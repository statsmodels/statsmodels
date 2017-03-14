from __future__ import division, print_function

import math
from collections import defaultdict
from math import log
import numpy as np
from numpy import hstack, vstack
from numpy.linalg import inv, svd
import scipy
import scipy.stats

from statsmodels.compat.python import range, string_types, iteritems
from statsmodels.iolib.summary import Summary
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import chain_dot
from statsmodels.tsa.tsatools import duplication_matrix, vec

import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.vector_ar import output as var_output, output
import statsmodels.tsa.vector_ar.irf as irf
import statsmodels.tsa.vector_ar.plotting as plot
from statsmodels.tsa.vector_ar.util import get_index, seasonal_dummies
from statsmodels.tsa.vector_ar.var_model import forecast, forecast_interval, \
    VAR, ma_rep, orth_ma_rep, test_normality
from statsmodels.tsa.coint_tables import c_sja, c_sjt


def select_order(data, maxlags, deterministic="nc", seasons=0, exog=None,
                 exog_coint=None, verbose=True):
    """
    Compute lag order selections based on each of the available information
    criteria.

    Parameters
    ----------
    data : array-like (nobs_tot x neqs)
        The observed data.
    maxlags : int
        All orders until maxlag will be compared according to the information
        criteria listed in the Results-section of this docstring.
    deterministic : str {"nc", "co", "ci", "lo", "li"}, default: "nc"
        * "nc" - no deterministic terms
        * "co" - constant outside the cointegration relation
        * "ci" - constant within the cointegration relation
        * "lo" - linear trend outside the cointegration relation
        * "li" - linear trend within the cointegration relation

        Combinations of these are possible (e.g. "cili" or "colo" for linear
        trend with intercept)
    seasons : int, default: 0
        Number of seasons.
    exog : ndarray (nobs_tot x neqs) or None, default: None
        Deterministic terms outside the cointegration relation.
    exog_coint: ndarray (nobs_tot x neqs) or None, default: None
        Deterministic terms inside the cointegration relation.
    verbose : bool, default: True
        If True, print table of info criteria and selected orders

    Returns
    -------
    selected_orders : dict
        Keys: Information criterion as string. {"aic", "bic", "hqic", "fpe"}
        Values: Number of lagged differences chosen by the corresponding
        information criterion.
    """
    ic = defaultdict(list)
    for p in range(1, maxlags + 2):  # +2 because k_ar_VECM == k_ar_VAR - 1
        exogs = []
        if "co" in deterministic or "ci" in deterministic:
            exogs.append(np.ones(len(data)).reshape(-1, 1))
        if "lo" in deterministic or "li" in deterministic:
            exogs.append(1 + np.arange(len(data)).reshape(-1, 1))
        if exog_coint is not None:
            exogs.append(exog_coint)
        if seasons > 0:
            exogs.append(seasonal_dummies(seasons, len(data)
                                          ).reshape(-1, seasons-1))
        if exog is not None:
            exogs.append(exog)
        exogs = hstack(exogs) if exogs else None
        var_model = VAR(data, exogs)
        # exclude some periods ==> same amount of data used for each lag order
        var_result = var_model._estimate_var(lags=p, offset=maxlags+1-p)

        for k, v in iteritems(var_result.info_criteria):
            ic[k].append(v)
    # -1+1 in the following line is only here for clarification.
    # -1 because k_ar_VECM == k_ar_VAR - 1
    # +1 because p == index +1 (we start with p=1, not p=0)
    selected_orders = dict((ic_name, np.array(ic_value).argmin() - 1 + 1)
                           for ic_name, ic_value in iteritems(ic))

    if verbose:
        output.print_ic_table(ic, selected_orders, vecm=True)

    return selected_orders


def _linear_trend(nobs, k_ar, coint=False):
    """
    Construct an ndarray representing a linear trend in a VECM. Note that the
    returned array's size is nobs and not nobs_tot so it cannot be used to
    construct the exog-argument of VECM's __init__ method.

    Parameters
    ----------
    nobs : int
        Number of observations excluding the presample.
    k_ar : int
        Number of lagged differences.
    coint : boolean, default: False
        If True (False), the returned array represents a linear trend inside
        (outside) the cointegration relation.

    Returns
    -------
    ret : ndarray (nobs)
        An ndarray representing a linear trend in a VECM
    """
    ret = np.arange(nobs) + k_ar
    if not coint:
        ret += 1
    return ret


def num_det_vars(det_string, seasons=0):
    """Gives the number of deterministic variables specified by det_string and
    seasons.

    Parameters
    ----------
    det_string : str {"nc", "co", "ci", "lo", "li"}
        * "nc" - no deterministic terms
        * "co" - constant outside the cointegration relation
        * "ci" - constant within the cointegration relation
        * "lo" - linear trend outside the cointegration relation
        * "li" - linear trend within the cointegration relation

        Combinations of these are possible (e.g. "cili" or "colo" for linear
        trend with intercept)
    seasons : int
        Number of seasons.

    Returns
    -------
    num : int
        Number of deterministic terms and number dummy variables for seasonal
        terms.
    """
    num = 0
    if "ci" in det_string or "co" in det_string:
        num += 1
    if "li" in det_string or "lo" in det_string:
        num += 1
    if seasons > 0:
        num += seasons - 1
    return num


def deterministic_to_exog(deterministic, seasons, nobs_tot, first_season=0,
                          seasons_centered=False, exog=None, exog_coint=None):
    """
    Use the VECM's deterministic terms to construct an array that is suitable
    to convey this information to VAR in form of the exog-argument for VAR's
    __init__ method.

    Parameters
    ----------
    deterministic : str
        See VECM's docstring for more information.
    seasons : int
        Number of seasons.
    nobs_tot : int
        Number of observations including the presample.
    first_season : int, default: 0
        Season of the first observation.
    seasons_centered : boolean, default: False
        If True, the seasonal dummy variables are demeaned such that they are
        orthogonal to an intercept term.
    exog : ndarray (nobs_tot x #det_terms) or None, default: None
        An ndarray representing deterministic terms outside the cointegration
        relation.
    exog_coint : ndarray (nobs_tot x #det_terms_coint) or None, default: None
        An ndarray representing deterministic terms inside the cointegration
        relation.

    Returns
    -------
    exog : ndarray or None
        None, if the function's arguments don't contain deterministic terms.
        Otherwise, an ndarray representing these deterministic terms.
    """
    exogs = []
    if "co" in deterministic or "ci" in deterministic:
        exogs.append(np.ones(nobs_tot))
    if exog_coint is not None:
        exogs.append(exog_coint)
    if "lo" in deterministic or "li" in deterministic:
        exogs.append(np.arange(nobs_tot))
    if seasons > 0:
        exogs.append(seasonal_dummies(seasons, nobs_tot,
                                      first_period=first_season,
                                      centered=seasons_centered))
    if exog is not None:
        exogs.append(exog)
    return np.column_stack(exogs) if exogs else None


def _mat_sqrt(_2darray):
    """Calculates the square root of a matrix.

    Parameters
    ----------
    _2darray : ndarray
        A 2-dimensional ndarray representing a square matrix.

    Returns
    -------
    result : ndarray
        Square root of the matrix given as function argument.
    """
    u_, s_, v_ = svd(_2darray, full_matrices=False)
    s_ = np.sqrt(s_)
    return chain_dot(u_, np.diag(s_), v_)


def _endog_matrices(endog_tot, exog, exog_coint, diff_lags, deterministic,
                    seasons=0, first_season=0):
    """Returns different matrices needed for parameter estimation (compare p.
    186 in [1]_). These matrices consist of elements of the data as well as
    elements representing deterministic terms. A tuple of consisting of these
    matrices is returned.

    Parameters
    ----------
    endog_tot : ndarray (neqs x nobs_tot)
        The whole sample including the presample.
    exog: ndarray (nobs_tot x neqs) or None
        Deterministic terms outside the cointegration relation.
    exog_coint: ndarray (nobs_tot x neqs) or None
        Deterministic terms inside the cointegration relation.
    diff_lags : int
        Number of lags in the VEC representation.
    deterministic : str {"nc", "co", "ci", "lo", "li"}
        * "nc" - no deterministic terms
        * "co" - constant outside the cointegration relation
        * "ci" - constant within the cointegration relation
        * "lo" - linear trend outside the cointegration relation
        * "li" - linear trend within the cointegration relation

        Combinations of these are possible (e.g. "cili" or "colo" for linear
        trend with intercept)
    seasons : int, default: 0
        Number of seasons. 0 (default) means no seasons.
    first_season : int, default: 0
        The season of the first observation. `0` means first season, `1` means
        second season, ..., `seasons-1` means the last season.

    Returns
    -------
    y_1_T : ndarray (neqs x nobs)
        The (transposed) data without the presample.
        .. math:: (y_1, \ldots, y_T)
    delta_y_1_T : ndarray (neqs x nobs)
        .. math:: (y_1, \ldots, y_T) - (y_0, \ldots, y_{T-1})
    y_min1 : ndarray (neqs x nobs)
        (dimensions assuming no deterministic terms are given)
        .. math:: (y_0, \ldots, y_{T-1}
    delta_x : ndarray (diff_lags*neqs x nobs)
        (dimensions assuming no deterministic terms are given)

    References
    ----------
    .. [1] Lutkepohl, H. 2005. *New Introduction to Multiple Time Series Analysis*. Springer.
    """
    # p. 286:
    p = diff_lags+1
    y = endog_tot
    K = y.shape[0]
    y_1_T = y[:, p:]
    T = y_1_T.shape[1]
    delta_y = np.diff(y)
    delta_y_1_T = delta_y[:, p-1:]

    y_min1 = y[:, p-1:-1]
    if "co" in deterministic and "ci" in deterministic:
        raise ValueError("Both 'co' and 'ci' as deterministic terms given. " +
                         "Please choose one of the two.")
    y_min1_stack = [y_min1]
    if "ci" in deterministic:  # pp. 257, 299, 306, 307
        y_min1_stack.append(np.ones(T))
    if "li" in deterministic:  # p. 299
        y_min1_stack.append(_linear_trend(T, p, coint=True))
    if exog_coint is not None:
        y_min1_stack.append(exog_coint[-T-1:-1].T)
    y_min1 = np.row_stack(y_min1_stack)

    # p. 286:
    delta_x = np.zeros((diff_lags*K, T))
    if diff_lags > 0:
        for j in range(delta_x.shape[1]):
            delta_x[:, j] = (delta_y[:, j+p-2:None if j-1 < 0 else j-1:-1]
                             .T.reshape(K*(p-1)))
    delta_x_stack = [delta_x]
    # p. 299, p. 303:
    if "co" in deterministic:
        delta_x_stack.append(np.ones(T))
    if seasons > 0:
        delta_x_stack.append(seasonal_dummies(seasons, delta_x.shape[1],
                                              first_period=first_season,
                                              centered=True).T)
    if "lo" in deterministic:
        delta_x_stack.append(_linear_trend(T, p))
    if exog is not None:
        delta_x_stack.append(exog[-T:].T)
    delta_x = np.row_stack(delta_x_stack)

    return y_1_T, delta_y_1_T, y_min1, delta_x


def _r_matrices(T, delta_x, delta_y_1_T, y_min1):
    """Returns two ndarrays needed for parameter estimation as well as the
    calculation of standard errors.

    Parameters
    ----------
    T : int
        nobs
    delta_x : ndarray (diff_lags*neqs x nobs)
        (dimensions assuming no deterministic terms are given)
    delta_y_1_T : ndarray (neqs x nobs)
        :math:`(y_1, \\ldots, y_T) - (y_0, \\ldots, y_{T-1})`
    y_min1 : ndarray (neqs x nobs)
        (dimensions assuming no deterministic terms are given)
        :math:`(y_0, \\ldots, y_{T-1}`

    Returns
    -------
    result : tuple
        A tuple of two ndarrays. (See p. 292 in [1]_ for the definition of
        R_0 and R_1.)

    References
    ----------
    .. [1] Lutkepohl, H. 2005. *New Introduction to Multiple Time Series Analysis*. Springer.
    """

    # todo: rewrite m such that a big (TxT) matrix is avoided
    m = np.identity(T) - (
        delta_x.T.dot(inv(delta_x.dot(delta_x.T))).dot(delta_x))  # p. 291
    r0 = delta_y_1_T.dot(m)  # p. 292
    r1 = y_min1.dot(m)
    return r0, r1


def _sij(delta_x, delta_y_1_T, y_min1):
    """Returns matrices and eigenvalues and -vectors used for parameter
    estimation and the calculation of a models loglikelihood.

    Parameters
    ----------
    delta_x : ndarray (diff_lags*neqs x nobs)
        (dimensions assuming no deterministic terms are given)
    delta_y_1_T : ndarray (neqs x nobs)
        :math:`(y_1, \\ldots, y_T) - (y_0, \\ldots, y_{T-1})`
    y_min1 : ndarray (neqs x nobs)
        (dimensions assuming no deterministic terms are given)
        :math:`(y_0, \\ldots, y_{T-1}`

    Returns
    -------
    result : tuple
        A tuple of five ndarrays as well as eigenvalues and -vectors of a
        certain (matrix) product of some of the returned ndarrays.
        (See pp. 294-295 in [1]_ for more information on
        :math:`S_0, S_1, \\lambda_i, \\v_i` for
        :math:`i \\in \\{1, \\dots, K\\}`.)

    References
    ----------
    .. [1] Lutkepohl, H. 2005. *New Introduction to Multiple Time Series Analysis*. Springer.
    """
    T = y_min1.shape[1]
    r0, r1 = _r_matrices(T, delta_x, delta_y_1_T, y_min1)
    s00 = np.dot(r0, r0.T) / T
    s01 = np.dot(r0, r1.T) / T
    s10 = s01.T
    s11 = np.dot(r1, r1.T) / T
    s11_ = inv(_mat_sqrt(s11))
    # p. 295:
    s01_s11_ = np.dot(s01, s11_)
    eig = np.linalg.eig(chain_dot(s01_s11_.T, inv(s00), s01_s11_))
    lambd = eig[0]
    v = eig[1]
    # reorder eig_vals to make them decreasing (and order eig_vecs accordingly)
    lambd_order = np.argsort(lambd)[::-1]
    lambd = lambd[lambd_order]
    v = v[:, lambd_order]
    return s00, s01, s10, s11, s11_, lambd, v


def coint_johansen(endog_tot, det_order, k_ar, coint_trend=None):
    """
    Perform the Johansen cointegration test for determining the cointegration
    rank of a VECM.

    Parameters
    ----------
    endog_tot : array-like (nobs_tot x neqs)

    det_order : int
        * -1 - no deterministic terms
        * 0 - constant term
        * 1 - linear trend
        * >1 - higher polynomial order
    k_ar : int, nonnegative
        Number of lagged differences in the model.
    coint_trend

    Returns
    -------
    result : Holder
        An object containing the results which can be accessed using
        dot-notation.
    """
    # TODO: describe coint_trend argument.

    from statsmodels.regression.linear_model import OLS
    tdiff = np.diff

    class Holder(object):
        pass

    def trimr(x, front, end):
        if end > 0:
            return x[front:-end]
        else:
            return x[front:]

    import statsmodels.tsa.tsatools as tsat
    mlag = tsat.lagmat

    def lag(x, lags):
        return x[:-lags]

    def detrend(y, order):
        if order == -1:
            return y
        return OLS(y, np.vander(np.linspace(-1, 1, len(y)),
                                order+1)).fit().resid

    def resid(y, x):
        if x.size == 0:
            return y
        r = y - np.dot(x, np.dot(np.linalg.pinv(x), y))
        return r

    nobs, neqs = endog_tot.shape

    # why this?  f is detrend transformed series, det_order is detrend data
    if det_order > -1:
        f = 0
    else:
        f = det_order

    if coint_trend is not None:
        f = coint_trend  # matlab has separate options

    endog_tot = detrend(endog_tot, det_order)
    dx = tdiff(endog_tot, 1, axis=0)
    # dx = trimr(dx, 1, 0)
    z = mlag(dx, k_ar)  # [k_ar-1:]
    z = trimr(z, k_ar, 0)
    z = detrend(z, f)

    dx = trimr(dx, k_ar, 0)

    dx = detrend(dx, f)
    # r0t = dx - z*(z\dx)
    r0t = resid(dx, z)  # diff on lagged diffs
    # lx = trimr(lag(endog_tot,k_ar), k_ar, 0)
    lx = lag(endog_tot, k_ar)
    lx = trimr(lx, 1, 0)
    dx = detrend(lx, f)
    # rkt = dx - z*(z\dx)
    rkt = resid(dx, z)  # level on lagged diffs
    skk = np.dot(rkt.T, rkt) / rkt.shape[0]
    sk0 = np.dot(rkt.T, r0t) / rkt.shape[0]
    s00 = np.dot(r0t.T, r0t) / r0t.shape[0]
    sig = np.dot(sk0, np.dot(inv(s00), sk0.T))
    tmp = inv(skk)
    # du, au = np.linalg.eig(np.dot(tmp, sig))
    au, du = np.linalg.eig(np.dot(tmp, sig))  # au is eval, du is evec
    # orig = np.dot(tmp, sig)

    # % Normalize the eigen vectors such that (du'skk*du) = I
    temp = inv(np.linalg.cholesky(np.dot(du.T, np.dot(skk, du))))
    dt = np.dot(du, temp)

    # JP: the next part can be done much  easier

    #%      NOTE: At this point, the eigenvectors are aligned by column. To
    #%            physically move the column elements using the MATLAB sort,
    #%            take the transpose to put the eigenvectors across the row

    #dt = transpose(dt)

    #% sort eigenvalues and vectors

    # au, auind = np.sort(diag(au))
    auind = np.argsort(au)
    # a = np.flipud(au)
    aind = np.flipud(auind)
    a = au[aind]
    # d = dt[aind, :]
    d = dt[:, aind]

    #%NOTE: The eigenvectors have been sorted by row based on auind and moved to array "d".
    #%      Put the eigenvectors back in column format after the sort by taking the
    #%      transpose of "d". Since the eigenvectors have been physically moved, there is
    #%      no need for aind at all. To preserve existing programming, aind is reset back to
    #%      1, 2, 3, ....

    # d  =  transpose(d)
    # test = np.dot(transpose(d), np.dot(skk, d))

    #%EXPLANATION:  The MATLAB sort function sorts from low to high. The flip realigns
    #%auind to go from the largest to the smallest eigenvalue (now aind). The original procedure
    #%physically moved the rows of dt (to d) based on the alignment in aind and then used
    #%aind as a column index to address the eigenvectors from high to low. This is a double
    #%sort. If you wanted to extract the eigenvector corresponding to the largest eigenvalue by,
    #%using aind as a reference, you would get the correct eigenvector, but with sorted
    #%coefficients and, therefore, any follow-on calculation would seem to be in error.
    #%If alternative programming methods are used to evaluate the eigenvalues, e.g. Frame method
    #%followed by a root extraction on the characteristic equation, then the roots can be
    #%quickly sorted. One by one, the corresponding eigenvectors can be generated. The resultant
    #%array can be operated on using the Cholesky transformation, which enables a unit
    #%diagonalization of skk. But nowhere along the way are the coefficients within the
    #%eigenvector array ever changed. The final value of the "beta" array using either method
    #%should be the same.

    #% Compute the trace and max eigenvalue statistics */
    lr1 = np.zeros(neqs)
    lr2 = np.zeros(neqs)
    cvm = np.zeros((neqs, 3))
    cvt = np.zeros((neqs, 3))
    iota = np.ones(neqs)
    t, junk = rkt.shape
    for i in range(0, neqs):
        tmp = trimr(np.log(iota-a), i, 0)
        lr1[i] = -t * np.sum(tmp, 0)  # columnsum ?
        # tmp = np.log(1-a)
        # lr1[i] = -t * np.sum(tmp[i:])
        lr2[i] = -t * np.log(1-a[i])
        cvm[i, :] = c_sja(neqs - i, det_order)
        cvt[i, :] = c_sjt(neqs - i, det_order)
        aind[i] = i

    result = Holder()
    # estimation results, residuals
    result.rkt = rkt
    result.r0t = r0t
    result.eig = a
    result.evec = d  # transposed compared to matlab ?
    result.lr1 = lr1
    result.lr2 = lr2
    result.cvt = cvt
    result.cvm = cvm
    result.ind = aind
    result.meth = 'johansen'

    return result


class VECM(tsbase.TimeSeriesModel):
    """
    Fit a VECM process
    .. math:: \\Delta y_t = \\Pi y_{t-1} + \\Gamma_1 \\Delta y_{t-1} + \\ldots + \\Gamma_{k_ar-1} \\Delta y_{t-k_ar+1} + u_t
    where
    .. math:: \\Pi = \\alpha \\beta'
    as described in chapter 7 of [1]_.

    Parameters
    ----------
    endog_tot : array-like (nobs_tot x neqs)
        2-d endogenous response variable.
    exog: ndarray (nobs_tot x neqs) or None
        Deterministic terms outside the cointegration relation.
    exog_coint: ndarray (nobs_tot x neqs) or None
        Deterministic terms inside the cointegration relation.
    dates : array-like of datetime, optional
        See :class:`statsmodels.tsa.base.tsa_model.TimeSeriesModel` for more
        information.
    freq : str, optional
        See :class:`statsmodels.tsa.base.tsa_model.TimeSeriesModel` for more
        information.
    missing : str, optional
        See :class:`statsmodels.base.model.Model` for more information.
    diff_lags : int
        Number of lags in the VEC representation
    coint_rank : int
        Cointegration rank, equals the rank of the matrix :math:`\\Pi` and the
        number of columns of :math:`\\alpha` and :math:`\\beta`.
    deterministic : str {"nc", "co", "ci", "lo", "li"}, default: "nc"
        * "nc" - no deterministic terms
        * "co" - constant outside the cointegration relation
        * "ci" - constant within the cointegration relation
        * "lo" - linear trend outside the cointegration relation
        * "li" - linear trend within the cointegration relation

        Combinations of these are possible (e.g. "cili" or "colo" for linear
        trend with intercept)
    seasons : int, default: 0
        Number of seasons. 0 means no seasons.
    first_season : int, default: 0
        Season of the first observation.

    References
    ----------
    .. [1] Lutkepohl, H. 2005. *New Introduction to Multiple Time Series Analysis*. Springer.
    """

    def __init__(self, endog_tot, exog=None, exog_coint=None, dates=None,
                 freq=None, missing="none", diff_lags=1, coint_rank=1,
                 deterministic="nc", seasons=0, first_season=0):
        super(VECM, self).__init__(endog_tot, exog, dates, freq,
                                   missing=missing)
        if exog_coint is not None and \
                not exog_coint.shape[0] == endog_tot.shape[0]:
            raise ValueError("exog_coint must have as many rows as enodg_tot!")
        if self.endog.ndim == 1:
            raise ValueError("Only gave one variable to VECM")
        self.y = self.endog.T
        self.exog_coint = exog_coint
        self.neqs = self.endog.shape[1]
        self.p = diff_lags + 1
        self.diff_lags = diff_lags
        self.coint_rank = coint_rank
        self.deterministic = deterministic
        self.seasons = seasons
        self.first_season = first_season
        self.load_coef_repr = "ec"  # name for loading coef. (alpha) in summary

    def fit(self, method="ml"):
        """
        Estimates the parameters of a VECM as described on pp. 269-304 in [1]_
        and returns a VECMResults object.

        Parameters
        ----------
        method : str {"ml"}, default: "ml"
            Estimation method to use.
            * "ml" - Maximum likelihood

        Returns
        -------
        est : VECMResults

        References
        -----
        .. [1] Lutkepohl, H. 2005. *New Introduction to Multiple Time Series Analysis*. Springer.
        """
        if method == "ml":
            return self._estimate_vecm_ml()
        else:
            raise ValueError("%s not recognized, must be among %s"
                             % (method, "ml"))

    def _estimate_vecm_ml(self):
        y_1_T, delta_y_1_T, y_min1, delta_x = _endog_matrices(
                self.y, self.exog, self.exog_coint, self.diff_lags,
                self.deterministic, self.seasons, self.first_season)
        T = y_1_T.shape[1]

        s00, s01, s10, s11, s11_, _, v = _sij(delta_x, delta_y_1_T, y_min1)

        beta_tilde = (v[:, :self.coint_rank].T.dot(s11_)).T
        # normalize beta tilde such that eye(r) forms the first r rows of it:
        beta_tilde = np.dot(beta_tilde, inv(beta_tilde[:self.coint_rank]))
        alpha_tilde = s01.dot(beta_tilde).dot(
                inv(beta_tilde.T.dot(s11).dot(beta_tilde)))
        gamma_tilde = (delta_y_1_T - alpha_tilde.dot(beta_tilde.T).dot(y_min1)
                       ).dot(delta_x.T).dot(inv(np.dot(delta_x, delta_x.T)))
        temp = (delta_y_1_T - alpha_tilde.dot(beta_tilde.T).dot(y_min1) -
                gamma_tilde.dot(delta_x))
        sigma_u_tilde = temp.dot(temp.T) / T

        return VECMResults(self.y, self.exog, self.exog_coint, self.p,
                           self.coint_rank, alpha_tilde, beta_tilde,
                           gamma_tilde, sigma_u_tilde,
                           deterministic=self.deterministic,
                           seasons=self.seasons, delta_y_1_T=delta_y_1_T,
                           y_min1=y_min1, delta_x=delta_x, model=self,
                           names=self.endog_names, dates=self.data.dates,
                           first_season=self.first_season)

    @property
    def lagged_param_names(self):
        """
        Returns
        -------
        param_names : list of str
            Returns a list of parameter names for the lagged endogenous
            parameters which are called :math:`\\Gamma` in [1]_
            (see chapter 6).
            If present in the model, also names for deterministic terms outside
            the cointegration relation are returned. They name the elements of
            the matrix C in [1]_ (p. 299).

        References
        ----------
        .. [1] Lutkepohl, H. 2005. *New Introduction to Multiple Time Series Analysis*. Springer.
        """
        param_names = []

        # 1. Deterministic terms outside cointegration relation
        if "co" in self.deterministic:
            param_names += ["const.%s" % n for n in self.endog_names]

        if self.seasons > 0:
            param_names += ["season%d.%s" % (s, n)
                            for s in range(1, self.seasons)
                            for n in self.endog_names]

        if "lo" in self.deterministic:
            param_names += ["lin_trend.%s" % n for n in self.endog_names]

        if self.exog is not None:
            param_names += ["exog%d.%s" % (exog_no, n)
                            for exog_no in range(1, self.exog.shape[1] + 1)
                            for n in self.endog_names]

        # 2. lagged endogenous terms
        param_names += [
            "L%d.%s.%s" % (i+1, n1, n2)
            for n2 in self.endog_names
            for i in range(self.p-1)
            for n1 in self.endog_names]

        return param_names

    @property
    def load_coef_param_names(self):
        """
        Returns
        -------
        param_names : list of str
            Returns a list of parameter names for the loading coefficients
            which are called :math:`\\alpha` in [1]_ (see chapter 6).

        References
        ----------
        .. [1] Lutkepohl, H. 2005. *New Introduction to Multiple Time Series Analysis*. Springer.
        """
        param_names = []

        if self.coint_rank == 0:
            return None

        # loading coefficients (alpha) # called "ec" in JMulTi, "ECT" in tsDyn,
        param_names += [               # and "_ce" in Stata
            self.load_coef_repr + "%d.%s" % (i+1, self.endog_names[j])
            for j in range(self.neqs)
            for i in range(self.coint_rank)
        ]

        return param_names

    @property
    def coint_param_names(self):
        """

        Returns
        -------
        param_names : list of str
            Returns a list of parameter names for the cointegration matrix
            as well as deterministic terms inside the cointegration relation
            (if present in the model).

        """
        # 1. cointegration matrix/vector
        param_names = []

        param_names += [("beta.%d." + self.load_coef_repr + "%d") % (j+1, i+1)
                        for i in range(self.coint_rank)
                        for j in range(self.neqs)]

        # 2. deterministic terms inside cointegration relation
        if "ci" in self.deterministic:
            param_names += ["const." + self.load_coef_repr + "%d" % (i+1)
                            for i in range(self.coint_rank)]

        if "li" in self.deterministic:
            param_names += ["lin_trend." + self.load_coef_repr + "%d" % (i+1)
                            for i in range(self.coint_rank)]

        if self.exog_coint is not None:
            param_names += ["exog_coint%d.%s" % (n+1, exog_no)
                            for exog_no in range(1, self.exog_coint.shape[1]+1)
                            for n in range(self.neqs)]

        return param_names


# -----------------------------------------------------------------------------
# VECMResults class

class VECMResults(object):
    """Class holding estimation related results of a vector error correction
    model (VECM).

    Parameters
    ----------
    endog_tot : ndarray (neqs x nobs_tot)
        Array of observations.
    exog: ndarray (nobs_tot x neqs) or None
        Deterministic terms outside the cointegration relation.
    exog_coint: ndarray (nobs_tot x neqs) or None
        Deterministic terms inside the cointegration relation.
    k_ar : int
        Lags in the VAR representation. This implies: Lags in the VEC
        representation = k_ar - 1
    coint_rank : int
        Cointegration rank, equals the rank of the matrix :math:`\\Pi` and the
        number of columns of :math:`\\alpha` and :math:`\\beta`.
    alpha : ndarray (neqs x coint_rank)
        Estimate for the parameter :math:`\\alpha` of a VECM.
    beta : ndarray (neqs x coint_rank)
        Estimate for the parameter :math:`\\beta` of a VECM.
    gamma : ndarray (neqs x neqs*(k_ar-1))
        Array containing the estimates of the p-1 parameter matrices
        :math:`\\Gamma_1, \\dots, \\Gamma_{p-1}` of a VECM(p-1). The
        submatrices are stacked horizontally from left to right.
    sigma_u : ndarray (neqs x neqs)
        Estimate of white noise process covariance matrix :math:`\\Sigma_u`.
    deterministic : str {"nc", "co", "ci", "lo", "li"}, default: "nc"
        * "nc" - no deterministic terms
        * "co" - constant outside the cointegration relation
        * "ci" - constant within the cointegration relation
        * "lo" - linear trend outside the cointegration relation
        * "li" - linear trend within the cointegration relation

        Combinations of these are possible (e.g. "cili" or "colo" for linear
        trend with intercept)
    seasons : int, default: 0
        Number of seasons. 0 means no seasons.
    first_season : int, default: 0
        Season of the first observation.
    delta_y_1_T : ndarray or None, default: None
        Auxilliary array for internal computations. It will be calculated if
        not given as parameter.
    y_min1 : ndarray or None, default: None
        Auxilliary array for internal computations. It will be calculated if
        not given as parameter.
    delta_x : ndarray or None, default: None
        Auxilliary array for internal computations. It will be calculated if
        not given as parameter.
    model : VECM
        An instance of the VECM class.
    names : list of str
        Each str in the list represents the name of a variable of the time
        series.
    dates : array-like
        For example a DatetimeIndex of length nobs_tot.

    Returns
    -------
    **Attributes**
    nobs : int
        Number of observations (excluding the presample).
    model : see Parameters
    y_all : see endog_tot in Parameters
    exog : see Parameters
    exog_coint : see Parameters
    names : see Parameters
    dates : see Parameters
    neqs : int
        Number of variables in the time series.
    k_ar : see Parameters
    deterministic : see Parameters
    seasons : see Parameters
    first_season : see Parameters
    alpha : see Parameters
    beta : see Parameters
    gamma : see Parameters
    sigma_u : see Parameters
    det_coef_coint : ndarray (#(determinist. terms inside the coint. rel.) x r)
        Estimated coefficients for the all deterministic terms inside the
        cointegration relation.
    const_coint : ndarray (1 x r)
        If there is a constant deterministic term inside the cointegration
        relation, then const_coint is the first row of det_coef_coint.
        Otherwise it's an ndarray of zeros.
    lin_trend_coint : ndarray (1 x r)
        If there is a linear deterministic term inside the cointegration
        relation, then lin_trend_coint contains the corresponding estimated
         coefficients. As such it represents the corresponding row of
         det_coef_coint. If there is no linear deterministic term inside the
         cointegration relation, then lin_trend_coint is an ndarray of zeros.
    exog_coint_coefs : ndarray (exog_coint.shape[1] x r) or None
        If deterministic terms inside the cointegration relation are passed via
        the exog_coint parameter, then exog_coint_coefs contains the
        corresponding estimated coefficients. As such exog_coint_coefs
        represents the last rows of det_coef_coint.
        If no deterministic terms were passed via the exog_coint parameter,
        this attribute is None.
    det_coef : ndarray (neqs x #(deterministic terms outside the coint. rel.))
        Estimated coefficients for the all deterministic terms outside the
        cointegration relation.
    const : ndarray (neqs x 1) or (neqs x 0)
        If a constant deterministic term outside the cointegration is specified
        within the deterministic parameter, then const is the first column of
        det_coef_coint. Otherwise it's an ndarray of size zero.
    seasonal : ndarray (neqs x seasons)
        If the seasons parameter is > 0, then seasonal contains the estimated
        coefficients corresponding to the seasonal terms. Otherwise it's an
        ndarray of size zero.
    lin_trend : ndarray (neqs x 1) or (neqs x 0)
        If a linear deterministic term outside the cointegration is specified
        within the deterministic parameter, then lin_trend contains the
        corresponding estimated coefficients. As such it represents the
        corresponding column of det_coef_coint. If there is no linear
        deterministic term outside the cointegration relation, then
        lin_trend is an ndarray of size zero.
    exog_coefs : ndarray (neqs x exog_coefs.shape[1])
        If deterministic terms outside the cointegration relation are passed
        via the exog parameter, then exog_coefs contains the corresponding
        estimated coefficients. As such exog_coefs represents the last columns
        of det_coef.
        If no deterministic terms were passed via the exog parameter, this
        attribute is an ndarray of size zero.
    delta_y_1_T : see Parameters
    y_min1 : see Parameters
    delta_x : see Parameters
    r : int
        Cointegration rank, equals the rank of the matrix :math:`\\Pi` and the
        number of columns of :math:`\\alpha` and :math:`\\beta`.
    llf : float
        The model's log-likelihood.
    cov_params : ndarray (d x d)
        Covariance matrix of the parameters. The number of rows and columns, d,
        is equal to neqs * (neqs+num_det_coef_coint + neqs*(k_ar-1)+number of
        deterministic dummy variables outside the cointegration relation). For
        the case with no deterministic terms this matrix is defined on p. 287
        in [1]_ as :math:`\\Sigma_{co}` and its relationship to the
        ML-estimators can be seen in eq. (7.2.21) on p. 296 in [1]_.
    cov_params_wo_det : ndarray
        Covariance matrix of the parameters
        :math:`\\tilde{\\Pi}, \\tilde{\\Gamma}` where
        :math:`\\tilde{\\Pi} = \\tilde{\\alpha} \\tilde{\\beta'}`.
        Equals cov_params without the rows and columns related to deterministic
        terms. This matrix is defined as :math:`\\Sigma_{co}` on p. 287 in [1]_.
    stderr_params : ndarray (d)
        Array containing the standard errors of :math:`\\Pi`, :math:`\\Gamma`,
        and estimated parameters related to deterministic terms.
    stderr_coint : ndarray (neqs+num_det_coef_coint x r)
        Array containing the standard errors of :math:`\\beta` and estimated
        parameters related to deterministic terms inside the cointegration
        relation.
    stderr_alpha :  ndarray (neqs x r)
        The standard errors of :math:`\\alpha`.
    stderr_beta : ndarray (neqs x r)
        The standard errors of :math:`\\beta`.
    stderr_det_coef_coint : ndarray (num_det_coef_coint x r)
        The standard errors of estimated the parameters related to
        deterministic terms inside the cointegration relation.
    stderr_gamma : ndarray (neqs x neqs*(k_ar-1))
        The standard errors of :math:`\\Gamma`.
    stderr_det_coef : ndarray (neqs x det. terms outside the coint. relation)
        The standard errors of estimated the parameters related to
        deterministic terms outside the cointegration relation.
    tvalues_alpha : ndarray (neqs x r)
    tvalues_beta : ndarray (neqs x r)
    tvalues_det_coef_coint : ndarray (num_det_coef_coint x r)
    tvalues_gamma : ndarray (neqs x neqs*(k_ar-1))
    tvalues_det_coef : ndarray (neqs x det. terms outside the coint. relation)
    pvalues_alpha : ndarray (neqs x r)
    pvalues_beta : ndarray (neqs x r)
    pvalues_det_coef_coint : ndarray (num_det_coef_coint x r)
    pvalues_gamma : ndarray (neqs x neqs*(k_ar-1))
    pvalues_det_coef : ndarray (neqs x det. terms outside the coint. relation)
    var_rep : (k_ar x neqs x neqs)
        KxK parameter matrices A_i of the corresponding VAR representation. If
        the return value is assigned to a variable A, these matrices can be
        accessed via A[i] for i=0, ..., k_ar-1.
    cov_var_repr : ndarray (neqs**2 * k_ar x neqs**2 * k_ar)
        This matrix is called :math:`\\Sigma^{co}_{\\alpha}` on p. 289 in [1]_.
        It is needed e.g. for impulse-response-analysis.
    fittedvalues : ndarray (nobs x neqs)
        The predicted in-sample values of the models' endogenous variables.
    resid : ndarray (nobs x neqs)
        The residuals.

    References
    ----------
    .. [1] Lutkepohl, H. 2005. *New Introduction to Multiple Time Series Analysis*. Springer.
    """

    def __init__(self, endog_tot, exog, exog_coint, k_ar,
                 coint_rank, alpha, beta, gamma, sigma_u, deterministic='nc',
                 seasons=0, first_season=0, delta_y_1_T=None, y_min1=None,
                 delta_x=None, model=None, names=None, dates=None):
        self.model = model
        self.y_all = endog_tot
        self.exog = exog
        self.exog_coint = exog_coint
        self.names = names
        self.dates = dates
        self.neqs = endog_tot.shape[0]
        self.k_ar = k_ar
        self.deterministic = deterministic
        self.seasons = seasons
        self.first_season = first_season

        self.r = coint_rank
        if alpha.dtype == np.complex128 and np.all(np.imag(alpha) == 0):
            alpha = np.real(alpha)
        if beta.dtype == np.complex128 and np.all(np.imag(beta) == 0):
            beta = np.real(beta)
        if gamma.dtype == np.complex128 and np.all(np.imag(gamma) == 0):
            gamma = np.real(gamma)

        self.alpha = alpha
        self.beta, self.det_coef_coint = np.vsplit(beta, [self.neqs])
        self.gamma, self.det_coef = np.hsplit(gamma,
                                              [self.neqs * (self.k_ar - 1)])

        if "ci" in deterministic:
            self.const_coint = self.det_coef_coint[:1, :]
        else:
            self.const_coint = np.zeros(self.r).reshape((1, -1))
        if "li" in deterministic:
            start = 1 if "ci" in deterministic else 0
            self.lin_trend_coint = self.det_coef_coint[start:start+1, :]
        else:
            self.lin_trend_coint = np.zeros(self.r).reshape(1, -1)
        if self.exog_coint is not None:
            start = ("ci" in deterministic) + ("li" in deterministic)
            self.exog_coint_coefs = self.det_coef_coint[start:, :]
        else:
            self.exog_coint_coefs = None

        split_const_season = 1 if "co" in deterministic else 0
        split_season_lin = split_const_season + ((seasons-1) if seasons else 0)
        if "lo" in deterministic:
            split_lin_exog = split_season_lin + 1
        else:
            split_lin_exog = split_season_lin
        self.const, self.seasonal, self.lin_trend, self.exog_coefs = \
            np.hsplit(self.det_coef,
                      [split_const_season, split_season_lin, split_lin_exog])

        self.sigma_u = sigma_u

        if y_min1 is not None and delta_x is not None \
                and delta_y_1_T is not None:
            self._delta_y_1_T = delta_y_1_T
            self._y_min1 = y_min1
            self._delta_x = delta_x
        else:
            _y_1_T, self._delta_y_1_T, self._y_min1, self._delta_x = \
                _endog_matrices(endog_tot, self.exog, k_ar,
                                deterministic, seasons)
        self.nobs = self._y_min1.shape[1]

    @cache_readonly
    def llf(self):  # Lutkepohl p. 295 (7.2.20)
        """Compute VECM(k_ar-1) loglikelihood, where k_ar-1 denotes the number
        of lagged differences.
        """
        K = self.neqs
        T = self.nobs
        r = self.r
        s00, _, _, _, _, lambd, _ = _sij(self._delta_x, self._delta_y_1_T,
                                         self._y_min1)
        return - K * T * log(2*math.pi) / 2  \
            - T * (log(np.linalg.det(s00)) + sum(np.log(1-lambd)[:r])) / 2  \
            - K * T / 2

    @cache_readonly
    def _cov_sigma(self):
        sigma_u = self.sigma_u
        d = duplication_matrix(self.neqs)
        d_K_plus = np.linalg.pinv(d)
        # compare p. 93, 297 Lutkepohl (2005)
        return 2 * chain_dot(d_K_plus, np.kron(sigma_u, sigma_u), d_K_plus.T)


    @cache_readonly
    def cov_params(self):  # p.296 (7.2.21)
        # Sigma_co described on p. 287
        beta = self.beta
        if self.det_coef_coint.size > 0:
            beta = vstack((beta, self.det_coef_coint))
        dt = self.deterministic
        num_det = ("co" in dt) + ("lo" in dt)
        num_det += (self.seasons-1) if self.seasons else 0
        if self.exog is not None:
            num_det += self.exog.shape[1]
        b_id = scipy.linalg.block_diag(beta,
                                       np.identity(self.neqs * (self.k_ar-1) +
                                                   num_det))

        y_min1 = self._y_min1
        b_y = beta.T.dot(y_min1)
        omega11 = b_y.dot(b_y.T)
        omega12 = b_y.dot(self._delta_x.T)
        omega21 = omega12.T
        omega22 = self._delta_x.dot(self._delta_x.T)
        omega = np.bmat([[omega11, omega12],
                         [omega21, omega22]]).A

        mat1 = b_id.dot(inv(omega)).dot(b_id.T)
        return np.kron(mat1, self.sigma_u)

    @cache_readonly
    def cov_params_wo_det(self):
        # rows & cols to be dropped (related to deterministic terms inside the
        # cointegration relation)
        start_i = self.neqs**2  # first elements belong to alpha @ beta.T
        end_i = start_i + self.neqs * self.det_coef_coint.shape[0]
        to_drop_i = np.arange(start_i, end_i)

        # rows & cols to be dropped (related to deterministic terms outside of
        # the cointegration relation)
        cov = self.cov_params
        cov_size = len(cov)
        to_drop_o = np.arange(cov_size-self.det_coef.size, cov_size)

        to_drop = np.union1d(to_drop_i, to_drop_o)

        mask = np.ones(cov.shape, dtype=bool)
        mask[to_drop] = False
        mask[:, to_drop] = False
        cov_size_new = mask.sum(axis=0)[0]
        return cov[mask].reshape((cov_size_new, cov_size_new))

    # standard errors:
    @cache_readonly
    def stderr_params(self):
        return np.sqrt(np.diag(self.cov_params))

    @cache_readonly
    def stderr_coint(self):
        """
        Notes
        -----
        See p. 297 in [1]_. Using the rule
        :math:`vec(B R) = (B' \kron I) vec(R)` for two matrices B and R which
        are compatible for multiplication. This is rule (3) on p. 662 in [1]_.

        References
        ----------
        .. [1] Lutkepohl, H. 2005. *New Introduction to Multiple Time Series Analysis*. Springer.
        """
        _, r1 = _r_matrices(self.nobs, self._delta_x, self._delta_y_1_T,
                            self._y_min1)
        r12 = r1[self.r:]
        if r12.size == 0:
            return np.zeros((self.r, self.r))
        mat1 = inv(r12.dot(r12.T))
        mat1 = np.kron(mat1.T, np.identity(self.r))
        det = self.det_coef_coint.shape[0]
        mat2 = np.kron(np.identity(self.neqs-self.r+det),
                       inv(chain_dot(
                               self.alpha.T, inv(self.sigma_u), self.alpha)))
        first_rows = np.zeros((self.r, self.r))
        last_rows_1d = np.sqrt(np.diag(mat1.dot(mat2)))
        last_rows = last_rows_1d.reshape((self.neqs-self.r+det, self.r),
                                         order="F")
        return vstack((first_rows,
                       last_rows))

    @cache_readonly
    def stderr_alpha(self):
        ret_1dim = self.stderr_params[:self.alpha.size]
        return ret_1dim.reshape(self.alpha.shape, order="F")

    @cache_readonly
    def stderr_beta(self):
        ret_1dim = self.stderr_coint[:self.beta.shape[0]]
        return ret_1dim.reshape(self.beta.shape, order="F")

    @cache_readonly
    def stderr_det_coef_coint(self):
        if self.det_coef_coint.size == 0:
            return self.det_coef_coint  # 0-size array
        ret_1dim = self.stderr_coint[self.beta.shape[0]:]
        return ret_1dim.reshape(self.det_coef_coint.shape, order="F")

    @cache_readonly
    def stderr_gamma(self):
        start = self.alpha.shape[0] * (self.beta.shape[0] +
                                       self.det_coef_coint.shape[0])
        ret_1dim = self.stderr_params[start:start+self.gamma.size]
        return ret_1dim.reshape(self.gamma.shape, order="F")
    
    @cache_readonly
    def stderr_det_coef(self):
        if self.det_coef.size == 0:
            return self.det_coef  # 0-size array
        ret1_1dim = self.stderr_params[-self.det_coef.size:]
        return ret1_1dim.reshape(self.det_coef.shape, order="F")

    # t-values:
    @cache_readonly
    def tvalues_alpha(self):
        return self.alpha / self.stderr_alpha

    @cache_readonly
    def tvalues_beta(self):
        first_rows = np.zeros((self.r, self.r))
        last_rows = self.beta[self.r:] / self.stderr_beta[self.r:]
        return vstack((first_rows,
                       last_rows))

    @cache_readonly
    def tvalues_det_coef_coint(self):
        if self.det_coef_coint.size == 0:
            return self.det_coef_coint  # 0-size array
        return self.det_coef_coint / self.stderr_det_coef_coint

    @cache_readonly
    def tvalues_gamma(self):
        return self.gamma / self.stderr_gamma

    @cache_readonly
    def tvalues_det_coef(self):
        if self.det_coef.size == 0:
            return self.det_coef  # 0-size array
        return self.det_coef / self.stderr_det_coef

    # p-values:
    @cache_readonly
    def pvalues_alpha(self):
        return (1-scipy.stats.norm.cdf(abs(self.tvalues_alpha))) * 2

    @cache_readonly
    def pvalues_beta(self):
        first_rows = np.zeros((self.r, self.r))
        tval_last = self.tvalues_beta[self.r:]
        last_rows = (1-scipy.stats.norm.cdf(abs(tval_last))) * 2  # student-t
        return vstack((first_rows,
                       last_rows))

    @cache_readonly
    def pvalues_det_coef_coint(self):
        if self.det_coef_coint.size == 0:
            return self.det_coef_coint  # 0-size array
        return (1-scipy.stats.norm.cdf(abs(self.tvalues_det_coef_coint))) * 2

    @cache_readonly
    def pvalues_gamma(self):
        return (1-scipy.stats.norm.cdf(abs(self.tvalues_gamma))) * 2

    @cache_readonly
    def pvalues_det_coef(self):
        if self.det_coef.size == 0:
            return self.det_coef  # 0-size array
        return (1-scipy.stats.norm.cdf(abs(self.tvalues_det_coef))) * 2

    # confidence intervals
    def _make_conf_int(self, est, stderr, alpha):
        struct_arr = np.zeros(est.shape, dtype=[("lower", float),
                                               ("upper", float)])
        struct_arr["lower"] = est - scipy.stats.norm.ppf(1 - alpha/2) * stderr
        struct_arr["upper"] = est + scipy.stats.norm.ppf(1 - alpha/2) * stderr
        return struct_arr

    def conf_int_alpha(self, alpha=0.05):
        return self._make_conf_int(self.alpha, self.stderr_alpha, alpha)

    def conf_int_beta(self, alpha=0.05):
        return self._make_conf_int(self.beta, self.stderr_beta, alpha)

    def conf_int_det_coef_coint(self, alpha=0.05):
        return self._make_conf_int(self.det_coef_coint,
                                   self.stderr_det_coef_coint, alpha)

    def conf_int_gamma(self, alpha=0.05):
        return self._make_conf_int(self.gamma, self.stderr_gamma, alpha)

    def conf_int_det_coef(self, alpha=0.05):
        return self._make_conf_int(self.det_coef, self.stderr_det_coef, alpha)

    @cache_readonly
    def var_rep(self):
        pi = self.alpha.dot(self.beta.T)
        gamma = self.gamma
        K = self.neqs
        A = np.zeros((self.k_ar, K, K))
        A[0] = pi + np.identity(K)
        if self.gamma.size > 0:
            A[0] += gamma[:, :K]
            A[self.k_ar-1] = - gamma[:, K*(self.k_ar-2):]
            for i in range(1, self.k_ar-1):
                A[i] = gamma[:, K*i:K*(i+1)] - gamma[:, K*(i-1):K*i]
        return A

    @cache_readonly
    def cov_var_repr(self):
        """
        Gives the covariance matrix of the vector consisting of the columns of
        the corresponding VAR coefficient matrices (i.e. vec(self.var_rep)).

        Returns
        -------
        cov : array (neqs**2 * k_ar x neqs**2 * k_ar)
        """
        # This implementation is using the fact that for a random variable x
        # with covariance matrix Sigma_x the following holds:
        # B @ x with B being a suitably sized matrix has the covariance matrix
        # B @ Sigma_x @ B.T. The arrays called vecm_var_transformation and
        # self.cov_params_wo_det in the code play the roles of B and Sigma_x
        # respectively. The elements of the random variable x are the elements
        # of the estimated matrices Pi (alpha @ beta.T) and Gamma.
        # Alternatively the following code (commented out) would yield the same
        # result (following p. 289 in Lutkepohl):
        # K, p = self.neqs, self.k_ar
        # w = np.identity(K * p)
        # w[np.arange(K, len(w)), np.arange(K, len(w))] *= (-1)
        # w[np.arange(K, len(w)), np.arange(len(w)-K)] = 1
        #
        # w_eye = np.kron(w, np.identity(K))
        #
        # return chain_dot(w_eye.T, self.cov_params, w_eye)

        if self.k_ar - 1 == 0:
            return self.cov_params_wo_det

        vecm_var_transformation = np.zeros((self.neqs**2 * self.k_ar,
                                            self.neqs**2 * self.k_ar))
        eye = np.identity(self.neqs**2)
        # for A_1:
        vecm_var_transformation[:self.neqs**2, :2*self.neqs**2] = hstack(
                (eye, eye))
        # for A_i, where i = 2, ..., k_ar-1
        for i in range(2, self.k_ar):
            start_row = self.neqs**2 + (i-2) * self.neqs**2
            start_col = self.neqs**2 + (i-2) * self.neqs**2
            vecm_var_transformation[start_row:start_row+self.neqs**2,
                start_col:start_col+2*self.neqs**2] = hstack((-eye, eye))
        # for A_p:
        vecm_var_transformation[-self.neqs**2:, -self.neqs**2:] = -eye
        return chain_dot(vecm_var_transformation, self.cov_params_wo_det,
                         vecm_var_transformation.T)

    def ma_rep(self, maxn=10):
        return ma_rep(self.var_rep, maxn)

    @cache_readonly
    def _chol_sigma_u(self):
        return np.linalg.cholesky(self.sigma_u)

    def orth_ma_rep(self, maxn=10, P=None):
        r"""Compute orthogonalized MA coefficient matrices using P matrix such
        that :math:`\\Sigma_u = PP^\\prime`. P defaults to the Cholesky
        decomposition of :math:`\\Sigma_u`

        Parameters
        ----------
        maxn : int
            Number of coefficient matrices to compute
        P : ndarray (neqs x neqs), optional
            Matrix such that Sigma_u = PP', defaults to Cholesky descomp

        Returns
        -------
        coefs : ndarray (maxn x neqs x neqs)
        """
        return orth_ma_rep(self, maxn, P)

    def predict(self, steps=5, alpha=None, exog_fc=None, exog_coint_fc=None):
        """

        Parameters
        ----------
        steps : int
            Prediction horizon.
        alpha : float between 0 and 1 or None
            If None, compute point forecast only.
            If float, compute confidence intervals too. In this case the
            argument stands for the confidence level.
        exog : ndarray (steps x self.exog.shape[1])
            If self.exog is not None, then information about the future values
            of exog have to be passed via this parameter. The ndarray may be
            larger in it's first dimension. In this case only the first steps
            rows will be considered.

        Returns
        -------
        forecast - ndarray (steps x neqs) or three ndarrays
            In case of a point forecast: each row of the returned ndarray
            represents the forecast of the neqs variables for a specific
            period. The first row (index [0]) is the forecast for the next
            period, the last row (index [steps-1]) is the steps-periods-ahead-
            forecast.
        """
        if self.exog is not None and exog_fc is None:
            raise ValueError("exog_fc is None: Please pass the future values "
                             "of the VECM's exog terms via the exog_fc "
                             "argument!")
        if self.exog is None and exog_fc is not None:
            raise ValueError("This VECMResult-instance's exog attribute is "
                             "None. Please don't pass a non-None value as the "
                             "method's exog_fc-argument.")
        if exog_fc is not None and exog_fc.shape[0] < steps:
            raise ValueError("The argument exog_fc must have at least steps "
                             "elements in its first dimension")

        if self.exog_coint is not None and exog_coint_fc is None:
            raise ValueError("exog_coint_fc is None: Please pass the future "
                             "values of the VECM's exog_coint terms via the "
                             "exog_coint_fc argument!")
        if self.exog_coint is None and exog_coint_fc is not None:
            raise ValueError("This VECMResult-instance's exog_coint attribute "
                             "is None. Please don't pass a non-None value as "
                             "the method's exog_coint_fc-argument.")
        if exog_coint_fc is not None and exog_coint_fc.shape[0] < steps - 1:
            raise ValueError("The argument exog_coint_fc must have at least "
                             "steps elements in its first dimension")

        last_observations = self.y_all.T[-self.k_ar:]
        exog = []
        trend_coefs = []

        # adding deterministic terms outside cointegration relation
        exog_const = np.ones(steps)
        nobs_tot = self.nobs + self.k_ar
        if self.const.size > 0:
            exog.append(exog_const)
            trend_coefs.append(self.const.T)

        if self.seasons > 0:
            first_future_season = (self.first_season + nobs_tot) % self.seasons
            exog_seasonal = seasonal_dummies(self.seasons, steps,
                                             first_future_season, True)
            exog.append(exog_seasonal)
            trend_coefs.append(self.seasonal.T)

        exog_lin_trend = _linear_trend(self.nobs, self.k_ar)
        exog_lin_trend = exog_lin_trend[-1] + 1 + np.arange(steps)
        if self.lin_trend.size > 0:
            exog.append(exog_lin_trend)
            trend_coefs.append(self.lin_trend.T)

        if exog_fc is not None:
            exog.append(exog_fc[:steps])
            trend_coefs.append(self.exog_coefs.T)

        # adding deterministic terms inside cointegration relation
        if "ci" in self.deterministic:
            exog.append(exog_const)
            trend_coefs.append(self.alpha.dot(self.const_coint.T).T)
        exog_lin_trend_coint = _linear_trend(self.nobs, self.k_ar, coint=True)
        exog_lin_trend_coint = exog_lin_trend_coint[-1] + 1 + np.arange(steps)
        if "li" in self.deterministic:
            exog.append(exog_lin_trend_coint)
            trend_coefs.append(self.alpha.dot(self.lin_trend_coint.T).T)

        if exog_coint_fc is not None:
            if exog_coint_fc.ndim == 1:
                exog_coint_fc = exog_coint_fc[:, None]  # make 2-D
            exog_coint_fc = np.vstack((self.exog_coint[-1:],
                                          exog_coint_fc[:steps-1]))
            exog.append(exog_coint_fc)
            trend_coefs.append(self.alpha.dot(self.exog_coint_coefs.T).T)

        # glueing all deterministics together
        exog = np.column_stack(exog) if exog != [] else None
        if trend_coefs != []:
            trend_coefs = np.row_stack(trend_coefs)
        else:
            trend_coefs = None

        # call the forecasting function of the VAR-module
        if alpha is not None:
            return forecast_interval(last_observations, self.var_rep,
                                     trend_coefs, self.sigma_u, steps,
                                     alpha=alpha,
                                     exog=exog)
        else:
            return forecast(last_observations, self.var_rep, trend_coefs,
                            steps, exog)

    def plot_forecast(self, steps, alpha=0.05, plot_conf_int=True,
                      n_last_obs=None):
        """
        Plot the forecast.

        Parameters
        ----------
        steps : int
            Prediction horizon.
        alpha : float between 0 and 1
            The confidence level.
        plot_conf_int : bool, default: True
            If True, plot bounds of confidence intervals.
        n_last_obs : int or None, default: None
            If int, restrict plotted history to n_last_obs observations.
            If None, include the whole history in the plot.
        """
        mid, lower, upper = self.predict(steps, alpha=alpha)

        y = self.y_all.T
        y = y[self.k_ar:] if n_last_obs is None else y[-n_last_obs:]
        plot.plot_var_forc(y, mid, lower, upper, names=self.names,
                           plot_stderr=plot_conf_int,
                           legend_options={"loc": "lower left"})

    def test_granger_causality(self, caused, causing=None, signif=0.05,
                               verbose=True):
        """
        Test for Granger-causality as described in chapter 7.6.3 of [1]_.
        Test H0: "`causing` does not Granger-cause the remaining variables of
        the system" against  H1: "`causing` is Granger-causal for the
        remaining variables".

        Parameters
        ----------
        caused : int or str or sequence of int or str
            If int or str, test whether the variable specified via this index
            (int) or name (str) is Granger-caused by the variable(s) specified
            by `causing`.
            If a sequence of int or str, test whether the corresponding
            variables are Granger-caused by the variable(s) specified
            by `causing`.
        causing : int or str or sequence of int or str or None, default: None
            If int or str, test whether the variable specified via this index
            (int) or name (str) is Granger-causing the variable(s) specified by
            `caused`.
            If a sequence of int or str, test whether the corresponding
            variables are Granger-causing the variable(s) specified by
            `caused`.
            If None, `causing` is assumed to be the complement of `caused`.
        signif : float between 0 and 1, default 5 %
            Significance level for computing critical values for test,
            defaulting to standard 0.95 level.
        verbose : bool
            If True, print a table with the results.

        Returns
        -------
        results : dict
            A dict holding the test's results. The dict's keys are:
            * "statistic" : float
                The claculated test statistic.
            * "crit_value" : float
                The critical value of the F-distribution.
            * "pvalue" : float
                The p-value corresponding to the test statistic.
            * "df" : float
                The degrees of freedom of the F-distribution.
            * "conclusion" : str {"reject", "fail to reject"}
                 Whether H0 can be rejected or not.
            * "signif" : float

        References
        ----------
        .. [1] Lutkepohl, H. 2005. *New Introduction to Multiple Time Series Analysis*. Springer.

        """
        if not (0 < signif < 1):
            raise ValueError("signif has to be between 0 and 1")

        allowed_types = (string_types, int)

        if isinstance(caused, allowed_types):
            caused = [caused]
        if not all(isinstance(c, allowed_types) for c in caused):
            raise TypeError("caused has to be of type string or int (or a "
                            "sequence of these types).")
        caused_ind = [get_index(self.names, c) for c in caused]

        if causing is not None:

            if isinstance(causing, allowed_types):
                causing = [causing]
            if not all(isinstance(c, allowed_types) for c in causing):
                raise TypeError("causing has to be of type string or int (or "
                                "a sequence of these types) or None.")
            causing = [self.names[c] if type(c) == int else c for c in causing]
            causing_ind = [get_index(self.names, c) for c in causing]

        if causing is None:
            causing_ind = [i for i in range(self.neqs) if i not in caused_ind]
            causing = [self.names[c] for c in causing_ind]

        y, k, t, p = self.y_all, self.neqs, self.nobs - 1, self.k_ar + 1
        exog = deterministic_to_exog(self.deterministic, self.seasons,
                                     nobs_tot=self.nobs + self.k_ar,
                                     first_season=self.first_season,
                                     seasons_centered=True, exog=self.exog,
                                     exog_coint=self.exog_coint)
        var_results = VAR(y.T, exog).fit(maxlags=p, trend="nc")

        # num_restr is called N in Lutkepohl
        num_restr = len(causing) * len(caused) * (p - 1)
        num_det_terms = num_det_vars(self.deterministic, self.seasons)
        if self.exog is not None:
            num_det_terms += self.exog.shape[1]
        if self.exog_coint is not None:
            num_det_terms += self.exog_coint.shape[1]

        # Make restriction matrix
        C = np.zeros((num_restr, k*num_det_terms + k**2 * (p-1)), dtype=float)
        cols_det = k * num_det_terms
        row = 0
        for j in range(p-1):
            for ing_ind in causing_ind:
                for ed_ind in caused_ind:
                    C[row, cols_det + ed_ind + k * ing_ind + k**2 * j] = 1
                    row += 1
        # print(C.shape)
        # print(var_results.params[:-k].shape)
        # a = np.vstack(vec(var_results.coefs[i])[:, None] for i in range(p-1))
        # Ca = np.dot(C, a)
        Ca = np.dot(C, vec(var_results.params[:-k].T))

        x_min_p_components = []
        if exog is not None:
            x_min_p_components.append(exog[-t:].T)
        # if "co" in self.deterministic or "ci" in self.deterministic:
        #     x_min_p_components.append(np.ones(t))
        # if "lo" in self.deterministic or "li" in self.deterministic:
        #     x_min_p_components.append(np.arange(t))
        # if self.seasons > 0:
        #     x_min_p_components.append(seasonal_dummies(self.seasons, t,
        #                                                first_period=1,
        #                                                centered=False).T)
        x_min_p = np.zeros((k * p, t))
        for i in range(p-1):  # fll first k * k_ar rows of x_min_p
            x_min_p[i*k:(i+1)*k, :] = y[:, p-1-i:-1-i] - y[:, :-p]
        x_min_p[-k:, :] = y[:, :-p]  # fill last rows of x_min_p
        x_min_p_components.append(x_min_p)
        x_min_p = np.row_stack(x_min_p_components)
        x_x = np.dot(x_min_p, x_min_p.T)  # k*k_ar x k*k_ar
        x_x_11 = inv(x_x)[:k * (p-1) + num_det_terms, :k * (p-1) + num_det_terms]  # k*(k_ar-1) x k*(k_ar-1)
        # For VAR-models with parameter restrictions the denominator in the
        # calculation of sigma_u is nobs and not (nobs-k*k_ar-num_det_terms).
        # Testing for Granger-causality means testing for restricted
        # parameters, thus the former of the two denominators is used. As
        # Lutkepohl states, both variants of the estimated sigma_u are
        # possible. (see Lutkepohl, p.198)
        # The choice of the denominator T has also the advantage of getting the
        # same results as the reference software JMulTi.
        sigma_u = var_results.sigma_u * (t-k*p-num_det_terms) / t
        sig_alpha_min_p = t * np.kron(x_x_11, sigma_u)  # k**2*(p-1)xk**2*(p-1)
        middle = inv(chain_dot(C, sig_alpha_min_p, C.T))

        wald_statistic = t * chain_dot(Ca.T, middle, Ca)
        f_statistic = wald_statistic / num_restr
        df = (num_restr, k * var_results.df_resid)
        f_distribution = scipy.stats.f(*df)

        pvalue = f_distribution.sf(f_statistic)
        crit_value = f_distribution.ppf(1 - signif)
        conclusion = 'fail to reject' if f_statistic < crit_value else 'reject'

        results = {
            "statistic": f_statistic,
            "crit_value": crit_value,
            "pvalue": pvalue,
            "df": df,
            "conclusion": conclusion,
            "signif":  signif
        }

        if verbose:
            summ = var_output.causality_summary(results, causing, caused, "f")
            print(summ)

        return results

    def test_inst_causality(self, causing, signif=0.05, verbose=True):
        """
        Test for instantaneous causality as described in chapters 3.6.3 and
        7.6.4 of [1]_.
        Test H0: "No instantaneous causality between caused and causing"
        against H1: "Instantaneous causality between caused and causing
        exists".
        Note that instantaneous causality is a symmetric relation
        (i.e. if causing is "instantaneously causing" caused, then also caused
        is "instantaneously causing" causing), thus the naming of the
        parameters (which is chosen to be in accordance with
        test_granger_causality()) may be misleading.

        Parameters
        ----------
        causing :
            If int or str, test whether the corresponding variable is causing
            the variable(s) specified in caused.
            If sequence of int or str, test whether the corresponding variables
            are causing the variable(s) specified in caused.
        signif : float between 0 and 1, default 5 %
            Significance level for computing critical values for test,
            defaulting to standard 0.95 level
        verbose : bool
            If True, print a table with the results.

        Returns
        -------
        results : dict
            A dict holding the test's results. The dict's keys are:
            * "statistic" : float
                The claculated test statistic.
            * "crit_value" : float
                The critical value of the \Chi^2-distribution.
            * "pvalue" : float
                The p-value corresponding to the test statistic.
            * "df" : float
                The degrees of freedom of the \Chi^2-distribution.
            * "conclusion" : str {"reject", "fail to reject"}
                 Whether H0 can be rejected or not.
            * "signif" : float

        Notes
        -----
        This method is not returning the same result as JMulTi. This is because
        the test is based on a VAR(k_ar) model in statsmodels (in accordance to
        pp. 104, 320-321 in [1]_) whereas JMulTi seems to be using a
        VAR(k_ar+1) model.

        References
        ----------
        .. [1] Lutkepohl, H. 2005. *New Introduction to Multiple Time Series Analysis*. Springer.
        """
        exog = deterministic_to_exog(self.deterministic, self.seasons,
                                     nobs_tot=self.nobs + self.k_ar,
                                     first_season=self.first_season,
                                     seasons_centered=True, exog=self.exog,
                                     exog_coint=self.exog_coint)

        # Note: JMulTi seems to be using k_ar+1 instead of k_ar
        k, t, p = self.neqs, self.nobs, self.k_ar
        # fit with trend "nc" because all trend information is already in exog
        var_results = VAR(self.y_all.T, exog).fit(maxlags=p, trend="nc")
        return var_results.test_inst_causality(causing=causing, signif=signif,
                                               verbose=verbose,
                                               names=self.names)

    def irf(self, periods=10):
        return irf.IRAnalysis(self, periods=periods, vecm=True)

    @cache_readonly
    def fittedvalues(self):
        """
        Returns
        -------
        fitted : array (nobs x neqs)
            The predicted in-sample values of the models' endogenous variables.
        """
        beta = self.beta
        if self.det_coef_coint.size > 0:
            beta = vstack((beta, self.det_coef_coint))
        pi = np.dot(self.alpha, beta.T)

        gamma = self.gamma
        if self.det_coef.size > 0:
            gamma = hstack((gamma, self.det_coef))
        delta_y = np.dot(pi, self._y_min1) + np.dot(gamma, self._delta_x)
        return (delta_y + self._y_min1[:self.neqs]).T

    @cache_readonly
    def resid(self):
        """
        Returns
        -------
        resid : array (nobs x neqs)
            The residuals.
        """
        return self.y_all.T[self.k_ar:] - self.fittedvalues

    def test_normality(self, signif=0.05, verbose=True):
        """
        Test assumption of normal-distributed errors using Jarque-Bera-style
        omnibus Chi^2 test

        Parameters
        ----------
        signif : float
            Test significance threshold
        verbose : bool
            If True, print summary with the test's results.

        Returns
        -------
        result : dict
            A dictionary with the test's results as key-value-pairs. The keys
            are 'statistic', 'crit_value', 'pvalue', 'df', 'conclusion', and
            'signif'.

        Notes
        -----
        H0 (null) : data are generated by a Gaussian-distributed process
        """
        return test_normality(self, signif=signif, verbose=verbose)

    def test_whiteness(self, nlags=10, signif=0.05, adjusted=False):
        """
        Test the whiteness of the residuals using the Portmanteau test as
        described in [1]_, chapter 8.4.1.

        Parameters
        ----------
        nlags : int > 0
        signif : float, between 0 and 1
        adjusted : bool, default False

        Returns
        -------
        result : dict
            A dictionary with the test's results as key-value-pairs. The keys
            are 'statistic', 'crit_value', 'pvalue', 'df', 'conclusion', and
            'signif'.

        References
        ----------
        .. [1] Lutkepohl, H. 2005. *New Introduction to Multiple Time Series Analysis*. Springer.
        """
        def cov(lag):
            """
            Parameters
            ----------
            lag : int >= 0

            Returns
            -------
            result : ndarray (neqs, neqs)
                The estimated autocovariance matrix of :math:`u_t` for lag
                `lag`.
            """
            u = np.asarray(self.resid).T
            u -= np.mean(u, axis=1).reshape((u.shape[0], 1))
            result = np.zeros((self.neqs, self.neqs))
            for t in range(lag, self.nobs):
                result += u[:, t:t+1].dot(u[:, t-lag:t-lag+1].T)
            result /= self.nobs
            return result

        statistic = 0
        # self.sigma_u instead of cov(0) is necessary to get the same
        # result as JMulTi. The difference between the two is that sigma_u is
        # calculated with the usual residuals while in cov(0) the
        # residuals are demeaned. To me JMulTi's behaviour seems a bit strange
        # because it uses the usual residuals here but demeaned residuals in
        # the calculation of autocovariances with lag > 0. (used in the
        # argument of trace() four rows below this comment.)
        c0_inv = inv(self.sigma_u)  # instead of inv(cov(0))
        if c0_inv.dtype == np.complex128 and np.all(np.imag(c0_inv) == 0):
            c0_inv = np.real(c0_inv)
        for t in range(1, nlags+1):
            ct = cov(t)
            to_add = np.trace(chain_dot(ct.T, c0_inv, ct, c0_inv))
            if adjusted:
                to_add /= (self.nobs - t)
            statistic += to_add
        statistic *= self.nobs**2 if adjusted else self.nobs

        df = self.neqs**2 * (nlags - self.k_ar + 1) - self.neqs * self.r
        dist = scipy.stats.chi2(df)
        pvalue = dist.sf(statistic)
        crit_value = dist.ppf(1 - signif)

        if statistic < crit_value:
            conclusion = 'fail to reject'
        else:
            conclusion = 'reject'
        results = {
            'statistic': statistic,
            'crit_value': crit_value,
            'pvalue': pvalue,
            'df': df,
            'conclusion': conclusion,
            'signif': signif
        }
        return results

    def plot_data(self, with_presample=False):
        """
        Plots the input time series.

        Parameters
        ----------
        with_presample : bool, default: False
            If False, the pre-sample data (the first k_ar values) will not be
            plotted.
        """
        y = self.y_all if with_presample else self.y_all[:, self.k_ar:]
        names = self.names
        dates = self.dates if with_presample else self.dates[self.k_ar:]
        plot.plot_mts(y.T, names=names, index=dates)

    def summary(self, alpha=.05):
        """
        Parameters
        ----------
        alpha : float between 0 and 1, default 0.05
            Significance level of the shown confidence intervals.

        Returns
        -------
        summary : statsmodels.iolib.summary.Summary
            A summary containing information about estimated parameters.
        """
        from statsmodels.iolib.summary import summary_params

        summary = Summary()

        def make_table(self, params, std_err, t_values, p_values, conf_int,
                       mask, names, title, strip_end=True):
            res = (self,
                   params[mask],
                   std_err[mask],
                   t_values[mask],
                   p_values[mask],
                   conf_int[mask]
                   )
            param_names = [
                '.'.join(name.split('.')[:-1]) if strip_end else name
                for name in np.array(names)[mask].tolist()]
            return summary_params(res, yname=None, xname=param_names,
                                  alpha=alpha, use_t=False, title=title)

        # ---------------------------------------------------------------------
        # Add tables with gamma and det_coef for each endogenous variable:
        lagged_params_components = []
        stderr_lagged_params_components = []
        tvalues_lagged_params_components = []
        pvalues_lagged_params_components = []
        conf_int_lagged_params_components = []
        if self.det_coef.size > 0:
            lagged_params_components.append(self.det_coef.flatten(order="F"))
            stderr_lagged_params_components.append(
                    self.stderr_det_coef.flatten(order="F"))
            tvalues_lagged_params_components.append(
                    self.tvalues_det_coef.flatten(order="F"))
            pvalues_lagged_params_components.append(
                    self.pvalues_det_coef.flatten(order="F"))
            conf_int = self.conf_int_det_coef(alpha=alpha)
            lower = conf_int["lower"].flatten(order="F")
            upper = conf_int["upper"].flatten(order="F")
            conf_int_lagged_params_components.append(np.column_stack(
                    (lower, upper)))
        if self.k_ar - 1 > 0:
            lagged_params_components.append(self.gamma.flatten())
            stderr_lagged_params_components.append(self.stderr_gamma.flatten())
            tvalues_lagged_params_components.append(
                    self.tvalues_gamma.flatten())
            pvalues_lagged_params_components.append(
                    self.pvalues_gamma.flatten())
            conf_int = self.conf_int_gamma(alpha=alpha)
            lower = conf_int["lower"].flatten()
            upper = conf_int["upper"].flatten()
            conf_int_lagged_params_components.append(np.column_stack(
                    (lower, upper)))

        # if gamma or det_coef exists, then make a summary-table for them:
        if len(lagged_params_components) != 0:
            lagged_params = hstack(lagged_params_components)
            stderr_lagged_params = hstack(stderr_lagged_params_components)
            tvalues_lagged_params = hstack(tvalues_lagged_params_components)
            pvalues_lagged_params = hstack(pvalues_lagged_params_components)
            conf_int_lagged_params = vstack(conf_int_lagged_params_components)

            for i in range(self.neqs):
                masks = []
                offset = 0
                # 1. Deterministic terms outside cointegration relation
                if "co" in self.deterministic:
                    masks.append(offset + np.array(i, ndmin=1))
                    offset += self.neqs
                if self.seasons > 0:
                    for _ in range(self.seasons-1):
                        masks.append(offset + np.array(i, ndmin=1))
                        offset += self.neqs
                if "lo" in self.deterministic:
                    masks.append(offset + np.array(i, ndmin=1))
                    offset += self.neqs
                if self.exog is not None:
                    for _ in range(self.exog.shape[1]):
                        masks.append(offset + np.array(i, ndmin=1))
                        offset += self.neqs
                # 2. Lagged endogenous terms
                if self.k_ar - 1 > 0:
                    start = i * self.neqs * (self.k_ar-1)
                    end = (i+1) * self.neqs * (self.k_ar-1)
                    masks.append(offset + np.arange(start, end))
                    # offset += self.neqs**2 * (self.k_ar-1)

                # Create the table
                mask = np.concatenate(masks)
                eq_name = self.model.endog_names[i]
                title = "Det. terms outside the coint. relation " + \
                        "& lagged endog. parameters for equation %s" % eq_name
                table = make_table(self, lagged_params, stderr_lagged_params,
                                   tvalues_lagged_params,
                                   pvalues_lagged_params,
                                   conf_int_lagged_params, mask,
                                   self.model.lagged_param_names, title)
                summary.tables.append(table)

        # ---------------------------------------------------------------------
        # Loading coefficients (alpha):
        a = self.alpha.flatten()
        se_a = self.stderr_alpha.flatten()
        t_a = self.tvalues_alpha.flatten()
        p_a = self.pvalues_alpha.flatten()
        ci_a = self.conf_int_alpha(alpha=alpha)
        lower = ci_a["lower"].flatten()
        upper = ci_a["upper"].flatten()
        ci_a = np.column_stack((lower, upper))
        a_names = self.model.load_coef_param_names
        alpha_masks = []
        for i in range(self.neqs):
            if self.r > 0:
                start = i * self.r
                end = start + self.r
                mask = np.arange(start, end)

            # Create the table
            alpha_masks.append(mask)

            eq_name = self.model.endog_names[i]
            title = "Loading coefficients (alpha) for equation %s" % eq_name
            table = make_table(self, a, se_a, t_a, p_a, ci_a, mask, a_names,
                               title)
            summary.tables.append(table)

        # ---------------------------------------------------------------------
        # Cointegration matrix/vector (beta) and det. terms inside coint. rel.:
        coint_components = []
        stderr_coint_components = []
        tvalues_coint_components = []
        pvalues_coint_components = []
        conf_int_coint_components = []
        if self.r > 0:
            coint_components.append(self.beta.T.flatten())
            stderr_coint_components.append(self.stderr_beta.T.flatten())
            tvalues_coint_components.append(self.tvalues_beta.T.flatten())
            pvalues_coint_components.append(self.pvalues_beta.T.flatten())
            conf_int = self.conf_int_beta(alpha=alpha)
            lower = conf_int["lower"].T.flatten()
            upper = conf_int["upper"].T.flatten()
            conf_int_coint_components.append(np.column_stack(
                    (lower, upper)))
        if self.det_coef_coint.size > 0:
            coint_components.append(self.det_coef_coint.flatten())
            stderr_coint_components.append(
                    self.stderr_det_coef_coint.flatten())
            tvalues_coint_components.append(
                    self.tvalues_det_coef_coint.flatten())
            pvalues_coint_components.append(
                    self.pvalues_det_coef_coint.flatten())
            conf_int = self.conf_int_det_coef_coint(alpha=alpha)
            lower = conf_int["lower"].flatten()
            upper = conf_int["upper"].flatten()
            conf_int_coint_components.append(np.column_stack((lower, upper)))
        coint = hstack(coint_components)
        stderr_coint = hstack(stderr_coint_components)
        tvalues_coint = hstack(tvalues_coint_components)
        pvalues_coint = hstack(pvalues_coint_components)
        conf_int_coint = vstack(conf_int_coint_components)
        coint_names = self.model.coint_param_names

        for i in range(self.r):
            masks = []
            offset = 0

            # 1. Cointegration matrix (beta)
            if self.r > 0:
                start = i * self.neqs
                end = start + self.neqs
                masks.append(offset + np.arange(start, end))
                offset += self.neqs * self.r

            # 2. Deterministic terms inside cointegration relation
            if "ci" in self.deterministic:
                masks.append(offset + np.array(i, ndmin=1))
                offset += self.r
            if "li" in self.deterministic:
                masks.append(offset + np.array(i, ndmin=1))
                offset += self.r
            if self.exog_coint is not None:
                for _ in range(self.exog_coint.shape[1]):
                    masks.append(offset + np.array(i, ndmin=1))
                    offset += self.r

            # Create the table
            mask = np.concatenate(masks)
            title = "Cointegration relations for " + \
                    "loading-coefficients-column %d" % (i+1)
            table = make_table(self, coint, stderr_coint, tvalues_coint,
                               pvalues_coint, conf_int_coint, mask,
                               coint_names, title)
            summary.tables.append(table)

        return summary
