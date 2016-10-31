from __future__ import division, print_function

import collections
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
from statsmodels.tsa.vector_ar.util import vech, get_index, seasonal_dummies
from statsmodels.tsa.vector_ar.var_model import forecast, forecast_interval, \
    VAR, ma_rep, orth_ma_rep, test_normality


def select_order(data, maxlags, deterministic="nc", seasons=0, verbose=True):
    """
    Compute lag order selections based on each of the available information
    criteria.

    Parameters
    ----------
    data : array (nobs_tot x neqs)
        The observed data.
    maxlags : int
    deterministic : str {"nc", "co", "ci", "lo", "li"}
        * "nc" - no deterministic terms
        * "co" - constant outside the cointegration relation
        * "ci" - constant within the cointegration relation
        * "lo" - linear trend outside the cointegration relation
        * "li" - linear trend within the cointegration relation

        Combinations of these are possible (e.g. "cili" or "colo" for linear
        trend with intercept)
    seasons : int
        Number of seasons.
    verbose : bool, default True
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
        # exclude some periods to same amount of data used for each lag
        # order
        # TODO: pass deterministic and seasons as parameter to VAR()
        exog = []
        if "co" in deterministic or "ci" in deterministic:
            exog.append(np.ones(len(data)).reshape(-1, 1))
        if "lo" in deterministic or "li" in deterministic:
            exog.append(np.arange(len(data)).reshape(-1, 1))
        if seasons > 0:
            exog.append(seasonal_dummies(seasons, len(data)).reshape(-1, seasons-1))
        exog = hstack(exog) if exog else None
        var_model = VAR(data, exog)
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


def num_det_vars(det_string, seasons=0):
    """Gives the number of deterministic variables.

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


def deterministic_to_exog(deterministic, seasons, len_data,
                          first_season=0, seasons_centered=False):
            exog = []
            if "co" in deterministic or "ci" in deterministic:
                exog.append(np.ones(len_data))
            if "lo" in deterministic or "li" in deterministic:
                exog.append(np.arange(len_data))
            if seasons > 0:
                exog.append(seasonal_dummies(seasons, len_data,
                                             first_period=first_season,
                                             centered=seasons_centered))
            return np.column_stack(exog) if exog else None


def mat_sqrt(_2darray):
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


def _endog_matrices(endog_tot, diff_lags, deterministic, seasons=0,
                    first_season=0):
    """Returns different matrices needed for parameter estimation (compare p.
    186 in [1]_). These matrices consist of elements of the data as well as
    elements representing deterministic terms. A tuple of consisting of these
    matrices is returned.

    Parameters
    ----------
    endog_tot : ndarray (neqs x total_nobs)
        The whole sample including the presample.
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
        # y_min1_mean = y_min1.mean(1)
        y_min1_stack.append(np.ones(T))
        # H = vstack((np.identity(neqs),
        #             - y_min1_mean))
        # y_min1 = H.T.dot(y_min1)

    if "li" in deterministic:  # p. 299
        y_min1_stack.append(np.arange(T) + p)
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
        delta_x_stack.append(np.arange(T)+1)
    delta_x = np.row_stack(delta_x_stack)
    return y_1_T, delta_y_1_T, y_min1, delta_x


def _block_matrix_ymin1_deltax(y_min1, delta_x):  # e.g. p.287 (7.2.4)
    """Returns an ndarray needed for parameter estimation as well as the
    calculation of standard errors.

    Parameters
    ----------
    y_min1 : ndarray (neqs x nobs)
        (dimensions assuming no deterministic terms are given)
        .. math:: (y_0, \ldots, y_{T-1}
    delta_x : ndarray (diff_lags*neqs x nobs)
        (dimensions assuming no deterministic terms are given)

    Returns
    -------
    result : ndarray (neqs*k_ar x neqs*k_ar)
        (dimensions assuming no deterministic terms are given)
        Inverse of a matrix consisting of four blocks. Each block is consists
        of matrix products of the function's arguments.
    """
    b = y_min1.dot(delta_x.T)
    return inv(vstack((hstack((y_min1.dot(y_min1.T), b)),
                       hstack((b.T, delta_x.dot(delta_x.T))))))


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
        .. math:: (y_1, \ldots, y_T) - (y_0, \ldots, y_{T-1})
    y_min1 : ndarray (neqs x nobs)
        (dimensions assuming no deterministic terms are given)
        .. math:: (y_0, \ldots, y_{T-1}

    Returns
    -------
    result : tuple
        A tuple of two ndarrays
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
        .. math:: (y_1, \ldots, y_T) - (y_0, \ldots, y_{T-1})
    y_min1 : ndarray (neqs x nobs)
        (dimensions assuming no deterministic terms are given)
        .. math:: (y_0, \ldots, y_{T-1}

    Returns
    -------
    result : tuple
        A tuple of five ndarrays as well as eigenvalues and -vectors of a
        certain (matrix) product of some of the returned ndarrays.
    """
    T = y_min1.shape[1]
    r0, r1 = _r_matrices(T, delta_x, delta_y_1_T, y_min1)
    s00 = np.dot(r0, r0.T) / T
    s01 = np.dot(r0, r1.T) / T
    s10 = s01.T
    s11 = np.dot(r1, r1.T) / T
    s11_ = inv(mat_sqrt(s11))
    # p. 295:
    s01_s11_ = np.dot(s01, s11_)
    eig = np.linalg.eig(chain_dot(s01_s11_.T, inv(s00), s01_s11_))
    lambd = eig[0]
    v = eig[1]
    return s00, s01, s10, s11, s11_, lambd, v


# VECM class: for known or unknown VECM

class VECM(tsbase.TimeSeriesModel):
    """
    Fit a VECM process
    .. math:: \Delta y_t = \Pi y_{t-1} + \Gamma_1 \Delta y_{t-1} + \ldots + \Gamma_{k_ar-1} \Delta y_{t-k_ar+1} + u_t
    where
    .. math:: \Pi = \alpha \beta'
    as described in chapter 7 of [1]_.

    Parameters
    ----------
    endog_tot : array-like
        2-d endogenous response variable.
    dates : array-like
        must match number of rows of endog
    diff_lags : int
        Number of lags in the VEC representation
    deterministic : str {"nc", "co", "ci", "lo", "li"}
        * "nc" - no deterministic terms
        * "co" - constant outside the cointegration relation
        * "ci" - constant within the cointegration relation
        * "lo" - linear trend outside the cointegration relation
        * "li" - linear trend within the cointegration relation

        Combinations of these are possible (e.g. "cili" or "colo" for linear
        trend with intercept)
    seasons : int
        Number of seasons. 0 (default) means no seasons.

    References
    ----------
    .. [1] Lutkepohl, H. 2005. *New Introduction to Multiple Time Series Analysis*. Springer.
    """

    def __init__(self, endog_tot, dates=None, freq=None, missing="none",
                 diff_lags=1, coint_rank=1, deterministic="nc", seasons=0,
                 first_season=0):
        super(VECM, self).__init__(endog_tot, None, dates, freq,
                                   missing=missing)
        if self.endog.ndim == 1:
            raise ValueError("Only gave one variable to VECM")
        self.y = self.endog.T  # TODO delete this line if y not necessary
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
        method : {"ls", "egls", "ml"}
            Estimation method to use.
        coint_rank : int
            Cointegration rank, equals the rank of the matrix \Pi and the
            number of columns of \alpha and \beta

        Returns
        -------
        est : VECMResults

        References
        -----
        [1]_ pp. 269-304
        """
        if method == "ls":
            return self._estimate_vecm_ls(self.diff_lags, self.deterministic,
                                          self.seasons, self.first_season)
        elif method == "egls":
            return self._estimate_vecm_egls(self.diff_lags, self.deterministic,
                                            self.seasons, self.coint_rank,
                                            self.first_season)
        elif method == "ml":
            return self._estimate_vecm_ml(self.diff_lags, self.deterministic,
                                          self.seasons, self.coint_rank,
                                          self.first_season)
        else:
            raise ValueError("%s not recognized, must be among %s"
                             % (method, ("ls", "egls", "ml")))

    def _ls_pi_gamma(self, delta_y_1_T, y_min1, delta_x, diff_lags,
                     deterministic):
        K = delta_y_1_T.shape[0]
        T = delta_y_1_T.shape[1]

        mat1 = hstack((delta_y_1_T.dot(y_min1.T), delta_y_1_T.dot(delta_x.T)))
        mat2 = _block_matrix_ymin1_deltax(y_min1, delta_x)
        est_pi_gamma = mat1.dot(mat2)  # p. 287 (equation (7.2.4))

        pi_cols = K
        if "ci" in deterministic:
            pi_cols += 1
        if "li" in deterministic:
            pi_cols += 1
        pi_hat, gamma_hat = np.hsplit(est_pi_gamma, [pi_cols])

        _A = delta_y_1_T - pi_hat.dot(y_min1) - gamma_hat.dot(delta_x)
        p = diff_lags+1
        sigma_u_hat = 1/(T-K*p) * np.dot(_A, _A.T)  # p. 287 (equation (7.2.5))

        return pi_hat, gamma_hat, sigma_u_hat

    def _estimate_vecm_ls(self, diff_lags, deterministic="nc", seasons=0,
                          first_season=0):
        # deterministic \in \{"c", "lo", \}, where
        # c=constant, lt=linear trend, s=seasonal terms
        y_1_T, delta_y_1_T, y_min1, delta_x = _endog_matrices(
                self.y, diff_lags, deterministic, seasons, first_season)

        pi_hat, gamma_hat, sigma_u_hat = self._ls_pi_gamma(delta_y_1_T, y_min1,
                                                           delta_x, diff_lags,
                                                           deterministic)
        return {"Pi_hat": pi_hat, "Gamma_hat": gamma_hat,
                "Sigma_u_hat": sigma_u_hat}
    
    def _estimate_vecm_egls(self, diff_lags, deterministic="nc", seasons=0,
                            r=1, first_season=0):
        y_1_T, delta_y_1_T, y_min1, delta_x = _endog_matrices(
                self.y, diff_lags, deterministic, seasons, first_season)
        T = y_1_T.shape[1]
        
        pi_hat, _gamma_hat, sigma_u_hat = self._ls_pi_gamma(delta_y_1_T,
                                                            y_min1, delta_x,
                                                            diff_lags,
                                                            deterministic)
        alpha_hat = pi_hat[:, :r]

        r0, r1 = _r_matrices(T, delta_x, delta_y_1_T, y_min1)
        r11 = r1[:r]
        r12 = r1[r:]
        _alpha_Sigma = alpha_hat.T.dot(inv(sigma_u_hat))
        # p. 292:
        beta_hhat = inv(_alpha_Sigma.dot(alpha_hat)).dot(_alpha_Sigma).dot(
                r0-alpha_hat.dot(r11)).dot(r12.T).dot(inv(r12.dot(r12.T))).T
        beta_hhat = vstack((np.identity(r),
                            beta_hhat))

        # ? Gamma_hhat necessary / computed via
        # (delta_y_1_T - alpha_hat.dot(beta_hhat.T).dot(y_min1)).dot(
        #     delta_x.dot(inv(np.dot(delta_x,delta_x.T))))
        
        # Gamma_hhat = 
        # TODO: Gamma?
        
        return {"alpha": alpha_hat, "beta": beta_hhat, 
                "Gamma": _gamma_hat, "Sigma_u": sigma_u_hat}
    
    def _estimate_vecm_ml(self, diff_lags, deterministic="nc", seasons=0, r=1,
                          first_season=0):
        y_1_T, delta_y_1_T, y_min1, delta_x = _endog_matrices(
                self.y, diff_lags, deterministic, seasons, first_season)
        T = y_1_T.shape[1]

        s00, s01, s10, s11, s11_, _, v = _sij(delta_x, delta_y_1_T, y_min1)

        beta_tilde = (v[:, :r].T.dot(s11_)).T
        # normalize beta tilde such that eye(r) forms the first r rows of it:
        beta_tilde = np.dot(beta_tilde, inv(beta_tilde[:r]))
        alpha_tilde = s01.dot(beta_tilde).dot(
                inv(beta_tilde.T.dot(s11).dot(beta_tilde)))
        gamma_tilde = (delta_y_1_T - alpha_tilde.dot(beta_tilde.T).dot(y_min1)
                       ).dot(delta_x.T).dot(inv(np.dot(delta_x, delta_x.T)))
        temp = (delta_y_1_T - alpha_tilde.dot(beta_tilde.T).dot(y_min1) -
                gamma_tilde.dot(delta_x))
        sigma_u_tilde = temp.dot(temp.T) / T

        return VECMResults(self.y, self.p, r, alpha_tilde, beta_tilde,
                           gamma_tilde, sigma_u_tilde,
                           deterministic=deterministic, seasons=seasons,
                           delta_y_1_T=delta_y_1_T, y_min1=y_min1,
                           delta_x=delta_x, model=self, names=self.endog_names,
                           dates=self.data.dates,
                           first_season=self.first_season)

    @property
    def lagged_param_names(self):
        """

        Returns
        -------
        param_names : list of str
            Returns a list of parameter names for the lagged endogenous
            parameters which are called Gamma in [1]_ (see chapter 6).
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
                            for n in self.endog_names
                            for s in range(1, self.seasons)]

        if "lo" in self.deterministic:
            param_names += ["lin_trend.%s" % n for n in self.endog_names]

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
            which are called alpha in [1]_ (see chapter 6).

        References
        ----------
        .. [1] Lutkepohl, H. 2005. *New Introduction to Multiple Time Series Analysis*. Springer.
        """
        param_names = []

        if self.coint_rank == 0:
            return None

        # loading coefficients (alpha) # called "ec" in JMulTi, "ECT" in tsDyn
        param_names += [               # called "_ce" in Stata
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

        param_names += [("beta.%d." + self.load_coef_repr + "%d") % (j, i+1)
                        for i in range(self.coint_rank)
                        for j in range(self.neqs)]

        # 2. deterministic terms inside cointegration relation
        if "ci" in self.deterministic:
            param_names += ["const." + self.load_coef_repr + "%d" % (i+1)
                            for i in range(self.coint_rank)]

        if "li" in self.deterministic:
            param_names += ["lin_trend." + self.load_coef_repr + "%d" % (i+1)
                            for i in range(self.coint_rank)]

        return param_names

    @property
    def sigma2_param_names(self, error_cov_type="unstructured"):
        """

        Parameters
        ----------
        error_cov_type : str {"diagonal", "unstructured"}
            If "diagonal", the variance of each variable is returned.
            If "unstructured", the covariance of each combination of variables
            is returned.

        Returns
        -------
        param_names : list of str
            Returns a list of parameter names.
        """
        param_names = []

        if self.error_cov_type == 'diagonal':
            param_names += [
                'sigma2.%s' % self.endog_names[i]
                for i in range(self.neqs)
            ]
        elif self.error_cov_type == 'unstructured':
            param_names += [
                ('sqrt.var.%s' % self.endog_names[i] if i == j else
                 'sqrt.cov.%s.%s' % (self.endog_names[j], self.endog_names[i]))
                for i in range(self.neqs)
                for j in range(i+1)
            ]
        else:
            raise ValueError("error_cov_type has to be either \"diagonal\" " +
                             "or \"unstructured\".")
        return param_names

# -----------------------------------------------------------------------------
# VECMResults class

class VECMResults(object):
    """Class holding estimation related results of a vector error correction
    model (VECM).

    Parameters
    ----------
    endog_tot : array

    level_var_lag_order : int

    coint_rank : int

    alpha : array (neqs x coint_rank)
    beta : array (neqs x coint_rank)
    gamma : array (neqs x neqs*(level_var_lag_order-1))
    sigma_u : array (neqs x neqs)
    deterministic : str {"nc", "co", "ci", "lo", "li"}
        * "nc" - no deterministic terms
        * "co" - constant outside the cointegration relation
        * "ci" - constant within the cointegration relation
        * "lo" - linear trend outside the cointegration relation
        * "li" - linear trend within the cointegration relation

        Combinations of these are possible (e.g. "cili" or "colo" for linear
        trend with intercept)
    seasons : int
        Number of seasons. 0 (default) means no seasons.
    model : VECM
        An instance of the VECM class.

    Returns
    -------
    **Attributes**

    y_all
    alpha
    beta
    gamma
    sigma_u
        Estimate of white noise process variance Var[u_t]

    deterministic
    neqs : int
        Number of variables per observation. Number of equations.
    k_ar : int
        Lags in the VAR representation. This implies: Lags in the VEC
        representation = k_ar - 1
    r : int
        Cointegration rank.
    T : int
        Number of observations after the presample

    y_min1 : ndarray (neqs x T)
        Observations at t=0 until t=T-1
    delta_y_1_T : ndarray (neqs x T)
        Observations at t=1 until t=T minus y_min1
    delta_x : ndarray ((neqs * (k_ar-1) + number of deterministic dummy variables
        outside the cointegration relation) x T)

    llf

    _cov_sigma
    num_det_coef_coint : int
        Number of estimated coefficients for deterministic terms within the
        cointegration relation

    cov_params : ndarray (d x d)
        ... where d equals neqs * (neqs+num_det_coef_coint + neqs*(k_ar-1)+number of
        deterministic dummy variables outside the cointegration relation)
    stderr_params : ndarray (d)
        ... where d is defined as for cov_params
    stderr_coint : ndarray (neqs+num_det_coef_coint x r)
    stderr_alpha ndarray (neqs x r)
    stderr_beta : ndarray (neqs x r)
    stderr_det_coef_coint ndarray (num_det_coef_coint x r)
    stderr_gamma : ndarray (neqs x neqs*(k_ar-1))
    stderr_det_coef : ndarray (neqs x number of deterministic dummy variables
        outside the cointegration relation)
    tvalues_alpha : ndarray (neqs x r)
    tvalues_beta : ndarray (neqs x r)
    tvalues_det_coef_coint
    tvalues_gamma : ndarray (neqs x neqs*(k_ar-1))
    tvalues_det_coef : ndarray (neqs x number of deterministic dummy variables
        outside the cointegration relation)
    pvalues_alpha : ndarray (neqs x r)
    pvalues_beta : ndarray (neqs x r)
    pvalues_det_coef_coint
    pvalues_gamma : ndarray (neqs x neqs*(k_ar-1))
    pvalues_det_coef : ndarray (neqs x number of deterministic dummy variables
        outside the cointegration relation)
    var_repr : (k_ar x neqs x neqs)
        KxK matrices A_i of the corresponding VAR representation. If the return
        value is assigned to a variable A, these matrices can be accessed via
        A[i], i=0, ..., k_ar-1.
    """

    def __init__(self, endog_tot, level_var_lag_order, coint_rank, alpha, beta,
                 gamma, sigma_u, deterministic='nc', seasons=0, first_season=0,
                 delta_y_1_T=None, y_min1=None, delta_x=None, model=None,
                 names=None, dates=None):
        self.model = model
        self.y_all = endog_tot
        self.names = names
        self.dates = dates
        self.neqs = endog_tot.shape[0]
        self.k_ar = level_var_lag_order
        self.deterministic = deterministic
        self.seasons = seasons
        self.first_season = first_season

        self.r = coint_rank
        self.alpha = alpha
        self.beta, self.det_coef_coint = np.vsplit(beta, [self.neqs])
        self.gamma, self.det_coef = np.hsplit(gamma, [self.neqs * (self.k_ar - 1)])

        if "ci" in deterministic:
            self.const_coint = self.det_coef_coint[:1, :]
        else:
            self.const_coint = np.zeros(self.neqs)[:, None]
        if "li" in deterministic:
            self.lin_trend_coint = self.det_coef_coint[-1:, :]
        else:
            self.lin_trend_coint = np.zeros(self.neqs)[:, None]

        split_const_season = 1 if "co" in deterministic else 0
        split_season_lin = split_const_season + ((seasons-1) if seasons else 0)
        self.const, self.seasonal, self.lin_trend = \
            np.hsplit(self.det_coef, [split_const_season, split_season_lin])

        self.sigma_u = sigma_u

        if y_min1 is not None or delta_x is not None or delta_y_1_T:
            self.delta_y_1_T = delta_y_1_T
            self.y_min1 = y_min1
            self.delta_x = delta_x
        else:
            _y_1_T, self.delta_y_1_T, self.y_min1, self.delta_x = \
                _endog_matrices(endog_tot, level_var_lag_order, deterministic,
                                seasons)
        self.nobs = self.y_min1.shape[1]

    @cache_readonly
    def llf(self):  # Lutkepohl p. 295 (7.2.20)
        """Compute VECM(k_ar) loglikelihood
        """
        K = self.neqs
        T = self.nobs
        r = self.r
        s00, _, _, _, _, lambd, _ = _sij(self.delta_x, self.delta_y_1_T,
                                         self.y_min1)
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
    def num_det_coef_coint(self):  # todo: check if used at all?
        number_of_params = 0 + ("ci" in self.deterministic) \
                           + ("li" in self.deterministic)
        return number_of_params

    @cache_readonly
    def cov_params(self):  # p.296 (7.2.21)
        # Sigma_co described on p. 287
        beta = self.beta
        if self.det_coef_coint.size > 0:
            beta = vstack((beta, self.det_coef_coint))
        dt = self.deterministic
        num_det = ("co" in dt) + ("lo" in dt)
        num_det += (self.seasons-1) if self.seasons else 0
        b_id = scipy.linalg.block_diag(beta,
                                       np.identity(self.neqs * (self.k_ar-1) +
                                                   num_det))

        y_min1 = self.y_min1
        b_y = beta.T.dot(y_min1)
        omega11 = b_y.dot(b_y.T)
        omega12 = b_y.dot(self.delta_x.T)
        omega21 = omega12.T
        omega22 = self.delta_x.dot(self.delta_x.T)
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
        _, r1 = _r_matrices(self.nobs, self.delta_x, self.delta_y_1_T,
                            self.y_min1)
        r12 = r1[self.r:]
        mat1 = inv(r12.dot(r12.T))
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
        ret_1dim = self.stderr_coint[:self.beta.size]
        return ret_1dim.reshape(self.beta.shape, order="F")

    @cache_readonly
    def stderr_det_coef_coint(self):
        if self.det_coef_coint.size == 0:
            return self.det_coef_coint  # 0-size array
        ret_1dim = self.stderr_coint[self.beta.size:]
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
    def pvalues_alpha(self):  # todo: student-t
        return (1-scipy.stats.norm.cdf(abs(self.tvalues_alpha))) * 2

    @cache_readonly
    def pvalues_beta(self):
        first_rows = np.zeros((self.r, self.r))
        tval_last = self.tvalues_beta[self.r:]
        last_rows = (1-scipy.stats.norm.cdf(abs(tval_last))) * 2  # student-t
        return vstack((first_rows,
                       last_rows))

    @cache_readonly
    def pvalues_det_coef_coint(self):  # todo: student-t
        if self.det_coef_coint.size == 0:
            return self.det_coef_coint  # 0-size array
        return (1-scipy.stats.norm.cdf(abs(self.tvalues_det_coef_coint))) * 2

    @cache_readonly
    def pvalues_gamma(self):  # todo: student-t
        return (1-scipy.stats.norm.cdf(abs(self.tvalues_gamma))) * 2

    @cache_readonly
    def pvalues_det_coef(self):  # todo: student-t
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
    def var_repr(self):
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
        the corresponding VAR coefficient matrices (i.e. vec(self.var_repr)).

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
        return ma_rep(self.var_repr, maxn)

    @cache_readonly
    def _chol_sigma_u(self):
        return np.linalg.cholesky(self.sigma_u)

    def orth_ma_rep(self, maxn=10, P=None):
        r"""Compute orthogonalized MA coefficient matrices using P matrix such
        that :math:`\Sigma_u = PP^\prime`. P defaults to the Cholesky
        decomposition of :math:`\Sigma_u`

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

    def predict(self, steps=5, alpha=None):
        """

        Parameters
        ----------
        steps : int
            Prediction horizon.
        alpha : float between 0 and 1 or None
            If None, compute point forecast only.
            If float, compute confidence intervals too. In this case the
            argument stands for the confidence level.

        Returns
        -------
        forecast - ndarray (steps x neqs) or three ndarrays
            In case of a point forecast: each row of the returned ndarray
            represents the forecast of the neqs variables for a specific
            period. The first row (index [0]) is the forecast for the next
            period, the last row (index [steps-1]) is the steps-periods-ahead-
            forecast.
        """
        last_observations = self.y_all.T[-self.k_ar:]
        exog = []
        trend_coefs = []

        exog_const = np.ones(steps)
        nobs_tot = self.nobs + self.k_ar
        if self.const.size > 0:
            exog.append(exog_const)
            trend_coefs.append(self.const.T)
        if "ci" in self.deterministic:
            exog.append(exog_const)
            trend_coefs.append(self.alpha.dot(self.const_coint).T)
        if self.seasons > 0:
            first_future_season = (self.first_season + nobs_tot) % self.seasons
            exog_seasonal = seasonal_dummies(self.seasons, steps,
                                             first_future_season, True)
            exog.append(exog_seasonal)
            trend_coefs.append(self.seasonal.T)

        exog_lin_trend = list(range(nobs_tot + 1, nobs_tot + steps + 1))
        if self.lin_trend.size > 0:
            exog.append(exog_lin_trend)
            trend_coefs.append(self.lin_trend.T)
        if "li" in self.deterministic:
            exog.append(exog_lin_trend)
            trend_coefs.append(self.alpha.dot(self.lin_trend_coint).T)

        exog = np.column_stack(exog) if exog != [] else None
        if trend_coefs != []:
            # np.real: ignore imaginary +0j (e.g. if det. terms = cili+seasons)
            trend_coefs = np.real(np.row_stack(trend_coefs))
        else:
            trend_coefs = None

        if alpha is not None:
            return forecast_interval(last_observations, self.var_repr,
                                     trend_coefs, self.sigma_u, steps,
                                     alpha=alpha,
                                     exog=exog)
        else:
            return forecast(last_observations, self.var_repr, trend_coefs,
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

        """
        mid, lower, upper = self.predict(steps, alpha=alpha)

        y = self.y_all.T
        y = y[self.k_ar:] if n_last_obs is None else y[-n_last_obs:]
        plot.plot_var_forc(y, mid, lower, upper, names=self.names,
                           plot_stderr=plot_conf_int)

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
            causing = [self.names[c] for c in caused_ind]

        y, k, t, p = self.y_all, self.neqs, self.nobs - 1, self.k_ar + 1
        exog = deterministic_to_exog(self.deterministic, self.seasons,
                                     len_data=self.nobs + self.k_ar,
                                     first_season=self.first_season,
                                     seasons_centered=True)
        var_results = VAR(y.T, exog).fit(maxlags=p, trend="nc")

        # num_restr is called N in Lutkepohl
        num_restr = len(causing) * len(caused) * (p - 1)
        num_det_terms = num_det_vars(self.deterministic, self.seasons)

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
                                     len_data=self.nobs + self.k_ar,
                                     first_season=self.first_season,
                                     seasons_centered=True)

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
        delta_y = np.dot(pi, self.y_min1) + np.dot(gamma, self.delta_x)
        return (delta_y + self.y_min1[:self.neqs]).T

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
        results : dict

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
                    start = (self.seasons-1) * i
                    masks.append(offset + np.arange(start,
                                                    start + self.seasons-1))
                    offset += (self.seasons-1) * self.neqs
                if "lo" in self.deterministic:
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
                title = "Det. terms outside coint. relation " + \
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
                # offset += self.r

            # Create the table
            mask = np.concatenate(masks)
            title = "Cointegration relations for " + \
                    "loading-coefficients-column %d" % (i+1)
            table = make_table(self, coint, stderr_coint, tvalues_coint,
                               pvalues_coint, conf_int_coint, mask,
                               coint_names, title)
            summary.tables.append(table)

        return summary
