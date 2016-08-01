from __future__ import division
import math
from math import log
import numpy as np
from numpy import hstack, vstack
from numpy.linalg import inv, svd
import scipy
import scipy.stats
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import chain_dot
from statsmodels.tsa.tsatools import duplication_matrix

import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.vector_ar.var_model import forecast, forecast_interval


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


def _endog_matrices(endog_tot, diff_lags, deterministic, seasons=0):
    """Returns different matrices needed for parameter estimation. These
    consist of elements of the data as well as elements representing
    deterministic terms. A tuple of consisting of these matrices is returned.

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
    seasons : int
        Number of seasons. 0 (default) means no seasons.

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
    [Lut2005]_, p. 286
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
    # if "co" in deterministic and "ci" in deterministic:
    #     raise ValueError("Both 'co' and 'ci' as deterministic terms given. "
    #                      + "Please choose one of the two.")
    # todo: optimize the following with np.row_stack()

    if "ci" in deterministic:  # pp. 257, 299, 306, 307
        # y_min1_mean = y_min1.mean(1)
        y_min1 = vstack((y_min1,
                         np.ones(T)))
        # H = vstack((np.identity(K),
        #             - y_min1_mean))
        # y_min1 = H.T.dot(y_min1)

    if "li" in deterministic:  # p. 299
        y_min1 = vstack((y_min1,
                         np.arange(T)))

    # p. 286:
    delta_x = np.zeros((diff_lags*K, T))
    for j in range(delta_x.shape[1]):
        delta_x[:, j] = (delta_y[:, j+p-2:None if j-1 < 0 else j-1:-1]
                         .T.reshape(K*(p-1)))
    # p. 299, p. 303:
    if "co" in deterministic:
        delta_x = vstack((delta_x,
                          np.ones(T)))
    if seasons > 0:
        season_dummy = np.zeros((seasons - 1, delta_x.shape[1]))
        for i in range(seasons - 1):
            season_dummy[i, i::seasons] = 1
        # season_dummy = season_dummy[:, ::-1]
        # season_dummy = np.hstack((season_dummy[:, 3:4],
        #   season_dummy[:, :-1]))
        # season_dummy = np.hstack((season_dummy[:, 2:4],
        #   season_dummy[:, :-2]))
        season_dummy = np.hstack((season_dummy[:, 1:4], season_dummy[:, :-3]))
        # season_dummy[1] = -season_dummy[1]
        season_dummy -= 1 / seasons
        delta_x = vstack((delta_x,
                          season_dummy))

    if "lo" in deterministic:
        delta_x = vstack((delta_x,
                          np.arange(T)+1))

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
    result : ndarray (K*p x K*p)
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
    # p. 294: optimizable: e.g. r0.dot(r1.T) == r1.dot(r0.T).T ==> s01==s10.T
    s00, s01, s10, s11 = (Ri.dot(Rj.T)/T for Ri in (r0, r1) for Rj in (r0, r1))
    s11_ = inv(mat_sqrt(s11))
    # p. 295:
    eig = np.linalg.eig(chain_dot(s11_, s10, inv(s00), s01, s11_))
    lambd = eig[0]
    v = eig[1]
    return s00, s01, s10, s11, s11_, lambd, v


# VECM class: for known or unknown VECM

class VECM(tsbase.TimeSeriesModel):
    """
    Fit a VECM process
    .. math:: \Delta y_t = \Pi y_{t-1} + \Gamma_1 \Delta y_{t-1} + \ldots + \Gamma_{p-1} \Delta y_{t-p+1} + u_t
    where
    .. math:: \Pi = \alpha \beta'.

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
    [Lut2005]_
    .. [Lut2005] Lutkepohl, H. 2005. *New Introduction to Multiple Time Series Analysis*. Springer.
    """

    def __init__(self, endog_tot, dates=None, freq=None, missing="none",
                 diff_lags=1, deterministic="nc", seasons=0):
        super(VECM, self).__init__(endog_tot, None, dates, freq,
                                   missing=missing)
        if self.endog.ndim == 1:
            raise ValueError("Only gave one variable to VECM")
        self.y = self.endog.T  # TODO delete this line if y not necessary
        self.neqs = self.endog.shape[1]
        self.p = diff_lags + 1
        self.diff_lags = diff_lags
        self.deterministic = deterministic
        self.seasons = seasons

    def fit(self, method="ml", coint_rank=1):
        """
        Estimates the parameters of a VECM and returns a VECMResults object.

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

        Notes
        -----
        [Lut2005]_ pp. 269-304
        """
        if method == "ls":
            return self._estimate_vecm_ls(self.diff_lags, self.deterministic,
                                          self.seasons)
        elif method == "egls":
            return self._estimate_vecm_egls(self.diff_lags, self.deterministic,
                                            self.seasons, coint_rank)
        elif method == "ml":
            return self._estimate_vecm_ml(self.diff_lags, self.deterministic,
                                          self.seasons, coint_rank)
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

    def _estimate_vecm_ls(self, diff_lags, deterministic="nc", seasons=0):
        # deterministic \in \{"c", "lo", \}, where
        # c=constant, lt=linear trend, s=seasonal terms
        y_1_T, delta_y_1_T, y_min1, delta_x = _endog_matrices(
                self.y, diff_lags, deterministic, seasons)

        pi_hat, gamma_hat, sigma_u_hat = self._ls_pi_gamma(delta_y_1_T, y_min1,
                                                           delta_x, diff_lags,
                                                           deterministic)
        return {"Pi_hat": pi_hat, "Gamma_hat": gamma_hat,
                "Sigma_u_hat": sigma_u_hat}
    
    def _estimate_vecm_egls(self, diff_lags, deterministic="nc", seasons=0,
                            r=1):
        y_1_T, delta_y_1_T, y_min1, delta_x = _endog_matrices(
                self.y, diff_lags, deterministic, seasons)
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
    
    def _estimate_vecm_ml(self, diff_lags, deterministic="nc", seasons=0, r=1):
        y_1_T, delta_y_1_T, y_min1, delta_x = _endog_matrices(
                self.y, diff_lags, deterministic, seasons)
        T = y_1_T.shape[1]

        s00, s01, s10, s11, s11_, _, v = _sij(delta_x, delta_y_1_T, y_min1)

        # print("s00 shape: " + str(s00.shape))
        # print("s01 shape: " + str(s01.shape))
        # print("s10 shape: " + str(s10.shape))
        # print("s11 shape: " + str(s11.shape))
        # print("s11_ shape: " + str(s11_.shape))

        beta_tilde = (v[:, :r].T.dot(s11_)).T
        # normalize beta tilde such that eye(r) forms the first r rows of it:
        beta_tilde = np.dot(beta_tilde, inv(beta_tilde[:r]))
        alpha_tilde = s01.dot(beta_tilde).dot(
                inv(beta_tilde.T.dot(s11).dot(beta_tilde)))
        # print("alpha shape: " + str(alpha_tilde.shape))
        # print("beta shape: " + str(beta_tilde.shape))
        # print("y_min1 shape: " + str(y_min1.shape))
        gamma_tilde = (delta_y_1_T - alpha_tilde.dot(beta_tilde.T).dot(y_min1)
                       ).dot(delta_x.T).dot(inv(np.dot(delta_x, delta_x.T)))
        temp = (delta_y_1_T - alpha_tilde.dot(beta_tilde.T).dot(y_min1) -
                gamma_tilde.dot(delta_x))
        sigma_u_tilde = temp.dot(temp.T) / T

        return VECMResults(self.y, self.p, r, alpha_tilde, beta_tilde,
                           gamma_tilde, sigma_u_tilde,
                           deterministic=deterministic, seasons=seasons,
                           delta_y_1_T=delta_y_1_T, y_min1=y_min1,
                           delta_x=delta_x)


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

    alpha : array (K x coint_rank)
        ... where K is the number of variables per observation
    beta : array (K x coint_rank)
        ... where K is the number of variables per observation
    gamma : array (K x K*(level_var_lag_order-1))
        ... where K is the number of variables per observation
    sigma_u : array (K x K)
        ... where K is the number of variables per observation
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
    K : int
        Number of variables per observation. Number of equations.
    p : int
        Lags in the VAR representation. This implies: Lags in the VEC
        representation = p - 1
    r : int
        Cointegration rank.
    T : int
        Number of observations after the presample

    y_min1 : ndarray (K x T)
        Observations at t=0 until t=T-1
    delta_y_1_T : ndarray (K x T)
        Observations at t=1 until t=T minus y_min1
    delta_x : ndarray ((K * (p-1) + number of deterministic dummy variables
        outside the cointegration relation) x T)

    llf

    _covar_sigma_u
    num_det_coef_coint : int
        Number of estimated coefficients for deterministic terms within the
        cointegration relation

    cov_params : ndarray (d x d)
        ... where d equals K * (K+num_det_coef_coint + K*(p-1)+number of
        deterministic dummy variables outside the cointegration relation)
    stderr_params : ndarray (d)
        ... where d is defined as for cov_params
    stderr_coint : ndarray (K+num_det_coef_coint x r)
    stderr_alpha ndarray (K x r)
    stderr_beta : ndarray (K x r)
    stderr_det_coef_coint ndarray (num_det_coef_coint x r)
    stderr_gamma : ndarray (K x K*(p-1))
    stderr_det_coef : ndarray (K x number of deterministic dummy variables
        outside the cointegration relation)
    tvalues_alpha : ndarray (K x r)
    tvalues_beta : ndarray (K x r)
    tvalues_det_coef_coint
    tvalues_gamma : ndarray (K x K*(p-1))
    tvalues_det_coef : ndarray (K x number of deterministic dummy variables
        outside the cointegration relation)
    pvalues_alpha : ndarray (K x r)
    pvalues_beta : ndarray (K x r)
    pvalues_det_coef_coint
    pvalues_gamma : ndarray (K x K*(p-1))
    pvalues_det_coef : ndarray (K x number of deterministic dummy variables
        outside the cointegration relation)
    var_repr : (p x K x K)
        KxK matrices A_i of the corresponding VAR representation. If the return
        value is assigned to a variable A, these matrices can be accessed via
        A[i], i=0, ..., p-1.
    """    # todo: aic, bic, bse, df_model, df_resid, fittedvalues, resid

    def __init__(self, endog_tot, level_var_lag_order, coint_rank, alpha, beta,
                 gamma, sigma_u, deterministic='cc', seasons=0,
                 delta_y_1_T=None, y_min1=None, delta_x=None):
        self.y_all = endog_tot
        self.K = endog_tot.shape[0]
        self.p = level_var_lag_order
        self.deterministic = deterministic
        self.seasons = seasons

        self.r = coint_rank
        self.alpha = alpha
        self.beta, self.det_coef_coint = np.vsplit(beta, [self.K])
        self.gamma, self.det_coef = np.hsplit(gamma, [self.K*(self.p-1)])

        if "ci" in deterministic:
            self.const_coint = self.det_coef_coint[:1, :]
        else:
            self.const_coint = np.zeros(self.K)[:, None]
        if "li" in deterministic:
            self.lin_trend_coint = self.det_coef_coint[-1:, :]
        else:
            self.lin_trend_coint = np.zeros(self.K)[:, None]

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
        self.T = self.y_min1.shape[1]

    @cache_readonly
    def llf(self):  # Lutkepohl p. 295 (7.2.20)
        """Compute VECM(p) loglikelihood
        """
        K = self.K
        T = self.T
        r = self.r
        s00, _, _, _, _, lambd, _ = _sij(self.delta_x, self.delta_y_1_T,
                                         self.y_min1)
        return - K * T * log(2*math.pi) / 2  \
            - T * (log(np.linalg.det(s00)) + sum(np.log(1-lambd)[:r])) / 2  \
            - K * T / 2

    @cache_readonly
    def _covar_sigma_u(self):
        sigma_u = self.sigma_u
        K = sigma_u.shape[0]
        d = duplication_matrix(K)
        d_K_plus = inv(np.dot(d.T, d)).dot(d.T)
        return 2 * chain_dot(d_K_plus, np.kron(sigma_u, sigma_u), d_K_plus.T)

    @cache_readonly
    def num_det_coef_coint(self):  # todo: check if used at all?
        number_of_params = 0 + ("ci" in self.deterministic) \
                           + ("li" in self.deterministic)
        return number_of_params

    @cache_readonly
    def cov_params(self):  # p.296 (7.2.21)
        # beta = self.beta
        beta = self.beta
        if self.det_coef_coint.size > 0:
            beta = vstack((beta, self.det_coef_coint))
        dt = self.deterministic
        num_det = ("co" in dt) + ("lo" in dt)
        num_det += (self.seasons-1) if self.seasons else 0
        b_id = scipy.linalg.block_diag(beta,
                                       np.identity(self.K * (self.p-1) +
                                                   num_det))

        y_min1 = self.y_min1
        # if self.num_det_coef_coint > 0:
        #     y_min1 = y_min1[:-self.num_det_coef_coint]
        b_y = beta.T.dot(y_min1)
        omega11 = b_y.dot(b_y.T)
        omega12 = b_y.dot(self.delta_x.T)
        omega21 = omega12.T
        omega22 = self.delta_x.dot(self.delta_x.T)
        omega = np.bmat([[omega11, omega12],
                         [omega21, omega22]]).A

        mat1 = b_id.dot(inv(omega)).dot(b_id.T)
        return np.kron(mat1, self.sigma_u)

    # standard errors:
    @cache_readonly
    def stderr_params(self):
        return np.sqrt(np.diag(self.cov_params))

    @cache_readonly
    def stderr_coint(self):
        _, r1 = _r_matrices(self.T, self.delta_x, self.delta_y_1_T,
                            self.y_min1)
        r12 = r1[self.r:]
        mat1 = inv(r12.dot(r12.T))
        det = self.det_coef_coint.shape[0]
        mat2 = np.kron(np.identity(self.K-self.r+det),
                       inv(chain_dot(
                               self.alpha.T, inv(self.sigma_u), self.alpha)))
        first_rows = np.zeros((self.r, self.r))
        last_rows_1d = np.sqrt(np.diag(mat1.dot(mat2)))
        last_rows = last_rows_1d.reshape((self.K-self.r+det, self.r),
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

    @cache_readonly
    def var_repr(self):
        pi = self.alpha.dot(self.beta.T)
        gamma = self.gamma
        K = self.K
        A = np.zeros((self.p, K, K))
        A[0] = pi + np.identity(K) + gamma[:, :K]
        A[self.p-1] = - gamma[:, K*(self.p-2):]
        for i in range(1, self.p-1):
            A[i] = gamma[:, K*i:K*(i+1)] - gamma[:, K*(i-1):K*i]
        return A

    def predict(self, steps=5, confidence_level_for_intervals=None):
        """

        Parameters
        ----------
        steps : int
            Prediction horizon.
        confidence_level_for_intervals : float between 0 and 1 or None
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
        last_observations = self.y_all.T[-self.p:]
        exog = []
        trend_coefs = []

        exog_const = np.ones(steps)
        if self.const.size > 0:
            exog.append(exog_const)
            trend_coefs.append(self.const.T)
        if "ci" in self.deterministic:
            exog.append(exog_const)
            trend_coefs.append(self.alpha.dot(self.const_coint).T)
        if self.seasons > 0:
            exog_seasonal = np.pad(np.identity(self.seasons-1),   # identity &
                                   ((0, 1), (0, 0)), "constant")  # 0-row below
            exog_seasonal -= 1 / self.seasons
            if steps < self.seasons:
                exog_seasonal = exog_seasonal[:steps]
            if steps > self.seasons:  # repeat matrix pattern until #rows=steps
                exog_seasonal = np.pad(exog_seasonal, ((0, steps-self.seasons),
                                                       (0, 0)), "wrap")
            exog.append(exog_seasonal)
            trend_coefs.append(self.seasonal.T)

        nobs_tot = self.T + self.p
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

        if confidence_level_for_intervals is not None:
            return forecast_interval(last_observations, self.var_repr,
                                     trend_coefs, self.sigma_u, steps,
                                     alpha=confidence_level_for_intervals,
                                     exog=exog)
        else:
            return forecast(last_observations, self.var_repr, trend_coefs,
                            steps, exog)
