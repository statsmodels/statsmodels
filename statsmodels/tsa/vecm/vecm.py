from math import log, pi
import numpy as np
from numpy import hstack, vstack
from numpy.linalg import inv, svd
import scipy
import scipy.stats
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import chain_dot
from statsmodels.tsa.tsatools import duplication_matrix, vec, vech, unvec

import statsmodels.tsa.base.tsa_model as tsbase

def mat_sqrt(_2darray):
    u_, s_, v_ = svd(_2darray, full_matrices=False)
    s_ = np.sqrt(s_)
    return chain_dot(u_, np.diag(s_), v_)

def _endog_matrices(endog_tot, diff_lags, deterministic):
    # p. 286:
    p = diff_lags+1
    y = endog_tot
    K = y.shape[0]
    y_1_T = y[:, p:]
    T = y_1_T.shape[1]
    delta_y = np.diff(y)
    delta_y_1_T = delta_y[:, p-1:]
    y_min1 = y[:, p-1:-1]
    if "lt" in deterministic:
        y_min1 = vstack((y_min1,
                         np.arange(T)))  # p. 299
    # p. 286:
    delta_x = np.zeros((diff_lags*K, T))
    for j in range(delta_x.shape[1]):
        delta_x[:, j] = (delta_y[:, j+p-2:None if j-1<0 else j-1:-1]
                         .T.reshape(K*(p-1)))
    # p. 299, p. 303:
    if "c" in deterministic:
        delta_x = vstack((delta_x,
                          np.ones(T)))
    if "s" in deterministic:  # TODO: How many seasons??
        seasons = np.zeros((3, delta_x.shape[1]))  # 3 = 4-1 (4=#seasons)
        for i in range(seasons.shape[0]):
            seasons[i, i::4] = 1
        #seasons = seasons[:, ::-1]
        #seasons = np.hstack((seasons[:, 3:4], seasons[:, :-1]))
        #seasons = np.hstack((seasons[:, 2:4], seasons[:, :-2]))
        seasons = np.hstack((seasons[:, 1:4], seasons[:, :-3]))
        # seasons[1] = -seasons[1]
        delta_x = vstack((delta_x,
                          seasons))
    return y_1_T, delta_y, delta_y_1_T, y_min1, delta_x


def _block_matrix_ymin1_deltax(y_min1, delta_x):  # e.g. p.287 (7.2.4)
    b = y_min1.dot(delta_x.T)
    return inv(vstack((hstack((y_min1.dot(y_min1.T), b)),
                       hstack((b.T, delta_x.dot(delta_x.T))))))


def _r_matrices(T, delta_x, delta_y_1_T, y_min1):
    m = np.identity(T) - (
        delta_x.T.dot(inv(delta_x.dot(delta_x.T))).dot(delta_x))  # p. 291
    r0 = delta_y_1_T.dot(m)  # p. 292
    r1 = y_min1.dot(m)
    return r0, r1


def _sij(delta_x, delta_y_1_T, y_min1):
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
    r"""
    Fit VECM process and do lag order selection

    .. math:: y_t = A_1 y_{t-1} + \ldots + A_p y_{t-p} + u_t

    Parameters
    ----------
    endog : array-like
        2-d endogenous response variable. The independent variable.
    dates : array-like
        must match number of rows of endog

    References
    ----------
    Lutkepohl (2005) New Introduction to Multiple Time Series Analysis
    """  # TODO: docstring + implementation
    
    # TODO: implementation
    def __init__(self, endog_tot, dates=None, freq=None, missing="none"):
        super(VECM, self).__init__(endog_tot, None, dates, freq, missing=missing)
        if self.endog.ndim == 1:
            raise ValueError("Only gave one variable to VECM")
        self.y = self.endog.T  # TODO delete this line if y not necessary
        self.neqs = self.endog.shape[1]

    def fit(self, max_diff_lags=None, method="ml", ic=None,
            deterministic="", verbose=False, coint_rank=None):
        """
        Fit the VECM

        Parameters
        ----------
        max_diff_lags : int
            Maximum number of lags to check for order selection, defaults to
            12 * (nobs/100.)**(1./4), see select_order function ##### ##### ##### #####
        method : {"ls", "egls", "ml"}
            Estimation method to use.
        ic : {"aic", "fpe", "hqic", "bic", None}
            Information criterion to use for VECM order selection.
            aic : Akaike
            fpe : Final prediction error
            hqic : Hannan-Quinn
            bic : Bayesian a.k.a. Schwarz
        deterministic, str {"", "c", "lt", "s"}
            "" - no deterministic terms
            "c" - constant
            "lt" - linear trend
            "s" - seasonal terms
            Combinations of these are possible (e.g. "clt" for linear trend 
            with intercept)

        Notes
        -----
        Lutkepohl pp. 146-153

        Returns
        -------
        est : VARResults
        """  # TODO: docstring + implementation
        
        # TODO: trend
        
        # select number of lags (=p-1)
        if ic is None:
            if max_diff_lags is None:
                diff_lags = 1
            else:
                diff_lags = max_diff_lags
        else:
            selections = self.select_order(max_diff_lags=max_diff_lags,
                                           verbose=verbose)
            if ic not in selections:
                raise ValueError("%s not recognized, must be among %s"
                                 % (ic, sorted(selections)))
            diff_lags = selections[ic]
            if verbose:
                print("Using %d based on %s criterion" %  (diff_lags, ic))
        self.p = diff_lags + 1
        # estimate parameters
        if method == "ls":
            return self._estimate_vecm_ls(diff_lags, deterministic)
        elif method == "egls":
            if coint_rank is None:
                coint_rank = 1
            return self._estimate_vecm_egls(diff_lags, deterministic,
                                            coint_rank)
        elif method == "ml":
            if coint_rank is None:
                coint_rank = 1
            return self._estimate_vecm_ml(diff_lags, deterministic,
                                          coint_rank)
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

        pi_cols = K if "lt" not in deterministic else K + 1
        pi_hat = est_pi_gamma[:, :pi_cols]

        gamma_hat = est_pi_gamma[:, pi_cols:]

        _A = delta_y_1_T - pi_hat.dot(y_min1) - gamma_hat.dot(delta_x)
        p = diff_lags+1
        sigma_u_hat = 1/(T-K*p) * np.dot(_A, _A.T)  # p. 287 (equation (7.2.5))

        return pi_hat, gamma_hat, sigma_u_hat

    def _estimate_vecm_ls(self, diff_lags, deterministic=""):
        # deterministic \in \{"c", "lt", "s"\}, where
        # c=constant, lt=linear trend, s=seasonal terms
        y_1_T, delta_y, delta_y_1_T, y_min1, delta_x = _endog_matrices(
                self.y, diff_lags, deterministic)
        pi_hat, gamma_hat, sigma_u_hat = self._ls_pi_gamma(delta_y_1_T, y_min1,
                                                           delta_x, diff_lags,
                                                           deterministic)
        return {"Pi_hat": pi_hat, "Gamma_hat": gamma_hat,
                "Sigma_u_hat": sigma_u_hat}
    
    def _estimate_vecm_egls(self, diff_lags, deterministic="", r=1):
        y_1_T, _dy, delta_y_1_T, y_min1, delta_x = _endog_matrices(
                self.y, diff_lags, deterministic)
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
        # (delta_y_1_T - alpha_hat.dot(beta_hhat.T).dot(y_min1)).dot(delta_x.dot(inv(np.dot(delta_x,delta_x.T))))
        
        # Gamma_hhat = 
        # TODO: Gamma?
        
        return {"alpha": alpha_hat, "beta": beta_hhat, 
                "Gamma": _gamma_hat, "Sigma_u": sigma_u_hat}
    
    def _estimate_vecm_ml(self, diff_lags, deterministic="", r=1):
        y_1_T, _dy, delta_y_1_T, y_min1, delta_x = _endog_matrices(
                self.y, diff_lags, deterministic)
        T = y_1_T.shape[1]

        s00, s01, s10, s11, s11_, _, v = _sij(delta_x, delta_y_1_T, y_min1)

        # print("s00 shape: " + str(s00.shape))
        # print("s01 shape: " + str(s01.shape))
        # print("s10 shape: " + str(s10.shape))
        # print("s11 shape: " + str(s11.shape))
        # print("s11_ shape: " + str(s11_.shape))

        beta_tilde = (v[:, :r].T.dot(s11_)).T
        # normalize beta tilde such that eye(r) forms the first r rows of it:
        beta_tilde = np.dot(beta_tilde, inv(beta_tilde[:r, :r]))
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
                           deterministic=deterministic, delta_y_1_T=delta_y_1_T,
                           y_min1=y_min1, delta_x=delta_x)
        # return {"alpha": np.array(alpha_tilde),
                # "beta": np.array(beta_tilde),
                # "Gamma": np.array(gamma_tilde),
                # "Sigma_u": np.array(sigma_u_tilde)}

    def select_order(self, max_diff_lags=None, verbose=True):
        """
        Compute lag order selections based on each of the available information
        criteria

        Parameters
        ----------
        max_diff_lags : int
            if None, defaults to 12 * (nobs/100.)**(1./4)
        verbose : bool, default True
            If True, print table of info criteria and selected orders

        Returns
        -------
        selections : dict {info_crit -> selected_order}
        """  # TODO: docstring + implementation. Function used in fit().
        pass

    def predict(self, params, start=None, end=None, lags=1, trend="c"):
        """
        Returns in-sample predictions or forecasts
        """  # TODO: docstring + implementation
        pass



# ------------------------------------------------------------------------------
# VECMResults class

class VECMResults(object):
    """Estimate VECM(p) process with fixed number of lags

    Parameters
    ----------
    endog_tot : array
    alpha : array
    beta : array
    gamma : array
    sigma_u : array
    lag_order : int
    coint_rank : int
    model : VECM model instance
    deterministic : str {'c', 'lt', 's'}
    names : array-like
        List of names of the endogenous variables in order of appearance in `endog`.
    dates


    Returns
    -------
    **Attributes**
    aic
    bic
    bse
    coefs : ndarray (p x K x K)
        Estimated A_i matrices, A_i = coefs[i-1]
    llf
    model
    neqs : int
        Number of variables (equations)
    nobs : int
    n_totobs : int
    order : int
        Order of VAR process
    params : ndarray (Kp + 1) x K
        A_i matrices and intercept in stacked form [int A_1 ... A_p]
    pvalues
    sigma_u : ndarray (K x K)
        Estimate of white noise process variance Var[u_t]
    sigma_u_mle
    stderr
    tvalues

    df_model
    df_resid
    fittedvalues
    resid
    sigma_u_mle --- need to adapt the ML estimator?
    """

    def __init__(self, endog_tot, level_var_lag_order, coint_rank, alpha, beta,
                 gamma, sigma_u,
                 model=None, deterministic='c', names=None, dates=None,
                 delta_y_1_T=None, y_min1=None, delta_x=None):
        self.y_all = endog_tot
        self.K = endog_tot.shape[0]
        self.p = level_var_lag_order
        self.deterministic = deterministic
        self.r = coint_rank
        self.alpha = alpha
        self.beta = beta[:self.K]
        self.det_coef_coint = beta[self.K:]
        self.gamma, self.det_coef = np.hsplit(gamma, [self.K*(self.p-1)])
        # = gamma[:, self.gamma.shape[1]:].reshape(gamma.shape[0], -1)
        self.sigma_u = sigma_u
        if y_min1 is not None or delta_x is not None or delta_y_1_T:
            self.delta_y_1_T = delta_y_1_T
            self.y_min1 = y_min1
            self.delta_x = delta_x
        else:
             _y_1_T, _delta_y, self.delta_y_1_T, self.y_min1, self.delta_x = \
                 _endog_matrices(endog_tot, level_var_lag_order, deterministic)
        self.T = self.y_min1.shape[1]
        # TODO: llf, se, t, p

    @cache_readonly
    def llf(self):  # Lutkepohl p. 295 (7.2.20)
        """Compute VECM(p) loglikelihood
        """
        K = self.K
        T = self.T
        r = self.r
        s00, _, _, _, _, lambd, _ = _sij(self.delta_x, self.delta_y_1_T,
                                              self.y_min1)
        return - K * T * log(2*pi) / 2  \
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
    def number_coint_params(self):  # tedo: check if used at all?
        number_of_params = 0 + ("lt" in self.deterministic)
        return number_of_params



    @cache_readonly
    def cov_params(self):  # p.296 (7.2.21)
        # beta = vstack((self.beta,
        #                self.det_coef_coint))
        #
        # b_id = scipy.linalg.block_diag(beta, np.identity(self.K*(self.p-1)))
        #
        # b_y = beta.T.dot(self.y_min1)
        # omega11 = b_y.dot(b_y.T)
        # omega12 = b_y.dot(self.delta_x.T)
        # omega21 = omega12.T
        # omega22 = self.delta_x.dot(self.delta_x.T)
        # omega = np.bmat([[omega11, omega12], [omega21, omega22]])
        #
        # print("\n\n\ndet terms")
        # print(self.det_coef.shape)
        # print("det terms in coint")
        # print(self.det_coef_coint.shape)
        # print(b_id.shape)
        # print(omega.shape)
        # print(b_id.T.shape)
        # mat1 = b_id.dot(inv(omega)
        #             ).dot(b_id.T)
        # return np.kron(mat1, self.sigma_u)

        # with first arg to np.kron estimated as in Lutkepohl
        return np.kron(_block_matrix_ymin1_deltax(self.y_min1, self.delta_x),
                      self.sigma_u)  # p.287

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
        last_rows = last_rows_1d.reshape((self.K-self.r+det, self.r), order="F")
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

    
    @cache_readonly
    def pvalues_alpha(self):
        return (1-scipy.stats.norm.cdf(abs(self.tvalues_alpha))) * 2  # student-t

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
        return (1-scipy.stats.norm.cdf(abs(self.tvalues_det_coef_coint))) * 2  # student-t

    @cache_readonly
    def pvalues_gamma(self):
        return (1-scipy.stats.norm.cdf(abs(self.tvalues_gamma))) * 2  # student-t

    @cache_readonly
    def pvalues_det_coef(self):
        if self.det_coef.size == 0:
            return self.det_coef  # 0-size array
        return (1-scipy.stats.norm.cdf(abs(self.tvalues_det_coef))) * 2  # student-t



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