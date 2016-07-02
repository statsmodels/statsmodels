import numpy as np
from numpy import hstack, vstack
from numpy.linalg import inv, svd
import scipy

import statsmodels.tsa.base.tsa_model as tsbase


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
    def __init__(self, endog, dates=None, freq=None, missing="none"):
        super(VECM, self).__init__(endog, None, dates, freq, missing=missing)
        if self.endog.ndim == 1:
            raise ValueError("Only gave one variable to VECM")
        self.y = self.endog  # TODO delete this line if y not necessary
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

    def _est_matrices(self, diff_lags, deterministic):
        # p. 286:
        p = diff_lags+1
        y = self.endog.T  # superclass turning DataFrame into ndarray?
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
            delta_x = vstack((delta_x,
                              seasons))
        return y, y_1_T, delta_y, delta_y_1_T, y_min1, delta_x

    def _ls_pi_gamma(self, delta_y_1_T, y_min1, delta_x, diff_lags,
                     deterministic):
        K = delta_y_1_T.shape[0]
        T = delta_y_1_T.shape[1]

        mat1 = hstack((delta_y_1_T.dot(y_min1.T), delta_y_1_T.dot(delta_x.T)))
        b = y_min1.dot(delta_x.T)
        mat2 = inv(vstack((hstack((y_min1.dot(y_min1.T), b)),
                           hstack((b.T, delta_x.dot(delta_x.T))))))
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
        y, y_1_T, delta_y, delta_y_1_T, y_min1, delta_x = self._est_matrices(
                diff_lags,deterministic)
        pi_hat, gamma_hat, sigma_u_hat = self._ls_pi_gamma(delta_y_1_T, y_min1,
                                                           delta_x, diff_lags,
                                                           deterministic)
        return {"Pi_hat": pi_hat, "Gamma_hat": gamma_hat,
                "Sigma_u_hat": sigma_u_hat}

    def _m_and_r_matrices(self, T, delta_x, delta_y_1_T, y_min1):
        m = np.identity(T) - (
            delta_x.T.dot(inv(delta_x.dot(delta_x.T))).dot(delta_x))  # p. 291
        r0 = delta_y_1_T.dot(m)  # p. 292
        r1 = y_min1.dot(m)
        return m, r0, r1
    
    def _estimate_vecm_egls(self, diff_lags, deterministic="", r=1):
        _y, y_1_T, _dy, delta_y_1_T, y_min1, delta_x = self._est_matrices(
                diff_lags, deterministic)
        T = y_1_T.shape[1]
        
        pi_hat, _gamma_hat, sigma_u_hat = self._ls_pi_gamma(delta_y_1_T,
                                                            y_min1, delta_x,
                                                            diff_lags,
                                                            deterministic)
        alpha_hat = pi_hat[:, :r]

        m, r0, r1 = self._m_and_r_matrices(T, delta_x, delta_y_1_T, y_min1)
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
    
    def _estimate_vecm_ml(self, diff_lags, deterministic_terms="", r=1):
        y, y_1_T, _dy, delta_y_1_T, y_min1, delta_x = self._est_matrices(
                diff_lags, deterministic_terms)
        K = y.shape[0]
        T = y_1_T.shape[1]
        
        m, r0, r1 = self._m_and_r_matrices(T, delta_x, delta_y_1_T, y_min1)
        # p. 294:
        s = np.bmat([[Ri.dot(Rj.T)/T for Rj in [r0, r1]]
                     for Ri in [r0, r1]])
        s = s.A
        s00 = s[:K, :K]
        s01 = s[:K, K:]
        s10 = s[K:, :K]
        s11 = s[K:, K:]
        u_, s_, v_ = svd(inv(s11), full_matrices=False)
        s_ = np.sqrt(s_)
        s11_ = u_.dot(np.diag(s_)).dot(v_)
        # p. 295:
        v = np.linalg.eig(s11_.dot(s10).dot(inv(s00)).dot(s01).dot(s11_))[1]
        beta_tilde = (v[:, :r].T.dot(s11_)).T
        # normalize beta tilde such that eye(r) forms the first r rows of it:
        beta_tilde = np.dot(beta_tilde, inv(beta_tilde[:r, :r]))
        alpha_tilde = s01.dot(beta_tilde).dot(
                inv(beta_tilde.T.dot(s11).dot(beta_tilde)))

        gamma_tilde = (delta_y_1_T - alpha_tilde*beta_tilde.T.dot(y_min1)).dot(
            delta_x.T).dot(inv(np.dot(delta_x, delta_x.T)))
        temp = (delta_y_1_T - alpha_tilde.dot(beta_tilde.T).dot(y_min1) -
                gamma_tilde.dot(delta_x))
        sigma_u_tilde = temp.dot(temp.T) / T
        
        return {"alpha": np.array(alpha_tilde),
                "beta": np.array(beta_tilde),
                "Gamma": np.array(gamma_tilde),
                "Sigma_u": np.array(sigma_u_tilde)}

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

    def to_var(self, max_diff_lags, method, deterministic):
        pass
        vecm_params = self.fit(max_diff_lags=max_diff_lags, method=method,
                               deterministic=deterministic)
        pi = vecm_params["alpha"].dot(vecm_params["beta"].T)
        gamma = vecm_params["Gamma"]
        if "lt" in deterministic:
            pi = pi[:, :-1]
        if "c" in deterministic:
            gamma = gamma[:, :-1]
        if "s" in deterministic:
            gamma = gamma[:, :-3]  # TODO: allow for #seasons != 3
        K = gamma.shape[0]
        p = 1 + round(gamma.shape[1]/K)
        A = np.zeros((p, K, K))
        A[0] = pi + np.identity(K) + gamma[:, :K]
        A[p-1] = - gamma[:, K*(p-2):]
        for i in range(1, p-1):
            A[i] = gamma[:, K*i:K*(i+1)] - gamma[:, K*(i-1):K*i]
        return np.concatenate(A, 1)
