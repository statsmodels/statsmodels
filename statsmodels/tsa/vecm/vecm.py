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
    """#TODO: docstring + implementation
    
    # TODO: implementation
    def __init__(self, endog, dates=None, freq=None, missing='none'):
        super(VECM, self).__init__(endog, None, dates, freq, missing=missing)
        if self.endog.ndim == 1:
            raise ValueError("Only gave one variable to VECM")
        self.y = self.endog #TODO delete this line if y not necessary
        self.neqs = self.endog.shape[1]

    def predict(self, params, start=None, end=None, lags=1, trend='c'):
        """
        Returns in-sample predictions or forecasts
        """#TODO: docstring + implementation
        pass

    def fit(self, maxlags=None, method='ols', ic=None, trend='c',
            verbose=False):
        """
        Fit the VECM

        Parameters
        ----------
        maxlags : int
            Maximum number of lags to check for order selection, defaults to
            12 * (nobs/100.)**(1./4), see select_order function
        method : {'ols'}
            Estimation method to use
        ic : {'aic', 'fpe', 'hqic', 'bic', None}
            Information criterion to use for VAR order selection.
            aic : Akaike
            fpe : Final prediction error
            hqic : Hannan-Quinn
            bic : Bayesian a.k.a. Schwarz
        verbose : bool, default False
            Print order selection output to the screen
        trend, str {"c", "ct", "ctt", "nc"}
            "c" - add constant
            "ct" - constant and trend
            "ctt" - constant, linear and quadratic trend
            "nc" - co constant, no trend
            Note that these are prepended to the columns of the dataset.

        Notes
        -----
        Lutkepohl pp. 146-153

        Returns
        -------
        est : VARResults
        """#TODO: docstring + implementation
        print("fit called") #TODO: delete this line
        
        #TODO: trend
        
        # select number of lags
        if ic is None:
            if maxlags is None:
                lags = 1
            else:
                lags = maxlags
        else:
            selections = self.select_order(maxlags=maxlags, verbose=verbose)
            if ic not in selections:
                raise Exception("%s not recognized, must be among %s"
                                % (ic, sorted(selections)))
            lags = selections[ic]
            if verbose:
                print('Using %d based on %s criterion' %  (lags, ic))
        print(lags)
        
        # estimate parameters
        self._estimate_vecm(lags)
        
    def _estimate_vecm(self, lags):
        """
        lags : int
        offset : int
            Periods to drop from beginning-- for order selection so it's an
            apples-to-apples comparison
        trend : string or None
            As per above
        """#TODO: docstring + implementation
        pass

    def estimation_matrices(data, diff_lags, deterministic_terms):
        p = diff_lags+1
        y_all = data.values.T
        K = y_all.shape[0]
        
        y_1_T = y_all[:, p:]
        T = y_1_T.shape[1]
        
        Delta_y_all = np.diff(y_all)
        Delta_y_1_T = Delta_y_all[:, p-1:]
        
        y_minus1 = y_all[:, p-1:-1]
        if 'lt' in deterministic_terms:
            y_minus1 = np.vstack((y_minus1, np.arange(T)))
        
        Delta_x = np.zeros((diff_lags*K, T))
        for j in range(Delta_x.shape[1]):
            Delta_x[:, j] = Delta_y_all[:, j+p-2:None if j-1<0 else j-1:-1].T.reshape(K*(p-1))
        if 'c' in deterministic_terms:
            Delta_x = np.vstack((Delta_x, np.ones(T)))
        if 's' in deterministic_terms: # TODO: How many seasons??
            seasons = np.zeros((3, Delta_x.shape[1])) # 3 = 4-1 (4=number_of_seasons)
            for i in range(seasons.shape[0]):
                seasons[i, i::4] = i+1
            Delta_x = np.vstack((Delta_x, seasons))
        
        return y_all, y_1_T, Delta_y_all, Delta_y_1_T, y_minus1, Delta_x

    def _ls_Pi_Gamma(Delta_y_1_T, y_minus1, Delta_x, deterministic_terms):
        mat1 = np.hstack((Delta_y_1_T.dot(y_minus1.T), Delta_y_1_T.dot(Delta_x.T)))

        B = y_minus1.dot(Delta_x.T)
        mat2 = np.linalg.inv(np.vstack( (np.hstack( (y_minus1.dot(y_minus1.T),   B                    ) ),
                                         np.hstack( (B.T,                        Delta_x.dot(Delta_x.T) )) ) ))

        estPiGamma = mat1.dot(mat2)

        Pi_hat_cols = K if 'lt' not in deterministic_terms else K+1
        Pi_hat = estPiGamma[:, :Pi_hat_cols]

        Gamma_hat = estPiGamma[:, Pi_hat_cols:]
        #print("Dy1T.shape: "+str(Delta_y_1_T.shape) + ",  Pi_hat.shape: "+str(Pi_hat.shape) + ",  Gamma_hat.shape: "+str(Gamma_hat.shape))
        #print("yminus1:" + str(y_minus1.shape))
        _A = Delta_y_1_T - Pi_hat.dot(y_minus1) - Gamma_hat.dot(Delta_x)
        p = diff_lags+1
        Sigma_u_hat = 1/(T-K*p) * np.dot(_A,_A.T)

        return (Pi_hat, Gamma_hat, Sigma_u_hat)

    def ls_estimator(data, diff_lags, deterministic_terms):    
        # deterministic_terms \in \{'c', 'lt', 's'\} c=constant, lt=linear trend, s=seasonal terms

        y_all, y_1_T, Delta_y_all, Delta_y_1_T, y_minus1, Delta_x = estimation_matrices(data, diff_lags, deterministic_terms)
        K = y_all.shape[0]
        T = y_1_T.shape[1]

        Pi_hat, Gamma_hat, Sigma_u_hat = _ls_Pi_Gamma(Delta_y_1_T, y_minus1, Delta_x, deterministic_terms)
        
        print("Pi:")
        print(Pi_hat)
        print("Gamma:")
        print(Gamma_hat)
        print("Sigma_u")
        print(Sigma_u_hat)
        
        return {"Pi_hat": Pi_hat, "Gamma_hat": Gamma_hat, "Sigma_u_hat": Sigma_u_hat}        

    def _M_and_R_matrices(T, Delta_x, Delta_y_1_T, y_minus1):
        M = np.identity(T) - Delta_x.T.dot(np.linalg.inv(Delta_x.dot(Delta_x.T))).dot(Delta_x)
        R_0 = Delta_y_1_T.dot(M)
        R_1 = y_minus1.dot(M)
        return (M, R_0, R_1)
    
    def egls_estimator(data, diff_lags, deterministic_terms='', r=1):    
        _ya, y_1_T, _Dya, Delta_y_1_T, y_minus1, Delta_x = estimation_matrices(data, diff_lags, deterministic_terms)
        K = y_all.shape[0]
        T = y_1_T.shape[1]
        
        Pi_hat, _Gamma_hat, Sigma_u_hat = _ls_Pi_Gamma(Delta_y_1_T, y_minus1, Delta_x, deterministic_terms)
        #print("Dy1T.shape: "+str(Delta_y_1_T.shape) + ",  Pi_hat.shape: "+str(Pi_hat.shape) + ",  Gamma_hat.shape: "+str(Gamma_hat.shape))
        #print("yminus1: " + str(y_minus1.shape) + ",  DX.shape: "+str(Delta_x.shape))
        alpha_hat = Pi_hat[:, :r]
        
        M, R_0, R_1 = _M_and_R_matrices(T, Delta_x, Delta_y_1_T, y_minus1)
        R_11 = R_1[:r]
        R_12 = R_1[r:]
        #print("Sigma_u_hat: \n"+str(Sigma_u_hat))
        _alpha_Sigma = alpha_hat.T.dot(np.linalg.inv(Sigma_u_hat))
        beta_hhat = np.linalg.inv(_alpha_Sigma.dot(alpha_hat)).dot(_alpha_Sigma).dot(R_0-alpha_hat.dot(R_11)).dot(R_12.T).dot(np.linalg.inv(R_12.dot(R_12.T))).T
        beta_hhat = np.vstack((1, beta_hhat))
        # ? Gamma_hhat necessary / computed via (Delta_y_1_T - alpha_hat.dot(beta_hhat.T).dot(y_minus1)).dot(Delta_x.dot(np.linalg.inv(np.dot(Delta_x,Delta_x.T))))
        
        print("alpha")
        print(alpha_hat.round(3))
        print("beta")
        print(beta_hhat.round(3))
    
    def ml_estimator(data, diff_lags, deterministic_terms='', r=1):
        _ya, y_1_T, _Dya, Delta_y_1_T, y_minus1, Delta_x = estimation_matrices(data, diff_lags, deterministic_terms)
        K = y_all.shape[0]
        T = y_1_T.shape[1]
        
        M, R_0, R_1 = _M_and_R_matrices(T, Delta_x, Delta_y_1_T, y_minus1)
        S = np.bmat([[Ri.dot(Rj.T)/T for Rj in [R_0, R_1]] for Ri in [R_0, R_1]])
        S00 = S[:K, :K]
        S01 = S[:K, K:]
        S10 = S[K:, :K]
        S11 = S[K:, K:]
        S11_ = scipy.linalg.sqrtm(S11.I)

        v = np.linalg.eig(S11_ * S10 * S00.I * S01 * S11_)[1]
        m_beta_tilde = (v[:,:r].T * S11_).T
        m_beta_tilde = m_beta_tilde * m_beta_tilde[:r,:r].I # normalize such that eye(r) forms the first r rows of beta tilde
        m_alpha_tilde = S01 * m_beta_tilde * (m_beta_tilde.T * S11 * m_beta_tilde).I

        m_Delta_y_1_T = np.matrix(Delta_y_1_T)
        m_y_minus1 = np.matrix(y_minus1)
        m_Delta_x = np.matrix(Delta_x)
        m_Gamma_tilde = (m_Delta_y_1_T - m_alpha_tilde*m_beta_tilde.T*m_y_minus1) * m_Delta_x.T * (m_Delta_x * m_Delta_x.T).I
        m_temp = (m_Delta_y_1_T - m_alpha_tilde*m_beta_tilde.T*m_y_minus1 - m_Gamma_tilde*m_Delta_x)
        m_Sigma_u_tilde = m_temp * m_temp.T / T
        print("alpha")
        print(m_alpha_tilde)
        print("beta")
        print(m_beta_tilde)
        print("==> Pi")
        print(m_alpha_tilde*m_beta_tilde.T)
        print("Gamma")
        print(m_Gamma_tilde)
        print("Sigma_u")
        print(m_Sigma_u_tilde)
        

    def select_order(self, maxlags=None, verbose=True):
        """
        Compute lag order selections based on each of the available information
        criteria

        Parameters
        ----------
        maxlags : int
            if None, defaults to 12 * (nobs/100.)**(1./4)
        verbose : bool, default True
            If True, print table of info criteria and selected orders

        Returns
        -------
        selections : dict {info_crit -> selected_order}
        """#TODO: docstring + implementation. Function used in fit().
        pass



