class HCCM(OLSModel):
    """
    Heteroskedasticity-Corrected Covariance Matrix estimation

    Parameters
    ----------
    design : array-like

    Methods
    -------
    Same as OLSModel except that fit accepts an additional argument
    to define the type of correction.

    """

    def __init__(self, design, hascons=True):
        super(HCCM, self).__init__(design, hascons)

    def fit(self, Y, robust='HC0'):
        '''
        Parameters
        -----------
        Y : array-like
            Response variable
        robust : string, optional
            Estimation of the heteroskedasticity robust covariance matrix
            Values can be "HC0", "HC1", "HC2", or "HC3"
            Default is HC0
        '''
        Z = self.whiten(Y)
        lfit = RegressionResults(np.dot(self.calc_beta, Z), Y,
                       normalized_cov_beta=self.normalized_cov_beta)
        lfit.predict = np.dot(self.design, lfit.beta)
        lfit.resid = Z - np.dot(self.wdesign, lfit.beta)
        if robust is "HC0": # HC0 (White 1980)
            lfit.scale = np.diag(lfit.resid**2)
        elif robust is "HC1": # HC1-3 MacKinnon and White (1985)
            lfit.scale = lfit.n/(lfit.n-lfit.df_model-1)*(np.diag(lfit.resid**2))
        elif robust is "HC2":
            h=np.diag(np.dot(np.dot(self.wdesign,self.normalized_cov_beta),
                    self.wdesign.T))
            lfit.scale=np.diag(lfit.resid**2/(1-h))
        elif robust is "HC3":
             h=np.diag(np.dot(np.dot(self.wdesign,self.normalized_cov_beta),
                    self.wdesign.T))
             lfit.scale=np.diag((lfit.resid/(1-h))**2)
        else:
            raise ValueError, "Robust option %s not understood" % robust
        lfit.df_resid = self.df_resid
        lfit.df_model = self.df_model
        lfit.Z = Z
        lfit.calc_beta = self.calc_beta # needed for cov_beta()
        return lfit

