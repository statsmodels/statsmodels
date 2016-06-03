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
        super(VAR, self).__init__(endog, None, dates, freq, missing=missing)
        if self.endog.ndim == 1:
            raise ValueError("Only gave one variable to VAR")
        self.y = self.endog #keep alias for now
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
        pass

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
        """#TODO: docstring + implementation
        pass



