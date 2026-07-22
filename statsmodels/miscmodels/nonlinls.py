"""
Non-linear least squares

Author: Josef Perktold based on scipy.optimize.curve_fit
"""

import numpy as np
from scipy import optimize

from statsmodels.base.model import Model


class Results:
    """
    Just a dummy placeholder for now

    Most results from RegressionResults can be used here.
    """


# def getjaccov(retval, n):
#    '''calculate something and raw covariance matrix from return of optimize.leastsq
#
#    I cannot figure out how to recover the Jacobian, or whether it is even
#    possible
#
#    this is a partial copy of scipy.optimize.leastsq
#    '''
#    info = retval[-1]
#    # n = len(x0)  # nparams, where do I get this
#    cov_x = None
#    if info in [1,2,3,4]:
#        from numpy.dual import inv
#        from numpy.linalg import LinAlgError
#        perm = np.take(np.eye(n), retval[1]['ipvt']-1,0)
#        r = np.triu(np.transpose(retval[1]['fjac'])[:n,:])
#        R = np.dot(r, perm)
#        try:
#            cov_x = inv(np.dot(np.transpose(R),R))
#        except LinAlgError:
#            print 'cov_x not available'
#            pass
#        return r, R, cov_x
#
# def _general_function(params, xdata, ydata, function):
#    return function(xdata, *params) - ydata
#
# def _weighted_general_function(params, xdata, ydata, function, weights):
#    return weights * (function(xdata, *params) - ydata)
#


class NonlinearLS(Model):  # or subclass a model
    r"""
    Base class for estimation of a non-linear model with least squares

    This class is supposed to be subclassed, and the subclass has to provide a method
    `_predict` that defines the non-linear function `f(params) that is predicting the endogenous
    variable. The model is assumed to be

    :math: y = f(params) + error

    and the estimator minimizes the sum of squares of the estimated error.

    :math: min_parmas \sum (y - f(params))**2

    f has to return the prediction for each observation. Exogenous or explanatory variables
    should be accessed as attributes of the class instance, and can be given as arguments
    when the instance is created.

    Similar to scipy.optimize.curve_fit. The main API difference is that
    `params` are array_like and not split up, so `n_params` information is
    needed. This also includes weights similar to curve_fit, but there is
    no general sigma yet (OLS and WLS, but no GLS).

    This is currently holding on to intermediate results that are not
    necessary but useful for testing.

    ``fit`` returns an instance of RegressionResults, in contrast to the
    linear model, results in this case are based on a local approximation,
    essentially y = f(X, params) is replaced by y = grad * params where grad
    is the Gradient or Jacobian with the shape (nobs, nparams). See for
    example Greene.

    Parameters
    ----------
    endog : array_like, optional
        The dependent (endogenous) variable.
    exog : array_like, optional
        The independent (exogenous) variable(s) used by `_predict`.
    weights : array_like, optional
        Weights used when computing the errors. Currently unused directly
        in `__init__`; see `sigma`.
    sigma : array_like, optional
        1-d array of standard deviations used to construct `weights` as
        ``1 / sigma``. Correlated errors (2-d `sigma`) are not supported.
    missing : str
        Available options are 'none', 'drop', and 'raise'. Currently not
        used in `__init__`.

    Warnings
    --------
    Weights are not correctly handled yet in the results statistics,
    but are included when estimating the parameters.

    Examples
    --------
    ::

        class Myfunc(NonlinearLS):

            def _predict(self, params):
                x = self.exog
                a, b, c = params
                return a*np.exp(-b*x) + c

    If we have data (y, x), we can create an instance and fit it with::

        mymod = Myfunc(y, x)
        myres = mymod.fit(nparams=3)

    and use the non-linear regression results, for example::

        myres.params
        myres.bse
        myres.tvalues
    """

    # NOTE: This needs to call super for data checking
    def __init__(self, endog=None, exog=None, weights=None, sigma=None, missing="none"):
        self.endog = endog
        self.exog = exog
        if sigma is not None:
            sigma = np.asarray(sigma)
            if sigma.ndim < 2:
                self.sigma = sigma
                self.weights = 1.0 / sigma
            else:
                raise ValueError("correlated errors are not handled yet")
        else:
            self.weights = None

    def predict(self, exog, params=None):
        """
        Return the predicted values for `params`

        Parameters
        ----------
        exog : array_like
            Not used directly, present for signature compatibility with
            `Model.predict`. The exogenous data stored on the instance is
            used instead.
        params : array_like, optional
            The parameters at which to evaluate the prediction function.

        Returns
        -------
        ndarray
            The predicted values from `_predict`.
        """
        # copied from GLS, Model has different signature
        return self._predict(params)

    def _predict(self, params):
        """
        Non-linear prediction function, to be defined by a subclass

        Parameters
        ----------
        params : array_like
            The parameters at which to evaluate the prediction function.
        """

    def start_value(self):
        """
        Return starting values for the parameters

        Returns
        -------
        None
            The base class does not provide starting values; subclasses
            can override this method to do so.
        """
        return None

    def geterrors(self, params, weights=None):
        """
        Return the (optionally weighted) residuals at `params`

        Parameters
        ----------
        params : array_like
            The parameters at which to evaluate the prediction function.
        weights : array_like, optional
            Weights to apply to the residuals. If None, `self.weights` is
            used when available.

        Returns
        -------
        ndarray
            The residuals, ``endog - predict(params)``, optionally
            weighted.
        """
        if weights is None:
            if self.weights is None:
                return self.endog - self._predict(params)
            else:
                weights = self.weights
        return weights * (self.endog - self._predict(params))

    def errorsumsquares(self, params):
        """
        Return the sum of squared (weighted) residuals at `params`

        Parameters
        ----------
        params : array_like
            The parameters at which to evaluate the prediction function.

        Returns
        -------
        float
            The sum of squared residuals.
        """
        return (self.geterrors(params) ** 2).sum()

    def fit(self, start_value=None, nparams=None, **kw):
        """
        Estimate the parameters of the model by non-linear least squares

        Parameters
        ----------
        start_value : array_like, optional
            Starting values for the optimization. If None, `start_value`
            (the method) is used, falling back to an array of 0.1s of
            length `nparams` if that also returns None.
        nparams : int, optional
            Number of parameters, only used to construct default starting
            values when `start_value` is not provided.
        **kw
            Additional keyword arguments passed through to
            `scipy.optimize.leastsq`.

        Returns
        -------
        RegressionResults
            The fitted regression results, based on a local linear
            approximation using the Jacobian of `_predict`.
        """
        # if hasattr(self, 'start_value'):
        # I added start_value even if it's empty, not sure about it
        # but it makes a visible placeholder

        if start_value is not None:
            p0 = start_value
        else:
            # nesting so that start_value is only calculated if it is needed
            p0 = self.start_value()
            if p0 is not None:
                pass
            elif nparams is not None:
                p0 = 0.1 * np.ones(nparams)
            else:
                raise ValueError("need information about start values for optimization")

        func = self.geterrors
        res = optimize.leastsq(func, p0, full_output=1, **kw)
        popt, pcov, infodict, errmsg, ier = res

        if ier not in [1, 2, 3, 4]:
            msg = "Optimal parameters not found: " + errmsg
            raise RuntimeError(msg)

        err = infodict["fvec"]

        ydata = self.endog
        if (len(ydata) > len(p0)) and pcov is not None:
            # this can use the returned errors instead of recalculating

            s_sq = (err**2).sum() / (len(ydata) - len(p0))
            pcov = pcov * s_sq
        else:
            pcov = None

        self.df_resid = len(ydata) - len(p0)
        self.df_model = len(p0)
        fitres = Results()
        fitres.params = popt
        fitres.pcov = pcov
        fitres.rawres = res
        self.wendog = self.endog  # add weights
        self.wexog = self.jac_predict(popt)
        pinv_wexog = np.linalg.pinv(self.wexog)
        self.normalized_cov_params = np.dot(pinv_wexog, np.transpose(pinv_wexog))

        # TODO: check effect of `weights` on result statistics
        # I think they are correctly included in cov_params
        # maybe not anymore, I'm not using pcov of leastsq
        # direct calculation with jac_predict misses the weights

        # if not weights is None
        #     fitres.wexogw = self.weights * self.jacpredict(popt)
        from statsmodels.regression.linear_model import RegressionResults

        beta = popt
        lfit = RegressionResults(
            self, beta, normalized_cov_params=self.normalized_cov_params
        )

        lfit.fitres = fitres  # mainly for testing
        self._results = lfit
        return lfit

    def fit_minimal(self, start_value, **kwargs):
        """
        Minimal fitting with no extra calculations

        Parameters
        ----------
        start_value : array_like
            Starting values for the optimization.
        **kwargs
            Additional keyword arguments passed through to
            `scipy.optimize.leastsq`.

        Returns
        -------
        tuple
            The raw return value of `scipy.optimize.leastsq`.
        """
        func = self.geterrors
        res = optimize.leastsq(func, start_value, full_output=0, **kwargs)
        return res

    def fit_random(self, ntries=10, rvs_generator=None, nparams=None):
        """
        Fit with random starting values

        This could be replaced with a global fitter.

        Parameters
        ----------
        ntries : int
            Number of random starting values to try.
        rvs_generator : callable, optional
            Function to generate random starting values given a `size`
            keyword. If None, `numpy.random.uniform` with low=-10, high=10
            is used.
        nparams : int, optional
            Number of parameters. If None, `self.nparams` is used.

        Returns
        -------
        ndarray
            Array with one row per try, containing the fitted parameters,
            residual, and other `leastsq` output concatenated with the
            random starting values used.
        """

        if nparams is None:
            nparams = self.nparams
        if rvs_generator is None:
            rvs = np.random.uniform(low=-10, high=10, size=(ntries, nparams))
        else:
            rvs = rvs_generator(size=(ntries, nparams))

        results = np.array([np.r_[self.fit_minimal(rv), rv] for rv in rvs])
        # selct best results and check how many solutions are within 1e-6 of best
        # not sure what leastsq returns
        return results

    def jac_predict(self, params):
        """
        Jacobian of prediction function using complex step derivative

        This assumes that the predict function does not use complex variable
        but is designed to do so.

        Parameters
        ----------
        params : array_like
            The parameters at which to evaluate the Jacobian.

        Returns
        -------
        ndarray
            The Jacobian of `_predict` with respect to `params`, with shape
            (nobs, nparams).
        """
        from statsmodels.tools.numdiff import approx_fprime_cs

        jaccs_err = approx_fprime_cs(params, self._predict)
        return jaccs_err


class Myfunc(NonlinearLS):

    # predict model.Model has a different signature
    #    def predict(self, params, exog=None):
    #        if not exog is None:
    #            x = exog
    #        else:
    #            x = self.exog
    #        a, b, c = params
    #        return a*np.exp(-b*x) + c

    def _predict(self, params):
        x = self.exog
        a, b, c = params
        return a * np.exp(-b * x) + c


if __name__ == "__main__":

    def func0(x, a, b, c):
        return a * np.exp(-b * x) + c

    def func(params, x):
        a, b, c = params
        return a * np.exp(-b * x) + c

    def error(params, x, y):
        return y - func(params, x)

    def error2(params, x, y):
        return (y - func(params, x)) ** 2

    x = np.linspace(0, 4, 50)
    params = np.array([2.5, 1.3, 0.5])
    y0 = func(params, x)
    y = y0 + 0.2 * np.random.normal(size=len(x))

    res = optimize.leastsq(error, params, args=(x, y), full_output=True)
    #    r, R, c = getjaccov(res[1:], 3)

    mod = Myfunc(y, x)
    resmy = mod.fit(nparams=3)

    cf_params, cf_pcov = optimize.curve_fit(func0, x, y)
    cf_bse = np.sqrt(np.diag(cf_pcov))
    print(res[0])
    print(cf_params)
    print(resmy.params)
    print(cf_bse)
    print(resmy.bse)
