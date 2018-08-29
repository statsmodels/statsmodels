from collections import namedtuple
import numpy as np
from scipy import stats

from statsmodels.tools.tools import Bunch

_MinimalWLSModel = namedtuple('_MinimalWLSModel', ['weights'])


class _MinimalWLS(object):
    """
    Minimal implementation of WLS optimized for performance.

    Parameters
    ----------
    endog : array-like
        1-d endogenous response variable. The dependent variable.
    exog : array-like
        A nobs x k array where `nobs` is the number of observations and `k`
        is the number of regressors. An intercept is not included by default
        and should be added by the user. See
        :func:`statsmodels.tools.add_constant`.
    weights : array-like, optional
        1d array of weights.  If you supply 1/W then the variables are pre-
        multiplied by 1/sqrt(W).  If no weights are supplied the default value
        is 1 and WLS reults are the same as OLS.

    Notes
    -----
    Need resid, scale, fittedvalues, model.weights!
        history['scale'].append(tmp_results.scale)
        if conv == 'dev':
            history['deviance'].append(self.deviance(tmp_results))
        elif conv == 'sresid':
            history['sresid'].append(tmp_results.resid/tmp_results.scale)
        elif conv == 'weights':
            history['weights'].append(tmp_results.model.weights)
    Does not perform and checks on the input data
    """

    def __init__(self, endog, exog, weights=1.0):
        self.endog = endog
        self.exog = exog
        self.weights = weights
        w_half = np.sqrt(weights)

        self.wendog = w_half * endog
        if np.isscalar(weights):
            self.wexog = w_half * exog
        else:
            self.wexog = w_half[:, None] * exog

    def fit(self, method='pinv'):
        """
        Minimal implementation of WLS optimized for performance.

        Parameters
        ----------
        method : str, optional
            Method to use to estimate parameters.  "pinv", "qr" or "lstsq"

              * "pinv" uses the Moore-Penrose pseudoinverse
                 to solve the least squares problem.
              * "qr" uses the QR factorization.
              * "lstsq" uses the least squares implementation in numpy.linalg

        Returns
        -------
        results : namedtuple
            Named tuple containing the fewest terms needed to implement
            iterative estimation in models. Currently

              * params : Estimated parameters
              * fittedvalues : Fit values using original data
              * resid : Residuals using original data
              * model : namedtuple with one field, weights
              * scale : scale computed using weighted residuals

        Notes
        -----
        Does not perform and checks on the input data

        See Also
        --------
        statsmodels.regression.linear_model.WLS
        """
        if method == 'pinv':
            pinv_wexog = np.linalg.pinv(self.wexog)
            params = pinv_wexog.dot(self.wendog)
        elif method == 'qr':
            Q, R = np.linalg.qr(self.wexog)
            params = np.linalg.solve(R, np.dot(Q.T, self.wendog))
        else:
            params, _, _, _ = np.linalg.lstsq(self.wexog, self.wendog,
                                              rcond=-1)

        fitted_values = self.exog.dot(params)
        resid = self.endog - fitted_values
        wresid = self.wendog - self.wexog.dot(params)
        df_resid = self.wexog.shape[0] - self.wexog.shape[1]
        scale = np.dot(wresid, wresid) / df_resid

        return Bunch(params=params, fittedvalues=fitted_values, resid=resid,
                     model=self, scale=scale)


def wls_prediction_std(res, exog=None, weights=None, alpha=0.05):
    """calculate standard deviation and confidence interval for prediction

    applies to WLS and OLS, not to general GLS,
    that is independently but not identically distributed observations

    Parameters
    ----------
    res : regression result instance
        results of WLS or OLS regression required attributes see notes
    exog : array_like (optional)
        exogenous variables for points to predict
    weights : scalar or array_like (optional)
        weights as defined for WLS (inverse of variance of observation)
    alpha : float (default: alpha = 0.05)
        confidence level for two-sided hypothesis

    Returns
    -------
    predstd : array_like, 1d
        standard error of prediction
        same length as rows of exog
    interval_l, interval_u : array_like
        lower und upper confidence bounds

    Notes
    -----
    The result instance needs to have at least the following
    res.model.predict() : predicted values or
    res.fittedvalues : values used in estimation
    res.cov_params() : covariance matrix of parameter estimates

    If exog is 1d, then it is interpreted as one observation,
    i.e. a row vector.

    testing status: not compared with other packages

    References
    ----------
    Greene p.111 for OLS, extended to WLS by analogy
    """

    covb = res.cov_params()
    if exog is None:
        exog = res.model.exog
        predicted = res.fittedvalues
        if weights is None:
            weights = res.model.weights
    else:
        exog = np.atleast_2d(exog)
        if covb.shape[1] != exog.shape[1]:
            raise ValueError('wrong shape of exog')
        predicted = res.model.predict(res.params, exog)
        if weights is None:
            weights = 1.
        else:
            weights = np.asarray(weights)
            if weights.size > 1 and len(weights) != exog.shape[0]:
                raise ValueError('weights and exog do not have matching shape')

    # full covariance:
    #   predvar = res3.mse_resid + np.diag(np.dot(X2, np.dot(covb, X2.T)))
    # predication variance only:
    predvar = res.mse_resid/weights + (exog * np.dot(covb, exog.T).T).sum(1)
    predstd = np.sqrt(predvar)
    tppf = stats.t.isf(alpha/2., res.df_resid)
    interval_u = predicted + tppf * predstd
    interval_l = predicted - tppf * predstd
    return predstd, interval_l, interval_u
