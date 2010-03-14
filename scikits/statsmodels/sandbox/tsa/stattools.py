"""
Statistical tools for time series analysis
"""

import numpy as np
from scipy import stats
import scikits.statsmodels as sm
from scikits.statsmodels.sandbox.tsa.varma_tools import lagmat
#from scikits.statsmodels.sandbox.rls import RLS

def dfuller(X, nlags=1, noconstant=False, trend=False):
    """
    Augmented Dickey-Fuller test for a time series X.

    Parameters
    ----------
    X - array-like
        The time series that might contain a unit root.
    nlags - int
        nlags should be >= 0.
        TODO: Allow string 'AIC' and 'BIC'
    noconstant - bool
        Include a constant or not.  Default is false, so that a constant
        term is estimated.
    trend : bool
        Include a linear time trend or not.  Default is false for no
        time trend.

    Notes
    -----
    The ADF test statistic is not returned.  Refer to a ADF table for
    inference.  Is nlags = 0, dfuller returns the classic Dickey-Fuller
    statistic.

    """
    if nlags < 0:
        raise ValueError, "nlags should be >= 0"
    X = np.asarray(X)
    nobs = float(len(X))
    xdiff = np.diff(X)
    xlag1 = X[:-1]
    xdiffp = lagmat(xdiff,nlags,trim='both')[:,1:]
    t = np.arange(1,nobs-nlags)
    RHS = np.column_stack((xlag1[nlags:],xdiffp))
    if trend:
        t = np.arange(1,nobs-nlags)
        RHS = np.column_stack((t,RHS))
    if not noconstant:
        RHS = sm.add_constant(RHS,prepend=True)
    results = sm.OLS(xdiff[nlags:],RHS).fit()
    return results.t()[-(nlags+1)]

def acorr(X,nlags=40, level=95):
    """
    Autocorrelation function

    Parameters
    ----------
    X : array-like
        The time series
    nlags : int
        The number of lags to estimate the autocorrelation function for.
        Default is 40.
    level : int
        The level of the confidence intervals.  Defaults is 95 % confidence
        intervals.

    Returns
    -------
    acf : array
        The values of the autocorrelation function
    conf_int(acf) : array
        The confidence interval of acf at level
    """
    X = np.asarray(X).squeeze()
    nobs = float(len(X))
    if nlags > nobs:
        raise ValueError, "X does not have %s observations" % nlags
    Xbar = np.mean(X)
    acf = np.zeros(nlags)
    acov0 = np.var(X)
    for i in range(1,nlags+1):
        acf[i-1] = np.dot(X[i:] - Xbar,X[:-i] - Xbar)
    acf = 1/nobs*acf/acov0
    varacf = np.ones(nlags)/nobs
    varacf[1:] *= 1 + 2*np.cumsum(acf[1:]**2)
    confint = np.array(zip(acf - stats.norm.ppf(1-(100 - level)/\
            200.)*np.sqrt(varacf), acf+stats.norm.ppf(1-(100-level)/200.)\
            *np.sqrt(varacf)))
    return acf,confint

def pacorr(X,nlags=40, method="ols"):
    """
    Partial autocorrelation function
    """
    X = np.asarray(X).squeeze()
    nobs = float(len(X))
    if nlags > nobs:
        raise ValueError, "X does not have %s observations" % nlags
    pacf = np.zeros(nlags)
    for i in range(1,nlags+1):
        pacf[i-1] = sm.OLS(X[i:],sm.add_constant(lagmat(X, i,
            trim="both")[:,1:], prepend=True)).fit().params[-1]
    return pacf



if __name__=="__main__":
    data = sm.datasets.macrodata.Load().data
    adf = dfuller(data['realgdp'],4)
    acf,ci = acorr(data['realgdp'])
    pacf = pacorr(data['realgdp'])

