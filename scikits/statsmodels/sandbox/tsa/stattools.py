"""
Statistical tools for time series analysis
"""

import numpy as np
from scipy import stats, signal
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
    X = np.asarray(X).squeeze()
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

def pergram(X, kernel='bartlett', log=True):
    """
    Returns the (log) periodogram for the natural frequency of X

    Parameters
    ----------
    X
    M : int
        Should this be hardcoded?
    kernel : str, optional
    Notes
    -----
    The autocovariances are normalized by len(X).
    The frequencies are calculated as
    If len(X) is odd M = (len(X) - 1)/2 else M = len(X)/2. Either way
        freq[i] = 2*[i+1]/T and len(freq) == M


    Reference
    ----------
    Based on Lutkepohl; Hamilton.

    Notes
    -----
    Doesn't look right yet.
    """
    X = np.asarray(X).squeeze()
    nobs = len(X)
    M = np.floor(nobs/2.)
    acov = np.zeros(M+1)
    acov[0] = np.var(X)
    Xbar = X.mean()
    for i in range(1,int(M+1)):
        acov[i] = np.dot(X[i:] - Xbar,X[:-i] - Xbar)
    acov /= nobs
    #    #TODO: make a list to check window
#    ell = np.r_[1,np.arange(1,M+1)*np.pi/nobs]
    if kernel == "bartlett":
        w = 1 - np.arange(M+1)/M

#    weights = exec('signal.'+window+'(M='str(M)')')
    j = np.arange(1,M+1)
    ell = np.linspace(0,np.pi,M)
    pergr = np.zeros_like(ell)
    for i,L in enumerate(ell):
        pergr[i] = 1/(2*np.pi)*acov[0] + 2 * np.sum(w[1:]*acov[1:]*np.cos(L*j))
    return pergr



if __name__=="__main__":
    data = sm.datasets.macrodata.Load().data
    adf = dfuller(data['realgdp'],4)
    acf,ci = acorr(data['realgdp'])
    pacf = pacorr(data['realgdp'])

