"""
Statistical tools for time series analysis
"""

import numpy as np
from scipy import stats, signal
from statsmodels.regression.linear_model import OLS, yule_walker
from statsmodels.tools.tools import add_constant
from tsatools import lagmat, lagmat2ds, add_trend
#from statsmodels.sandbox.tsa import var
from adfvalues import *
#from statsmodels.sandbox.rls import RLS

#NOTE: now in two places to avoid circular import
#TODO: I like the bunch pattern for this too.
class ResultsStore(object):
    def __str__(self):
        return self._str

def _autolag(mod, endog, exog, startlag, maxlag, method, modargs=(),
        fitargs=(), regresults=False):
    """
    Returns the results for the lag length that maximimizes the info criterion.

    Parameters
    ----------
    mod : Model class
        Model estimator class.
    modargs : tuple
        args to pass to model.  See notes.
    fitargs : tuple
        args to pass to fit.  See notes.
    lagstart : int
        The first zero-indexed column to hold a lag.  See Notes.
    maxlag : int
        The highest lag order for lag length selection.
    method : str {"aic","bic","t-stat"}
        aic - Akaike Information Criterion
        bic - Bayes Information Criterion
        t-stat - Based on last lag

    Returns
    -------
    icbest : float
        Best information criteria.
    bestlag : int
        The lag length that maximizes the information criterion.


    Notes
    -----
    Does estimation like mod(endog, exog[:,:i], *modargs).fit(*fitargs)
    where i goes from lagstart to lagstart+maxlag+1.  Therefore, lags are
    assumed to be in contiguous columns from low to high lag length with
    the highest lag in the last column.
    """
#TODO: can tcol be replaced by maxlag + 2?
#TODO: This could be changed to laggedRHS and exog keyword arguments if this
#    will be more general.

    results = {}
    method = method.lower()
    for lag in range(startlag, startlag+maxlag+1):
        mod_instance = mod(endog, exog[:,:lag], *modargs)
        results[lag] = mod_instance.fit()

    if method == "aic":
        icbest, bestlag = min((v.aic,k) for k,v in results.iteritems())
    elif method == "bic":
        icbest, bestlag = min((v.bic,k) for k,v in results.iteritems())
    elif method == "t-stat":
        lags = sorted(results.keys())[::-1]
#        stop = stats.norm.ppf(.95)
        stop = 1.6448536269514722
        for lag in range(startlag + maxlag, startlag - 1, -1):
            icbest = np.abs(results[lag].tvalues[-1])
            if np.abs(icbest) >= stop:
                bestlag = lag
                icbest = icbest
                break
    else:
        raise ValueError("Information Criterion %s not understood.") % method

    if not regresults:
        return icbest, bestlag
    else:
        return icbest, bestlag, results

#this needs to be converted to a class like HetGoldfeldQuandt, 3 different returns are a mess
# See:
#Ng and Perron(2001), Lag length selection and the construction of unit root
#tests with good size and power, Econometrica, Vol 69 (6) pp 1519-1554
#TODO: include drift keyword, only valid with regression == "c"
# just changes the distribution of the test statistic to a t distribution
#TODO: autolag is untested
def adfuller(x, maxlag=None, regression="c", autolag='AIC',
    store=False, regresults=False):
    '''Augmented Dickey-Fuller unit root test

    The Augmented Dickey-Fuller test can be used to test for a unit root in a
    univariate process in the presence of serial correlation.

    Parameters
    ----------
    x : array_like, 1d
        data series
    maxlag : int
        Maximum lag which is included in test, default 12*(nobs/100)^{1/4}
    regression : str {'c','ct','ctt','nc'}
        Constant and trend order to include in regression
        * 'c' : constant only
        * 'ct' : constant and trend
        * 'ctt' : constant, and linear and quadratic trend
        * 'nc' : no constant, no trend
    autolag : {'AIC', 'BIC', 't-stat', None}
        * if None, then maxlag lags are used
        * if 'AIC' or 'BIC', then the number of lags is chosen to minimize the
          corresponding information criterium
        * 't-stat' based choice of maxlag.  Starts with maxlag and drops a
          lag until the t-statistic on the last lag length is significant at
          the 95 % level.
    store : bool
        If True, then a result instance is returned additionally to
        the adf statistic
    regresults : bool
        If True, the full regression results are returned.

    Returns
    -------
    adf : float
        Test statistic
    pvalue : float
        MacKinnon's approximate p-value based on MacKinnon (1994)
    usedlag : int
        Number of lags used.
    nobs : int
        Number of observations used for the ADF regression and calculation of
        the critical values.
    critical values : dict
        Critical values for the test statistic at the 1 %, 5 %, and 10 % levels.
        Based on MacKinnon (2010)
    icbest : float
        The maximized information criterion if autolag is not None.
    regresults : RegressionResults instance
        The
    resstore : (optional) instance of ResultStore
        an instance of a dummy class with results attached as attributes

    Notes
    -----
    The null hypothesis of the Augmented Dickey-Fuller is that there is a unit
    root, with the alternative that there is no unit root. If the pvalue is
    above a critical size, then we cannot reject that there is a unit root.

    The p-values are obtained through regression surface approximation from
    MacKinnon 1994, but using the updated 2010 tables.
    If the p-value is close to significant, then the critical values should be
    used to judge whether to accept or reject the null.

    The autolag option and maxlag for it are described in Greene.

    Examples
    --------
    see example script

    References
    ----------
    Greene
    Hamilton


    P-Values (regression surface approximation)
    MacKinnon, J.G. 1994.  "Approximate asymptotic distribution functions for
    unit-root and cointegration tests.  `Journal of Business and Economic
    Statistics` 12, 167-76.

    Critical values
    MacKinnon, J.G. 2010. "Critical Values for Cointegration Tests."  Queen's
    University, Dept of Economics, Working Papers.  Available at
    http://ideas.repec.org/p/qed/wpaper/1227.html

    '''

    if regresults:
        store = True

    trenddict = {None:'nc', 0:'c', 1:'ct', 2:'ctt'}
    if regression is None or isinstance(regression, int):
        regression = trenddict[regression]
    regression = regression.lower()
    if regression not in ['c','nc','ct','ctt']:
        raise ValueError("regression option %s not understood") % regression
    x = np.asarray(x)
    nobs = x.shape[0]

    if maxlag is None:
        #from Greene referencing Schwert 1989
        maxlag = int(np.ceil(12. * np.power(nobs/100., 1/4.)))

    xdiff = np.diff(x)
    xdall = lagmat(xdiff[:,None], maxlag, trim='both', original='in')
    nobs = xdall.shape[0]

    xdall[:,0] = x[-nobs-1:-1] # replace 0 xdiff with level of x
    xdshort = xdiff[-nobs:]

    if store:
        resstore = ResultsStore()
    if autolag:
        if regression != 'nc':
            fullRHS = add_trend(xdall, regression, prepend=True)
        else:
            fullRHS = xdall
        startlag = fullRHS.shape[1] - xdall.shape[1] + 1 # 1 for level
        #search for lag length with smallest information criteria
        #Note: use the same number of observations to have comparable IC
        #aic and bic: smaller is better

        if not regresults:
            icbest, bestlag = _autolag(OLS, xdshort, fullRHS, startlag,
                                       maxlag, autolag)
        else:
            icbest, bestlag, alres = _autolag(OLS, xdshort, fullRHS, startlag,
                                        maxlag, autolag, regresults=regresults)
            resstore.autolag_results = alres

        bestlag -= startlag  #convert to lag not column index

        #rerun ols with best autolag
        xdall = lagmat(xdiff[:,None], bestlag, trim='both', original='in')
        nobs = xdall.shape[0]
        xdall[:,0] = x[-nobs-1:-1] # replace 0 xdiff with level of x
        xdshort = xdiff[-nobs:]
        usedlag = bestlag
    else:
        usedlag = maxlag
        icbest = None
    if regression != 'nc':
        resols = OLS(xdshort, add_trend(xdall[:,:usedlag+1], regression)).fit()
    else:
        resols = OLS(xdshort, xdall[:,:usedlag+1]).fit()

    adfstat = resols.tvalues[0]
#    adfstat = (resols.params[0]-1.0)/resols.bse[0]
    # the "asymptotically correct" z statistic is obtained as
    # nobs/(1-np.sum(resols.params[1:-(trendorder+1)])) (resols.params[0] - 1)
    # I think this is the statistic that is used for series that are integrated
    # for orders higher than I(1), ie., not ADF but cointegration tests.

    # Get approx p-value and critical values
    pvalue = mackinnonp(adfstat, regression=regression, N=1)
    critvalues = mackinnoncrit(N=1, regression=regression, nobs=nobs)
    critvalues = {"1%" : critvalues[0], "5%" : critvalues[1],
            "10%" : critvalues[2]}
    if store:
        resstore.resols = resols
        resstore.maxlag = maxlag
        resstore.usedlag = usedlag
        resstore.adfstat = adfstat
        resstore.critvalues = critvalues
        resstore.nobs = nobs
        resstore.H0 = "The coefficient on the lagged level equals 1 - unit root"
        resstore.HA = "The coefficient on the lagged level < 1 - stationary"
        resstore.icbest = icbest
        return adfstat, pvalue, critvalues, resstore
    else:
        if not autolag:
            return adfstat, pvalue, usedlag, nobs, critvalues
        else:
            return adfstat, pvalue, usedlag, nobs, critvalues, icbest

def acovf(x, unbiased=False, demean=True, fft=False):
    '''
    Autocovariance for 1D

    Parameters
    ----------
    x : array
       time series data
    unbiased : bool
       if True, then denominators is n-k, otherwise n
    fft : bool
        If True, use FFT convolution.  This method should be preferred
        for long time series.

    Returns
    -------
    acovf : array
        autocovariance function
    '''
    n = len(x)
    if demean:
        xo = x - x.mean();
    else:
        xo = x
    if unbiased:
#        xi = np.ones(n);
#        d = np.correlate(xi, xi, 'full')
        xi = np.arange(1,n+1)
        d = np.hstack((xi,xi[:-1][::-1])) # faster, is correlate more general?
    else:
        d = n
    if fft:
        nobs = len(xo)
        Frf = np.fft.fft(xo, n=nobs*2)
        acov = np.fft.ifft(Frf*np.conjugate(Frf))[:nobs]/d
        return acov.real
    else:
        return (np.correlate(xo, xo, 'full')/d)[n-1:]

def q_stat(x,nobs, type="ljungbox"):
    """
    Return's Ljung-Box Q Statistic

    x : array-like
        Array of autocorrelation coefficients.  Can be obtained from acf.
    nobs : int
        Number of observations in the entire sample (ie., not just the length
        of the autocorrelation function results.

    Returns
    -------
    q-stat : array
        Ljung-Box Q-statistic for autocorrelation parameters
    p-value : array
        P-value of the Q statistic

    Notes
    ------
    Written to be used with acf.
    """
    x = np.asarray(x)
    if type=="ljungbox":
        ret = nobs*(nobs+2)*np.cumsum((1./(nobs-np.arange(1,
            len(x)+1)))*x**2)
    chi2 = stats.chi2.sf(ret,np.arange(1,len(x)+1))
    return ret,chi2

#NOTE: Changed unbiased to False
#see for example
# http://www.itl.nist.gov/div898/handbook/eda/section3/autocopl.htm
def acf(x, unbiased=False, nlags=40, confint=None, qstat=False, fft=False,
        alpha=None):
    '''
    Autocorrelation function for 1d arrays.

    Parameters
    ----------
    x : array
       Time series data
    unbiased : bool
       If True, then denominators for autocovariance are n-k, otherwise n
    nlags: int, optional
        Number of lags to return autocorrelation for.
    confint : scalar, optional
        The use of confint is deprecated. See `alpha`.
        If a number is given, the confidence intervals for the given level are
        returned. For instance if confint=95, 95 % confidence intervals are
        returned where the standard deviation is computed according to
        Bartlett\'s formula.
    qstat : bool, optional
        If True, returns the Ljung-Box q statistic for each autocorrelation
        coefficient.  See q_stat for more information.
    fft : bool, optional
        If True, computes the ACF via FFT.
    alpha : scalar, optional
        If a number is given, the confidence intervals for the given level are
        returned. For instance if alpha=.05, 95 % confidence intervals are
        returned where the standard deviation is computed according to
        Bartlett\'s formula.

    Returns
    -------
    acf : array
        autocorrelation function
    confint : array, optional
        Confidence intervals for the ACF. Returned if confint is not None.
    qstat : array, optional
        The Ljung-Box Q-Statistic.  Returned if q_stat is True.
    pvalues : array, optional
        The p-values associated with the Q-statistics.  Returned if q_stat is
        True.

    Notes
    -----
    The acf at lag 0 (ie., 1) is returned.

    This is based np.correlate which does full convolution. For very long time
    series it is recommended to use fft convolution instead.

    If unbiased is true, the denominator for the autocovariance is adjusted
    but the autocorrelation is not an unbiased estimtor.
    '''
    nobs = len(x)
    d = nobs # changes if unbiased
    if not fft:
        avf = acovf(x, unbiased=unbiased, demean=True)
        #acf = np.take(avf/avf[0], range(1,nlags+1))
        acf = avf[:nlags+1]/avf[0]
    else:
        #JP: move to acovf
        x0 = x - x.mean()
        Frf = np.fft.fft(x0, n=nobs*2) # zero-pad for separability
        if unbiased:
            d = nobs - np.arange(nobs)
        acf = np.fft.ifft(Frf * np.conjugate(Frf))[:nobs]/d
        acf /= acf[0]
        #acf = np.take(np.real(acf), range(1,nlags+1))
        acf = np.real(acf[:nlags+1])   #keep lag 0
    if not (confint or qstat or alpha):
        return acf
    if not confint is None:
        import warnings
        warnings.warn("confint is deprecated. Please use the alpha keyword",
                      FutureWarning)
        varacf = np.ones(nlags+1)/nobs
        varacf[0] = 0
        varacf[1] = 1./nobs
        varacf[2:] *= 1 + 2*np.cumsum(acf[1:-1]**2)
        interval = stats.norm.ppf(1-(100-confint)/200.)*np.sqrt(varacf)
        confint = np.array(zip(acf-interval, acf+interval))
        if not qstat:
            return acf, confint
    if alpha is not None:
        varacf = np.ones(nlags+1)/nobs
        varacf[0] = 0
        varacf[1] = 1./nobs
        varacf[2:] *= 1 + 2*np.cumsum(acf[1:-1]**2)
        interval = stats.norm.ppf(1-alpha/2.)*np.sqrt(varacf)
        confint = np.array(zip(acf-interval, acf+interval))
        if not qstat:
            return acf, confint
    if qstat:
        qstat, pvalue = q_stat(acf[1:], nobs=nobs)  #drop lag 0
        if (confint is not None or alpha is not None):
            return acf, confint, qstat, pvalue
        else:
            return acf, qstat, pvalue

def pacf_yw(x, nlags=40, method='unbiased'):
    '''Partial autocorrelation estimated with non-recursive yule_walker

    Parameters
    ----------
    x : 1d array
        observations of time series for which pacf is calculated
    nlags : int
        largest lag for which pacf is returned
    method : 'unbiased' (default) or 'mle'
        method for the autocovariance calculations in yule walker

    Returns
    -------
    pacf : 1d array
        partial autocorrelations, maxlag+1 elements

    Notes
    -----
    This solves yule_walker for each desired lag and contains
    currently duplicate calculations.
    '''
    xm = x - x.mean()
    pacf = [1.]
    for k in range(1, nlags+1):
        pacf.append(yule_walker(x, k, method=method)[0][-1])
    return np.array(pacf)

#NOTE: this is incorrect.
def pacf_ols(x, nlags=40):
    '''Calculate partial autocorrelations

    Parameters
    ----------
    x : 1d array
        observations of time series for which pacf is calculated
    nlags : int
        Number of lags for which pacf is returned.  Lag 0 is not returned.

    Returns
    -------
    pacf : 1d array
        partial autocorrelations, maxlag+1 elements

    Notes
    -----
    This solves a separate OLS estimation for each desired lag.
    '''
    #TODO: add warnings for Yule-Walker
    #NOTE: demeaning and not using a constant gave incorrect answers?
    #JP: demeaning should have a better estimate of the constant
    #maybe we can compare small sample properties with a MonteCarlo
    xlags, x0 = lagmat(x, nlags, original='sep')
    #xlags = sm.add_constant(lagmat(x, nlags), prepend=True)
    xlags = add_constant(xlags, prepend=True)
    pacf = [1.]
    for k in range(1, nlags+1):
        res = OLS(x0[k:], xlags[k:,:k+1]).fit()
         #np.take(xlags[k:], range(1,k+1)+[-1],

        pacf.append(res.params[-1])
    return np.array(pacf)

def pacf(x, nlags=40, method='ywunbiased', alpha=None):
    '''Partial autocorrelation estimated

    Parameters
    ----------
    x : 1d array
        observations of time series for which pacf is calculated
    nlags : int
        largest lag for which pacf is returned
    method : 'ywunbiased' (default) or 'ywmle' or 'ols'
        specifies which method for the calculations to use:

        - yw or ywunbiased : yule walker with bias correction in denominator
          for acovf
        - ywm or ywmle : yule walker without bias correction
        - ols - regression of time series on lags of it and on constant
        - ld or ldunbiased : Levinson-Durbin recursion with bias correction
        - ldb or ldbiased : Levinson-Durbin recursion without bias correction

    alpha : scalar, optional
        If a number is given, the confidence intervals for the given level are
        returned. For instance if alpha=.05, 95 % confidence intervals are
        returned where the standard deviation is computed according to
        1/sqrt(len(x))

    Returns
    -------
    pacf : 1d array
        partial autocorrelations, nlags elements, including lag zero
    confint : array, optional
        Confidence intervals for the PACF. Returned if confint is not None.

    Notes
    -----
    This solves yule_walker equations or ols for each desired lag
    and contains currently duplicate calculations.
    '''

    if method == 'ols':
        ret = pacf_ols(x, nlags=nlags)
    elif method in ['yw', 'ywu', 'ywunbiased', 'yw_unbiased']:
        ret = pacf_yw(x, nlags=nlags, method='unbiased')
    elif method in ['ywm', 'ywmle', 'yw_mle']:
        ret = pacf_yw(x, nlags=nlags, method='mle')
    elif method in ['ld', 'ldu', 'ldunbiase', 'ld_unbiased']:
        acv = acovf(x, unbiased=True)
        ld_ = levinson_durbin(acv, nlags=nlags, isacov=True)
        #print 'ld', ld_
        ret = ld_[2]
    elif method in ['ldb', 'ldbiased', 'ld_biased']: #inconsistent naming with ywmle
        acv = acovf(x, unbiased=False)
        ld_ = levinson_durbin(acv, nlags=nlags, isacov=True)
        ret = ld_[2]
    else:
        raise ValueError('method not available')
    if alpha is not None:
        varacf = 1./len(x)
        interval = stats.norm.ppf(1. - alpha/2.) * np.sqrt(varacf)
        confint = np.array(zip(ret-interval, ret+interval))
        return ret, confint
    else:
        return ret



def ccovf(x, y, unbiased=True, demean=True):
    ''' crosscovariance for 1D

    Parameters
    ----------
    x, y : arrays
       time series data
    unbiased : boolean
       if True, then denominators is n-k, otherwise n

    Returns
    -------
    ccovf : array
        autocovariance function

    Notes
    -----
    This uses np.correlate which does full convolution. For very long time
    series it is recommended to use fft convolution instead.
    '''
    n = len(x)
    if demean:
        xo = x - x.mean();
        yo = y - y.mean();
    else:
        xo = x
        yo = y
    if unbiased:
        xi = np.ones(n);
        d = np.correlate(xi, xi, 'full')
    else:
        d = n
    return (np.correlate(xo,yo,'full') / d)[n-1:]

def ccf(x, y, unbiased=True):
    '''cross-correlation function for 1d

    Parameters
    ----------
    x, y : arrays
       time series data
    unbiased : boolean
       if True, then denominators for autocovariance is n-k, otherwise n

    Returns
    -------
    ccf : array
        cross-correlation function of x and y

    Notes
    -----
    This is based np.correlate which does full convolution. For very long time
    series it is recommended to use fft convolution instead.

    If unbiased is true, the denominator for the autocovariance is adjusted
    but the autocorrelation is not an unbiased estimtor.

    '''
    cvf = ccovf(x, y, unbiased=unbiased, demean=True)
    return cvf / (np.std(x) * np.std(y))


def periodogram(X):
    """
    Returns the periodogram for the natural frequency of X

    Parameters
    ----------
    X : array-like
        Array for which the periodogram is desired.

    Returns
    -------
    pgram : array
        1./len(X) * np.abs(np.fft.fft(X))**2


    References
    ----------
    Brockwell and Davis.
    """
    X = np.asarray(X)
#    if kernel == "bartlett":
#        w = 1 - np.arange(M+1.)/M   #JP removed integer division

    pergr = 1./len(X) * np.abs(np.fft.fft(X))**2
    pergr[0] = 0. # what are the implications of this?
    return pergr

#copied from nitime and scikits\statsmodels\sandbox\tsa\examples\try_ld_nitime.py
#TODO: check what to return, for testing and trying out returns everything
def levinson_durbin(s, nlags=10, isacov=False):
    '''Levinson-Durbin recursion for autoregressive processes

    Parameters
    ----------
    s : array_like
        If isacov is False, then this is the time series. If iasacov is true
        then this is interpreted as autocovariance starting with lag 0
    nlags : integer
        largest lag to include in recursion or order of the autoregressive
        process
    isacov : boolean
        flag to indicate whether the first argument, s, contains the autocovariances
        or the data series.

    Returns
    -------
    sigma_v : float
        estimate of the error variance ?
    arcoefs : ndarray
        estimate of the autoregressive coefficients
    pacf : ndarray
        partial autocorrelation function
    sigma : ndarray
        entire sigma array from intermediate result, last value is sigma_v
    phi : ndarray
        entire phi array from intermediate result, last column contains
        autoregressive coefficients for AR(nlags) with a leading 1

    Notes
    -----
    This function returns currently all results, but maybe we drop sigma and
    phi from the returns.

    If this function is called with the time series (isacov=False), then the sample
    autocovariance function is calculated with the default options (biased, no fft).

    '''
    s = np.asarray(s)
    order = nlags  #rename compared to nitime
    #from nitime

##    if sxx is not None and type(sxx) == np.ndarray:
##        sxx_m = sxx[:order+1]
##    else:
##        sxx_m = ut.autocov(s)[:order+1]
    if isacov:
        sxx_m = s
    else:
        sxx_m = acovf(s)[:order+1]  #not tested

    phi = np.zeros((order+1, order+1), 'd')
    sig = np.zeros(order+1)
    # initial points for the recursion
    phi[1,1] = sxx_m[1]/sxx_m[0]
    sig[1] = sxx_m[0] - phi[1,1]*sxx_m[1]
    for k in xrange(2,order+1):
        phi[k,k] = (sxx_m[k] - np.dot(phi[1:k,k-1], sxx_m[1:k][::-1]))/sig[k-1]
        for j in xrange(1,k):
            phi[j,k] = phi[j,k-1] - phi[k,k]*phi[k-j,k-1]
        sig[k] = sig[k-1]*(1 - phi[k,k]**2)

    sigma_v = sig[-1]
    arcoefs = phi[1:,-1]
    pacf_ = np.diag(phi).copy()
    pacf_[0] = 1.
    return sigma_v, arcoefs, pacf_, sig, phi  #return everything



def grangercausalitytests(x, maxlag, addconst=True, verbose=True):
    '''four tests for granger non causality of 2 timeseries

    all four tests give similar results
    `params_ftest` and `ssr_ftest` are equivalent based on F test which is
    identical to lmtest:grangertest in R

    Parameters
    ----------
    x : array, 2d, (nobs,2)
        data for test whether the time series in the second column Granger
        causes the time series in the first column
    maxlag : integer
        the Granger causality test results are calculated for all lags up to
        maxlag
    verbose : bool
        print results if true

    Returns
    -------
    results : dictionary
        all test results, dictionary keys are the number of lags. For each
        lag the values are a tuple, with the first element a dictionary with
        teststatistic, pvalues, degrees of freedom, the second element are
        the OLS estimation results for the restricted model, the unrestricted
        model and the restriction (contrast) matrix for the parameter f_test.

    Notes
    -----
    TODO: convert to class and attach results properly

    The Null hypothesis for grangercausalitytests is that the time series in
    the second column, x2, does NOT Granger cause the time series in the first
    column, x1. Grange causality means that past values of x2 have a
    statistically significant effect on the current value of x1, taking past
    values of x1 into account as regressors. We reject the null hypothesis
    that x2 does not Granger cause x1 if the pvalues are below a desired size
    of the test.

    The null hypothesis for all four test is that the coefficients
    corresponding to past values of the second time series are zero.

    'params_ftest', 'ssr_ftest' are based on F distribution

    'ssr_chi2test', 'lrtest' are based on chi-square distribution

    References
    ----------
    http://en.wikipedia.org/wiki/Granger_causality
    Greene: Econometric Analysis

    '''
    from scipy import stats # lazy import

    resli = {}

    for mlg in range(1, maxlag+1):
        result = {}
        if verbose:
            print '\nGranger Causality'
            print 'number of lags (no zero)', mlg
        mxlg = mlg #+ 1 # Note number of lags starting at zero in lagmat

        # create lagmat of both time series
        dta = lagmat2ds(x, mxlg, trim='both', dropex=1)

        #add constant
        if addconst:
            dtaown = add_constant(dta[:,1:mxlg+1])
            dtajoint = add_constant(dta[:,1:])
        else:
            raise ValueError('Not Implemented')
            dtaown = dta[:,1:mxlg]
            dtajoint = dta[:,1:]

        #run ols on both models without and with lags of second variable
        res2down = OLS(dta[:,0], dtaown).fit()
        res2djoint = OLS(dta[:,0], dtajoint).fit()

        #print results
        #for ssr based tests see: http://support.sas.com/rnd/app/examples/ets/granger/index.htm
        #the other tests are made-up

        # Granger Causality test using ssr (F statistic)
        fgc1 = (res2down.ssr-res2djoint.ssr)/res2djoint.ssr/(mxlg)*res2djoint.df_resid
        if verbose:
            print 'ssr based F test:         F=%-8.4f, p=%-8.4f, df_denom=%d, df_num=%d' % \
              (fgc1, stats.f.sf(fgc1, mxlg, res2djoint.df_resid), res2djoint.df_resid, mxlg)
        result['ssr_ftest'] = (fgc1, stats.f.sf(fgc1, mxlg, res2djoint.df_resid), res2djoint.df_resid, mxlg)

        # Granger Causality test using ssr (ch2 statistic)
        fgc2 = res2down.nobs*(res2down.ssr-res2djoint.ssr)/res2djoint.ssr
        if verbose:
            print 'ssr based chi2 test:   chi2=%-8.4f, p=%-8.4f, df=%d' %  \
              (fgc2, stats.chi2.sf(fgc2, mxlg), mxlg)
        result['ssr_chi2test'] = (fgc2, stats.chi2.sf(fgc2, mxlg), mxlg)

        #likelihood ratio test pvalue:
        lr = -2*(res2down.llf-res2djoint.llf)
        if verbose:
            print 'likelihood ratio test: chi2=%-8.4f, p=%-8.4f, df=%d' %  \
              (lr, stats.chi2.sf(lr, mxlg), mxlg)
        result['lrtest'] = (lr, stats.chi2.sf(lr, mxlg), mxlg)

        # F test that all lag coefficients of exog are zero
        rconstr = np.column_stack((np.zeros((mxlg-1,mxlg-1)), np.eye(mxlg-1, mxlg-1),\
                                   np.zeros((mxlg-1, 1))))
        rconstr = np.column_stack((np.zeros((mxlg,mxlg)), np.eye(mxlg, mxlg),\
                                   np.zeros((mxlg, 1))))
        ftres = res2djoint.f_test(rconstr)
        if verbose:
            print 'parameter F test:         F=%-8.4f, p=%-8.4f, df_denom=%d, df_num=%d' % \
              (ftres.fvalue, ftres.pvalue, ftres.df_denom, ftres.df_num)
        result['params_ftest'] = (np.squeeze(ftres.fvalue)[()],
                                  np.squeeze(ftres.pvalue)[()],
                                  ftres.df_denom, ftres.df_num)

        resli[mxlg] = (result, [res2down, res2djoint, rconstr])

    return resli

def coint(y1, y2, regression="c"):
    """
    This is a simple cointegration test. Uses unit-root test on residuals to
    test for cointegrated relationship

    See Hamilton (1994) 19.2

    Parameters
    ----------
    y1 : array_like, 1d
        first element in cointegrating vector
    y2 : array_like
        remaining elements in cointegrating vector
    c : str {'c'}
        Included in regression
        * 'c' : Constant

    Returns
    -------
    coint_t : float
        t-statistic of unit-root test on residuals
    pvalue : float
        MacKinnon's approximate p-value based on MacKinnon (1994)
    crit_value : dict
        Critical values for the test statistic at the 1 %, 5 %, and 10 % levels.

    Notes
    -----
    The Null hypothesis is that there is no cointegration, the alternative
    hypothesis is that there is cointegrating relationship. If the pvalue is
    small, below a critical size, then we can reject the hypothesis that there
    is no cointegrating relationship.

    P-values are obtained through regression surface approximation from
    MacKinnon 1994.

    References
    ----------
    MacKinnon, J.G. 1994.  "Approximate asymptotic distribution functions for
        unit-root and cointegration tests.  `Journal of Business and Economic
        Statistics` 12, 167-76.

    """
    regression = regression.lower()
    if regression not in ['c','nc','ct','ctt']:
        raise ValueError("regression option %s not understood") % regression
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    if regression == 'c':
        y2 = add_constant(y2)
    st1_resid = OLS(y1, y2).fit().resid #stage one residuals
    lgresid_cons = add_constant(st1_resid[0:-1])
    uroot_reg = OLS(st1_resid[1:], lgresid_cons).fit()
    coint_t = (uroot_reg.params[0]-1)/uroot_reg.bse[0]
    pvalue = mackinnonp(coint_t, regression="c", N=2, lags=None)
    crit_value = mackinnoncrit(N=1, regression="c", nobs=len(y1))
    return coint_t, pvalue, crit_value

__all__ = ['acovf', 'acf', 'pacf', 'pacf_yw', 'pacf_ols', 'ccovf', 'ccf',
           'periodogram', 'q_stat', 'coint']

if __name__=="__main__":
    import statsmodels.api as sm
    data = sm.datasets.macrodata.load().data
    x = data['realgdp']
# adf is tested now.
    adf = adfuller(x,4, autolag=None)
    adfbic = adfuller(x, autolag="bic")
    adfaic = adfuller(x, autolag="aic")
    adftstat = adfuller(x, autolag="t-stat")

# acf is tested now
    acf1,ci1,Q,pvalue = acf(x, nlags=40, confint=95, qstat=True)
    acf2, ci2,Q2,pvalue2 = acf(x, nlags=40, confint=95, fft=True, qstat=True)
    acf3,ci3,Q3,pvalue3 = acf(x, nlags=40, confint=95, qstat=True, unbiased=True)
    acf4, ci4,Q4,pvalue4 = acf(x, nlags=40, confint=95, fft=True, qstat=True,
            unbiased=True)

# pacf is tested now
#    pacf1 = pacorr(x)
#    pacfols = pacf_ols(x, nlags=40)
#    pacfyw = pacf_yw(x, nlags=40, method="mle")
    y = np.random.normal(size=(100,2))
    grangercausalitytests(y,2)

