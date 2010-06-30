"""
Statistical tools for time series analysis
"""

import numpy as np
from scipy import stats, signal
import scikits.statsmodels as sm
from scikits.statsmodels.sandbox.tsa.tsatools import lagmat, lagmat2ds
from adfvalues import *
#from scikits.statsmodels.sandbox.rls import RLS

#NOTE: now in two places to avoid circular import
#TODO: I like the bunch pattern for this too.
class ResultsStore(object):
    def __str__(self):
        return self._str

def add_trend(X, trend="c", prepend=False):
    """
    Adds a trend and/or constant to an array.

    Parameters
    ----------
    X : array-like
        Original array of data.
    trend : str {"c","ct","ctt"}
        "c" add constnat
        "ct" add constant and linear trend
        "ctt" add constant and linear and quadratic trend.
    prepend : bool
        If True, prepends the new data to the columns of X.

    Notes
    -----
    Returns columns as ["ctt","ct","c"] whenever applicable.  There is currently
    no checking for an existing constant or trend.

    See also
    --------
    scikits.statsmodels.add_constant
    """
    #TODO: could be generalized for trend of aribitrary order
    trend = trend.lower()
    if trend == "c":    # handles structured arrays
        return sm.add_constant(X, prepend=prepend)
    elif trend == "ct":
        trendorder = 1
    elif trend == "ctt":
        trendorder = 2
    else:
        raise ValueError("trend %s not understood") % trend
    X = np.asanyarray(X)
    nobs = len(X)
    trendarr = np.vander(np.arange(1,nobs+1, dtype=float), trendorder+1)
    if not X.dtype.names:
        if not prepend:
            X = np.column_stack((X, trendarr))
        else:
            X = np.column_stack((trendarr, X))
    else:
        return_rec = data.__clas__ is np.recarray
        if trendorder == 1:
            dt = [('trend',float),('const',float)]
        elif trendorder == 2:
            dt = [('trend_squared', float),('trend',float),('const',float)]
        trendarr = trendarr.view(dt)
        if prepend:
            X = nprf.append_fields(trendarr, X.dtype.names, [X[i] for i
                in data.dtype.names], usemask=False, asrecarray=return_rec)
        else:
            X = nprf.append_fields(X, trendarr.dtype.names, [trendarr[i] for i
                in trendarr.dtype.names], usemask=false, asrecarray=return_rec)
    return X


def _autolag(mod, endog, exog, modargs=(), fitargs=(), lagstart=1,
        maxlag=None, method=None):
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
    method : str {"aic","bic","t-stat","hic"}
        aic - Akaike Information Criterion
        bic - Bayes Information Criterion
        t-stat - Based on last lag
        hq - Hannan-Quinn

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
    for lag in range(int(lagstart),int(maxlag+1)):
        results[lag] = mod(endog, exog[:,:lag]).fit(*fitargs)
    if method == "aic":
        icbest, bestlag = max((v.aic,k) for k,v in results.iteritems())
    elif method == "bic":
        icbest, bestlag = max((v.bic,k) for k,v in results.iteritems())
    elif method == "t-stat":
        lags = sorted(results.keys())[::-1]
#        stop = stats.norm.ppf(.95)
        stop = 1.6448536269514722
        i = 0
        icbest, bestlag = results[lags[i]].t(-1), lags[i]
        i += 1
        while not (abs(icbest) >= stop):
            lastt, bestlag = results[lags[i]].t(-1), lags[i]
            i += 1
    elif method == "hq":
        icbest, bestlag = max((v.hqic,k) for k,v in results.iteritems())
    else:
        raise ValueError("Information Criterion %s not understood.") % method
    return icbest, bestlag

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
    If the p-value is close to significant, then the critical values should be
    used to judge whether to accept or reject the null.

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
    MacKinnon, J.G. 2010. "Critical Values for Cointegration Tests."
        Queen's University, Dept of Economics, Working Papers.  Available at
        http://ideas.repec.org/p/qed/wpaper/1227.html
    '''
    regression = regression.lower()
    if regression not in ['c','nc','ct','ctt']:
        raise ValueError("regression option %s not understood") % regression
    x = np.asarray(x)
    nobs = x.shape[0]

    if regression == 'c':
        trendorder = 0
    elif regression == 'nc':
        trendorder = -1
    elif regression == 'ct':
        trendorder = 1
    elif regression == 'ctt':
        trendorder = 2
    # only make the trend once with biggest nobs
    trend = np.vander(np.arange(nobs), trendorder+1)

    if maxlag is None:
        #from Greene referencing Schwert 1989
        maxlag = 12. * np.power(nobs/100., 1/4.)

    xdiff = np.diff(x)
    xdall = lagmat(xdiff[:,None], maxlag, trim='both')
    nobs = xdall.shape[0]

    xdall[:,0] = x[-nobs-1:-1] # replace 0 xdiff with level of x
    xdshort = xdiff[-nobs:]
#    xdshort = x[-nobs:]
#TODO: allow for 2nd xdshort as endog, with Phillips Perron or DF test?

    if store:
        resstore = ResultsStore()
    if autolag:
        if trendorder is not -1:
            fullRHS = np.column_stack((trend[:nobs],xdall))
        else:
            fullRHS = xdall
        lagstart = trendorder + 1
        #search for lag length with highest information criteria
        #Note: I use the same number of observations to have comparable IC
        icbest, bestlag = _autolag(sm.OLS, xdshort, fullRHS, lagstart=lagstart,
                maxlag=maxlag, method=autolag)

        #rerun ols with best autolag
        xdall = lagmat(xdiff[:,None], bestlag, trim='both')
        nobs = xdall.shape[0]
#        trend = np.vander(np.arange(nobs), trendorder+1)
        xdall[:,0] = x[-nobs-1:-1] # replace 0 xdiff with level of x
        xdshort = xdiff[-nobs:]
        usedlag = bestlag
    else:
        usedlag = maxlag

    resols = sm.OLS(xdshort, np.column_stack([xdall[:,:usedlag+1],
        trend[:nobs]])).fit()
    #NOTE: should be usedlag+1 since the first column is the level?
    adfstat = resols.t(0)
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
        resstore.usedlag = usedlag
        resstore.adfstat = adfstat
        resstore.critvalues = critvalues
        resstore.nobs = nobs
        resstore.H0 = "The coefficient on the lagged level equals 1"
        resstore.HA = "The coefficient on the lagged level < 1"
        resstore.icbest = icbest
        return adfstat, pvalue, critvalues, resstore
    else:
        if not autolag:
            return adfstat, pvalue, usedlag, nobs, critvalues
        else:
            return adfstat, pvalue, usedlag, nobs, critvalues, icbest

def acovf(x, unbiased=False, demean=True):
    '''
    Autocovariance for 1D

    Parameters
    ----------
    x : array
       time series data
    unbiased : boolean
       if True, then denominators is n-k, otherwise n

    Returns
    -------
    acovf : array
        autocovariance function

    Notes
    -----
    This uses np.correlate which does full convolution. For very long time
    series it is recommended to use fft convolution instead.
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
def acf(x, unbiased=False, nlags=40, confint=None, qstat=False, fft=False):
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
    confint : float or None, optional
        If True, the confidence intervals for the given level are returned.
        For instance if confint=95, 95 % confidence intervals are returned.
    qstat : bool, optional
        If True, returns the Ljung-Box q statistic for each autocorrelation
        coefficient.  See q_stat for more information.
    fft : bool, optional
        If True, computes the ACF via FFT.

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
    The acf at lag 0 (ie., 1) is *not* returned.

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
        acf = avf/avf[0]
    else:
        #JP: move to acovf
        x0 = x - x.mean()
        Frf = np.fft.fft(x0, n=nobs*2) # zero-pad for separability
        if unbiased:
            d = nobs - np.arange(nobs)
        acf = np.fft.ifft(Frf * np.conjugate(Frf))[:nobs]/d
        acf /= acf[0]
        acf = np.take(np.real(acf), range(1,nlags+1))
    if not (confint or qstat):
        return acf
# Based on Bartlett's formula for MA(q) processes
#NOTE: not sure if this is correct, or needs to be centered or what.
    if confint:
        varacf = np.ones(nlags)/nobs
        varacf[1:] *= 1 + 2*np.cumsum(acf[:-1]**2)
        interval = stats.norm.ppf(1-(100-confint)/200.)*np.sqrt(varacf)
        confint = np.array(zip(acf-interval, acf+interval))
        if not qstat:
            return acf, confint
    if qstat:
        qstat, pvalue = q_stat(acf, nobs=nobs)
        if confint is not None:
            return acf, confint, qstat, pvalue
        else:
            return acf, qstat

def pacf_yw(x, nlags=40, method='unbiased'):
    '''Partial autocorrelation estimated with non-recursive yule_walker

    Parameters
    ----------
    x : 1d array
        observations of time series for which pacf is calculated
    maxlag : int
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
        pacf.append(sm.regression.yule_walker(x, k, method=method)[0][-1])
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
    xlags = lagmat(x, nlags)
    x0 = xlags[:,0]
    xlags = xlags[:,1:]
    xlags = sm.add_constant(lagmat(x, nlags), prepend=True)
    pacf = [1.]
    for k in range(1, nlags+1):
        res = sm.OLS(x0[k:], xlags[k:,:k+1]).fit()
         #np.take(xlags[k:], range(1,k+1)+[-1],

        pacf.append(res.params[-1])
    return np.array(pacf)

def pacf(x, nlags=40, method='unbiased'):

    if method == 'ols':
        return pacf_ols(x, nlags=nlags)
    elif method in ['yw', 'ywu', 'ywunbiased', 'yw_unbiased']:
        return pacf_yw(x, nlags=nlags, method='unbiased')
    elif method in ['ywm', 'ywmle', 'yw_mle']:
        return pacf_yw(x, nlags=nlags, method='mle')
    else:
        raise ValueError('method not available')



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
    #JP: this should use covf and fft for speed and accuracy for longer time series
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

def grangercausalitytests(x, maxlag):
    '''four tests for granger causality of 2 timeseries

    this is a proof-of concept implementation
    not cleaned up, has some duplicate calculations,
    memory intensive - builds full lag array for variables
    prints results
    not verified with other packages,
    all four tests give similar results (1 and 4 identical)

    Parameters
    ----------
    x : array, 2d, (nobs,2)
        data for test whether the time series in the second column Granger
        causes the time series in the first column
    maxlag : integer
        the Granger causality test results are calculated for all lags up to
        maxlag

    Returns
    -------
    None : no returns
        all test results are currently printed

    Notes
    -----
    TODO: convert to function that returns and compare with other packages

    '''
    from scipy import stats # lazy import
    import scikits.statsmodels as sm  # absolute import for now

    for mlg in range(1, maxlag+1):
        print '\nGranger Causality'
        print 'number of lags (no zero)', mlg
        mxlg = mlg + 1 # Note number of lags starting at zero in lagmat

        # create lagmat of both time series
        dta = lagmat2ds(x, mxlg, trim='both', dropex=1)

        #add constant
        dtaown = sm.add_constant(dta[:,1:mxlg])
        dtajoint = sm.add_constant(dta[:,1:])

        #run ols on both models without and with lags of second variable
        res2down = sm.OLS(dta[:,0], dtaown).fit()
        res2djoint = sm.OLS(dta[:,0], dtajoint).fit()

        #print results
        #for ssr based tests see: http://support.sas.com/rnd/app/examples/ets/granger/index.htm
        #the other tests are made-up

        # Granger Causality test using ssr (F statistic)
        fgc1 = (res2down.ssr-res2djoint.ssr)/res2djoint.ssr/(mxlg-1)*res2djoint.df_resid
        print 'ssr based F test:         F=%-8.4f, p=%-8.4f, df_denom=%d, df_num=%d' % \
              (fgc1, stats.f.sf(fgc1, mxlg-1, res2djoint.df_resid), res2djoint.df_resid, mxlg-1)

        # Granger Causality test using ssr (ch2 statistic)
        fgc2 = res2down.nobs*(res2down.ssr-res2djoint.ssr)/res2djoint.ssr
        print 'ssr based chi2 test:   chi2=%-8.4f, p=%-8.4f, df=%d' %  \
              (fgc2, stats.chi2.sf(fgc2, mxlg-1), mxlg-1)

        #likelihood ratio test pvalue:
        lr = -2*(res2down.llf-res2djoint.llf)
        print 'likelihood ratio test: chi2=%-8.4f, p=%-8.4f, df=%d' %  \
              (lr, stats.chi2.sf(lr, mxlg-1), mxlg-1)

        # F test that all lag coefficients of exog are zero
        rconstr = np.column_stack((np.zeros((mxlg-1,mxlg-1)), np.eye(mxlg-1, mxlg-1),\
                                   np.zeros((mxlg-1, 1))))
        ftres = res2djoint.f_test(rconstr)
        print 'parameter F test:         F=%-8.4f, p=%-8.4f, df_denom=%d, df_num=%d' % \
              (ftres.fvalue, ftres.pvalue, ftres.df_denom, ftres.df_num)


__all__ = ['acovf', 'acf', 'acf', 'pacf_yw', 'pacf_ols', 'pergram', 'q_stat']

if __name__=="__main__":
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

