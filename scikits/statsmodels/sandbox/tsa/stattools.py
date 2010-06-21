"""
Statistical tools for time series analysis
"""

import numpy as np
from scipy import stats, signal
import scikits.statsmodels as sm
from scikits.statsmodels.sandbox.tsa.tsatools import lagmat, lagmat2ds
from scikits.statsmodels.sandbox.tools.stattools import ResultsStore
from adfvalues import *
#from scikits.statsmodels.sandbox.rls import RLS

#taken from econpy until we have large set of critical values
adf_cv1 = '''
One-sided test of H0: Unit root vs. H1: Stationary
Approximate asymptotic critical values (t-ratio):
------------------------------------------------------------
  1%      5%      10%      Model
------------------------------------------------------------
-2.56   -1.94   -1.62     Simple ADF (no constant or trend)
-3.43   -2.86   -2.57     ADF with constant (no trend)
-3.96   -3.41   -3.13     ADF with constant & trend
------------------------------------------------------------'''
#NOTE: Don't have critical values or p-values for trend polynomial
# greater than 2.
#NOTE: I like the ResultsStore idea.  When a post-estimation test is
# run as a mix-in, then this can attach a test_results dict to the Results
# object.  If the test is standalone it can return a TestResults class
# or not as asked.
#TODO: rename, unitroot could be a super class for all unit root tests to
# mix-in to time series models
#NOTE: ALL tests should note in the docstring the null and the alternative
# and the null and alternative should be placed in the results holder.
#TODO: include drift keyword, only valid with regression == "c"
# just changes the distribution of the test statistic to a t distribution
def adfuller(x, maxlag=None, regression="c", autolag='AIC',
    store=False):
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
    store : {False, True}
        If True, then a result instance is returned additionally to
        the adf statistic

    Returns
    -------
    adf : float
        Test statistic
    usedlag : int
        Number of lags used.
    pvalue : float
        MacKinnon's approximate p-value based on MacKinnon (1994)
    critical values : dict
        Critical values for the test statistic at the 1 %, 5 %, and 10 % levels.
        Based on MacKinnon (2010)
    resstore : (optional) instance of ResultStore
        an instance of a dummy class with results attached as attributes

    Notes
    -----
    If the p-value is close to significant, then the critical values should be
    used to judge whether to accept or reject the null.

    The pvalues are (will be) interpolated from the table of critical
    values. NOT YET DONE

    still requires pvalues and maybe some cleanup

    ''Verification''

    Looks correctly sized in Monte Carlo studies.
    Differs from R tseries results in second decimal, based on a few examples

    Examples
    --------
    see example script

    References
    ----------
    Greene
    Hamilton

    Critical Values (Canonical reference)
    Fuller, W.A. 1996. `Introduction to Statistical Time Series.` 2nd ed.
        New York: Wiley.

    P-Values (regression surface approximation)
    MacKinnon, J.G. 1994.  "Approximate asymptotic distribution functions for
        unit-root and cointegration tests.  `Journal of Business and Economic
        Statistics` 12, 167-76.

    '''
    regression = regression.lower()
    if regression not in ['c','nc','ct','ctt']:
        raise ValueError("regression option %s not understood") % regression
    x = np.asarray(x)
    nobs = x.shape[0]
    if maxlag is None:
        #from Greene referencing Schwert 1989
        maxlag = 12. * np.power(nobs/100., 1/4.)

    xdiff = np.diff(x)

    xdall = lagmat(xdiff[:,None], maxlag, trim='both')
    nobs = xdall.shape[0]
    if regression == 'c':
        trendorder = 0
    elif regression == 'nc':
        trendorder = -1
    elif regression == 'ct':
        trendorder = 1
    elif regression == 'ctt':
        trendorder = 2
    trend = np.vander(np.arange(nobs), trendorder+1)
    xdall[:,0] = x[-nobs-1:-1] # replace 0 xdiff with level of x
    xdshort = xdiff[-nobs:]
#    xdshort = x[-nobs:]
#TODO: allow for this as endog, with Phillips Perron or DF test?

    if store:
        resstore = ResultsStore()

    if autolag:
        autolag = autolag.lower()
        #search for lag length with highest information criteria
        #Note: I use the same number of observations to have comparable IC
        results = {}
        for mlag in range(1,int(maxlag+1)):  # +1 so maxlag is inclusive
            results[mlag] = sm.OLS(xdshort, np.column_stack([xdall[:,:mlag+1],
                trend])).fit()
            #NOTE: mlag+1 since level is in first column.

        if autolag == 'aic':
            icbest, bestlag = max((v.aic,k) for k,v in results.iteritems())
        elif autolag == 'bic':
            icbest, bestlag = max((v.bic,k) for k,v in results.iteritems())
        elif autolag == 't-stat':
            lags = sorted(results.keys())[::-1]
            stop = stats.norm.ppf(.95)
            i = 0
            lastt, bestlag = results[lags[i]].t(-trendorder-1)
            i += 1
            while not (abs(lastt) >= stop):
                lastt, bestlag = results[lags[i]].t(-trendorder-1)
                i += 1
        else:
            raise ValueError("autolag can be None, 'aic', 'bic', or 't-stat'")

        #rerun ols with best autolag
#        xdall = lagmat(xdiff[:,None], bestlag, trim='forward')
        xdall = lagmat(xdiff[:,None], bestlag, trim='both')
        # Why trim forward here and not above?
        nobs = xdall.shape[0]
        trend = np.vander(np.arange(nobs), trendorder+1)
        xdall[:,0] = x[-nobs-1:-1] # replace 0 xdiff with level of x
        xdshort = xdiff[-nobs:]
#        xdshort = x[-nobs:]
# NOTE: switched the above.  xdiff should be endog for augmented df
        usedlag = bestlag
    else:
        usedlag = maxlag

    resols = sm.OLS(xdshort, np.column_stack([xdall[:,:usedlag+1],trend])).fit()
    #NOTE: should be usedlag+1 since the first column is the level?
    adfstat = resols.t(0)
#    adfstat = (resols.params[0]-1.0)/resols.bse[0]
# the "asymptotically correct" z statistic is obtained as
# nobs/(1-np.sum(resols.params[1:-(trendorder+1)])) (resols.params[0] - 1)
# I think this is the statistic that is used for series that are integrated
# for orders higher than I(1), ie., not ADF but cointegration tests.

    # Get approx p-value and critical values
    pvalue = mackinnonp(adfstat, regression=regression, N=1)
    critvalues = mackinnoncrit(adfstat, regression=regression, nobs=nobs)
    critvalues = {"1%" : critvalues[0], "5%" : critvalues[1],
            "10%" : critvalues[2]}
    if store:
        resstore.resols = resols
        resstore.usedlag = usedlag
        resstore.adfstat = adfstat
        resstore.critvalues = critvalues
        resstore.H0 = "The coefficient on the lagged level equals 1"
        resstore.HA = "The coefficient on the lagged level < 1"
        return adfstat, pvalue, critvalues, resstore
    else:
        return adfstat, usedlag, pvalue, critvalues


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



if __name__=="__main__":
    data = sm.datasets.macrodata.load().data
    adf = adfuller(data['realgdp'],4, autolag=None)
    acf,ci = acorr(data['realgdp'])
    pacf = pacorr(data['realgdp'])
    x = np.random.normal(size=(100,2))
    grangercausalitytests(x,2)

