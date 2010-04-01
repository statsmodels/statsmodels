# -*- coding: utf-8 -*-
"""Various Statistical Tests


Warning: Work in progress


Author: josef-pktd
License: BSD
"""


import numpy as np
from scipy import stats
import scikits.statsmodels as sm
from scikits.statsmodels.sandbox.tsa import acf
from scikits.statsmodels.sandbox.tools.tools_tsa import lagmat

class ResultsStore(object):
    def __str__(self):
        return self._str

def acorr_ljungbox(x, lags=None, boxpierce=False):
    '''Ljung-Box test for no autocorrelation


    Parameters
    ----------
    x : array_like, 1d
        data series
    lags : None, int or array_like
        If lags is an integer then this is taken to be the largest lag
        that is included, the test result is reported for all smaller lag length.
        If lags is a list or array, then all lags are included up to the largest
        lag in the list, however only the tests for the lags in the list are
        reported.
        If lags is None, then the default maxlag is 12*(nobs/100)^{1/4}
    boxpierce : {False, True}
        If true, then additional to the results of the Ljung-Box test also the
        Box-Pierce test results are returned

    Returns
    -------
    lbvalue : float or array
        test statistic
    pvalue : float or array
        p-value based on chi-square distribution
    bpvalue : (optionsal), float or array
        test statistic for Box-Pierce test
    bppvalue : (optional), float or array
        p-value based for Box-Pierce test on chi-square distribution

    Notes
    -----
    Ljung-Box and Box-Pierce statistic differ in their scaling of the
    autocorrelation function. Ljung-Box test is reported to have better
    small sample properties.

    could be extended to work with more than one series
    1d or nd ? axis ? ravel ?
    needs more testing

    ''Verification''

    Looks correctly sized in Monte Carlo studies.
    not yet compared to verified values

    Examples
    --------
    see example script

    References
    ----------
    Greene
    Wikipedia

    '''
    x = np.asarray(x)
    nobs = x.shape[0]
    if lags is None:
        lags = range(1,nobs/4.)  #TODO: check default
    elif isinstance(lags, int):
        lags = range(1,lags+1)
    maxlag = max(lags)
    lags = np.asarray(lags)

    acfx = acf(x, unbiased=False)   # normalize by nobs not (nobs-nlags)
    acf2norm = acfx[1:maxlag+1]**2 / (nobs - np.arange(1,maxlag+1))
    qljungbox = nobs * (nobs+2) * np.cumsum(acf2norm)[lags-1]
    pval = stats.chi2.sf(qljungbox, lags)
    if not boxpierce:
        return qljungbox, pval
    else:
        qboxpierce = nobs * np.cumsum(acfx[1:maxlag+1]**2)[lags]
        pvalbp = stats.chi2.sf(qboxpierce, lags)
        return qljungbox, pval, qboxpierce, pvalbp

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

def unitroot_adf(x, maxlag=None, trendorder=0, autolag='AIC', store=False):
    '''Augmented Dickey-Fuller unit root test

    Parameters
    ----------
    x : array_like, 1d
        data series
    maxlag : int
        maximum lag which is included in test, default 12*(nobs/100)^{1/4}
    trendorder : int
        constant and trend order to include in regression
        * -1: no constant no trend
        *  0: constant only
        * p>0 : trend polynomial of order p
    autolag : {'AIC', 'BIC', None}
        * if None, then maxlag lags are used
        * if 'AIC' or 'BIC', then the number of lags is chosen to minimize the
          corresponding information criterium
        * TODO: t-statistic based choice of maxlag
    store : {False, True}
        If true, then a result instance is returned additionally to
        the adf statistic

    Returns
    -------
    adf : float
        test statistic
    pvalue : NOT YET IMPLEMENTED
    resstore : (optional) instance of ResultStore
        an instance of a dummy class with results attached as attributes

    Notes
    -----
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
    Wikipedia

    '''
    x = np.asarray(x)
    nobs = x.shape[0]
    if maxlag is None:
        #from Greene referencing Schwert 1989
        maxlag = 12. * np.power(nobs/100., 1/4.)


    xdiff = np.diff(x)
    #
    xdall = lagmat(xdiff[:,None], maxlag, trim='both')
    nobs = xdall.shape[0]
    trend = np.vander(np.arange(nobs), trendorder+1)
    xdall[:,0] = x[-nobs-1:-1] # replace 0 xdiff with level of x
    #xdshort = xdiff[-nobs:]
    xdshort = x[-nobs:]

    if store: resstore = ResultsStore()

    if autolag:
        #search for lag length with highest information criteria
        #Note: I use the same number of observations to have comparable IC
        results = {}
        for mlag in range(1,maxlag):
            results[mlag] = sm.OLS(xdshort, np.column_stack([xdall[:,:mlag],trend])).fit()

        if autolag.lower() == 'aic':
            bestic, icbestlag = max((v.aic,k) for k,v in results.iteritems())
        elif autolag.lower() == 'bic':
            icbest, icbestlag = max((v.bic,k) for k,v in results.iteritems())
        else:
            raise ValueError("autolag can only be None, 'AIC' or 'BIC'")

        #rerun ols with best ic
        xdall = lagmat(xdiff[:,None], icbestlag, trim='forward')
        nobs = xdall.shape[0]
        trend = np.vander(np.arange(nobs), trendorder+1)
        xdall[:,0] = x[-nobs-1:-1] # replace 0 xdiff with level of x
        #xdshort = xdiff[-nobs:]
        xdshort = x[-nobs:]
        usedlag = icbestlag
    else:
        usedlag = maxlag

    resols = sm.OLS(xdshort, np.column_stack([xdall[:,:usedlag],trend])).fit()
    adfstat = resols.t(0)
    adfstat = (resols.params[0]-1.0)/resols.bse[0]
    if store:
        resstore.resols = resols
        resstore.usedlag = usedlag
        return adfstat, resstore
    else:
        return adfstat


def acorr_lm(x, maxlag=None, autolag='AIC', store=False):
    '''Lagrange Multiplier tests for autocorrelation

    not checked yet, copied from unitrood_adf with adjustments
    check array shapes because of the addition of the constant.

    Notes
    -----
    If x is calculated as y^2 for a time series y, then this test corresponds
    to the Engel test for autoregressive conditional heteroscedasticity (ARCH).
    TODO: get details and verify

    '''

    x = np.asarray(x)
    nobs = x.shape[0]
    if maxlag is None:
        #for adf from Greene referencing Schwert 1989
        maxlag = 12. * np.power(nobs/100., 1/4.)#nobs//4  #TODO: check default, or do AIC/BIC


    xdiff = np.diff(x)
    #
    xdall = lagmat(x[:-1,None], maxlag, trim='both')
    nobs = xdall.shape[0]
    xdall = np.c_[np.ones((nobs,1)), xdall]
    xshort = x[-nobs:]

    if store: resstore = ResultsStore()

    if autolag:
        #search for lag length with highest information criteria
        #Note: I use the same number of observations to have comparable IC
        results = {}
        for mlag in range(1,maxlag):
            results[mlag] = sm.OLS(xshort, xdall[:,:mlag+1]).fit()

        if autolag.lower() == 'aic':
            bestic, icbestlag = max((v.aic,k) for k,v in results.iteritems())
        elif autolag.lower() == 'bic':
            icbest, icbestlag = max((v.bic,k) for k,v in results.iteritems())
        else:
            raise ValueError("autolag can only be None, 'AIC' or 'BIC'")

        #rerun ols with best ic
        xdall = lagmat(x[:,None], icbestlag, trim='forward')
        nobs = xdall.shape[0]
        xdall = np.c_[np.ones((nobs,1)), xdall]
        xshort = x[-nobs:]
        usedlag = icbestlag
    else:
        usedlag = maxlag

    resols = sm.OLS(xshort, xdall[:,:usedlag+1]).fit()
    fval = resols.fvalue
    fpval = resols.f_pvalue
    lm = nobs * resols.rsquared
    lmpval = stats.chi2.sf(lm, usedlag)
    # Note: degrees of freedom for LM test is nvars minus constant = usedlags
    return fval, fpval, lm, lmpval

    if store:
        resstore.resols = resols
        resstore.usedlag = usedlag
        return fval, fpval, lm, lmpval, resstore
    else:
        return fval, fpval, lm, lmpval


def het_breushpagan(y,x):
    '''Lagrange Multiplier Heteroscedasticity Test by Breush-Pagan

    Notes
    -----
    assumes x contains constant (for counting dof)

    need to check this again, is different in Greene p224

    References
    ----------
    http://en.wikipedia.org/wiki/Breusch%E2%80%93Pagan_test
    Greene
    '''
    x = np.asarray(x)
    y = np.asarray(y)**2
    nobs, nvars = x.shape
    resols = sm.OLS(y, x).fit()
    fval = resols.fvalue
    fpval = resols.f_pvalue
    lm = nobs * resols.rsquared
    # Note: degrees of freedom for LM test is nvars minus constant
    return lm, stats.chi2.sf(lm, nvars-1), fval, fpval

def het_white(y, x, retres=False):
    '''Lagrange Multiplier Heteroscedasticity Test by White

    Notes
    -----
    assumes x contains constant (for counting dof)

    question: does f-statistic make sense? constant ?

    References
    ----------

    Greene section 11.4.1 5th edition p. 222
    '''
    x = np.asarray(x)
    y = np.asarray(y)**2
    if x.ndim == 1:
        raise ValueError('x should have constant and at least one more variable')
    nobs, nvars0 = x.shape
    i0,i1 = np.triu_indices(nvars0)
    exog = x[:,i0]*x[:,i1]
    nobs, nvars = exog.shape
    assert nvars == nvars0*(nvars0-1)/2. + nvars0
    resols = sm.OLS(y**2, exog).fit()
    fval = resols.fvalue
    fpval = resols.f_pvalue
    lm = nobs * resols.rsquared
    # Note: degrees of freedom for LM test is nvars minus constant
    lmpval = stats.chi2.sf(lm, nvars-1)
    return lm, lmpval, fval, fpval

def het_goldfeldquandt(y, x, idx, split=None, retres=False):
    '''test whether variance is the same in 2 subsamples

    Parameters
    ----------
    y : array_like
        endogenous variable
    x : array_like
        exogenous variable, regressors
    idx : integer
        column index of variable according to which observations are
        sorted for the split
    split : None or integer or float in intervall (0,1)
        index at which sample is split.
        If 0<split<0 then split is interpreted as fraction of the observations
        in the first sample
    retres : boolean
        if true, then an instance of a result class is returned,
        otherwise 2 numbers, fvalue and p-value, are returned

    Returns
    -------
    (fval, pval) or res
    fval : float
        value of the F-statistic
    pval : float
        p-value of the hypothesis that the variance in one subsample is larger
        than in the other subsample
    res : instance of result class
        The class instance is just a storage for the intermediate and final
        results that are calculated

    Notes
    -----

    TODO:
    add resultinstance - DONE
    maybe add drop-middle as option
    maybe allow for several breaks

    recommendation for users: use this function as pattern for more flexible
        split in tests, e.g. drop middle.

    can do Chow test for structural break in same way

    ran sanity check
    '''
    x = np.asarray(x)
    y = np.asarray(y)**2
    nobs, nvars = x.shape
    if split is None:
        split = nobs//2
    elif (0<split) and (split<1):
        split = int(nobs*split)

    xsortind = np.argsort(x[:,idx])
    y = y[xsortind]
    x = x[xsortind,:]
    resols1 = sm.OLS(y[:split], x[:split]).fit()
    resols2 = sm.OLS(y[split:], x[split:]).fit()
    fval = resols1.mse_resid/resols2.mse_resid
    if fval>1:
        fpval = stats.f.sf(fval, resols1.df_resid, resols2.df_resid)
        ordering = 'larger'
    else:
        fval = 1./fval;
        fpval = stats.f.sf(fval, resols2.df_resid, resols1.df_resid)
        ordering = 'smaller'

    if retres:
        res = ResultsStore()
        res.__doc__ = 'Test Results for Goldfeld-Quandt test of heterogeneity'
        res.fval = fval
        res.fpval = fpval
        res.df_fval = (resols2.df_resid, resols1.df_resid)
        res.resols1 = resols1
        res.resols2 = resols2
        res.ordering = ordering
        res.split = split
        #res.__str__
        res._str = '''The Goldfeld-Quandt test for null hypothesis that the
variance in the second subsample is %s than in the first subsample:
    F-statistic =%8.4f and p-value =%8.4f''' % (ordering, fval, fpval)

        return res
    else:
        return fval, fpval


class HetGoldfeldQuandt(object):
    '''test whether variance is the same in 2 subsamples

    Parameters
    ----------
    y : array_like
        endogenous variable
    x : array_like
        exogenous variable, regressors
    idx : integer
        column index of variable according to which observations are
        sorted for the split
    split : None or integer or float in intervall (0,1)
        index at which sample is split.
        If 0<split<0 then split is interpreted as fraction of the observations
        in the first sample
    retres : boolean
        if true, then an instance of a result class is returned,
        otherwise 2 numbers, fvalue and p-value, are returned

    Returns
    -------
    (fval, pval) or res
    fval : float
        value of the F-statistic
    pval : float
        p-value of the hypothesis that the variance in one subsample is larger
        than in the other subsample
    res : instance of result class
        The class instance is just a storage for the intermediate and final
        results that are calculated

    Notes
    -----

    TODO:
    add resultinstance - DONE
    maybe add drop-middle as option
    maybe allow for several breaks

    recommendation for users: use this function as pattern for more flexible
        split in tests, e.g. drop middle.

    can do Chow test for structural break in same way

    ran sanity check
    '''
    def run(self, x, y, idx, split=None, attach=True):
        '''see class docstring'''
        x = np.asarray(x)
        y = np.asarray(y)**2
        nobs, nvars = x.shape
        if split is None:
            split = nobs//2
        elif (0<split) and (split<1):
            split = int(nobs*split)

        xsortind = np.argsort(x[:,idx])
        y = y[xsortind]
        x = x[xsortind,:]
        resols1 = sm.OLS(y[:split], x[:split]).fit()
        resols2 = sm.OLS(y[split:], x[split:]).fit()
        fval = resols1.mse_resid/resols2.mse_resid
        if fval>1:
            fpval = stats.f.sf(fval, resols1.df_resid, resols2.df_resid)
            ordering = 'larger'
        else:
            fval = 1./fval;
            fpval = stats.f.sf(fval, resols2.df_resid, resols1.df_resid)
            ordering = 'smaller'

        if attach:
            res = self
            res.__doc__ = 'Test Results for Goldfeld-Quandt test of heterogeneity'
            res.fval = fval
            res.fpval = fpval
            res.df_fval = (resols2.df_resid, resols1.df_resid)
            res.resols1 = resols1
            res.resols2 = resols2
            res.ordering = ordering
            res.split = split
            #res.__str__
            #TODO: check if string works
            res.__str__ = '''The Goldfeld-Quandt test for null hypothesis that the
    variance in the second subsample is %s than in the first subsample:
        F-statistic =%8.4f and p-value =%8.4f''' % (ordering, fval, fpval)

        return fval, fpval

    def __call__(self, x, y, idx, split=None):
        return self.run(x, y, idx, split=None, attach=False)

hetgoldfeldquandt2 = HetGoldfeldQuandt()
hetgoldfeldquandt2.__doc__ = hetgoldfeldquandt2.run.__doc__




def neweywestcov(resid, x):
    ''' did not run  yet
    from regstats2
    if idx(29) % HAC (Newey West)
     L = round(4*(nobs/100)^(2/9));
     % L = nobs^.25; % as an alternative
     hhat = repmat(residuals',p,1).*X';
     xuux = hhat*hhat';
     for l = 1:L;
        za = hhat(:,(l+1):nobs)*hhat(:,1:nobs-l)';
        w = 1 - l/(L+1);
        xuux = xuux + w*(za+za');
     end
     d = struct;
     d.covb = xtxi*xuux*xtxi;
    '''
    nobs = resid.shape[0]   #TODO: check this can only be 1d
    nlags = round(4*(nobs/100)^(2/9))
    hhat = resid * x.T
    xuux = np.dot(hhat, hhat.T)
    for lag in range(nlags):
        za = np.dot(hhat[:,lag:nobs] * hhat[:,:nobs-lag].T)
        w = 1 - lag/(nobs + 1.)
        xuux = xuux + np.dot(w, za+za.T)
    xtxi = np.linalg.inv(np.dot(x.T, x))  #QR instead?
    covbNW = np.dot(xtxi, np.dot(xuux, xtxi))

    return covbNW



class StatTestMC(object):
    """class to run Monte Carlo study on a statistical test'''

    TODO
    print summary, for quantiles and for histogram
    draft in trying out script log

    """

    def __init__(self, dgp, statistic):
        self.dgp = dgp #staticmethod(dgp)  #no self
        self.statistic = statistic # staticmethod(statistic)  #no self

    def run(self, nrepl, statindices=None, dgpargs=[], statsargs=[]):
        '''run the actual Monte Carlo and save results


        '''
        self.nrepl = nrepl
        self.statindices = statindices
        self.dgpargs = dgpargs
        self.statsargs = statsargs

        dgp = self.dgp
        statfun = self.statistic # name ?

        #single return statistic
        if statindices is None:
            self.nreturn = nreturns = 1
            mcres = np.zeros(nrepl)
            for ii in range(nrepl-1):
                x = dgp(*dgpargs) #(1e-4+np.random.randn(nobs)).cumsum()
                mcres[ii] = statfun(x, *statsargs) #unitroot_adf(x, 2,trendorder=0, autolag=None)
        #more than one return statistic
        else:
            self.nreturn = nreturns = len(statindices)
            self.mcres = mcres = np.zeros((nrepl, nreturns))
            for ii in range(nrepl-1):
                x = dgp(*dgpargs) #(1e-4+np.random.randn(nobs)).cumsum()
                ret = statfun(x, *statsargs)
                mcres[ii] = [ret[i] for i in statindices]

        self.mcres = mcres

    def histogram(self, idx=None, critval=None):
        '''calculate histogram values

        does not do any plotting
        '''
        if self.mcres.ndim == 2:
            if  not idx is None:
                mcres = self.mcres[:,idx]
            else:
                raise ValueError('currently only 1 statistic at a time')
        else:
            mcres = self.mcres

        if critval is None:
            histo = np.histogram(mcres, bins=10)
        else:
            if not critval[0] == -np.inf:
                bins=np.r_[-np.inf, critval, np.inf]
            if not critval[0] == -np.inf:
                bins=np.r_[bins, np.inf]
            histo = np.histogram(mcres,
                                 bins=np.r_[-np.inf, critval, np.inf])

        self.histo = histo
        self.cumhisto = np.cumsum(histo[0])*1./self.nrepl
        self.cumhistoreversed = np.cumsum(histo[0][::-1])[::-1]*1./self.nrepl
        return histo, self.cumhisto, self.cumhistoreversed

    def quantiles(self, idx=None, frac=[0.01, 0.025, 0.05, 0.1, 0.975]):
        '''calculate quantiles of Monte Carlo results

        '''

        if self.mcres.ndim == 2:
            if not idx is None:
                mcres = self.mcres[:,idx]
            else:
                raise ValueError('currently only 1 statistic at a time')
        else:
            mcres = self.mcres

        self.frac = frac = np.asarray(frac)
        self.mcressort = mcressort = np.sort(self.mcres)
        return frac, mcressort[(self.nrepl*frac).astype(int)]

if __name__ == '__main__':

    examples = []
    if 'adf' in examples:

        x = np.random.randn(20)
        print acorr_ljungbox(x)
        print unitroot_adf(x)

        nrepl = 100
        nobs = 100
        mcres = np.zeros(nrepl)
        for ii in range(nrepl-1):
            x = (1e-4+np.random.randn(nobs)).cumsum()
            mcres[ii] = unitroot_adf(x, 2,trendorder=0, autolag=None)

        print (mcres<-2.57).sum()
        print np.histogram(mcres)
        mcressort = np.sort(mcres)
        for ratio in [0.01, 0.025, 0.05, 0.1]:
            print ratio, mcressort[int(nrepl*ratio)]

        print 'critical values in Green table 20.5'
        print 'sample size = 100'
        print 'with constant'
        print '0.01: -19.8,  0.025: -16.3, 0.05: -13.7, 0.01: -11.0, 0.975: 0.47'

        print '0.01: -3.50,  0.025: -3.17, 0.05: -2.90, 0.01: -2.58, 0.975: 0.26'
        crvdg = dict([map(float,s.split(':')) for s in ('0.01: -19.8,  0.025: -16.3, 0.05: -13.7, 0.01: -11.0, 0.975: 0.47'.split(','))])
        crvd = dict([map(float,s.split(':')) for s in ('0.01: -3.50,  0.025: -3.17, 0.05: -2.90, 0.01: -2.58, 0.975: 0.26'.split(','))])
        '''
        >>> crvd
        {0.050000000000000003: -13.699999999999999, 0.97499999999999998: 0.46999999999999997, 0.025000000000000001: -16.300000000000001, 0.01: -11.0}
        >>> sorted(crvd.values())
        [-16.300000000000001, -13.699999999999999, -11.0, 0.46999999999999997]
        '''

        #for trend = 0
        crit_5lags0p05 =-4.41519 + (-14.0406)/nobs + (-12.575)/nobs**2
        print crit_5lags0p05


        adfstat, resstore = unitroot_adf(x, 2,trendorder=0, autolag=None, store=1)

        print (mcres>crit_5lags0p05).sum()

        print resstore.resols.model.exog[-5:]
        print x[-5:]

        print np.histogram(mcres, bins=[-np.inf, -3.5, -3.17, -2.9 , -2.58,  0.26, np.inf])

        print mcressort[(nrepl*(np.array([0.01, 0.025, 0.05, 0.1, 0.975]))).astype(int)]


        def randwalksim(nobs=100, drift=0.0):
            return (drift+np.random.randn(nobs)).cumsum()

        def normalnoisesim(nobs=500, loc=0.0):
            return (loc+np.random.randn(nobs))

        def adf20(x):
            return unitroot_adf(x, 2,trendorder=0, autolag=None)

        print '\nResults with MC class'
        mc1 = StatTestMC(randwalksim, adf20)
        mc1.run(1000)
        print mc1.histogram(critval=[-3.5, -3.17, -2.9 , -2.58,  0.26])
        print mc1.quantiles()

        print '\nLjung Box'

        def lb4(x):
            s,p = acorr_ljungbox(x, lags=4)
            return s[-1], p[-1]

        def lb4(x):
            s,p = acorr_ljungbox(x, lags=1)
            return s[0], p[0]

        print 'Results with MC class'
        mc1 = StatTestMC(normalnoisesim, lb4)
        mc1.run(1000, statindices=[0,1])
        print mc1.histogram(1, critval=[0.01, 0.025, 0.05, 0.1, 0.975])
        print mc1.quantiles(1)
        print mc1.quantiles(0)
        print mc1.histogram(0)

    nobs = 100
    x = np.ones((20,2))
    x[:,1] = np.arange(20)
    y = x.sum(1) + 1.01*(1+1.5*(x[:,1]>10))*np.random.rand(20)
    print het_goldfeldquandt(y,x, 1)

    y = x.sum(1) + 1.01*(1+0.5*(x[:,1]>10))*np.random.rand(20)
    print het_goldfeldquandt(y,x, 1)

    y = x.sum(1) + 1.01*(1-0.5*(x[:,1]>10))*np.random.rand(20)
    print het_goldfeldquandt(y,x, 1)

    print het_breushpagan(y,x)
    print het_white(y,x)

    f,p = het_goldfeldquandt(y,x, 1)
    print f, p
    resgq = het_goldfeldquandt(y,x, 1, retres=True)
    print resgq

    #this is just a syntax check:
    print neweywestcov(y, x)

    resols1 = sm.OLS(y, x).fit()
    print neweywestcov(resols1.resid, x)
    print resols1.cov_params()
    print resols1.HC0_se
    print resols1.cov_HC0


