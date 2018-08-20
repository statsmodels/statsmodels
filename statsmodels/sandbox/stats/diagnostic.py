# -*- coding: utf-8 -*-
"""Various Statistical Tests


Author: josef-pktd
License: BSD-3

Notes
-----
Almost fully verified against R or Gretl, not all options are the same.
In many cases of Lagrange multiplier tests both the LM test and the F test is
returned. In some but not all cases, R has the option to choose the test
statistic. Some alternative test statistic results have not been verified.


TODO
* refactor to store intermediate results
* how easy is it to attach a test that is a class to a result instance,
  for example CompareCox as a method compare_cox(self, other) ?
* StatTestMC has been moved and should be deleted

missing:

* pvalues for breaks_hansen
* additional options, compare with R, check where ddof is appropriate
* new tests:
  - breaks_ap, more recent breaks tests
  - specification tests against nonparametric alternatives


"""
from __future__ import print_function
from statsmodels.compat.python import map
import numpy as np
from scipy import stats
from statsmodels.regression.linear_model import OLS


#TODO: I like the bunch pattern for this too.
class ResultsStore(object):
    def __str__(self):
        return self._str


class CompareCox(object):
    '''Cox Test for non-nested models

    Parameters
    ----------
    results_x : Result instance
        result instance of first model
    results_z : Result instance
        result instance of second model
    attach : bool


    Formulas from Greene, section 8.3.4 translated to code

    produces correct results for Example 8.3, Greene


    '''


    def run(self, results_x, results_z, attach=True):
        '''run Cox test for non-nested models

        Parameters
        ----------
        results_x : Result instance
            result instance of first model
        results_z : Result instance
            result instance of second model
        attach : bool
            If true, then the intermediate results are attached to the instance.

        Returns
        -------
        tstat : float
            t statistic for the test that including the fitted values of the
            first model in the second model has no effect.
        pvalue : float
            two-sided pvalue for the t statistic

        Notes
        -----
        Tests of non-nested hypothesis might not provide unambiguous answers.
        The test should be performed in both directions and it is possible
        that both or neither test rejects. see ??? for more information.

        References
        ----------
        ???

        '''

        if not np.allclose(results_x.model.endog, results_z.model.endog):
            raise ValueError('endogenous variables in models are not the same')
        nobs = results_x.model.endog.shape[0]
        x = results_x.model.exog
        z = results_z.model.exog
        sigma2_x = results_x.ssr/nobs
        sigma2_z = results_z.ssr/nobs
        yhat_x = results_x.fittedvalues
        yhat_z = results_z.fittedvalues
        res_dx = OLS(yhat_x, z).fit()
        err_zx = res_dx.resid
        res_xzx = OLS(err_zx, x).fit()
        err_xzx = res_xzx.resid

        sigma2_zx = sigma2_x + np.dot(err_zx.T, err_zx)/nobs
        c01 = nobs/2. * (np.log(sigma2_z) - np.log(sigma2_zx))
        v01 = sigma2_x * np.dot(err_xzx.T, err_xzx) / sigma2_zx**2
        q = c01 / np.sqrt(v01)
        pval = 2*stats.norm.sf(np.abs(q))

        if attach:
            self.res_dx = res_dx
            self.res_xzx = res_xzx
            self.c01 = c01
            self.v01 = v01
            self.q = q
            self.pvalue = pval
            self.dist = stats.norm

        return q, pval

    def __call__(self, results_x, results_z):
        return self.run(results_x, results_z, attach=False)


compare_cox = CompareCox()
compare_cox.__doc__ = CompareCox.__doc__


class CompareJ(object):
    '''J-Test for comparing non-nested models

    Parameters
    ----------
    results_x : Result instance
        result instance of first model
    results_z : Result instance
        result instance of second model
    attach : bool


    From description in Greene, section 8.3.3

    produces correct results for Example 8.3, Greene - not checked yet
    #currently an exception, but I don't have clean reload in python session

    check what results should be attached

    '''


    def run(self, results_x, results_z, attach=True):
        '''run J-test for non-nested models

        Parameters
        ----------
        results_x : Result instance
            result instance of first model
        results_z : Result instance
            result instance of second model
        attach : bool
            If true, then the intermediate results are attached to the instance.

        Returns
        -------
        tstat : float
            t statistic for the test that including the fitted values of the
            first model in the second model has no effect.
        pvalue : float
            two-sided pvalue for the t statistic

        Notes
        -----
        Tests of non-nested hypothesis might not provide unambiguous answers.
        The test should be performed in both directions and it is possible
        that both or neither test rejects. see ??? for more information.

        References
        ----------
        ???

        '''
        if not np.allclose(results_x.model.endog, results_z.model.endog):
            raise ValueError('endogenous variables in models are not the same')
        nobs = results_x.model.endog.shape[0]
        y = results_x.model.endog
        x = results_x.model.exog
        z = results_z.model.exog
        #sigma2_x = results_x.ssr/nobs
        #sigma2_z = results_z.ssr/nobs
        yhat_x = results_x.fittedvalues
        #yhat_z = results_z.fittedvalues
        res_zx = OLS(y, np.column_stack((yhat_x, z))).fit()
        self.res_zx = res_zx  #for testing
        tstat = res_zx.tvalues[0]
        pval = res_zx.pvalues[0]
        if attach:
            self.res_zx = res_zx
            self.dist = stats.t(res_zx.df_resid)
            self.teststat = tstat
            self.pvalue = pval

        return tstat, pval

    def __call__(self, results_x, results_z):
        return self.run(results_x, results_z, attach=False)


compare_j = CompareJ()
compare_j.__doc__ = CompareJ.__doc__


def _het_goldfeldquandt2_old(y, x, idx, split=None, retres=False):
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
        If 0<split<1 then split is interpreted as fraction of the observations
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
    y = np.asarray(y)
    nobs, nvars = x.shape
    if split is None:
        split = nobs//2
    elif (0<split) and (split<1):
        split = int(nobs*split)

    xsortind = np.argsort(x[:,idx])
    y = y[xsortind]
    x = x[xsortind,:]
    resols1 = OLS(y[:split], x[:split]).fit()
    resols2 = OLS(y[split:], x[split:]).fit()
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
        If 0<split<1 then split is interpreted as fraction of the observations
        in the first sample
    drop : None, float or int
        If this is not None, then observation are dropped from the middle part
        of the sorted series. If 0<split<1 then split is interpreted as fraction
        of the number of observations to be dropped.
        Note: Currently, observations are dropped between split and
        split+drop, where split and drop are the indices (given by rounding if
        specified as fraction). The first sample is [0:split], the second
        sample is [split+drop:]
    alternative : string, 'increasing', 'decreasing' or 'two-sided'
        default is increasing. This specifies the alternative for the p-value
        calculation.

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
    The Null hypothesis is that the variance in the two sub-samples are the
    same. The alternative hypothesis, can be increasing, i.e. the variance in
    the second sample is larger than in the first, or decreasing or two-sided.

    Results are identical R, but the drop option is defined differently.
    (sorting by idx not tested yet)
    '''
    #TODO: can do Chow test for structural break in same way
    def run(self, y, x, idx=None, split=None, drop=None,
            alternative='increasing', attach=True):
        '''see class docstring'''
        x = np.asarray(x)
        y = np.asarray(y)#**2
        nobs, nvars = x.shape
        if split is None:
            split = nobs//2
        elif (0<split) and (split<1):
            split = int(nobs*split)

        if drop is None:
            start2 = split
        elif (0<drop) and (drop<1):
            start2 = split + int(nobs*drop)
        else:
            start2 = split + drop

        if not idx is None:
            xsortind = np.argsort(x[:,idx])
            y = y[xsortind]
            x = x[xsortind,:]

        resols1 = OLS(y[:split], x[:split]).fit()
        resols2 = OLS(y[start2:], x[start2:]).fit()
        fval = resols2.mse_resid/resols1.mse_resid
        #if fval>1:
        if alternative.lower() in ['i', 'inc', 'increasing']:
            fpval = stats.f.sf(fval, resols1.df_resid, resols2.df_resid)
            ordering = 'increasing'
        elif alternative.lower() in ['d', 'dec', 'decreasing']:
            fval = fval;
            fpval = stats.f.sf(1./fval, resols2.df_resid, resols1.df_resid)
            ordering = 'decreasing'
        elif alternative.lower() in ['2', '2-sided', 'two-sided']:
            fpval_sm = stats.f.cdf(fval, resols2.df_resid, resols1.df_resid)
            fpval_la = stats.f.sf(fval, resols2.df_resid, resols1.df_resid)
            fpval = 2*min(fpval_sm, fpval_la)
            ordering = 'two-sided'
        else:
            raise ValueError('invalid alternative')

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
            res._str = '''The Goldfeld-Quandt test for null hypothesis that the
    variance in the second subsample is %s than in the first subsample:
        F-statistic =%8.4f and p-value =%8.4f''' % (ordering, fval, fpval)

        return fval, fpval, ordering
        #return self

    def __str__(self):
        try:
            return self._str
        except AttributeError:
            return repr(self)

    #TODO: missing the alternative option in call
    def __call__(self, y, x, idx=None, split=None, drop=None,
                 alternative='increasing'):
        return self.run(y, x, idx=idx, split=split, drop=drop, attach=False,
                        alternative=alternative)

het_goldfeldquandt = HetGoldfeldQuandt()
het_goldfeldquandt.__doc__ = het_goldfeldquandt.run.__doc__


def _neweywestcov(resid, x):
    '''
    Did not run yet

    from regstats2 ::

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
    nlags = int(round(4*(nobs/100.)**(2/9.)))
    hhat = resid * x.T
    xuux = np.dot(hhat, hhat.T)
    for lag in range(nlags):
        za = np.dot(hhat[:,lag:nobs], hhat[:,:nobs-lag].T)
        w = 1 - lag/(nobs + 1.)
        xuux = xuux + np.dot(w, za+za.T)
    xtxi = np.linalg.inv(np.dot(x.T, x))  #QR instead?
    covbNW = np.dot(xtxi, np.dot(xuux, xtxi))

    return covbNW


def _recursive_olsresiduals2(olsresults, skip):
    '''this is my original version based on Greene and references

    keep for now for comparison and benchmarking
    '''
    y = olsresults.model.endog
    x = olsresults.model.exog
    nobs, nvars = x.shape
    rparams = np.nan * np.zeros((nobs,nvars))
    rresid = np.nan * np.zeros((nobs))
    rypred = np.nan * np.zeros((nobs))
    rvarraw = np.nan * np.zeros((nobs))

    #XTX = np.zeros((nvars,nvars))
    #XTY = np.zeros((nvars))

    x0 = x[:skip]
    y0 = y[:skip]
    XTX = np.dot(x0.T, x0)
    XTY = np.dot(x0.T, y0) #xi * y   #np.dot(xi, y)
    beta = np.linalg.solve(XTX, XTY)
    rparams[skip-1] = beta
    yipred = np.dot(x[skip-1], beta)
    rypred[skip-1] = yipred
    rresid[skip-1] = y[skip-1] - yipred
    rvarraw[skip-1] = 1+np.dot(x[skip-1],np.dot(np.linalg.inv(XTX),x[skip-1]))
    for i in range(skip,nobs):
        xi = x[i:i+1,:]
        yi = y[i]
        xxT = np.dot(xi.T, xi)  #xi is 2d 1 row
        xy = (xi*yi).ravel() # XTY is 1d  #np.dot(xi, yi)   #np.dot(xi, y)
        print(xy.shape, XTY.shape)
        print(XTX)
        print(XTY)
        beta = np.linalg.solve(XTX, XTY)
        rparams[i-1] = beta  #this is beta based on info up to t-1
        yipred = np.dot(xi, beta)
        rypred[i] = yipred
        rresid[i] = yi - yipred
        rvarraw[i] = 1 + np.dot(xi,np.dot(np.linalg.inv(XTX),xi.T))
        XTX += xxT
        XTY += xy

    i = nobs
    beta = np.linalg.solve(XTX, XTY)
    rparams[i-1] = beta

    rresid_scaled = rresid/np.sqrt(rvarraw)   #this is N(0,sigma2) distributed
    nrr = nobs-skip
    sigma2 = rresid_scaled[skip-1:].var(ddof=1)
    rresid_standardized = rresid_scaled/np.sqrt(sigma2) #N(0,1) distributed
    rcusum = rresid_standardized[skip-1:].cumsum()
    #confidence interval points in Greene p136 looks strange?
    #this assumes sum of independent standard normal
    #rcusumci = np.sqrt(np.arange(skip,nobs+1))*np.array([[-1.],[+1.]])*stats.norm.sf(0.025)
    a = 1.143 #for alpha=0.99  =0.948 for alpha=0.95
    #following taken from Ploberger,
    crit = a*np.sqrt(nrr)
    rcusumci = (a*np.sqrt(nrr) + a*np.arange(0,nobs-skip)/np.sqrt(nrr)) \
                  * np.array([[-1.],[+1.]])
    return (rresid, rparams, rypred, rresid_standardized, rresid_scaled,
            rcusum, rcusumci)


def recursive_olsresiduals(olsresults, skip=None, lamda=0.0, alpha=0.95):
    '''calculate recursive ols with residuals and cusum test statistic

    Parameters
    ----------
    olsresults : instance of RegressionResults
        uses only endog and exog
    skip : int or None
        number of observations to use for initial OLS, if None then skip is
        set equal to the number of regressors (columns in exog)
    lamda : float
        weight for Ridge correction to initial (X'X)^{-1}
    alpha : {0.95, 0.99}
        confidence level of test, currently only two values supported,
        used for confidence interval in cusum graph

    Returns
    -------
    rresid : array
        recursive ols residuals
    rparams : array
        recursive ols parameter estimates
    rypred : array
        recursive prediction of endogenous variable
    rresid_standardized : array
        recursive residuals standardized so that N(0,sigma2) distributed, where
        sigma2 is the error variance
    rresid_scaled : array
        recursive residuals normalize so that N(0,1) distributed
    rcusum : array
        cumulative residuals for cusum test
    rcusumci : array
        confidence interval for cusum test, currently hard coded for alpha=0.95


    Notes
    -----
    It produces same recursive residuals as other version. This version updates
    the inverse of the X'X matrix and does not require matrix inversion during
    updating. looks efficient but no timing

    Confidence interval in Greene and Brown, Durbin and Evans is the same as
    in Ploberger after a little bit of algebra.

    References
    ----------
    jplv to check formulas, follows Harvey
    BigJudge 5.5.2b for formula for inverse(X'X) updating
    Greene section 7.5.2

    Brown, R. L., J. Durbin, and J. M. Evans. “Techniques for Testing the
    Constancy of Regression Relationships over Time.”
    Journal of the Royal Statistical Society. Series B (Methodological) 37,
    no. 2 (1975): 149-192.

    '''

    y = olsresults.model.endog
    x = olsresults.model.exog
    nobs, nvars = x.shape
    if skip is None:
        skip = nvars
    rparams = np.nan * np.zeros((nobs,nvars))
    rresid = np.nan * np.zeros((nobs))
    rypred = np.nan * np.zeros((nobs))
    rvarraw = np.nan * np.zeros((nobs))


    #intialize with skip observations
    x0 = x[:skip]
    y0 = y[:skip]
    #add Ridge to start (not in jplv
    XTXi = np.linalg.inv(np.dot(x0.T, x0)+lamda*np.eye(nvars))
    XTY = np.dot(x0.T, y0) #xi * y   #np.dot(xi, y)
    #beta = np.linalg.solve(XTX, XTY)
    beta = np.dot(XTXi, XTY)
    #print('beta', beta
    rparams[skip-1] = beta
    yipred = np.dot(x[skip-1], beta)
    rypred[skip-1] = yipred
    rresid[skip-1] = y[skip-1] - yipred
    rvarraw[skip-1] = 1 + np.dot(x[skip-1],np.dot(XTXi, x[skip-1]))
    for i in range(skip,nobs):
        xi = x[i:i+1,:]
        yi = y[i]
        #xxT = np.dot(xi.T, xi)  #xi is 2d 1 row
        xy = (xi*yi).ravel() # XTY is 1d  #np.dot(xi, yi)   #np.dot(xi, y)
        #print(xy.shape, XTY.shape
        #print(XTX
        #print(XTY

        # get prediction error with previous beta
        yipred = np.dot(xi, beta)
        rypred[i] = yipred
        residi = yi - yipred
        rresid[i] = residi

        #update beta and inverse(X'X)
        tmp = np.dot(XTXi, xi.T)
        ft = 1 + np.dot(xi, tmp)

        XTXi = XTXi - np.dot(tmp,tmp.T) / ft  #BigJudge equ 5.5.15

        #print('beta', beta
        beta = beta + (tmp*residi / ft).ravel()  #BigJudge equ 5.5.14
#        #version for testing
#        XTY += xy
#        beta = np.dot(XTXi, XTY)
#        print((tmp*yipred / ft).shape
#        print('tmp.shape, ft.shape, beta.shape', tmp.shape, ft.shape, beta.shape
        rparams[i] = beta
        rvarraw[i] = ft



    i = nobs
    #beta = np.linalg.solve(XTX, XTY)
    #rparams[i] = beta

    rresid_scaled = rresid/np.sqrt(rvarraw)   #this is N(0,sigma2) distributed
    nrr = nobs-skip
    #sigma2 = rresid_scaled[skip-1:].var(ddof=1)  #var or sum of squares ?
            #Greene has var, jplv and Ploberger have sum of squares (Ass.:mean=0)
    #Gretl uses: by reverse engineering matching their numbers
    sigma2 = rresid_scaled[skip:].var(ddof=1)
    rresid_standardized = rresid_scaled/np.sqrt(sigma2) #N(0,1) distributed
    rcusum = rresid_standardized[skip-1:].cumsum()
    #confidence interval points in Greene p136 looks strange. Cleared up
    #this assumes sum of independent standard normal, which does not take into
    #account that we make many tests at the same time
    #rcusumci = np.sqrt(np.arange(skip,nobs+1))*np.array([[-1.],[+1.]])*stats.norm.sf(0.025)
    if alpha == 0.95:
        a = 0.948 #for alpha=0.95
    elif alpha == 0.99:
        a = 1.143 #for alpha=0.99
    elif alpha == 0.90:
        a = 0.850
    else:
        raise ValueError('alpha can only be 0.9, 0.95 or 0.99')

    #following taken from Ploberger,
    crit = a*np.sqrt(nrr)
    rcusumci = (a*np.sqrt(nrr) + 2*a*np.arange(0,nobs-skip)/np.sqrt(nrr)) \
                 * np.array([[-1.],[+1.]])
    return (rresid, rparams, rypred, rresid_standardized, rresid_scaled,
            rcusum, rcusumci)


#def breaks_cusum(recolsresid):
#    '''renormalized cusum test for parameter stability based on recursive residuals
#
#
#    still incorrect: in PK, the normalization for sigma is by T not T-K
#    also the test statistic is asymptotically a Wiener Process, Brownian motion
#    not Brownian Bridge
#    for testing: result reject should be identical as in standard cusum version
#
#    References
#    ----------
#    Ploberger, Werner, and Walter Kramer. “The Cusum Test with Ols Residuals.”
#    Econometrica 60, no. 2 (March 1992): 271-285.
#
#    '''
#    resid = recolsresid.ravel()
#    nobssigma2 = (resid**2).sum()
#    #B is asymptotically a Brownian Bridge
#    B = resid.cumsum()/np.sqrt(nobssigma2) # use T*sigma directly
#    nobs = len(resid)
#    denom = 1. + 2. * np.arange(nobs)/(nobs-1.) #not sure about limits
#    sup_b = np.abs(B/denom).max()
#    #asymptotically distributed as standard Brownian Bridge
#    crit = [(1,1.63), (5, 1.36), (10, 1.22)]
#    #Note stats.kstwobign.isf(0.1) is distribution of sup.abs of Brownian Bridge
#    #>>> stats.kstwobign.isf([0.01,0.05,0.1])
#    #array([ 1.62762361,  1.35809864,  1.22384787])
#    pval = stats.kstwobign.sf(sup_b)
#    return sup_b, pval, crit


def breaks_AP(endog, exog, skip):
    '''supLM, expLM and aveLM by Andrews, and Andrews,Ploberger

    p-values by B Hansen

    just idea for computation of sequence of tests with given change point
    (Chow tests)
    run recursive ols both forward and backward, match the two so they form a
    split of the data, calculate sum of squares for residuals and get test
    statistic for each breakpoint between skip and nobs-skip
    need to put recursive ols (residuals) into separate function

    alternative: B Hansen loops over breakpoints only once and updates
        x'x and xe'xe
    update: Andrews is based on GMM estimation not OLS, LM test statistic is
       easy to compute because it only requires full sample GMM estimate (p.837)
       with GMM the test has much wider applicability than just OLS



    for testing loop over single breakpoint Chow test function

    '''
    pass


#delete when testing is finished
class StatTestMC(object):
    """class to run Monte Carlo study on a statistical test'''

    TODO
    print(summary, for quantiles and for histogram
    draft in trying out script log


    this has been copied to tools/mctools.py, with improvements

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
            if idx is not None:
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
    from statsmodels.stats.diagnostic import unitroot_adf

    examples = ['adf']
    if 'adf' in examples:

        x = np.random.randn(20)
        print(acorr_ljungbox(x,4))
        print(unitroot_adf(x))

        nrepl = 100
        nobs = 100
        mcres = np.zeros(nrepl)
        for ii in range(nrepl-1):
            x = (1e-4+np.random.randn(nobs)).cumsum()
            mcres[ii] = unitroot_adf(x, 2,trendorder=0, autolag=None)[0]

        print((mcres<-2.57).sum())
        print(np.histogram(mcres))
        mcressort = np.sort(mcres)
        for ratio in [0.01, 0.025, 0.05, 0.1]:
            print(ratio, mcressort[int(nrepl*ratio)])

        print('critical values in Green table 20.5')
        print('sample size = 100')
        print('with constant')
        print('0.01: -19.8,  0.025: -16.3, 0.05: -13.7, 0.01: -11.0, 0.975: 0.47')

        print('0.01: -3.50,  0.025: -3.17, 0.05: -2.90, 0.01: -2.58, 0.975: 0.26')
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
        print(crit_5lags0p05)


        adfstat, _,_,resstore = unitroot_adf(x, 2,trendorder=0, autolag=None, store=1)

        print((mcres>crit_5lags0p05).sum())

        print(resstore.resols.model.exog[-5:])
        print(x[-5:])

        print(np.histogram(mcres, bins=[-np.inf, -3.5, -3.17, -2.9 , -2.58,  0.26, np.inf]))

        print(mcressort[(nrepl*(np.array([0.01, 0.025, 0.05, 0.1, 0.975]))).astype(int)])


        def randwalksim(nobs=100, drift=0.0):
            return (drift+np.random.randn(nobs)).cumsum()

        def normalnoisesim(nobs=500, loc=0.0):
            return (loc+np.random.randn(nobs))

        def adf20(x):
            return unitroot_adf(x, 2,trendorder=0, autolag=None)[:2]

        print('\nResults with MC class')
        mc1 = StatTestMC(randwalksim, adf20)
        mc1.run(1000, statindices=[0,1])
        print(mc1.histogram(0, critval=[-3.5, -3.17, -2.9 , -2.58,  0.26]))
        print(mc1.quantiles(0))

        print('\nLjung Box')

        def lb4(x):
            s,p = acorr_ljungbox(x, lags=4)
            return s[-1], p[-1]

        def lb4(x):
            s,p = acorr_ljungbox(x, lags=1)
            return s[0], p[0]

        print('Results with MC class')
        mc1 = StatTestMC(normalnoisesim, lb4)
        mc1.run(1000, statindices=[0,1])
        print(mc1.histogram(1, critval=[0.01, 0.025, 0.05, 0.1, 0.975]))
        print(mc1.quantiles(1))
        print(mc1.quantiles(0))
        print(mc1.histogram(0))

    nobs = 100
    x = np.ones((nobs,2))
    x[:,1] = np.arange(nobs)/20.
    y = x.sum(1) + 1.01*(1+1.5*(x[:,1]>10))*np.random.rand(nobs)
    print(het_goldfeldquandt(y,x, 1))

    y = x.sum(1) + 1.01*(1+0.5*(x[:,1]>10))*np.random.rand(nobs)
    print(het_goldfeldquandt(y,x, 1))

    y = x.sum(1) + 1.01*(1-0.5*(x[:,1]>10))*np.random.rand(nobs)
    print(het_goldfeldquandt(y,x, 1))

    print(het_breuschpagan(y,x))
    print(het_white(y,x))

    f, fp, fo = het_goldfeldquandt(y,x, 1)
    print(f, fp)
    resgq = het_goldfeldquandt(y,x, 1, retres=True)
    print(resgq)

    #this is just a syntax check:
    print(_neweywestcov(y, x))

    resols1 = OLS(y, x).fit()
    print(_neweywestcov(resols1.resid, x))
    print(resols1.cov_params())
    print(resols1.HC0_se)
    print(resols1.cov_HC0)

    y = x.sum(1) + 10.*(1-0.5*(x[:,1]>10))*np.random.rand(nobs)
    print(HetGoldfeldQuandt().run(y,x, 1, alternative='dec'))
