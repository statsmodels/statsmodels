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
from statsmodels.compat.python import iteritems, lrange, map, long
import numpy as np
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import acf, adfuller
from statsmodels.tsa.tsatools import lagmat
from statsmodels.compat.numpy import np_matrix_rank

#get the old signature back so the examples work
def unitroot_adf(x, maxlag=None, trendorder=0, autolag='AIC', store=False):
    return adfuller(x, maxlag=maxlag, regression=trendorder, autolag=autolag,
                    store=store, regresults=False)


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


def acorr_ljungbox(x, lags=None, boxpierce=False):
    '''Ljung-Box test for no autocorrelation

    Parameters
    ----------
    x : array_like, 1d
        data series, regression residuals when used as diagnostic test
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

    TODO: could be extended to work with more than one series
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
        lags = lrange(1,41)  #TODO: check default; SS: changed to 40
    elif isinstance(lags, (int, long)):
        lags = lrange(1,lags+1)
    maxlag = max(lags)
    lags = np.asarray(lags)

    acfx = acf(x, nlags=maxlag) # normalize by nobs not (nobs-nlags)
                             # SS: unbiased=False is default now
#    acf2norm = acfx[1:maxlag+1]**2 / (nobs - np.arange(1,maxlag+1))
    acf2norm = acfx[1:maxlag+1]**2 / (nobs - np.arange(1,maxlag+1))

    qljungbox = nobs * (nobs+2) * np.cumsum(acf2norm)[lags-1]
    pval = stats.chi2.sf(qljungbox, lags)
    if not boxpierce:
        return qljungbox, pval
    else:
        qboxpierce = nobs * np.cumsum(acfx[1:maxlag+1]**2)[lags-1]
        pvalbp = stats.chi2.sf(qboxpierce, lags)
        return qljungbox, pval, qboxpierce, pvalbp

def acorr_lm(x, maxlag=None, autolag='AIC', store=False, regresults=False):
    '''Lagrange Multiplier tests for autocorrelation

    This is a generic Lagrange Multiplier test for autocorrelation. I don't
    have a reference for it, but it returns Engle's ARCH test if x is the
    squared residual array. A variation on it with additional exogenous
    variables is the Breusch-Godfrey autocorrelation test.

    Parameters
    ----------
    resid : ndarray, (nobs,)
        residuals from an estimation, or time series
    maxlag : int
        highest lag to use
    autolag : None or string
        If None, then a fixed number of lags given by maxlag is used.
    store : bool
        If true then the intermediate results are also returned

    Returns
    -------
    lm : float
        Lagrange multiplier test statistic
    lmpval : float
        p-value for Lagrange multiplier test
    fval : float
        fstatistic for F test, alternative version of the same test based on
        F test for the parameter restriction
    fpval : float
        pvalue for F test
    resstore : instance (optional)
        a class instance that holds intermediate results. Only returned if
        store=True

    See Also
    --------
    het_arch
    acorr_breusch_godfrey
    acorr_ljung_box

    '''

    if regresults:
        store = True

    x = np.asarray(x)
    nobs = x.shape[0]
    if maxlag is None:
        #for adf from Greene referencing Schwert 1989
        maxlag = int(np.ceil(12. * np.power(nobs/100., 1/4.)))#nobs//4  #TODO: check default, or do AIC/BIC


    xdiff = np.diff(x)
    #
    xdall = lagmat(x[:,None], maxlag, trim='both')
    nobs = xdall.shape[0]
    xdall = np.c_[np.ones((nobs,1)), xdall]
    xshort = x[-nobs:]

    if store: resstore = ResultsStore()

    if autolag:
        #search for lag length with highest information criteria
        #Note: I use the same number of observations to have comparable IC
        results = {}
        for mlag in range(1, maxlag+1):
            results[mlag] = OLS(xshort, xdall[:,:mlag+1]).fit()

        if autolag.lower() == 'aic':
            bestic, icbestlag = min((v.aic,k) for k,v in iteritems(results))
        elif autolag.lower() == 'bic':
            icbest, icbestlag = min((v.bic,k) for k,v in iteritems(results))
        else:
            raise ValueError("autolag can only be None, 'AIC' or 'BIC'")

        #rerun ols with best ic
        xdall = lagmat(x[:,None], icbestlag, trim='both')
        nobs = xdall.shape[0]
        xdall = np.c_[np.ones((nobs,1)), xdall]
        xshort = x[-nobs:]
        usedlag = icbestlag
        if regresults:
            resstore.results = results
    else:
        usedlag = maxlag

    resols = OLS(xshort, xdall[:,:usedlag+1]).fit()
    fval = resols.fvalue
    fpval = resols.f_pvalue
    lm = nobs * resols.rsquared
    lmpval = stats.chi2.sf(lm, usedlag)
    # Note: degrees of freedom for LM test is nvars minus constant = usedlags
    #return fval, fpval, lm, lmpval

    if store:
        resstore.resols = resols
        resstore.usedlag = usedlag
        return lm, lmpval, fval, fpval, resstore
    else:
        return lm, lmpval, fval, fpval

def het_arch(resid, maxlag=None, autolag=None, store=False, regresults=False,
             ddof=0):
    '''Engle's Test for Autoregressive Conditional Heteroscedasticity (ARCH)

    Parameters
    ----------
    resid : ndarray, (nobs,)
        residuals from an estimation, or time series
    maxlag : int
        highest lag to use
    autolag : None or string
        If None, then a fixed number of lags given by maxlag is used.
    store : bool
        If true then the intermediate results are also returned
    ddof : int
        Not Implemented Yet
        If the residuals are from a regression, or ARMA estimation, then there
        are recommendations to correct the degrees of freedom by the number
        of parameters that have been estimated, for example ddof=p+a for an
        ARMA(p,q) (need reference, based on discussion on R finance mailinglist)

    Returns
    -------
    lm : float
        Lagrange multiplier test statistic
    lmpval : float
        p-value for Lagrange multiplier test
    fval : float
        fstatistic for F test, alternative version of the same test based on
        F test for the parameter restriction
    fpval : float
        pvalue for F test
    resstore : instance (optional)
        a class instance that holds intermediate results. Only returned if
        store=True

    Notes
    -----
    verified agains R:FinTS::ArchTest

    '''

    return acorr_lm(resid**2, maxlag=maxlag, autolag=autolag, store=store,
                    regresults=regresults)


def acorr_breusch_godfrey(results, nlags=None, store=False):
    '''Breusch Godfrey Lagrange Multiplier tests for residual autocorrelation

    Parameters
    ----------
    results : Result instance
        Estimation results for which the residuals are tested for serial
        correlation
    nlags : int
        Number of lags to include in the auxiliary regression. (nlags is
        highest lag)
    store : bool
        If store is true, then an additional class instance that contains
        intermediate results is returned.

    Returns
    -------
    lm : float
        Lagrange multiplier test statistic
    lmpval : float
        p-value for Lagrange multiplier test
    fval : float
        fstatistic for F test, alternative version of the same test based on
        F test for the parameter restriction
    fpval : float
        pvalue for F test
    resstore : instance (optional)
        a class instance that holds intermediate results. Only returned if
        store=True

    Notes
    -----
    BG adds lags of residual to exog in the design matrix for the auxiliary
    regression with residuals as endog,
    see Greene 12.7.1.

    References
    ----------
    Greene Econometrics, 5th edition

    '''

    x = np.asarray(results.resid)
    exog_old = results.model.exog
    nobs = x.shape[0]
    if nlags is None:
        #for adf from Greene referencing Schwert 1989
        nlags = np.trunc(12. * np.power(nobs/100., 1/4.))#nobs//4  #TODO: check default, or do AIC/BIC
        nlags = int(nlags)

    x = np.concatenate((np.zeros(nlags), x))

    #xdiff = np.diff(x)
    #
    xdall = lagmat(x[:,None], nlags, trim='both')
    nobs = xdall.shape[0]
    xdall = np.c_[np.ones((nobs,1)), xdall]
    xshort = x[-nobs:]
    exog = np.column_stack((exog_old, xdall))
    k_vars = exog.shape[1]

    if store: resstore = ResultsStore()

    resols = OLS(xshort, exog).fit()
    ft = resols.f_test(np.eye(nlags, k_vars, k_vars - nlags))
    fval = ft.fvalue
    fpval = ft.pvalue
    fval = np.squeeze(fval)[()]   #TODO: fix this in ContrastResults
    fpval = np.squeeze(fpval)[()]
    lm = nobs * resols.rsquared
    lmpval = stats.chi2.sf(lm, nlags)
    # Note: degrees of freedom for LM test is nvars minus constant = usedlags
    #return fval, fpval, lm, lmpval

    if store:
        resstore.resols = resols
        resstore.usedlag = nlags
        return lm, lmpval, fval, fpval, resstore
    else:
        return lm, lmpval, fval, fpval


msg = "Use acorr_breusch_godfrey, acorr_breush_godfrey will be removed " \
      "in 0.9 \n (Note: misspelling missing 'c'),"

acorr_breush_godfrey = np.deprecate(acorr_breusch_godfrey, 'acorr_breush_godfrey',
                               'acorr_breusch_godfrey',
                               msg)


def het_breuschpagan(resid, exog_het):
    '''Breusch-Pagan Lagrange Multiplier test for heteroscedasticity

    The tests the hypothesis that the residual variance does not depend on
    the variables in x in the form

    :math: \sigma_i = \\sigma * f(\\alpha_0 + \\alpha z_i)

    Homoscedasticity implies that $\\alpha=0$


    Parameters
    ----------
    resid : arraylike, (nobs,)
        For the Breusch-Pagan test, this should be the residual of a regression.
        If an array is given in exog, then the residuals are calculated by
        the an OLS regression or resid on exog. In this case resid should
        contain the dependent variable. Exog can be the same as x.
        TODO: I dropped the exog option, should I add it back?
    exog_het : array_like, (nobs, nvars)
        This contains variables that might create data dependent
        heteroscedasticity.

    Returns
    -------
    lm : float
        lagrange multiplier statistic
    lm_pvalue :float
        p-value of lagrange multiplier test
    fvalue : float
        f-statistic of the hypothesis that the error variance does not depend
        on x
    f_pvalue : float
        p-value for the f-statistic

    Notes
    -----
    Assumes x contains constant (for counting dof and calculation of R^2).
    In the general description of LM test, Greene mentions that this test
    exaggerates the significance of results in small or moderately large
    samples. In this case the F-statistic is preferrable.

    *Verification*

    Chisquare test statistic is exactly (<1e-13) the same result as bptest
    in R-stats with defaults (studentize=True).

    Implementation
    This is calculated using the generic formula for LM test using $R^2$
    (Greene, section 17.6) and not with the explicit formula
    (Greene, section 11.4.3).
    The degrees of freedom for the p-value assume x is full rank.

    References
    ----------
    http://en.wikipedia.org/wiki/Breusch%E2%80%93Pagan_test
    Greene 5th edition
    Breusch, Pagan article

    '''

    x = np.asarray(exog_het)
    y = np.asarray(resid)**2
    nobs, nvars = x.shape
    resols = OLS(y, x).fit()
    fval = resols.fvalue
    fpval = resols.f_pvalue
    lm = nobs * resols.rsquared
    # Note: degrees of freedom for LM test is nvars minus constant
    return lm, stats.chi2.sf(lm, nvars-1), fval, fpval

het_breushpagan = np.deprecate(het_breuschpagan, 'het_breushpagan', 'het_breuschpagan',
                               "Use het_breuschpagan, het_breushpagan will be "
                               "removed in 0.9 \n(Note: misspelling missing 'c')")

def het_white(resid, exog, retres=False):
    '''White's Lagrange Multiplier Test for Heteroscedasticity

    Parameters
    ----------
    resid : array_like
        residuals, square of it is used as endogenous variable
    exog : array_like
        possible explanatory variables for variance, squares and interaction
        terms are included in the auxilliary regression.
    resstore : instance (optional)
        a class instance that holds intermediate results. Only returned if
        store=True

    Returns
    -------
    lm : float
        lagrange multiplier statistic
    lm_pvalue :float
        p-value of lagrange multiplier test
    fvalue : float
        f-statistic of the hypothesis that the error variance does not depend
        on x. This is an alternative test variant not the original LM test.
    f_pvalue : float
        p-value for the f-statistic

    Notes
    -----
    assumes x contains constant (for counting dof)

    question: does f-statistic make sense? constant ?

    References
    ----------
    Greene section 11.4.1 5th edition p. 222
    now test statistic reproduces Greene 5th, example 11.3

    '''
    x = np.asarray(exog)
    y = np.asarray(resid)
    if x.ndim == 1:
        raise ValueError('x should have constant and at least one more variable')
    nobs, nvars0 = x.shape
    i0,i1 = np.triu_indices(nvars0)
    exog = x[:,i0]*x[:,i1]
    nobs, nvars = exog.shape
    assert nvars == nvars0*(nvars0-1)/2. + nvars0
    resols = OLS(y**2, exog).fit()
    fval = resols.fvalue
    fpval = resols.f_pvalue
    lm = nobs * resols.rsquared
    # Note: degrees of freedom for LM test is nvars minus constant
    #degrees of freedom take possible reduced rank in exog into account
    #df_model checks the rank to determine df
    #extra calculation that can be removed:
    assert resols.df_model == np_matrix_rank(exog) - 1
    lmpval = stats.chi2.sf(lm, resols.df_model)
    return lm, lmpval, fval, fpval

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

def linear_harvey_collier(res):
    '''Harvey Collier test for linearity

    The Null hypothesis is that the regression is correctly modeled as linear.

    Parameters
    ----------
    res : Result instance

    Returns
    -------
    tvalue : float
        test statistic, based on ttest_1sample
    pvalue : float
        pvalue of the test

    Notes
    -----
    TODO: add sort_by option

    This test is a t-test that the mean of the recursive ols residuals is zero.
    Calculating the recursive residuals might take some time for large samples.

    '''
    #I think this has different ddof than
    #B.H. Baltagi, Econometrics, 2011, chapter 8
    #but it matches Gretl and R:lmtest, pvalue at decimal=13
    rr = recursive_olsresiduals(res, skip=3, alpha=0.95)
    from scipy import stats

    return stats.ttest_1samp(rr[3][3:], 0)

def linear_rainbow(res, frac = 0.5):
    '''Rainbow test for linearity

    The Null hypothesis is that the regression is correctly modelled as linear.
    The alternative for which the power might be large are convex, check

    Parameters
    ----------
    res : Result instance

    Returns
    -------
    fstat : float
        test statistic based of F test
    pvalue : float
        pvalue of the test

    '''

    nobs = res.nobs
    endog = res.model.endog
    exog = res.model.exog
    lowidx = np.ceil(0.5 * (1 - frac) * nobs).astype(int)
    uppidx = np.floor(lowidx + frac * nobs).astype(int)
    mi_sl = slice(lowidx, uppidx)
    res_mi = OLS(endog[mi_sl], exog[mi_sl]).fit()
    nobs_mi = res_mi.model.endog.shape[0]
    ss_mi = res_mi.ssr
    ss = res.ssr
    fstat = (ss - ss_mi) / (nobs-nobs_mi) / ss_mi  * res_mi.df_resid
    from scipy import stats
    pval = stats.f.sf(fstat, nobs - nobs_mi, res_mi.df_resid)
    return fstat, pval

def linear_lm(resid, exog, func=None):
    '''Lagrange multiplier test for linearity against functional alternative

    limitations: Assumes currently that the first column is integer.
    Currently it doesn't check whether the transformed variables contain NaNs,
    for example log of negative number.

    Parameters
    ----------
    resid : ndarray
        residuals of a regression
    exog : ndarray
        exogenous variables for which linearity is tested
    func : callable
        If func is None, then squares are used. func needs to take an array
        of exog and return an array of transformed variables.

    Returns
    -------
    lm : float
       Lagrange multiplier test statistic
    lm_pval : float
       p-value of Lagrange multiplier tes
    ftest : ContrastResult instance
       the results from the F test variant of this test

    Notes
    -----
    written to match Gretl's linearity test.
    The test runs an auxilliary regression of the residuals on the combined
    original and transformed regressors.
    The Null hypothesis is that the linear specification is correct.

    '''
    from scipy import stats

    if func is None:
        func = lambda x: np.power(x, 2)

    exog_aux = np.column_stack((exog, func(exog[:,1:])))

    nobs, k_vars = exog.shape
    ls = OLS(resid, exog_aux).fit()
    ftest = ls.f_test(np.eye(k_vars - 1, k_vars * 2 - 1, k_vars))
    lm = nobs * ls.rsquared
    lm_pval = stats.chi2.sf(lm, k_vars - 1)
    return lm, lm_pval, ftest

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


def breaks_hansen(olsresults):
    '''test for model stability, breaks in parameters for ols, Hansen 1992

    Parameters
    ----------
    olsresults : instance of RegressionResults
        uses only endog and exog

    Returns
    -------
    teststat : float
        Hansen's test statistic
    crit : structured array
        critical values at alpha=0.95 for different nvars
    pvalue Not yet
    ft, s : arrays
        temporary return for debugging, will be removed

    Notes
    -----
    looks good in example, maybe not very powerful for small changes in
    parameters

    According to Greene, distribution of test statistics depends on nvar but
    not on nobs.

    Test statistic is verified against R:strucchange

    References
    ----------
    Greene section 7.5.1, notation follows Greene

    '''
    y = olsresults.model.endog
    x = olsresults.model.exog
    resid = olsresults.resid
    nobs, nvars = x.shape
    resid2 = resid**2
    ft = np.c_[x*resid[:,None], (resid2 - resid2.mean())]
    s = ft.cumsum(0)
    assert (np.abs(s[-1]) < 1e10).all()  #can be optimized away
    F = nobs*(ft[:,:,None]*ft[:,None,:]).sum(0)
    S = (s[:,:,None]*s[:,None,:]).sum(0)
    H = np.trace(np.dot(np.linalg.inv(F), S))
    crit95 = np.array([(2,1.9),(6,3.75),(15,3.75),(19,4.52)],
                      dtype = [('nobs',int), ('crit', float)])
    #TODO: get critical values from Bruce Hansens' 1992 paper
    return H, crit95, ft, s

def breaks_cusumolsresid(olsresidual, ddof=0):
    '''cusum test for parameter stability based on ols residuals

    Parameters
    ----------
    olsresiduals : ndarray
        array of residuals from an OLS estimation
    ddof : int
        number of parameters in the OLS estimation, used as degrees of freedom
        correction for error variance.

    Returns
    -------
    sup_b : float
        test statistic, maximum of absolute value of scaled cumulative OLS
        residuals
    pval : float
        Probability of observing the data under the null hypothesis of no
        structural change, based on asymptotic distribution which is a Brownian
        Bridge
    crit: list
        tabulated critical values, for alpha = 1%, 5% and 10%

    Notes
    -----
    tested agains R:strucchange

    Not clear: Assumption 2 in Ploberger, Kramer assumes that exog x have
    asymptotically zero mean, x.mean(0) = [1, 0, 0, ..., 0]
    Is this really necessary? I don't see how it can affect the test statistic
    under the null. It does make a difference under the alternative.
    Also, the asymptotic distribution of test statistic depends on this.

    From examples it looks like there is little power for standard cusum if
    exog (other than constant) have mean zero.

    References
    ----------
    Ploberger, Werner, and Walter Kramer. “The Cusum Test with Ols Residuals.”
    Econometrica 60, no. 2 (March 1992): 271-285.

    '''
    resid = olsresidual.ravel()
    nobs = len(resid)
    nobssigma2 = (resid**2).sum()
    if ddof > 0:
        #print('ddof', ddof, 1. / (nobs - ddof) * nobs
        nobssigma2 = nobssigma2 / (nobs - ddof) * nobs
    #B is asymptotically a Brownian Bridge
    B = resid.cumsum()/np.sqrt(nobssigma2) # use T*sigma directly
    sup_b = np.abs(B).max() #asymptotically distributed as standard Brownian Bridge
    crit = [(1,1.63), (5, 1.36), (10, 1.22)]
    #Note stats.kstwobign.isf(0.1) is distribution of sup.abs of Brownian Bridge
    #>>> stats.kstwobign.isf([0.01,0.05,0.1])
    #array([ 1.62762361,  1.35809864,  1.22384787])
    pval = stats.kstwobign.sf(sup_b)
    return sup_b, pval, crit

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


