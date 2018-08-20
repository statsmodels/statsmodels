# -*- coding: utf-8 -*-
"""Assorted Statistical Tests

Author: josef-pktd
License: BSD-3

Notes
-----
Almost fully verified against R or Gretl, not all options are the same.
In many cases of Lagrange multiplier tests both the LM test and the F test is
returned. In some but not all cases, R has the option to choose the test
statistic. Some alternative test statistic results have not been verified.

TODO:
* refactor to store intermediate results
* how easy is it to attach a test that is a class to a result instance,
  for example CompareCox as a method compare_cox(self, other) ?
* StatTestMC has been moved and should be deleted

MISSING:
* pvalues for breaks_hansen
* additional options, compare with R, check where ddof is appropriate
* new tests:
  - breaks_ap, more recent breaks tests
  - specification tests against nonparametric alternatives
"""
from __future__ import print_function

import numpy as np
from scipy import stats

from statsmodels.compat.python import iteritems, long

# collect some imports of verified (at least one example) functions
from statsmodels.sandbox.stats.diagnostic import (  # noqa:F841
    CompareCox, CompareJ,
    compare_cox, compare_j, HetGoldfeldQuandt,
    het_goldfeldquandt,
    recursive_olsresiduals
    )

from ._lilliefors import (kstest_fit, lilliefors,  # noqa:F841
                          lillifors, kstest_normal,
                          kstest_exponential)  # lillifors is deprecated
from ._adnorm import normal_ad  # noqa:F841


class ResultsStore(object):
    def __str__(self):
        return self._str


def unitroot_adf(x, maxlag=None, trendorder=0, autolag='AIC', store=False):
    # Wrap the newer implementation to retain the older signature so the
    # example files continue to work.
    from statsmodels.tsa.stattools import adfuller
    return adfuller(x, maxlag=maxlag, regression=trendorder, autolag=autolag,
                    store=store, regresults=False)


# ----------------------------------------------------------------------
# Tests for Homoscedasticity/Heteroscedasticity

def het_white(resid, exog, retres=False):
    """
    White's Lagrange Multiplier Test for Heteroscedasticity

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
    Assumes x contains constant (for counting dof)

    Question: does f-statistic make sense? constant ?  # TODO: answer this

    References
    ----------
    Greene section 11.4.1 5th edition p. 222
    now test statistic reproduces Greene 5th, example 11.3
    """
    from statsmodels.regression.linear_model import OLS

    x = np.asarray(exog)
    y = np.asarray(resid)
    if x.ndim == 1:
        raise ValueError("`x` input should be at least 2-dimensional")

    nobs, nvars0 = x.shape
    i0, i1 = np.triu_indices(nvars0)
    exog = x[:, i0] * x[:, i1]
    nobs, nvars = exog.shape
    assert nvars == nvars0 * (nvars0 - 1) / 2. + nvars0

    resols = OLS(y**2, exog).fit()
    fval = resols.fvalue
    fpval = resols.f_pvalue
    lm = nobs * resols.rsquared
    # Note: degrees of freedom for LM test is nvars minus constant
    # degrees of freedom take possible reduced rank in exog into account
    # df_model checks the rank to determine df
    # extra calculation that can be removed:
    assert resols.df_model == np.linalg.matrix_rank(exog) - 1

    lmpval = stats.chi2.sf(lm, resols.df_model)
    return lm, lmpval, fval, fpval


def het_breuschpagan(resid, exog_het):
    '''Breusch-Pagan Lagrange Multiplier test for heteroscedasticity

    The tests the hypothesis that the residual variance does not depend on
    the variables in x in the form

    :math: \sigma_i = \\sigma * f(\\alpha_0 + \\alpha z_i)

    Homoscedasticity implies that $\\alpha=0$


    Parameters
    ----------
    resid : array-like
        For the Breusch-Pagan test, this should be the residual of a regression.
        If an array is given in exog, then the residuals are calculated by
        the an OLS regression or resid on exog. In this case resid should
        contain the dependent variable. Exog can be the same as x.
        TODO: I dropped the exog option, should I add it back?
    exog_het : array_like
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
    from statsmodels.regression.linear_model import OLS

    x = np.asarray(exog_het)
    y = np.asarray(resid)**2
    nobs, nvars = x.shape
    resols = OLS(y, x).fit()
    fval = resols.fvalue
    fpval = resols.f_pvalue
    lm = nobs * resols.rsquared
    # Note: degrees of freedom for LM test is nvars minus constant
    return lm, stats.chi2.sf(lm, nvars-1), fval, fpval


het_breushpagan = np.deprecate(het_breuschpagan,
                               "het_breushpagan", "het_breuschpagan",
                               "Use het_breuschpagan, het_breushpagan will be "
                               "removed in 0.9 \n"
                               "(Note: misspelling missing 'c')")


# ----------------------------------------------------------------------
# Tests for No-Autocorrelation

def acorr_ljungbox(x, lags=None, boxpierce=False):
    """
    Ljung-Box test for no autocorrelation

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
        If lags is None, then the default maxlag is 'min((nobs // 2 - 2), 40)'
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

    *Verification*

    Looks correctly sized in Monte Carlo studies.
    not yet compared to verified values

    Examples
    --------
    see example script

    References
    ----------
    Greene
    Wikipedia
    """
    from statsmodels.tsa.stattools import acf

    x = np.asarray(x)
    nobs = x.shape[0]
    if lags is None:
        lags = np.arange(1, min((nobs // 2 - 2), 40) + 1)
    elif isinstance(lags, (int, long)):
        lags = np.arange(1, lags + 1)
    lags = np.asarray(lags)
    maxlag = max(lags)
    acfx = acf(x, nlags=maxlag) # normalize by nobs not (nobs-nlags)
                                # SS: unbiased=False is default now
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
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tsa.tsatools import lagmat

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
    resid : ndarray
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
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tsa.tsatools import lagmat

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

# ----------------------------------------------------------------------
# Structural Break/Parameter Stability Tests


def breaks_cusumolsresid(olsresidual, ddof=0):
    """
    cusum test for parameter stability based on ols residuals

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
    Tested against R:strucchange

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
    """
    resid = olsresidual.ravel()
    nobs = len(resid)
    nobssigma2 = (resid**2).sum()
    if ddof > 0:
        nobssigma2 = nobssigma2 / (nobs - ddof) * nobs

    B = resid.cumsum() / np.sqrt(nobssigma2)  # use T*sigma directly
    # B is asymptotically a Brownian Bridge

    sup_b = np.abs(B).max()
    # sup_b is asymptotically distributed as standard Brownian Bridge

    crit = [(1, 1.63), (5, 1.36), (10, 1.22)]
    # NOTE: stats.kstwobign.isf(0.1) is distribution
    # of sup.abs of Brownian Bridge
    # >>> stats.kstwobign.isf([0.01, 0.05, 0.1])
    # array([ 1.62762361,  1.35809864,  1.22384787])
    pval = stats.kstwobign.sf(sup_b)
    return sup_b, pval, crit


def breaks_hansen(olsresults):
    """
    Test for model stability, breaks in parameters for ols, Hansen 1992

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
    Looks good in example, maybe not very powerful for small changes in
    parameters

    According to Greene, distribution of test statistics depends on nvar but
    not on nobs.

    Test statistic is verified against R:strucchange

    References
    ----------
    Greene section 7.5.1, notation follows Greene
    """
    y = olsresults.model.endog
    x = olsresults.model.exog
    resid = olsresults.resid
    nobs, nvars = x.shape
    resid2 = resid**2
    ft = np.c_[x * resid[:, None], (resid2 - resid2.mean())]
    s = ft.cumsum(0)
    assert (np.abs(s[-1]) < 1e10).all()  # can be optimized away
    F = nobs * (ft[:, :, None] * ft[:, None, :]).sum(0)
    S = (s[:, :, None] * s[:, None, :]).sum(0)
    H = np.trace(np.dot(np.linalg.inv(F), S))
    crit95 = np.array([(2, 1.9), (6, 3.75), (15, 3.75), (19, 4.52)],
                      dtype=[('nobs', int), ('crit', float)])
    # TODO: get critical values from Bruce Hansens' 1992 paper
    return H, crit95, ft, s


# ----------------------------------------------------------------------
# Tests for Linearity/Functional Form

def linear_harvey_collier(res):
    """
    Harvey Collier test for linearity

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
    """
    # I think this has different ddof than
    # B.H. Baltagi, Econometrics, 2011, chapter 8
    # but it matches Gretl and R:lmtest, pvalue at decimal=13
    rr = recursive_olsresiduals(res, skip=3, alpha=0.95)

    return stats.ttest_1samp(rr[3][3:], 0)


def linear_rainbow(res, frac=0.5):
    """
    Rainbow test for linearity

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
    """
    from statsmodels.regression.linear_model import OLS

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

    fstat = (ss - ss_mi) / (nobs - nobs_mi) / ss_mi * res_mi.df_resid
    pval = stats.f.sf(fstat, nobs - nobs_mi, res_mi.df_resid)
    return fstat, pval


def linear_lm(resid, exog, func=None):
    """
    Lagrange multiplier test for linearity against functional alternative

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
    Written to match Gretl's linearity test.
    The test runs an auxilliary regression of the residuals on the combined
    original and transformed regressors.
    The Null hypothesis is that the linear specification is correct.
    """
    from statsmodels.regression.linear_model import OLS

    if func is None:
        func = lambda x: np.power(x, 2)

    exog_aux = np.column_stack((exog, func(exog[:, 1:])))

    nobs, k_vars = exog.shape
    ls = OLS(resid, exog_aux).fit()
    ftest = ls.f_test(np.eye(k_vars - 1, k_vars * 2 - 1, k_vars))
    lm = nobs * ls.rsquared
    lm_pval = stats.chi2.sf(lm, k_vars - 1)
    return lm, lm_pval, ftest
