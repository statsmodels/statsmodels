# -*- coding: utf-8 -*-
"""
Various Statistical Tests

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

missing:

* pvalues for breaks_hansen
* additional options, compare with R, check where ddof is appropriate
* new tests:
  - breaks_ap, more recent breaks tests
  - specification tests against nonparametric alternatives
"""
from statsmodels.compat.python import iteritems

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import acf, adfuller
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tools.validation import array_like, int_like, bool_like


# get the old signature back so the examples work
def unitroot_adf(x, maxlag=None, trendorder=0, autolag='AIC', store=False):
    return adfuller(x, maxlag=maxlag, regression=trendorder, autolag=autolag,
                    store=store, regresults=False)


# TODO: I like the bunch pattern for this too.
class ResultsStore(object):
    def __str__(self):
        return self._str


class CompareCox(object):
    """
    Cox Test for non-nested models

    Parameters
    ----------
    results_x : Result instance
        result instance of first model
    results_z : Result instance
        result instance of second model
    attach : bool


    Formulas from Greene, section 8.3.4 translated to code

    produces correct results for Example 8.3, Greene
    """

    def run(self, results_x, results_z, attach=True):
        """run Cox test for non-nested models

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

        """

        if not np.allclose(results_x.model.endog, results_z.model.endog):
            raise ValueError('endogenous variables in models are not the same')
        nobs = results_x.model.endog.shape[0]
        x = results_x.model.exog
        z = results_z.model.exog
        sigma2_x = results_x.ssr / nobs
        sigma2_z = results_z.ssr / nobs
        yhat_x = results_x.fittedvalues
        yhat_z = results_z.fittedvalues
        res_dx = OLS(yhat_x, z).fit()
        err_zx = res_dx.resid
        res_xzx = OLS(err_zx, x).fit()
        err_xzx = res_xzx.resid

        sigma2_zx = sigma2_x + np.dot(err_zx.T, err_zx) / nobs
        c01 = nobs / 2. * (np.log(sigma2_z) - np.log(sigma2_zx))
        v01 = sigma2_x * np.dot(err_xzx.T, err_xzx) / sigma2_zx ** 2
        q = c01 / np.sqrt(v01)
        pval = 2 * stats.norm.sf(np.abs(q))

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


class CompareJ(object):
    """J-Test for comparing non-nested models

    Parameters
    ----------
    results_x : Result instance
        result instance of first model
    results_z : Result instance
        result instance of second model
    attach : bool


    From description in Greene, section 8.3.3

    produces correct results for Example 8.3, Greene - not checked yet
    #currently an exception, but I do not have clean reload in python session

    check what results should be attached

    """

    def run(self, results_x, results_z, attach=True):
        """run J-test for non-nested models

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

        """
        if not np.allclose(results_x.model.endog, results_z.model.endog):
            raise ValueError('endogenous variables in models are not the same')
        nobs = results_x.model.endog.shape[0]
        y = results_x.model.endog
        x = results_x.model.exog
        z = results_z.model.exog
        # sigma2_x = results_x.ssr/nobs
        # sigma2_z = results_z.ssr/nobs
        yhat_x = results_x.fittedvalues
        # yhat_z = results_z.fittedvalues
        res_zx = OLS(y, np.column_stack((yhat_x, z))).fit()
        self.res_zx = res_zx  # for testing
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


def acorr_ljungbox(x, lags=None, boxpierce=False, model_df=0, period=None,
                   return_df=None):
    """
    Ljung-Box test of autocorrelation in residuals.

    Parameters
    ----------
    x : array_like
        The data series. The data is demeaned before the test statistic is
        computed.
    lags : {None, int, array_like}
        If lags is an integer then this is taken to be the largest lag
        that is included, the test result is reported for all smaller lag
        length. If lags is a list or array, then all lags are included up to
        the largest lag in the list, however only the tests for the lags in
        the list are reported. If lags is None, then the default maxlag is
        currently min((nobs // 2 - 2), 40). After 0.12 this will change to
        min(10, nobs // 5). The default number of lags changes if period
        is set.
    boxpierce : {False, True}
        If true, then additional to the results of the Ljung-Box test also the
        Box-Pierce test results are returned
    model_df : int
        Number of degrees of freedom consumed by the model. In an ARMA model,
        this value is usually p+q where p is the AR order and q is the MA
        order. This value is subtracted from the degrees-of-freedom used in
        the test so that the adjusted dof for the statistics are
        lags - model_df. If lags - model_df <= 0, then NaN is returned.
    period : {int, None}
        The period of a Seasonal time series.  Used to compute the max lag
        for seasonal data which uses min(2*period, nobs // 5) if set. If None,
        then the default rule is used to set the number of lags. When set, must
        be >= 2.
    return_df : bool
        Flag indicating whether to return the result as a single DataFrame
        with columns lb_stat, lb_pvalue, and optionally bp_stat and bp_pvalue.
        After 0.12, this will become the only return method.  Set to True
        to return the DataFrame or False to continue returning the 2 - 4
        output. If not set, a warning is raised.

    Returns
    -------
    lbvalue : float or array
        The Ljung-Box test statistic.
    pvalue : float or array
        The p-value based on chi-square distribution. The p-value is computed
        as 1.0 - chi2.cdf(lbvalue, dof) where dof is lag - model_df. If
        lag - model_df <= 0, then NaN is returned for the pvalue.
    bpvalue : (optional), float or array
        The test statistic for Box-Pierce test.
    bppvalue : (optional), float or array
        The p-value based for Box-Pierce test on chi-square distribution.
        The p-value is computed as 1.0 - chi2.cdf(bpvalue, dof) where dof is
        lag - model_df. If lag - model_df <= 0, then NaN is returned for the
        pvalue.

    Notes
    -----
    Ljung-Box and Box-Pierce statistic differ in their scaling of the
    autocorrelation function. Ljung-Box test is has better finite-sample
    properties.

    Examples
    --------
    >>> data = sm.datasets.sunspots.load_pandas().data
    >>> res = sm.tsa.ARMA(data['SUNACTIVITY'], (1,1)).fit(disp=-1)
    >>> sm.stats.acorr_ljungbox(res.resid, lags=[10], return_df=True)
           lb_stat     lb_pvalue
    10  214.106992  1.827374e-40

    References
    ----------
    .. [*] Green, W. "Econometric Analysis," 5th ed., Pearson, 2003.
    """
    x = array_like(x, 'x')
    period = int_like(period, 'period', optional=True)
    return_df = bool_like(return_df, 'return_df', optional=True)
    model_df = int_like(model_df, 'model_df', optional=False)
    if period is not None and period <= 1:
        raise ValueError('period must be >= 2')
    if model_df < 0:
        raise ValueError('model_df must be >= 0')
    nobs = x.shape[0]
    if period is not None:
        lags = np.arange(1, min(nobs // 5, 2 * period) + 1, dtype=np.int)
    elif lags is None:
        # TODO: Switch to min(10, nobs//5) after 0.12
        import warnings
        warnings.warn("The default value of lags is changing.  After 0.12, "
                      "this value will become min(10, nobs//5). Directly set"
                      "lags to silence this warning.", FutureWarning)
        # Future
        # lags = np.arange(1, min(nobs // 5, 10) + 1, dtype=np.int)
        lags = np.arange(1, min((nobs // 2 - 2), 40) + 1, dtype=np.int)
    elif not isinstance(lags, Iterable):
        lags = int_like(lags, 'lags')
        lags = np.arange(1, lags + 1)
    lags = array_like(lags, 'lags', dtype=np.int)
    maxlag = lags.max()
    # normalize by nobs not (nobs-nlags)
    # SS: unbiased=False is default now
    sacf = acf(x, nlags=maxlag, fft=False)
    sacf2 = sacf[1:maxlag + 1] ** 2 / (nobs - np.arange(1, maxlag + 1))
    qljungbox = nobs * (nobs + 2) * np.cumsum(sacf2)[lags - 1]
    adj_lags = lags - model_df
    pval = np.full_like(qljungbox, np.nan)
    loc = adj_lags > 0
    pval[loc] = stats.chi2.sf(qljungbox[loc], adj_lags[loc])

    if return_df is None:
        msg = ("The value returned will change to a single DataFrame after "
               "0.12 is released.  Set return_df to True to use to return a "
               "DataFrame now.  Set return_df to False to silence this "
               "warning.")
        import warnings
        warnings.warn(msg, FutureWarning)

    if not boxpierce:
        if return_df:
            return pd.DataFrame({"lb_stat": qljungbox, "lb_pvalue": pval},
                                index=lags)
        return qljungbox, pval

    qboxpierce = nobs * np.cumsum(sacf[1:maxlag + 1] ** 2)[lags - 1]
    pvalbp = np.full_like(qljungbox, np.nan)
    pvalbp[loc] = stats.chi2.sf(qboxpierce[loc], adj_lags[loc])
    if return_df:
        return pd.DataFrame({"lb_stat": qljungbox, "lb_pvalue": pval,
                             "bp_stat": qboxpierce, "bp_pvalue": pvalbp},
                            index=lags)

    return qljungbox, pval, qboxpierce, pvalbp


def acorr_lm(x, maxlag=None, autolag='AIC', store=False, regresults=False):
    """
    Lagrange Multiplier tests for autocorrelation

    This is a generic Lagrange Multiplier test for autocorrelation. I do not
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
    """

    if regresults:
        store = True

    x = np.asarray(x)
    nobs = x.shape[0]
    if maxlag is None:
        # for adf from Greene referencing Schwert 1989
        maxlag = int(np.ceil(12. * np.power(nobs / 100., 1 / 4.)))
        # nobs//4  #TODO: check default, or do AIC/BIC

    xdiff = np.diff(x)
    #
    xdall = lagmat(x[:, None], maxlag, trim='both')
    nobs = xdall.shape[0]
    xdall = np.c_[np.ones((nobs, 1)), xdall]
    xshort = x[-nobs:]

    if store:
        resstore = ResultsStore()

    if autolag:
        # search for lag length with highest information criteria
        # Note: I use the same number of observations to have comparable IC
        results = {}
        for mlag in range(1, maxlag + 1):
            results[mlag] = OLS(xshort, xdall[:, :mlag + 1]).fit()

        if autolag.lower() == 'aic':
            bestic, icbestlag = min((v.aic, k) for k, v in iteritems(results))
        elif autolag.lower() == 'bic':
            icbest, icbestlag = min((v.bic, k) for k, v in iteritems(results))
        else:
            raise ValueError("autolag can only be None, 'AIC' or 'BIC'")

        # rerun ols with best ic
        xdall = lagmat(x[:, None], icbestlag, trim='both')
        nobs = xdall.shape[0]
        xdall = np.c_[np.ones((nobs, 1)), xdall]
        xshort = x[-nobs:]
        usedlag = icbestlag
        if regresults:
            resstore.results = results
    else:
        usedlag = maxlag

    resols = OLS(xshort, xdall[:, :usedlag + 1]).fit()
    fval = resols.fvalue
    fpval = resols.f_pvalue
    lm = nobs * resols.rsquared
    lmpval = stats.chi2.sf(lm, usedlag)
    # Note: degrees of freedom for LM test is nvars minus constant = usedlags
    # return fval, fpval, lm, lmpval

    if store:
        resstore.resols = resols
        resstore.usedlag = usedlag
        return lm, lmpval, fval, fpval, resstore
    else:
        return lm, lmpval, fval, fpval


def het_arch(resid, maxlag=None, autolag=None, store=False, regresults=False,
             ddof=0):
    """
    Engle's Test for Autoregressive Conditional Heteroscedasticity (ARCH).

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
        of parameters that have been estimated, for example ddof=p+q for an
        ARMA(p,q) (need reference, based on discussion on R finance
        mailinglist)

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
    verified against R:FinTS::ArchTest
    """
    return acorr_lm(resid ** 2, maxlag=maxlag, autolag=autolag, store=store,
                    regresults=regresults)


def acorr_breusch_godfrey(results, nlags=None, store=False):
    """Breusch Godfrey Lagrange Multiplier tests for residual autocorrelation

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

    """

    x = np.asarray(results.resid).squeeze()
    if x.ndim != 1:
        raise ValueError('Model resid must be a 1d array. Cannot be used on'
                         ' multivariate models.')
    exog_old = results.model.exog
    nobs = x.shape[0]
    if nlags is None:
        # for adf from Greene referencing Schwert 1989
        nlags = np.trunc(12. * np.power(nobs / 100.,
                                        1 / 4.))  # nobs//4  #TODO: check default, or do AIC/BIC
        nlags = int(nlags)

    x = np.concatenate((np.zeros(nlags), x))

    # xdiff = np.diff(x)
    #
    xdall = lagmat(x[:, None], nlags, trim='both')
    nobs = xdall.shape[0]
    xdall = np.c_[np.ones((nobs, 1)), xdall]
    xshort = x[-nobs:]
    exog = np.column_stack((exog_old, xdall))
    k_vars = exog.shape[1]

    if store:
        resstore = ResultsStore()

    resols = OLS(xshort, exog).fit()
    ft = resols.f_test(np.eye(nlags, k_vars, k_vars - nlags))
    fval = ft.fvalue
    fpval = ft.pvalue
    fval = np.squeeze(fval)[()]  # TODO: fix this in ContrastResults
    fpval = np.squeeze(fpval)[()]
    lm = nobs * resols.rsquared
    lmpval = stats.chi2.sf(lm, nlags)
    # Note: degrees of freedom for LM test is nvars minus constant = usedlags
    # return fval, fpval, lm, lmpval

    if store:
        resstore.resols = resols
        resstore.usedlag = nlags
        return lm, lmpval, fval, fpval, resstore
    else:
        return lm, lmpval, fval, fpval


def het_breuschpagan(resid, exog_het):
    r"""Breusch-Pagan Lagrange Multiplier test for heteroscedasticity

    The tests the hypothesis that the residual variance does not depend on
    the variables in x in the form

    :math: \sigma_i = \sigma * f(\alpha_0 + \alpha z_i)

    Homoscedasticity implies that $\alpha=0$


    Parameters
    ----------
    resid : array_like
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
    samples. In this case the F-statistic is preferable.

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
    https://en.wikipedia.org/wiki/Breusch%E2%80%93Pagan_test
    Greene 5th edition
    Breusch, Pagan article

    """

    x = np.asarray(exog_het)
    y = np.asarray(resid) ** 2
    nobs, nvars = x.shape
    resols = OLS(y, x).fit()
    fval = resols.fvalue
    fpval = resols.f_pvalue
    lm = nobs * resols.rsquared
    # Note: degrees of freedom for LM test is nvars minus constant
    return lm, stats.chi2.sf(lm, nvars - 1), fval, fpval


def het_white(resid, exog, retres=False):
    """White's Lagrange Multiplier Test for Heteroscedasticity

    Parameters
    ----------
    resid : array_like
        residuals, square of it is used as endogenous variable
    exog : array_like
        possible explanatory variables for variance, squares and interaction
        terms are included in the auxiliary regression.
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

    """
    x = np.asarray(exog)
    y = np.asarray(resid)
    if x.ndim == 1:
        raise ValueError(
            'x should have constant and at least one more variable')
    nobs, nvars0 = x.shape
    i0, i1 = np.triu_indices(nvars0)
    exog = x[:, i0] * x[:, i1]
    nobs, nvars = exog.shape
    assert nvars == nvars0 * (nvars0 - 1) / 2. + nvars0
    resols = OLS(y ** 2, exog).fit()
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


def _het_goldfeldquandt2_old(y, x, idx, split=None, retres=False):
    """test whether variance is the same in 2 subsamples

    Parameters
    ----------
    y : array_like
        endogenous variable
    x : array_like
        exogenous variable, regressors
    idx : int
        column index of variable according to which observations are
        sorted for the split
    split : {None, int, float}
        If an int, the index at which sample is split.
        If a float 0<split<1 then split is interpreted as fraction of the
        observations in the first sample
    retres : bool
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
    """
    x = np.asarray(x)
    y = np.asarray(y)
    nobs, nvars = x.shape
    if split is None:
        split = nobs // 2
    elif (0 < split) and (split < 1):
        split = int(nobs * split)

    xsortind = np.argsort(x[:, idx])
    y = y[xsortind]
    x = x[xsortind, :]
    resols1 = OLS(y[:split], x[:split]).fit()
    resols2 = OLS(y[split:], x[split:]).fit()
    fval = resols1.mse_resid / resols2.mse_resid
    if fval > 1:
        fpval = stats.f.sf(fval, resols1.df_resid, resols2.df_resid)
        ordering = 'larger'
    else:
        fval = 1. / fval
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
        # res.__str__
        res._str = """The Goldfeld-Quandt test for null hypothesis that the
variance in the second subsample is %s than in the first subsample:
    F-statistic =%8.4f and p-value =%8.4f""" % (ordering, fval, fpval)

        return res
    else:
        return fval, fpval


class HetGoldfeldQuandt(object):
    """
    Test whether variance is the same in 2 subsamples

    Parameters
    ----------
    y : array_like
        endogenous variable
    x : array_like
        exogenous variable, regressors
    idx : int
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
    alternative : str, 'increasing', 'decreasing' or 'two-sided'
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
    """

    # TODO: can do Chow test for structural break in same way
    def run(self, y, x, idx=None, split=None, drop=None,
            alternative='increasing', attach=True):
        """see class docstring"""
        x = np.asarray(x)
        y = np.asarray(y)  # **2
        nobs, nvars = x.shape
        if split is None:
            split = nobs // 2
        elif (0 < split) and (split < 1):
            split = int(nobs * split)

        if drop is None:
            start2 = split
        elif (0 < drop) and (drop < 1):
            start2 = split + int(nobs * drop)
        else:
            start2 = split + drop

        if idx is not None:
            xsortind = np.argsort(x[:, idx])
            y = y[xsortind]
            x = x[xsortind, :]

        resols1 = OLS(y[:split], x[:split]).fit()
        resols2 = OLS(y[start2:], x[start2:]).fit()
        fval = resols2.mse_resid / resols1.mse_resid
        # if fval>1:
        if alternative.lower() in ['i', 'inc', 'increasing']:
            fpval = stats.f.sf(fval, resols1.df_resid, resols2.df_resid)
            ordering = 'increasing'
        elif alternative.lower() in ['d', 'dec', 'decreasing']:
            fval = fval
            fpval = stats.f.sf(1. / fval, resols2.df_resid, resols1.df_resid)
            ordering = 'decreasing'
        elif alternative.lower() in ['2', '2-sided', 'two-sided']:
            fpval_sm = stats.f.cdf(fval, resols2.df_resid, resols1.df_resid)
            fpval_la = stats.f.sf(fval, resols2.df_resid, resols1.df_resid)
            fpval = 2 * min(fpval_sm, fpval_la)
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
            # res.__str__
            # TODO: check if string works
            res._str = """The Goldfeld-Quandt test for null hypothesis that the
    variance in the second subsample is %s than in the first subsample:
        F-statistic =%8.4f and p-value =%8.4f""" % (ordering, fval, fpval)

        return fval, fpval, ordering
        # return self

    def __str__(self):
        try:
            return self._str
        except AttributeError:
            return repr(self)

    # TODO: missing the alternative option in call
    def __call__(self, y, x, idx=None, split=None, drop=None,
                 alternative='increasing'):
        return self.run(y, x, idx=idx, split=split, drop=drop, attach=False,
                        alternative=alternative)


het_goldfeldquandt = HetGoldfeldQuandt()


def linear_harvey_collier(res, order_by=None):
    """
    Harvey Collier test for linearity

    The Null hypothesis is that the regression is correctly modeled as linear.

    Parameters
    ----------
    res : Result instance
    order_by :

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


def linear_rainbow(res, frac=0.5, order_by=None, use_distance=False,
                   center=None):
    """
    Rainbow test for linearity

    The null hypothesis is the fit of the model using full sample is the same
    as using a central subset. The alternative is that the fits are difference.
    The rainbow test has power against many different forms of nonlinearity.

    Parameters
    ----------
    res : RegressionResults instance
        A results instance from a linear regresson.
    frac : float
        The fraction of the data to include in the center model.
    order_by : ndarray, str, List[str]
        If an ndarray, the values in the array are used to sort the
        observations.  If a string or a list of strings, these are interprted
        as column name(s) which are then used to lexigographically sort the
        data.
    use_distance : bool
        Flag indicating whether data should be ordered by the Mahalanobis
        distance to the center.
    center : float, int
        If a float, the center is center * nobs of the ordered data.  If an
        integer, must be in [0, nobs) and is interpreted as the observation
        of the ordered data to use.

    Returns
    -------
    fstat : float
        test statistic based of F test
    pvalue : float
        pvalue of the test

    Notes
    -----
    This test assumes residuals are homoskedastic and may reject a correct
    linear specification if the residuals are heteroskedastic.
    """

    nobs = res.nobs
    endog = res.model.endog
    exog = res.model.exog
    # TODO: Add centering here
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
    Currently it does not check whether the transformed variables contain NaNs,
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
    The test runs an auxiliary regression of the residuals on the combined
    original and transformed regressors.
    The Null hypothesis is that the linear specification is correct.
    """

    if func is None:
        func = lambda x: np.power(x, 2)

    exog_aux = np.column_stack((exog, func(exog[:, 1:])))

    nobs, k_vars = exog.shape
    ls = OLS(resid, exog_aux).fit()
    ftest = ls.f_test(np.eye(k_vars - 1, k_vars * 2 - 1, k_vars))
    lm = nobs * ls.rsquared
    lm_pval = stats.chi2.sf(lm, k_vars - 1)
    return lm, lm_pval, ftest


def spec_white(resid, exog):
    """
    White's Two-Moment Specification Test

    Parameters
    ----------
    resid : array_like
        OLS residuals
    exog : array_like
        OLS design matrix

    Returns
    -------
    stat : float
        test statistic
    pval : float
        chi-square p-value for test statistic
    dof : int
        degrees of freedom

    Notes
    -----
    Implements the two-moment specification test described by White's
    Theorem 2 (1980, p. 823) which compares the standard OLS covariance
    estimator with White's heteroscedasticity-consistent estimator. The
    test statistic is shown to be chi-square distributed.

    Null hypothesis is homoscedastic and correctly specified.

    Assumes the OLS design matrix contains an intercept term and at least
    one variable. The intercept is removed to calculate the test statistic.

    Interaction terms (squares and crosses of OLS regressors) are added to
    the design matrix to calculate the test statistic.

    Degrees-of-freedom (full rank) = nvar + nvar * (nvar + 1) / 2

    Linearly dependent columns are removed to avoid singular matrix error.

    Reference
    ---------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
    estimator and a direct test for heteroscedasticity. Econometrica,
    48: 817-838.

    See also het_white.
    """
    x = np.asarray(exog)
    e = np.asarray(resid)
    if x.ndim == 1:
        raise ValueError('X should have a constant and at least one variable')

    # add interaction terms
    i0, i1 = np.triu_indices(x.shape[1])
    exog = np.delete(x[:, i0] * x[:, i1], 0, 1)

    # collinearity check - see _fit_collinear
    atol = 1e-14
    rtol = 1e-13
    tol = atol + rtol * exog.var(0)
    r = np.linalg.qr(exog, mode='r')
    mask = np.abs(r.diagonal()) < np.sqrt(tol)
    exog = exog[:, np.where(~mask)[0]]

    # calculate test statistic
    sqe = e * e
    sqmndevs = sqe - np.mean(sqe)
    d = np.dot(exog.T, sqmndevs)
    devx = exog - np.mean(exog, axis=0)
    devx *= sqmndevs[:, None]
    b = devx.T.dot(devx)
    stat = d.dot(np.linalg.solve(b, d))

    # chi-square test
    dof = devx.shape[1]
    pval = stats.chi2.sf(stat, dof)
    return stat, pval, dof


def _neweywestcov(resid, x):
    """
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
    """
    nobs = resid.shape[0]  # TODO: check this can only be 1d
    nlags = int(round(4 * (nobs / 100.) ** (2 / 9.)))
    hhat = resid * x.T
    xuux = np.dot(hhat, hhat.T)
    for lag in range(nlags):
        za = np.dot(hhat[:, lag:nobs], hhat[:, :nobs - lag].T)
        w = 1 - lag / (nobs + 1.)
        xuux = xuux + np.dot(w, za + za.T)
    xtxi = np.linalg.inv(np.dot(x.T, x))  # QR instead?
    covbNW = np.dot(xtxi, np.dot(xuux, xtxi))

    return covbNW


def _recursive_olsresiduals2(olsresults, skip):
    """this is my original version based on Greene and references

    keep for now for comparison and benchmarking
    """
    y = olsresults.model.endog
    x = olsresults.model.exog
    nobs, nvars = x.shape
    rparams = np.nan * np.zeros((nobs, nvars))
    rresid = np.nan * np.zeros((nobs))
    rypred = np.nan * np.zeros((nobs))
    rvarraw = np.nan * np.zeros((nobs))

    # XTX = np.zeros((nvars,nvars))
    # XTY = np.zeros((nvars))

    x0 = x[:skip]
    y0 = y[:skip]
    XTX = np.dot(x0.T, x0)
    XTY = np.dot(x0.T, y0)  # xi * y   #np.dot(xi, y)
    beta = np.linalg.solve(XTX, XTY)
    rparams[skip - 1] = beta
    yipred = np.dot(x[skip - 1], beta)
    rypred[skip - 1] = yipred
    rresid[skip - 1] = y[skip - 1] - yipred
    rvarraw[skip - 1] = 1 + np.dot(x[skip - 1],
                                   np.dot(np.linalg.inv(XTX), x[skip - 1]))
    for i in range(skip, nobs):
        xi = x[i:i + 1, :]
        yi = y[i]
        xxT = np.dot(xi.T, xi)  # xi is 2d 1 row
        xy = (xi * yi).ravel()  # XTY is 1d  #np.dot(xi, yi)   #np.dot(xi, y)
        print(xy.shape, XTY.shape)
        print(XTX)
        print(XTY)
        beta = np.linalg.solve(XTX, XTY)
        rparams[i - 1] = beta  # this is beta based on info up to t-1
        yipred = np.dot(xi, beta)
        rypred[i] = yipred
        rresid[i] = yi - yipred
        rvarraw[i] = 1 + np.dot(xi, np.dot(np.linalg.inv(XTX), xi.T))
        XTX += xxT
        XTY += xy

    i = nobs
    beta = np.linalg.solve(XTX, XTY)
    rparams[i - 1] = beta

    rresid_scaled = rresid / np.sqrt(
        rvarraw)  # this is N(0,sigma2) distributed
    nrr = nobs - skip
    sigma2 = rresid_scaled[skip - 1:].var(ddof=1)
    rresid_standardized = rresid_scaled / np.sqrt(sigma2)  # N(0,1) distributed
    rcusum = rresid_standardized[skip - 1:].cumsum()
    # confidence interval points in Greene p136 looks strange?
    # this assumes sum of independent standard normal
    # rcusumci = np.sqrt(np.arange(skip,nobs+1))*np.array([[-1.],[+1.]])*stats.norm.sf(0.025)
    a = 1.143  # for alpha=0.99  =0.948 for alpha=0.95
    # following taken from Ploberger,
    crit = a * np.sqrt(nrr)
    rcusumci = (a * np.sqrt(nrr) + a * np.arange(0, nobs - skip) / np.sqrt(
        nrr)) \
               * np.array([[-1.], [+1.]])
    return (rresid, rparams, rypred, rresid_standardized, rresid_scaled,
            rcusum, rcusumci)


def recursive_olsresiduals(olsresults, skip=None, lamda=0.0, alpha=0.95):
    """
    Calculate recursive ols with residuals and Cusum test statistic

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

    """

    y = olsresults.model.endog
    x = olsresults.model.exog
    nobs, nvars = x.shape
    if skip is None:
        skip = nvars
    rparams = np.nan * np.zeros((nobs, nvars))
    rresid = np.nan * np.zeros((nobs))
    rypred = np.nan * np.zeros((nobs))
    rvarraw = np.nan * np.zeros((nobs))

    # intialize with skip observations
    x0 = x[:skip]
    y0 = y[:skip]
    # add Ridge to start (not in jplv
    XTXi = np.linalg.inv(np.dot(x0.T, x0) + lamda * np.eye(nvars))
    XTY = np.dot(x0.T, y0)  # xi * y   #np.dot(xi, y)
    # beta = np.linalg.solve(XTX, XTY)
    beta = np.dot(XTXi, XTY)
    # print('beta', beta
    rparams[skip - 1] = beta
    yipred = np.dot(x[skip - 1], beta)
    rypred[skip - 1] = yipred
    rresid[skip - 1] = y[skip - 1] - yipred
    rvarraw[skip - 1] = 1 + np.dot(x[skip - 1], np.dot(XTXi, x[skip - 1]))
    for i in range(skip, nobs):
        xi = x[i:i + 1, :]
        yi = y[i]
        # xxT = np.dot(xi.T, xi)  #xi is 2d 1 row
        xy = (xi * yi).ravel()  # XTY is 1d  #np.dot(xi, yi)   #np.dot(xi, y)
        # print(xy.shape, XTY.shape
        # print(XTX
        # print(XTY

        # get prediction error with previous beta
        yipred = np.dot(xi, beta)
        rypred[i] = yipred
        residi = yi - yipred
        rresid[i] = residi

        # update beta and inverse(X'X)
        tmp = np.dot(XTXi, xi.T)
        ft = 1 + np.dot(xi, tmp)

        XTXi = XTXi - np.dot(tmp, tmp.T) / ft  # BigJudge equ 5.5.15

        # print('beta', beta
        beta = beta + (tmp * residi / ft).ravel()  # BigJudge equ 5.5.14
        #        #version for testing
        #        XTY += xy
        #        beta = np.dot(XTXi, XTY)
        #        print((tmp*yipred / ft).shape
        #        print('tmp.shape, ft.shape, beta.shape', tmp.shape, ft.shape, beta.shape
        rparams[i] = beta
        rvarraw[i] = ft

    i = nobs
    # beta = np.linalg.solve(XTX, XTY)
    # rparams[i] = beta

    rresid_scaled = rresid / np.sqrt(
        rvarraw)  # this is N(0,sigma2) distributed
    nrr = nobs - skip
    # sigma2 = rresid_scaled[skip-1:].var(ddof=1)  #var or sum of squares ?
    # Greene has var, jplv and Ploberger have sum of squares (Ass.:mean=0)
    # Gretl uses: by reverse engineering matching their numbers
    sigma2 = rresid_scaled[skip:].var(ddof=1)
    rresid_standardized = rresid_scaled / np.sqrt(sigma2)  # N(0,1) distributed
    rcusum = rresid_standardized[skip - 1:].cumsum()
    # confidence interval points in Greene p136 looks strange. Cleared up
    # this assumes sum of independent standard normal, which does not take into
    # account that we make many tests at the same time
    # rcusumci = np.sqrt(np.arange(skip,nobs+1))*np.array([[-1.],[+1.]])*stats.norm.sf(0.025)
    if alpha == 0.95:
        a = 0.948  # for alpha=0.95
    elif alpha == 0.99:
        a = 1.143  # for alpha=0.99
    elif alpha == 0.90:
        a = 0.850
    else:
        raise ValueError('alpha can only be 0.9, 0.95 or 0.99')

    # following taken from Ploberger,
    crit = a * np.sqrt(nrr)
    rcusumci = (a * np.sqrt(nrr) + 2 * a * np.arange(0, nobs - skip) / np.sqrt(
        nrr)) \
               * np.array([[-1.], [+1.]])
    return (rresid, rparams, rypred, rresid_standardized, rresid_scaled,
            rcusum, rcusumci)


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
    looks good in example, maybe not very powerful for small changes in
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
    resid2 = resid ** 2
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


def breaks_cusumolsresid(olsresidual, ddof=0):
    """
    Cusum test for parameter stability based on ols residuals

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
    tested against R:strucchange

    Not clear: Assumption 2 in Ploberger, Kramer assumes that exog x have
    asymptotically zero mean, x.mean(0) = [1, 0, 0, ..., 0]
    Is this really necessary? I do not see how it can affect the test statistic
    under the null. It does make a difference under the alternative.
    Also, the asymptotic distribution of test statistic depends on this.

    From examples it looks like there is little power for standard cusum if
    exog (other than constant) have mean zero.

    References
    ----------
    Ploberger, Werner, and Walter Kramer. “The Cusum Test with OLS Residuals.”
    Econometrica 60, no. 2 (March 1992): 271-285.
    """
    resid = olsresidual.ravel()
    nobs = len(resid)
    nobssigma2 = (resid ** 2).sum()
    if ddof > 0:
        # print('ddof', ddof, 1. / (nobs - ddof) * nobs
        nobssigma2 = nobssigma2 / (nobs - ddof) * nobs
    # B is asymptotically a Brownian Bridge
    B = resid.cumsum() / np.sqrt(nobssigma2)  # use T*sigma directly
    sup_b = np.abs(
        B).max()  # asymptotically distributed as standard Brownian Bridge
    crit = [(1, 1.63), (5, 1.36), (10, 1.22)]
    # Note stats.kstwobign.isf(0.1) is distribution of sup.abs of Brownian Bridge
    # >>> stats.kstwobign.isf([0.01,0.05,0.1])
    # array([ 1.62762361,  1.35809864,  1.22384787])
    pval = stats.kstwobign.sf(sup_b)
    return sup_b, pval, crit


# def breaks_cusum(recolsresid):
#    """renormalized cusum test for parameter stability based on recursive residuals
#
#
#    still incorrect: in PK, the normalization for sigma is by T not T-K
#    also the test statistic is asymptotically a Wiener Process, Brownian motion
#    not Brownian Bridge
#    for testing: result reject should be identical as in standard cusum version
#
#    References
#    ----------
#    Ploberger, Werner, and Walter Kramer. “The Cusum Test with OLS Residuals.”
#    Econometrica 60, no. 2 (March 1992): 271-285.
#
#    """
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
    """supLM, expLM and aveLM by Andrews, and Andrews,Ploberger

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

    """
    pass


if __name__ == '__main__':

    examples = ['adf']
    if 'adf' in examples:

        x = np.random.randn(20)
        print(acorr_ljungbox(x, 4, return_df=True))
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
