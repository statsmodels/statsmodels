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
# TODO: Check all input
from statsmodels.compat.pandas import deprecate_kwarg
from statsmodels.compat.python import iteritems

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from statsmodels.stats.contrast import ContrastResults
from statsmodels.regression.linear_model import OLS, RegressionResultsWrapper
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tools.validation import (array_like, int_like, bool_like,
                                          string_like, dict_like)
from statsmodels.stats._lilliefors import (kstest_fit, lilliefors,
                                           kstest_normal, kstest_exponential)
from statsmodels.stats._adnorm import normal_ad, anderson_statistic

__all__ = ["kstest_fit", "lilliefors", "kstest_normal", "kstest_exponential",
           "normal_ad", "compare_cox", "compare_j", "acorr_breusch_godfrey",
           "acorr_ljungbox", "acorr_lm", "het_arch", "het_breuschpagan",
           "het_goldfeldquandt", "het_white", "spec_white", "linear_lm",
           "linear_rainbow", "linear_harvey_collier", "anderson_statistic"]


def check_nested(small, large):
    """
    Check if a larger exog nests a smaller exog

    Parameters
    ----------
    small : ndarray
        exog from smaller model
    large : ndarray
        exog from larger model

    Returns
    -------
    bool
        True if small is nested by large
    """

    if small.shape[1] > large.shape[1]:
        return False
    coef = np.linalg.lstsq(large, small, rcond=None)[0]
    err = small - large @ coef
    return np.linalg.matrix_rank(np.c_[large, err]) == large.shape[1]


class ResultsStore(object):
    def __str__(self):
        try:
            return self._str
        except AttributeError:
            return self.__class__.__name__


def compare_cox(results_x, results_z, store=False):
    """
    Compute the Cox test for non-nested models

    Parameters
    ----------
    results_x : Result instance
        result instance of first model
    results_z : Result instance
        result instance of second model
    store : bool
        If true, then the intermediate results are returned.

    Returns
    -------
    tstat : float
        t statistic for the test that including the fitted values of the
        first model in the second model has no effect.
    pvalue : float
        two-sided pvalue for the t statistic
    res_store : ResultsStore, optional
        Intermediate results. Returned if store is True.

    Notes
    -----
    Tests of non-nested hypothesis might not provide unambiguous answers.
    The test should be performed in both directions and it is possible
    that both or neither test rejects. see [1]_ for more information.

    Formulas from [1]_, section 8.3.4 translated to code

    Matches results for Example 8.3 in Greene

    References
    ----------
    .. [1] Greene, W. H. Econometric Analysis. New Jersey. Prentice Hall;
       5th edition. (2002).
    """
    if not np.allclose(results_x.model.endog, results_z.model.endog):
        raise ValueError('endogenous variables in models are not the same')
    nobs = results_x.model.endog.shape[0]
    x = results_x.model.exog
    z = results_z.model.exog

    nested = False
    if x.shape[1] <= z.shape[1]:
        nested = nested or check_nested(x, z)
    else:
        nested = nested or check_nested(z, x)
    if nested:
        raise ValueError("The exog in results_x and in results_z are nested."
                         "Cox comparison requires that models are non-nested.")
    sigma2_x = results_x.ssr / nobs
    sigma2_z = results_z.ssr / nobs
    yhat_x = results_x.fittedvalues
    res_dx = OLS(yhat_x, z).fit()
    err_zx = res_dx.resid
    res_xzx = OLS(err_zx, x).fit()
    err_xzx = res_xzx.resid

    sigma2_zx = sigma2_x + np.dot(err_zx.T, err_zx) / nobs
    c01 = nobs / 2. * (np.log(sigma2_z) - np.log(sigma2_zx))
    v01 = sigma2_x * np.dot(err_xzx.T, err_xzx) / sigma2_zx ** 2
    q = c01 / np.sqrt(v01)
    pval = 2 * stats.norm.sf(np.abs(q))

    if store:
        res = ResultsStore()
        res.res_dx = res_dx
        res.res_xzx = res_xzx
        res.c01 = c01
        res.v01 = v01
        res.q = q
        res.pvalue = pval
        res.dist = stats.norm
        return q, pval, res

    return q, pval


class CompareCox(object):
    """
    Cox Test for non-nested models

    .. deprecated::

       CompareCox is deprecated in favor of compare_cox.
       CompareCox will be removed after 0.12.
    """

    def __init__(self):
        import warnings
        warnings.warn("CompareCox is deprecated in favor of compare_cox and "
                      "will be removed after 0.12.", FutureWarning)

    def run(self, results_x, results_z, attach=False):
        results = compare_cox(results_x, results_z, store=attach)
        if attach:
            res = results[-1]
            self.res_dx = res.res_dx
            self.res_xzx = res.res_xzx
            self.c01 = res.c01
            self.v01 = res.v01
            self.q = res.q
            self.pvalue = res.pvalue
            self.dist = res.dist

        return results

    def __call__(self, results_x, results_z):
        return self.run(results_x, results_z, attach=False)


def compare_j(results_x, results_z, store=False):
    """
    Compute the J-test for non-nested models

    Parameters
    ----------
    results_x : Result instance
        result instance of first model
    results_z : Result instance
        result instance of second model
    store : bool
        If true, then the intermediate results are returned.

    Returns
    -------
    tstat : float
        t statistic for the test that including the fitted values of the
        first model in the second model has no effect.
    pvalue : float
        two-sided pvalue for the t statistic
    res_store : ResultsStore, optional
        Intermediate results. Returned if store is True.

    Notes
    -----
    From description in Greene, section 8.3.3. Matches results for Example
    8.3, Greene.

    Tests of non-nested hypothesis might not provide unambiguous answers.
    The test should be performed in both directions and it is possible
    that both or neither test rejects. see Greene for more information.

    References
    ----------
    .. [1] Greene, W. H. Econometric Analysis. New Jersey. Prentice Hall;
       5th edition. (2002).
    """
    # TODO: Allow cov to be specified
    if not np.allclose(results_x.model.endog, results_z.model.endog):
        raise ValueError('endogenous variables in models are not the same')
    y = results_x.model.endog
    x = results_x.model.exog
    z = results_z.model.exog
    nested = False
    if x.shape[1] <= z.shape[1]:
        nested = nested or check_nested(x, z)
    else:
        nested = nested or check_nested(z, x)
    if nested:
        raise ValueError("The exog in results_x and in results_z are nested."
                         "J comparison requires that models are non-nested.")
    yhat_x = results_x.fittedvalues
    res_zx = OLS(y, np.column_stack((yhat_x, z))).fit()
    tstat = res_zx.tvalues[0]
    pval = res_zx.pvalues[0]
    if store:
        res = ResultsStore()
        res.res_zx = res_zx
        res.dist = stats.t(res_zx.df_resid)
        res.teststat = tstat
        res.pvalue = pval
        return tstat, pval, res

    return tstat, pval


class CompareJ(object):
    """
    J-Test for comparing non-nested models

    .. deprecated::

       CompareJ is deprecated in favor of compare_j.
       CompareJ will be removed after 0.12.
    """

    def __init__(self):
        import warnings
        warnings.warn("CompareJ is deprecated in favor of compare_j and will "
                      "be removed after 0.12.", FutureWarning)

    def run(self, results_x, results_z, attach=True):
        res = compare_j(results_x, results_z, store=attach)
        tstat, pval = res[:2]
        if attach:
            self.res_zx = res[-1].res_zx
            self.dist = res[-1].dist
            self.teststat = res[-1].teststat
            self.pvalue = res[-1].pvalue

        return tstat, pval

    def __call__(self, results_x, results_z):
        return self.run(results_x, results_z, attach=False)


def compare_encompassing(results_x, results_z, cov_type="nonrobust",
                         cov_kwargs=None):
    r"""
    Davidson-MacKinnon encompassing test for comparing non-nested models

    Parameters
    ----------
    results_x : Result instance
        result instance of first model
    results_z : Result instance
        result instance of second model
    cov_type : str
        Covariance type. The default is 'nonrobust` which uses the classic
        OLS covariance estimator. Specify one of 'HC0', 'HC1', 'HC2', 'HC3'
        to use White's covariance estimator. All covariance types supported
        by ``OLS.fit`` are accepted.
    cov_kwargs : dict
        Dictionary of covariance options passed to ``OLS.fit``. See OLS.fit
        for more details.

    Returns
    -------
    DataFrame
        A DataFrame with two rows and four columns. The row labeled x
        contains results for the null that the model contained in
        results_x is equivalent to the encompassing model. The results in
        the row labeled z correspond to the test that the model contained
        in results_z are equivalent to the encompassing model. The columns
        are the test statistic, its p-value, and the numerator and
        denominator degrees of freedom. The test statistic has an F
        distribution. The numerator degree of freedom is the number of
        variables in the encompassing model that are not in the x or z model.
        The denominator degree of freedom is the number of observations minus
        the number of variables in the nesting model.

    Notes
    -----
    The null is that the fit produced using x is the same as the fit
    produced using both x and z. When testing whether x is encompassed,
    the model estimated is

    .. math::

        Y = X\beta + Z_1\gamma + \epsilon

    where :math:`Z_1` are the columns of :math:`Z` that are not spanned by
    :math:`X`. The null is :math:`H_0:\gamma=0`. When testing whether z is
    encompassed, the roles of :math:`X` and :math:`Z` are reversed.

    Implementation of  Davidson and MacKinnon (1993)'s encompassing test.
    Performs two Wald tests where models x and z are compared to a model
    that nests the two. The Wald tests are performed by using an OLS
    regression.
    """
    if not isinstance(results_x, RegressionResultsWrapper):
        raise TypeError("results_x must come from a linear regression model")
    if not isinstance(results_z, RegressionResultsWrapper):
        raise TypeError("results_z must come from a linear regression model")
    y = results_x.model.endog
    x = results_x.model.exog
    z = results_z.model.exog

    def _test_nested(endog, a, b, cov_type, cov_kwargs):
        err = b - a @ np.linalg.lstsq(a, b, rcond=None)[0]
        u, s, v = np.linalg.svd(err)
        eps = np.finfo(np.double).eps
        tol = s.max(axis=-1, keepdims=True) * max(err.shape) * eps
        non_zero = np.abs(s) > tol
        if check_nested(b, a):
            raise RuntimeError("Models are nested. Test requires non-nested "
                               "models.")
        aug = err @ v[:, non_zero]
        aug_reg = np.hstack([a, aug])
        k_a = aug.shape[1]
        k = aug_reg.shape[1]

        res = OLS(endog, aug_reg).fit(cov_type=cov_type, cov_kwds=cov_kwargs)
        r_matrix = np.zeros((k_a, k))
        r_matrix[:, -k_a:] = np.eye(k_a)
        test = res.wald_test(r_matrix, use_f=True)
        stat, pvalue = float(np.squeeze(test.statistic)), float(test.pvalue)
        df_num, df_denom = int(test.df_num), int(test.df_denom)
        return stat, pvalue, df_num, df_denom

    x_nested = _test_nested(y, x, z, cov_type, cov_kwargs)
    z_nested = _test_nested(y, z, x, cov_type, cov_kwargs)
    return pd.DataFrame([x_nested, z_nested],
                        index=['x', 'z'],
                        columns=['stat', 'pvalue', 'df_num', 'df_denom'])


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
        Box-Pierce test results are returned.
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

    See Also
    --------
    statsmodels.regression.linear_model.OLS.fit
        Regression model fitting.
    statsmodels.regression.linear_model.RegressionResults
        Results from linear regression models.

    Notes
    -----
    Ljung-Box and Box-Pierce statistic differ in their scaling of the
    autocorrelation function. Ljung-Box test is has better finite-sample
    properties.

    References
    ----------
    .. [*] Green, W. "Econometric Analysis," 5th ed., Pearson, 2003.

    Examples
    --------
    >>> data = sm.datasets.sunspots.load_pandas().data
    >>> res = sm.tsa.ARMA(data['SUNACTIVITY'], (1,1)).fit(disp=-1)
    >>> sm.stats.acorr_ljungbox(res.resid, lags=[10], return_df=True)
           lb_stat     lb_pvalue
    10  214.106992  1.827374e-40
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


@deprecate_kwarg('maxlag', 'nlag')
def acorr_lm(resid, nlag=None, autolag='AIC', store=False, *, period=None,
             ddof=0, cov_type='nonrobust', cov_kwargs=None):
    """
    Lagrange Multiplier tests for autocorrelation.

    This is a generic Lagrange Multiplier test for autocorrelation. Returns
    Engle's ARCH test if resid is the squared residual array. Breusch-Godfrey
    is a variation on this test with additional exogenous variables.

    Parameters
    ----------
    resid : array_like
        Time series to test.
    nlag : int
        Highest lag to use. The behavior of this parameter will change
        after 0.12.
    autolag : {None, str}
        If None, then a fixed number of lags given by maxlag is used. This
        parameter is deprecated and will be removed after 0.12.  Searching
        for model specification cannot control test siez.
    store : bool
        If true then the intermediate results are also returned.
    period : {int, None}
        The period of a Seasonal time series.  Used to compute the max lag
        for seasonal data which uses min(2*period, nobs // 5) if set. If None,
        then the default rule is used to set the number of lags. When set, must
        be >= 2.
    ddof : int
        The number of degrees of freedom consumed by the model used to
        produce resid. The default value is 0.
    cov_type : str
        Covariance type. The default is 'nonrobust` which uses the classic
        OLS covariance estimator. Specify one of 'HC0', 'HC1', 'HC2', 'HC3'
        to use White's covariance estimator. All covariance types supported
        by ``OLS.fit`` are accepted.
    cov_kwargs : dict
        Dictionary of covariance options passed to ``OLS.fit``. See OLS.fit for
        more details.

    Returns
    -------
    lm : float
        Lagrange multiplier test statistic.
    lmpval : float
        The p-value for Lagrange multiplier test.
    fval : float
        The f statistic of the F test, alternative version of the same
        test based on F test for the parameter restriction.
    fpval : float
        The pvalue of the F test.
    res_store : ResultsStore, optional
        Intermediate results. Only returned if store=True.

    See Also
    --------
    het_arch
        Conditional heteroskedasticity testing.
    acorr_breusch_godfrey
        Breusch-Godfrey test for serial correlation.
    acorr_ljung_box
        Ljung-Box test for serial correlation.

    Notes
    -----
    The test statistic is computed as (nobs - ddof) * r2 where r2 is the
    R-squared from a regression on the residual on nlag lags of the
    residual.
    """
    resid = array_like(resid, "resid", ndim=1)
    cov_type = string_like(cov_type, 'cov_type')
    cov_kwargs = {} if cov_kwargs is None else cov_kwargs
    cov_kwargs = dict_like(cov_kwargs, 'cov_kwargs')
    nobs = resid.shape[0]
    if period is not None and nlag is None:
        maxlag = min(nobs // 5, 2 * period)
    elif nlag is None:
        # TODO: Switch to min(10, nobs//5) after 0.12
        import warnings
        warnings.warn("The default value of lags is changing.  After 0.12, "
                      "this value will become min(10, nobs//5). Directly set"
                      "maxlags or period to silence this warning.",
                      FutureWarning)
        # Future
        # maxlag = min(nobs // 5, 10)
        # Old: for adf from Greene referencing Schwert 1989
        maxlag = int(np.ceil(12. * np.power(nobs / 100., 1 / 4.)))
    else:
        maxlag = nlag

    xdall = lagmat(resid[:, None], maxlag, trim='both')
    nobs = xdall.shape[0]
    xdall = np.c_[np.ones((nobs, 1)), xdall]
    xshort = resid[-nobs:]

    if store:
        res_store = ResultsStore()

    if autolag:
        # TODO: Deprecate this
        #   Use same rules as autocorr
        # search for lag length with highest information criteria
        # Note: I use the same number of observations to have comparable IC
        import warnings
        warnings.warn("autolag is deprecated and will be removed after 0.12. "
                      "Model selection before testing fails to control test"
                      "size. Set autolag to False to silence this warning")
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
        xdall = lagmat(resid[:, None], icbestlag, trim='both')
        nobs = xdall.shape[0]
        xdall = np.c_[np.ones((nobs, 1)), xdall]
        xshort = resid[-nobs:]
        usedlag = icbestlag
        if store:
            res_store.results = results
    else:
        usedlag = maxlag

    resols = OLS(xshort, xdall[:, :usedlag + 1]).fit(cov_type=cov_type,
                                                     cov_kwargs=cov_kwargs)
    fval = resols.fvalue
    fpval = resols.f_pvalue
    if cov_type == "nonrobust":
        lm = (nobs - ddof) * resols.rsquared
        lmpval = stats.chi2.sf(lm, usedlag)
        # Note: degrees of freedom for LM test is nvars minus constant = usedlags
    else:
        r_matrix = np.hstack((np.ones((usedlag,1)), np.eye(usedlag)))
        test_stat = resols.wald_test(r_matrix, use_f=False)
        lm = test_stat.fvalue
        lmpval = test_stat.pvalue

    if store:
        res_store.resols = resols
        res_store.usedlag = usedlag
        return lm, lmpval, fval, fpval, res_store
    else:
        return lm, lmpval, fval, fpval


def het_arch(resid, maxlag=None, autolag=None, store=False, ddof=0):
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
        If the residuals are from a regression, or ARMA estimation, then there
        are recommendations to correct the degrees of freedom by the number
        of parameters that have been estimated, for example ddof=p+q for an
        ARMA(p,q).

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
    res_store : ResultsStore, optional
        Intermediate results. Returned if store is True.

    Notes
    -----
    verified against R:FinTS::ArchTest
    """
    return acorr_lm(resid ** 2, maxlag=maxlag, autolag=autolag, store=store,
                    ddof=ddof)


def acorr_breusch_godfrey(results, nlags=None, store=False):
    """
    Breusch-Godfrey Lagrange Multiplier tests for residual autocorrelation.

    Parameters
    ----------
    results : Result instance
        Estimation results for which the residuals are tested for serial
        correlation.
    nlags : int
        Number of lags to include in the auxiliary regression. (nlags is
        highest lag).
    store : bool
        If store is true, then an additional class instance that contains
        intermediate results is returned.

    Returns
    -------
    lm : float
        Lagrange multiplier test statistic.
    lmpval : float
        The p-value for Lagrange multiplier test.
    fval : float
        The value of the f statistic for F test, alternative version of the
        same test based on F test for the parameter restriction.
    fpval : float
        The pvalue for F test.
    res_store : ResultsStore
        A class instance that holds intermediate results. Only returned if
        store=True.

    Notes
    -----
    BG adds lags of residual to exog in the design matrix for the auxiliary
    regression with residuals as endog. See [1]_, section 12.7.1.

    References
    ----------
    .. [1] Greene, W. H. Econometric Analysis. New Jersey. Prentice Hall;
      5th edition. (2002).
    """

    x = np.asarray(results.resid).squeeze()
    if x.ndim != 1:
        raise ValueError('Model resid must be a 1d array. Cannot be used on'
                         ' multivariate models.')
    exog_old = results.model.exog
    nobs = x.shape[0]
    if nlags is None:
        # for adf from Greene referencing Schwert 1989
        # nobs//4  #TODO: check default, or do AIC/BIC
        nlags = np.trunc(12. * np.power(nobs / 100., 1 / 4.))
        nlags = int(nlags)

    x = np.concatenate((np.zeros(nlags), x))

    xdall = lagmat(x[:, None], nlags, trim='both')
    nobs = xdall.shape[0]
    xdall = np.c_[np.ones((nobs, 1)), xdall]
    xshort = x[-nobs:]
    exog = np.column_stack((exog_old, xdall))
    k_vars = exog.shape[1]

    resols = OLS(xshort, exog).fit()
    ft = resols.f_test(np.eye(nlags, k_vars, k_vars - nlags))
    fval = ft.fvalue
    fpval = ft.pvalue
    fval = np.squeeze(fval)[()]  # TODO: fix this in ContrastResults
    fpval = np.squeeze(fpval)[()]
    lm = nobs * resols.rsquared
    lmpval = stats.chi2.sf(lm, nlags)
    # Note: degrees of freedom for LM test is nvars minus constant = usedlags

    if store:
        res_store = ResultsStore()
        res_store.resols = resols
        res_store.usedlag = nlags
        return lm, lmpval, fval, fpval, res_store
    else:
        return lm, lmpval, fval, fpval


def het_breuschpagan(resid, exog_het):
    r"""
    Breusch-Pagan Lagrange Multiplier test for heteroscedasticity

    The tests the hypothesis that the residual variance does not depend on
    the variables in x in the form

    :math: \sigma_i = \sigma * f(\alpha_0 + \alpha z_i)

    Homoscedasticity implies that :math:`\alpha=0`.

    Parameters
    ----------
    resid : array_like
        For the Breusch-Pagan test, this should be the residual of a
        regression. If an array is given in exog, then the residuals are
        calculated by the an OLS regression or resid on exog. In this case
        resid should contain the dependent variable. Exog can be the same as x.
    exog_het : array_like
        This contains variables suspected of being related to
        heteroscedasticity in resid.

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

    **Verification**

    Chisquare test statistic is exactly (<1e-13) the same result as bptest
    in R-stats with defaults (studentize=True).

    **Implementation**

    This is calculated using the generic formula for LM test using $R^2$
    (Greene, section 17.6) and not with the explicit formula
    (Greene, section 11.4.3).
    The degrees of freedom for the p-value assume x is full rank.

    References
    ----------
    .. [1] Greene, W. H. Econometric Analysis. New Jersey. Prentice Hall;
       5th edition. (2002).
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


def het_white(resid, exog):
    """
    White's Lagrange Multiplier Test for Heteroscedasticity

    Parameters
    ----------
    resid : array_like
        residuals, square of it is used as endogenous variable
    exog : array_like
        possible explanatory variables for variance, squares and interaction
        terms are included in the auxiliary regression.

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
    Assumes x contains constant (for counting dof).

    question: does f-statistic make sense? constant ?

    References
    ----------
    Greene section 11.4.1 5th edition p. 222. Test statistic reproduces
    Greene 5th, example 11.3.
    """
    x = np.asarray(exog)
    y = np.asarray(resid)
    if x.ndim == 1:
        raise ValueError('x should have constant and at least one more'
                         'variable')
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


def het_goldfeldquandt(y, x, idx=None, split=None, drop=None,
                       alternative='increasing', store=False):
    """
    Goldfeld-Quandt homoskedasticity test.

    This test examines whether the residual variance is the same in 2
    subsamples.

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
        If an integer, this is the index at which sample is split.
        If a float in 0<split<1 then split is interpreted as fraction
        of the observations in the first sample. If None, uses nobs//2.
    drop : {None, int, float}
        If this is not None, then observation are dropped from the middle
        part of the sorted series. If 0<split<1 then split is interpreted
        as fraction of the number of observations to be dropped.
        Note: Currently, observations are dropped between split and
        split+drop, where split and drop are the indices (given by rounding
        if specified as fraction). The first sample is [0:split], the
        second sample is [split+drop:]
    alternative : {'increasing', 'decreasing', 'two-sided'}
        The default is increasing. This specifies the alternative for the
        p-value calculation.
    store : bool
        Flag indicating to return the regression results

    Returns
    -------
    fval : float
        value of the F-statistic
    pval : float
        p-value of the hypothesis that the variance in one subsample is
        larger than in the other subsample
    ordering : str
        The ordering used in the alternative.
    res_store : ResultsStore, optional
        Storage for the intermediate and final results that are calculated

    Notes
    -----
    The Null hypothesis is that the variance in the two sub-samples are the
    same. The alternative hypothesis, can be increasing, i.e. the variance
    in the second sample is larger than in the first, or decreasing or
    two-sided.

    Results are identical R, but the drop option is defined differently.
    (sorting by idx not tested yet)
    """
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

    if store:
        res = ResultsStore()
        res.__doc__ = 'Test Results for Goldfeld-Quandt test of' \
                      'heterogeneity'
        res.fval = fval
        res.fpval = fpval
        res.df_fval = (resols2.df_resid, resols1.df_resid)
        res.resols1 = resols1
        res.resols2 = resols2
        res.ordering = ordering
        res.split = split
        res._str = """\
The Goldfeld-Quandt test for null hypothesis that the variance in the second
subsample is %s than in the first subsample:
F-statistic =%8.4f and p-value =%8.4f""" % (ordering, fval, fpval)

        return fval, fpval, ordering, res

    return fval, fpval, ordering


class HetGoldfeldQuandt(object):
    """
    Test whether variance is the same in 2 subsamples

    .. deprecated::

       HetGoldfeldQuandt is deprecated in favor of het_goldfeldquandt.
       HetGoldfeldQuandt will be removed after 0.12.

    See Also
    --------
    het_goldfeldquant
        Goldfeld-Quant heteroskedasticity test.
    """

    def __init__(self):
        import warnings
        warnings.warn("HetGoldfeldQuandt is deprecated in favor of"
                      "het_goldfeldquandt. HetGoldfeldQuandt will be removed"
                      "after 0.12.",
                      FutureWarning)

    def run(self, y, x, idx=None, split=None, drop=None,
            alternative='increasing', attach=True):
        """
        .. deprecated::

           Use het_goldfeldquant instead.

        See Also
        --------
        het_goldfeldquant
            Goldfeld-Quant heteroskedasticity test.
        """
        res = het_goldfeldquandt(y, x, idx=idx, split=split, drop=drop,
                                 alternative=alternative, store=attach)
        if attach:
            store = res[-1]
            self.__doc__ = store.__doc__
            self.fval = store.fval
            self.fpval = store.fpval
            self.df_fval = store.df_fval
            self.resols1 = store.resols1
            self.resols2 = store.resols2
            self.ordering = store.ordering
            self.split = store.split
            self._str = store._str

        return res[:3]

    def __call__(self, y, x, idx=None, split=None, drop=None,
                 alternative='increasing'):
        return self.run(y, x, idx=idx, split=split, drop=drop,
                        attach=False, alternative=alternative)


def linear_reset(result, power=3, test_type="fitted", use_f=False,
                 cov_type="nonrobust", cov_kwargs=None):
    """
    Ramsey's RESET test for neglected nonlinearity

    Parameters
    ----------
    result : Result instance
    power : {int, List[int]}
        The maximum power to include in the model, if an integer. Includes
        powers 2, 3, ..., power. If an list of integers, includes all powers
        in the list.
    test_type : str
        The type of augmentation to use:

        * "fitted" : (default) Augment regressors with powers of fitted values.
        * "exog" : Augment exog with powers of exog. Excludes binary
          regressors.
        * "princomp": Augment exog with powers of first principal component of
          exog.
    use_f : bool
    cov_type : str
    cov_kwargs : {dict, None}

    Returns
    -------
    ContrastResults
        Test results for Ramsey's Reset test. See notes for implementation
        details.

    Notes
    -----
    """
    if not isinstance(result, RegressionResultsWrapper):
        raise TypeError("result must come from a linear regression model")
    test_type = string_like(test_type, "test_type",
                            options=("fitted", "exog", "princomp"))
    cov_kwargs = dict_like(cov_kwargs, "cov_kwargs", optional=True)
    use_f = bool_like(use_f, "use_f")
    if isinstance(power, int):
        if power < 2:
            raise ValueError("power must be >= 2")
        power = np.arange(2, power + 1, dtype=np.int)
    else:
        try:
            power = np.array(power, dtype=np.int)
        except Exception:
            raise ValueError("power must be an integer or list of integers")
        if power.ndim != 1 or len(set(power)) != power.shape[0] or \
                (power < 2).any():
            raise ValueError("power must contains distinct integers all >= 2")
    exog = result.model.exog
    if test_type == "fitted":
        aug = result.fittedvalues[:, None]
    elif test_type == "exog":
        # Remove constant and binary
        aug = result.model.exog
        non_min_max = ((exog != exog.max(1)) & (exog != exog.min(1)))
        non_binary = non_min_max.any(axis=1)
        if not non_binary.any():
            raise ValueError("Model contains only constant or binary data")
        aug = aug[:, non_binary]
    else:
        from statsmodels.multivariate.pca import PCA
        aug = exog
        pca = PCA(aug, ncomp=1, standardize=False, demean=result.k_constant,
                  method="nipls")
        aug = pca.factors[:, :1]
    aug_exog = np.hstack([exog] + [aug ** p for p in power])
    mod_class = result.model.__class__
    mod = mod_class(result.data.endog, aug_exog)
    cov_kwargs = {} if cov_kwargs is None else cov_kwargs
    res = mod.fit(cov_type=cov_type, cov_kwargs=cov_kwargs)
    nrestr = aug_exog.shape[1] - exog.shape[1]
    nparams = aug_exog.shape[1]
    r_mat = np.c_[np.zeros(nrestr, nparams - nrestr), np.eye(nrestr)]
    test = res.wald_test(r_mat)
    if use_f:
        test = ContrastResults(F=test.stat, df_num=nrestr,
                               df_denom=exog.shape[0] - aug_exog.shape[1])
    return test


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

    # TODO: Remove the restriction
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
        def func(x):
            return np.power(x, 2)

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
        Confidence level of test, currently only two values supported,
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
    rresid = np.nan * np.zeros(nobs)
    rypred = np.nan * np.zeros(nobs)
    rvarraw = np.nan * np.zeros(nobs)

    # intialize with skip observations
    x0 = x[:skip]
    y0 = y[:skip]
    # add Ridge to start (not in jplv
    xtxi = np.linalg.inv(np.dot(x0.T, x0) + lamda * np.eye(nvars))
    xty = np.dot(x0.T, y0)  # xi * y   #np.dot(xi, y)
    beta = np.dot(xtxi, xty)
    rparams[skip - 1] = beta
    yipred = np.dot(x[skip - 1], beta)
    rypred[skip - 1] = yipred
    rresid[skip - 1] = y[skip - 1] - yipred
    rvarraw[skip - 1] = 1 + np.dot(x[skip - 1], np.dot(xtxi, x[skip - 1]))
    for i in range(skip, nobs):
        xi = x[i:i + 1, :]
        yi = y[i]

        # get prediction error with previous beta
        yipred = np.dot(xi, beta)
        rypred[i] = yipred
        residi = yi - yipred
        rresid[i] = residi

        # update beta and inverse(X'X)
        tmp = np.dot(xtxi, xi.T)
        ft = 1 + np.dot(xi, tmp)

        xtxi = xtxi - np.dot(tmp, tmp.T) / ft  # BigJudge equ 5.5.15

        beta = beta + (tmp * residi / ft).ravel()  # BigJudge equ 5.5.14
        rparams[i] = beta
        rvarraw[i] = ft

    rresid_scaled = rresid / np.sqrt(rvarraw)  # N(0,sigma2) distributed
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
    if alpha == 0.95:
        a = 0.948  # for alpha=0.95
    elif alpha == 0.99:
        a = 1.143  # for alpha=0.99
    elif alpha == 0.90:
        a = 0.850
    else:
        raise ValueError('alpha can only be 0.9, 0.95 or 0.99')

    # following taken from Ploberger,
    # crit = a * np.sqrt(nrr)
    rcusumci = (a * np.sqrt(nrr) + 2 * a * np.arange(0, nobs - skip) / np.sqrt(
        nrr)) * np.array([[-1.], [+1.]])
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
    x = olsresults.model.exog
    resid = olsresults.resid
    nobs, nvars = x.shape
    resid2 = resid ** 2
    ft = np.c_[x * resid[:, None], (resid2 - resid2.mean())]
    score = ft.cumsum(0)
    f = nobs * (ft[:, :, None] * ft[:, None, :]).sum(0)
    s = (score[:, :, None] * score[:, None, :]).sum(0)
    h = np.trace(np.dot(np.linalg.inv(f), s))
    crit95 = np.array([(2, 1.9), (6, 3.75), (15, 3.75), (19, 4.52)],
                      dtype=[('nobs', int), ('crit', float)])
    # TODO: get critical values from Bruce Hansens' 1992 paper
    return h, crit95


def breaks_cusumolsresid(resid, ddof=0):
    """
    Cusum test for parameter stability based on ols residuals

    Parameters
    ----------
    resid : ndarray
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
    resid = resid.ravel()
    nobs = len(resid)
    nobssigma2 = (resid ** 2).sum()
    if ddof > 0:
        nobssigma2 = nobssigma2 / (nobs - ddof) * nobs
    # b is asymptotically a Brownian Bridge
    b = resid.cumsum() / np.sqrt(nobssigma2)  # use T*sigma directly
    # asymptotically distributed as standard Brownian Bridge
    sup_b = np.abs(b).max()
    crit = [(1, 1.63), (5, 1.36), (10, 1.22)]
    # Note stats.kstwobign.isf(0.1) is distribution of sup.abs of Brownian
    # Bridge
    # >>> stats.kstwobign.isf([0.01,0.05,0.1])
    # array([ 1.62762361,  1.35809864,  1.22384787])
    pval = stats.kstwobign.sf(sup_b)
    return sup_b, pval, crit

# def breaks_cusum(recolsresid):
#    """renormalized cusum test for parameter stability based on recursive
#    residuals
#
#
#    still incorrect: in PK, the normalization for sigma is by T not T-K
#    also the test statistic is asymptotically a Wiener Process, Brownian
#    motion
#    not Brownian Bridge
#    for testing: result reject should be identical as in standard cusum
#    version
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
#    #Note stats.kstwobign.isf(0.1) is distribution of sup.abs of Brownian
#    Bridge
#    #>>> stats.kstwobign.isf([0.01,0.05,0.1])
#    #array([ 1.62762361,  1.35809864,  1.22384787])
#    pval = stats.kstwobign.sf(sup_b)
#    return sup_b, pval, crit
