:orphan:

.. _diagnostics:

Regression Diagnostics and Specification Tests
==============================================


Introduction
------------

In many cases of statistical analysis, we are not sure whether our statistical
model is correctly specified. For example when using ols, then linearity and
homoscedasticity are assumed, some test statistics additionally assume that
the errors are normally distributed or that we have a large sample.
Since our results depend on these statistical assumptions, the results are
only correct of our assumptions hold (at least approximately).

One solution to the problem of uncertainty about the correct specification is
to use robust methods, for example robust regression or robust covariance
(sandwich) estimators. The second approach is to test whether our sample is
consistent with these assumptions.

The following briefly summarizes specification and diagnostics tests for
linear regression.

Note: Not all statistical tests in the sandbox are fully tested, and the API
will still change. Some of the tests are still on the wishlist.

Heteroscedasticity Tests
------------------------

For these test the null hypothesis is that all observations have the same
error variance, i.e. errors are homoscedastic. The tests differ in which kind
of heteroscedasticity is considered as alternative hypothesis. They also vary
in the power of the test for different types of heteroscedasticity.

:py:func:`het_breushpagan <statsmodels.stats.diagnostic.het_breushpagan>`
    Lagrange Multiplier Heteroscedasticity Test by Breush-Pagan

:py:func:`het_white <statsmodels.stats.diagnostic.het_white>`
    Lagrange Multiplier Heteroscedasticity Test by White

:py:func:`het_goldfeldquandt <statsmodels.stats.diagnostic.het_goldfeldquandt>`
    test whether variance is the same in 2 subsamples


Autocorrelation Tests
---------------------

This group of test whether the regression residuals are not autocorrelated.
They assume that observations are ordered by time.

:py:func:`durbin_watson <statsmodels.stats.diagnostic.durbin_watson>`
  - Durbin-Watson test for no autocorrelation of residuals
  - printed with summary()

:py:func:`acorr_ljungbox <statsmodels.stats.diagnostic.acorr_ljungbox>`
  - Ljung-Box test for no autocorrelation of residuals
  - also returns Box-Pierce statistic

:py:func:`acorr_breush_godfrey <statsmodels.stats.diagnostic.acorr_breush_godfrey>`
  - Breush-Pagan test for no autocorrelation of residuals


missing
  - ?


Non-Linearity Tests
-------------------

:py:func:`linear_harvey_collier <statsmodels.stats.diagnostic.linear_harvey_collier>`
  - Multiplier test for Null hypothesis that linear specification is
    correct

:py:func:`acorr_linear_rainbow <statsmodels.stats.diagnostic.acorr_linear_rainbow>`
  - Multiplier test for Null hypothesis that linear specification is
    correct.

:py:func:`acorr_linear_lm <statsmodels.stats.diagnostic.acorr_linear_lm>`
  - Lagrange Multiplier test for Null hypothesis that linear specification is
    correct. This tests against specific functional alternatives.


Tests for Structural Change, Parameter Stability
------------------------------------------------

Test whether all or some regression coefficient are constant over the
entire data sample.

Known Change Point
^^^^^^^^^^^^^^^^^^

OneWayLS :
  - flexible ols wrapper for testing identical regression coefficients across
    predefined subsamples (eg. groups)

missing
  - predictive test: Greene, number of observations in subsample is smaller than
    number of regressors


Unknown Change Point
^^^^^^^^^^^^^^^^^^^^

:py:func:`breaks_cusumolsresid <statsmodels.stats.diagnostic.breaks_cusumolsresid>`
  - cusum test for parameter stability based on ols residuals

:py:func:`breaks_hansen <statsmodels.stats.diagnostic.breaks_hansen>`
  - test for model stability, breaks in parameters for ols, Hansen 1992

:py:func:`recursive_olsresiduals <statsmodels.stats.diagnostic.recursive_olsresiduals>`
  Calculate recursive ols with residuals and cusum test statistic. This is
  currently mainly helper function for recursive residual based tests.
  However, since it uses recursive updating and doesn't estimate separate
  problems it should be also quite efficient as expanding OLS function.

missing
  - supLM, expLM, aveLM  (Andrews, Andrews/Ploberger)
  - R-structchange also has musum (moving cumulative sum tests)
  - test on recursive parameter estimates, which are there?


Mutlicollinearity Tests
--------------------------------

conditionnum (statsmodels.stattools)
  - -- needs test vs Stata --
  - cf Grene (3rd ed.) pp 57-8

numpy.linalg.cond
  - (for more general condition numbers, but no behind the scenes help for
    design preparation)

Variance Inflation Factors
  This is currently together with influence and outlier measures
  (with some links to other tests here: http://www.stata.com/help.cgi?vif)


Normality and Distribution Tests
--------------------------------

:py:func:`jarque_bera <statsmodels.stats.tools.jarque_bera>`
  - printed with summary()
  - test for normal distribution of residuals

Normality tests in scipy stats
  need to find list again

:py:func:`omni_normtest <statsmodels.stats.tools.omni_normtest>`
  - test for normal distribution of residuals
  - printed with summary()

:py:func:`normal_ad <statsmodels.stats.diagnostic.normal_ad>`
  - Anderson Darling test for normality with estimated mean and variance

:py:func:`kstest_normal <statsmodels.stats.diagnostic.kstest_normal>` :py:func:`lillifors <statsmodels.stats.diagnostic.lillifors>`
  Lillifors test for normality, this is a Kolmogorov-Smirnov tes with for
  normality with estimated mean and variance. lillifors is an alias for
  kstest_normal

qqplot, scipy.stats.probplot

other goodness-of-fit tests for distributions in scipy.stats and enhancements
  - kolmogorov-smirnov
  - anderson : Anderson-Darling
  - likelihood-ratio, ...
  - chisquare tests, powerdiscrepancy : needs wrapping (for binning)


Outlier and Influence Diagnostic Measures
-----------------------------------------

These measures try to identify observations that are outliers, with large
residual, or observations that have a large influence on the regression
estimates. Robust Regression, RLM, can be used to both estimate in an outlier
robust way as well as identify outlier. The advantage of RLM that the
estimation results are not strongly influenced even if there are many
outliers, while most of the other measures are better in identifying
individual outliers and might not be able to identify groups of outliers.

robust regression results RLM
    example from example_rlm.py ::

        import statsmodels.api as sm

        ### Example for using Huber's T norm with the default
        ### median absolute deviation scaling

        data = sm.datasets.stackloss.Load()
        data.exog = sm.add_constant(data.exog)
        huber_t = sm.RLM(data.endog, data.exog, M=sm.robust.norms.HuberT())
        hub_results = huber_t.fit()
        print hub_results.weights

    And the weights give an idea of how much a particular observation is
    down-weighted according to the scaling asked for.

:py:class:`Influence <statsmodels.stats.outliers_influence.Influence>`
   Class in stats.outliers_influence, most standard measures for outliers
   and influence are available as methods or attributes given a fitted
   OLS model. This is mainly written for OLS, some but not all measures
   are also valid for other models.
   Some of these statistics can be calculated from an OLS results instance,
   others require that an OLS is estimated for each left out variable.

   resid_press
   resid_studentized_external
   resid_studentized_internal
   ess_press
   hat_matrix_diag
   cooks_distance - Cook's Distance `Wikipedia <http://en.wikipedia.org/wiki/Cook%27s_distance>`_ (with some other links)
   cov_ratio
   dfbetas
   dffits
   dffits_internal
   det_cov_params_not_obsi
   params_not_obsi
   sigma2_not_obsi



Unit Root Tests
---------------

:py:func:`unitroot_adf <statsmodels.stats.diagnostic.unitroot_adf>`
  - same as adfuller but with different signature


