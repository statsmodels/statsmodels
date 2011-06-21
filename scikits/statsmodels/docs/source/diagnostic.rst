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

het_breushpagan (scikits.sandbox.tools.stattools) :
    Lagrange Multiplier Heteroscedasticity Test by Breush-Pagan

het_white (scikits.sandbox.tools.stattools) :
    Lagrange Multiplier Heteroscedasticity Test by White

het_goldfeldquandt (scikits.sandbox.tools.stattools) :
    test whether variance is the same in 2 subsamples


Autocorrelation Tests
---------------------

This group of test whether the regression residuals are not autocorrelated.
They assume that observations are ordered by time.

durbin_watson (scikits.stattools) :
  - Durbin-Watson test for no autocorrelation of residuals
  - printed with summary()

acorr_ljungbox (scikits.sandbox.tools.stattools) :
  - Ljung-Box test for no autocorrelation of residuals
  - also returns Box-Pierce statistic

acorr_lm
  - Lagrange Multiplier tests for autocorrelation
  - not checked yet, might not make sense

missing
  - Breush-Godfrey test, in stata and Greene 12.7.1
  -


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

(Note: considerable cleaning still required)

recursive_olsresiduals(olsresults, skip=None, lamda=0.0, alpha=0.95):
  - calculate recursive ols with residuals and cusum test statistic

breaks_cusumolsresid :
  - cusum test for parameter stability based on ols residuals

breaks_hansen :
  - test for model stability, breaks in parameters for ols, Hansen 1992

missing
  - supLM, expLM, aveLM  (Andrews, Andrews/Ploberger)
  - R-structchange also has musum (moving cumulative sum tests)

Mutlicollinearity Tests
--------------------------------

conditionnum (scikits.statsmodels.stattools) -- needs test vs Stata --
cf Grene (3rd ed.) pp 57-8
numpy.linalg.cond (for more general condition numbers, but no behind
the scenes help for design preparation)

missing
  - Variance Inflation Factors
    (with some links to other tests here: http://www.stata.com/help.cgi?vif)

Outlier Diagnosis
-----------------

  - robust regression results
    example from example_rlm.py ::

        import scikits.statsmodels.api as sm

        ### Example for using Huber's T norm with the default
        ### median absolute deviation scaling

        data = sm.datasets.stackloss.Load()
        data.exog = sm.add_constant(data.exog)
        huber_t = sm.RLM(data.endog, data.exog, M=sm.robust.norms.HuberT())
        hub_results = huber_t.fit()
        print hub_results.weights

    And the weights give an idea of how much a particular observation is
    down-weighted according to the scaling asked for.

missing :
   - Cook's Distance
     http://en.wikipedia.org/wiki/Cook%27s_distance (with some other links)


Normality and Distribution Tests
--------------------------------

jarque_bera (scikits.stats.tools) :
  - printed with summary()
  - test for normal distribution of residuals

omni_normtest (scikits.stats.tools) :
  - printed with summary()
  - test for normal distribution of residuals

qqplot, scipy.stats.probplot

other goodness-of-fit tests for distributions in scipy.stats and enhancements
  - kolmogorov-smirnov
  - anderson : Anderson-Darling
  - likelihood-ratio, ...
  - chisquare tests, powerdiscrepancy : needs wrapping (for binning)


Non-Linearity Tests
-------------------

nothing yet ???



Unit Root Tests
---------------

unitroot_adf
  - Augmented Dickey-Fuller test for unit roots


