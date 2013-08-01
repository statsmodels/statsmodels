.. currentmodule:: statsmodels.stats

.. _stats:


Statistics :mod:`stats`
=======================

This section collects various statistical tests and tools.
Some can be used independently of any models, some are intended as extension to the
models and model results.

API Warning: The functions and objects in this category are spread out in
various modules and might still be moved around. We expect that in future the
statistical tests will return class instances with more informative reporting
instead of only the raw numbers.


.. _stattools:


Residual Diagnostics and Specification Tests
--------------------------------------------

.. currentmodule:: statsmodels.stats.stattools

.. autosummary::
   :toctree: generated/

   durbin_watson
   jarque_bera
   omni_normtest

.. currentmodule:: statsmodels.stats.diagnostic

.. autosummary::
   :toctree: generated/

   acorr_ljungbox
   acorr_breush_godfrey

   HetGoldfeldQuandt
   het_goldfeldquandt
   het_breushpagan
   het_white
   het_arch

   linear_harvey_collier
   linear_rainbow
   linear_lm

   breaks_cusumolsresid
   breaks_hansen
   recursive_olsresiduals

   CompareCox
   compare_cox
   CompareJ
   compare_j

   unitroot_adf

   normal_ad
   kstest_normal
   lillifors

Outliers and influence measures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: statsmodels.stats.outliers_influence

.. autosummary::
   :toctree: generated/

   OLSInfluence
   variance_inflation_factor

See also the notes on :ref:`notes on regression diagnostics <diagnostics>`

Sandwich Robust Covariances
---------------------------

The following functions calculate covariance matrices and standard errors for
the parameter estimates that are robust to heteroscedasticity and
autocorrelation in the errors. Similar to the methods that are available
for the LinearModelResults, these methods are designed for use with OLS.

.. currentmodule:: statsmodels.stats

.. autosummary::
   :toctree: generated/

   sandwich_covariance.cov_hac
   sandwich_covariance.cov_nw_panel
   sandwich_covariance.cov_nw_groupsum
   sandwich_covariance.cov_cluster
   sandwich_covariance.cov_cluster_2groups
   sandwich_covariance.cov_white_simple

The following are standalone versions of the heteroscedasticity robust
standard errors attached to LinearModelResults

.. autosummary::
   :toctree: generated/

   sandwich_covariance.cov_hc0
   sandwich_covariance.cov_hc1
   sandwich_covariance.cov_hc2
   sandwich_covariance.cov_hc3

   sandwich_covariance.se_cov


Goodness of Fit Tests and Measures
----------------------------------

some tests for goodness of fit for univariate distributions

.. currentmodule:: statsmodels.stats.gof

.. autosummary::
   :toctree: generated/

   powerdiscrepancy
   gof_chisquare_discrete
   gof_binning_discrete
   chisquare_effectsize

.. currentmodule:: statsmodels.stats.diagnostic

.. autosummary::
   :toctree: generated/

   normal_ad
   kstest_normal
   lillifors

Non-Parametric Tests
--------------------

.. currentmodule:: statsmodels.sandbox.stats.runs

.. autosummary::
   :toctree: generated/

   mcnemar
   symmetry_bowker
   median_test_ksample
   runstest_1samp
   runstest_2samp
   cochrans_q
   Runs

.. currentmodule:: statsmodels.stats.descriptivestats

.. autosummary::
   :toctree: generated/

   sign_test

.. _interrater:   

Interrater Reliability and Agreement
------------------------------------

The main function that statsmodels has currently available for interrater
agreement measures and tests is Cohen's Kappa. Fleiss' Kappa is currently
only implemented as a measures but without associated results statistics.

.. currentmodule:: statsmodels.stats.inter_rater

.. autosummary::
   :toctree: generated/

   cohens_kappa
   fleiss_kappa
   to_table
   aggregate_raters

Multiple Tests and Multiple Comparison Procedures
-------------------------------------------------

`multipletests` is a function for p-value correction, which also includes p-value
correction based on fdr in `fdrcorrection`.
`tukeyhsd` performs simulatenous testing for the comparison of (independent) means.
These three functions are verified.
GroupsStats and MultiComparison are convenience classes to multiple comparisons similar
to one way ANOVA, but still in developement

.. currentmodule:: statsmodels.sandbox.stats.multicomp

.. autosummary::
   :toctree: generated/

   multipletests
   fdrcorrection0

   GroupsStats
   MultiComparison
   TukeyHSDResults

.. currentmodule:: statsmodels.stats.multicomp

.. autosummary::
   :toctree: generated/

   pairwise_tukeyhsd

The following functions are not (yet) public

.. currentmodule:: statsmodels.sandbox.stats.multicomp

.. autosummary::
   :toctree: generated/

   varcorrection_pairs_unbalanced
   varcorrection_pairs_unequal
   varcorrection_unbalanced
   varcorrection_unequal

   StepDown
   catstack
   ccols
   compare_ordered
   distance_st_range
   ecdf
   get_tukeyQcrit
   homogeneous_subsets
   line
   maxzero
   maxzerodown
   mcfdr
   qcrit
   randmvn
   rankdata
   rejectionline
   set_partition
   set_remove_subs
   tiecorrect

.. _tost:

Basic Statistics and t-Tests with frequency weights
---------------------------------------------------

Besides basic statistics, like mean, variance, covariance and correlation for
data with case weights, the classes here provide one and two sample tests
for means. The t-tests have more options than those in scipy.stats, but are
more restrictive in the shape of the arrays. Confidence intervals for means
are provided based on the same assumptions as the t-tests.

Additionally, tests for equivalence of means are available for one sample and
for two, either paired or independent, samples. These tests are based on TOST,
two one-sided tests, which have as null hypothesis that the means are not
"close" to each other.

.. currentmodule:: statsmodels.stats.weightstats

.. autosummary::
   :toctree: generated/

   DescrStatsW
   CompareMeans
   ttest_ind
   ttost_ind
   ttost_paired
   ztest
   ztost
   zconfint

weightstats also contains tests and confidence intervals based on summary
data

.. currentmodule:: statsmodels.stats.weightstats

.. autosummary::
   :toctree: generated/

   _tconfint_generic
   _tstat_generic
   _zconfint_generic
   _zstat_generic
   _zstat_generic2


Power and Sample Size Calculations
----------------------------------

The :mod:`power` module currently implements power and sample size calculations
for the t-tests, normal based test, F-tests and Chisquare goodness of fit test.
The implementation is class based, but the module also provides
three shortcut functions, ``tt_solve_power``, ``tt_ind_solve_power`` and
``zt_ind_solve_power`` to solve for any one of the parameters of the power
equations.


.. currentmodule:: statsmodels.stats.power

.. autosummary::
   :toctree: generated/

   TTestIndPower
   TTestPower
   GofChisquarePower
   NormalIndPower
   FTestAnovaPower
   FTestPower
   tt_solve_power
   tt_ind_solve_power
   zt_ind_solve_power


.. _proportion_stats:

Proportion
----------


Also available are hypothesis test, confidence intervals and effect size for
proportions that can be used with NormalIndPower.

.. currentmodule:: statsmodels.stats.proportion

.. autosummary::
   :toctree: generated

   proportion_confint
   proportion_effectsize

   binom_test
   binom_test_reject_interval
   binom_tost
   binom_tost_reject_interval


   proportions_ztest
   proportions_ztost
   proportions_chisquare
   proportions_chisquare_allpairs
   proportions_chisquare_pairscontrol

   proportion_effectsize
   power_binom_tost
   power_ztost_prop
   samplesize_confint_proportion


Moment Helpers
--------------

When there are missing values, then it is possible that a correlation or
covariance matrix is not positive semi-definite. The following three
functions can be used to find a correlation or covariance matrix that is
positive definite and close to the original matrix.

.. currentmodule:: statsmodels.stats.correlation_tools

.. autosummary::
   :toctree: generated/

	corr_nearest
	corr_clipped
	cov_nearest

These are utility functions to convert between central and non-central moments, skew,
kurtosis and cummulants.

.. currentmodule:: statsmodels.stats.moment_helpers

.. autosummary::
   :toctree: generated/

   cum2mc
   mc2mnc
   mc2mvsk
   mnc2cum
   mnc2mc
   mnc2mvsk
   mvsk2mc
   mvsk2mnc
   cov2corr
   corr2cov
   se_cov

