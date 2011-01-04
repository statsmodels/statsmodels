.. currentmodule:: scikits.statsmodels.stats

.. _stats:


Statistics :mod:`stats`
=======================

Introduction
------------

This section collects various statistical tests and tools.
Some can be used independently of any models, some are intended as extension to the
models and model results.

API Warning: The functions and objects in this category are spread out in various modules
and might still be moved around.

Debugging in Sphinx and rst sucks, where the missing invisible whitespace ????



.. _stattools:


Residual Diagnostics and Specification Tests
--------------------------------------------

.. currentmodule:: scikits.statsmodels.stattools

.. autosummary::
   :toctree: generated/

   durbin_watson
   jarque_bera
   omni_normtest

.. currentmodule:: scikits.statsmodels.sandbox.tools.stattools

.. autosummary::
   :toctree: generated/

   acorr_ljungbox
   acorr_lm

   breaks_cusumolsresid
   breaks_hansen
   CompareCox
   CompareJ
   compare_cox
   compare_j
   het_breushpagan
   HetGoldfeldQuandt
   het_goldfeldquandt
   het_goldfeldquandt2
   het_white
   unitroot_adf
   neweywestcov
   recursive_olsresiduals
   recursive_olsresiduals2


Univariate Distributions by non-linear Transformations
------------------------------------------------------

here for debugging Sphinx



Goodness of Fit Tests and Measures
----------------------------------

 some tests for goodness of fit for univariate distributions

.. currentmodule:: scikits.statsmodels.stats.gof

.. autosummary::
   :toctree: generated/

   powerdiscrepancy
   gof_chisquare_discrete
   gof_binning_discrete



Non-Parametric Tests
--------------------

.. currentmodule:: scikits.statsmodels.sandbox.stats.runs

.. autosummary::
   :toctree: generated/

   mcnemar
   median_test_ksample
   runstest_1samp
   runstest_2samp
   cochran_q
   Runs


Multiple Tests and Multiple Comparison Procedures
-------------------------------------------------

`multipletests` is a function for p-value correction, which also includes p-value
correction based on fdr in `fdrcorrection`.
`tukeyhsd` performs simulatenous testing for the comparison of (independent) means.
These three functions are verified.
GroupsStats and MultiComparison are convenience classes to multiple comparisons similar
to one way ANOVA, but still in developement

.. currentmodule:: scikits.statsmodels.sandbox.stats.multicomp

.. autosummary::
   :toctree: generated/

   multipletests
   fdrcorrection0
   tukeyhsd

   GroupsStats
   MultiComparison

The following functions are not (yet) public (here for my own benefit, JP)

.. currentmodule:: scikits.statsmodels.sandbox.stats.multicomp

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
   test_tukeyhsd
   tiecorrect




Basic Statistics and t-Tests with frequency weights
---------------------------------------------------

.. currentmodule:: scikits.statsmodels.stats.weightstats

.. autosummary::
   :toctree: generated/

   CompareMeans
   DescrStatsW
   tstat_generic


Moment Helpers
--------------

These are utility functions to convert between central and non-central moments, skew,
kurtosis and cummulants.

.. currentmodule:: scikits.statsmodels.stats.moment_helpers

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




