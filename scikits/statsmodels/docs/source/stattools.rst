.. currentmodule:: scikits.statsmodels.stattools


.. _stattools:

Statistical Tests for Residual Analysis
=======================================

This is a collection of statistical tests that can be used independent from
any models. They are used in :mod: `statsmodels` to perform tests of the
underlying assumptions of a statistical model, e.g. durbin_watson for
autocorrelation of residuals, jarque_bera for normal distribution of residuals.
Further tests on the properties of the residuals can be obtained from
scipy.stats.


.. autosummary::
   :toctree: generated/

   durbin_watson
   jarque_bera
   omni_normtest
   conditionnum
