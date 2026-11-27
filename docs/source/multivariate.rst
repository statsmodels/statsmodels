.. module:: statsmodels.multivariate
   :synopsis: Models for multivariate data

.. currentmodule:: statsmodels.multivariate

.. _multivariate:


Multivariate Statistics :mod:`multivariate`
===========================================

This section includes methods and algorithms from multivariate statistics.


Principal Component Analysis
----------------------------

.. module:: statsmodels.multivariate.pca
   :synopsis: Principal Component Analaysis

.. currentmodule:: statsmodels.multivariate.pca

.. autosummary::
   :toctree: generated/

   PCA
   pca


Factor Analysis
---------------

.. currentmodule:: statsmodels.multivariate.factor

.. autosummary::
   :toctree: generated/

   Factor
   FactorResults


Factor Rotation
---------------

.. currentmodule:: statsmodels.multivariate.factor_rotation

.. autosummary::
   :toctree: generated/

   rotate_factors
   target_rotation
   procrustes
   promax


Canonical Correlation
---------------------

.. currentmodule:: statsmodels.multivariate.cancorr

.. autosummary::
   :toctree: generated/

   CanCorr


MANOVA
------

.. currentmodule:: statsmodels.multivariate.manova

.. autosummary::
   :toctree: generated/

   MANOVA


Repeated-measures data
^^^^^^^^^^^^^^^^^^^^^^

MANOVA can be used for repeated-measures analyses when the repeated
measurements are represented as multiple dependent variables. In this
wide-format representation, each subject contributes one row, with
separate columns for measurements taken under different conditions
or at different time points.

The multivariate model can test hypotheses about predictors and
linear combinations of the dependent variables. Questions about
within-subject effects require appropriate specification of
multivariate contrasts.

The :class:`MANOVA` implementation is based on multivariate
regression and provides multivariate hypothesis tests through
:meth:`MANOVA.mv_test`. It does not provide a dedicated
repeated-measures interface or automatically construct
within-subject contrasts.

MANOVA and univariate repeated-measures ANOVA have different
formulations and assumptions. For longitudinal data in long format,
particularly when observations are unbalanced or incomplete,
linear mixed-effects models and generalized estimating equations
may also be appropriate. See :ref:`anova_ref` for an overview
of these alternatives.


MultivariateOLS
---------------

`_MultivariateOLS` is a model class with limited features. Currently it
supports multivariate hypothesis tests and is used as backend for MANOVA.

.. currentmodule:: statsmodels.multivariate.multivariate_ols

.. autosummary::
   :toctree: generated/

   _MultivariateOLS
   _MultivariateOLSResults
   MultivariateTestResults


Multivariate Least Squares
---------------------------

`MultivariateLS` is a multivariate least-squares model for a set of
continuous, jointly-modeled response variables.

.. currentmodule:: statsmodels.multivariate.multivariate_ols

.. autosummary::
   :toctree: generated/

   MultivariateLS
   MultivariateLSResults
