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
