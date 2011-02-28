.. currentmodule:: scikits.statsmodels.sandbox.distributions

.. _distributions:


Distributions
=============

Introduction
------------

This section collects various additional functions and methods for statistical
distributions.


Sandbox Warning: The functions and objects in this category are still in the sandbox.
Many functions or classes have been tested on individual examples, but don't have a
(consistent or complete) test suite yet.


Distribution Extras
-------------------


.. currentmodule:: scikits.statsmodels.sandbox.distributions.extras

*Skew Distributions*

.. autosummary::
   :toctree: generated/

   SkewNorm_gen
   SkewNorm2_gen
   ACSkewT_gen
   skewnorm2

*Distributions based on Gram-Charlier expansion*

.. autosummary::
   :toctree: generated/

   pdf_moments_st
   pdf_mvsk
   pdf_moments
   NormExpan_gen

*cdf of multivariate normal* wrapper for scipy.stats


.. autosummary::
   :toctree: generated/

   mvstdnormcdf
   mvnormcdf

Univariate Distributions by non-linear Transformations
------------------------------------------------------

Univariate distributions can be generated from a non-linear transformation of an
existing univariate distribution. `Transf_gen` is a class that can generate a new
distribution from a monotonic transformation, `TransfTwo_gen` can use hump-shaped
or u-shaped transformation, such as abs or square. The remaining objects are
special cases.

.. currentmodule:: scikits.statsmodels.sandbox.distributions.transformed

.. autosummary::
   :toctree: generated/

   TransfTwo_gen
   Transf_gen

   ExpTransf_gen
   LogTransf_gen
   SquareFunc

   absnormalg
   invdnormalg

   loggammaexpg
   lognormalg
   negsquarenormalg

   squarenormalg
   squaretg
