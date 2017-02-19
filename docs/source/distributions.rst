.. module:: statsmodels.sandbox.distributions
   :synopsis: Probability distributions

.. currentmodule:: statsmodels.sandbox.distributions

.. _distributions:


Distributions
=============

This section collects various additional functions and methods for statistical
distributions.

Empirical Distributions
-----------------------

.. module:: statsmodels.distributions.empirical_distribution
   :synopsis: Tools for working with empirical distributions

.. currentmodule:: statsmodels.distributions.empirical_distribution

.. autosummary::
   :toctree: generated/

   ECDF
   StepFunction
   monotone_fn_inverter

Distribution Extras
-------------------


.. module:: statsmodels.sandbox.distributions.extras
   :synopsis: Probability distributions and random number generators

.. currentmodule:: statsmodels.sandbox.distributions.extras

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

.. module:: statsmodels.sandbox.distributions.transformed
   :synopsis: Experimental probability distributions and random number generators

.. currentmodule:: statsmodels.sandbox.distributions.transformed

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
