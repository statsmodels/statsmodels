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

Count Distributions
-------------------

The `discrete` module contains classes for count distributions that are based
on discretizing a continuous distribution, and specific count distributions
that are not available in scipy.distributions like generalized poisson and
zero-inflated count models.

The latter are mainly in support of the corresponding models in
`statsmodels.discrete`. Some methods are not specifically implemented and will
use potentially slow inherited generic methods.

.. module:: statsmodels.distributions.discrete
   :synopsis: Support for count distributions

.. currentmodule:: statsmodels.distributions.discrete

.. autosummary::
   :toctree: generated/

   DiscretizedCount
   DiscretizedModel
   genpoisson_p
   zigenpoisson
   zinegbin
   zipoisson

Copula
------

The `copula` sub-module provides classes to model the dependence between
parameters. Copulae are used to construct a multivariate joint distribution and
provide a set of functions like sampling, PDF, CDF.

.. module:: statsmodels.distributions.copula.api
   :synopsis: Copula for modeling parameter dependence

.. currentmodule:: statsmodels.distributions.copula.api

.. autosummary::
   :toctree: generated/

   CopulaDistribution
   ArchimedeanCopula
   FrankCopula
   ClaytonCopula
   GumbelCopula
   GaussianCopula
   StudentTCopula
   ExtremeValueCopula
   IndependenceCopula

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


Helper Functions
----------------

.. module:: statsmodels.tools.rng_qrng
   :synopsis: Tools for working with random variable generation

.. currentmodule:: statsmodels.tools.rng_qrng

.. autosummary::
   :toctree: generated/

   check_random_state
