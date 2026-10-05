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
   ECDFDiscrete
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
   rvs_kernel

The Archimedean generator transforms used by ``ArchimedeanCopula`` and the
Pickands dependence functions used by ``ExtremeValueCopula`` are in

.. module:: statsmodels.distributions.copula.transforms
   :synopsis: Archimedean copula generator transforms
.. currentmodule:: statsmodels.distributions.copula.transforms

and

.. module:: statsmodels.distributions.copula.depfunc_ev
   :synopsis: Pickands dependence functions for extreme value copulas
.. currentmodule:: statsmodels.distributions.copula.depfunc_ev

respectively.

Bernstein Distribution
-----------------------

Univariate and bivariate distributions estimated nonparametrically on the
unit hypercube using Bernstein polynomials, e.g. for use as the marginal or
copula component of a semiparametric model.

.. module:: statsmodels.distributions.bernstein
   :synopsis: Distributions based on Bernstein polynomials

.. currentmodule:: statsmodels.distributions.bernstein

.. autosummary::
   :toctree: generated/

   BernsteinDistribution
   BernsteinDistributionUV
   BernsteinDistributionBV

Mixture of Distributions
--------------------------

Tools for combining component distributions into a mixture and generating
random samples from the mixture.

.. module:: statsmodels.distributions.mixture_rvs
   :synopsis: Mixtures of distributions

.. currentmodule:: statsmodels.distributions.mixture_rvs

.. autosummary::
   :toctree: generated/

   MixtureDistribution
   mixture_rvs
   mv_mixture_rvs

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


Helper Functions
----------------

.. module:: statsmodels.tools.rng_qrng
   :synopsis: Tools for working with random variable generation

.. currentmodule:: statsmodels.tools.rng_qrng

.. autosummary::
   :toctree: generated/

   check_random_state
