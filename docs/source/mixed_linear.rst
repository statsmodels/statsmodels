.. currentmodule:: statsmodels.regression.mixed_linear_model

.. _mixedlmmod:

Linear Mixed Effects Models
===========================

Linear Mixed Effects models are used for regression analyses involving
dependent data.  Such data arise when working with longitudinal and
other study designs in which multiple observations are made on each
subject.  Some specific linear mixed effects models are

* *Random intercepts models*, where all responses in a group are
  additively shifted by a value that is specific to the group.

* *Random slopes models*, where the responses in a group follow a
  (conditional) mean trajectory that is linear in the observed
  covariates, with the slopes (and possibly intercepts) varying by
  group.

* *Variance components models*, where the levels of one or more
  categorical covariates are associated with draws from distributions.
  These random terms additively determine the conditional mean of each
  observation based on its covariate values.

The Statsmodels implementation of LME is primarily group-based,
meaning that random effects must be independently-realized for
responses in different groups.  There are two types of random effects
in our implementation of mixed models: (i) random coefficients that
have an unknown covariance matrix, and (ii) random coefficients that
are independent draws from a common univariate distribution.  For
both (i) and (ii), the random effects influence the conditional mean
of a group through their matrix/vector product with a group-specific
design matrix.

Statsmodels MixedLM handles most non-crossed random effects models,
and some crossed models.  To include crossed random effects in a
model, it is necessary to treat the entire dataset as a single group.
The variance components arguments to the model can then be used to
define models with various combinations of crossed and non-crossed
random effects.

The Statsmodels LME framework currently supports post-estimation
inference via Wald tests and confidence intervals on the coefficients,
profile likelihood analysis, likelihood ratio testing, and AIC.

Examples
--------

.. ipython:: python

  import statsmodels.api as sm
  import statsmodels.formula.api as smf

  data = sm.datasets.get_rdataset("dietox", "geepack").data

  md = smf.mixedlm("Weight ~ Time", data, groups=data["Pig"])
  mdf = md.fit()
  print(mdf.summary())

Detailed examples can be found here

* `Mixed LM <examples/notebooks/generated/mixed_lm_example.html>`__

There some notebook examples on the Wiki:
`Wiki notebooks for MixedLM <https://github.com/statsmodels/statsmodels/wiki/Examples#linear-mixed-models>`_



Technical Documentation
-----------------------

The data are partitioned into disjoint groups.
The probability model for group :math:`i` is:

.. math::

    Y = X\beta + Z\gamma + Q_1\eta_1 + \cdots + Q_k\eta_k + \epsilon

where

* :math:`n_i` is the number of observations in group :math:`i`
* :math:`Y` is a :math:`n_i` dimensional response vector
* :math:`X` is a :math:`n_i * k_{fe}` dimensional matrix of fixed effects
  coefficients
* :math:`\beta` is a :math:`k_{fe}`-dimensional vector of fixed effects slopes
* :math:`Z` is a :math:`n_i * k_{re}` dimensional matrix of random effects
  coefficients
* :math:`\gamma` is a :math:`k_{re}`-dimensional random vector with mean 0
  and covariance matrix :math:`\Psi`; note that each group
  gets its own independent realization of gamma.
* :math:`Q_j` is a :math: `n_i \time q_j` dimensional design matrix for the
  :math: `j`th variance component.
* :math:`\eta_j` is a :math:`q_j`-dimensional random vector containing independent
  and identically ditsributed values with variance :math:`\tau_j^2`.
* :math:`\epsilon` is a :math:`n_i` dimensional vector of i.i.d normal
  errors with mean 0 and variance :math:`\sigma^2`; the :math:`\epsilon`
  values are independent both within and between groups

:math:`Y, X, \{Q_j\}` and :math:`Z` must be entirely observed.  :math:`\beta`,
:math:`\Psi`, and :math:`\sigma^2` are estimated using ML or REML estimation,
and :math:`\gamma`, :math:`\{\eta_j\}` and :math:`\epsilon` are
random so define the probability model.

The marginal mean structure is :math:`E[Y|X,Z] = X*\beta`.  If only
the marginal mean structure is of interest, GEE is a good alternative
to mixed models.


Notation:

* :math:`cov_{re}` is the random effects covariance matrix (referred
  to above as :math:`\Psi`) and :math:`scale` is the (scalar) error
  variance.  There is also a single estimated variance parameter
  :math:`\tau_j^2` for each variance component.  For a single group,
  the marginal covariance matrix of endog given exog is
  :math:`scale*I + Z * cov_{re} * Z`, where :math:`Z` is the design
  matrix for the random effects in one group.

Notes
^^^^^

1. Three different parameterizations are used here in different
places.  The regression slopes (usually called :math:`fe_{params}`) are
identical in all three parameterizations, but the variance parameters
differ.  The parameterizations are:

* The *natural parameterization* in which :math:`cov(endog) = scale*I + Z *
  cov_{re} * Z`, as described above.  This is the main parameterization
  visible to the user.

* The *profile parameterization* in which :math:`cov(endog) = I +
  Z * cov_{re1} * Z`.  This is the parameterization of the profile
  likelihood that is maximized to produce parameter estimates.
  (see Lindstrom and Bates for details).  The *natural* :math:`cov_{re}` is
  equal to the *profile*  :math:`cov_{re1}` times scale.

* The *square root parameterization* in which we work with the
  Cholesky factor of :math:`cov_{re1}` instead of :math:`cov_{re1}` directly.

All three parameterizations can be *packed* by concatenating :math:`fe_{params}`
together with the lower triangle of the dependence structure.  Note
that when unpacking, it is important to either square or reflect the
dependence structure depending on which parameterization is being
used.

2. The optimization strategy is to optionally perform a few EM steps,
followed by optionally performing a few steepest descent steps,
followed by conjugate gradient descent using one of the scipy gradient
optimizers.  The EM and steepest descent steps are used to get
adequate starting values for the conjugate gradient optimization,
which is much faster.

References
^^^^^^^^^^

The primary reference for the implementation details is:

*   MJ Lindstrom, DM Bates (1988).  *Newton Raphson and EM algorithms for
    linear mixed effects models for repeated measures data*.  Journal of
    the American Statistical Association. Volume 83, Issue 404, pages 1014-1022.

See also this more recent document:

* http://econ.ucsb.edu/~doug/245a/Papers/Mixed%20Effects%20Implement.pdf

All the likelihood, gradient, and Hessian calculations closely follow
Lindstrom and Bates.

The following two documents are written more from the perspective of
users:

* http://lme4.r-forge.r-project.org/lMMwR/lrgprt.pdf

* http://lme4.r-forge.r-project.org/slides/2009-07-07-Rennes/3Longitudinal-4.pdf

.. Class hierachy: TODO

   General references for this class of models are

Module Reference
----------------

.. module:: statsmodels.regression.mixed_linear_model
   :synopsis: Mixed Linear Models


The model class is:

.. autosummary::
   :toctree: generated/

   MixedLM

The result classe are:

.. autosummary::
   :toctree: generated/

   MixedLMResults
