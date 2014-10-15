.. currentmodule:: statsmodels.regression.mixed_linear_model


.. _mixedlmmod:

Linear Mixed Effects Models
===========================


Linear Mixed Effects models are used for regression analyses involving
dependent data.  Such data arise when working with longitudinal and
other study designs in which multiple observations are made on each
subject.  Two specific mixed effects models are "random intercepts
models", where all responses in a single group are additively shifted
by a value that is specific to the group, and "random slopes models",
where the values follow a mean trajectory that is linear in observed
covariates, with both the slopes and intercept being specific to the
group.  The Statsmodels MixedLM implementation allows arbitrary random
effects design matrices to be specified for the groups, so these and
other types of random effects models can all be fit.

The Statsmodels LME framework currently supports post-estimation
inference via Wald tests and confidence intervals on the coefficients,
profile likelihood analysis, likelihood ratio testing, and AIC.  Some
limitations of the current implementation are that it does not support
structure more complex on the residual errors (they are always
homoscedastic), and it does not support crossed random effects.  We
hope to implement these features for the next release.

Examples
--------

..code-block:: python

  import statsmodels.api as sm
  import statsmodels.formula.api as smf

  data = sm.datasets.get_rdataset("dietox", "geepack").data
  
  md = smf.mixedlm("Weight ~ Time", data, groups=data["Pig"])
  mdf = md.fit()
  print(mdf.summary())

Detailed examples can be found here

.. toctree::
   :maxdepth: 2

   examples/notebooks/generated/

There some notebook examples on the Wiki:
`Wiki notebooks for MixedLM <https://github.com/statsmodels/statsmodels/wiki/Examples#linear-mixed-models>`_



Technical Documentation
-----------------------

The data are partitioned into disjoint groups.  The probability model
for group i is:

Y = X*beta + Z*gamma + epsilon

where

* n_i is the number of observations in group i
* Y is a n_i dimensional response vector
* X is a n_i x k_fe dimensional matrix of fixed effects
  coefficients
* beta is a k_fe-dimensional vector of fixed effects slopes
* Z is a n_i x k_re dimensional matrix of random effects
  coefficients
* gamma is a k_re-dimensional random vector with mean 0
  and covariance matrix Psi; note that each group
  gets its own independent realization of gamma.
* epsilon is a n_i dimensional vector of iid normal
  errors with mean 0 and variance sigma^2; the epsilon
  values are independent both within and between groups

Y, X and Z must be entirely observed.  beta, Psi, and sigma^2 are
estimated using ML or REML estimation, and gamma and epsilon are
random so define the probability model.

The mean structure is E[Y|X,Z] = X*beta.  If only the mean structure
is of interest, GEE is a good alternative to mixed models.

The primary reference for the implementation details is:

MJ Lindstrom, DM Bates (1988).  "Newton Raphson and EM algorithms for
linear mixed effects models for repeated measures data".  Journal of
the American Statistical Association. Volume 83, Issue 404, pages
1014-1022.

See also this more recent document:

http://econ.ucsb.edu/~doug/245a/Papers/Mixed%20Effects%20Implement.pdf

All the likelihood, gradient, and Hessian calculations closely follow
Lindstrom and Bates.

The following two documents are written more from the perspective of
users:

http://lme4.r-forge.r-project.org/lMMwR/lrgprt.pdf

http://lme4.r-forge.r-project.org/slides/2009-07-07-Rennes/3Longitudinal-4.pdf

Notation:

* `cov_re` is the random effects covariance matrix (referred to above
  as Psi) and `scale` is the (scalar) error variance.  For a single
  group, the marginal covariance matrix of endog given exog is scale*I
  + Z * cov_re * Z', where Z is the design matrix for the random
  effects in one group.

Notes:

1. Three different parameterizations are used here in different
places.  The regression slopes (usually called `fe_params`) are
identical in all three parameterizations, but the variance parameters
differ.  The parameterizations are:

* The "natural parameterization" in which cov(endog) = scale*I + Z *
  cov_re * Z', as described above.  This is the main parameterization
  visible to the user.

* The "profile parameterization" in which cov(endog) = I +
  Z * cov_re1 * Z'.  This is the parameterization of the profile
  likelihood that is maximized to produce parameter estimates.
  (see Lindstrom and Bates for details).  The "natural" cov_re is
  equal to the "profile" cov_re1 times scale.

* The "square root parameterization" in which we work with the
  Cholesky factor of cov_re1 instead of cov_re1 directly.

All three parameterizations can be "packed" by concatenating fe_params
together with the lower triangle of the dependence structure.  Note
that when unpacking, it is important to either square or reflect the
dependence structure depending on which parameterization is being
used.

2. The situation where the random effects covariance matrix is
singular is numerically challenging.  Small changes in the covariance
parameters may lead to large changes in the likelihood and
derivatives.

3. The optimization strategy is to optionally perform a few EM steps,
followed by optionally performing a few steepest descent steps,
followed by conjugate gradient descent using one of the scipy gradient
optimizers.  The EM and steepest descent steps are used to get
adequate starting values for the conjugate gradient optimization,
which is much faster.


.. todo::

   General references for this class of models are

Module Reference
----------------

The model class is:

.. autosummary::
   :toctree: generated/

   MixedLM

The result classe are:

.. autosummary::
   :toctree: generated/

   MixedLMResults
