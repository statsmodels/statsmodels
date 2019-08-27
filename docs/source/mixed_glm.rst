.. currentmodule:: statsmodels.genmod.bayes_mixed_glm

Generalized Linear Mixed Effects Models
=======================================

Generalized Linear Mixed Effects (GLIMMIX) models are generalized
linear models with random effects in the linear predictors.
statsmodels currently supports estimation of binomial and Poisson
GLIMMIX models using two Bayesian methods: the Laplace approximation
to the posterior, and a variational Bayes approximation to the
posterior.  Both methods provide point estimates (posterior means) and
assessments of uncertainty (posterior standard deviation).

The current implementation only supports independent random effects.

Technical Documentation
-----------------------

Unlike statsmodels mixed linear models, the GLIMMIX implementation is
not group-based.  Groups are created by interacting all random effects
with a categorical variable.  Note that this creates large, sparse
random effects design matrices `exog_vc`.  Internally, `exog_vc` is
converted to a scipy sparse matrix.  When passing the arguments
directly to the class initializer, a sparse matrix may be passed.
When using formulas, a dense matrix is created then converted to
sparse.  For very large problems, it may not be feasible to use
formulas due to the size of this dense intermediate matrix.

References
^^^^^^^^^^

Blei, Kucukelbir, McAuliffe (2017).  Variational Inference: A review
for Statisticians https://arxiv.org/pdf/1601.00670.pdf

Module Reference
----------------

.. module:: statsmodels.genmod.bayes_mixed_glm
   :synopsis: Bayes Mixed Generalized Linear Models


The model classes are:

.. autosummary::
   :toctree: generated/

   BinomialBayesMixedGLM
   PoissonBayesMixedGLM

The result class is:

.. autosummary::
   :toctree: generated/

   BayesMixedGLMResults
