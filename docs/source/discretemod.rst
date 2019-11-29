.. currentmodule:: statsmodels.discrete.discrete_model


.. _discretemod:

Regression with Discrete Dependent Variable
===========================================

Regression models for limited and qualitative dependent variables. The module
currently allows the estimation of models with binary (Logit, Probit), nominal
(MNLogit), or count (Poisson, NegativeBinomial) data.

Starting with version 0.9, this also includes new count models, that are still
experimental in 0.9, NegativeBinomialP, GeneralizedPoisson and zero-inflated
models, ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP and
ZeroInflatedGeneralizedPoisson.

See `Module Reference`_ for commands and arguments.

Examples
--------

.. ipython:: python
  :okwarning:

  # Load the data from Spector and Mazzeo (1980)
  import statsmodels.api as sm
  spector_data = sm.datasets.spector.load_pandas()
  spector_data.exog = sm.add_constant(spector_data.exog)

  # Logit Model
  logit_mod = sm.Logit(spector_data.endog, spector_data.exog)
  logit_res = logit_mod.fit()
  print(logit_res.summary())

Detailed examples can be found here:


* `Overview <examples/notebooks/generated/discrete_choice_overview.html>`__
* `Examples <examples/notebooks/generated/discrete_choice_example.html>`__

Technical Documentation
-----------------------

Currently all models are estimated by Maximum Likelihood and assume
independently and identically distributed errors.

All discrete regression models define the same methods and follow the same
structure, which is similar to the regression results but with some methods
specific to discrete models. Additionally some of them contain additional model
specific methods and attributes.


References
^^^^^^^^^^

General references for this class of models are::

    A.C. Cameron and P.K. Trivedi.  `Regression Analysis of Count Data`.
        Cambridge, 1998

    G.S. Madalla. `Limited-Dependent and Qualitative Variables in Econometrics`.
        Cambridge, 1983.

    W. Greene. `Econometric Analysis`. Prentice Hall, 5th. edition. 2003.

Module Reference
----------------

.. module:: statsmodels.discrete.discrete_model
   :synopsis: Models for discrete data

The specific model classes are:

.. autosummary::
   :toctree: generated/

   Logit
   Probit
   MNLogit
   Poisson
   NegativeBinomial
   NegativeBinomialP
   GeneralizedPoisson

.. currentmodule:: statsmodels.discrete.count_model
.. module:: statsmodels.discrete.count_model

.. autosummary::
   :toctree: generated/

   ZeroInflatedPoisson
   ZeroInflatedNegativeBinomialP
   ZeroInflatedGeneralizedPoisson

.. currentmodule:: statsmodels.discrete.conditional_models
.. module:: statsmodels.discrete.conditional_models

.. autosummary::
   :toctree: generated/

   ConditionalLogit
   ConditionalMNLogit
   ConditionalPoisson

The specific result classes are:

.. currentmodule:: statsmodels.discrete.discrete_model

.. autosummary::
   :toctree: generated/

   LogitResults
   ProbitResults
   CountResults
   MultinomialResults
   NegativeBinomialResults
   GeneralizedPoissonResults

.. currentmodule:: statsmodels.discrete.count_model

.. autosummary::
   :toctree: generated/

   ZeroInflatedPoissonResults
   ZeroInflatedNegativeBinomialResults
   ZeroInflatedGeneralizedPoissonResults


:class:`DiscreteModel` is a superclass of all discrete regression models. The
estimation results are returned as an instance of one of the subclasses of
:class:`DiscreteResults`. Each category of models, binary, count and
multinomial, have their own intermediate level of model and results classes.
This intermediate classes are mostly to facilitate the implementation of the
methods and attributes defined by :class:`DiscreteModel` and
:class:`DiscreteResults`.

.. currentmodule:: statsmodels.discrete.discrete_model

.. autosummary::
   :toctree: generated/

   DiscreteModel
   DiscreteResults
   BinaryModel
   BinaryResults
   CountModel
   MultinomialModel

.. currentmodule:: statsmodels.discrete.count_model

.. autosummary::
   :toctree: generated/

   GenericZeroInflated
