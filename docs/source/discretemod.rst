.. currentmodule:: scikits.statsmodels.discrete.discrete_model


.. _discretemod:

Regression with Discrete Dependent Variable
===========================================

Note: These models have just been moved out of the sandbox. Large parts of the
statistical results are verified and tested, but this module has not seen much
use yet and we can still expect some changes.


Introduction
------------

:mod:discretemod contains regression models for limited dependent and
qualitative variables.

This currently includes models when the dependent variable is discrete,
either binary (Logit, Probit), (ordered) ordinal data (MNLogit) or
count data (Poisson). Currently all models are estimated by Maximum Likelihood
and assume independently and identically distributed errors.

All discrete regression models define the same methods and follow the same
structure, which is similar to the regression results but with some methods
specific to discrete models. Additionally some of them contain additional model
specific methods and attributes.

Example::

  # Load the data from Spector and Mazzeo (1980)
  spector_data = sm.datasets.spector.load()
  spector_data.exog = sm.add_constant(spector_data.exog)

  # Linear Probability Model using OLS
  lpm_mod = sm.OLS(spector_data.endog,spector_data.exog)
  lpm_res = lpm_mod.fit()

  # Logit Model
  logit_mod = sm.Logit(spector_data.endog, spector_data.exog)
  logit_res = logit_mod.fit()

  # Probit Model
  probit_mod = sm.Probit(spector_data.endog, spector_data.exog)
  probit_res = probit_mod.fit()

  # Since the parameters have different parameterization across non-linear
  # models, we can use the average marginal effect instead to compare the
  # models results.

  >>> lpm_res.params[:-1]
  array([ 0.46385168,  0.01049512,  0.37855479])
  >>> logit_res.margeff()
  array([ 0.36258083,  0.01220841,  0.3051777 ])
  >>> probit_res.margeff()
  array([ 0.36078629,  0.01147926,  0.31651986])



References
^^^^^^^^^^

General references for this class of models are::

    A.C. Cameron and P.K. Trivedi.  `Regression Analysis of Count Data`.
        Cambridge, 1998

    G.S. Madalla. `Limited-Dependent and Qualitative Variables in Econometrics`.
        Cambridge, 1983.

    W. Greene. `Econometric Analysis`. Prentice Hall, 5th. edition. 2003.


Examples
^^^^^^^^

see the `examples` and the `tests` folders, and the docstrings of the
individual model classes.


Module Reference
----------------

The specific model classes are:

.. autosummary::
   :toctree: generated/

   Logit
   Probit
   MNLogit
   Poisson

:class:`DiscreteModel` is a superclass of all discrete regression models. The
estimation results are returned as an instance of :class:`DiscreteResults`

.. autosummary::
   :toctree: generated/

   DiscreteModel
   DiscreteResults

