
.. module:: statsmodels.othermod
.. currentmodule:: statsmodels.othermod


.. _othermod:


Other Models :mod:`othermod`
==============================

:mod:`statsmodels.othermod` contains model classes that do not fit into
any other category, for example models for a response variable ``endog`` that
has support on the unit interval or is positive or non-negative.

:mod:`statsmodels.othermod` contains models that are, or will be fully developed
in contrast to :mod:`statsmodels.miscmodels` which contains mainly examples
for the use of the generic likelihood model setup.

Status is experimental. The api and implementation will need to adjust as we
support more types of models, for example models with multiple exog and
multiple link functions.


Interval Models :mod:`betareg`
------------------------------

Models for continuous dependent variables that are in the unit interval such
as fractions. These Models are estimated by full Maximum Likelihood. 
Dependent variables on the unit interval can also be estimate by 
Quasi Maximum Likelihood using models for binary endog, such as Logit and
GLM-Binomial. (The implementation of discrete.Probit assumes binary endog and
cannot estimate a QMLE for continuous dependent variable.)

.. module:: statsmodels.othermod.betareg
.. currentmodule:: statsmodels.othermod.betareg

.. autosummary::
   :toctree: generated/

   BetaModel
   BetaResults
