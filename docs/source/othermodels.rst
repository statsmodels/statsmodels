
.. module:: statsmodels.othermod
.. currentmodule:: statsmodels.othermod


.. _othermod:


Other Models :mod:`othermod`
==============================

:mod:`statsmodels.other` contains model classes and that do not fit into
any other category, for example models for a response variable ``endog`` that
has support on the unit interval or is positive or non-negative.

:mod:`statsmodels.other` contains models that are, or will be fully developed
in contrast to :mod:`statsmodels.miscmodels` which contains mainly examples
for the use of the generic likelihood model setup.

Status is experimental. The api and implementation will need to adjust as we
support more types of models, for example models with multiple exog and
multiple link functions.


Interval Models :mod:`betareg`
------------------------------

.. module:: statsmodels.othermod.betareg
.. currentmodule:: statsmodels.othermod.betareg

.. autosummary::
   :toctree: generated/

   BetaModel
   BetaResults
