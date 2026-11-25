.. module:: statsmodels.extremes
.. currentmodule:: statsmodels.extremes

.. _extremes:


Extreme Value Models :mod:`extremes`
====================================

:mod:`statsmodels.extremes` contains models from extreme value theory in
which the parameters of an extreme value distribution depend on explanatory
variables.

Status is experimental. The api can change.


Threshold Exceedances :mod:`generalized_pareto`
-----------------------------------------------

Exceedances of a variable over a high threshold are approximately
generalized Pareto distributed. ``GeneralizedParetoModel`` estimates the
generalized Pareto distribution by maximum likelihood, with the log of the
scale and, optionally, the shape as linear functions of explanatory
variables. Choosing the threshold and declustering dependent exceedances are
left to the user.

.. module:: statsmodels.extremes.generalized_pareto
.. currentmodule:: statsmodels.extremes.generalized_pareto

.. autosummary::
   :toctree: generated/

   GeneralizedParetoModel
   GeneralizedParetoResults


References
----------

Coles, Stuart. An Introduction to Statistical Modeling of Extreme Values.
Springer, 2001.

Davison, A. C., and R. L. Smith. "Models for Exceedances over High
Thresholds." Journal of the Royal Statistical Society, Series B 52, no. 3
(1990): 393-442.
