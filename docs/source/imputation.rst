.. module:: statsmodels.imputation.mice
.. currentmodule:: statsmodels.imputation.mice

.. _imputation:


Multiple Imputation with Chained Equations
==========================================

The MICE module allows most Statsmodels models to be fit to a dataset
with missing values on the independent and/or dependent variables, and
provides rigorous standard errors for the fitted parameters.  The
basic idea is to treat each variable with missing values as the
dependent variable in a regression, with some or all of the remaining
variables as its predictors.  The MICE procedure cycles through these
models, fitting each in turn, then uses a procedure called "predictive
mean matching" (PMM) to generate random draws from the predictive
distributions determined by the fitted models.  These random draws
become the imputed values for one imputed data set.

By default, each variable with missing variables is modeled using a
linear regression with main effects for all other variables in the
data set.  Note that even when the imputation model is linear, the PMM
procedure preserves the domain of each variable.  Thus, for example,
if all observed values for a given variable are positive, all imputed
values for the variable will always be positive.  The user also has
the option to specify which model is used to produce imputed values
for each variable.

.. code


Classes
-------

.. currentmodule:: statsmodels.imputation.mice

.. autosummary::
   :toctree: generated/

   MICE
   MICEData


Implementation Details
----------------------

Internally, this function uses
`pandas.isnull <http://pandas.pydata.org/pandas-docs/stable/missing_data.html#working-with-missing-data>`_.
Anything that returns True from this function will be treated as missing data.
