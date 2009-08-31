.. currentmodule:: scikits.statsmodels.regression


.. _regression:

Regression
==========


Introduction
------------

Regression contains linear models with independently and identically
distributed errors and for errors with heteroscedasticit or autocorrelation

The statistical model is assumed to be

y = X b + u,  where u is distributed with mean 0 and covariance \Sigma

depending on the assumption on V, we have currently four classes available

* GLS : generalized least squares for arbitrary covariance V
* OLS : ordinary least squares for i.i.d. errors
* WLS : eighted least squares for heteroscedastic errors
* GLSAR : feasible generalized least squares with autocorrelated AR(p) errors

All regression models define the same methods and follow the same structure,
and can be used in a similar fashion. Some of them contain additional model
spedific methods and attributes.

GLS is the superclass of the other regression classes. class hierachy

yule_walker is not a full model class, but a function that estimate the
parameters of a univariate autoregressive process, AR(p). It is used in GLSAR,
but it can also be used independently of any models. yule_walker only
calculates the estimates and the standard deviation of the lag parameters but
not the additional regression statistics. We hope to include yule-walker in
future in a separate univariate time series class. A similar result can be
obtained with GLSAR if only the constant is included as regressors. In this
case the parameter estimates of the lag estimates are not reported, however
additional statistics, for example aic, become available.

Examples
--------



Module Reference
----------------

.. autosummary::
   :toctree: generated/

   OLS
   GLS
   WLS
   GLSAR
   yule_walker


Technical Documentation
-----------------------

.. toctree::
   :maxdepth: 1

   regression_techn1
