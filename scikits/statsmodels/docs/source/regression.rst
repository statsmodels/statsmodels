.. currentmodule:: scikits.statsmodels.regression


.. _regression:

Regression
==========


Introduction
------------

Regression contains linear models with independently and identically
distributed errors and for errors with heteroscedasticity or autocorrelation

The statistical model is assumed to be

 :math:`Y = X\beta + \mu`,  where :math:`\mu\sim N\left(0,\sigma^{2}\Sigma\right)`

depending on the assumption on :math:`\Sigma`, we have currently four classes available

* GLS : generalized least squares for arbitrary covariance :math:`\Sigma`
* OLS : ordinary least squares for i.i.d. errors :math:`\Sigma=\textbf{I}`
* WLS : weighted least squares for heteroskedastic errors :math:`\text{diag}\left(\Sigma\right)`
* GLSAR : feasible generalized least squares with autocorrelated AR(p) errors :math:`\Sigma=\Sigma\left(\rho\right)`

All regression models define the same methods and follow the same structure,
and can be used in a similar fashion. Some of them contain additional model
spedific methods and attributes.

GLS is the superclass of the other regression classes.

Class hierachy:

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
