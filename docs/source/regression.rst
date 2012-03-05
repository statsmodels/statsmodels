.. currentmodule:: statsmodels.regression.linear_model


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
* WLS : weighted least squares for heteroskedastic errors :math:`\text{diag}\left  (\Sigma\right)`
* GLSAR : feasible generalized least squares with autocorrelated AR(p) errors
  :math:`\Sigma=\Sigma\left(\rho\right)`

All regression models define the same methods and follow the same structure,
and can be used in a similar fashion. Some of them contain additional model
specific methods and attributes.

GLS is the superclass of the other regression classes.

Class hierachy: TODO

yule_walker is not a full model class, but a function that estimate the
parameters of a univariate autoregressive process, AR(p). It is used in GLSAR,
but it can also be used independently of any models. yule_walker only
calculates the estimates and the standard deviation of the lag parameters but
not the additional regression statistics. We hope to include yule-walker in
future in a separate univariate time series class. A similar result can be
obtained with GLSAR if only the constant is included as regressors. In this
case the parameter estimates of the lag estimates are not reported, however
additional statistics, for example aic, become available.


Attributes
^^^^^^^^^^
The following is more verbose description of the attributes which is mostly common to all
regression classes

pinv_wexog : array
    | `pinv_wexog` is the `p` x `n` Moore-Penrose pseudoinverse of the
    | whitened design matrix. Approximately equal to
    | :math:`\left(X^{T}\Sigma^{-1}X\right)^{-1}X^{T}\Psi`
    | where :math:`\Psi` is given by :math:`\Psi\Psi^{T}=\Sigma^{-1}`
cholsimgainv : array
    | n x n upper triangular matrix such that
    | :math:`\Psi\Psi^{T}=\Sigma^{-1}`
    | :math:`cholsigmainv=\Psi^{T}`
df_model : float
    The model degrees of freedom is equal to `p` - 1, where `p` is the number
    of regressors.  Note that the intercept is not counted as using a degree
    of freedom here.
df_resid : float
    The residual degrees of freedom is equal to the number of observations
    `n` less the number of parameters `p`.  Note that the intercept is counted as
    using a degree of freedom here.
llf : float
    The value of the likelihood function of the fitted model.
nobs : float
    The number of observations `n`
normalized_cov_params : array
    | A `p` x `p` array
    | :math:`(X^{T}\Sigma^{-1}X)^{-1}`
sigma : array
    | `sigma` is the n x n strucutre of the covariance matrix of the error terms
    | :math:`\mu\sim N\left(0,\sigma^{2}\Sigma\right)`
wexog : array
    | `wexog` is the whitened design matrix.
    | :math:`\Psi^{T}X`
wendog : array
    | The whitened response variable.
    | :math:`\Psi^{T}Y`



References
^^^^^^^^^^

General reference for regression models::

    D.C. Montgomery and E.A. Peck. "Introduction to Linear Regression
        Analysis." 2nd. Ed., Wiley, 1992.

Econometrics references for regression models::

    R. Davidson and J.G. MacKinnon. "Econometric Theory and Methods," Oxford,
        2004.

    W. Green.  "Econometric Analysis," 5th ed., Pearson, 2003.

Examples
--------

see also the `examples` and the `tests` folders


Module Reference
----------------

Model Classes
^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   OLS
   GLS
   WLS
   GLSAR
   yule_walker

Results Class
^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   RegressionResults


Technical Documentation
-----------------------

.. toctree::
   :maxdepth: 1

   regression_techn1
