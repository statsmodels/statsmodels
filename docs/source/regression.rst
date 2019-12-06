.. currentmodule:: statsmodels.regression.linear_model


.. _regression:

Linear Regression
=================

Linear models with independently and identically distributed errors, and for
errors with heteroscedasticity or autocorrelation. This module allows
estimation by ordinary least squares (OLS), weighted least squares (WLS),
generalized least squares (GLS), and feasible generalized least squares with
autocorrelated AR(p) errors.

See `Module Reference`_ for commands and arguments.

Examples
--------

.. ipython:: python

    # Load modules and data
    import numpy as np
    import statsmodels.api as sm
    spector_data = sm.datasets.spector.load(as_pandas=False)
    spector_data.exog = sm.add_constant(spector_data.exog, prepend=False)

    # Fit and summarize OLS model
    mod = sm.OLS(spector_data.endog, spector_data.exog)
    res = mod.fit()
    print(res.summary())

Detailed examples can be found here:


* `OLS <examples/notebooks/generated/ols.html>`__
* `WLS <examples/notebooks/generated/wls.html>`__
* `GLS <examples/notebooks/generated/gls.html>`__
* `Recursive LS <examples/notebooks/generated/recursive_ls.html>`__
* `Rolling LS <examples/notebooks/generated/rolling_ls.html>`__

Technical Documentation
-----------------------

The statistical model is assumed to be

 :math:`Y = X\beta + \mu`,  where :math:`\mu\sim N\left(0,\Sigma\right).`

Depending on the properties of :math:`\Sigma`, we have currently four classes available:

* GLS : generalized least squares for arbitrary covariance :math:`\Sigma`
* OLS : ordinary least squares for i.i.d. errors :math:`\Sigma=\textbf{I}`
* WLS : weighted least squares for heteroskedastic errors :math:`\text{diag}\left  (\Sigma\right)`
* GLSAR : feasible generalized least squares with autocorrelated AR(p) errors
  :math:`\Sigma=\Sigma\left(\rho\right)`

All regression models define the same methods and follow the same structure,
and can be used in a similar fashion. Some of them contain additional model
specific methods and attributes.

GLS is the superclass of the other regression classes except for RecursiveLS,
RollingWLS and RollingOLS.

.. Class hierachy: TODO

.. yule_walker is not a full model class, but a function that estimate the
.. parameters of a univariate autoregressive process, AR(p). It is used in GLSAR,
.. but it can also be used independently of any models. yule_walker only
.. calculates the estimates and the standard deviation of the lag parameters but
.. not the additional regression statistics. We hope to include yule-walker in
.. future in a separate univariate time series class. A similar result can be
.. obtained with GLSAR if only the constant is included as regressors. In this
.. case the parameter estimates of the lag estimates are not reported, however
.. additional statistics, for example aic, become available.

References
^^^^^^^^^^

General reference for regression models:

* D.C. Montgomery and E.A. Peck. "Introduction to Linear Regression Analysis." 2nd. Ed., Wiley, 1992.

Econometrics references for regression models:

* R.Davidson and J.G. MacKinnon. "Econometric Theory and Methods," Oxford, 2004.
* W.Green. "Econometric Analysis," 5th ed., Pearson, 2003.

.. toctree::
..   :maxdepth: 1
..
..   regression_techn1

Attributes
^^^^^^^^^^

The following is more verbose description of the attributes which is mostly
common to all regression classes

pinv_wexog : array
    The `p` x `n` Moore-Penrose pseudoinverse of the whitened design matrix.
    It is approximately equal to
    :math:`\left(X^{T}\Sigma^{-1}X\right)^{-1}X^{T}\Psi`, where
    :math:`\Psi` is defined such that :math:`\Psi\Psi^{T}=\Sigma^{-1}`.
cholsimgainv : array
    The `n` x `n` upper triangular matrix :math:`\Psi^{T}` that satisfies
    :math:`\Psi\Psi^{T}=\Sigma^{-1}`.
df_model : float
    The model degrees of freedom. This is equal to `p` - 1, where `p` is the
    number of regressors. Note that the intercept is not counted as using a
    degree of freedom here.
df_resid : float
    The residual degrees of freedom. This is equal `n - p` where `n` is the
    number of observations and `p` is the number of parameters. Note that the
    intercept is counted as using a degree of freedom here.
llf : float
    The value of the likelihood function of the fitted model.
nobs : float
    The number of observations `n`
normalized_cov_params : array
    A `p` x `p` array equal to :math:`(X^{T}\Sigma^{-1}X)^{-1}`.
sigma : array
    The `n` x `n` covariance matrix of the error terms:
    :math:`\mu\sim N\left(0,\Sigma\right)`.
wexog : array
    The whitened design matrix :math:`\Psi^{T}X`.
wendog : array
    The whitened response variable :math:`\Psi^{T}Y`.

Module Reference
----------------

.. module:: statsmodels.regression.linear_model
   :synopsis: Least squares linear models

Model Classes
^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   OLS
   GLS
   WLS
   GLSAR
   yule_walker
   burg

.. module:: statsmodels.regression.quantile_regression
   :synopsis: Quantile regression

.. currentmodule:: statsmodels.regression.quantile_regression

.. autosummary::
   :toctree: generated/

   QuantReg

.. module:: statsmodels.regression.recursive_ls
   :synopsis: Recursive least squares using the Kalman Filter

.. currentmodule:: statsmodels.regression.recursive_ls

.. autosummary::
   :toctree: generated/

   RecursiveLS

.. module:: statsmodels.regression.rolling
   :synopsis: Rolling (moving) least squares

.. currentmodule:: statsmodels.regression.rolling

.. autosummary::
   :toctree: generated/

   RollingWLS
   RollingOLS

.. module:: statsmodels.regression.process_regression
   :synopsis: Process regression

.. currentmodule:: statsmodels.regression.process_regression

.. autosummary::
   :toctree: generated/

   GaussianCovariance
   ProcessMLE

.. module:: statsmodels.regression.dimred
   :synopsis: Dimension reduction methods

.. currentmodule:: statsmodels.regression.dimred

.. autosummary::
   :toctree: generated/

    SlicedInverseReg
    PrincipalHessianDirections
    SlicedAverageVarianceEstimation


Results Classes
^^^^^^^^^^^^^^^

Fitting a linear regression model returns a results class. OLS has a
specific results class with some additional methods compared to the
results class of the other linear models.

.. currentmodule:: statsmodels.regression.linear_model

.. autosummary::
   :toctree: generated/

   RegressionResults
   OLSResults
   PredictionResults

.. currentmodule:: statsmodels.base.elastic_net

.. autosummary::
   :toctree: generated/

    RegularizedResults

.. currentmodule:: statsmodels.regression.quantile_regression

.. autosummary::
   :toctree: generated/

   QuantRegResults

.. currentmodule:: statsmodels.regression.recursive_ls

.. autosummary::
   :toctree: generated/

   RecursiveLSResults

.. currentmodule:: statsmodels.regression.rolling

.. autosummary::
   :toctree: generated/

   RollingRegressionResults

.. currentmodule:: statsmodels.regression.process_regression

.. autosummary::
   :toctree: generated/

   ProcessMLEResults

.. currentmodule:: statsmodels.regression.dimred

.. autosummary::
   :toctree: generated/

   DimReductionResults
