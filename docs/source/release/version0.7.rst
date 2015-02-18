:orphan:

===========
0.7 Release
===========

Release 0.7.0
=============

Release summary

The following major new features appear in this version.

Principal Component Analysis
----------------------------

A new class-based Principal Component Analysis has been added.  This
class replaces the function-based PCA that previously existed in the
sandbox.  This change bring a number of new features, including:

* Options to control the standardization (demeaning/studentizing)
* Scree plotting
* Information criteria for selecting the number of factors
* R-squared plots to assess component fit
* NIPALS implementation when only a small number of components are required and the dataset is large
* Missing-value filling using the EM algorithm

.. code-block:: python

   import statsmodels.api as sm
   from statsmodels.tools.pca import PCA

   data = sm.datasets.fertility.load_pandas().data

   columns = map(str, range(1960, 2012))
   data.set_index('Country Name', inplace=True)
   dta = data[columns]
   dta = dta.dropna()

   pca_model = PCA(dta.T, standardize=False, demean=True)
   pca_model.plot_scree()

*Note* : A function version is also available which is compatible with the
call in the sandbox.  The function version is just a thin wrapper around the
class-based PCA implementation.

Regression graphics for GLM/GEE
-------------------------------

Added variable plots, partial residual plots, and CERES residual plots
are available for GLM and GEE models by calling the methods
`plot_added_variable`, `plot_partial_residuals`, and
`plot_ceres_residuals` that are attached to the results classes.

State Space Models
------------------

State space methods provide a flexible structure for the estimation and
analysis of a wide class of time series models. The Statsmodels implementation
allows specification of state models, fast Kalman filtering, and built-in
methods to facilitate maximum likelihood estimation of arbitrary models. One of
the primary goals of this module is to allow end users to create and estimate
their own models. Below is a short example demonstrating the ease with which a
local level model can be specified and estimated:

.. code-block:: python

   import statsmodels.api as sm
   import pandas as pd

   data = sm.datasets.nile.load_pandas().data
   data.index = pd.DatetimeIndex(data.year.astype(int).astype(str), freq='AS')

   # Setup the state space representation
   mod = sm.tsa.statespace.MLEModel(data['volume'], k_states=1)
   mod['design', :] = mod['transition', :] = mod['selection', :] = 1.
   mod.initialize_approximate_diffuse()

   # Describe how parameters enter the model
   def updater(mod, params):
       mod['obs_cov',0,0] = params[0]
       mod['state_cov',0,0] = params[1]
   mod.updater = updater
   mod.transformer = lambda mod, params: params**2

   # Specify start parameters and parameter names
   mod.start_params = [1e2,1e2]
   mod.param_names = ['sigma2.measurement', 'sigma2.level']

   # Fit the model with maximum likelihood estimation
   res = mod.fit()
   print res.summary()

The documentation and example notebooks provide further examples of how to
create ad-hoc models (as above) as well as more complex models. Included in
this release is a full-fledged model making use of the state space
infrastructure to estimate SARIMAX models. See below for more details.

Time Series Models (ARIMA) with Seasonal Effects
------------------------------------------------

A model for estimating seasonal autoregressive integrated moving average models
with exogenous regressors (SARIMAX) has been added by taking advantage of the
new state space functionality. It can be used very similarly to the existing
`ARIMA` model, but works on a wider range of specifications, including:

* Additive and multiplicative seasonal effects
* Flexible trend specications
* Regression with SARIMA errors
* Regression with time-varying coefficients
* Measurement error in the endogenous variables

Below is a short example fitting a model with a number of these components,
including exogenous data, a linear trend, and annual multiplicative seasonal
effects.

.. code-block:: python

   import statsmodels.api as sm
   import pandas as pd

   data = sm.datasets.macrodata.load_pandas().data
   data.index = pd.DatetimeIndex(start='1959-01-01', end='2009-09-01', freq='QS')
   endog = data['realcons']
   exog = data['m1']

   mod = sm.tsa.SARIMAX(endog, exog=exog, order=(1,1,1),
                        trend='t', seasonal_order=(0,0,1,4))
   res = mod.fit()
   print res.summary()

Other important new features
----------------------------

* Bullet
* List
* of
* new
* features

Major Bugs fixed
----------------

* Bullet
* list
* use :ghissue:`XXX` to link to issue.

Backwards incompatible changes and deprecations
-----------------------------------------------

* List backwards incompatible changes

Development summary and credits
-------------------------------

A blurb about the number of changes and the contributors list.

.. note::

   Obtained by running ``git log v0.6.0..HEAD --format='* %aN <%aE>' | sed 's/@/\-at\-/' | sed 's/<>//' | sort -u``.

