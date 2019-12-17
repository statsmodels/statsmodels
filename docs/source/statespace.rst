.. module:: statsmodels.tsa.statespace
   :synopsis: Statespace models for time-series analysis

.. currentmodule:: statsmodels.tsa.statespace


.. _statespace:


Time Series Analysis by State Space Methods :mod:`statespace`
=============================================================

:mod:`statsmodels.tsa.statespace` contains classes and functions that are
useful for time series analysis using state space methods.

A general state space model is of the form

.. math::

  y_t & = Z_t \alpha_t + d_t + \varepsilon_t \\
  \alpha_{t+1} & = T_t \alpha_t + c_t + R_t \eta_t \\

where :math:`y_t` refers to the observation vector at time :math:`t`,
:math:`\alpha_t` refers to the (unobserved) state vector at time
:math:`t`, and where the irregular components are defined as

.. math::

  \varepsilon_t \sim N(0, H_t) \\
  \eta_t \sim N(0, Q_t) \\

The remaining variables (:math:`Z_t, d_t, H_t, T_t, c_t, R_t, Q_t`) in the
equations are matrices describing the process. Their variable names and
dimensions are as follows

Z : `design`          :math:`(k\_endog \times k\_states \times nobs)`

d : `obs_intercept`   :math:`(k\_endog \times nobs)`

H : `obs_cov`         :math:`(k\_endog \times k\_endog \times nobs)`

T : `transition`      :math:`(k\_states \times k\_states \times nobs)`

c : `state_intercept` :math:`(k\_states \times nobs)`

R : `selection`       :math:`(k\_states \times k\_posdef \times nobs)`

Q : `state_cov`       :math:`(k\_posdef \times k\_posdef \times nobs)`

In the case that one of the matrices is time-invariant (so that, for
example, :math:`Z_t = Z_{t+1} ~ \forall ~ t`), its last dimension may
be of size :math:`1` rather than size `nobs`.

This generic form encapsulates many of the most popular linear time series
models (see below) and is very flexible, allowing estimation with missing
observations, forecasting, impulse response functions, and much more.

**Example: AR(2) model**

An autoregressive model is a good introductory example to putting models in
state space form. Recall that an AR(2) model is often written as:

.. math::

   y_t = \phi_1 y_{t-1} + \phi_2 y_{t-2} + \epsilon_t,
   \quad \epsilon_t \sim N(0, \sigma^2)

This can be put into state space form in the following way:

.. math::

   y_t & = \begin{bmatrix} 1 & 0 \end{bmatrix} \alpha_t \\
   \alpha_{t+1} & = \begin{bmatrix}
      \phi_1 & \phi_2 \\
           1 &      0
   \end{bmatrix} \alpha_t + \begin{bmatrix} 1 \\ 0 \end{bmatrix} \eta_t

Where

.. math::

   Z_t \equiv Z = \begin{bmatrix} 1 & 0 \end{bmatrix}

and

.. math::

   T_t \equiv T & = \begin{bmatrix}
      \phi_1 & \phi_2 \\
           1 &      0
   \end{bmatrix} \\
   R_t \equiv R & = \begin{bmatrix} 1 \\ 0 \end{bmatrix} \\
   \eta_t \equiv \epsilon_{t+1} & \sim N(0, \sigma^2)

There are three unknown parameters in this model:
:math:`\phi_1, \phi_2, \sigma^2`.

Models and Estimation
---------------------

The following are the main estimation classes, which can be accessed through
`statsmodels.tsa.statespace.api` and their result classes.

Seasonal Autoregressive Integrated Moving-Average with eXogenous regressors (SARIMAX)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `SARIMAX` class is an example of a fully fledged model created using the
statespace backend for estimation. `SARIMAX` can be used very similarly to
:ref:`tsa <tsa>` models, but works on a wider range of models by adding the
estimation of additive and multiplicative seasonal effects, as well as
arbitrary trend polynomials.

.. autosummary::
   :toctree: generated/

   sarimax.SARIMAX
   sarimax.SARIMAXResults

For an example of the use of this model, see the
`SARIMAX example notebook <examples/notebooks/generated/statespace_sarimax_stata.html>`__
or the very brief code snippet below:


.. code-block:: python

   # Load the statsmodels api
   import statsmodels.api as sm

   # Load your dataset
   endog = pd.read_csv('your/dataset/here.csv')

   # We could fit an AR(2) model, described above
   mod_ar2 = sm.tsa.SARIMAX(endog, order=(2,0,0))
   # Note that mod_ar2 is an instance of the SARIMAX class

   # Fit the model via maximum likelihood
   res_ar2 = mod_ar2.fit()
   # Note that res_ar2 is an instance of the SARIMAXResults class

   # Show the summary of results
   print(res_ar2.summary())

   # We could also fit a more complicated model with seasonal components.
   # As an example, here is an SARIMA(1,1,1) x (0,1,1,4):
   mod_sarimax = sm.tsa.SARIMAX(endog, order=(1,1,1),
                                seasonal_order=(0,1,1,4))
   res_sarimax = mod_sarimax.fit()

   # Show the summary of results
   print(res_sarimax.summary())

The results object has many of the attributes and methods you would expect from
other statsmodels results objects, including standard errors, z-statistics,
and prediction / forecasting.

Behind the scenes, the `SARIMAX` model creates the design and transition
matrices (and sometimes some of the other matrices) based on the model
specification.

Unobserved Components
^^^^^^^^^^^^^^^^^^^^^

The `UnobservedComponents` class is another example of a statespace model.

.. autosummary::
   :toctree: generated/

   structural.UnobservedComponents
   structural.UnobservedComponentsResults

For examples of the use of this model, see the `example notebook <examples/notebooks/generated/statespace_structural_harvey_jaeger.html>`__ or a notebook on using the unobserved components model to `decompose a time series into a trend and cycle <examples/notebooks/generated/statespace_cycles.html>`__ or the very brief code snippet below:

.. code-block:: python

   # Load the statsmodels api
   import statsmodels.api as sm

   # Load your dataset
   endog = pd.read_csv('your/dataset/here.csv')

   # Fit a local level model
   mod_ll = sm.tsa.UnobservedComponents(endog, 'local level')
   # Note that mod_ll is an instance of the UnobservedComponents class

   # Fit the model via maximum likelihood
   res_ll = mod_ll.fit()
   # Note that res_ll is an instance of the UnobservedComponentsResults class

   # Show the summary of results
   print(res_ll.summary())

   # Show a plot of the estimated level and trend component series
   fig_ll = res_ll.plot_components()

   # We could further add a damped stochastic cycle as follows
   mod_cycle = sm.tsa.UnobservedComponents(endog, 'local level', cycle=True,
                                           damped_cycle=true,
                                           stochastic_cycle=True)
   res_cycle = mod_cycle.fit()

   # Show the summary of results
   print(res_cycle.summary())

   # Show a plot of the estimated level, trend, and cycle component series
   fig_cycle = res_cycle.plot_components()

Vector Autoregressive Moving-Average with eXogenous regressors (VARMAX)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `VARMAX` class is an example of a multivariate statespace model.

.. autosummary::
   :toctree: generated/

   varmax.VARMAX
   varmax.VARMAXResults

For an example of the use of this model, see the `VARMAX example notebook <examples/notebooks/generated/statespace_varmax.html>`__ or the very brief code snippet below:

.. code-block:: python

   # Load the statsmodels api
   import statsmodels.api as sm

   # Load your (multivariate) dataset
   endog = pd.read_csv('your/dataset/here.csv')

   # Fit a local level model
   mod_var1 = sm.tsa.VARMAX(endog, order=(1,0))
   # Note that mod_var1 is an instance of the VARMAX class

   # Fit the model via maximum likelihood
   res_var1 = mod_var1.fit()
   # Note that res_var1 is an instance of the VARMAXResults class

   # Show the summary of results
   print(res_var1.summary())

   # Construct impulse responses
   irfs = res_ll.impulse_responses(steps=10)

Dynamic Factor Models
^^^^^^^^^^^^^^^^^^^^^

The `DynamicFactor` class is another example of a multivariate statespace
model.

.. autosummary::
   :toctree: generated/

   dynamic_factor.DynamicFactor
   dynamic_factor.DynamicFactorResults

For an example of the use of this model, see the `Dynamic Factor example notebook <examples/notebooks/generated/statespace_dfm_coincident.html>`__ or the very brief code snippet below:

.. code-block:: python

   # Load the statsmodels api
   import statsmodels.api as sm

   # Load your dataset
   endog = pd.read_csv('your/dataset/here.csv')

   # Fit a local level model
   mod_dfm = sm.tsa.DynamicFactor(endog, k_factors=1, factor_order=2)
   # Note that mod_dfm is an instance of the DynamicFactor class

   # Fit the model via maximum likelihood
   res_dfm = mod_dfm.fit()
   # Note that res_dfm is an instance of the DynamicFactorResults class

   # Show the summary of results
   print(res_ll.summary())

   # Show a plot of the r^2 values from regressions of
   # individual estimated factors on endogenous variables.
   fig_dfm = res_ll.plot_coefficients_of_determination()

Linear Exponential Smoothing Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `ExponentialSmoothing` class is an implementation of linear exponential
smoothing models using a state space approach.

**Note**: this model is available at `sm.tsa.statespace.ExponentialSmoothing`;
it is not the same as the model available at `sm.tsa.ExponentialSmoothing`.
See below for details of the differences between these classes.

.. autosummary::
   :toctree: generated/

   exponential_smoothing.ExponentialSmoothing
   exponential_smoothing.ExponentialSmoothingResults

A very brief code snippet follows:

.. code-block:: python

   # Load the statsmodels api
   import statsmodels.api as sm

   # Load your dataset
   endog = pd.read_csv('your/dataset/here.csv')

   # Simple exponential smoothing, denoted (A,N,N)
   mod_ses = sm.tsa.statespace.ExponentialSmoothing(endog)
   res_ses = mod_ses.fit()

   # Holt's linear method, denoted (A,A,N)
   mod_h = sm.tsa.statespace.ExponentialSmoothing(endog, trend=True)
   res_h = mod_h.fit()

   # Damped trend model, denoted (A,Ad,N)
   mod_dt = sm.tsa.statespace.ExponentialSmoothing(endog, trend=True,
                                                   damped_trend=True)
   res_dt = mod_dt.fit()

   # Holt-Winters' trend and seasonality method, denoted (A,A,A)
   # (assuming that `endog` has a seasonal periodicity of 4, for example if it
   # is quarterly data).
   mod_hw = sm.tsa.statespace.ExponentialSmoothing(endog, trend=True,
                                                   seasonal=4)
   res_hw = mod_hw.fit()

**Differences between Statsmodels' exponential smoothing model classes**

There are several differences between this model class, available at
`sm.tsa.statespace.ExponentialSmoothing`, and the model class available at
`sm.tsa.ExponentialSmoothing`.

- This model class only supports *linear* exponential smoothing models, while
  `sm.tsa.ExponentialSmoothing` also supports multiplicative models.
- This model class puts the exponential smoothing models into state space form
  and then applies the Kalman filter to estimate the states, while
  `sm.tsa.ExponentialSmoothing` is based on exponential smoothing recursions.
  In some cases, this can mean that estimating parameters with this model class
  will be somewhat slower than with `sm.tsa.ExponentialSmoothing`.
- This model class can produce confidence intervals for forecasts, based on an
  assumption of Gaussian errors, while `sm.tsa.ExponentialSmoothing` does not
  support confidence intervals.
- This model class supports concentrating initial values out of the objective
  function, which can improve performance when there are many initial states to
  estimate (for example when the seasonal periodicity is large).
- This model class supports many advanced features available to state space
  models, such as diagnostics and fixed parameters.

**Note**: this class is based on a "multiple sources of error" (MSOE) state
space formulation and not a "single source of error" (SSOE) formulation.

Custom state space models
^^^^^^^^^^^^^^^^^^^^^^^^^

The true power of the state space model is to allow the creation and estimation
of custom models. Usually that is done by extending the following two classes,
which bundle all of state space representation, Kalman filtering, and maximum
likelihood fitting functionality for estimation and results output.

.. autosummary::
   :toctree: generated/

   mlemodel.MLEModel
   mlemodel.MLEResults

For a basic example demonstrating creating and estimating a custom state space
model, see the `Local Linear Trend example notebook <examples/notebooks/generated/statespace_local_linear_trend.html>`__.
For a more sophisticated example, see the source code for the `SARIMAX` and
`SARIMAXResults` classes, which are built by extending `MLEModel` and
`MLEResults`.

In simple cases, the model can be constructed entirely using the MLEModel
class. For example, the AR(2) model from above could be constructed and
estimated using only the following code:

.. code-block:: python

   import numpy as np
   from scipy.signal import lfilter
   import statsmodels.api as sm

   # True model parameters
   nobs = int(1e3)
   true_phi = np.r_[0.5, -0.2]
   true_sigma = 1**0.5

   # Simulate a time series
   np.random.seed(1234)
   disturbances = np.random.normal(0, true_sigma, size=(nobs,))
   endog = lfilter([1], np.r_[1, -true_phi], disturbances)

   # Construct the model
   class AR2(sm.tsa.statespace.MLEModel):
       def __init__(self, endog):
           # Initialize the state space model
           super(AR2, self).__init__(endog, k_states=2, k_posdef=1,
                                     initialization='stationary')

           # Setup the fixed components of the state space representation
           self['design'] = [1, 0]
           self['transition'] = [[0, 0],
                                     [1, 0]]
           self['selection', 0, 0] = 1

       # Describe how parameters enter the model
       def update(self, params, transformed=True, **kwargs):
           params = super(AR2, self).update(params, transformed, **kwargs)

           self['transition', 0, :] = params[:2]
           self['state_cov', 0, 0] = params[2]

       # Specify start parameters and parameter names
       @property
       def start_params(self):
           return [0,0,1]  # these are very simple

   # Create and fit the model
   mod = AR2(endog)
   res = mod.fit()
   print(res.summary())

This results in the following summary table::

                              Statespace Model Results                           
   ==============================================================================
   Dep. Variable:                      y   No. Observations:                 1000
   Model:                            AR2   Log Likelihood               -1389.437
   Date:                Wed, 26 Oct 2016   AIC                           2784.874
   Time:                        00:42:03   BIC                           2799.598
   Sample:                             0   HQIC                          2790.470
                                  - 1000                                         
   Covariance Type:                  opg                                         
   ==============================================================================
                    coef    std err          z      P>|z|      [0.025      0.975]
   ------------------------------------------------------------------------------
   param.0        0.4395      0.030     14.730      0.000       0.381       0.498
   param.1       -0.2055      0.032     -6.523      0.000      -0.267      -0.144
   param.2        0.9425      0.042     22.413      0.000       0.860       1.025
   ===================================================================================
   Ljung-Box (Q):                       24.25   Jarque-Bera (JB):                 0.22
   Prob(Q):                              0.98   Prob(JB):                         0.90
   Heteroskedasticity (H):               1.05   Skew:                            -0.04
   Prob(H) (two-sided):                  0.66   Kurtosis:                         3.02
   ===================================================================================
   
   Warnings:
   [1] Covariance matrix calculated using the outer product of gradients (complex-step).

The results object has many of the attributes and methods you would expect from
other statsmodels results objects, including standard errors, z-statistics,
and prediction / forecasting.

More advanced usage is possible, including specifying parameter
transformations, and specifying names for parameters for a more informative
output summary.

Overview of usage
-----------------

All state space models follow the typical Statsmodels pattern:

1. Construct a **model instance** with an input dataset
2. Apply parameters to the model (for example, using `fit`) to construct a **results instance**
3. Interact with the results instance to examine the estimated parameters, explore residual diagnostics, and produce forecasts, simulations, or impulse responses.

An example of this pattern is as follows:

.. code-block:: python

  # Load in the example macroeconomic dataset
  dta = sm.datasets.macrodata.load_pandas().data
  # Make sure we have an index with an associated frequency, so that
  # we can refer to time periods with date strings or timestamps
  dta.index = pd.date_range('1959Q1', '2009Q3', freq='QS')

  # Step 1: construct an SARIMAX model for US inflation data
  model = sm.tsa.SARIMAX(dta.infl, order=(4, 0, 0), trend='c')

  # Step 2: fit the model's parameters by maximum likelihood
  results = model.fit()

  # Step 3: explore / use results

  # - Print a table summarizing estimation results
  print(results.summary())

  # - Print only the estimated parameters
  print(results.params)

  # - Create diagnostic figures based on standardized residuals:
  #   (1) time series graph
  #   (2) histogram
  #   (3) Q-Q plot
  #   (4) correlogram
  results.plot_diagnostics()

  # - Examine diagnostic hypothesis tests
  # Jarque-Bera: [test_statistic, pvalue, skewness, kurtosis]
  print(results.test_normality(method='jarquebera'))
  # Goldfeld-Quandt type test: [test_statistic, pvalue]
  print(results.test_heteroskedasticity(method='breakvar'))
  # Ljung-Box test: [test_statistic, pvalue] for each lag
  print(results.test_serial_correlation(method='ljungbox'))

  # - Forecast the next 4 values
  print(results.forecast(4))

  # - Forecast until 2020Q4
  print(results.forecast('2020Q4'))

  # - Plot in-sample dynamic prediction starting in 2005Q1
  #   and out-of-sample forecasts until 2010Q4 along with
  #   90% confidence intervals
  predict_results = results.get_prediction(start='2005Q1', end='2010Q4', dynamic=True)
  predict_df = predict_results.summary_frame(alpha=0.10)
  fig, ax = plt.subplots()
  predict_df['mean'].plot(ax=ax)
  ax.fill_between(predict_df.index, predict_df['mean_ci_lower'],
                  predict_df['mean_ci_upper'], alpha=0.2)

  # - Simulate two years of new data after the end of the sample
  print(results.simulate(8, anchor='end'))

  # - Impulse responses for two years
  print(results.impulse_responses(8))

Basic methods and attributes for estimation / filtering / smoothing
-------------------------------------------------------------------

The most-used methods for a state space model are:

- :py:meth:`fit <mlemodel.MLEModel.fit>` - estimate parameters via maximum
  likelihood and return a results object (this object will have also performed
  Kalman filtering and smoothing at the estimated parameters). This is the most
  commonly used method.
- :py:meth:`smooth <mlemodel.MLEModel.smooth>` - return a results object
  associated with a given vector of parameters after performing Kalman
  filtering and smoothing
- :py:meth:`loglike <mlemodel.MLEModel.loglike>` - compute the log-likelihood
  of the data using a given vector of parameters

Some useful attributes of a state space model are:

- :py:meth:`param_names <mlemodel.MLEModel.param_names>` - names of the
  parameters used by the model
- :py:meth:`state_names <mlemodel.MLEModel.state_names>` - names of the
  elements of the (unobserved) state vector
- :py:meth:`start_params <mlemodel.MLEModel.start_params>` - initial parameter
  estimates used a starting values for numerical maximum likelihood
  optimization

Other methods that are used less often are:

- :py:meth:`filter <mlemodel.MLEModel.filter>` - return a results object
  associated with a given vector of parameters after only performing Kalman
  filtering (but not smoothing)
- :py:meth:`simulation_smoother <mlemodel.MLEModel.simulation_smoother>` -
  return an object that can perform simulation smoothing

Output and postestimation methods and attributes
------------------------------------------------

Commonly used methods include:

- :py:meth:`summary <mlemodel.MLEResults.summary>` - construct a table that
  presents model fit statistics, estimated parameters, and other summary output
- :py:meth:`predict <mlemodel.MLEResults.predict>` - compute in-sample
  predictions and out-of-sample forecasts (point estimates only)
- :py:meth:`get_prediction <mlemodel.MLEResults.get_prediction>` - compute
  in-sample predictions and out-of-sample forecasts, including confidence
  intervals
- :py:meth:`forecast <mlemodel.MLEResults.forecast>` - compute out-of-sample
  forecasts (point estimates only) (this is a convenience wrapper around
  `predict`)
- :py:meth:`get_forecast <mlemodel.MLEResults.get_forecast>` - compute
  out-of-sample forecasts, including confidence intervals (this is a
  convenience wrapper around `get_prediction`)
- :py:meth:`simulate <mlemodel.MLEResults.simulate>` - simulate new data
  according to the state space model
- :py:meth:`impulse_responses <mlemodel.MLEResults.impulse_responses>` -
  compute impulse responses from the state space model

Commonly used attributes include:

- :py:meth:`params <mlemodel.MLEResults.params>` - estimated parameters
- :py:meth:`bse <mlemodel.MLEResults.bse>` - standard errors of estimated
  parameters
- :py:meth:`pvalues <mlemodel.MLEResults.pvalues>` - p-values associated with
  estimated parameters
- :py:meth:`llf <mlemodel.MLEResults.llf>` - log-likelihood of the data at
  the estimated parameters
- :py:meth:`sse <mlemodel.MLEResults.sse>`,
  :py:meth:`mse <mlemodel.MLEResults.mse>`, and
  :py:meth:`mae <mlemodel.MLEResults.mae>` - sum of squared errors,
  mean square error, and mean absolute error
- Information criteria, including: :py:meth:`aic <mlemodel.MLEResults.aic>`,
  :py:meth:`aicc <mlemodel.MLEResults.aicc>`,
  :py:meth:`bic <mlemodel.MLEResults.bic>`, and
  :py:meth:`hquc <mlemodel.MLEResults.hqic>`
- :py:meth:`fittedvalues <mlemodel.MLEResults.fittedvalues>` - fitted values
  from the model (note that these are one-step-ahead predictions)
- :py:meth:`resid <mlemodel.MLEResults.resid>` - residuals from the model (note
  that these are one-step-ahead prediction errors)

Estimates and covariances of the unobserved state
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It can be useful to compute estimates of the unobserved state vector
conditional on the observed data. These are available in the results object
:py:meth:`states <mlemodel.MLEResults.states>`, which contains the following
elements:

- `states.filtered` - filtered (one-sided) estimates of the state vector. The
  estimate of the state vector at time `t` is based on the observed data up
  to and including time `t`.
- `states.smoothed` - smoothed (two-sided) estimates of the state vector. The
  estimate of the state vector at time `t` is based on all observed data in
  the sample.
- `states.filtered_cov` - filtered (one-sided) covariance of the state vector
- `states.smoothed_cov` - smoothed (two-sided) covariance of the state vector

Each of these elements are Pandas `DataFrame` objects.

As an example, in a "local level + seasonal" model estimated via the
`UnobservedComponents` components class we can get an estimates of the
underlying level and seasonal movements of a series over time.

.. code-block:: python

  fig, axes = plt.subplots(3, 1, figsize=(8, 8))

  # Retrieve monthly retail sales for clothing
  from pandas_datareader.data import DataReader
  clothing = DataReader('MRTSSM4481USN', 'fred', start='1992').asfreq('MS')['MRTSSM4481USN']

  # Construct a local level + seasonal model
  model = sm.tsa.UnobservedComponents(clothing, 'llevel', seasonal=12)
  results = model.fit()

  # Plot the data, the level, and seasonal
  clothing.plot(ax=axes[0])
  results.states.smoothed['level'].plot(ax=axes[1])
  results.states.smoothed['seasonal'].plot(ax=axes[2])

Residual diagnostics
^^^^^^^^^^^^^^^^^^^^

Three diagnostic tests are available after estimation of any statespace model,
whether built in or custom, to help assess whether the model conforms to the
underlying statistical assumptions. These tests are:

- :py:meth:`test_normality <mlemodel.MLEResults.test_normality>`
- :py:meth:`test_heteroskedasticity <mlemodel.MLEResults.test_heteroskedasticity>`
- :py:meth:`test_serial_correlation <mlemodel.MLEResults.test_serial_correlation>`

A number of standard plots of regression residuals are available for the same
purpose. These can be produced using the command
:py:meth:`plot_diagnostics <mlemodel.MLEResults.plot_diagnostics>`.

Applying estimated parameters to an updated or different dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are three methods that can be used to apply estimated parameters from a
results object to an updated or different dataset:

- :py:meth:`append <mlemodel.MLEResults.append>` - retrieve a new results
  object with additional observations that follow after the end of the current
  sample appended to it (so the new results object contains both the current
  sample and the additional observations)
- :py:meth:`extend <mlemodel.MLEResults.extend>` - retrieve a new results
  object for additional observations that follow after end of the current
  sample (so the new results object contains only the new observations but NOT
  the current sample)
- :py:meth:`apply <mlemodel.MLEResults.apply>` - retrieve a new results object
  for a completely different dataset

One cross-validation exercise on time-series data involves fitting a model's
parameters based on a training sample (observations through time `t`) and
then evaluating the fit of the model using a test sample (observations `t+1`,
`t+2`, ...). This can be conveniently done using either `apply` or `extend`. In
the example below, we use the `extend` method.

.. code-block:: python

  # Load in the example macroeconomic dataset
  dta = sm.datasets.macrodata.load_pandas().data
  # Make sure we have an index with an associated frequency, so that
  # we can refer to time periods with date strings or timestamps
  dta.index = pd.date_range('1959Q1', '2009Q3', freq='QS')

  # Separate inflation data into a training and test dataset
  training_endog = dta['infl'].iloc[:-1]
  test_endog = dta['infl'].iloc[-1:]

  # Fit an SARIMAX model for inflation
  training_model = sm.tsa.SARIMAX(training_endog, order=(4, 0, 0))
  training_results = training_model.fit()

  # Extend the results to the test observations
  test_results = training_results.extend(test_endog)

  # Print the sum of squared errors in the test sample,
  # based on parameters computed using only the training sample
  print(test_results.sse)

Additional options and tools
----------------------------

All state space models have the following options and tools:

Holding some parameters fixed and estimating the rest
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:meth:`fit_constrained <mlemodel.MLEModel.fit_constrained>` method
allows fixing some parameters to known values and then estimating the rest via
maximum likelihood. An example of this is:

.. code-block:: python

  # Construct a model
  model = sm.tsa.SARIMAX(endog, order=(1, 0, 0))

  # To find out the parameter names, use:
  print(model.param_names)

  # Fit the model with a fixed value for the AR(1) coefficient:
  results = model.fit_constrained({'ar.L1': 0.5})

Alternatively, you can use the
:py:meth:`fix_params <mlemodel.MLEModel.fix_params>` context manager:

.. code-block:: python

  # Construct a model
  model = sm.tsa.SARIMAX(endog, order=(1, 0, 0))

  # Fit the model with a fixed value for the AR(1) coefficient using the
  # context manager
  with model.fix_params({'ar.L1': 0.5}):
      results = model.fit()

Low memory options
^^^^^^^^^^^^^^^^^^

When the observed dataset is very large and / or the state vector of the model
is high-dimensional (for example when considering long seasonal effects), the
default memory requirements can be too large. For this reason, the `fit`,
`filter`, and `smooth` methods accept an optional `low_memory=True` argument,
which can considerably reduce memory requirements and speed up model fitting.

Note that when using `low_memory=True`, not all results objects will be
available. However, residual diagnostics, in-sample (non-dynamic) prediction,
and out-of-sample forecasting are all still available.

Low-level state space representation and Kalman filtering
---------------------------------------------------------

While creation of custom models will almost always be done by extending
`MLEModel` and `MLEResults`, it can be useful to understand the superstructure
behind those classes.

Maximum likelihood estimation requires evaluating the likelihood function of
the model, and for models in state space form the likelihood function is
evaluated as a byproduct of running the Kalman filter.

There are two classes used by `MLEModel` that facilitate specification of the
state space model and Kalman filtering: `Representation` and `KalmanFilter`.

The `Representation` class is the piece where the state space model
representation is defined. In simple terms, it holds the state space matrices
(`design`, `obs_intercept`, etc.; see the introduction to state space models,
above) and allows their manipulation.

`FrozenRepresentation` is the most basic results-type class, in that it takes a
"snapshot" of the state space representation at any given time. See the class
documentation for the full list of available attributes.

.. autosummary::
   :toctree: generated/

   representation.Representation
   representation.FrozenRepresentation

The `KalmanFilter` class is a subclass of Representation that provides
filtering capabilities. Once the state space representation matrices have been
constructed, the :py:meth:`filter <kalman_filter.KalmanFilter.filter>`
method can be called, producing a `FilterResults` instance; `FilterResults` is
a subclass of `FrozenRepresentation`.

The `FilterResults` class not only holds a frozen representation of the state
space model (the design, transition, etc. matrices, as well as model
dimensions, etc.) but it also holds the filtering output, including the
:py:attr:`filtered state <kalman_filter.FilterResults.filtered_state>` and
loglikelihood (see the class documentation for the full list of available
results). It also provides a
:py:meth:`predict <kalman_filter.FilterResults.predict>` method, which allows
in-sample prediction or out-of-sample forecasting. A similar method,
:py:meth:`predict <kalman_filter.FilterResults.get_prediction>`, provides
additional prediction or forecasting results, including confidence intervals.

.. autosummary::
   :toctree: generated/

   kalman_filter.KalmanFilter
   kalman_filter.FilterResults
   kalman_filter.PredictionResults

The `KalmanSmoother` class is a subclass of `KalmanFilter` that provides
smoothing capabilities. Once the state space representation matrices have been
constructed, the :py:meth:`filter <kalman_smoother.KalmanSmoother.smooth>`
method can be called, producing a `SmootherResults` instance; `SmootherResults`
is a subclass of `FilterResults`.

The `SmootherResults` class holds all the output from `FilterResults`, but
also includes smoothing output, including the
:py:attr:`smoothed state <kalman_filter.SmootherResults.smoothed_state>` and
loglikelihood (see the class documentation for the full list of available
results). Whereas "filtered" output at time `t` refers to estimates conditional
on observations up through time `t`, "smoothed" output refers to estimates
conditional on the entire set of observations in the dataset.

.. autosummary::
   :toctree: generated/

   kalman_smoother.KalmanSmoother
   kalman_smoother.SmootherResults

The `SimulationSmoother` class is a subclass of `KalmanSmoother` that further
provides simulation and simulation smoothing capabilities. The
:py:meth:`simulation_smoother <simulation_smoother.SimulationSmoother.simulation_smoother>`
method can be called, producing a `SimulationSmoothResults` instance.

The `SimulationSmoothResults` class has a `simulate` method, that allows
performing simulation smoothing to draw from the joint posterior of the state
vector. This is useful for Bayesian estimation of state space models via Gibbs
sampling.

.. autosummary::
   :toctree: generated/

   simulation_smoother.SimulationSmoother
   simulation_smoother.SimulationSmoothResults


Statespace Tools
----------------

There are a variety of tools used for state space modeling or by the SARIMAX
class:

.. autosummary::
   :toctree: generated/

   tools.companion_matrix
   tools.diff
   tools.is_invertible
   tools.constrain_stationary_univariate
   tools.unconstrain_stationary_univariate
   tools.constrain_stationary_multivariate
   tools.unconstrain_stationary_multivariate
   tools.validate_matrix_shape
   tools.validate_vector_shape
