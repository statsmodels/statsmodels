.. module:: statsmodels.tsa
   :synopsis: Time-series analysis

.. currentmodule:: statsmodels.tsa


.. _tsa:


Time Series analysis :mod:`tsa`
===============================

:mod:`statsmodels.tsa` contains model classes and functions that are useful
for time series analysis. Basic models include univariate autoregressive models (AR),
vector autoregressive models (VAR) and univariate autoregressive moving average models
(ARMA). Non-linear models include Markov switching dynamic regression and
autoregression. It also includes descriptive statistics for time series, for example autocorrelation, partial
autocorrelation function and periodogram, as well as the corresponding theoretical properties
of ARMA or related processes. It also includes methods to work with autoregressive and
moving average lag-polynomials.
Additionally, related statistical tests and some useful helper functions are available.

Estimation is either done by exact or conditional Maximum Likelihood or conditional
least-squares, either using Kalman Filter or direct filters.

Currently, functions and classes have to be imported from the corresponding module, but
the main classes will be made available in the statsmodels.tsa namespace. The module
structure is within statsmodels.tsa is

- stattools : empirical properties and tests, acf, pacf, granger-causality,
  adf unit root test, kpss test, bds test, ljung-box test and others.
- ar_model : univariate autoregressive process, estimation with conditional
  and exact maximum likelihood and conditional least-squares
- arima_model : univariate ARMA process, estimation with conditional
  and exact maximum likelihood and conditional least-squares
- statespace : Comprehensive statespace model specification and estimation. See
  the :ref:`statespace documentation <statespace>`.
- vector_ar, var : vector autoregressive process (VAR) and vector error correction
  models, estimation, impulse response analysis, forecast error variance decompositions,
  and data visualization tools. See the :ref:`vector_ar documentation <var>`.
- kalmanf : estimation classes for ARMA and other models with exact MLE using
  Kalman Filter
- arma_process : properties of arma processes with given parameters, this
  includes tools to convert between ARMA, MA and AR representation as well as
  acf, pacf, spectral density, impulse response function and similar
- sandbox.tsa.fftarma : similar to arma_process but working in frequency domain
- tsatools : additional helper functions, to create arrays of lagged variables,
  construct regressors for trend, detrend and similar.
- filters : helper function for filtering time series
- regime_switching : Markov switching dynamic regression and autoregression models

Some additional functions that are also useful for time series analysis are in
other parts of statsmodels, for example additional statistical tests.

Some related functions are also available in matplotlib, nitime, and
scikits.talkbox. Those functions are designed more for the use in signal
processing where longer time series are available and work more often in the
frequency domain.


.. currentmodule:: statsmodels.tsa


Descriptive Statistics and Tests
""""""""""""""""""""""""""""""""

.. autosummary::
   :toctree: generated/

   stattools.acovf
   stattools.acf
   stattools.pacf
   stattools.pacf_yw
   stattools.pacf_ols
   stattools.pacf_burg
   stattools.ccovf
   stattools.ccf
   stattools.adfuller
   stattools.kpss
   stattools.zivot_andrews
   stattools.coint
   stattools.bds
   stattools.q_stat
   stattools.grangercausalitytests
   stattools.levinson_durbin
   stattools.innovations_algo
   stattools.innovations_filter
   stattools.levinson_durbin_pacf
   stattools.arma_order_select_ic
   x13.x13_arima_select_order
   x13.x13_arima_analysis

Estimation
""""""""""

The following are the main estimation classes, which can be accessed through
statsmodels.tsa.api and their result classes

Univariate Autoregressive Processes (AR)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The basic autoregressive model in Statsmodels is:

.. currentmodule:: statsmodels.tsa

.. autosummary::
   :toctree: generated/

   ar_model.AutoReg
   ar_model.AutoRegResults
   ar_model.ar_select_order

The `ar_model.AutoReg` model estimates parameters using conditional MLE (OLS),
and supports exogenous regressors (an AR-X model) and seasonal effects.

AR-X and related models can also be fitted with the `arima.ARIMA` class and the
`SARIMAX` class (using full MLE via the Kalman Filter).

Finally, the old class, `ar_model.AR`, is still available but it has been
deprecated.

.. autosummary::
   :toctree: generated/

   ar_model.AR
   ar_model.ARResults

Autoregressive Moving-Average Processes (ARMA) and Kalman Filter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Basic ARIMA model and results classes are as follows:

.. currentmodule:: statsmodels.tsa

.. autosummary::
   :toctree: generated/

   arima.model.ARIMA
   arima.model.ARIMAResults

This model allows estimating parameters by various methods (including
conditional MLE via the Hannan-Rissanen method and full MLE via the Kalman
filter). It is a special case of the `SARIMAX` model, and it includes a large
number of inherited features from the :ref:`state space <statespace>` models
(including prediction / forecasting, residual diagnostics, simulation and
impulse responses, etc.).

Prior to version 0.11, the basic ARIMA model and results classes were the
following:

.. autosummary::
   :toctree: generated/

   arima_model.ARMA
   arima_model.ARMAResults
   arima_model.ARIMA
   arima_model.ARIMAResults

These classes are still available, but they are no longer recommended.

Exponential Smoothing
~~~~~~~~~~~~~~~~~~~~~

Linear and non-linear exponential smoothing models are available:

.. currentmodule:: statsmodels.tsa.holtwinters

.. autosummary::
   :toctree: generated/

   ExponentialSmoothing
   SimpleExpSmoothing
   Holt
   HoltWintersResults

Separately, linear and non-linear exponential smoothing models have also been
implemented based on the "innovations" state space approach. In addition to the
usual support for parameter fitting, in-sample prediction, and out-of-sample
forecasting, these models also support prediction intervals, simulation, and
more.

.. currentmodule:: statsmodels.tsa

.. autosummary::
   :toctree: generated/

   exponential_smoothing.ets.ETSModel
   exponential_smoothing.ets.ETSResults

Finally, linear exponential smoothing models have also been separately
implemented as a special case of the general state space framework (this is
separate from the "innovations" state space approach described above). Although
this approach does not allow for the non-linear (multiplicative) exponential
smoothing models, it includes all features of :ref:`state space <statespace>`
models (including prediction / forecasting, residual diagnostics, simulation
and impulse responses, etc.).

.. currentmodule:: statsmodels.tsa

.. autosummary::
   :toctree: generated/

   statespace.exponential_smoothing.ExponentialSmoothing
   statespace.exponential_smoothing.ExponentialSmoothingResults

ARMA Process
""""""""""""

The following are tools to work with the theoretical properties of an ARMA
process for given lag-polynomials.

.. autosummary::
   :toctree: generated/

   arima_process.ArmaProcess
   arima_process.ar2arma
   arima_process.arma2ar
   arima_process.arma2ma
   arima_process.arma_acf
   arima_process.arma_acovf
   arima_process.arma_generate_sample
   arima_process.arma_impulse_response
   arima_process.arma_pacf
   arima_process.arma_periodogram
   arima_process.deconvolve
   arima_process.index2lpol
   arima_process.lpol2index
   arima_process.lpol_fiar
   arima_process.lpol_fima
   arima_process.lpol_sdiff

.. currentmodule:: statsmodels.sandbox.tsa.fftarma

.. autosummary::
   :toctree: generated/

   ArmaFft

.. currentmodule:: statsmodels.tsa

Statespace Models
"""""""""""""""""
See the :ref:`statespace documentation <statespace>`.


Vector ARs and Vector Error Correction Models
"""""""""""""""""""""""""""""""""""""""""""""
See the :ref:`vector_ar documentation. <var>`

Regime switching models
"""""""""""""""""""""""

.. currentmodule:: statsmodels.tsa.regime_switching.markov_regression
.. autosummary::
   :toctree: generated/

   MarkovRegression

.. currentmodule:: statsmodels.tsa.regime_switching.markov_autoregression
.. autosummary::
   :toctree: generated/

   MarkovAutoregression


Time Series Filters
"""""""""""""""""""

.. currentmodule:: statsmodels.tsa.filters.bk_filter
.. autosummary::
   :toctree: generated/

   bkfilter

.. currentmodule:: statsmodels.tsa.filters.hp_filter
.. autosummary::
   :toctree: generated/

   hpfilter

.. currentmodule:: statsmodels.tsa.filters.cf_filter
.. autosummary::
   :toctree: generated/

   cffilter

.. currentmodule:: statsmodels.tsa.filters.filtertools
.. autosummary::
   :toctree: generated/

   convolution_filter
   recursive_filter
   miso_lfilter
   fftconvolve3
   fftconvolveinv


.. currentmodule:: statsmodels.tsa.seasonal
.. autosummary::
   :toctree: generated/

   seasonal_decompose
   STL
   DecomposeResult

TSA Tools
"""""""""

.. currentmodule:: statsmodels.tsa.tsatools

.. autosummary::
   :toctree: generated/

   add_lag
   add_trend
   detrend
   lagmat
   lagmat2ds

VARMA Process
"""""""""""""

.. currentmodule:: statsmodels.tsa.varma_process
.. autosummary::
   :toctree: generated/

   VarmaPoly

Interpolation
"""""""""""""

.. currentmodule:: statsmodels.tsa.interp.denton
.. autosummary::
   :toctree: generated/

   dentonm


Deterministic Processes
"""""""""""""""""""""""

Deterministic processes simplify creating deterministic sequences with time
trend or seasonal patterns. They also provide methods to simplify generating
deterministic terms for out-of-sample forecasting. A
:class:`~statsmodels.tsa.deterministic.DeterministicProcess` can be directly
used with :class:`~statsmodels.tsa.ar_model.AutoReg` to construct complex
deterministic dynamics and to forecast without constructing exogenous trends.

.. currentmodule:: statsmodels.tsa.deterministic
.. autosummary::
   :toctree: generated/

   DeterministicProcess
   TimeTrend
   Seasonality
   Fourier
   CalendarTimeTrend
   CalendarSeasonality
   CalendarFourier
   DeterministicTerm
   CalendarDeterministicTerm
   FourierDeterministicTerm
   TimeTrendDeterministicTerm

Users who wish to write custom deterministic terms must subclass
:class:`~statsmodels.tsa.deterministic.DeterministicTerm`.

.. currentmodule:: statsmodels.tsa.deterministic
.. autosummary::
   :toctree: generated/

   DeterministicTerm

Forecasting Models
""""""""""""""""""
.. module:: statsmodels.tsa.forecasting
   :synopsis: Models designed for forecasting

.. currentmodule:: statsmodels.tsa.forecasting

The Theta Model
~~~~~~~~~~~~~~~
The Theta model is a simple forecasting method that combines a linear time
trend with a Simple Exponential Smoother (Assimakopoulos & Nikolopoulos).
An estimator for the parameters of the Theta model and methods to forecast
are available in:

.. module:: statsmodels.tsa.forecasting.theta
   :synopsis: Models designed for forecasting

.. currentmodule:: statsmodels.tsa.forecasting.theta

.. autosummary::
   :toctree: generated/

   ThetaModel
   ThetaModelResults

Forecasting after STL Decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:class:`statsmodels.tsa.seasonal.STL` is commonly used to remove seasonal
components from a time series. The deseasonalized time series can then
be modeled using a any non-seasonal model, and forecasts are constructed
by adding the forecast from the non-seasonal model to the estimates of
the seasonal component from the final full-cycle which are forecast using
a random-walk model.

.. module:: statsmodels.tsa.forecasting.stl
   :synopsis: Models designed for forecasting

.. currentmodule:: statsmodels.tsa.forecasting.stl

.. autosummary::
   :toctree: generated/

   STLForecast
   STLForecastResults

Prediction Results
""""""""""""""""""
Most forecasting methods support a ``get_prediction`` method that return
a ``PredictionResults`` object that contains both the prediction, its
variance and can construct a prediction interval.

Results Class
~~~~~~~~~~~~~

.. module:: statsmodels.tsa.base.prediction
   :synopsis: Shared objects for predictive methods

.. currentmodule:: statsmodels.tsa.base.prediction

.. autosummary::
   :toctree: generated/

   PredictionResults

