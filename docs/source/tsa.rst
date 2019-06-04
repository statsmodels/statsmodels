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
 - regime_switching : Markov switching dynamic regression and autoregression
   models



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
   stattools.periodogram
   stattools.adfuller
   stattools.kpss
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

Univariate Autogressive Processes (AR)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: statsmodels.tsa

.. autosummary::
   :toctree: generated/

   ar_model.AR
   ar_model.ARResults


Autogressive Moving-Average Processes (ARMA) and Kalman Filter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: statsmodels.tsa

The basic ARIMA model and results classes that should be the starting point for
for most users are:

.. autosummary::
   :toctree: generated/

   arima_model.ARMA
   arima_model.ARMAResults
   arima_model.ARIMA
   arima_model.ARIMAResults

Some advanced underlying low-level classes and functions that can be used to
compute the log-likelihood function for ARMA-type models include (note that
these are rarely needed by end-users):

.. autosummary::
   :toctree: generated/

   kalmanf.kalmanfilter.KalmanFilter
   innovations.arma_innovations.arma_innovations
   innovations.arma_innovations.arma_loglike
   innovations.arma_innovations.arma_loglikeobs
   innovations.arma_innovations.arma_score
   innovations.arma_innovations.arma_scoreobs


Exponential Smoothing
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: statsmodels.tsa

.. autosummary::
   :toctree: generated/

   holtwinters.ExponentialSmoothing
   holtwinters.SimpleExpSmoothing
   holtwinters.Holt
   holtwinters.HoltWintersResults


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

.. currentmodule:: statsmodels

.. autosummary::
   :toctree: generated/

   sandbox.tsa.fftarma.ArmaFft

.. currentmodule:: statsmodels.tsa

Statespace Models
"""""""""""""""""
See the :ref:`statespace documentation. <statespace>`.


Vector ARs and Vector Error Correction Models
"""""""""""""""""""""""""""""""""""""""""""""
See the :ref:`vector_ar documentation. <var>`.

Regime switching models
"""""""""""""""""""""""

.. autosummary::
   :toctree: generated/

   regime_switching.markov_regression.MarkovRegression
   regime_switching.markov_autoregression.MarkovAutoregression


Time Series Filters
"""""""""""""""""""

.. autosummary::
   :toctree: generated/

   filters.bk_filter.bkfilter
   filters.hp_filter.hpfilter
   filters.cf_filter.cffilter
   filters.filtertools.convolution_filter
   filters.filtertools.recursive_filter
   filters.filtertools.miso_lfilter
   filters.filtertools.fftconvolve3
   filters.filtertools.fftconvolveinv
   seasonal.seasonal_decompose


TSA Tools
"""""""""

.. currentmodule:: statsmodels.tsa

.. autosummary::
   :toctree: generated/

   tsatools.add_trend
   tsatools.detrend
   tsatools.lagmat
   tsatools.lagmat2ds

VARMA Process
"""""""""""""

.. autosummary::
   :toctree: generated/

   varma_process.VarmaPoly

Interpolation
"""""""""""""

.. autosummary::
   :toctree: generated/

   interp.denton.dentonm
