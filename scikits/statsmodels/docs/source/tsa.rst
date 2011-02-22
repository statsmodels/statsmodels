.. currentmodule:: scikits.statsmodels.tsa


.. _tsa:


Time Series analysis :mod:`tsa`
===============================

:mod:`scikits.statmodels.tsa` contains model classes and functions that are useful
for time series analysis. This currently includes univariate autoregressive models (AR),
vector autoregressive models (VAR) and univariate autoregressive moving average models
(ARMA). It also includes descriptive statistics for time series, for example autocorrelation, partial
autocorrelation function and periodogram, as well as the corresponding theoretical properties
of ARMA or related processes. It also includes methods to work with autoregressive and
moving average lag-polynomials.
Additionally, related statistical tests and some useful helper functions are available.

Estimation is either done by exact or conditional Maximum Likelihood or conditional
least-squares, either using Kalman Filter or direct filters.

Currently, functions and classes have to be imported from the corresponding module, but
the main classes will be made available in the statsmodels.tsa namespace. The module
structure is within scikits.statsmodels.tsa is

 - stattools : empirical properties and tests, acf, pacf, granger-causality,
 	  adf unit root test, ljung-box test and others.
 - var : contains univariate AR and multivariate AR (VAR) estimation models, either
      conditional LS or MLE or exact MLE.
 - arma_mle : estimation class for univariate ARMA with conditional least squares or
 	  conditional MLE
 - kalmanf : estimation classes for ARMA and other models with exact MLE using Kalman Filter
      (currently still in sandbox.tsa)
 - arma_process : properties of arma processes with given parameters, this includes tools
      to convert between ARMA, MA and AR representation as well as acf, pacf, spectral density,
      impulse response function and similar
 - sandbox.tsa.fftarma : similar to arma_process but working in frequency domain
 - tsatools : additional helper functions, to create arrays of lagged variables, construct
      regressors for trend, detrend and similar.
 - filters :

Some additional functions that are also useful for time series analysis are in other parts
of statsmodels, for example additional statistical tests.

Some related functions are also available in matplotlib, nitime, and scikits.talkbox.
Those functions are designed more for the use in signal processing where longer time
series are available and work more often in the frequency domain.


.. currentmodule:: scikits.statsmodels.tsa


Time Series Properties
""""""""""""""""""""""

.. autosummary::
   :toctree: generated/

   stattools.acovf
   stattools.acf
   stattools.pacf
   stattools.pacf_yw
   stattools.pacf_ols
   stattools.ccovf
   stattools.ccf
   stattools.pergram
   stattools.adfuller
   stattools.q_stat
   stattools.grangercausalitytests
   stattools.levinson_durbin

Estimation
""""""""""

.. autosummary::
   :toctree: generated/

   ar.AR
   ar.ARResults
   var.varmod.VAR
   var.varmod.VARResults
   arma_mle.Arma

ARMA and Kalman Filter
~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: scikits.statsmodels.tsa

.. autosummary::
   :toctree: generated/

   arima.ARMA
   arima.ARMAResults
   kalmanf.kalmanf.StateSpaceModel
   kalmanf.kalmanf.kalmanfilter
   kalmanf.kalmanf.kalmansmooth

.. currentmodule:: scikits.statsmodels.tsa

ARMA Process
""""""""""""

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

.. currentmodule:: scikits.statsmodels

.. autosummary::
   :toctree: generated/

   sandbox.tsa.fftarma.ArmaFft

.. currentmodule:: scikits.statsmodels.tsa

Other Time Series Filters
"""""""""""""""""""""""""

.. autosummary::
   :toctree: generated/

   filters.arfilter
   filters.fftconvolve3
   filters.fftconvolveinv
   filters.miso_lfilter


TSA Tools
"""""""""

.. autosummary::
   :toctree: generated/

   tsatools.add_constant
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
