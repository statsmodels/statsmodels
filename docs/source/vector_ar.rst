:orphan:

.. module:: statsmodels.tsa.vector_ar.var_model
   :synopsis: Vector autoregressions

.. currentmodule:: statsmodels.tsa.vector_ar.var_model

.. _var:

Vector Autoregressions :mod:`tsa.vector_ar`
===========================================

VAR(p) processes
----------------

We are interested in modeling a :math:`T \times K` multivariate time series
:math:`Y`, where :math:`T` denotes the number of observations and :math:`K` the
number of variables. One way of estimating relationships between the time series
and their lagged values is the *vector autoregression process*:

.. math::

   Y_t = A_1 Y_{t-1} + \ldots + A_p Y_{t-p} + u_t

   u_t \sim {\sf Normal}(0, \Sigma_u)

where :math:`A_i` is a :math:`K \times K` coefficient matrix.

We follow in large part the methods and notation of `Lutkepohl (2005)
<http://www.springer.com/economics/econometrics/book/978-3-540-40172-8?otherVersion=978-3-540-26239-8>`__,
which we will not develop here.

Model fitting
~~~~~~~~~~~~~

.. note::

    The classes referenced below are accessible via the
    :mod:`statsmodels.tsa.api` module.

To estimate a VAR model, one must first create the model using an `ndarray` of
homogeneous or structured dtype. When using a structured or record array, the
class will use the passed variable names. Otherwise they can be passed
explicitly:

.. ipython:: python
    :suppress:

    import pandas as pd
    pd.options.display.max_rows = 10
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.style.use('ggplot')

.. ipython:: python

    # some example data
    import numpy as np
    import pandas
    import statsmodels.api as sm
    from statsmodels.tsa.api import VAR, DynamicVAR
    mdata = sm.datasets.macrodata.load_pandas().data

    # prepare the dates index
    dates = mdata[['year', 'quarter']].astype(int).astype(str)
    quarterly = dates["year"] + "Q" + dates["quarter"]
    from statsmodels.tsa.base.datetools import dates_from_str
    quarterly = dates_from_str(quarterly)

    mdata = mdata[['realgdp','realcons','realinv']]
    mdata.index = pandas.DatetimeIndex(quarterly)
    data = np.log(mdata).diff().dropna()

    # make a VAR model
    model = VAR(data)

.. note::

   The :class:`VAR` class assumes that the passed time series are
   stationary. Non-stationary or trending data can often be transformed to be
   stationary by first-differencing or some other method. For direct analysis of
   non-stationary time series, a standard stable VAR(p) model is not
   appropriate.

To actually do the estimation, call the `fit` method with the desired lag
order. Or you can have the model select a lag order based on a standard
information criterion (see below):

.. ipython:: python

    results = model.fit(2)

    results.summary()


Several ways to visualize the data using `matplotlib` are available.

Plotting input time series:

.. ipython:: python

    @savefig var_plot_input.png
    results.plot()


Plotting time series autocorrelation function:

.. ipython:: python

    @savefig var_plot_acorr.png
    results.plot_acorr()


Lag order selection
~~~~~~~~~~~~~~~~~~~

Choice of lag order can be a difficult problem. Standard analysis employs
likelihood test or information criteria-based order selection. We have
implemented the latter, accessible through the :class:`VAR` class:

.. ipython:: python

    model.select_order(15)

When calling the `fit` function, one can pass a maximum number of lags and the
order criterion to use for order selection:

.. ipython:: python

    results = model.fit(maxlags=15, ic='aic')

Forecasting
~~~~~~~~~~~

The linear predictor is the optimal h-step ahead forecast in terms of
mean-squared error:

.. math::

   y_t(h) = \nu + A_1 y_t(h − 1) + \cdots + A_p y_t(h − p)

We can use the `forecast` function to produce this forecast. Note that we have
to specify the "initial value" for the forecast:

.. ipython:: python

    lag_order = results.k_ar
    results.forecast(data.values[-lag_order:], 5)

The `forecast_interval` function will produce the above forecast along with
asymptotic standard errors. These can be visualized using the `plot_forecast`
function:

.. ipython:: python

   @savefig var_forecast.png
   results.plot_forecast(10)

Impulse Response Analysis
-------------------------

*Impulse responses* are of interest in econometric studies: they are the
estimated responses to a unit impulse in one of the variables. They are computed
in practice using the MA(:math:`\infty`) representation of the VAR(p) process:

.. math::

    Y_t = \mu + \sum_{i=0}^\infty \Phi_i u_{t-i}

We can perform an impulse response analysis by calling the `irf` function on a
`VARResults` object:

.. ipython:: python

    irf = results.irf(10)

These can be visualized using the `plot` function, in either orthogonalized or
non-orthogonalized form. Asymptotic standard errors are plotted by default at
the 95% significance level, which can be modified by the user.

.. note::

    Orthogonalization is done using the Cholesky decomposition of the estimated
    error covariance matrix :math:`\hat \Sigma_u` and hence interpretations may
    change depending on variable ordering.

.. ipython:: python

    @savefig var_irf.png
    irf.plot(orth=False)


Note the `plot` function is flexible and can plot only variables of interest if
so desired:

.. ipython:: python

    @savefig var_realgdp.png
    irf.plot(impulse='realgdp')

The cumulative effects :math:`\Psi_n = \sum_{i=0}^n \Phi_i` can be plotted with
the long run effects as follows:

.. ipython:: python

    @savefig var_irf_cum.png
    irf.plot_cum_effects(orth=False)


Forecast Error Variance Decomposition (FEVD)
--------------------------------------------

Forecast errors of component j on k in an i-step ahead forecast can be
decomposed using the orthogonalized impulse responses :math:`\Theta_i`:

.. math::

    \omega_{jk, i} = \sum_{i=0}^{h-1} (e_j^\prime \Theta_i e_k)^2 / \mathrm{MSE}_j(h)

    \mathrm{MSE}_j(h) = \sum_{i=0}^{h-1} e_j^\prime \Phi_i \Sigma_u \Phi_i^\prime e_j

These are computed via the `fevd` function up through a total number of steps ahead:

.. ipython:: python

    fevd = results.fevd(5)

    fevd.summary()

They can also be visualized through the returned :class:`FEVD` object:

.. ipython:: python

    @savefig var_fevd.png
    results.fevd(20).plot()


Statistical tests
-----------------

A number of different methods are provided to carry out hypothesis tests about
the model results and also the validity of the model assumptions (normality,
whiteness / "iid-ness" of errors, etc.).

Granger causality
~~~~~~~~~~~~~~~~~

One is often interested in whether a variable or group of variables is "causal"
for another variable, for some definition of "causal". In the context of VAR
models, one can say that a set of variables are Granger-causal within one of the
VAR equations. We will not detail the mathematics or definition of Granger
causality, but leave it to the reader. The :class:`VARResults` object has the
`test_causality` method for performing either a Wald (:math:`\chi^2`) test or an
F-test.

.. ipython:: python

    results.test_causality('realgdp', ['realinv', 'realcons'], kind='f')

Normality
~~~~~~~~~

Whiteness of residuals
~~~~~~~~~~~~~~~~~~~~~~

Dynamic Vector Autoregressions
------------------------------

.. note::

    To use this functionality, `pandas <https://pypi.python.org/pypi/pandas>`__
    must be installed. See the `pandas documentation
    <http://pandas.pydata.org>`__ for more information on the below data
    structures.

One is often interested in estimating a moving-window regression on time series
data for the purposes of making forecasts throughout the data sample. For
example, we may wish to produce the series of 2-step-ahead forecasts produced by
a VAR(p) model estimated at each point in time.

.. ipython:: python

    np.random.seed(1)
    import pandas.util.testing as ptest
    ptest.N = 500
    data = ptest.makeTimeDataFrame().cumsum(0)
    data

    var = DynamicVAR(data, lag_order=2, window_type='expanding')

The estimated coefficients for the dynamic model are returned as a
:class:`pandas.Panel` object, which can allow you to easily examine, for
example, all of the model coefficients by equation or by date:

.. ipython:: python
   :okwarning:

    import datetime as dt

    var.coefs

    # all estimated coefficients for equation A
    var.coefs.minor_xs('A').info()

    # coefficients on 11/30/2001
    var.coefs.major_xs(dt.datetime(2001, 11, 30)).T

Dynamic forecasts for a given number of steps ahead can be produced using the
`forecast` function and return a :class:`pandas.DataMatrix` object:

.. ipython:: python

    var.forecast(2)

The forecasts can be visualized using `plot_forecast`:

.. ipython:: python

    @savefig dvar_forecast.png
    var.plot_forecast(2)

Class Reference
---------------

.. module:: statsmodels.tsa.vector_ar
   :synopsis: Vector autoregressions and related tools

.. currentmodule:: statsmodels.tsa.vector_ar

.. autosummary::
   :toctree: generated/

   var_model.VAR
   var_model.VARProcess
   var_model.VARResults
   irf.IRAnalysis
   var_model.FEVD
   dynamic.DynamicVAR

