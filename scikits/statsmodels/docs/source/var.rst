.. currentmodule:: scikits.statsmodels.tsa.var


.. _var:

Vector Autoregressions :mod:`tsa.var`
=====================================

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

We follow in large part the theory the methods described in `Lutkepohl (2005)
<http://www.springer.com/economics/econometrics/book/978-3-540-26239-8>`__,
which we will not develop here.

Model fitting
~~~~~~~~~~~~~

To estimate a VAR model, one must first create the model using an `ndarray` of
homogeneous or structured dtype. When using a structured or record array, the
class will use the passed variable names. Otherwise they can be passed
explicitly:

::

    # some example data
    >>> mdata = sm.datasets.macrodata.load().data
    >>> mdata = mdata[['realgdp','realcons','realinv']]
    >>> names = mdata.dtype.names
    >>> data = mdata.view((float,3))
    >>> data = np.diff(np.log(data), axis=0)

    >>> model = VAR(data, names=names)

.. note::

   The :class:`VAR` class assumes that the passed time series are
   stationary. Non-stationary or trending data can often be transformed to be
   stationary by first-differencing or some other method. For direct analysis of
   non-stationary time series, a standard stable VAR(p) model is not
   appropriate.

To actually do the estimation, call the `fit` method with the desired lag
order. Or you can have the model select a lag order based on a standard
information criterion (see below):

::

    >>> results = model.fit(2)

    >>> results.summary()

      Summary of Regression Results
    ==================================
    Model:                         VAR
    Method:                        OLS
    Date:           Thu, 24, Feb, 2011
    Time:                     18:55:52
    --------------------------------------------------------------------
    No. of Equations:         3.00000    BIC:                   -27.5830
    Nobs:                     200.000    HQIC:                  -27.7892
    Log likelihood:           1962.57    FPE:                7.42129e-13
    AIC:                     -27.9293    Det(Omega_mle):     6.69358e-13
    --------------------------------------------------------------------
    Results for equation realgdp
    ==============================================================================
                     coefficient       std. error           t-stat            prob
    ------------------------------------------------------------------------------
    const               0.001527         0.001119            1.365           0.174
    L1.realgdp          0.005460         0.000969            5.634           0.000
    L1.realcons        -0.023903         0.005863           -4.077           0.000
    L1.realinv         -0.279435         0.169663           -1.647           0.101
    L2.realgdp         -0.100468         0.146924           -0.684           0.495
    L2.realcons        -1.970974         0.888892           -2.217           0.028
    L2.realinv          0.675016         0.131285            5.142           0.000
    ==============================================================================

    Results for equation realcons
    ==============================================================================
                     coefficient       std. error           t-stat            prob
    ------------------------------------------------------------------------------
    const               0.268640         0.113690            2.363           0.019
    L1.realgdp          4.414162         0.687825            6.418           0.000
    L1.realcons         0.033219         0.026194            1.268           0.206
    L1.realinv          0.025739         0.022683            1.135           0.258
    L2.realgdp          0.225479         0.137234            1.643           0.102
    L2.realcons         0.008221         0.173522            0.047           0.962
    L2.realinv         -0.123174         0.150267           -0.820           0.413
    ==============================================================================

    Results for equation realinv
    ==============================================================================
                     coefficient       std. error           t-stat            prob
    ------------------------------------------------------------------------------
    const               0.380786         0.909114            0.419           0.676
    L1.realgdp          0.290458         0.145904            1.991           0.048
    L1.realcons         0.232499         0.126350            1.840           0.067
    L1.realinv          0.800281         0.764416            1.047           0.296
    L2.realgdp         -0.007321         0.025786           -0.284           0.777
    L2.realcons         0.023504         0.022330            1.053           0.294
    L2.realinv         -0.124079         0.135098           -0.918           0.360
    ==============================================================================

    Correlation matrix of residuals
                 realgdp  realcons   realinv
    realgdp     1.000000  0.603316  0.750722
    realcons    0.603316  1.000000  0.131951
    realinv     0.750722  0.131951  1.000000

Several ways to visualize the data using `matplotlib` are available.

Plotting input time series:

::

	>>> model.plot()

.. plot:: plots/var_plot_input.py

Plotting time series autocorrelation function:

::

	>>> model.plot_acorr()

.. plot:: plots/var_plot_acorr.py


Lag order selection
~~~~~~~~~~~~~~~~~~~

Forecasting
~~~~~~~~~~~

Impulse response analysis
-------------------------

.. plot:: plots/var_plot_irf.py

.. plot:: plots/var_plot_irf_cum.py
