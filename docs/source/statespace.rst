.. currentmodule:: statsmodels.tsa.statespace


.. _statespace:


Time Series Analysis by State Space Methods :mod:`statespace`
=============================================================

:mod:`statsmodels.tsa.statespace` contains classes and functions that are
useful for time series analysis using state space methods.

A general state space model is of the form

.. math::

  y_t & = Z_t \alpha_t + d_t + \varepsilon_t \\
  \alpha_t & = T_t \alpha_{t-1} + c_t + R_t \eta_t \\

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

Example: AR(2) model
^^^^^^^^^^^^^^^^^^^^

An autoregresive model is a good introductory example to putting models in
state space form. Recall that an AR(2) model is often written as:

.. math::

   y_t = \phi_1 y_{t-1} + \phi_2 y_{t-2} + \epsilon_t

This can be put into state space form in the following way:

.. math::

   y_t & = \begin{bmatrix} 1 & 0 \end{bmatrix} \alpha_t \\
   \alpha_t & = \begin{bmatrix}
      \phi_1 & \phi_2 \\
           1 &      0
   \end{bmatrix} \alpha_{t-1} + \begin{bmatrix} 1 \\ 0 \end{bmatrix} \eta_t

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
   \eta_t & \sim N(0, \sigma^2)

There are three unknown parameters in this model:
:math:`\phi_1, \phi_2, \sigma^2`.

Models and Estimation
---------------------

The following are the main estimation classes, which can be accessed through
`statsmodels.tsa.statespace.api` and their result classes.

Seasonal Autoregressive Integrated Moving-Average with eXogenous regoressors (SARIMAX)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `SARIMAX` class is an example of a fully fledged model created using the
statespace backend for estimation. `SARIMAX` can be used very similarly to
:ref:`tsa <tsa>` models, but works on a wider range of models by adding the
estimation of additive and multiplicative seasonal effects, as well as
arbitrary trend polynomials.

.. autosummary::
   :toctree: generated/

   sarimax.SARIMAX
   sarimax.SARIMAXResults

A very brief example of how to use this model::

   # Load the statsmodels api
   import statsmodels.api as sm

   # Load your dataset
   endog = pd.read_csv('your/dataset/here.csv')

   # Create the model, here an SARIMA(1,1,1) x (0,1,1,4) model
   mod = sm.tsa.statespace.SARIMAX(endog, order=(1,1,1), seasonal_order=(0,1,1,4))
   # Note that mod is an instance of the SARIMAX class

   # Fit the model via maximum likelihood
   res = mod.fit()
   # Note that res is an instance of the SARIMAXResults class

   # Show the summary of results
   print res.summary()

The results object has many of the attributes and methods you would expect from
other Statsmodels results objects, including standard errors, z-statistics,
and prediction / forecasting.

Behind the scenes, the `SARIMAX` model creates the design and transition
matrices (and sometimes some of the other matrices) based on the model
specification.

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
model, see the :ref:`Local Linear Trend example notebook <statespace_local_linear_trend_notebook>`.
For a more sophisticated example, see the source code for the `SARIMAX` and
`SARIMAXResults` classes, which are built by extending `MLEModel` and
`MLEResults`.

In simple cases, the model can be constructed entirely using the MLEModel
class. For example, the AR(2) model from above could be constructed and
estimated using only the following code::

   import numpy as np
   from scipy.signal import lfilter
   import statsmodels.api as sm

   # True model parameters
   nobs = 1e3
   true_phi = np.r_[0.5, -0.2]
   true_sigma = 1**0.5

   # Simulate a time series
   np.random.seed(1234)
   disturbances = np.random.normal(0, true_sigma, size=(nobs,))
   endog = lfilter([1], np.r_[1, -true_phi], disturbances)

   # Construct the model
   mod = sm.tsa.statespace.MLEModel(endog, k_states=2, k_posdef=1)

   # Setup the fixed components of the state space representation
   mod['design'] = [1, 0]
   mod['transition'] = [[0, 0],
                        [1, 0]]
   mod['selection', 0, 0] = 1

   # Tell the model how to update the state space representation
   # given parameters specified by the optimizer
   def updater(mod, params):
       mod['transition', 0, :] = params[:2]
       mod['state_cov', 0, 0] = params[2]
   mod.updater = updater

   # State space models must be initialized; we use this
   # method because AR(p) models are assumed to be stationary
   mod.initialize_stationary()

   # Fit the model via maximum likelihood
   # Note: must specify start parameters
   mod.start_params = [0,0,1]
   res = mod.fit()
   print res.summary()

This results in the following summary table::

                              Statespace Model Results                           
   ==============================================================================
   Dep. Variable:                      y   No. Observations:                 1000
   Model:                       MLEModel   Log Likelihood               -1389.437
   Date:                Mon,  1 Jan 2000   AIC                           2784.874
   Time:                        11:11:00   BIC                           2799.598
   Sample:                             0   HQIC                          2790.470
                                  - 1000                                         
   ==============================================================================
                    coef    std err          z      P>|z|      [95.0% Conf. Int.]
   ------------------------------------------------------------------------------
   param.0        0.4395      0.031     14.195      0.000         0.379     0.500
   param.1       -0.2055      0.031     -6.635      0.000        -0.266    -0.145
   param.2        0.9425      0.042     22.366      0.000         0.860     1.025
   ==============================================================================

The results object has many of the attributes and methods you would expect from
other Statsmodels results objects, including standard errors, z-statistics,
and prediction / forecasting.

More advanced usage is possible, including specifying parameter
transformations, and specifing names for parameters for a more informative
output summary. Note that in most cases, it will be more convenient to specify
models as a subclass of MLEModel, as is done in the local linear trend example.

State space representation and Kalman filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While creation of custom models will almost always be done by extending
`MLEModel` and `MLEResults`, it can be useful to understand the superstructure
behind those classes.

Maximum likelihood estimation requires evaluating the likelihood function of
the model, and for models in state space form the likelihood function is
evaluted as a byproduct of running the Kalman filter.

There are two superclasses of `MLEModel` that facilitate specification of the
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
in-sample prediction or out-of-sample forecasting.

.. autosummary::
   :toctree: generated/

   kalman_filter.KalmanFilter
   kalman_filter.FilterResults


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
   tools.validate_matrix_shape
   tools.validate_vector_shape
