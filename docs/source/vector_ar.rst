:orphan:

.. module:: statsmodels.tsa.vector_ar.var_model
   :synopsis: Vector autoregressions

.. currentmodule:: statsmodels.tsa.vector_ar.var_model

.. _var:

Vector Autoregressions :mod:`tsa.vector_ar`
===========================================

:mod:`statsmodels.tsa.vector_ar` contains methods that are useful
for simultaneously modeling and analyzing multiple time series using
:ref:`Vector Autoregressions (VAR) <var>` and
:ref:`Vector Error Correction Models (VECM) <vecm>`.

.. _var_process:

VAR(p) processes
----------------

We are interested in modeling a :math:`T \times K` multivariate time series
:math:`Y`, where :math:`T` denotes the number of observations and :math:`K` the
number of variables. One way of estimating relationships between the time series
and their lagged values is the *vector autoregression process*:

.. math::

   Y_t = \nu + A_1 Y_{t-1} + \ldots + A_p Y_{t-p} + u_t

   u_t \sim {\sf Normal}(0, \Sigma_u)

where :math:`A_i` is a :math:`K \times K` coefficient matrix.

We follow in large part the methods and notation of `Lutkepohl (2005)
<https://www.springer.com/gb/book/9783540401728>`__,
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
   :okwarning:

   # some example data
   import numpy as np
   import pandas
   import statsmodels.api as sm
   from statsmodels.tsa.api import VAR
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
   :okwarning:

   results = model.fit(2)
   results.summary()

Several ways to visualize the data using `matplotlib` are available.

Plotting input time series:

.. ipython:: python
   :okwarning:

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

Class Reference
~~~~~~~~~~~~~~~

.. module:: statsmodels.tsa.vector_ar
   :synopsis: Vector autoregressions and related tools

.. currentmodule:: statsmodels.tsa.vector_ar.var_model


.. autosummary::
   :toctree: generated/

   VAR
   VARProcess
   VARResults


Post-estimation Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Several process properties and additional results after
estimation are available for vector autoregressive processes.

.. currentmodule:: statsmodels.tsa.vector_ar.var_model
.. autosummary::
   :toctree: generated/

   LagOrderResults

.. currentmodule:: statsmodels.tsa.vector_ar.hypothesis_test_results
.. autosummary::
   :toctree: generated/

   HypothesisTestResults
   NormalityTestResults
   WhitenessTestResults


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
   :okwarning:

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


.. currentmodule:: statsmodels.tsa.vector_ar.irf
.. autosummary::
   :toctree: generated/

   IRAnalysis

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


.. currentmodule:: statsmodels.tsa.vector_ar.var_model
.. autosummary::
   :toctree: generated/

   FEVD

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

As pointed out in the beginning of this document, the white noise component
:math:`u_t` is assumed to be normally distributed. While this assumption
is not required for parameter estimates to be consistent or asymptotically
normal, results are generally more reliable in finite samples when residuals
are Gaussian white noise. To test whether this assumption is consistent with
a data set, :class:`VARResults` offers the `test_normality` method.

.. ipython:: python

    results.test_normality()

Whiteness of residuals
~~~~~~~~~~~~~~~~~~~~~~

To test the whiteness of the estimation residuals (this means absence of
significant residual autocorrelations) one can use the `test_whiteness`
method of :class:`VARResults`.


.. currentmodule:: statsmodels.tsa.vector_ar.hypothesis_test_results
.. autosummary::
   :toctree: generated/

   HypothesisTestResults
   CausalityTestResults
   NormalityTestResults
   WhitenessTestResults

.. _svar:

Structural Vector Autoregressions
---------------------------------

There are a matching set of classes that handle some types of Structural VAR models.

.. module:: statsmodels.tsa.vector_ar.svar_model
   :synopsis: Structural vector autoregressions and related tools

.. currentmodule:: statsmodels.tsa.vector_ar.svar_model

.. autosummary::
   :toctree: generated/

   SVAR
   SVARProcess
   SVARResults

.. _vecm:

Vector Error Correction Models (VECM)
-------------------------------------

Vector Error Correction Models are used to study short-run deviations from
one or more permanent stochastic trends (unit roots). A VECM models the
difference of a vector of time series by imposing structure that is implied
by the assumed number of stochastic trends. :class:`VECM` is used to
specify and estimate these models.

A VECM(:math:`k_{ar}-1`) has the following form

.. math::

    \Delta y_t = \Pi y_{t-1} + \Gamma_1 \Delta y_{t-1} + \ldots
                   + \Gamma_{k_{ar}-1} \Delta y_{t-k_{ar}+1} + u_t

where

.. math::

    \Pi = \alpha \beta'

as described in chapter 7 of [1]_.

A VECM(:math:`k_{ar} - 1`) with deterministic terms has the form

.. math::

   \Delta y_t = \alpha \begin{pmatrix}\beta' & \eta'\end{pmatrix} \begin{pmatrix}y_{t-1} \\
                D^{co}_{t-1}\end{pmatrix} + \Gamma_1 \Delta y_{t-1} + \dots + \Gamma_{k_{ar}-1} \Delta y_{t-k_{ar}+1} + C D_t + u_t.

In :math:`D^{co}_{t-1}` we have the deterministic terms which are inside
the cointegration relation (or restricted to the cointegration relation).
:math:`\eta` is the corresponding estimator. To pass a deterministic term
inside the cointegration relation, we can use the `exog_coint` argument.
For the two special cases of an intercept and a linear trend there exists
a simpler way to declare these terms: we can pass ``"ci"`` and ``"li"``
respectively to the `deterministic` argument. So for an intercept inside
the cointegration relation we can either pass ``"ci"`` as `deterministic`
or `np.ones(len(data))` as `exog_coint` if `data` is passed as the
`endog` argument. This ensures that :math:`D_{t-1}^{co} = 1` for all
:math:`t`.

We can also use deterministic terms outside the cointegration relation.
These are defined in :math:`D_t` in the formula above with the
corresponding estimators in the matrix :math:`C`. We specify such terms by
passing them to the `exog` argument. For an intercept and/or linear trend
we again have the possibility to use `deterministic` alternatively. For
an intercept we pass ``"co"`` and for a linear trend we pass ``"lo"`` where
the `o` stands for `outside`.

The following table shows the five cases considered in [2]_. The last
column indicates which string to pass to the `deterministic` argument for
each of these cases.

====  ===============================  ===================================  =============
Case  Intercept                        Slope of the linear trend            `deterministic`
====  ===============================  ===================================  =============
I     0                                0                                    ``"nc"``
II    :math:`- \alpha \beta^T \mu`     0                                    ``"ci"``
III   :math:`\neq 0`                   0                                    ``"co"``
IV    :math:`\neq 0`                   :math:`- \alpha \beta^T \gamma`      ``"coli"``
V     :math:`\neq 0`                   :math:`\neq 0`                       ``"colo"``
====  ===============================  ===================================  =============

.. currentmodule:: statsmodels.tsa.vector_ar.vecm
.. autosummary::
   :toctree: generated/

   VECM
   coint_johansen
   JohansenTestResult
   select_order
   select_coint_rank
   VECMResults
   CointRankResults


References
----------
.. [1] Lütkepohl, H. 2005. *New Introduction to Multiple Time Series Analysis*. Springer.

.. [2] Johansen, S. 1995. *Likelihood-Based Inference in Cointegrated *
       *Vector Autoregressive Models*. Oxford University Press.
