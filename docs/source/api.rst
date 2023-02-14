API Reference
=============
The main statsmodels API is split into models:

* ``statsmodels.api``: Cross-sectional models and methods. Canonically imported
  using ``import statsmodels.api as sm``.
* ``statsmodels.tsa.api``: Time-series models and methods. Canonically imported
  using ``import statsmodels.tsa.api as tsa``.
* ``statsmodels.formula.api``: A convenience interface for specifying models
  using formula strings and DataFrames. This API directly exposes the ``from_formula``
  class method of models that support the formula API. Canonically imported using
  ``import statsmodels.formula.api as smf``

The API focuses on models and the most frequently used statistical test, and tools.
:ref:`api-structure:Import Paths and Structure` explains the design of the two API modules and how
importing from the API differs from directly importing from the module where the
model is defined. See the detailed topic pages in the :ref:`user-guide:User Guide` for a complete
list of available models, statistics, and tools.

``statsmodels.api``
-------------------

Regression
~~~~~~~~~~
.. autosummary::

   ~statsmodels.regression.linear_model.OLS
   ~statsmodels.regression.linear_model.WLS
   ~statsmodels.regression.linear_model.GLS
   ~statsmodels.regression.linear_model.GLSAR
   ~statsmodels.regression.recursive_ls.RecursiveLS
   ~statsmodels.regression.rolling.RollingOLS
   ~statsmodels.regression.rolling.RollingWLS

Imputation
~~~~~~~~~~
.. autosummary::

   ~statsmodels.imputation.bayes_mi.BayesGaussMI
   ~statsmodels.imputation.bayes_mi.MI
   ~statsmodels.imputation.mice.MICE
   ~statsmodels.imputation.mice.MICEData

Generalized Estimating Equations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::

   ~statsmodels.genmod.generalized_estimating_equations.GEE
   ~statsmodels.genmod.generalized_estimating_equations.NominalGEE
   ~statsmodels.genmod.generalized_estimating_equations.OrdinalGEE

Generalized Linear Models
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::

   ~statsmodels.genmod.generalized_linear_model.GLM
   ~statsmodels.gam.generalized_additive_model.GLMGam
   ~statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM
   ~statsmodels.genmod.bayes_mixed_glm.PoissonBayesMixedGLM

Discrete and Count Models
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::

   ~statsmodels.discrete.discrete_model.Logit
   ~statsmodels.discrete.discrete_model.Probit
   ~statsmodels.discrete.discrete_model.MNLogit
   ~statsmodels.miscmodels.ordinal_model.OrderedModel
   ~statsmodels.discrete.discrete_model.Poisson
   ~statsmodels.discrete.discrete_model.NegativeBinomial
   ~statsmodels.discrete.discrete_model.NegativeBinomialP
   ~statsmodels.discrete.discrete_model.GeneralizedPoisson
   ~statsmodels.discrete.count_model.ZeroInflatedPoisson
   ~statsmodels.discrete.count_model.ZeroInflatedNegativeBinomialP
   ~statsmodels.discrete.count_model.ZeroInflatedGeneralizedPoisson
   ~statsmodels.discrete.conditional_models.ConditionalLogit
   ~statsmodels.discrete.conditional_models.ConditionalMNLogit
   ~statsmodels.discrete.conditional_models.ConditionalPoisson


Multivariate Models
~~~~~~~~~~~~~~~~~~~
.. autosummary::

   ~statsmodels.multivariate.factor.Factor
   ~statsmodels.multivariate.manova.MANOVA
   ~statsmodels.multivariate.pca.PCA

Other Models
~~~~~~~~~~~~
.. autosummary::

   ~statsmodels.regression.mixed_linear_model.MixedLM
   ~statsmodels.duration.survfunc.SurvfuncRight
   ~statsmodels.duration.hazard_regression.PHReg
   ~statsmodels.regression.quantile_regression.QuantReg
   ~statsmodels.robust.robust_linear_model.RLM
   ~statsmodels.othermod.betareg.BetaModel

Graphics
~~~~~~~~
.. autosummary::

   ~statsmodels.graphics.gofplots.ProbPlot
   ~statsmodels.graphics.gofplots.qqline
   ~statsmodels.graphics.gofplots.qqplot
   ~statsmodels.graphics.gofplots.qqplot_2samples

Statistics
~~~~~~~~~~
.. autosummary::

   ~statsmodels.stats.descriptivestats.Description
   ~statsmodels.stats.descriptivestats.describe

Tools
~~~~~
.. autosummary::

   ~statsmodels.__init__.test
   ~statsmodels.tools.tools.add_constant
   ~statsmodels.iolib.smpickle.load_pickle
   ~statsmodels.tools.print_version.show_versions
   ~statsmodels.tools.web.webdoc


``statsmodels.tsa.api``
-----------------------

Statistics and Tests
~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~statsmodels.tsa.stattools.acf
   ~statsmodels.tsa.stattools.acovf
   ~statsmodels.tsa.stattools.adfuller
   ~statsmodels.tsa.stattools.bds
   ~statsmodels.tsa.stattools.ccf
   ~statsmodels.tsa.stattools.ccovf
   ~statsmodels.tsa.stattools.coint
   ~statsmodels.tsa.stattools.kpss
   ~statsmodels.tsa.stattools.pacf
   ~statsmodels.tsa.stattools.pacf_ols
   ~statsmodels.tsa.stattools.pacf_yw
   ~statsmodels.tsa.stattools.q_stat
   ~statsmodels.tsa.stattools.range_unit_root_test
   ~statsmodels.tsa.stattools.zivot_andrews


Univariate Time-Series Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~statsmodels.tsa.ar_model.AutoReg
   ~statsmodels.tsa.ardl.ARDL
   ~statsmodels.tsa.arima.model.ARIMA
   ~statsmodels.tsa.statespace.sarimax.SARIMAX
   ~statsmodels.tsa.ardl.ardl_select_order
   ~statsmodels.tsa.stattools.arma_order_select_ic
   ~statsmodels.tsa.arima_process.arma_generate_sample
   ~statsmodels.tsa.arima_process.ArmaProcess
   ~statsmodels.tsa.ardl.UECM

Exponential Smoothing
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~statsmodels.tsa.holtwinters.ExponentialSmoothing
   ~statsmodels.tsa.holtwinters.Holt
   ~statsmodels.tsa.holtwinters.SimpleExpSmoothing
   ~statsmodels.tsa.statespace.exponential_smoothing.ExponentialSmoothing
   ~statsmodels.tsa.exponential_smoothing.ets.ETSModel


Multivariate Time Series Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~statsmodels.tsa.statespace.dynamic_factor.DynamicFactor
   ~statsmodels.tsa.statespace.dynamic_factor_mq.DynamicFactorMQ
   ~statsmodels.tsa.vector_ar.var_model.VAR
   ~statsmodels.tsa.statespace.varmax.VARMAX
   ~statsmodels.tsa.vector_ar.svar_model.SVAR
   ~statsmodels.tsa.vector_ar.vecm.VECM
   ~statsmodels.tsa.statespace.structural.UnobservedComponents

Filters and Decompositions
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~statsmodels.tsa.seasonal.seasonal_decompose
   ~statsmodels.tsa.seasonal.STL
   ~statsmodels.tsa.seasonal.MSTL
   ~statsmodels.tsa.filters.bk_filter.bkfilter
   ~statsmodels.tsa.filters.cf_filter.cffilter
   ~statsmodels.tsa.filters.hp_filter.hpfilter

Markov Regime Switching Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~statsmodels.tsa.regime_switching.markov_autoregression.MarkovAutoregression
   ~statsmodels.tsa.regime_switching.markov_regression.MarkovRegression

Forecasting
~~~~~~~~~~~

.. autosummary::

   ~statsmodels.tsa.forecasting.stl.STLForecast
   ~statsmodels.tsa.forecasting.theta.ThetaModel

Time-Series Tools
~~~~~~~~~~~~~~~~~

.. autosummary::

   ~statsmodels.tsa.tsatools.add_lag
   ~statsmodels.tsa.tsatools.add_trend
   ~statsmodels.tsa.tsatools.detrend
   ~statsmodels.tsa.tsatools.lagmat
   ~statsmodels.tsa.tsatools.lagmat2ds
   ~statsmodels.tsa.deterministic.DeterministicProcess

X12/X13 Interface
~~~~~~~~~~~~~~~~~

.. autosummary::

   ~statsmodels.tsa.x13.x13_arima_analysis
   ~statsmodels.tsa.x13.x13_arima_select_order

``statsmodels.formula.api``
---------------------------

Models
~~~~~~

The lower case names are aliases to the `from_formula` method of the
corresponding model class. The function descriptions of the methods exposed in
the formula API are generic. See the documentation for the parent model for
details.

.. autosummary::
   :toctree: generated/

   ~statsmodels.formula.api.gls
   ~statsmodels.formula.api.wls
   ~statsmodels.formula.api.ols
   ~statsmodels.formula.api.glsar
   ~statsmodels.formula.api.mixedlm
   ~statsmodels.formula.api.glm
   ~statsmodels.formula.api.gee
   ~statsmodels.formula.api.ordinal_gee
   ~statsmodels.formula.api.nominal_gee
   ~statsmodels.formula.api.rlm
   ~statsmodels.formula.api.logit
   ~statsmodels.formula.api.probit
   ~statsmodels.formula.api.mnlogit
   ~statsmodels.formula.api.poisson
   ~statsmodels.formula.api.negativebinomial
   ~statsmodels.formula.api.quantreg
   ~statsmodels.formula.api.phreg
   ~statsmodels.formula.api.glmgam
   ~statsmodels.formula.api.conditional_logit
   ~statsmodels.formula.api.conditional_mnlogit
   ~statsmodels.formula.api.conditional_poisson
