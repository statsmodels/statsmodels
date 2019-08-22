API
---
The main ~statsmodels API is split into models:

* ``~statsmodels.api``: Cross-sectional models and methods. Canonically imported
  using ``import ~statsmodels.api as sm``.
* ``~statsmodels.tsa.api``: Time-series models and methods. Canonically imported
  using ``import ~statsmodels.tsa.api as tsa``.

The API focuses on models and the most frequently used statistical test, and tools.  See the
detailed topic pages for a more complete list of available models, statistics, and tools.

``statsmodels.api``
===================

Regression
~~~~~~~~~~
.. autosummary::

   ~statsmodels.regression.linear_model.OLS
   ~statsmodels.regression.linear_model.GLS
   ~statsmodels.regression.linear_model.GLSAR
   ~statsmodels.regression.linear_model.WLS
   ~statsmodels.regression.recursive_ls.RecursiveLS
   ~statsmodels.regression.rolling.RollingOLS
   ~statsmodels.regression.rolling.RollingWLS

Imputation
~~~~~~~~~~
.. autosummary::

   ~statsmodels.imputation.bayes_mi.BayesGaussMI
   ~statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM
   ~statsmodels.multivariate.factor.Factor
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
   ~statsmodels.genmod.bayes_mixed_glm.PoissonBayesMixedGLM

Discrete and Count Models
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::

   ~statsmodels.discrete.discrete_model.GeneralizedPoisson
   ~statsmodels.discrete.discrete_model.Logit
   ~statsmodels.discrete.discrete_model.MNLogit
   ~statsmodels.discrete.discrete_model.Poisson
   ~statsmodels.discrete.discrete_model.Probit
   ~statsmodels.discrete.discrete_model.NegativeBinomial
   ~statsmodels.discrete.discrete_model.NegativeBinomialP
   ~statsmodels.discrete.count_model.ZeroInflatedGeneralizedPoisson
   ~statsmodels.discrete.count_model.ZeroInflatedNegativeBinomialP
   ~statsmodels.discrete.count_model.ZeroInflatedPoisson

Multivarite Models
~~~~~~~~~~~~~~~~~~
.. autosummary::

   ~statsmodels.multivariate.manova.MANOVA
   ~statsmodels.multivariate.pca.PCA

Misc Models
~~~~~~~~~~~
.. autosummary::

   ~statsmodels.regression.mixed_linear_model.MixedLM
   ~statsmodels.duration.hazard_regression.PHReg
   ~statsmodels.regression.quantile_regression.QuantReg
   ~statsmodels.robust.robust_linear_model.RLM
   ~statsmodels.duration.survfunc.SurvfuncRight


Graphics
~~~~~~~~
.. autosummary::

   ~statsmodels.graphics.gofplots.ProbPlot
   ~statsmodels.graphics.gofplots.qqline
   ~statsmodels.graphics.gofplots.qqplot
   ~statsmodels.graphics.gofplots.qqplot_2samples

Tools
~~~~~
.. autosummary::

   ~statsmodels.__init__.test
   ~statsmodels.tools.tools.add_constant
   ~statsmodels.tools.tools.categorical
   ~statsmodels.iolib.smpickle.load_pickle
   ~statsmodels.tools.print_version.show_versions
   ~statsmodels.tools.web.webdoc


``statsmodels.tsa.api``
=======================

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
   ~statsmodels.tsa.stattools.periodogram
   ~statsmodels.tsa.stattools.q_stat

Univariate Time Series Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~statsmodels.tsa.ar_model.AR
   ~statsmodels.tsa.arima_model.ARIMA
   ~statsmodels.tsa.arima_model.ARMA
   ~statsmodels.tsa.statespace.sarimax.SARIMAX
   ~statsmodels.tsa.stattools.arma_order_select_ic

Exponential Smoothing
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~statsmodels.tsa.holtwinters.ExponentialSmoothing
   ~statsmodels.tsa.holtwinters.Holt
   ~statsmodels.tsa.holtwinters.SimpleExpSmoothing


Multivariate Models
~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~statsmodels.tsa.statespace.dynamic_factor.DynamicFactor
   ~statsmodels.tsa.vector_ar.var_model.VAR
   ~statsmodels.tsa.statespace.varmax.VARMAX
   ~statsmodels.tsa.vector_ar.svar_model.SVAR
   ~statsmodels.tsa.vector_ar.vecm.VECM
   ~statsmodels.tsa.vector_ar.dynamic.DynamicVAR
   ~statsmodels.tsa.statespace.structural.UnobservedComponents

Tools
~~~~~

.. autosummary::

   ~statsmodels.tsa.tsatools.add_lag
   ~statsmodels.tsa.tsatools.add_trend
   ~statsmodels.tsa.arima_process.arma_generate_sample
   ~statsmodels.tsa.arima_process.ArmaProcess
   ~statsmodels.tsa.tsatools.detrend
   ~statsmodels.tsa.tsatools.lagmat
   ~statsmodels.tsa.tsatools.lagmat2ds
   ~statsmodels.tsa.seasonal.seasonal_decompose

Markov Switching
~~~~~~~~~~~~~~~~

.. autosummary::

   ~statsmodels.tsa.regime_switching.markov_autoregression.MarkovAutoregression
   ~statsmodels.tsa.regime_switching.markov_regression.MarkovRegression

X12/X13 Interface
~~~~~~~~~~~~~~~~~

.. autosummary::

   ~statsmodels.tsa.x13.x13_arima_analysis
   ~statsmodels.tsa.x13.x13_arima_select_order
