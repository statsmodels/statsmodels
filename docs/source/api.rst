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
`Import Paths and Structure`_ explains the design of the two API modules and how
importing from the API differs from directly importing from the module where the
model is defined. See the detailed topic pages in the :ref:`user-guide:User Guide` for a complete
list of available models, statistics, and tools.

``statsmodels.api``
-------------------

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

Multivariate Models
~~~~~~~~~~~~~~~~~~~
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
   ~statsmodels.tsa.stattools.periodogram
   ~statsmodels.tsa.stattools.q_stat

Univariate Time-Series Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~statsmodels.tsa.ar_model.AR
   ~statsmodels.tsa.arima_model.ARIMA
   ~statsmodels.tsa.arima_model.ARMA
   ~statsmodels.tsa.statespace.sarimax.SARIMAX
   ~statsmodels.tsa.stattools.arma_order_select_ic
   ~statsmodels.tsa.arima_process.arma_generate_sample
   ~statsmodels.tsa.arima_process.ArmaProcess

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
   ~statsmodels.tsa.statespace.structural.UnobservedComponents

Filters and Decompositions
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~statsmodels.tsa.seasonal.seasonal_decompose
   ~statsmodels.tsa.seasonal.STL
   ~statsmodels.tsa.filters.bk_filter.bkfilter
   ~statsmodels.tsa.filters.cf_filter.cffilter
   ~statsmodels.tsa.filters.hp_filter.hpfilter

Markov Regime Switching Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~statsmodels.tsa.regime_switching.markov_autoregression.MarkovAutoregression
   ~statsmodels.tsa.regime_switching.markov_regression.MarkovRegression

Time-Series Tools
~~~~~~~~~~~~~~~~~

.. autosummary::

   ~statsmodels.tsa.tsatools.add_lag
   ~statsmodels.tsa.tsatools.add_trend
   ~statsmodels.tsa.tsatools.detrend
   ~statsmodels.tsa.tsatools.lagmat
   ~statsmodels.tsa.tsatools.lagmat2ds

X12/X13 Interface
~~~~~~~~~~~~~~~~~

.. autosummary::

   ~statsmodels.tsa.x13.x13_arima_analysis
   ~statsmodels.tsa.x13.x13_arima_select_order

``statsmodels.formula.api``
---------------------------

Models
~~~~~~

The function descriptions of the methods exposed in the formula API are generic.
See the documentation for the parent model for details.

.. autosummary::
   :toctree: generated/

   ~statsmodels.formula.api.gls
   ~statsmodels.formula.api.wls
   ~statsmodels.formula.api.ols
   ~statsmodels.formula.api.glsar
   ~statsmodels.formula.api.mixedlm
   ~statsmodels.formula.api.glm
   ~statsmodels.formula.api.rlm
   ~statsmodels.formula.api.mnlogit
   ~statsmodels.formula.api.logit
   ~statsmodels.formula.api.probit
   ~statsmodels.formula.api.poisson
   ~statsmodels.formula.api.negativebinomial
   ~statsmodels.formula.api.quantreg
   ~statsmodels.formula.api.phreg
   ~statsmodels.formula.api.ordinal_gee
   ~statsmodels.formula.api.nominal_gee
   ~statsmodels.formula.api.gee
   ~statsmodels.formula.api.glmgam


.. _importpaths:

Import Paths and Structure
--------------------------

We offer two ways of importing functions and classes from statsmodels:

1. `API import for interactive use`_

   + Allows tab completion

2. `Direct import for programs`_

   + Avoids importing unnecessary modules and commands

API Import for interactive use
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For interactive use the recommended import is:

.. code-block:: python

    import statsmodels.api as sm

Importing `statsmodels.api` will load most of the public parts of statsmodels.
This makes most functions and classes conveniently available within one or two
levels, without making the "sm" namespace too crowded.

To see what functions and classes available, you can type the following (or use
the namespace exploration features of IPython, Spyder, IDLE, etc.):

.. code-block:: python

    >>> dir(sm)
    ['GLM', 'GLS', 'GLSAR', 'Logit', 'MNLogit', 'OLS', 'Poisson', 'Probit', 'RLM',
    'WLS', '__builtins__', '__doc__', '__file__', '__name__', '__package__',
    'add_constant', 'categorical', 'datasets', 'distributions', 'families',
    'graphics', 'iolib', 'nonparametric', 'qqplot', 'regression', 'robust',
    'stats', 'test', 'tools', 'tsa', 'version']

    >>> dir(sm.graphics)
    ['__builtins__', '__doc__', '__file__', '__name__', '__package__',
    'abline_plot', 'beanplot', 'fboxplot', 'interaction_plot', 'qqplot',
    'rainbow', 'rainbowplot', 'violinplot']

    >>> dir(sm.tsa)
    ['AR', 'ARMA', 'SVAR', 'VAR', '__builtins__', '__doc__',
    '__file__', '__name__', '__package__', 'acf', 'acovf', 'add_lag',
    'add_trend', 'adfuller', 'ccf', 'ccovf', 'datetools', 'detrend',
    'filters', 'grangercausalitytests', 'interp', 'lagmat', 'lagmat2ds',
    'pacf', 'pacf_ols', 'pacf_yw', 'periodogram', 'q_stat', 'stattools',
    'tsatools', 'var']

Notes
^^^^^

The `api` modules may not include all the public functionality of statsmodels. If
you find something that should be added to the api, please file an issue on
github or report it to the mailing list.

The subpackages of statsmodels include `api.py` modules that are mainly
intended to collect the imports needed for those subpackages. The `subpackage/api.py`
files are imported into statsmodels api, for example ::

     from .nonparametric import api as nonparametric

Users do not need to load the `subpackage/api.py` modules directly.

Direct import for programs
~~~~~~~~~~~~~~~~~~~~~~~~~~

``statsmodels`` submodules are arranged by topic (e.g. `discrete` for discrete
choice models, or `tsa` for time series analysis). Our directory tree (stripped
down) looks something like this::

    statsmodels/
        __init__.py
        api.py
        discrete/
            __init__.py
            discrete_model.py
            tests/
                results/
        tsa/
            __init__.py
            api.py
            tsatools.py
            stattools.py
            arima_model.py
            arima_process.py
            vector_ar/
                __init__.py
                var_model.py
                tests/
                    results/
            tests/
                results/
        stats/
            __init__.py
            api.py
            stattools.py
            tests/
        tools/
            __init__.py
            tools.py
            decorators.py
            tests/

The submodules that can be import heavy contain an empty `__init__.py`, except
for some testing code for running tests for the submodules. The intention is to
change all directories to have an `api.py` and empty `__init__.py` in the next
release.

Import examples
^^^^^^^^^^^^^^^

Functions and classes::

    from statsmodels.regression.linear_model import OLS, WLS
    from statsmodels.tools.tools import rank, add_constant

Modules ::

    from statsmodels.datasets import macrodata
    import statsmodels.stats import diagnostic

Modules with aliases ::

    import statsmodels.regression.linear_model as lm
    import statsmodels.stats.diagnostic as smsdia
    import statsmodels.stats.outliers_influence as oi

We do not have currently a convention for aliases of submodules.

