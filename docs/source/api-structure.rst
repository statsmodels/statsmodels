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
