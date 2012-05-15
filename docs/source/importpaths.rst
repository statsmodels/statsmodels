
Import Paths and Structure
==========================

We have two ways of importing functions and classes from statsmodels. One
way is designed for interactive use, where tab completion makes it easy to see
what is available and to save on typing. The other way is designed for use in
a program, where we do not want to import modules that are not needed for
the purpose on hand.

The API Import
--------------

importing statsmodels.api will load most of the public parts of statsmodels
and make functions and classes available within one or two levels.
or directly from a module (long path) to get minimal imports

For interactive use the recommended import is

>>> import statsmodels.api as sm

The following illustrates what is currently available (and changes as new
functionality is added). Instead of using dir(xxx), we can get the same
information by tab completion in most python editors, for example Ipython and
Spyder, or IDLE.

>>> dir(sm)

['GLM', 'GLS', 'GLSAR', 'Logit', 'MNLogit', 'OLS', 'Poisson', 'Probit', 'RLM',
'WLS', '__builtins__', '__doc__', '__file__', '__name__', '__package__',
'add_constant', 'categorical', 'datasets', 'distributions', 'families',
'graphics', 'iolib', 'nonparametric', 'qqplot', 'regression', 'robust',
'stats', 'test', 'tools', 'tsa', 'version']

>>> dir(sm.nonparametric)
['KDE', '__builtins__', '__doc__', '__file__', '__name__',
'__package__', 'bandwidths', 'lowess']

>>> dir(sm.graphics)
['__builtins__', '__doc__', '__file__', '__name__', '__package__',
'abline_plot', 'beanplot', 'fboxplot', 'interaction_plot', 'qqplot',
'rainbow', 'rainbowplot', 'violinplot']

>>> dir(sm.tsa)
['AR', 'ARMA', 'DynamicVAR', 'SVAR', 'VAR', '__builtins__', '__doc__',
'__file__', '__name__', '__package__', 'acf', 'acovf', 'add_lag',
'add_trend', 'adfuller', 'ccf', 'ccovf', 'datetools', 'detrend',
'filters', 'grangercausalitytests', 'interp', 'lagmat', 'lagmat2ds',
'pacf', 'pacf_ols', 'pacf_yw', 'periodogram', 'q_stat', 'stattools',
'tsatools', 'var']

>>> dir(sm.tsa.tsatools)
...

The idea is to be able to access the commonly used models and functions of
statsmodels from "sm" (statsmodels.api) directly or within one level. We add
one level so the `sm` namespace does not get too crowded.

The `api` modules contain the main public functionality of statsmodels.
Functions that are not in the `api's` are not clearly marked as to whether they
are considered public or private. If you find something that should be
added to the api, then please file an issue on github or report it to the
mailing list.

**Detail**

In the subpackages of statsmodels we have `api.py` modules that are mainly to
collect the imports from a subpackage. Those `subpackage/api.py` are imported
into statsmodels api, for example ::

     from .nonparametric import api as nonparametric

Users do not need to load the `subpackage/api.py` modules directly.


Direct Imports - File and Directory Structure
---------------------------------------------
Our directory tree stripped down looks something like::

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

The submodules are arranged by topic, `discrete` for discrete choice models,
or `tsa` for time series analysis. The submodules that can be import heavy
contain an empty `__init__.py`, except for some testing code for running tests
for the submodules. The intention is to change all directories to have an
`api.py` and empty `__init__.py` in the next release.

The following are some examples for imports that are used within statsmodels.

importing functions and classes::

    from statsmodels.regression.linear_model import OLS, WLS
    from statsmodels.tools.tools import rank, add_constant

importing modules ::

    from statsmodels.datasets import macrodata
    import statsmodels.stats import diagnostic

importing modules with alias ::

    import statsmodels.regression.linear_model as lm
    import statsmodels.stats.diagnostic as smsdia
    import statsmodels.stats.outliers_influence as oi

We do not have currently a convention for aliases of submodules.

