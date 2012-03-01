Naming Conventions
------------------

File and Directory Names
~~~~~~~~~~~~~~~~~~~~~~~~
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

The submodules are arranged by topic, `discrete` for discrete choice models, or `tsa` for time series
analysis. The submodules that can be import heavy contain an empty __init__.py, except for some testing
code for running tests for the submodules. The namespace to be imported in in `api.py`. That way, we
can import selectively and not have to import a lot of code that we don't need. Helper functions are
usually put in files named `tools.py` and statistical functions, such as statistical tests are placed 
in `stattools.py`. Everything has directores for :ref:`tests <testing>`.

Variable Names
~~~~~~~~~~~~~~
All of our models assume that data is arranged with variables in columns. Thus, internally the data
is all 2d arrays. By convention, we will prepend a `k_` to variable names that indicate moving over 
axis 1 (columns), and `n_` to variables that indicate moving over axis 0 (rows). The main exception to
the underscore is that `nobs` should indicate the number of observations. For example, in the 
time-series ARMA model we have::

    k_ar - The number of AR lags included in the RHS variables
    k_ma - The number of MA lags included in the RHS variables
    k_trend - The number of trend variables included in the RHS variables
    k_exog - The number of exogenous variables included in the RHS variables exluding the trend terms
    n_totobs - The total number of observations for the LHS variables including the pre-sample values
