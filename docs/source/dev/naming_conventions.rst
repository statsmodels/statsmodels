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
code for running tests for the submodules. The namespace to be imported is in `api.py`. That way, we
can import selectively and do not have to import a lot of code that we do not need. Helper functions are
usually put in files named `tools.py` and statistical functions, such as statistical tests are placed
in `stattools.py`. Everything has directories for :ref:`tests <testing>`.

`endog` & `exog`
~~~~~~~~~~~~~~~~

Our working definition of a statistical model is an object that has
both endogenous and exogenous data defined as well as a statistical
relationship.  In place of endogenous and exogenous one can often substitute
the terms left hand side (LHS) and right hand side (RHS), dependent and
independent variables, regressand and regressors, outcome and design, response
variable and explanatory variable, respectively.  The usage is quite often
domain specific; however, we have chosen to use `endog` and `exog` almost
exclusively, since the principal developers of statsmodels have a background
in econometrics, and this feels most natural.  This means that all of the
models are objects with `endog` and `exog` defined, though in some cases
`exog` is None for convenience (for instance, with an autoregressive process).
Each object also defines a `fit` (or similar) method that returns a
model-specific results object.  In addition there are some functions, e.g. for
statistical tests or convenience functions.

See also the related explanation in :ref:`endog_exog`.

Variable Names
~~~~~~~~~~~~~~
All of our models assume that data is arranged with variables in columns. Thus, internally the data
is all 2d arrays. By convention, we will prepend a `k_` to variable names that indicate moving over
axis 1 (columns), and `n_` to variables that indicate moving over axis 0 (rows). The main exception to
the underscore is that `nobs` should indicate the number of observations. For example, in the
time-series ARMA model we have::

    `k_ar` - The number of AR lags included in the RHS variables
    `k_ma` - The number of MA lags included in the RHS variables
    `k_trend` - The number of trend variables included in the RHS variables
    `k_exog` - The number of exogenous variables included in the RHS variables excluding the trend terms
    `n_totobs` - The total number of observations for the LHS variables including the pre-sample values


Options
~~~~~~~
We are using similar options in many classes, methods and functions. They
should follow a standardized pattern if they recur frequently. ::

    `missing` ['none', 'drop', 'raise'] define whether inputs are checked for
        nans, and how they are treated
    `alpha` (float in (0, 1)) significance level for hypothesis tests and
        confidence intervals, e.g. `alpha=0.05`

patterns ::

    `return_xxx` : boolean to indicate optional or different returns
        (not `ret_xxx`)
