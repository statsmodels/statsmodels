Getting started
===============

This very simple case-study is designed to get you up-and-running quickly with
``statsmodels``. Starting from raw data, we will show the steps needed to
estimate a statistical model and to draw a diagnostic plot. We will only use
functions provided by ``statsmodels`` or its ``pandas`` and ``patsy``
dependencies.

Loading modules and functions
-----------------------------

After `installing statsmodels and its dependencies <install.html>`_, we load a
few modules and functions:

.. ipython:: python

    import statsmodels.api as sm
    import pandas
    from patsy import dmatrices

`pandas <https://pandas.pydata.org/>`_ builds on ``numpy`` arrays to provide
rich data structures and data analysis tools. The ``pandas.DataFrame`` function
provides labelled arrays of (potentially heterogenous) data, similar to the
``R`` "data.frame". The ``pandas.read_csv`` function can be used to convert a
comma-separated values file to a ``DataFrame`` object.

`patsy <https://github.com/pydata/patsy>`_ is a Python library for describing
statistical models and building `Design Matrices
<https://en.wikipedia.org/wiki/Design_matrix>`_ using ``R``-like formulas.

.. note::

   This example uses the API interface.  See :ref:`importpaths` for information on
   the difference between importing the API interfaces (``statsmodels.api`` and
   ``statsmodels.tsa.api``) and directly importing from the module that defines
   the model.

Data
----

We download the `Guerry dataset
<https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Guerry.html>`_, a
collection of historical data used in support of Andre-Michel Guerry's 1833
*Essay on the Moral Statistics of France*. The data set is hosted online in
comma-separated values format (CSV) by the `Rdatasets
<https://github.com/vincentarelbundock/Rdatasets/>`_ repository.
We could download the file locally and then load it using ``read_csv``, but
``pandas`` takes care of all of this automatically for us:

.. ipython:: python

    df = sm.datasets.get_rdataset("Guerry", "HistData").data

The `Input/Output doc page <iolib.html>`_ shows how to import from various
other formats.

We select the variables of interest and look at the bottom 5 rows:

.. ipython:: python

    vars = ['Department', 'Lottery', 'Literacy', 'Wealth', 'Region']
    df = df[vars]
    df[-5:]

Notice that there is one missing observation in the *Region* column. We
eliminate it using a ``DataFrame`` method provided by ``pandas``:

.. ipython:: python

    df = df.dropna()
    df[-5:]

Substantive motivation and model
--------------------------------

We want to know whether literacy rates in the 86 French departments are
associated with per capita wagers on the Royal Lottery in the 1820s. We need to
control for the level of wealth in each department, and we also want to include
a series of dummy variables on the right-hand side of our regression equation to
control for unobserved heterogeneity due to regional effects. The model is
estimated using ordinary least squares regression (OLS).


Design matrices (*endog* & *exog*)
----------------------------------

To fit most of the models covered by ``statsmodels``, you will need to create
two design matrices. The first is a matrix of endogenous variable(s) (i.e.
dependent, response, regressand, etc.). The second is a matrix of exogenous
variable(s) (i.e. independent, predictor, regressor, etc.). The OLS coefficient
estimates are calculated as usual:

.. math::

    \hat{\beta} = (X'X)^{-1} X'y

where :math:`y` is an :math:`N \times 1` column of data on lottery wagers per
capita (*Lottery*). :math:`X` is :math:`N \times 7` with an intercept, the
*Literacy* and *Wealth* variables, and 4 region binary variables.

The ``patsy`` module provides a convenient function to prepare design matrices
using ``R``-like formulas. You can find more information `here <https://patsy.readthedocs.io/en/latest/>`_.

We use ``patsy``'s ``dmatrices`` function to create design matrices:

.. ipython:: python

    y, X = dmatrices('Lottery ~ Literacy + Wealth + Region', data=df, return_type='dataframe')

The resulting matrices/data frames look like this:

.. ipython:: python

    y[:3]
    X[:3]

Notice that ``dmatrices`` has

* split the categorical *Region* variable into a set of indicator variables.
* added a constant to the exogenous regressors matrix.
* returned ``pandas`` DataFrames instead of simple numpy arrays. This is useful because DataFrames allow ``statsmodels`` to carry-over meta-data (e.g. variable names) when reporting results.

The above behavior can of course be altered. See the `patsy doc pages
<https://patsy.readthedocs.io/en/latest/>`_.

Model fit and summary
---------------------

Fitting a model in ``statsmodels`` typically involves 3 easy steps:

1. Use the model class to describe the model
2. Fit the model using a class method
3. Inspect the results using a summary method

For OLS, this is achieved by:

.. ipython:: python

    mod = sm.OLS(y, X)    # Describe model
    res = mod.fit()       # Fit model
    print(res.summary())   # Summarize model


The ``res`` object has many useful attributes. For example, we can extract
parameter estimates and r-squared by typing:


.. ipython:: python

    res.params
    res.rsquared

Type ``dir(res)`` for a full list of attributes.

For more information and examples, see the `Regression doc page <regression.html>`_

Diagnostics and specification tests
-----------------------------------

``statsmodels`` allows you to conduct a range of useful `regression diagnostics
and specification tests
<stats.html#residual-diagnostics-and-specification-tests>`_.  For instance,
apply the Rainbow test for linearity (the null hypothesis is that the
relationship is properly modelled as linear):

.. ipython:: python

    sm.stats.linear_rainbow(res)

Admittedly, the output produced above is not very verbose, but we know from
reading the `docstring <generated/statsmodels.stats.diagnostic.linear_rainbow.html>`_
(also, ``print(sm.stats.linear_rainbow.__doc__)``) that the
first number is an F-statistic and that the second is the p-value.

``statsmodels`` also provides graphics functions. For example, we can draw a
plot of partial regression for a set of regressors by:

.. ipython:: python

    @savefig gettingstarted_0.png
    sm.graphics.plot_partregress('Lottery', 'Wealth', ['Region', 'Literacy'],
                                 data=df, obs_labels=False)

Documentation
-------------
Documentation can be accessed from an IPython session
using :func:`~statsmodels.tools.web.webdoc`.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   ~statsmodels.tools.web.webdoc

More
----

Congratulations! You're ready to move on to other topics in the
`Table of Contents <index.html#table-of-contents>`_
