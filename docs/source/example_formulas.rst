.. _formula_examples:

Fitting models using R-style formulas
=====================================

Since version 0.5.0, ``statsmodels`` allows users to fit statistical
models using R-style formulas. Internally, ``statsmodels`` uses the
`patsy <https://patsy.readthedocs.io/en/latest/>`_ package to convert formulas and
data to the matrices that are used in model fitting. The formula
framework is quite powerful; this tutorial only scratches the surface. A
full description of the formula language can be found in the ``patsy``
docs:

-  `Patsy formula language description <https://patsy.readthedocs.io/en/latest/>`_

Loading modules and functions
-----------------------------

.. ipython:: python

    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import numpy as np
    import pandas

Notice that we called ``statsmodels.formula.api`` in addition to the usual
``statsmodels.api``. In fact, ``statsmodels.api`` is used here only to load
the dataset. The ``formula.api`` hosts many of the same
functions found in ``api`` (e.g. OLS, GLM), but it also holds lower case
counterparts for most of these models. In general, lower case models
accept ``formula`` and ``df`` arguments, whereas upper case ones take
``endog`` and ``exog`` design matrices. ``formula`` accepts a string
which describes the model in terms of a ``patsy`` formula. ``df`` takes
a `pandas <https://pandas.pydata.org/>`_ data frame.

``dir(smf)`` will print a list of available models.

Formula-compatible models have the following generic call signature:
``(formula, data, subset=None, *args, **kwargs)``

OLS regression using formulas
-----------------------------

To begin, we fit the linear model described on the `Getting
Started <gettingstarted.html>`_ page. Download the data, subset columns,
and list-wise delete to remove missing observations:

.. ipython:: python

    df = sm.datasets.get_rdataset("Guerry", "HistData").data
    df = df[['Lottery', 'Literacy', 'Wealth', 'Region']].dropna()
    df.head()

Fit the model:

.. ipython:: python

    mod = smf.ols(formula='Lottery ~ Literacy + Wealth + Region', data=df)
    res = mod.fit()
    print(res.summary())

Categorical variables
---------------------

Looking at the summary printed above, notice that ``patsy`` determined
that elements of *Region* were text strings, so it treated *Region* as a
categorical variable. ``patsy``'s default is also to include an
intercept, so we automatically dropped one of the *Region* categories.

If *Region* had been an integer variable that we wanted to treat
explicitly as categorical, we could have done so by using the ``C()``
operator:

.. ipython:: python

    res = smf.ols(formula='Lottery ~ Literacy + Wealth + C(Region)', data=df).fit()
    print(res.params)


Examples more advanced features ``patsy``'s categorical variables
function can be found here: `Patsy: Contrast Coding Systems for
categorical variables <contrasts.html>`_

Operators
---------

We have already seen that "~" separates the left-hand side of the model
from the right-hand side, and that "+" adds new columns to the design
matrix.

Removing variables
~~~~~~~~~~~~~~~~~~

The "-" sign can be used to remove columns/variables. For instance, we
can remove the intercept from a model by:

.. ipython:: python

    res = smf.ols(formula='Lottery ~ Literacy + Wealth + C(Region) -1 ', data=df).fit()
    print(res.params)


Multiplicative interactions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

":" adds a new column to the design matrix with the product of the other
two columns. "\*" will also include the individual columns that were
multiplied together:

.. ipython:: python

    res1 = smf.ols(formula='Lottery ~ Literacy : Wealth - 1', data=df).fit()
    res2 = smf.ols(formula='Lottery ~ Literacy * Wealth - 1', data=df).fit()
    print(res1.params)
    print(res2.params)


Many other things are possible with operators. Please consult the `patsy
docs <https://patsy.readthedocs.io/en/latest/formulas.html>`_ to learn
more.

Functions
---------

You can apply vectorized functions to the variables in your model:

.. ipython:: python

    res = smf.ols(formula='Lottery ~ np.log(Literacy)', data=df).fit()
    print(res.params)


Define a custom function:

.. ipython:: python

    def log_plus_1(x):
        return np.log(x) + 1.0

    res = smf.ols(formula='Lottery ~ log_plus_1(Literacy)', data=df).fit()
    print(res.params)

.. _patsy-namespaces:

Namespaces
----------

Notice that all of the above examples use the calling namespace to look for the functions to apply. The namespace used can be controlled via the ``eval_env`` keyword. For example, you may want to give a custom namespace using the :class:`patsy:patsy.EvalEnvironment` or you may want to use a "clean" namespace, which we provide by passing ``eval_func=-1``. The default is to use the caller's namespace. This can have (un)expected consequences, if, for example, someone has a variable names ``C`` in the user namespace or in their data structure passed to ``patsy``, and ``C`` is used in the formula to handle a categorical variable. See the `Patsy API Reference <https://patsy.readthedocs.io/en/latest/API-reference.html>`_ for more information.

Using formulas with models that do not (yet) support them
---------------------------------------------------------

Even if a given ``statsmodels`` function does not support formulas, you
can still use ``patsy``'s formula language to produce design matrices.
Those matrices can then be fed to the fitting function as ``endog`` and
``exog`` arguments.

To generate ``numpy`` arrays:

.. ipython:: python

    import patsy
    f = 'Lottery ~ Literacy * Wealth'
    y, X = patsy.dmatrices(f, df, return_type='matrix')
    print(y[:5])
    print(X[:5])

``y`` and ``X`` would be instances of ``patsy.DesignMatrix`` which is a subclass of ``numpy.ndarray``.

To generate pandas data frames:

.. ipython:: python

    f = 'Lottery ~ Literacy * Wealth'
    y, X = patsy.dmatrices(f, df, return_type='dataframe')
    print(y[:5])
    print(X[:5])

.. ipython:: python

    print(sm.OLS(y, X).fit().summary())
