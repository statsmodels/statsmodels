.. :tocdepth: 2

Welcome to Statsmodels's Documentation
======================================

:mod:`statsmodels` is a Python module that provides classes and functions for the estimation
of many different statistical models, as well as for conducting statistical tests, and statistical
data exploration. An extensive list of result statistics are avalable for each estimator.
The results are tested against existing statistical packages to ensure that they are correct. The
package is released under the open source Modified BSD (3-clause) license.
The online documentation is hosted at `sourceforge <http://statsmodels.sourceforge.net/>`__.


Minimal Examples
----------------

Since version ``0.5.0`` of ``statsmodels``, you can use R-style formulas
together with ``pandas`` data frames to fit your models. Here is a simple
example using ordinary least squares:   

.. code-block:: python

    import numpy as np
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Load data
    dat = sm.datasets.get_rdataset("Guerry", "HistData").data

    # Fit regression model (using the natural log of one of the regressors)
    results = smf.ols('Lottery ~ Literacy + np.log(Pop1831)', data=dat).fit()

    # Inspect the results
    print results.summary()

You can also use ``numpy`` arrays instead of formulas:

.. code-block:: python

    import numpy as np
    import statsmodels.api as sm

    # Generate artificial data (2 regressors + constant)
    nobs = 100
    X = np.random.random((nobs, 2)) 
    X = sm.add_constant(X)
    beta = [1, .1, .5]
    e = np.random.random(nobs)
    y = np.dot(X, beta) + e 

    # Fit regression model
    results = sm.OLS(y, X).fit()

    # Inspect the results
    print results.summary()

Have a look at `dir(results)` to see available results. Attributes are
described in `results.__doc__` and results methods have their own docstrings.

Basic Documentation
-------------------

.. toctree::
    :maxdepth: 3

    release/index
    gettingstarted
    example_formulas
    install
    about

Information about the structure and development of
statsmodels:

.. toctree::
   :maxdepth: 1

   endog_exog
   importpaths
   pitfalls
   dev/index
   dev/internal

Table of Contents
-----------------

.. toctree::
   :maxdepth: 3

   regression
   glm
   gee
   rlm
   mixed_linear
   discretemod
   anova
   tsa
   statespace
   duration
   stats
   nonparametric
   gmm
   emplike
   miscmodels
   distributions
   graphics
   iolib
   tools
   datasets/index
   sandbox


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

