.. statsmodels documentation master file, created by
   sphinx-quickstart on Sat Aug 22 00:38:34 2009.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to statsmodels's documentation!
=======================================

:mod:`scikits.statsmodels` is a pure python package that provides classes and
functions for the estimation of several categories of statistical models. These
currently include linear regression models, OLS, GLS, WLS and GLS with AR(p)
errors, generalized linear models for six distribution families and
M-estimators for robust linear models. An extensive list of result statistics
are avalable for each estimation problem

Quickstart for the impatient
----------------------------

**License:** BSD

**Requirements:** python 2.4. to 2.6 and latest releases of numpy and scipy

**Repository:** http://code.launchpad.net/statsmodels

**Online Documentation:** http://statsmodels.sourceforge.net/


**Installation:**

::

  easy_install scikits.statsmodels

or get the source from pypi, sourceforge, or from the launchpad repository and

::

  setup.py install

**Usage:**

Get the data, run the estimation, and look at the results.
For example, here is a minimal ordinary least squares case ::

  import numpy as np
  import scikits.statsmodels as sm

  # get data
  nsample = 100
  x = np.linspace(0,10, 100)
  X = sm.tools.add_constant(np.column_stack((x, x**2)))
  beta = np.array([1, 0.1, 10])
  y = np.dot(X, beta) + np.random.normal(size=nsample)

  # run the regression
  results = sm.OLS(y, X).fit()

  # look at the results
  print results.summary()

  and look at `dir(results)` to see some of the results
  that are available

**Note:**
Due to our infrequent official releases, it should be noted that the trunk
branch in the launchpad repository given above will have the most recent
code and is for the most part quite stable and fine for daily use.


Table of Content
----------------

.. toctree::
   :maxdepth: 4


   introduction
   related
   regression
   glm
   rlm
   stattools
   tools
   internal
   sandbox

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

