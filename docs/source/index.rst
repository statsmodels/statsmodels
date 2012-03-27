.. :tocdepth: 2

Welcome to Statsmodels's Documentation
======================================

:mod:`statsmodels` is a Python module that provides classes and functions for the estimation 
of many different statistical models, as well as for conducting statistical tests, and statistical
data exploration. An extensive list of result statistics are avalable for each estimator.
The results are tested against existing statistical packages to ensure that they are correct. The
package is released under the open source Simplied BSD (3-clause) license. The online documentation
is hosted at `sourceforge <http://statsmodels.sourceforge.net/>`__.

Google Summer of Code 2012
--------------------------

Statsmodels is participating for the fourth time in `GSoC <http://www.google-melange.com/gsoc/homepage/google/gsoc2012>`__  under the umbrella of the `Python Software Foundation <http://python.org/psf/>`__. We have set up a `wiki page <https://github.com/statsmodels/statsmodels/wiki/GSoC-Ideas>`__ with ideas for projects. Feel free to contribute to the ideas page or contact the mailing list if you are interested in applying so we can coordinate on developing your application and project - the earlier the better.

Getting Started
---------------

Get the data, run the estimation, and look at the results.
For example, here is a minimal ordinary least squares example

.. code-block:: python

  import numpy as np
  import statsmodels.api as sm

  # get data
  nsample = 100
  x = np.linspace(0,10, 100)
  X = sm.add_constant(np.column_stack((x, x**2)))
  beta = np.array([1, 0.1, 10])
  y = np.dot(X, beta) + np.random.normal(size=nsample)

  # run the regression
  results = sm.OLS(y, X).fit()

  # look at the results
  print results.summary()

Have a look at `dir(results)` to see available results. Attributes are 
described in `results.__doc__` and results methods have their own docstrings.


Table of Contents
-----------------

.. toctree::
   :maxdepth: 1

   introduction
   related
   dev/index
   pitfalls

.. toctree::
   :maxdepth: 2

   regression
   glm
   rlm
   discretemod
   tsa
   stats
   tools
   miscmodels
   dev/internal
   gmm
   distributions
   graphics
   datasets/index
   sandbox

Related Projects
----------------

See our :ref:`related projects page <related>`.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
