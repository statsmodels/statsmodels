.. :tocdepth: 2

Welcome to Statsmodels's Documentation
======================================

:mod:`scikits.statsmodels` is a Python module that provides classes and functions for the estimation 
of many different statistical models, as well as for conducting statistical tests, and statistical 
data exploration. An extensive list of result statistics are avalable for each estimator. 
The results are tested against existing statistical packages to ensure that they are correct. The 
pacakge is released under the open source Simplied BSD (2-clause) license. The online documentation
is hosted at `sourceforge <http://statsmodels.sourceforge.net/>`__. 


You can contanct us on our 
`mailing list <http://groups.google.com/group/pystatsmodels?hl=en>`__.

Installation
------------

Using setuptools
~~~~~~~~~~~~~~~~

To obtain the latest released version of statsmodels using `setuptools <http://pypi.python.org/pypi/setuptools>__`::

    easy_install scikits.statsmodels

Or follow `this link to our PyPI page <http://pypi.python.org/pypi/scikits.statsmodels>`__.

Obtaining the Source
~~~~~~~~~~~~~~~~~~~~

We do not release very often but the master branch of our source code is 
usually fine for everyday use. You can get the latest source from our 
`github repository <https://www.github.com/statsmodels/statsmodels>`__. Or if you have git installed::

    git clone git://github.com/statsmodels/statsmodels.git

If you want to keep up to date with the source on github just periodically do::

    git pull

in the statsmodels directory.

Installation from Source
~~~~~~~~~~~~~~~~~~~~~~~~

Once you have obtained the source, you can do (with appropriate permissions)::

    python setup.py install

For the 0.3 release, you might want to do::

    python setup.py build --with-cython
    python setup.py install

To enable the Cython-based Kalman filter used by the ARMA model. You will need a C compiler.

Dependencies
~~~~~~~~~~~~

* `Python <www.python.org>`__ >= 2.5, including Python 3.x 
* `NumPy <http://www.scipy.org/>`__ (>=1.4) and `SciPy <http://www.scipy.org/>`__ (>=0.7)

.. tested with Python 2.5., 2.6, 2.7 and 3.2
.. (tested with numpy 1.4.1, 1.5.1 and 1.6.0, scipy 0.7.2, 0.8.0, 0.9.0)
.. do we need to tell people about testing?

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

* `Matplotlib <http://matplotlib.sourceforge.net/>`__ is needed for plotting functions and running many of the examples. 
* `Nose <http://www.somethingaboutorange.com/mrl/projects/nose/>`__ is required to run the test suite.

Getting Involved
----------------

* Join us on our `Mailing List <http://groups.google.com/group/pystatsmodels?hl=en>`__ 
  to ask user questions or discuss development.
* Report bugs on our `Bug Tracker <https://bugs.launchpad.net/statsmodels>`__.
* Found a bug and want to contribute a patch? Want to contribute a new model 
  or some statistical tests? What to get involved in code development? 
  Have a look out our :doc:`dev/index`.

Getting Started
---------------

Get the data, run the estimation, and look at the results.
For example, here is a minimal ordinary least squares example

.. code-block:: python

  import numpy as np
  import scikits.statsmodels.api as sm

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

  and look at `dir(results)` to see some of the results
  that are available


Table of Contents
-----------------

.. toctree::
   :maxdepth: 1

   introduction
   related
   dev/index

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
   datasets
   sandbox

Related Projects
----------------

See our :ref:`related projects page <related>`.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
