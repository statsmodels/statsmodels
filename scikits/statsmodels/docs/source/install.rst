.. _install:

Installation
------------

Using setuptools
~~~~~~~~~~~~~~~~

To obtain the latest released version of statsmodels using `setuptools <http://pypi.python.org/pypi/setuptools>`__::

    easy_install -U scikits.statsmodels

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
