:orphan:

.. _install:

Installation
============

Pre-packaged binaries
---------------------

To obtain the latest released version of statsmodels using pip::

    pip install -U statsmodels

Or follow `this link to our PyPI page <https://pypi.python.org/pypi/statsmodels>`__, download
the wheel or source and install.

Statsmodels is also available in through conda provided by
`Anaconda <https://www.continuum.io/downloads>`__. The latest release can
be installed using::

    conda install -c conda-forge statsmodels

For Windows users, unofficial recent binaries (wheels) are occasionally
available `here <https://www.lfd.uci.edu/~gohlke/pythonlibs/#statsmodels>`__.

Obtaining the Source
--------------------

We do not release very often but the master branch of our source code is
usually fine for everyday use. You can get the latest source from our
`github repository <https://github.com/statsmodels/statsmodels>`__. Or if you
have git installed::

    git clone git://github.com/statsmodels/statsmodels.git

If you want to keep up to date with the source on github just periodically do::

    git pull

in the statsmodels directory.

Installation from Source
------------------------

You will need a C compiler installed to build statsmodels. If you are building
from the github source and not a source release, then you will also need
Cython. You can follow the instructions below to get a C compiler setup for
Windows.

If your system is already set up with pip, a compiler, and git, you can try::

    pip install git+https://github.com/statsmodels/statsmodels

If you do not have pip installed or want to do the installation more manually,
you can also type::

.. code-block:: bash

    python setup.py install

Or even more manually

.. code-block:: bash

    python setup.py build
    python setup.py install

statsmodels can also be installed in `develop` mode which installs statsmodels
into the current python environment in-place. The advantage of this is that
edited modules will immediately be re-interpreted when the python interpreter
restarts without having to re-install statsmodels.

.. code-block:: bash

    python setup.py develop

Linux
^^^^^

If you are using Linux, we assume that you are savvy enough to install `gcc` on
your own. More than likely, its already installed.

Windows
^^^^^^^

It is strongly recommended to use 64-bit Python if possible.

Getting the right compiler is especially confusing for Windows users. Over time,
Python has been built using a variety of different Windows C compilers.
`This guide <https://wiki.python.org/moin/WindowsCompilers>`_ should help
clarify which version of Python uses which compiler by default.

Mac
^^^

Installing statsmodels on MacOS will requires installing `gcc` which provides
a suitable C compiler. We recommend installing Xcode and the Command Line
Tools.

Dependencies
------------

The current minimum dependencies are:

* `Python <https://www.python.org>`__ >= 2.7, including Python 3.4+
* `NumPy <http://www.scipy.org/>`__ >= 1.11
* `SciPy <http://www.scipy.org/>`__ >= 0.18
* `Pandas <http://pandas.pydata.org/>`__ >= 0.19
* `Patsy <https://patsy.readthedocs.io/en/latest/>`__ >= 0.4.0
* `Cython <http://cython.org/>`__ >= 0.24 is required to build the code from
  github but not from a source distribution.

Given the long release cycle, Statsmodels follows a loose time-based policy for
dependencies: minimal dependencies are lagged about one and a half to two
years. Our next planned update of minimum versions in `setup.py` is expected in
September 2018, when we will update to reflect Numpy >= 1.12 (released January
2017), Scipy >= 0.19 (released March 2017) and Pandas >= 0.20 (released May
2017).

Optional Dependencies
---------------------

* `Matplotlib <http://matplotlib.org/>`__ >= 1.5 is needed for plotting
  functions and running many of the examples.
* If installed, `X-12-ARIMA <http://www.census.gov/srd/www/x13as/>`__ or
  `X-13ARIMA-SEATS <http://www.census.gov/srd/www/x13as/>`__ can be used
  for time-series analysis.
* `pytest <https://docs.pytest.org/en/latest/>`__ is required to run
  the test suite.
* `IPython <http://ipython.org>`__ >= 3.0 is required to build the
  docs locally or to use the notebooks.
* `joblib <http://pythonhosted.org/joblib/>`__ >= 0.9 can be used to accelerate distributed
  estimation for certain models.
* `jupyter <http://jupyter.org/>`__ is needed to run the notebooks.
