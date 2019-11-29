:orphan:

.. _install:

Installing statsmodels
======================

The easiest way to install statsmodels is to install it as part of the `Anaconda <https://docs.continuum.io/anaconda/>`_
distribution, a cross-platform distribution for data analysis and scientific
computing. This is the recommended installation method for most users.

Instructions for installing from PyPI, source or a development version are also provided.


Python Support
--------------

statsmodels supports Python 3.5, 3.6 and 3.7.

Anaconda
--------
statsmodels is available through conda provided by
`Anaconda <https://www.continuum.io/downloads>`__. The latest release can
be installed using:

.. code-block:: bash

   conda install -c conda-forge statsmodels

PyPI (pip)
----------

To obtain the latest released version of statsmodels using pip:

.. code-block:: bash

    pip install statsmodels

Follow `this link to our PyPI page <https://pypi.org/project/statsmodels/>`__ to directly
download wheels or source.

For Windows users, unofficial recent binaries (wheels) are occasionally
available `here <https://www.lfd.uci.edu/~gohlke/pythonlibs/#statsmodels>`__.

Obtaining the Source
--------------------

We do not release very often but the master branch of our source code is
usually fine for everyday use. You can get the latest source from our
`github repository <https://github.com/statsmodels/statsmodels>`__. Or if you
have git installed:

.. code-block:: bash

    git clone git://github.com/statsmodels/statsmodels.git

If you want to keep up to date with the source on github just periodically do:

.. code-block:: bash

    git pull

in the statsmodels directory.

Installation from Source
------------------------

You will need a C compiler installed to build statsmodels. If you are building
from the github source and not a source release, then you will also need
Cython. You can follow the instructions below to get a C compiler setup for
Windows.

If your system is already set up with pip, a compiler, and git, you can try:

.. code-block:: bash

    pip install git+https://github.com/statsmodels/statsmodels

If you do not have pip installed or want to do the installation more manually,
you can also type:

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

Compilers
~~~~~~~~~

Linux
^^^^^

If you are using Linux, we assume that you are savvy enough to install `gcc` on
your own. More than likely, it is already installed.

Windows
^^^^^^^

It is strongly recommended to use 64-bit Python if possible.

Getting the right compiler is especially confusing for Windows users. Over time,
Python has been built using a variety of different Windows C compilers.
`This guide <https://wiki.python.org/moin/WindowsCompilers>`_ should help
clarify which version of Python uses which compiler by default.

Mac
^^^

Installing statsmodels on MacOS requires installing `gcc` which provides
a suitable C compiler. We recommend installing Xcode and the Command Line
Tools.

Dependencies
------------

The current minimum dependencies are:

* `Python <https://www.python.org>`__ >= 3.5
* `NumPy <https://www.scipy.org/>`__ >= 1.14
* `SciPy <https://www.scipy.org/>`__ >= 1.0
* `Pandas <https://pandas.pydata.org/>`__ >= 0.21
* `Patsy <https://patsy.readthedocs.io/en/latest/>`__ >= 0.5.1

Cython is required to build from a git checkout but not to run or install from PyPI:

* `Cython <https://cython.org/>`__ >= 0.29 is required to build the code from
  github but not from a source distribution.

Given the long release cycle, statsmodels follows a loose time-based policy for
dependencies: minimal dependencies are lagged about one and a half to two
years. Our next planned update of minimum versions is expected in the first
half of 2020.

Optional Dependencies
---------------------

* `cvxopt <https://cvxopt.org/>`__ is required for regularized fitting of
  some models.
* `Matplotlib <https://matplotlib.org/>`__ >= 2.2 is needed for plotting
  functions and running many of the examples.
* If installed, `X-12-ARIMA <https://www.census.gov/srd/www/x13as/>`__ or
  `X-13ARIMA-SEATS <https://www.census.gov/srd/www/x13as/>`__ can be used
  for time-series analysis.
* `pytest <https://docs.pytest.org/en/latest/>`__ is required to run
  the test suite.
* `IPython <https://ipython.org>`__ >= 5.0 is required to build the
  docs locally or to use the notebooks.
* `joblib <http://pythonhosted.org/joblib/>`__ >= 0.9 can be used to accelerate distributed
  estimation for certain models.
* `jupyter <https://jupyter.org/>`__ is needed to run the notebooks.
