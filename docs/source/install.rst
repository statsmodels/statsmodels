:orphan:

.. _install:

Installation
------------

Using setuptools
~~~~~~~~~~~~~~~~

To obtain the latest released version of statsmodels using `setuptools <https://pypi.python.org/pypi/setuptools>`__::

    pip install -U statsmodels

Or follow `this link to our PyPI page <https://pypi.python.org/pypi/statsmodels>`__.

Statsmodels is also available in `Anaconda <https://www.continuum.io/downloads>`__::

    conda install statsmodels

Obtaining the Source
~~~~~~~~~~~~~~~~~~~~

We do not release very often but the master branch of our source code is 
usually fine for everyday use. You can get the latest source from our 
`github repository <https://github.com/statsmodels/statsmodels>`__. Or if you have git installed::

    git clone git://github.com/statsmodels/statsmodels.git

If you want to keep up to date with the source on github just periodically do::

    git pull

in the statsmodels directory.

Windows Nightly Binaries
~~~~~~~~~~~~~~~~~~~~~~~~

If you are not able to follow the build instructions below, we upload nightly builds of the GitHub repository to `http://www.statsmodels.org/binaries/ <http://www.statsmodels.org/binaries/>`__.

Installation from Source
~~~~~~~~~~~~~~~~~~~~~~~~

You will need a C compiler installed to build statsmodels. If you are building from the github source and not a source release, then you will also need Cython. You can follow the instructions below to get a C compiler setup for Windows.

Linux
^^^^^

Once you have obtained the source, you can do (with appropriate permissions)::

    python setup.py install

Or::

    python setup.py build
    python setup.py install

Windows
^^^^^^^

You can build 32-bit version of the code on windows using mingw32.

First, get and install `mingw32 <http://www.mingw.org/>`__. Then, you'll need to edit distutils.cfg. This is usually found somewhere like C:\Python27\Lib\distutils\distutils.cfg. Add these lines::

    [build]
    compiler=mingw32

Then in the statsmodels directory do::

    python setup.py build
    python setup.py install

OR

You can build 32-bit or 64-bit versions of the code using the Microsoft SDK. Detailed instructions can be found on the Cython wiki `here <http://wiki.cython.org/64BitCythonExtensionsOnWindows>`__. The gist of these instructions follow. You will need to download the free Windows SDK C/C++ compiler from Microsoft. You must use the **Microsoft Windows SDK for Windows 7 and .NET Framework 3.5 SP1** to be comptible with Python 2.6, 2.7, 3.1, and 3.2. The link for the 3.5 SP1 version is

`http://www.microsoft.com/downloads/en/details.aspx?familyid=71DEB800-C591-4F97-A900-BEA146E4FAE1&displaylang=en <http://www.microsoft.com/downloads/en/details.aspx?familyid=71DEB800-C591-4F97-A900-BEA146E4FAE1&displaylang=en>`__

For Python 3.3, you need to use the **Microsoft Windows SDK for Windows 7 and .NET Framework 4**, available from

`http://www.microsoft.com/en-us/download/details.aspx?id=8279 <http://www.microsoft.com/en-us/download/details.aspx?id=8279>`__

For 7.0, get the ISO file GRMSDKX_EN_DVD.iso for AMD64. After you install this, open the SDK Command Shell (Start -> All Programs -> Microsoft Windows SDK v7.0 -> CMD Shell). CD to the statsmodels directory and type::

    set DISTUTILS_USE_SDK=1

To build a 64-bit application type::

    setenv /x64 /release

To build a 32-bit application type::

    setenv /x86 /release

The prompt should change colors to green. Then proceed as usual to install::

    python setup.py build
    python setup.py install

For 7.1, the instructions are exactly the same, except you use the download link provided above and make sure you are using SDK 7.1.

If you want to accomplish the same without opening up the SDK CMD SHELL, then you can use these commands at the CMD Prompt or in a batch file.::

    setlocal EnableDelayedExpansion
    CALL "C:\Program Files\Microsoft SDKs\Windows\v7.0\Bin\SetEnv.cmd" /x64 /release
    set DISTUTILS_USE_SDK=1

Replace `/x64` with `/x86` and `v7.0` with `v7.1` as needed.


Dependencies
~~~~~~~~~~~~

* `Python <https://www.python.org>`__ >= 2.6, including Python 3.x 
* `NumPy <http://www.scipy.org/>`__ >= 1.5.1
* `SciPy <http://www.scipy.org/>`__ >= 0.9.0
* `Pandas <http://pandas.pydata.org/>`__ >= 0.7.1
* `Patsy <https://patsy.readthedocs.org>`__ >= 0.3.0
* `Cython <http://cython.org/>`__ >= 20.1, Needed if you want to build the code from github and not a source distribution. You must use Cython >= 0.20.1 if you're on Python 3.4. Earlier versions may work for Python < 3.4.

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

* `Matplotlib <http://matplotlib.org/>`__ >= 1.1 is needed for plotting functions and running many of the examples. 
* If installed, `X-12-ARIMA <http://www.census.gov/srd/www/x13as/>`__ or `X-13ARIMA-SEATS <http://www.census.gov/srd/www/x13as/>`__ can be used for time-series analysis.
* `Nose <https://nose.readthedocs.org/en/latest>`__ is required to run the test suite.
* `IPython <http://ipython.org>`__ >= 1.0 is required to build the docs locally.
