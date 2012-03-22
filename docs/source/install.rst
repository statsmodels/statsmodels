:orphan:

.. _install:

Installation
------------

Using setuptools
~~~~~~~~~~~~~~~~

To obtain the latest released version of statsmodels using `setuptools <http://pypi.python.org/pypi/setuptools>`__::

    easy_install -U statsmodels

Or follow `this link to our PyPI page <http://pypi.python.org/pypi/statsmodels>`__.

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

While not strictly necessary for 0.4, it is recommended that you will have a C compiler installed to take advantage of Cython code where available. You can follow the instructions below to get a C compiler setup for Windows.

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

You can build 32-bit or 64-bit versions of the code using the Microsoft SDK. Detailed instructions can be found on the Cython wiki `here <http://wiki.cython.org/64BitCythonExtensionsOnWindows>`__. The gist of these instructions follow. You will need to download the free Windows SDK C/C++ compiler from Microsoft. You must use the **Microsoft Windows SDK for Windows 7 and .NET Framework 3.5 SP1** to be comptible with Python 2.6, 2.7, 3.1, and 3.2. Other Python versions are as yet untested. Please report results to the mailing list. The link for the 3.5 version is

`http://www.microsoft.com/downloads/en/details.aspx?familyid=71DEB800-C591-4F97-A900-BEA146E4FAE1&displaylang=en <http://www.microsoft.com/downloads/en/details.aspx?familyid=71DEB800-C591-4F97-A900-BEA146E4FAE1&displaylang=en>`__

Get the ISO file GRMSDKX_EN_DVD.iso for AMD64. After you install this, open the SDK Command Shell (Start -> All Programs -> Microsoft Windows SDK v7.0 -> CMD Shell). CD to the statsmodels directory and type::

    set DISTUTILS_USE_SDK=1

To build a 64-bit application type::

    setenv /x64 /release

To build a 32-bit application type::

    setenv /x86 /release

The prompt should change colors to green. Then proceed as usual to install::

    python setup.py build
    python setup.py install


Dependencies
~~~~~~~~~~~~

* `Python <http://www.python.org>`__ >= 2.5, including Python 3.x 
* `NumPy <http://www.scipy.org/>`__ (>=1.4)
* `SciPy <http://www.scipy.org/>`__ (>=0.7)
* `Pandas <http://pandas.pydata.org/>`__ >= 0.7.1
* `Cython <http://cython.org/>`__ >= 15.1, Still optional but recommended for building from non-source distributions. That is, it will be used when building the source from github and not from a zipped source distribution archive.

.. tested with Python 2.5., 2.6, 2.7 and 3.2
.. (tested with numpy 1.4.1, 1.5.1 and 1.6.0, scipy 0.7.2, 0.8.0, 0.9.0)
.. do we need to tell people about testing?

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

* `Matplotlib <http://matplotlib.sourceforge.net/>`__ is needed for plotting functions and running many of the examples. 
* `Nose <http://www.somethingaboutorange.com/mrl/projects/nose/>`__ is required to run the test suite.
