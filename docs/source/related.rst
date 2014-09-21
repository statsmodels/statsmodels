.. _related:

.. currentmodule:: statsmodels


Related Packages
================

These are some python packages that have a related purpose and can be
useful in combination with statsmodels. The selection in this list is
biased towards packages that might be directly useful for data handling and statistical analysis, and towards those that have a BSD compatible license, which implies that we are not restricted in looking at the source to learn of different ways of implementation or of different algorithms. The following descriptions are taken from the websites with small adjustments.


Data Handling
-------------

Pandas
^^^^^^

PyPI: https://pypi.python.org/pypi/pandas
Documentation: http://pandas.pydata.org/

This project provides high-performance, easy-to-use data structures and data analysis tools. 

License: New BSD
Language: Python, Cython

*Comments*

Uses statsmodels as optional dependency for statistical analysis, but has
additional statistical and econometrics algorithms that focus on panel data analysis, mostly in the time dimension. It has several data structures that allow dictionary access to the underlying 1, 2, or 3 dimensional arrays. It was initially focused on a two-dimensional representation of the data, but now also allows for different representation of three-dimensional arrays. It allows for arbitrary axis labels, but also offers a convenient time series class.


Tabular
^^^^^^^

PyPI: https://pypi.python.org/pypi/tabular
Documentation: http://web.mit.edu/yamins/www/tabular/

Tabular is a package of Python modules for working with tabular data. Its main object is the tabarray class, a data structure for holding and manipulating tabular data.

The tabarray object is based on the ndarray object from the Numerical Python package (NumPy), and the Tabular package is built to interface well with NumPy in general.

License: MIT
Language: Python

*Comments*

Uses numpy's structured arrays as basic building block. Focused on spreadsheet-style operations for working with two-dimensional tables and associated data handling and analysis. It is instructive to read the code of tabular for working with structured arrays.


Data Analysis
-------------

PyMC
^^^^

PyPI: https://pypi.python.org/pypi/pymc
Documentation: https://pymc-devs.github.io/pymc/

"Bayesian estimation, particularly using Markov chain Monte Carlo (MCMC), is an increasingly relevant approach to statistical estimation. PyMC is a python module that implements the Metropolis-Hastings algorithm as a python class, and is extremely flexible and applicable to a large suite of problems."

License: Apache, Academic Free License
Language: Python, C, Fortran

*Comments*
This is to some extent the modern Bayesian analog of statsmodels. It is by
far the most mature project in this group including statsmodels.

Work is ongoing on PyMC 3, which provides a significant rewrite of the package. See the `source on github <https://github.com/pymc-devs/pymc>`_ for more information.


Nitime
^^^^^^

PyPI: https://pypi.python.org/pypi/nitime
Documentation: http://nipy.org/nitime/

Nitime is a library for time-series analysis of data from neuroscience experiments.

It contains a core of numerical algorithms for time-series analysis both in the time and spectral domains, a set of container objects to represent
time-series, and auxiliary objects that expose a high level interface to the numerical machinery and make common analysis tasks easy to express with
compact and semantically clear code.

License: BSD
Language: Python

*Comments*
Although focused on neuroscience, the algorithms for time series analysis are independent of the data representation and can be used with numpy arrays. Current focus is on spectral analysis including coherence between several time series.


scikit-learn
^^^^^^^^^^^^

PyPI: https://pypi.python.org/pypi/scikit-learn
Documentation: http://scikit-learn.org/stable/

Machine learning in Python.

Simple and efficient tools for data mining and data analysis. Accessible to everybody, and reusable in various contexts. Built on NumPy, SciPy, and 
matplotlib.

License: BSD
Language: Python, Cython


Domain-specific Data Analysis
-----------------------------

The following packages contain interesting statistical algorithms, however
they are tightly focused on their application, and are or might be more
difficult to use "from the outside". (Descriptions are taken from websites)

Pymvpa
^^^^^^

PyMVPA is a Python module intended to ease pattern classification analyses of large datasets
http://www.pymvpa.org/
License: MIT

Nipy
^^^^

Nipy aims to provide a complete Python environment for the analysis of
structural and functional neuroimaging data
http://nipy.sourceforge.net/
License: BSD

Biopython
^^^^^^^^^

Biopython is a set of tools for biological computation
http://biopython.org/wiki/Main_Page
License: http://www.biopython.org/DIST/LICENSE   similar to MIT (?))

PySAL
^^^^^

A library for exploratory spatial analysis and geocomputation
http://code.google.com/p/pysal/
License: BSD


Other packages
--------------

There exist a large number of machine learning packages in python, many of them with a well established code base. Unfortunately, none of the packages with a wider coverage of algorithms has a scipy compatible license. A listing can be found at http://mloss.org/software/language/python/. Other packages are available that provide additional functionality, especially openopt which offers additional optimization routines compared to the ones in scipy.
