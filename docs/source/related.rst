.. _related:

.. currentmodule:: statsmodels


Related Packages
================

These are some python packages that have a related purpose and can be
useful in combination with statsmodels. The selection in this list is
biased towards packages that might be directly useful for data handling and
statistical analysis, and towards those that have a BSD compatible license,
which implies that we are not restricted in looking at the source to learn
of different ways of implementation or of different algorithms.
The following descriptions are taken from the websites with small adjustments.



Data Handling
-------------

Scikits.timeseries
^^^^^^^^^^^^^^^^^^

http://pypi.python.org/pypi/scikits.timeseries

"Time series manipulation

The scikits.timeseries module provides classes and functions for manipulating,
reporting, and plotting time series of various frequencies. The focus is on
convenient data access and manipulation while leveraging the existing
mathematical functionality in Numpy and SciPy."

Licence: BSD
Language: Python, C, binary distributions available


*Comments*

Timeseries is based on numpys MaskedArray and is designed for handling data
with missing values. It also includes functions for statistical analysis.


Pandas
^^^^^^

http://pypi.python.org/pypi/pandas

"This project aims to provide the following
 * A set of fast NumPy-based data structures optimized for panel, time series,
   and cross-sectional data analysis.
 * A set of tools for loading such data from various sources and providing
   efficient ways to persist the data.
 * A robust statistics and econometrics library which closely integrates with
   the core data structures."

License: New BSD
Language: Python, Cython,
binary distribution available for win32-py25, but easy to build with MinGW

*Comments*

Uses statsmodels as optional dependency for statistical analysis, but has
additional statistical and econometrics algorithms that focus on panel data
analysis, mostly in the time dimension. It has several data structures that
allow dictionary access to the underlying 1, 2, or 3 dimensional arrays. It
was initially focused on a two-dimensional representation of the data, but
now also allows for different representation of three-dimensional arrays. It
allows for arbitrary axis labels, but offers also a convenient time series
class.


Tabular
^^^^^^^

http://pypi.python.org/pypi/tabular

"Tabular data container and associated convenience routines in Python

Tabular is a package of Python modules for working with tabular data. Its main
object is the tabarray class, a data structure for holding and manipulating
tabular data.

The tabarray object is based on the ndarray object from the Numerical Python
package (NumPy), and the Tabular package is built to interface well with NumPy
in general. "

License: MIT
Language: Python

*Comments*

Uses numpys structured arrays as basic building block. Focused on
spreadsheet-style operations for working with two-dimensional tables and
associated data handling and analysis.
It is instructive to read the code of tabular for working with structured
arrays.


La
^^

http://pypi.python.org/pypi/la

"Label the rows, columns, any dimension, of your NumPy arrays.

The main class of the la package is a labeled array, larry. A larry consists of
a data array and a label list. The data array is stored as a NumPy array and
the label list as a list of lists. "

License: BSD
Language: Python

*Comments*

The data handling is in intention similar to pandas but closer to working
with standard numpy ndarrays. The main addition to numpy arrays are
arbitrary labels for each axis of the array. Larry delegates to numpy
functions but does not subclass numpy's ndarrays. It also provides functions
for basic descriptive statistics.




Data Analysis
-------------

Pymc
^^^^

http://pypi.python.org/pypi/pymc

"Bayesian estimation, particularly using Markov chain Monte Carlo (MCMC), is
an increasingly relevant approach to statistical estimation.
PyMC is a python module that implements the Metropolis-Hastings algorithm
as a python class, and is extremely flexible and applicable to a large suite
of problems.""

License: MIT, Academic Free License (?)
Language: Python, C, Fortran
binary (bundle ?) installer

*Comments*
This is to some extent the modern Bayesian analog of statsmodels. It is by
far the most mature project in this group including statsmodels.


Scikits.talkbox
^^^^^^^^^^^^^^^

http://pypi.python.org/pypi/scikits.talkbox

Talkbox is set of python modules for speech/signal processing. The goal of this
toolbox is to be a sandbox for features which may end up in scipy at some
point.

License: BSD
Language: Python, C optional


*Comments*

Although specialized on speech processing, talkbox has some accessible and
useful functions for time series analysis, especially a fast implementation
for estimating AR models (with ...) and spectral density based on estimated
AR coefficients.


Nitime
^^^^^^
http://github.com/fperez/nitime

"Nitime is a library for time-series analysis of data from neuroscience experiments.

It contains a core of numerical algorithms for time-series analysis both in
the time and spectral domains, a set of container objects to represent
time-series, and auxiliary objects that expose a high level interface to the
numerical machinery and make common analysis tasks easy to express with
compact and semantically clear code."

License: BSD
Language: Python

*Comments*
Althoug focused on neuroscience, the algorithms for time series analysis are
independent of the data representation and can be used with numpy arrays.
Current focus is on spectral analysis including coherence between several
time series.


KF - Kalman Filter
^^^^^^^^^^^^^^^^^^

http://pypi.python.org/pypi/KF

"This project was started to test different avaiable tools to track mutual
funds and hedge fund using Capital Asset Pricing Model (CAPM thereafter)
introduced my Sharpe and Arbitrage Pricing Theory (APT thereafter) introduced
by Ross.
"

 * License : BSD -check
 * Language Python (requires cvxopt)


*Comments*
Very young project but with a similar, although narrower, focus as pandas
and (parts of) statsmodels. Uses Kalman Filter for rolling linear regression
and allows for equality and inequality constraints in the estimation.
Includes its own time series class, and the estimation seems (?) to depend on
it.



Domain-specific Data Analysis
-----------------------------

The following packages contain interesting statistical algorithms, however
they are tightly focused on their application, and are or might be more
difficult to use "from the outside". (Descriptions are taken from websites)

Pymvpa
^^^^^^

PyMVPA is a Python module intended to ease pattern classification analyses of
large datasets
http://pymvpa.org/
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

Pysal
^^^^^

A library for exploratory spatial analysis and geocomputation
http://code.google.com/p/pysal/
License: BSD

glu-genetics
^^^^^^^^^^^^

A broad array of tools to store, clean, and analyze data generated by
whole-genome or candidate gene association scans.
http://code.google.com/p/glu-genetics/
License: BSD


Other packages
--------------

There exists a large number of machine learning packages in python, many of
them with a well established code base. Unfortunately, none of the packages
with a wider coverage of algorithms has a scipy compatible license.
A listing can be found at http://mloss.org/software/language/python/
scikits.learn includes several machine learning algorithms and is currently
undergoing a cleanup and enhancement http://pypi.python.org/pypi/scikits.learn/0.1 .

Other packages are available that provide additional functionality,
especially openopt which offers additional optimization routines compared to
the ones in scipy.



