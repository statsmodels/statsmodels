.. currentmodule:: scikits.statsmodels

************
Introduction
************

About
=====

The :mod:`scikits.statsmodels` module provides classes and functions for
manipulating, reporting, and plotting time series of various frequencies.
The focus is on convenient data access and manipulation while leveraging
the existing mathematical functionality in numpy and scipy.

If the following scenarios sound familiar to you, then you will likely find
the :mod:`scikits.statsmodels` module useful:

* Compare many time series with different ranges of data (eg. stock prices);
* Create time series plots with intelligently spaced axis labels;
* Convert a daily time series to monthly by taking the average value during
  each month;
* Work with data that has missing values;
* Determine the last business day of the previous month/quarter/year for
  reporting purposes;
* Compute a moving standard deviation *efficiently*.

These are just some of the scenarios that are made very simple with the
:mod:`scikits.statsmodels` module.

History
=======

The :mod:`scikits.statsmodels` module was originally developed by Matt Knox to
manipulate financial and economic data of weekday (Monday-Friday), monthly, and
quarterly frequencies and to compare data series of differing frequencies. Matt
created a large number of frequency conversion algorithms (implemented in C for
extra speed) for reshaping the series. The initial version was released winter
2006 as a module in the (now defunct) :mod:`SciPy` sandbox.

Pierre Gerard-Marchant rewrote the original prototype late December 2006 and
adapted it to be based on the :class:`numpy.ma.MaskedArray` class for handling
missing data in order to work with environmental time series.
