:orphan:

===========
0.8 Release
===========

Release 0.8.0
=============

See also changes in the unreleased 0.7

Release summary
---------------

The main features of this release are several new time series models based
on the statespace framework, multiple imputation using MICE as well as many
other enhancements. The codebase also has been updated to be compatible with
recent numpy and pandas releases.

Statsmodels is using now github to store the updated documentation which
is available under
http://www.statsmodels.org/stable for the last release, and
http://www.statsmodels.org/dev/ for the development version.

This is the last release that supports Python 2.6.


**Warning**

API stability is not guaranteed for new features, although even in this case
changes will be made in a backwards compatible way if possible. The stability
of a new feature depends on how much time it was already in statsmodels master
and how much usage it has already seen.
If there are specific known problems or limitations, then they are mentioned
in the docstrings.


The following major new features appear in this version.

Statespace Models
-----------------

Building on the statespace framework and models added in 0.7, this release
includes additional models that build on it.
Authored by Chad Fulton largely during GSOC 2015

Kalman Smoother #2434

Postestimation #2566

Diagnostics #2431

Unobserved Components #2432

Multivariate Models - VARMAX, Dynamic Factors #2563

recursive least squares in regression #2830

other

* improved missing data handling #2770, #2809
* ongoing refactoring and bug fixes in fringes and corner cases


New functionality in statistics
-------------------------------

Contingency Tables #2418 (Kerby Shedden)

Local FDR, multiple testing #2297 (Kerby Shedden)

Mediation Analysis #2352 (Kerby Shedden)

other:

* weighted quantiles in DescrStatsW #2707 (Kerby Shedden)


Duration
--------

Kaplan Meier Survival Function #2614 (Kerby Shedden)

Cumulative incidence rate function #3016 (Kerby Shedden)

other:

* frequency weights in Kaplan-Meier #2992 (Kerby Shedden)


Imputation
----------

new subpackage in `statsmodels.imputation`

MICE #2076  (Frank Cheng GSOC 2014 and Kerby Shedden)

Imputation by regression on Order Statistic  #3019 (Paul Hobson)


Time Series Analysis
--------------------

Markov Switching Model #2980 (Chad Fulton)

KPSS stationarity, unit root test #2775 (N-Wouda)

BDS nonlinear dependence test #934 (Chad Fulton)


Penalized Estimation
--------------------

Elastic net: fit_regularized with L1/L2 penalization has been added to
OLS, GLM and PHReg (Kerby Shedden)


GLM
---

Tweedie is now available as new family #2872 (Peter Quackenbush, Josef Perktold)

other:

* frequency weights for GLM (currently without full support) #
* more flexible convergence options #2803 (Peter Quackenbush)


Multivariate
------------

new subpackage that currently contains PCA

PCA was added in 0.7 to statsmodels.tools and is now in statsmodels.multivariate


Documentation
-------------

New doc build with latest jupyter and Python 3 compatibility (Tom Augspurger)


Other important improvements
----------------------------

several existing functions have received improvements


* seasonal_decompose: improved periodicity handling #2987 (ssktotoro ?)
* tools add_constant, add_trend: refactoring and pandas compatibility #2240 (Kevin Sheppard)
* acf, pacf, acovf: option for missing handling #3020 (joesnacks ?)
* acf, pacf plots: allow array of lags #2989 (Kevin Sheppard)
* io SimpleTable (summary): allow names with special characters #3015 (tvanessa ?)
* tsa tools lagmat, lagmat2ds: pandas support #2310 #3042 (Kevin Sheppard)
* CompareMeans: from_data, summary methods #2754 (Valery Tyumen)



Major Bugs fixed
----------------

* see github issues

Backwards incompatible changes and deprecations
-----------------------------------------------

* ???
* predict now returns a pandas Series if the exog argument is a DataFrame
* PCA moved to multivariate compared to 0.7


Development summary and credits
-------------------------------

Besides receiving contributions for new and improved features and for bugfixes,
important contributions to general maintenance came from

* Kevin Sheppard
* Pierre Barbier de Reuille
* Tom Augsburger

and the general maintainer and code reviewer

* Josef Perktold

Additionally, many users contributed by participation in github issues and
providing feedback.

Thanks to all of the contributors for the 0.8 release:

.. note::

   * Ashish
   * Brendan
   * Brendan Condon
   * BrianLondon
   * Chad Fulton
   * Chris Fonnesbeck
   * Christoph T. Weidemann
   * James Kerns
   * Josef Perktold
   * Kerby Shedden
   * Kevin Sheppard
   * Leoyzen
   * Matthew Brett
   * Niels Wouda
   * Paul Hobson
   * Pierre Barbier de Reuille
   * Pietro Battiston
   * Ralf Gommers
   * Roman Ring
   * Skipper Seabold
   * Soren Fuglede Jorgensen
   * Thomas Cokelaer
   * Tom Augspurger
   * ValeryTyumen
   * Vanessa
   * Yaroslav Halchenko
   * joesnacks
   * kokes
   * matiumerca
   * rlan
   * ssktotoro
   * thequackdaddy
   * vegcev

Thanks to all of the contributors for the 0.7 release:

.. note::

   * Alex Griffing
   * Antony Lee
   * Chad Fulton
   * Christoph Deil
   * Daniel Sullivan
   * Hans-Martin von Gaudecker
   * Jan Schulz
   * Joey Stockermans
   * Josef Perktold
   * Kerby Shedden
   * Kevin Sheppard
   * Kiyoto Tamura
   * Louis-Philippe Lemieux Perreault
   * Padarn Wilson
   * Ralf Gommers
   * Saket Choudhary
   * Skipper Seabold
   * Tom Augspurger
   * Trent Hauck
   * Vincent Arel-Bundock
   * chebee7i
   * donbeo
   * gliptak
   * hlin117
   * jerry dumblauskas
   * jonahwilliams
   * kiyoto
   * neilsummers
   * waynenilsen

These lists of names are automatically generated based on git log, and may not be
complete.
