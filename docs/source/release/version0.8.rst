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
on the statespace framework, multiple imputation using MICE and many other
enhancements. The codebase also has been updated to be compatible with
recent numpy and pandas releases.

Statsmodels is using now github to store the updated documentation which
is available under
http://www.statsmodels.org/stable for the last release, and
http://www.statsmodels.org/dev/ for the development version.

This is the last release that supports Python 2.6.


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

plus ongoing refactoring and bug fixes in fringes and corner cases


New functionality in statistics
-------------------------------

Contingency Tables #2418 (Kerby Shedden)

Local FDR, multiple testing #2297 (Kerby Shedden)

Mediation Analysis #2352 (Kerby Shedden)


Duration
--------

Survival Function #2614 ?

Cumulative incidence function


Imputatation
------------
new subpackage in `statsmodels.imputation`

MICE #2076  (Frank Cheng GSOC 2014 and Kerby Shedden)


Time Series Analysis
--------------------

KPSS unit root test #2775? (N-Wouda)


Penalized Estimation
--------------------

elastic net: L1/L2 penalization in OLS, GLM and PHReg (Kerby Shedden)


GLM
---

Tweedie is now available as new family #2872 (thequackdaddy with Josef Perktold)

frequency weights have been added (currently without full support)


Multivariate
------------

new subpackage that currently contains PCA

PCA was added in 0.7 to statsmodels.tools and is now in statsmodels.multivariate


Other important new features
----------------------------

* Bullet
* List
* of
* new
* features

Major Bugs fixed
----------------

* Bullet
* list
* use :ghissue:`XXX` to link to issue.

Backwards incompatible changes and deprecations
-----------------------------------------------

* List backwards incompatible changes

Development summary and credits
-------------------------------

A blurb about the number of changes and the contributors list.

.. note::

   Obtained by running ``git log v0.6.0..HEAD --format='* %aN <%aE>' | sed 's/@/\-at\-/' | sed 's/<>//' | sort -u``.

