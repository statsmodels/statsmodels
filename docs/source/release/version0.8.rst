:orphan:

===========
0.8 Release
===========

Release 0.8.0
=============

See also changes in the unrelease 0.7

Release summary

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

Authored by Kerby Shedden

Contingency Tables #2418

Local FDR, multiple testing #2297

Mediation Analysis #2352
Survival Function #2614 ?
MICE #2076 with Frank Cheng GSOC 2014



Time Series Analysis
--------------------

KPSS unit root test (N-Wouda) #2775?


Penalized Estimation
--------------------

elstic net



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

