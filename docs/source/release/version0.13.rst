:orphan:

==============
Release 0.13.0
==============

Release summary
===============

statsmodels is using github to store the updated documentation. Two version are available:

- `Stable <https://www.statsmodels.org/>`_, the latest release
- `Development <https://www.statsmodels.org/devel/>`_, the latest build of the main branch

**Warning**

API stability is not guaranteed for new features, although even in
this case changes will be made in a backwards compatible way if
possible. The stability of a new feature depends on how much time it
was already in statsmodels main and how much usage it has already
seen.  If there are specific known problems or limitations, then they
are mentioned in the docstrings.

Stats
-----
**Issues Closed**: TBD

**Pull Requests Merged**: TBD

The Highlights
==============

Time series analysis
--------------------

Fixed parameters in ARIMA estimators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Allow fixing parameters in ARIMA estimator Hannan-Rissanen
  (:func:`statsmodels.tsa.arima.estimators.hannan_rissanen`) through the new
  ``fixed_params`` argument


What's new - an overview
========================

The following lists the main new features of statsmodels 0.13.0. In addition,
release 0.13.0 includes bug fixes, refactorings and improvements in many areas.

Major Feature
-------------
- Allow fixing parameters in ARIMA estimator Hannan-Rissanen (:pr:`7497`)

Submodules
----------

``tsa``
~~~~~~~
- Allow fixing parameters in ARIMA estimator Hannan-Rissanen (:pr:`7497`)