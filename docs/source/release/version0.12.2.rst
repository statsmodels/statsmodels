:orphan:

==============
Release 0.12.2
==============

Release summary
===============

Statsmodels 0.12.2 is a bug-fix release with no new features
compared to 0.12.1. Notable changes include fixes for a bug that could lead
to incorrect results in forecasts with the new ARIMA model (when `d > 0` and
`trend='t'`) and a bug in the LM test for autocorrelation.

Documentation
-------------

Documentation for the current release and for ongoing development are available at:

- `Stable <https://www.statsmodels.org/>`_, the latest release
- `Development <https://www.statsmodels.org/devel/>`_, the latest build of the master branch

Stats
-----
**Issues Closed**: 42

**Pull Requests Merged**: 4

Major Bugs Fixed
================

The primary bugs fixed include:

- :pr:`7250`: Bug in forecasting with new ARIMA model when `d > 0` and `trend='t'`.
- :pr:`7259`: Bug in LM test for autocorrelation

See github issues for a list of all bug fixes included in this release

- `Closed bugs <https://github.com/statsmodels/statsmodels/pulls?utf8=%E2%9C%93&q=is%3Apr+is%3Amerged+milestone%3A0.12.2+label%3Atype-bug/>`_
- `Closed bugs (wrong result) <https://github.com/statsmodels/statsmodels/pulls?q=is%3Apr+is%3Amerged+milestone%3A0.12.2+label%3Atype-bug-wrong/>`_


Development summary and credits
===============================

Besides receiving contributions for new and improved features and for bugfixes,
important contributions to general maintenance for this release came from

- Chad Fulton
- Brock Mendel
- Peter Quackenbush
- Kerby Shedden
- Kevin Sheppard

and the general maintainer and code reviewer

- Josef Perktold

Additionally, many users contributed by participation in github issues and
providing feedback.

Thanks to all of the contributors for the 0.12.2 release (based on git log):

- Chad Fulton
- Graham Inggs
- Joris Van Den Bossche
- Josef Perktold
- Kevin Sheppard
- Mike Ovyan
- Natalie Heer


These lists of names are automatically generated based on git log, and may not
be complete.

Merged Pull Requests
--------------------

The following Pull Requests were merged since the last release:

- :pr:`7123`: DOC: Port missed doc fix
- :pr:`7221`: MAINT: Backport fixes for 0.12.2 compat release
- :pr:`7222`: Backports
- :pr:`7291`: Backports
