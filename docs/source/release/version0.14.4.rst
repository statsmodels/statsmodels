:orphan:

==============
Release 0.14.4
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
**Issues Closed**: 1

**Pull Requests Merged**: 1


The Highlights
==============
This release bring official Pyodide support to a statsmodel release. It is otherwise identical to
the previous release.

Special thanks to Agriya Khetarpal for working through Pyodide-specific issues, and
improving other areas of statsmodels while doing so.


Merged Pull Requests
--------------------

The following Pull Requests were merged since the last release:

- :pr:`9365`: Backport of #9270: add Pyodide support and CI jobs for v0.14.x

The main enhancements and new features are listed below.

Enhancements
------------

* Post-hoc methods (`pairwise_tukeyhsd` and `allpairtest`) can now be directly called from the results of a repeated measures ANOVA (`AnovaRM`). (:issue:`9491`)

The main bug fixes are listed below.

Bug Fixes
