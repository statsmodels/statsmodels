:orphan:

==============
Release 0.14.5
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
This release fixes an import issue when using SciPy 1.16 or later. It also
fixes some small future issues and ensures that the test suit passes
against recent releases of upstream projects.

Merged Pull Requests
--------------------

The following Pull Requests were merged since the last release:

- :pr:`9586`: MAINT: Remove lazywhere
