:orphan:

==============
Release 0.14.2
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
**Issues Closed**: 22

**Pull Requests Merged**: 24


The Highlights
==============
This release brings compatibility with NumPy 2.0.0. This is the only key feature
of this release. Several minor patches have been backported. These either
fix bugs that have been documented, improve the documentation or are necessary for
NumPy 2.0 compatability.

NumPy 2.0 is only available for Python 3.9+. This means that the minimum Python
has been increased to 3.9 to match. NumPy 2 is only required to build statsmodels,
and statsmodels will continue to run on NumPy 1.22.3+.

Note that when running using NumPy 2, all dependencies that use build against NumPy
(e.g., Scipy and pandas) must be NumPy 2 compatible. You can continue to run against
NumPy 1.22 - 1.26 along with other components of the scientific Python stack until
all required dependencies have been updated.


What's new - an overview
========================

The following lists the main new features of statsmodels 0.14.2. In addition,
release 0.14.2 includes bug fixes, refactorings and improvements in many areas.

Submodules
----------


``dependencies``
~~~~~~~~~~~~~~~~
- Bump github/codeql-action from 2 to 3  (:pr:`9098`)
- Bump ts-graphviz/setup-graphviz from 1 to 2  (:pr:`9149`)



``multivariate``
~~~~~~~~~~~~~~~~
- Add MultivariateLS  (:pr:`8919`)



``robust``
~~~~~~~~~~
- Outlier robust covariance - rebased  (:pr:`8129`)



``stats``
~~~~~~~~~
- Outlier robust covariance - rebased  (:pr:`8129`)



``tsa.statespace``
~~~~~~~~~~~~~~~~~~
- Ensure ARIMA simulation is reproducable  (:pr:`9165`)





bug-wrong
---------

A new issue label `type-bug-wrong` indicates bugs that cause that incorrect
numbers are returned without warnings.
(Regular bugs are mostly usability bugs or bugs that raise an exception for
unsupported use cases.)
`see tagged issues <https://github.com/statsmodels/statsmodels/issues?q=is%3Aissue+label%3Atype-bug-wrong+is%3Aclosed+milestone%3A0.14>`_


Major Bugs Fixed
================

See github issues for a list of bug fixes included in this release

- `Closed bugs <https://github.com/statsmodels/statsmodels/pulls?utf8=%E2%9C%93&q=is%3Apr+is%3Amerged+milestone%3A0.14+label%3Atype-bug>`_
- `Closed bugs (wrong result) <https://github.com/statsmodels/statsmodels/pulls?q=is%3Apr+is%3Amerged+milestone%3A0.14+label%3Atype-bug-wrong>`_


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

Thanks to all of the contributors for the 0.14.2 release (based on git log):

- Josef Perktold
- Kevin Sheppard
- Manlai Amar
- Michel De Ruiter
- Trinh Quoc Anh
- Zhengbo Wang
- cppt
- dependabot[bot]
- s174139


These lists of names are automatically generated based on git log, and may not
be complete.

Merged Pull Requests
--------------------

The following Pull Requests were merged since the last release:

- :pr:`9029`: Update seasonal.py
- :pr:`9098`: Bump github/codeql-action from 2 to 3
- :pr:`9110`: BLD: Update minimums
- :pr:`9111`: MAINT: Fix future issues in pandas
- :pr:`9115`: MAINT: Clean up and silence some warnings
- :pr:`9117`: edited requirements.txt
- :pr:`9124`: MAINT: Fix future issues due to array shapes
- :pr:`9142`: Fix linting error
- :pr:`9143`: Fix string formatting
- :pr:`9144`: MAINT: Replace quarterly string identified
- :pr:`9149`: Bump ts-graphviz/setup-graphviz from 1 to 2
- :pr:`9150`: MAINT: Fixes for future changes
- :pr:`9158`: DOC: Fix broken in `linear_regression_diagnostics_plots`
- :pr:`9165`: BUG: Ensure ARIMA simulation is reproducable
- :pr:`9192`: DOC: fixed boxpierece typos
- :pr:`9195`: MAINT: Make compatability with NumPy 2
- :pr:`9200`: Cherry pick commits from 0.15 for 0.14.3
