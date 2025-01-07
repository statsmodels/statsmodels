:orphan:

==============
Release 0.14.1
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
**Issues Closed**: 41

**Pull Requests Merged**: 22


The Highlights
==============
This is a bug-fix and compatability focused release. There are two enhancements to the graphics module.

What's new - an overview
========================

The following lists the main new features of statsmodels 0.14.1. In addition,
release 0.14.1 includes bug fixes, refactorings and improvements in many areas.

Submodules
----------


``Performance``
~~~~~~~~~~~~~~~
- Faster computation of revision impacts  (:pr:`8937`)



``backport``
~~~~~~~~~~~~
- Mnlogit wald tests, ravel, string cov_names  (:pr:`8907`)



``base``
~~~~~~~~
- Mnlogit wald tests, ravel, string cov_names  (:pr:`8907`)



``discrete``
~~~~~~~~~~~~
- Mnlogit wald tests, ravel, string cov_names  (:pr:`8907`)



``distributions``
~~~~~~~~~~~~~~~~~
- Correct signature of `CopulaDistribution`  (:pr:`8946`)



``genmod``
~~~~~~~~~~
- Add get_margeff to GLM  (:pr:`8889`)



``graphics``
~~~~~~~~~~~~
- Fix inclusion of plots  (:pr:`8963`)
- ccf to optionally return confidence intervals (:pr:`8782`)
- Plot cross-correlations and auto/cross-correlation matrix (:pr:`8783`:)
- plot prediction curve over scatter in GLMGamResults.plot_partial (:pr:`8881`)



``maintenance``
~~~~~~~~~~~~~~~
- Update nightly location  (:pr:`8939`)
- Make changes for deprecations  (:pr:`8940`)
- Switch from == to is for type comparrison  (:pr:`8988`)
- Insert some initial NumPy caps  (:pr:`8989`)
- Block pandas 2.1.0  (:pr:`8990`)



``regression``
~~~~~~~~~~~~~~
- Correct typo in WLS.loglike docstring  (:pr:`8900`)



``stats``
~~~~~~~~~
- 2-sample z-test unequal variances case  (:pr:`8959`)



``tsa``
~~~~~~~
- Ccf to optionally return confidence intervals  (:pr:`8782`)
- Fix inconsistency in `var_model.py`  (:pr:`8948`)



``tsa.statespace``
~~~~~~~~~~~~~~~~~~
- Faster computation of revision impacts  (:pr:`8937`)





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

Thanks to all of the contributors for the 0.14.1 release (based on git log):

- Artem Glebov
- Chad Fulton
- Josef Perktold
- Kevin Sheppard
- Melissa Wu
- Rebecca N. Palmer
- Sebastian Pölsterl
- Tartopohm
- Tom Adamczewski


These lists of names are automatically generated based on git log, and may not
be complete.

Merged Pull Requests
--------------------

The following Pull Requests were merged since the last release:

- :pr:`8782`: ENH/TST: ccf to optionally return confidence intervals
- :pr:`8783`: ENH: Plot cross-correlations and auto/cross-correlation matrix
- :pr:`8881`: ENH: plot prediction curve over scatter in GLMGamResults.plot_partial
- :pr:`8886`: DOC: Correct links to notebooks
- :pr:`8900`: DOC: correct typo in WLS.loglike docstring
- :pr:`8907`: BUG: mnlogit wald tests, ravel, string cov_names
- :pr:`8930`: MAINT: Remove deprecated utility
- :pr:`8932`: CLN: Fix typos
- :pr:`8939`: MAINT: Update nightly location
- :pr:`8940`: MAINT: Make changes for deprecations
- :pr:`8941`: DOC: Add install instructions for nightly
- :pr:`8942`: BUG: Writing read-only arry on pandas 2/CoW
- :pr:`8946`: DOC: correct signature of `CopulaDistribution`
- :pr:`8948`: DOC: fix inconsistency in `var_model.py`
- :pr:`8963`: DOC: Fix inclusion of plots
- :pr:`8974`: DOC: Include correct plot in scatter_ellipse
- :pr:`8988`: STY: Switch from == to is for type comparrison
- :pr:`8989`: MAINT: Insert some initial NumPy caps
- :pr:`8990`: MAINT: Block pandas 2.1.0
