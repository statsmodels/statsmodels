:orphan:

==============
Release 0.14.3
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

**Pull Requests Merged**: 5


The Highlights
==============
This release if a packaging and modernization release. It solves two key issues:

1. Corrects the build procedure for MacOS on both x86_64 and arm64
2. Improves compatibility with recent pandas releases

This release is NumPy 2.0 compatible. NumPy 2.0 is only available for Python 3.9+.
This means that the minimum Python
has been increased to 3.9 to match. NumPy 2 is only required to build statsmodels,
and statsmodels will continue to run on NumPy 1.22.3+.

Note that when running using NumPy 2, all dependencies that use build against NumPy
(e.g., Scipy and pandas) must be NumPy 2 compatible. You can continue to run against
NumPy 1.22 - 1.26 along with other components of the scientific Python stack until
all required dependencies have been updated.


What's new - an overview
========================

There are no new features in release 0.14.3.

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


Merged Pull Requests
--------------------

The following Pull Requests were merged since the last release:

- :pr:`9356`: MAINT: Backport changes needed for 0.14.3 release
- :pr:`9358`: TST: Relax tolerance on test that fails for dynamic factor
- :pr:`9359`: MAINT: Run pyupgrade on 0.14 branch
- :pr:`9363`: DOC: Add release note for 0.14.3
- :pr:`9364`: DOC: Spelling