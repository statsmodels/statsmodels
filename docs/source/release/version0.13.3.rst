:orphan:

==============
Release 0.13.3
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
**Issues Closed**: 79

**Pull Requests Merged**: 7


The Highlights
==============


What's new - an overview
========================

The following lists the main new features of statsmodels 0.13.3. In addition,
release 0.13.3 includes bug fixes, refactorings and improvements in many areas.

Submodules
----------


``maintenance``
~~~~~~~~~~~~~~~
- Backport Python 3.11 to 0.13.x branch  (:pr:`8484`)





bug-wrong
---------

A new issue label `type-bug-wrong` indicates bugs that cause that incorrect
numbers are returned without warnings.
(Regular bugs are mostly usability bugs or bugs that raise an exception for
unsupported use cases.)
`see tagged issues <https://github.com/statsmodels/statsmodels/issues?q=is%3Aissue+label%3Atype-bug-wrong+is%3Aclosed+milestone%3A0.13.3/>`_


Major Bugs Fixed
================

See github issues for a list of bug fixes included in this release

- `Closed bugs <https://github.com/statsmodels/statsmodels/pulls?utf8=%E2%9C%93&q=is%3Apr+is%3Amerged+milestone%3A0.13.3+label%3Atype-bug/>`_
- `Closed bugs (wrong result) <https://github.com/statsmodels/statsmodels/pulls?q=is%3Apr+is%3Amerged+milestone%3A0.13.3+label%3Atype-bug-wrong/>`_


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

Thanks to all of the contributors for the 0.13.3 release (based on git log):

- Ewout Ter Hoeven
- Kevin Sheppard


These lists of names are automatically generated based on git log, and may not
be complete.

Merged Pull Requests
--------------------

The following Pull Requests were merged since the last release:

- :pr:`8470`: TST: Relax tolerance on tests that marginally fail
- :pr:`8473`: MAINT: Future fixes for 0.13
- :pr:`8474`: MAINT: Try to fix object issue
- :pr:`8479`: MAINT: Update doc build instructions
- :pr:`8480`: MAINT: Update doc build instructions
- :pr:`8483`: DOC: Fix warnings
- :pr:`8484`: MAINT: Backport Python 3.11 to 0.13.x branch
