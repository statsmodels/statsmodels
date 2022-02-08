:orphan:

==============
Release 0.13.1
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
**Issues Closed**: 13

**Pull Requests Merged**: 15



What's new - an overview
========================
This a bug fix and deprecation only release.


``maintenance``
---------------
- Merge pull request #7787 from gmcmacran/loglogDoc  (:pr:`7845`)
- Merge pull request #7791 from Wooqo/fix-hw  (:pr:`7846`)
- Merge pull request #7795 from bashtage/bug-none-kpss  (:pr:`7847`)
- Merge pull request #7801 from bashtage/change-setup  (:pr:`7850`)
- Merge pull request #7812 from joaomacalos/zivot-andrews-docs  (:pr:`7852`)
- Merge pull request #7799 from bashtage/update-codecov  (:pr:`7853`)
- Merge pull request #7820 from rgommers/scipy-imports  (:pr:`7854`)
- BACKPORT Merge pull request #7844 from bashtage/future-pandas  (:pr:`7855`)
- Merge pull request #7816 from tncowart/unalias_links  (:pr:`7857`)
- Merge pull request #7832 from larsoner/dep  (:pr:`7858`)


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

Thanks to all of the contributors for the 0.13.1 release (based on git log):

- Josef Perktold
- Kevin Sheppard


These lists of names are automatically generated based on git log, and may not
be complete.

Merged Pull Requests
--------------------

The following Pull Requests were merged since the last release:

- :pr:`7845`: BACKPORT: Merge pull request #7787 from gmcmacran/loglogDoc
- :pr:`7846`: BACKPORT: Merge pull request #7791 from Wooqo/fix-hw
- :pr:`7847`: BACKPORT: Merge pull request #7795 from bashtage/bug-none-kpss
- :pr:`7850`: BACKPORT: Merge pull request #7801 from bashtage/change-setup
- :pr:`7852`: BACKPORT: Merge pull request #7812 from joaomacalos/zivot-andrews-docs
- :pr:`7853`: BACKPORT: Merge pull request #7799 from bashtage/update-codecov
- :pr:`7854`: BACKPORT: Merge pull request #7820 from rgommers/scipy-imports
- :pr:`7855`: BACKPORT Merge pull request #7844 from bashtage/future-pandas
- :pr:`7857`: BACKPORT: Merge pull request #7816 from tncowart/unalias_links
- :pr:`7858`: BACKPORT: Merge pull request #7832 from larsoner/dep
- :pr:`7876`: BACKPORT: Merge pull request #7874 from bashtage/scalar-wald
- :pr:`7877`: BACKPORT: Merge pull request #7842 from bashtage/deprecate-cols
- :pr:`7878`: BACKPORT: Merge pull request #7839 from guilhermesilveira/main
- :pr:`7879`: BACKPORT: Merge pull request #7868 from josef-pkt/tst_penalized_convergence
- :pr:`7880`: MAINT: Update pyproject for 3.10
