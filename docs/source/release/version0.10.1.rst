:orphan:

==============
Release 0.10.1
==============

Release summary
===============
This is a bug fix-only release

Development summary and credits
===============================

Besides receiving contributions for new and improved features and for bugfixes,
important contributions to general maintenance for this release came from

* Chad Fulton
* Brock Mendel
* Peter Quackenbush
* Kerby Shedden
* Kevin Sheppard

and the general maintainer and code reviewer

* Josef Perktold

These lists of names are automatically generated based on git log, and may not
be complete.

Merged Pull Requests
--------------------

The following Pull Requests were merged since the last release:

* :pr:`5784`: MAINT: implement parts of #5220, deprecate ancient aliases
* :pr:`5892`: BUG: fix pandas compat
* :pr:`5893`: BUG: exponential smoothing - damped trend gives incorrect param, predictions
* :pr:`5895`: DOC: improvements to BayesMixedGLM docs, argument checking
* :pr:`5897`: MAINT: Use pytest.raises to check error message
* :pr:`5903`: BUG: Fix kwargs update bug in linear model fit_regularized
* :pr:`5917`: BUG: TVTP for Markov regression
* :pr:`5921`: BUG: Ensure exponential smoothers has continuous double data
* :pr:`5930`: BUG: Limit lags in KPSS
* :pr:`5933`: MAINT: Fix test that fails with positive probability
* :pr:`5935`: CLN: port parts of #5220
* :pr:`5940`: MAINT: Fix linting failures
* :pr:`5944`: BUG: Restore ResettableCache
* :pr:`5951`: BUG: Fix mosaic plot with missing category
* :pr:`5971`: BUG: Fix a future issue in ExpSmooth
