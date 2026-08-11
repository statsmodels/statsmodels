:orphan:

==============
Release 0.12.1
==============

Release summary
===============
This is a bug fix release.

Development summary and credits
===============================

Besides receiving contributions for new and improved features and for bugfixes,
important contributions to general maintenance for this release came from

* Kevin Sheppard
* Chad Fulton
* Josef Perktold
* Kerby Shedden
* Pratyush Sharan

Merged Pull Requests
--------------------

The following Pull Requests were merged since the last release:

* :pr:`7016`: BLD: avoid setuptools version 50, windows build problem
* :pr:`7017`: BUG: param names with higher order trend in VARMAX
* :pr:`7020`: REF: Don't validate specification in SARIMAX when cloning to get extended time varying matrices
* :pr:`7025`: BUG: Ensure bestlag is defined in autolag
* :pr:`7028`: BUG: Correct axis None case
* :pr:`7040`: Bug fix ets get prediction
* :pr:`7052`: ENH: handle/warn for singularities in MixedLM
* :pr:`7055`: BUG: Fix squeeze when nsimulation is 1
* :pr:`7073`: DOC: fix several doc issues in stats functions
* :pr:`7088`: DOC: augmented docstrings from statsmodels.base.optimizer
* :pr:`7090`: DOC: Fix contradicting KPSS-statistics interpretations in stationarity_detrending_adf_kpss.ipynb
* :pr:`7093`: BUG: Correct prediction intervals for ThetaModel
* :pr:`7109`: some fixes in the doc of grangercausalitytests
* :pr:`7116`: BUG: don't raise error in impacts table if no news.
* :pr:`7118`: MAINT: Fix issues in main branches of dependencies
