:orphan:

==============
Release 0.10.2
==============

Release summary
===============
This is a bug release and adds compatibility with Python 3.8.

Development summary and credits
===============================

Besides receiving contributions for new and improved features and for bugfixes,
important contributions to general maintenance for this release came from

* Chad Fulton
* Qingqing Mao
* Diego Mazon
* Brock Mendel
* Guglielmo Saggiorato
* Kevin Sheppard
* Tim Staley

Merged Pull Requests
--------------------

The following Pull Requests were merged since the last release:

* :pr:`5935`: CLN: port parts of #5220
* :pr:`5951`: BUG: Fix mosaic plot with missing category
* :pr:`5996`: BUG: Limit lags in KPSS
* :pr:`5998`: Replace alpha=0.05 with alpha=alpha
* :pr:`6030`: Turn relative import into an absolute import
* :pr:`6044`: DOC: Fix notebook due to pandas index change
* :pr:`6046`: DOC: Remove DynamicVAR
* :pr:`6091`: MAINT/SEC: Remove unnecessary pickle use
* :pr:`6092`: MAINT: Ensure r download cache works
* :pr:`6093`: MAINT: Fix new cache name
* :pr:`6117`: BUG: Remove extra LICENSE.txt and setup.cfg
* :pr:`6105`: Update correlation_tools.py
* :pr:`6050`: BUG: MLEModel now passes nobs to Representation
* :pr:`6205`: MAINT: Exclude pytest-xdist 1.30
* :pr:`6246`: TST: Add Python 3.8 environment
