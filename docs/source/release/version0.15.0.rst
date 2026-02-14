:orphan:

==============
Release 0.15.0
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


The Highlights
==============

New time series analysis tools
------------------------------

Partial Cross-Correlation Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:func:`~statsmodels.tsa.stattools.pccf` computes the partial
cross-correlation function between two time series. The PCCF measures
the correlation between two time series at different lags after removing
the linear dependence on intermediate observations.

A companion plotting function
:func:`~statsmodels.graphics.tsaplots.plot_pccf` is also available.
