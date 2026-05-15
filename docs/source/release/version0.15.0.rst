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

New models and extensions to models
------------------------------------

Modified Efficient Importance Sampling (MEIS) for non-Gaussian state space models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:class:`~statsmodels.tsa.statespace.meis.MEISMixin` adds MEIS-based
maximum likelihood estimation to any ``MLEModel`` subclass, enabling
parameter estimation for state space models with non-Gaussian observations
(e.g., Student's-t stochastic volatility, count data models). The
implementation follows Koopman, Lit & Nguyen (2018), *"Modified efficient
importance sampling for partially non-Gaussian state space models"*,
Statistica Neerlandica, 73(1), 44-62.

Key classes:

- :class:`~statsmodels.tsa.statespace.meis.MEISMixin` -- mixin adding
  ``fit_meis()`` and ``smooth_signal_meis()`` to ``MLEModel`` subclasses
- :class:`~statsmodels.tsa.statespace.meis.MEISImportanceDensity` --
  iterative construction of the Gaussian importance density
- :class:`~statsmodels.tsa.statespace.meis.MEISLikelihood` --
  importance-sampling log-likelihood with bias correction
- :class:`~statsmodels.tsa.statespace.meis.MEISResults` -- results
  container with ``smooth_signal()`` for signal extraction
- :class:`~statsmodels.tsa.statespace.meis.DurbinKoopmanSimulator` --
  simulation smoother following Durbin & Koopman (2002)

See :pr:`xxxx`.
