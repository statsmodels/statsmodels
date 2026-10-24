:orphan:

==============
Release 0.16.0
==============

Release summary
===============

statsmodels is using github to store the updated documentation. Two versions
are available:

- `Stable <https://www.statsmodels.org/>`_, the latest release
- `Development <https://www.statsmodels.org/devel/>`_, the latest build of the main branch

Bug Fixes
=========

- The tricube kernel in ``statsmodels.nonparametric.kernels`` returns 0
  outside of its support, :math:`|u| > 1`, instead of the peak value. Results
  of :class:`~statsmodels.nonparametric.kernel_regression.KernelReg` and
  :class:`~statsmodels.nonparametric.kernel_regression.KernelCensoredReg`
  that use ``ckertype="tricube"`` change. :issue:`10541`
- The default ``value`` of
  :func:`~statsmodels.stats.rates.power_poisson_ratio_2indep` is now 1, the
  null hypothesis of equal rates. The previous default of 0 gave a power of 0,
  1 or ``nan``. The value must now be positive. :issue:`10511`
- :func:`~statsmodels.stats.proportion.power_binom_tost` returns a power of 0
  instead of a negative value if the rejection region of the exact test is
  empty. :issue:`10507`

Breaking Changes and Deprecations
=================================

- The ``return_df`` argument of
  :func:`~statsmodels.stats.diagnostic.acorr_ljungbox` has no effect, since
  the function always returns a ``DataFrame``. Passing it now raises a
  ``FutureWarning``, and it will be removed in statsmodels 0.17.0.
  :issue:`8968`
