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
- :func:`~statsmodels.stats.meta_analysis.combine_effects` truncates the
  DerSimonian-Laird estimate of ``tau2`` (``method_re="dl"`` or ``"chi2"``) at
  zero. The random effects results change when the estimate was negative.
  The ``h2`` and ``i2`` attributes of the results are at least 1 and 0,
  matching R metafor. They were below 1 and negative when ``q < k - 1``. Lists
  and tuples are accepted as input. :pr:`10522` :issue:`10520`
- :func:`~statsmodels.stats.gof.gof_binning_discrete` and
  :func:`~statsmodels.stats.gof.gof_chisquare_discrete` build the bins from the
  support of the distribution instead of a fixed window of the integers from
  -1000 to 1000. Loc shifts and large means no longer collapse the bins or
  drop observations, the bin edges are exact for means of 1e9 and larger, and
  distributions for which the generic ``ppf`` of SciPy fails to converge, such
  as ``skellam`` with large parameters, work. ``nsupp`` below 1 raises a
  ``ValueError``. :pr:`10529` :issue:`4359`
- The results of
  :func:`~statsmodels.stats.rates.power_equivalence_poisson_2indep` and
  :func:`~statsmodels.stats.rates.power_equivalence_neginb_2indep` change for
  ``nobs_ratio != 1``. The variances used the wrong sample size and sample size
  ratio, and the rates were interchanged in the negative binomial variance. The
  ``ftotal`` null variance of
  :func:`~statsmodels.stats.rates.power_equivalence_neginb_2indep` and
  :func:`~statsmodels.stats.rates.power_negbin_ratio_2indep` is now equal to the
  score variance when ``dispersion=0``. A ``nobs_ratio`` with a value that is
  not positive and finite now raises a ``ValueError`` instead of returning
  ``nan``. :pr:`10521` :issue:`10519`

Breaking Changes and Deprecations
=================================

- The ``return_df`` argument of
  :func:`~statsmodels.stats.diagnostic.acorr_ljungbox` has no effect, since
  the function always returns a ``DataFrame``. Passing it now raises a
  ``FutureWarning``, and it will be removed in statsmodels 0.17.0.
  :issue:`8968`
- The ``linear`` argument of ``predict`` in
  :class:`~statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM`,
  :class:`~statsmodels.genmod.bayes_mixed_glm.PoissonBayesMixedGLM` and
  :class:`~statsmodels.genmod.bayes_mixed_glm.BayesMixedGLMResults` is
  deprecated in favor of ``which``, matching
  :meth:`~statsmodels.genmod.generalized_linear_model.GLM.predict`. Use
  ``which="linear"`` instead of ``linear=True`` and ``which="mean"`` instead
  of ``linear=False``. Passing ``linear`` now raises a ``FutureWarning``, and
  it will be removed in statsmodels 0.17.0. :issue:`8562`
