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
