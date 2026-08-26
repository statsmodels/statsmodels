:orphan:

==============
Release 0.15.0
==============

Release summary
===============

This note covers all changes merged into ``main`` between the ``v0.15.0.dev0``
tag (2023-05-05) and the current development head (2026-08-26).

statsmodels is using github to store the updated documentation. Two versions
are available:

- `Stable <https://www.statsmodels.org/>`_, the latest release
- `Development <https://www.statsmodels.org/devel/>`_, the latest build of the main branch

**Warning**

API stability is not guaranteed for new features, although even in this case
changes will be made in a backwards compatible way if possible. The
stability of a new feature depends on how much time it was already in
statsmodels main and how much usage it has already seen. If there are
specific known problems or limitations, then they are mentioned in the
docstrings.

Release Statistics
-------------------

- **Issues closed**: 355
- **Pull requests merged**: 649
- **Non-merge commits**: 1730
- **Contributors** (by git log author, unique names): 169
- **Time span**: 2023-05-05 through 2026-08-26

The Highlights
===============

SPEC-007: consistent use of ``rng`` for randomness
---------------------------------------------------

statsmodels is standardizing on a single ``rng`` keyword for supplying
entropy (an integer seed, an array of integers, a NumPy ``Generator``, or a
``RandomState``) wherever a model, estimator, or plotting function needs
randomness, in line with the community's `SPEC 007
<https://scientific-python.org/specs/spec-0007/>`_ convention. The older
``random_state`` and ``seed`` keywords are deprecated in favor of ``rng``.
Passing the old keyword still works and is transparently remapped to
``rng``, but it now raises a ``FutureWarning`` and will be removed in a
future release. This is one of the largest cross-cutting changes in this
release and touches, among others:

- State space models (``MLEResults.simulate``, simulation smoothers, impulse
  response simulation): ``random_state`` -> ``rng``.
- Distributions: copulas, ``BernsteinDistribution``, ``DiscretizedCount``,
  ``MixtureDistribution``, and related ``rvs``-style methods:
  ``random_state`` -> ``rng``.
- ``MixedLM``, ``nonlinls``, GAM cross-validation, and several ``sandbox``
  distributions: ``random_state`` -> ``rng``.
- Nonparametric estimation (``KDEMultivariate``, ``KDEMultivariateConditional``,
  ``KernelReg``, ``KernelCensoredReg``, ``TestRegCoefC``/``TestRegCoefD``):
  ``seed`` -> ``rng``.
- VAR/SVAR/IRF simulation and Monte Carlo error bands (``varsim``,
  ``VAR.simulate_var``, ``VAR.plotsim``, ``VARResults.irf_errband_mc``,
  ``VARResults.irf_resim``, ``SVARResults.sirf_errband_mc``, and the
  ``IRAnalysis.plot``/``plot_cum_effects``/``errband_mc``/``err_band_sz1``/
  ``err_band_sz2``/``err_band_sz3``/``cum_errband_mc`` family): ``seed`` -> ``rng``.
- ``ARDL.bounds_test``, ``graphics.functional.hdrboxplot`` (``seed`` and
  ``kernel_seed``), and ``sandbox.panel.random_panel.PanelSample``: ``seed`` -> ``rng``.
- The internal ``statsmodels.tools.rng_qrng.check_random_state`` helper
  (which also accepts ``scipy.stats.qmc.QMCEngine`` instances) is now used
  consistently across these code paths to turn whatever is passed via
  ``rng`` into an actual ``Generator``/``RandomState`` instance.

See *Breaking Changes and Deprecations* below for what this means for
existing code. :pr:`9737`, :pr:`9615`, :pr:`9831`, :pr:`9947`, :pr:`9950`

NamedTuple return values replace bare tuples
------------------------------------------------

Many statsmodels functions historically returned a plain tuple whose *length*
depended on the arguments passed, so that ``adfuller(x)`` and
``adfuller(x, store=True)`` returned a different number of values. This makes
results hard to unpack defensively, hard to document, and hard to type.

These functions now return purpose-built ``NamedTuple`` result classes with a
fixed set of fields; fields that were not requested are ``None``. Because a
``NamedTuple`` is a ``tuple``, positional unpacking, indexing and comparison
against plain tuples all continue to work, and field access such as
``res.pvalue`` becomes available.

The migration follows a single rule:

- Where the ``NamedTuple`` unpacks exactly like the tuple it replaces, it is
  simply returned now, with no deprecation and no warning. This covers, among
  others, ``pacf``/``ccf``/``pccf`` with ``alpha`` set, ``lagmat`` with
  ``original="sep"``, ``kdensity``/``kdensityfft`` with the default
  ``retgrid=True``, ``plot_partregress`` with ``ret_coords=True``, and the
  ``store=True`` paths of the ``stats.diagnostic`` tests.
- Where adopting it would change how many values are unpacked, the legacy
  tuple is still returned and a ``FutureWarning`` is raised. Pass
  ``result_object=True`` to opt in now, or ``result_object=False`` to keep
  the current behaviour and silence the warning. The default changes in 0.16.

Functions whose result shape never varied were converted outright, with no
flag and no warning: ``block_jackknife``, ``q_stat``, ``pacf_burg``,
``levinson_durbin``, ``levinson_durbin_pacf``,
``breakvar_heteroskedasticity_test``, ``coint``, ``cffilter``, ``hpfilter``,
``hamilton_filter``, ``forecast_interval``, the IRF/SIRF error-band methods,
the ARIMA parameter estimators, and ``RegressionResults.compare_lr_test``.

The migration was completed by converting the remaining ``Holder``/
``HolderTuple`` result objects across ``stats`` and ``robust`` (covariance
and scale estimators, ``proportion``, ``rates``, ``nonparametric``,
``multivariate``, ``effect_size``, ``oneway``, and more) to documented
``NamedTuple`` classes, and ``HolderTuple`` itself is now deprecated (see
*Breaking Changes and Deprecations* below). Where a converted class used to
support unpacking into a short tuple like ``statistic, pvalue = result``, a
``compat_2tuple_unpack`` decorator preserves that behaviour, with a
``FutureWarning``, during the transition.

:pr:`10025`, :pr:`10027`, :pr:`10029`, :pr:`10030`, :pr:`10031`, :pr:`10035`,
:pr:`10072`

Formula engine: patsy is no longer the only option
----------------------------------------------------

statsmodels now has an abstracted formula-handling layer
(``statsmodels.formula``) that can use either ``patsy`` (the default engine
when it is installed, for backward compatibility) or `formulaic
<https://matthewwardrop.github.io/formulaic/>`_ as the engine behind the
formula interface (``smf.ols("y ~ x", data=df)``, etc.). The engine can be
selected explicitly with the ``SM_FORMULA_ENGINE`` environment variable
(``"patsy"`` or ``"formulaic"``). ``formulaic`` is now a required runtime
dependency (``formulaic>=1.1.0``) even when ``patsy`` continues to be used
as the default engine. This lays the groundwork for statsmodels to move away
from ``patsy``, which has been in low-maintenance mode for several years.
:pr:`9423`, :pr:`9470`

Build system: meson-python replaces setuptools
-------------------------------------------------

statsmodels' build backend switched from ``setuptools`` (with a custom
``setup.py``) to `meson-python <https://meson-python.readthedocs.io/>`_.
Anyone building statsmodels from source needs Meson/Ninja available and a
build environment satisfying the new build requirements (``numpy>=2.0``,
``scipy>=1.13``, ``cython>=3.0.13``). This does not affect users installing
prebuilt wheels from PyPI. :pr:`9634`

Polars DataFrame support
---------------------------

Models and the formula API now accept `Polars <https://pola.rs/>`_
``DataFrame``/``Series`` objects wherever pandas objects are accepted.
Polars input is converted to pandas at the data-entry point
(``handle_data``, and the formula-handling layer), so all internal
computation continues to use pandas/NumPy unchanged; column names, index
information, and predictions with Polars ``exog`` are preserved. Polars is
an optional dependency: code paths that do not receive Polars objects are
unaffected, and the relevant tests are skipped when Polars is not
installed. :pr:`9804`

New robust estimation tools
------------------------------

Several new robust estimators and supporting tools were added:

- :class:`statsmodels.robust.covariance.CovDetMCD` (minimum covariance
  determinant with deterministic starts), :class:`~statsmodels.robust.covariance.CovDetS`
  (S-estimator for mean/covariance with deterministic starts), and
  :class:`~statsmodels.robust.covariance.CovDetMM` (an MM-estimator built on
  top of ``CovDetS``). These are preliminary/experimental APIs. :pr:`9227`, :pr:`8129`
- :class:`statsmodels.robust.resistant_linear_model.RLMDetSMM`, an
  MM-estimator for regression using S-estimator starting values, plus
  additional robust norms and supporting tools. :pr:`9186`
- Fixes and additions to ``scale.Huber`` and a new robust M-scale estimator.
  :pr:`9210`

New models and statistical tests
------------------------------------

- :class:`statsmodels.multivariate.multivariate_ols.MultivariateLS`, a new
  multivariate least-squares model. :pr:`8919`
- :func:`statsmodels.tsa.stattools.leybourne` implementing the Leybourne-McCabe
  stationarity test. :pr:`9399`
- A two-sample z-test for the unequal-variances case. :pr:`8959`
- ``"one-sided"`` alternative hypotheses for ``proportion_confint`` and
  ``confint_poisson``. :pr:`9249`, :pr:`9255`
- Games-Howell post-hoc test added alongside a fix to Tukey's HSD for the
  unequal-variance case. :pr:`9487`
- A sample-size calculation for the Wilcoxon/Mann-Whitney test. :pr:`9401`
- Order validation for the Hannan-Rissanen ARMA estimator. :pr:`9819`
- :func:`statsmodels.tsa.stattools.pccf`, the partial cross-correlation
  function, together with a companion
  :func:`~statsmodels.graphics.tsaplots.plot_pccf`. :pr:`9802`
- :func:`statsmodels.tsa.filters.hamilton_filter.hamilton_filter`, Hamilton's
  regression-based alternative to the HP filter. :pr:`9957`, :pr:`9991`
- :func:`statsmodels.tsa.stattools.block_jackknife`, a delete-k (block)
  jackknife estimator of bias and standard error. :pr:`10001`
- ``ARDL`` models can now use a ``"ctt"`` trend. :pr:`9518`
- ``x13_arima_analysis`` gained seasonality fit diagnostics and an optional
  raw spec parameter. :pr:`9498`, :pr:`9550`
- A Jonckheere-Terpstra test for ordered k-sample alternatives
  (:func:`statsmodels.stats.nonparametric.jonckheere_terpstra`), following
  Terpstra (1952) and Jonckheere (1954). :pr:`9874`, :pr:`10067`, :pr:`10075`
- A Diebold-Mariano test for equal predictive accuracy of two forecasts
  (:func:`statsmodels.tsa.stattools.diebold_mariano_test`), with an
  optional Harvey et al. (1997) small-sample correction. :pr:`10066`
- A Pesaran-Timmermann test of directional predictive accuracy
  (:func:`statsmodels.stats.diagnostic.pesaran_timmermann`). :pr:`10055`
- Local false discovery rate estimation
  (:func:`statsmodels.stats.multitest.local_fdr_correction`), based on the
  Grenander estimator of the p-value density. :pr:`10069`

New and improved plots
--------------------------

- :func:`statsmodels.graphics.tsaplots.plot_ccf` and
  :func:`~statsmodels.graphics.tsaplots.plot_accf_grid` for plotting
  cross-correlations and cross-correlation matrices, and ``ccf`` gained an
  option to return confidence intervals. :pr:`8782`, :pr:`8783`
- :func:`statsmodels.graphics.tsaplots.seasonal_diagnostic_plot`, a new
  seasonal diagnostic plot. :pr:`9787`
- :func:`statsmodels.graphics.regressionplots.add_ellipse` for adding
  confidence ellipses to scatter plots. :pr:`9815`
- ``qqplot_2samples`` accepts additional plot keyword arguments. :pr:`9544`

GLM and other model enhancements
------------------------------------

- ``GLMResults.get_margeff`` (marginal effects for GLM). :pr:`8889`
- GLM models now preserve the names of input pandas Series. :pr:`9130`
- ``het_white`` gained an option to omit interaction (cross) terms. :pr:`9691`
- Faster computation of state space "news"/revision impacts, and a
  significant performance optimization of VECM to avoid an :math:`O(T^2)`
  projection matrix. :pr:`8937`, :pr:`9720`
- :func:`statsmodels.stats.stattools.medcouple` gained an :math:`O(N \log N)`
  algorithm (``use_fast=True``, the default), replacing the previous
  :math:`O(N^2)` implementation, which remains available via
  ``use_fast=False``. :pr:`9571`

Platform and packaging compatibility
----------------------------------------

- Cython 3 compatibility, and compatibility of the ``tsa.statespace`` Cython
  code with SciPy ILP64 builds. :pr:`9078`, :pr:`9798`
- Experimental Pyodide/WebAssembly support and CI jobs. :pr:`9270`, :pr:`9343`
- Free-threaded (no-GIL) CPython compatibility work, including
  free-threading-compatible Cython modules and CI coverage. :pr:`9717`

Stricter input validation for string-valued options
---------------------------------------------------------

Late in the release cycle, essentially every string-valued parameter that
accepts a fixed set of options (``method``, ``alternative``, ``trend``, and
similar) was audited and, where it wasn't already, routed through
:func:`statsmodels.tools.validation.string_like` with an explicit
``options=`` tuple (:pr:`10161`, plus follow-ups :pr:`10167`, :pr:`10173`).
This is the largest single change in this release by number of call sites
touched, and it changes behavior in two distinct ways:

- **Previously-silent bad input now raises a clean, documented**
  ``ValueError``. A number of functions had validation gaps where an
  unrecognized string either silently fell through to a default branch (for
  example ``VECM``'s ``deterministic``, ``seasonal_decompose``'s ``model``,
  and ``oneway``'s ``use_var`` family) or produced a confusing
  ``KeyError``/``NameError`` instead of the documented error (for example
  :func:`~statsmodels.tsa.arima.specification.SARIMAXSpecification.validate_estimator`).
  Code that was accidentally relying on one of these fallback paths, rather
  than passing a value from the documented ``{...}`` set, will now see a
  ``ValueError`` where it previously ran (possibly incorrectly) without
  complaint.
- **Undocumented short-form aliases now emit a** ``FutureWarning``
  **instead of working silently.** The most widespread example is the
  ``alternative`` parameter used throughout ``stats`` and ``tsa`` for
  hypothesis-test direction (``"two-sided"``/``"larger"``/``"smaller"``, or
  ``"increasing"``/``"decreasing"``/``"two-sided"`` for heteroskedasticity
  tests): informal short forms such as ``"2s"``, ``"l"``, ``"s"``, ``"i"``,
  ``"inc"``, ``"d"``, ``"dec"``, or ``"2"`` were accepted but never
  documented. These still work today, but now raise a ``FutureWarning``
  naming the documented spelling to switch to, and will stop being accepted
  after statsmodels 0.16 (:pr:`10170`, :pr:`10180`). This affects, among others,
  :class:`~statsmodels.stats.weightstats.DescrStatsW`/
  :class:`~statsmodels.stats.weightstats.CompareMeans` and the module-level
  ``ztest``/``zconfint``/``ztost``/``ttest_ind``/``ttost_ind`` functions in
  :mod:`statsmodels.stats.weightstats`, :func:`~statsmodels.stats.diagnostic.het_goldfeldquandt`,
  :func:`~statsmodels.tsa.stattools.breakvar_heteroskedasticity_test` (and the
  state space/ETS ``test_heteroskedasticity`` methods built on it),
  ``PredictionResults.t_test``/``PredictionResultsBase.t_test``, most of
  :mod:`statsmodels.stats.power`, several functions in
  :mod:`statsmodels.stats.proportion` and :mod:`statsmodels.stats.rates`,
  :func:`~statsmodels.stats.oneway.confint_noncentrality`,
  :func:`~statsmodels.stats.meta_analysis.effectsize_2proportions` (whose
  ``statistic`` parameter separately gained ``"rd"``/``"rr"``/``"or"``/
  ``"arcsine"`` as deprecated aliases for ``"diff"``/``"risk-ratio"``/
  ``"odds-ratio"``/``"arcsin"``), and
  :func:`~statsmodels.stats._lilliefors.ksstat` (whose ``alternative`` gained
  deprecated aliases for the ``scipy.stats.kstest``-style spellings
  ``"two_sided"``/``"less"``/``"greater"``). Pass the documented spelling to
  silence the warning; the deprecated forms will be removed, not just
  undocumented, starting after statsmodels 0.16.

A few consequential bug fixes
---------------------------------

A few of the more consequential correctness fixes in this release (see
*Notable Bug Fixes* below for the full list):

- ``families.Binomial.deriv()`` was missing a division by ``n`` and returned
  an incorrect value; it now correctly returns ``1 - 2 * mu / n``. :pr:`9862`
- The log-likelihood computation for ``ETSModel`` was corrected. :pr:`9400`
- A state space model transition-timing bug was fixed. :pr:`9688`
- ``anova_lm`` silently returned ``NaN`` p-values when models were passed in
  reverse order. :pr:`9852`
- Numerical instability in VIF was fixed by standardizing the design matrix
  before computing it. :pr:`9835`
- ``wald_test_terms`` reported the raw number of constraint rows as
  ``df_constraint``, rather than the rank-adjusted degrees of freedom that
  ``wald_test`` itself already computes; this was wrong for rank-deficient
  models (e.g. incomplete factorial designs). :pr:`9907`
- The adjusted (unbiased) ``ccovf``/``acovf`` normalized by ``len(x) - k``
  rather than by the actual number of overlapping observation pairs, which
  is only the same thing when the two series are equal length. :pr:`9916`
- ``breakvar_heteroskedasticity_test`` (and the state space/ETS
  ``test_heteroskedasticity`` methods built on it) referred the ratio of two
  sums of squares directly to ``F(numer_dof, denom_dof)``; that ratio is only
  ``F``-distributed after rescaling by ``denom_dof / numer_dof``, so p-values
  were wrong whenever missing observations left the two subsets with
  different numbers of usable residuals. Balanced samples were unaffected.
  :pr:`10171`
- Every ``TreatmentEffectResults`` produced by
  :class:`~statsmodels.treatment.treatment_effects.TreatmentEffect`'s
  ``ra``/``aipw``/``aipw_wls``/``ipw_ra`` methods was labeled ``.method =
  "IPW"``, regardless of which method actually produced it. Each method now
  labels its own result correctly. :pr:`10164`
- ``BinomialBayesMixedGLM.fit``/``PoissonBayesMixedGLM.fit`` (documented as
  equivalent to ``fit_map``) called ``fit_map`` and discarded its return
  value, so ``.fit()`` always returned ``None`` instead of the fitted
  results instance -- any code using the documented ``.fit()`` entry point
  (rather than calling ``.fit_map()`` directly) could not get a usable
  result. :pr:`10195`
- The state space univariate filter/smoother (used for exact diffuse
  initialization, and as the automatic fallback whenever the multivariate
  filter hits a singular forecast-error covariance) computed the smoothed
  measurement disturbance in a whitened basis and never transformed it
  back, so ``smoothed_measurement_disturbance`` was wrong by an
  observation-dependent factor for any model that exercised this code
  path -- off by as much as 124 in one of the affected test cases. The
  corresponding disturbance *covariance* cannot be recovered the same way
  from what the univariate recursions compute, so that quantity now raises
  a warning instead of silently returning a value in the wrong basis.
  :pr:`9979`
- ``GLS.hessian_factor`` returned incorrect values for both non-scalar
  ``sigma`` cases: for a 1-d (heteroskedastic-weights) ``sigma`` it
  returned the whitening factor ``1 / sqrt(sigma)`` instead of the
  Hessian weight ``1 / sigma`` (:pr:`10196`), and for a full 2-d
  (non-diagonal) ``sigma`` its output does not correspond to the actual
  Hessian at all; rather than continue to return a plausible-looking but
  wrong answer, the 2-d case now raises ``NotImplementedError``
  (:pr:`10203`, see *Breaking Changes and Deprecations* below).
- In the non-IRLS gradient-optimizer path of ``GLM.fit``, the fallback that
  is supposed to reuse ``normalized_cov_params`` when the observed Hessian
  cannot be inverted was unreachable dead code, so ``bse``/``cov_params()``
  silently came back as all-``NaN`` any time the Hessian inversion failed,
  even though a usable covariance estimate from the optimizer was
  available. :pr:`9794`
- ``LikelihoodModel.fit(method="newton")`` used the opposite sign
  convention from every other optimizer for its internal score/Hessian
  closures. This made no difference to the Newton step itself, but it
  meant the ``ridge_factor`` Hessian regularization (used to stabilize
  the solve when the Hessian is poorly conditioned) was applied with the
  wrong sign -- shrinking the regularized Hessian's magnitude instead of
  increasing it, the opposite of what regularization is supposed to do.
  This is most consequential for models fit with a non-default
  ``ridge_factor`` or a near-singular Hessian. :pr:`10184`
- ``Tweedie`` GLM log-likelihood (``1 < var_power < 2``, the compound
  Poisson-Gamma case commonly used for claim-severity/insurance-style
  data) computed ``log(wright_bessel(...))``, which overflows to ``inf``
  before the log is taken for a range of realistic ``endog``/``mu``/scale
  combinations, silently producing an infinite or garbage
  log-likelihood. Fixed by using ``scipy.special.log_wright_bessel``
  directly, which does not have this overflow. Requires SciPy >= 1.14 to
  take effect; on older SciPy (or 32-bit platforms, where
  ``log_wright_bessel`` is not accurate enough) the previous,
  overflow-prone computation is still used. :pr:`10179`, :pr:`10186`,
  :pr:`10188`


Breaking Changes and Deprecations
===================================

Previously-silent wrong results now raise or warn
--------------------------------------------------------

A few of the correctness fixes described above change what a call does,
not just the numbers it returns, because the previous behavior had no
correct fallback:

- ``GLS.hessian_factor`` (and anything built on it, e.g. ``GLS.hessian``)
  raises ``NotImplementedError`` for a full 2-d (non-diagonal) ``sigma``,
  instead of silently returning a value that does not correspond to the
  actual Hessian. The 1-d (heteroskedastic-weights) and scalar ``sigma``
  cases are unaffected and continue to work. :pr:`10203`
- The state space simulation smoother's smoothed measurement disturbance
  *covariance* (as opposed to the disturbance itself, which is now
  computed correctly, see above) cannot be recovered in the original
  basis from what the univariate filter/smoother computes, so requesting
  it now raises a warning instead of silently returning a value in the
  wrong basis. :pr:`9979`
- ``psturng`` (the studentized range p-value approximation underlying
  Tukey's HSD and the Games-Howell test) raises ``ValueError`` for degrees
  of freedom ``1 <= v < 2`` combined with a very small p-value, instead of
  returning a fabricated ``0.1``. Neither R's ``ptukey`` nor the
  literature this implementation follows supports a real computation in
  that region. :pr:`7327`
- ``MixedLM.fit``'s warning for keyword arguments it does not recognize
  changed from ``RuntimeWarning`` to ``FutureWarning``, and now states
  that a future version will raise instead of dropping the argument.
  Code that specifically filters ``RuntimeWarning`` to silence this
  message will need to filter ``FutureWarning`` instead. :pr:`9695`

``seed``/``random_state`` -> ``rng`` (SPEC-007)
---------------------------------------------------

As described above, wherever a function or model previously accepted
``seed`` or ``random_state`` to control randomness, it now accepts ``rng``
instead. The old keyword names still work but emit a ``FutureWarning``
pointing at ``rng``; they will be removed in a future release. If your code
passes ``seed=`` or ``random_state=`` by keyword to statsmodels functions,
you should switch to ``rng=`` to avoid the warning (and future breakage).
Positional usage is unaffected in most cases since ``rng`` occupies the same
position the old keyword did.

Variable-length tuple returns become NamedTuples
----------------------------------------------------

As described above, functions that returned a tuple whose length depended on
their arguments are moving to fixed-shape ``NamedTuple`` results. Where the
``NamedTuple`` unpacks exactly like the tuple it replaces there is nothing to
do: existing code keeps working and no warning is raised.

Where adopting it *would* change how many values are unpacked, the affected
call now emits a ``FutureWarning`` and continues to return the legacy tuple.
This applies to:

- :func:`~statsmodels.tsa.stattools.adfuller`,
  :func:`~statsmodels.tsa.stattools.kpss`,
  :func:`~statsmodels.tsa.stattools.range_unit_root_test` (``store=False``),
  and :func:`~statsmodels.tsa.stattools.acf` when only one of ``qstat`` or
  ``alpha`` is given.
- :func:`~statsmodels.regression.linear_model.yule_walker` (``inv=False``)
  and ``OLSResults.el_test``.
- :func:`~statsmodels.stats.diagnostic.acorr_lm`,
  :func:`~statsmodels.stats.diagnostic.acorr_breusch_godfrey`,
  :func:`~statsmodels.stats.diagnostic.het_arch`,
  :func:`~statsmodels.stats.diagnostic.compare_cox`,
  :func:`~statsmodels.stats.diagnostic.compare_j` and
  :func:`~statsmodels.stats.diagnostic.het_goldfeldquandt` with
  ``store=False``.
- :func:`~statsmodels.nonparametric.kde.kdensity` and
  :func:`~statsmodels.nonparametric.kde.kdensityfft` with ``retgrid=False``,
  and :func:`~statsmodels.graphics.regressionplots.plot_partregress` with
  ``ret_coords=False``.

Pass ``result_object=True`` to adopt the new result now, or
``result_object=False`` to keep the old return type and silence the warning.
The default becomes the ``NamedTuple`` in 0.16.

``RegressionResults.compare_lr_test`` always returned three values, so it was
converted directly to a ``CompareLRTestResult`` with no deprecation period; it
still unpacks as a three-tuple.

``HolderTuple`` deprecated
------------------------------

``statsmodels.stats.base.HolderTuple``, used internally as the return type
for many statistical tests before the ``NamedTuple`` migration above, is now
deprecated and will be removed after statsmodels 0.16. It is no longer
constructed anywhere internally. Code that checked
``isinstance(result, HolderTuple)`` or relied on ``HolderTuple``'s specific
2-tuple-unpacking behaviour should switch to the documented ``NamedTuple``
result class and named attribute access (e.g. ``result.statistic``,
``result.pvalue``) instead. :pr:`10072`

Undocumented ``alternative`` short forms deprecated
--------------------------------------------------------

As described above (*Stricter input validation for string-valued options*),
short, undocumented spellings of the ``alternative`` hypothesis-direction
parameter (``"2s"``, ``"l"``, ``"s"``, ``"i"``, ``"inc"``, ``"d"``, ``"dec"``,
``"2"``, and a few compare/statistic aliases in
:mod:`~statsmodels.stats.meta_analysis` and
:mod:`~statsmodels.stats._lilliefors`) now raise a ``FutureWarning`` naming
the documented replacement instead of working silently, and will be removed
after statsmodels 0.16. :pr:`10170`, :pr:`10173`, :pr:`10180`

Several previously-undocumented, silently-accepted string values elsewhere
were similarly tightened to raise ``ValueError`` for anything outside the
documented set -- this is a validation fix, not a deprecation, so there is no
warning period; code passing a value outside the documented ``{...}`` set
needs to be corrected directly. :pr:`10161`, :pr:`10167`

Unused estimator classes deprecated
-----------------------------------------

The following classes and one function were found, during a systematic
coverage audit, to have no callers anywhere in the codebase and no test
coverage. They now raise a ``FutureWarning`` on construction/use and will be
removed after statsmodels 0.16: ``NonlinearLS``, ``MLEGLS``, ``TSMLEModel``,
``GLSHet``, ``GLSHet2``, ``TsaDescriptive``, and the ``_Var`` class in
``tsa.varma_process`` (whose own docstring already called it "Obsolete").
``nonparametric.smoothers_lowess_old.lowess`` gets the same treatment as a
function -- its own docstring examples already point at the actively
maintained :func:`statsmodels.nonparametric.lowess`. If you rely on any of
these, please open an issue. :pr:`10156`

``FactorResults.uniq_stderr`` is now a method, not a property
--------------------------------------------------------------------

``FactorResults.uniq_stderr`` previously accepted a documented ``kurt``
argument that could never actually be supplied, because the method was
wrapped in ``@cache_readonly`` and so was only ever accessed as a bare
attribute (``result.uniq_stderr``). The ``cache_readonly`` wrapper has been
removed so ``kurt`` is usable as documented; this means existing code must
change ``result.uniq_stderr`` to ``result.uniq_stderr()``. There is no
deprecation period for this one, since the old attribute-style access could
never have supplied ``kurt`` correctly in the first place. :pr:`10175`

Minimum dependency versions raised
--------------------------------------

- NumPy: 1.18 -> 1.23.5
- SciPy: 1.4 -> 1.8
- pandas: 1.0 -> 1.4
- patsy: 0.5.2 -> 0.5.6
- ``formulaic``: new required runtime dependency, >=1.1.0
- Building from source now requires NumPy >= 2.0, SciPy >= 1.13, and
  Cython >= 3.0.13 (see the meson-python migration above). This does not
  affect users installing wheels from PyPI.

Deprecated parameters removed entirely
------------------------------------------

The following previously-*deprecated* (not previously-working) parameters
and behaviors were removed as part of a general deprecation clean-up
(:pr:`9936`):

- ``grangercausalitytests``: the ``verbose`` parameter (deprecated since
  0.14) has been removed. The function no longer prints results; use the
  returned dictionary as before.
- ``AutoReg``/``ar_select_order``: the ``old_names`` parameter (pre-0.12
  variable naming, deprecated since 0.13) has been removed.
- ``kpss``: passing ``nlags=None`` now raises a ``ValueError`` instead of
  warning and silently falling back to ``'auto'``. Pass ``'auto'``,
  ``'legacy'``, or an explicit integer.
- A number of internal compatibility shims for very old NumPy/SciPy/Python
  versions were removed from ``statsmodels.compat``, including
  ``compat.numpy.lstsq``, ``NP_LT_114``, ``compat.python.asstr``,
  ``asunicode``, ``lfilter``, and ``compat.scipy.SP_LT_16``/``SP_LT_17``
  (along with the vendored ``multivariate_t`` fallback they guarded). These
  were internal implementation details, not public API, but could have been
  imported directly.

Vendored pandas private APIs
--------------------------------

pandas has been privatizing or removing several small utilities that
statsmodels relied on (``cache_readonly``, ``deprecate_kwarg``,
``Appender``, ``Substitution``). statsmodels now vendors its own copies of
these (in ``statsmodels.compat.pandas`` and
``statsmodels.tools.docstring_helpers``) so behavior stays stable across
pandas versions, including pandas 3. :pr:`9615`, :pr:`9820`, :pr:`9831`

Other removals
-------------------

- The long-empty ``statsmodels.interface`` package was removed. :pr:`9721`
- ``_lazywhere`` was removed in favor of ``apply_where``. :pr:`9543`
- ``scipy.interpolate.interp2d`` (removed upstream in recent SciPy) is no
  longer relied on by ``TableDist``. :pr:`9832`


New Features and Enhancements
================================

.. rubric:: Enhancements

- Outlier-robust covariance estimation. :pr:`8129`
- ``ccf`` can optionally return confidence intervals. :pr:`8782`
- Plot cross-correlations and the auto/cross-correlation matrix. :pr:`8783`
- Plot the prediction curve over a scatter plot in
  ``GLMGamResults.plot_partial``. :pr:`8881`
- Add ``get_margeff`` to GLM. :pr:`8889`
- Add ``MultivariateLS``. :pr:`8919`
- Faster computation of state space revision impacts. :pr:`8937`
- Two-sample z-test, unequal-variances case. :pr:`8959`
- Improve lag selection in ``pacf``. :pr:`9016`
- Add Cython 3 compatibility. :pr:`9078`
- GLM models now save the names of input pandas Series. :pr:`9130`
- Robust: additional tools and norms. :pr:`9186`
- Add ``CovDetMCD``, ``CovDetMM``, ``RLMDetSMM``, and related estimators. :pr:`9227`
- Add a ``"one-sided"`` alternative for ``proportion_confint``. :pr:`9249`
- Add an alternative option to ``confint_poisson``. :pr:`9255`
- Add optional parameters to ``summary_col`` to indicate fixed effects. :pr:`9280`
- Ensure returned arrays are owned (not views). :pr:`9334`
- Improve precision of a diagnostic printout (``mean_diff:.3g``). :pr:`9388`
- Add the Leybourne-McCabe stationarity test. :pr:`9399`
- Add a sample-size calculation for Wilcoxon/Mann-Whitney tests. :pr:`9401`
- More reliable casting of pandas data. :pr:`9407`
- Add an abstracted formula engine supporting ``patsy`` and ``formulaic``. :pr:`9423`
- Add ``ruff`` lint support. :pr:`9453`
- ``x13_arima_analysis`` can produce seasonality fit diagnostics. :pr:`9498`
- Allow the ARDL model to use a ``"ctt"`` trend. :pr:`9518`
- Add plot keyword arguments to ``qqplot_2samples``. :pr:`9544`
- ``x13_arima_analysis`` gained an optional raw spec parameter. :pr:`9550`
- Support array-like and pandas-like data more broadly. :pr:`9582`
- Add a "no cross terms" option to White's heteroscedasticity test. :pr:`9691`
- Add missing attributes to ``AutoReg``. :pr:`9750`
- Add a seasonal diagnostic plot to ``graphics.tsaplots``. :pr:`9787`
- Make ``tsa.statespace`` Cython usage compatible with SciPy ILP64 builds. :pr:`9798`
- Allow seasonal-differencing-only models with non-seasonal estimators. :pr:`9811`
- Add ``add_ellipse`` to graphics, and support passing ``x``/``y`` arrays. :pr:`9815`
- Add order validation to the Hannan-Rissanen estimator. :pr:`9819`
- Vendor ``Appender`` and ``Substitution`` docstring helpers from pandas. :pr:`9820`
- Vendor ``cache_readonly`` and ``deprecate_kwarg`` from pandas' private API. :pr:`9831`
- Report the last root-finder value in the ``solve_power`` convergence warning. :pr:`9885`
- Consistently use ``rng`` to move towards SPEC-007. :pr:`9950`
- Add the partial cross-correlation function ``pccf`` and ``plot_pccf``. :pr:`9802`
- Add the Hamilton filter. :pr:`9957`
- Add a delete-k (block) jackknife estimator. :pr:`10001`
- Allow pre-calculated error bands to be passed to the IRF plots. :pr:`9816`
- Support ``fixed_params`` in ``innovations_mle``. :pr:`9845`
- Raise an informative error for impossible one-sided ``solve_power`` cases. :pr:`9895`
- Add a ``min_diag`` option to ``cov_nearest`` for zero or negative
  diagonal entries. :pr:`9898`
- ``acf``/``pacf`` accept a list of lags in addition to ``maxlag``. :pr:`10016`
- Return ``NamedTuple`` results in place of variable-length tuples. :pr:`10025`,
  :pr:`10027`, :pr:`10029`, :pr:`10030`, :pr:`10035`, :pr:`10072`
- Accept Polars ``DataFrame``/``Series`` input in models and the formula
  API. :pr:`9804`
- Add the Jonckheere-Terpstra ordered trend test. :pr:`9874`
- Add the Diebold-Mariano test of equal predictive accuracy. :pr:`10066`
- Add the Pesaran-Timmermann test of directional predictive accuracy. :pr:`10055`
- Add local false discovery rate estimation (``local_fdr_correction``). :pr:`10069`
- Add ``LocalProjections``, a Jordà (2005) local-projections estimator for
  impulse response functions with Newey-West HAC standard errors. :pr:`9871`
- Implement an L1-penalized solver for GLM. :pr:`10101`
- Add CRV3 (cluster-jackknife) cluster-robust inference for ``OLS``/
  ``WLS``. :pr:`10103`
- Warn when ``exog`` is (numerically) singular in the ``*LS`` model
  family, instead of silently returning an unreliable fit. :pr:`10140`
- Make the ``ndim`` check in ``array_like`` orthogonal to ``maxdim``, so
  the two can be combined instead of one silently overriding the
  other. :pr:`10090`
- ``NominalGEE`` accepts non-numeric ``groups`` labels (for example
  strings), instead of failing to cast them to ``float64``
  internally. :pr:`10182`
- Robust linear model (``RLM``) scale-estimator callables passed via
  ``scale_est`` may now optionally accept the fitted model and residuals,
  in addition to the previously-supported single-argument (residuals
  only) form, which continues to work unchanged. :pr:`10191`
- Add ``fit_regularized`` to ``HurdleCountModel``. :pr:`10204`

.. rubric:: Performance

- Optimize VECM memory/speed by avoiding an :math:`O(T^2)` projection matrix. :pr:`9720`
- Improve the performance of ``ConditionalMNLogit``. :pr:`9036`
- Add an :math:`O(N \log N)` algorithm for ``medcouple``. :pr:`9571`


Notable Bug Fixes
====================

- Make ``MICEData`` iterable; each iteration step advances the MICE chain by
  one update cycle and yields the current imputed dataset. :issue:`7110`
- Fix a typo in the ``InfeasibleTestError`` exception string. :pr:`8878`
- Correct diagnostics for changes in pandas. :pr:`8887`
- MNLogit Wald tests: fix ``ravel``, string ``cov_names``. :pr:`8907`
- Fix writing a read-only array under pandas 2 copy-on-write. :pr:`8942`
- Fix an issue in ``seasonal.py``. :pr:`9029`
- Ensure ARIMA simulation is reproducible. :pr:`9165`
- Fix ``scale.Huber`` and add a robust M-scale. :pr:`9210`
- Correct ``cov_kwargs`` -> ``cov_kwds``. :pr:`9240`
- Ensure the Zivot-Andrews test does not overwrite its input. :pr:`9311`
- Avoid an in-place modification bug. :pr:`9385`
- Correct ``resid`` from ``UECM``. :pr:`9390`
- Correct the x/y label location in ``qqplot_2sample``. :pr:`9394`
- Remove an incorrect ``method`` assignment in GLM's ``summary2``. :pr:`9396`
- Ensure the Hessian is skipped where appropriate. :pr:`9398`
- Correct the log-likelihood computation for ``ETSModel``. :pr:`9400`
- Ensure VAR can forecast with 0 lags. :pr:`9413`
- Correct ``DatetimeIndex`` handling. :pr:`9457`
- Correct handling of ``PeriodIndex`` in ``seasonal_decompose``. :pr:`9461`
- SVAR: fix ``A``/``B`` dtype and a one-parameter score shape bug. :pr:`9468`
- Fix formula ``eval`` depth in model selection. :pr:`9471`
- Tukey's HSD: fix an unused variance and add Games-Howell for the
  unequal-variance case. :pr:`9487`
- Fix a bug in ``Runs.runs_test`` for the case of a single run. :pr:`9524`
- Make the Binomial family more robust to the corner case ``mu=0``,
  ``endog=0``. :pr:`9581`
- Fix the ``add_trend`` error message to correctly identify constant columns. :pr:`9636`
- Fix conversion of 1-d arrays to scalars. :pr:`9673`
- Fix a state space model transition-timing bug. :pr:`9688`
- Pass ``alpha`` through to ``plot_predict``. :pr:`9728`
- Fix an incorrect length comparison in endpoint transformation logic. :pr:`9729`
- Fix compilation errors in ``statespace/meson.build``. :pr:`9738`
- Fix patsy ``eval_env`` handling in ``FormulaManager``. :pr:`9739`
- Raise an error for invalid ``endog`` input in ``emplike.DescStat``. :pr:`9747`
- Add an informative error message when Hessian inversion fails in
  ``fit_regularized``. :pr:`9757`
- Replace bare ``except`` clauses with ``except Exception``. :pr:`9758`
- Treat empty docstrings as ``None`` in the ``Docstring`` class. :pr:`9773`
- Fix ``use_boxcox`` control flow in ``ExponentialSmoothing.fit``. :pr:`9797`
- Override the ``resid`` property in ``UECMResults``. :pr:`9812`
- Avoid a division by zero in ``estimate_location``. :pr:`9814`
- ``L-BFGS-B``: respect ``disp=False`` instead of always printing output. :pr:`9823`
- Remove a dead assignment to ``cov_p`` in GLM's ``fit``. :pr:`9826`
- Fix the ``GLMInfluence.hat_matrix_diag`` method name. :pr:`9830`
- Fix VIF numerical instability by standardizing the design matrix. :pr:`9835`
- Skip summary diagnostics when ``slim=True``. :pr:`9844`
- Fix ``anova_lm`` silently returning ``NaN`` p-values when models are passed
  in reverse order. :pr:`9852`
- Set ``k_exog_user`` on ``SVARResults`` so ``summary()`` works. :pr:`9853`
- Fix ``Binomial.deriv()`` to correctly return ``1 - 2*mu/n`` (it was
  missing the division by ``n``). :pr:`9862`
- Record the robust scale in ``RLM.fit_history``. :pr:`9866`
- Fix the ``NegativeBinomial`` check for the optional ``alpha`` parameter. :pr:`9877`
- Return ``nan`` from ``Power.solve_power`` when it fails to converge,
  rather than a misleading value. :pr:`9884`
- Correct several parameter names in docstrings (``prob_infl``,
  ``bin_edges``, ``pred_kwds``, ``param_nums``, ``mu1_low``). :pr:`9886`
- Fix ``DiscreteResults`` crashing with ``full_output=0``. :pr:`9887`
- Fix an ``ccovf`` shape mismatch for arrays of different lengths. :pr:`9888`
- ``describe``/``Description`` now handle a 0-row (empty) input gracefully. :pr:`9899`
- Fix an issue with random generation. :pr:`9901`
- Attach ``mlefit`` attributes to the results instance so they appear in
  ``dir()``. :pr:`9902`
- Do not pass ``hess`` to ``L-BFGS-B``/``TNC`` in ``_fit_minimize``, which
  do not accept it. :pr:`9908`
- Read the entropy integration limits from the kernel. :pr:`9919`
- Populate ``_retain_cols`` in ``out_of_sample`` without requiring a prior
  ``in_sample`` call. :pr:`9920`
- Correct a test that relied on the removed random-state singleton. :pr:`9924`
- Fix an import failure when matplotlib is not installed. :pr:`9925`
- Unify ``group_sums`` orientation and fix ``group_demean``. :pr:`9933`
- Fix the ``scale`` attribute and ``resid_pearson`` for a fixed-scale
  ``cov_type``. :pr:`9824`
- Pass ``ax`` through to ``dot_plot`` in ``CombineResults.plot_forest``. :pr:`9829`
- Filter unsupported keyword arguments in ``MixedLM.fit`` instead of raising
  an ``AttributeError``. :pr:`9906`
- Fix a Sison-Glaz confidence-interval failure for small or sparse
  counts. :pr:`9909`
- Fix the removal of the ``compat`` ``lstsq`` shim. :pr:`9958`
- Raise on non-2x2 tables in ``stats.mcnemar``. :pr:`9974`
- Respect caller warning filters in the discrete ``fit_regularized`` (l1)
  path. :pr:`9976`
- Reject ``None`` in ``string_like`` and ``array_like`` unless
  ``optional=True``. :pr:`9985`, :pr:`9987`
- Do not re-validate the specification when extending SARIMAX results, so an
  ``exog`` constant column no longer blocks ``extend``. :pr:`9992`
- ``score_test`` returns a documented ``NamedTuple`` result rather than a
  plain tuple (see the *NamedTuple return values* highlight
  above). :pr:`9993`, :pr:`10072`
- Select the correct axis in ``drop_missing``. :pr:`9994`
- Ensure ``AutoReg`` (and other) ``summary()`` calls still work after
  ``remove_data()``. :pr:`10002`, :pr:`10009`
- Report the correct accepted types in ``dict_like``. :pr:`10005`
- Clip Wilson ``proportion_confint`` bounds to ``[0, 1]``. :pr:`10010`
- Give ``sign_test`` a clear error when every observation ties with
  ``mu0``. :pr:`10012`
- ``multipletests`` no longer raises ``ZeroDivisionError`` on an empty
  p-value array. :pr:`10013`
- ``maxabs`` and ``iqr`` no longer raise on empty input, matching the other
  ``eval_measures``. :pr:`10014`
- Use the non-missing sample size for the ``acf`` confidence interval and
  Q-statistic when NaNs are handled. :pr:`10017`
- Raise an explicit error rather than dividing by zero in
  ``acf``/``acovf``. :pr:`10020`
- ``linear_rainbow(..., use_distance=True)`` now centers on the exog
  centroid, so the result no longer depends on the arbitrary order
  observations happen to be stored in. :pr:`9903`
- ``ARDLResults.apply``/``append`` lost the per-variable exog lag order,
  because they inherited ``AutoRegResults.apply``, which always
  reconstructs the cloned model as a plain ``AutoReg``. :pr:`9915`
- The adjusted ``ccovf`` divided by ``len(x) - k`` instead of the actual
  number of overlapping observation pairs. :pr:`9916`
- ``wald_test_terms`` now reports the rank-adjusted degrees of freedom for
  rank-deficient models instead of the raw constraint-matrix row
  count. :pr:`9907`
- Cast the ``np.repeat`` argument to platform ``intp`` size in the
  Jonckheere-Terpstra test so it works on 32-bit platforms (Pyodide). :pr:`10075`
- ``breakvar_heteroskedasticity_test`` (and the state space and ETS
  ``test_heteroskedasticity`` methods built on it) referred the raw ratio of
  the two sums of squares to ``F(numer_dof, denom_dof)``.  The ratio of sums
  is that ``F`` only after rescaling by ``denom_dof / numer_dof``, so the
  p-values were wrong whenever missing observations left the two subsets with
  different numbers of usable residuals -- for example a multivariate state
  space model with a ragged edge.  The ``use_f=False`` variant had its
  multiplier and its degrees of freedom interchanged, and the ``decreasing``
  alternative did not swap the degrees of freedom when it inverted the
  statistic.  Balanced samples, which is the usual case, are unaffected.
- Fix edge cases in the :math:`O(N \log N)` ``medcouple`` path. :pr:`10084`
- Check the sign of the smallest eigenvalue before taking its square root
  when forming a condition number, instead of letting a tiny negative
  value (floating-point noise) raise. :pr:`10088`
- Fix ``MNLogit.resid_response`` raising ``ValueError`` instead of
  returning residuals. :pr:`10089`
- Forward a kwarg that ``MixedLM.from_formula`` was silently dropping
  instead of passing to the superclass constructor. :pr:`10105`
- Pivot the QR factorization used in ``tools.matrix_rank``, so rank is
  computed correctly for matrices that need pivoting for numerical
  stability. :pr:`10106`
- Fix numerous small bugs in ``robust.norms``, ``RLM``, and
  ``stats.stattools``. :pr:`10113`
- Add a missing ``self`` in an ``ETSModel`` update path. :pr:`10120`
- Correct the ``distargs`` usage in ``robust.scale.scale_trimmed``. :pr:`10130`
- Fix a line-style bug in the Bland-Altman agreement plot. :pr:`10131`
- Enable the ``percentile`` option in ``_select_sigma`` for kernel
  bandwidth selection. :pr:`10132`
- Fix a sign/orientation bug (``factor.py`` reversed the intended
  direction). :pr:`10133`
- Only initialize the trend component in exponential smoothing when the
  model actually has one. :pr:`10134`
- Correct the Hessian choice in ``othermod.betareg``. :pr:`10135`
- Ensure the bar gap size is computed correctly in ``mosaic_plot``. :pr:`10136`
- Ensure ``SVAR`` raises for options it does not actually implement,
  instead of silently ignoring them. :pr:`10137`
- Fix several bugs found in a systematic full-codebase scan, including in
  ``MixedLM`` and ``stats.multivariate_tools``. :pr:`10139`
- Fix additional small bugs, including in ``iolib.table``. :pr:`10141`
- Correct the shape of the values returned by ``CanCorr``. :pr:`10143`
- Fix ``OLSInfluence._ols_xnoti`` crashing on every call. :pr:`10152`
- Fix ``RLMDetSMM.fit`` crashing with its own documented ``h=None``
  default. :pr:`10154`
- Fix ``MICEData`` using the observed-row index instead of the full index
  when building ``predict_miss_kwds``. :pr:`10163`
- Guard against ``zero_kwds=None`` in ``effectsize_2proportions``. :pr:`10165`
- Fix a crash in SARIMAX time-varying regression when the state vector
  also includes differencing. :pr:`10172`
- Coerce the ``offset`` argument with ``array_like`` in
  ``PoissonZiGMLE``, instead of failing on plain Python
  sequences. :pr:`10174`
- Coerce ``cov_null`` with ``array_like`` in ``stats.multivariate``
  instead of requiring a NumPy array. :pr:`10176`
- ``get_prediction`` for GLM-like models now always has a linear
  predictor available when one is requested. :pr:`10178`
- Correct the knot-centering computation in ``get_knots_bsplines`` for
  splines with few interior knots, where it previously produced
  incorrect (non-equally-spaced) knots or raised. :pr:`10177`
- Pass ``transformed`` through to the likelihood when computing the
  ``MarkovSwitching`` Hessian, matching ``score``. :pr:`10187`, :issue:`10148`
- ``wald_test`` (chi-square path, the default) raised ``AttributeError``
  for any results class without a ``df_resid`` attribute, such as
  ``MarkovRegressionResults``/``MarkovAutoregressionResults``, even though
  ``df_resid`` is only needed for the F-test (``use_f=True``)
  path. :pr:`9297`
- ``BinomialBayesMixedGLM.fit``/``PoissonBayesMixedGLM.fit`` always
  returned ``None`` instead of the fitted results instance (see
  *Breaking Changes and Deprecations* above). ``VariedCovStruct.summary()``
  (in ``genmod.cov_struct``) printed directly instead of returning a
  string like the other covariance-structure ``summary()``
  methods. :pr:`10195`
- ``GLS.hessian_factor`` was wrong for both non-scalar ``sigma`` cases,
  and ``ProcessMLE.covariance()`` omitted the ``exp()`` link transform on
  the scale/smoothing parameters for models not built from a formula,
  silently producing wrong (and sometimes ``NaN``, through a negative
  variance) covariance matrices. :pr:`10196`; see also :pr:`10203` and
  *Breaking Changes and Deprecations* above.
- In the non-IRLS gradient-optimizer path of ``GLM.fit``, a valid
  ``normalized_cov_params`` fallback was discarded whenever the observed
  Hessian could not be inverted, so ``bse`` came back all-``NaN`` even
  though a usable covariance estimate existed. :pr:`9794`
- The ``ridge_factor`` Hessian regularization in
  ``LikelihoodModel.fit(method="newton")`` was applied with the wrong
  sign for the "newton" branch specifically. :pr:`10184`
- Fix the ``Tweedie`` GLM log-likelihood overflowing to ``inf`` for
  ``1 < var_power < 2`` by using ``scipy.special.log_wright_bessel``
  (SciPy >= 1.14). :pr:`10179`, :pr:`10186`, :pr:`10188`
- ``psturng``/Tukey's HSD/Games-Howell: raise a clear error instead of
  returning a fabricated p-value for degrees of freedom ``1 <= v < 2``
  with an extreme statistic; also fixes wording in related error
  messages. :pr:`7327`
- ``MNLogit.score_test(exog_extra=...)`` crashed with ``AttributeError``
  because ``MNLogit`` did not implement ``score_factor``/
  ``hessian_factor``. :pr:`10185`
- ``emplikeAFT.predict`` used ``endog`` where it meant ``exog``, so
  passing new data to predict from raised or produced nonsensical
  output. :pr:`10197`
- Two contour-plotting bugs in ``emplike`` descriptive statistics:
  ``DescStatUV.plot_contour``'s default levels were in decreasing order,
  which recent Matplotlib rejects outright, and
  ``DescStatMV.mv_mean_contour`` contoured the unbounded ``-2`` log
  log-likelihood ratio against levels documented as significance levels
  instead of the already-computed p-value, making the plotted region
  degenerate. :pr:`10197`
- ``rvs_kernel``'s Beta-kernel perturbation step ignored the ``rng``
  argument and always drew from SciPy's global default state, so two
  calls with identically-seeded generators did not reproduce the same
  output. :pr:`10198`
- ``Representation.initialize_components`` raised ``TypeError`` on every
  call (missing the required ``k_states`` argument in its internal
  ``Initialization.from_components`` call). :pr:`10200`
- ``miso_lfilter`` selected the wrong output column for any number of
  input variables other than 2 or 3 (an ``IndexError`` for 1 variable,
  silently wrong output with no error for 4 or more). :pr:`10201`


Build, Packaging, and Infrastructure
========================================

- Migrate the build backend from ``setuptools``/``setup.py`` to
  ``meson-python``. :pr:`9634`
- Update minimum dependency versions (multiple passes). :pr:`9110`, :pr:`9112`
- Add experimental Pyodide/WebAssembly support and CI jobs, including fixing
  an OpenBLAS symbol error under Emscripten. :pr:`9270`, :pr:`9343`
- Avoid non-deterministic ordering in ``include_dirs`` lists (reproducible
  builds). :pr:`9296`
- Further clean-up of the build configuration. :pr:`9632`
- Generate free-threading (no-GIL) compatible Cython modules. :pr:`9717`
- Ensure the ``libm`` C math library is linked for all build targets. :pr:`9778`
- Remove the ``oldest-supported-numpy`` build workaround now that NumPy 2 is
  the floor for building from source. :pr:`9312`
- CI: add Python 3.13/3.14 (including free-threaded 3.14t) jobs, drop active
  Python 3.9 testing, and pin GitHub Actions to full commit SHAs for supply
  chain hardening. :pr:`9547`, :pr:`9656`, :pr:`9709`, :pr:`9913`, :pr:`9843`
- Routine dependency updates for GitHub Actions were kept current via
  dependabot throughout the release cycle (``actions/checkout``,
  ``actions/setup-python``, ``actions/setup-node``, ``github/codeql-action``,
  ``pypa/cibuildwheel``, ``r-lib/actions/setup-pandoc``, and
  ``ts-graphviz/setup-graphviz``) across roughly two dozen pull requests
  not individually itemized here.
- Improve the documentation-build requirements. :pr:`9949`
- Improve notebook generation. :pr:`9990`
- Add a CI run for the X-13ARIMA-SEATS tests. :pr:`10021`
- Add a lint-only CI workflow (``ruff`` + ``flake8``). :pr:`10064`
- Improve the documentation-generation CI job, and switch the X-13ARIMA-SEATS
  CI job to build with coverage and use a different binary installation
  method. :pr:`10052`, :pr:`10048`, :pr:`10051`
- Remove the coveralls integration. :pr:`10080`
- Routine dependabot bumps for ``pypa/cibuildwheel`` and
  ``actions/github-script``. :pr:`10070`, :pr:`10071`
- Also look for ``.exe``-suffixed binaries when locating the X-13ARIMA-SEATS
  executable on Windows. :pr:`10087`


Documentation
================

In addition to numerous individual typo, notebook, and docstring
corrections, this release cycle included a large, systematic effort to
bring docstrings across the codebase in line with the numpydoc standard
(module by module: ``discrete``, ``genmod``, ``stats``, ``tsa``/
``statespace``, ``base``/``compat``/``datasets``, ``graphics``,
``imputation``/``multivariate``/``nonparametric``, ``emplike``/``duration``,
``treatment``/``gam``, ``tools``, ``othermod``/``regression``/``robust``,
and more), plus a documentation theme change to ``pydata-sphinx-theme`` and
a pass over example notebooks to fix formatting and broken links. A second,
final pass in the closing weeks of the cycle brought the remaining modules
up to the same standard and fixed up the stragglers it turned up along the
way: tools (:pr:`10107`), robust (:pr:`10108`), stats (:pr:`10110`),
othermod/treatment/multivariate (:pr:`10111`), base/datasets/compat
(:pr:`10112`), regression (:pr:`10114`), formula/graphics/imputation
(:pr:`10116`), core ``tsa`` routines (:pr:`10117`),
discrete/duration/gam/genmod (:pr:`10119`),
distributions/emplike/iolib/miscmodels (:pr:`10121`), nonparametric
(:pr:`10123`), vector_ar (:pr:`10124`), statespace (:pr:`10127`), and
dataset docstrings (:pr:`10128`), plus general clean-up of ``numpy``/
``pandas`` usage (:pr:`10115`), ``rng`` parameter docstrings (:pr:`10145`),
and the ``AGENTS.md`` guidance used to drive this pass (:pr:`10125`).

- Correct links to notebooks. :pr:`8886`
- Correct a typo in the ``WLS.loglike`` docstring. :pr:`8900`
- Add install instructions for the nightly build. :pr:`8941`
- Correct the signature of ``CopulaDistribution``. :pr:`8946`
- Fix an inconsistency in ``var_model.py``. :pr:`8948`
- Fix inclusion of plots in the docs. :pr:`8963`
- Include the correct plot in ``scatter_ellipse`` docs. :pr:`8974`
- Various small typo fixes. :pr:`9011`, :pr:`9082`, :pr:`9192`, :pr:`9208`,
  :pr:`9285`, :pr:`9397`, :pr:`9462`, :pr:`9532`, :pr:`9558`, :pr:`9626`,
  :pr:`9848`, :pr:`9850`, :pr:`9873`, :pr:`9941`
- Fix broken plots/content in ``linear_regression_diagnostics_plots``. :pr:`9158`
- Fix interaction and other example notebooks. :pr:`9216`, :pr:`9218`,
  :pr:`9551`, :pr:`9552`, :pr:`9554`, :pr:`9617`, :pr:`9621`, :pr:`9683`,
  :pr:`9718`, :pr:`9724`, :pr:`9784`, :pr:`9864`
- Update the ``ztest``/``ztest_mean`` p-value description. :pr:`9226`
- Improve documentation for regression diagnostics, stats, and summary. :pr:`9230`
- Generate docs for ``plot_ccf`` and ``plot_accf_grid``. :pr:`9299`
- Fix documentation of ``AutoReg``. :pr:`9310`
- Add a ``CITATION`` file. :pr:`9346`
- Improve documentation of ``acf`` and ``plot_acf``. :pr:`9348`
- Clarify notation for the error term in the regression docs. :pr:`9361`
- Fix docstring formula display in the SVAR class. :pr:`9372`
- Improve docs for ``ExponentialSmoothing`` and related places. :pr:`9391`
- Update the mediation tutorial documentation. :pr:`9422`
- Remove an empty cell from an ARMA example notebook. :pr:`9483`
- Fix a broken link to a citation reference. :pr:`9561`
- Document currently supported Python versions. :pr:`9588`
- Fix the Gamma ``loglike_obs`` docstring and clarify weight
  parameterization; align Gamma/Negative-Binomial notation in the GLM
  families table. :pr:`9660`, :pr:`9890`, :pr:`9892`, :pr:`9893`
- Fix a broken academic reference in ``anova.py``. :pr:`9749`
- Fix an import in the api-structure page. :pr:`9755`
- Add the seasonal diagnostic plot to the docs. :pr:`9788`
- Correct the ``PredictionResults.conf_int`` docstring. :pr:`9813`
- Fix incorrect parameter names in ``deconvolve``, ``powerdiscrepancy``, and
  ``VECMResults.predict`` docstrings, and fix formula rendering in
  ``powerdiscrepancy``. :pr:`9838`, :pr:`9839`
- Switch the documentation theme to ``pydata-sphinx-theme``. :pr:`9861`
- Improve math formulas in ``robust.norms`` docstrings. :pr:`9876`
- Add missing ``PoissonResults``/``NegativeBinomialPResults`` to the
  discrete-models autosummary. :pr:`9914`
- Systematic docstring fixes by module: discrete (:pr:`9929`), genmod
  (:pr:`9930`), stats (:pr:`9931`), tsa/statespace (:pr:`9934`), base/compat/
  datasets (:pr:`9935`), graphics (:pr:`9937`), imputation/multivariate/
  nonparametric (:pr:`9938`), othermod/regression/robust (:pr:`9940`), tools
  (:pr:`9945`), statespace (:pr:`9946`), emplike/duration (:pr:`9943`),
  treatment/gam (:pr:`9944`).
- Update notebooks for the deprecations introduced in this release. :pr:`9939`
- Improve the ``robust.norms`` docstrings. :pr:`9766`
- Add an ARIMA tutorial notebook. :pr:`9792`
- Add a plot for the Hamilton filter. :pr:`9991`
- Add this release note. :pr:`9951`
- Many small documentation fixes, including for the new notebook and the
  ``STL`` docstring. :pr:`9952`, :pr:`9954`, :pr:`9960`, :pr:`9961`
- Fix the ``NegativeBinomialP.fit`` docstring, notebook title levels, and a
  misplaced reference. :pr:`9962`, :pr:`9963`
- Allow all notebooks to run again. :pr:`9955`
- Document that ``exog`` is matched by position for non-formula models. :pr:`9967`
- Remove docstring sections that did not render correctly. :pr:`9969`
- Use HTTPS for the MixedLM reference, clarify the ``add_constant``
  ``prepend`` default, fix the ANOVA example link, and list all GEE
  covariance structures. :pr:`9996`, :pr:`9997`, :pr:`9999`, :pr:`10000`
- Correct the ``recipr0`` summary line and the discrete results
  parameters. :pr:`10006`, :pr:`10011`
- Remove five documented parameters that are not in the signature. :pr:`10028`
- Add numpydoc ``Parameters`` sections to the new ``NamedTuple`` result
  classes. :pr:`10031`
- Add an AI-use policy for contributions, and an ``AGENTS.md`` for AI coding
  agents. :pr:`10045`, :pr:`10078`
- Move ``README.rst`` to ``README.md``. :pr:`10079`, :pr:`10081`
- Clarify the ``anova_lm`` Type I/II/III sums-of-squares documentation. :pr:`9309`
- Add an explanation of the Benjamini-Hochberg procedure to the
  ``fdrcorrection`` docstring. :pr:`4216`
- Correct typos in the Hurdle Count Model example notebook. :pr:`9477`
- Fix the ``statsmodels.family`` -> ``statsmodels.families`` submodule name
  in the docs. :pr:`7568`
- Reorganize and improve the ``robust.norms`` docstrings. :pr:`8975`, :pr:`10061`
- Fix the ETS simple-exponential-smoothing equations. :pr:`9484`
- Clarify ``GLMGam`` out-of-sample prediction and the ``GLSAR`` ``rho``
  argument. :pr:`9998`, :pr:`10047`
- Various small documentation and rst fixes. :pr:`10033`, :pr:`10034`,
  :pr:`10036`, :pr:`10037`, :pr:`10038`, :pr:`10040`, :pr:`10041`,
  :pr:`10046`, :pr:`10053`, :pr:`10057`, :pr:`10062`, :pr:`10063`
- Clarify how to access ``TukeyHSD`` rejection decisions and
  p-values. :pr:`9956`
- Improve the ``yule_walker`` documentation. :pr:`10076`
- Reduce Sphinx cross-reference noise/warnings. :pr:`10097`
- Fix a typo in the WLS example notebook's row labels, and remove an
  unused ``scipy`` import and cell left over from
  it. :pr:`10099`, :pr:`10100`
- Fix incorrect parameter types recorded in the regression
  docstrings. :pr:`10104`
- Fix the ``UECM`` docstring. :pr:`10118`
- Replace broken OECD glossary links in the ``endog``/``exog``
  documentation. :pr:`10122`
- Improve the ``pacf`` docstring. :pr:`10169`


Testing, Linting, and Maintenance
=====================================

A substantial amount of routine maintenance went into keeping the test
suite green against upstream changes in NumPy, SciPy, and pandas (including
pandas copy-on-write and preparation for pandas 3), adopting ``ruff`` for
linting in addition to ``flake8``, running ``isort``/``pyupgrade`` across
the codebase, relaxing overly tight test tolerances, and improving thread
safety of the test suite ahead of free-threaded CPython support.

In the final weeks of the cycle, a systematic coverage audit went through
results-class attributes and methods, computational code paths, and
summary/table content that had no test asserting on it, adding regression
tests and turning up several of the bug fixes listed above.
:pr:`10150`, :pr:`10151`, :pr:`10153`, :pr:`10155`

Selected items:

- Reduce direct use of the global ``np.random`` state in the library and in
  tests. :pr:`9878`, :pr:`9879`, :pr:`9737`
- Prepare for pandas 3 (string dtype changes, removed features). :pr:`9245`,
  :pr:`9247`, :pr:`9602`, :pr:`9689`, :pr:`9722`
- Adopt ``ruff`` for linting. :pr:`9453`, :pr:`9642`, :pr:`9643`, :pr:`9650`
- Run ``isort`` across the codebase. :pr:`9855`
- Remove the obsolete, empty ``statsmodels.interface`` package. :pr:`9721`
- Improve thread safety of the test suite. :pr:`9742`, :pr:`9904`, :pr:`9910`
- Add CI coverage for Python 3.13/3.14 and free-threaded CPython. :pr:`9547`,
  :pr:`9656`, :pr:`9709`
- Move from ``isort`` to ``ruff`` for import sorting. :pr:`9981`
- Reduce mutation of model state inside ``fit()`` methods. :pr:`9972`
- Remove long-standing anti-patterns across ``genmod``, ``multivariate``,
  ``robust``, ``tsa``, ``stats`` and ``tools``, and extend the same
  conventions to the remaining modules. :pr:`9973`, :pr:`9977`, :pr:`9978`,
  :pr:`9980`, :pr:`9984`
- Use ``pathlib`` in place of ``os.path``. :pr:`9988`
- Remove unproductive ``if __name__ == "__main__"`` blocks, converting the
  useful ones into tests. :pr:`10023`
- Archive unused ``statsmodels.sandbox`` files and remove leftover debug
  code. :pr:`10018`, :pr:`10019`
- Remove further deprecations and outdated compatibility code. :pr:`10015`,
  :pr:`10026`
- Raise the declared Python floor to the actual minimum of 3.10, and improve
  the formula-engine specification. :pr:`9953`, :pr:`9995`
- Add tests for the ``summary()``-after-``remove_data()`` pattern across
  models. :pr:`10003`, :pr:`10007`, :pr:`10008`
- Add a marker for joblib-dependent tests and fix a test on older
  SciPy. :pr:`9948`, :pr:`10022`
- Clean up the examples and assorted lint. :pr:`9959`, :pr:`9989`
- Update the declared NumPy minimum to reflect the version actually
  required, and remove the legacy NumPy code it made unreachable. :pr:`10032`
- Reduce warning noise in the test suite (new ``filterwarnings`` entries and
  ``pytest.warns`` wrappers for warnings introduced by the ``NamedTuple``
  migration). :pr:`10068`
- Remove the now-redundant ``method`` validation in ``yule_walker`` (already
  performed by ``string_like``). :pr:`10077`
- Rename misleadingly-named WLS equivalence tests, and clean up remaining
  small issues and lint. :pr:`10039`, :pr:`10062`, :pr:`10082`
- Prefer ``pandas.read_csv`` over ``numpy.genfromtxt`` for reading example
  data. :pr:`10054`
- Protect against pandas 4 changes. :pr:`10058`, :pr:`10065`
- Improve the issue and pull-request templates. :pr:`10050`, :pr:`10060`
- Assorted small maintenance ahead of the release. :pr:`10056`
- Test the remaining edge cases in the Jonckheere-Terpstra
  test. :pr:`10083`
- Move the ``NamedTuple`` result classes away from a shared limited-iteration
  mixin, standardize field names, and simplify the mix of ``NamedTuple`` and
  ``dataclass`` usage introduced earlier in the
  cycle. :pr:`10093`, :pr:`10095`, :pr:`10096`, :pr:`10098`
- Restore a behavior change that had been introduced
  accidentally. :pr:`10094`
- Improve import performance in some cases. :pr:`10102`
- Move non-core code out of the main package. :pr:`10168`
- Re-enable a previously-skipped test, and change the warning class expected
  from ``fit_collinear`` and from tests running under
  WASM. :pr:`10138`, :pr:`10142`, :pr:`10144`
- Silence expected-but-noisy singularity warnings in the test
  suite. :pr:`10146`
- Add tests for the ``rng`` argument selector. :pr:`10147`
- Add a marker for matplotlib-dependent tests. :pr:`10166`
- CI: work around a Cython/conda incompatibility that intermittently broke
  the legacy conda test job. :pr:`10158`, :pr:`10160`, :pr:`10162`
- Add ``tools/check_public_api_coverage.py`` and
  ``tools/class_coverage_report.py``, AST-based scripts that find public
  API surface and estimation-class code with no test coverage, plus a CI
  job that runs them with a baseline so the zero-coverage set cannot grow;
  this tooling drove much of the coverage-motivated bug-hunting elsewhere
  in this release. :pr:`10189`
- Standardize fully on ``ruff`` for linting and drop ``flake8`` from CI
  and pre-commit, now that ``ruff`` covers the rules previously split
  across both tools. :pr:`10192`, :pr:`10193`
- Add further regression tests from the public-API coverage audit for
  ``statsmodels.test``, ``docstring_helpers``, ``eval_measures.stde``,
  ``moment_helpers.mnc2mvsk``, ``gof.gof_chisquare_discrete``/
  ``gof_binning_discrete``, ``RegressionFDR.threshold``,
  ``weightstats.DescrStatsW.ttost_mean``/``CompareMeans.ztost_ind``,
  ``datasets.utils.clear_data_home``, ``iolib.table.SimpleTable.pad``,
  ``GenericLikelihoodModel.reduceparams``/``nloglike``, and
  ``DistributedModel.fit_joblib``/``DistributedResults.predict``, each
  checked against an independent reference rather than only asserting no
  exception is raised. :pr:`10194`
- Add coverage for ``VARProcess``/``VARResults`` autocorrelation
  methods. :pr:`10199`
- Reduce the number of Linux CI jobs to speed up completion. :pr:`10181`
- Further ``pandas``-compatibility maintenance (``factor.py``,
  ``grouputils.py``, an ``x13`` test). :pr:`10206`
- Skip a test requiring an exact ``LinAlgError`` message on
  WASM/Pyodide. :pr:`10202`


Major Bugs Fixed
====================

See github issues for a list of bug fixes included in this release

- `Closed bugs <https://github.com/statsmodels/statsmodels/pulls?utf8=%E2%9C%93&q=is%3Apr+is%3Amerged+milestone%3A0.15+label%3Atype-bug>`_
- `Closed bugs (wrong result) <https://github.com/statsmodels/statsmodels/pulls?q=is%3Apr+is%3Amerged+milestone%3A0.15+label%3Atype-bug-wrong>`_


Development summary and credits
===================================

Thanks to everyone who contributed code, documentation, bug reports, and
review to this release cycle. The following list of contributors is
generated from ``git log`` between ``v0.15.0.dev0`` and the current
development head, and may not be complete or fully deduplicated across
differently-configured git identities:

Achraf Ez, Aditi Juneja, Adrian Ross, Agriya Khetarpal, Alex Alborghetti,
Alexander Fischer, Andrés, Andrés López, Anh Trinh, Aniket, Aniket Singh
Yadav, Anselm Hahn, Antoine Mayerowitz, Anton Karpov, Anuraag Pandhi, Artem
Glebov, Ayush Gupta, Ben, Benjamin Leff, Bortlesboat, Caleb Lindgren, Chad
Fulton, Christine P. Chai, Clément Fauchereau, Daan Knoope, David Ivanov,
Deshan, Dhairya Motta, Dhruvil Darji, Eden Rochman, Elton Chang, Erich
Morisse, Eugen Goebel, Evan Lyall, Evgeni Burovski, FuturMix, Hadi Dayekh,
Harish Bhavandla, Hood Chatham, IsaacP, IntegralIndefinida, Illia
Polovnikov, Iman, Jake Soloff, Jesse W. Collins, Jim Varanelli,
Joey Scanga, Josef Perktold, Joshua Markovic, Justin Mahlik, Kaif,
Kakarot35, Kayvan Zahiri, Kevin Sheppard, Kevin Gregory, Kumar Aditya,
Lakshmi786, Loi Nguyen, Luke J, Maciej Skorski, Manlai Amar, Marc Bresson,
Mathias Hauser, Maxime Gourguechon, Melissa Wu, Michał Górny, Michel de
Ruiter, Naimish Machchhar, Pranav Achar, Puneet Dixit, Rahul Rathnavel K,
Ralf Gommers, Rebecca N. Palmer, Ritika shrestha, RoyS, Seaic Mac
Murchadha, Sebastian Pölsterl, Shamus, Solaris-star, Sreekant Baheti,
Tartopohm, Vedant Madane, Vikram Kumar, Viktor, Vitaliy, Vladimir
Saraikin, Wali Reheman, Will Tirone, YangWu1227, Zbigniew
Jędrzejewski-Szmek, Zhang Hong, Zhengbo Wang, adarshsm, alekracicot,
camaramm, chuenchen309, cjck944084735-dot, genrichez, hass-nation, lev,
libokai, louisabraham, mkzung, star1327p, uttam12331, whn, and many
others.

These lists are automatically generated based on ``git log`` and may not be
complete.


Merged Pull Requests
-----------------------

The following Pull Requests were merged since the last release:

- :pr:`4216`: DOC: Added explanation of fdr_bh to docstring of fdrcorrection
- :pr:`7326`: BUG: Fix libsturng issue #7324
- :pr:`7568`: MAINT: Fix incorrect submodule name (statsmodels.family -> sm.families)
- :pr:`8129`: ENH: Outlier robust covariance - rebased
- :pr:`8782`: ENH/TST: ccf to optionally return confidence intervals
- :pr:`8783`: ENH: Plot cross-correlations and auto/cross-correlation matrix
- :pr:`8865`: MAINT: Move from Styler.applymap to map
- :pr:`8866`: DOC: Add admonitions for changes and deprecations
- :pr:`8867`: DEV: Start of 0.15 branch
- :pr:`8870`: TST: install missing \*.csv files needed by tsa.stl tests
- :pr:`8872`: MAINT: Add CI for install and sdist install
- :pr:`8874`: Backport of #8870 and #8872
- :pr:`8875`: TST: Relax tolerance on overly tight test
- :pr:`8876`: TST: Relax tolerance on overly tight test
- :pr:`8878`: BUG Fix typo in InfeasibleTestError exception string
- :pr:`8881`: ENH: plot prediction curve over scatter in GLMGamResults.plot_partial
- :pr:`8886`: DOC: Correct links to notebooks
- :pr:`8887`: BUG: Correct diagnostics for changes in pandas
- :pr:`8889`: ENH: add get_margeff to GLM
- :pr:`8897`: MAINT: Update for future pandas changes
- :pr:`8900`: DOC: correct typo in WLS.loglike docstring
- :pr:`8907`: BUG: mnlogit wald tests, ravel, string cov_names
- :pr:`8919`: ENH: add MultivariateLS
- :pr:`8930`: MAINT: Remove deprecated utility
- :pr:`8932`: CLN: Fix typos
- :pr:`8937`: ENH/PERF: faster computation of revision impacts
- :pr:`8939`: MAINT: Update nightly location
- :pr:`8940`: MAINT: Make changes for deprecations
- :pr:`8941`: DOC: Add install instructions for nightly
- :pr:`8942`: BUG: Writing read-only arry on pandas 2/CoW
- :pr:`8946`: DOC: correct signature of `CopulaDistribution`
- :pr:`8948`: DOC: fix inconsistency in `var_model.py`
- :pr:`8959`: ENH: 2-sample z-test unequal variances case
- :pr:`8963`: DOC: Fix inclusion of plots
- :pr:`8974`: DOC: Include correct plot in scatter_ellipse
- :pr:`8975`: DOC: docstrings in robust.norms, improve, reorganize
- :pr:`8988`: STY: Switch from == to is for type comparrison
- :pr:`8989`: MAINT: Insert some initial NumPy caps
- :pr:`8990`: MAINT: Block pandas 2.1.0
- :pr:`8992`: Bump actions/checkout from 3 to 4
- :pr:`9011`: DOC: fix small typo
- :pr:`9016`: ENH: Improve lag selection in pacf
- :pr:`9029`: Update seasonal.py
- :pr:`9036`: ENH: Improved performance of the ConditionalMNLogit class
- :pr:`9041`: Backport 0.14.1
- :pr:`9046`: Forward port
- :pr:`9059`: TST: Ensure value is float
- :pr:`9078`: ENH: Add compatability with Cython 3
- :pr:`9082`: DOC: fix typo
- :pr:`9083`: CI: Ensure non-zero exit fails
- :pr:`9086`: Bump actions/setup-python from 4 to 5
- :pr:`9087`: MAINT: Use RandomState in-place of np.random.seed
- :pr:`9088`: MAINT: Protect against future pandas changes to merge/sorting
- :pr:`9089`: MAINT: Use modern freq names
- :pr:`9092`: Backport 0.14.1
- :pr:`9098`: Bump github/codeql-action from 2 to 3
- :pr:`9101`: refactor code to drop constant columns
- :pr:`9106`: MAINT: Explore NumPy 2 compatability
- :pr:`9110`: BLD: Update minimums
- :pr:`9111`: MAINT: Fix future issues in pandas
- :pr:`9112`: Update mins v2
- :pr:`9113`: MAINT: Remove conditions producing warnings
- :pr:`9115`: MAINT: Clean up and silence some warnings
- :pr:`9116`: CI: Update pip pre to 3.12
- :pr:`9117`: edited requirements.txt
- :pr:`9124`: MAINT: Fix future issues due to array shapes
- :pr:`9126`: MAINT: Fixes for pre-release testing
- :pr:`9130`: ENH: GLM models now save the names of input Pandas Series
- :pr:`9142`: Fix linting error
- :pr:`9143`: Fix string formatting
- :pr:`9144`: MAINT: Replace quarterly string identified
- :pr:`9149`: Bump ts-graphviz/setup-graphviz from 1 to 2
- :pr:`9150`: MAINT: Fixes for future changes
- :pr:`9158`: DOC: Fix broken in `linear_regression_diagnostics_plots`
- :pr:`9165`: BUG: Ensure ARIMA simulation is reproducable
- :pr:`9186`: ENH: robust: tools and more norms
- :pr:`9192`: DOC: fixed boxpierece typos
- :pr:`9195`: MAINT: Make compatability with NumPy 2
- :pr:`9200`: Cherry pick commits from 0.15 for 0.14.3
- :pr:`9203`: DOC: Add release note
- :pr:`9208`: DOC: fixed typos init_training_endog
- :pr:`9210`: BUG/ENH: fix scale.Huber and add robust MScale
- :pr:`9212`: DOC: Final docs for 0.14.2
- :pr:`9213`: DOC: Final docs for 0.14.2
- :pr:`9216`: DOC: Fix interactions notebook
- :pr:`9218`: DOC: Fix multiple issues in notebooks
- :pr:`9226`: DOC: Update pvalue description in weightstats.py of `ztest` and `ztest_mean`
- :pr:`9227`: ENH: add CovDetMCD and det for regression
- :pr:`9230`: DOC: Improve docs of `regression_diagnostics.html`, `stats.html`, `summary`
- :pr:`9240`: BUG: Correct cov_kwargs -> cov_kwds
- :pr:`9245`: MAINT: Fix issues with pandas 3
- :pr:`9247`: MAINT: Additional fixes for pandas 3
- :pr:`9249`: added "one-sided" alternative for proportion_confint
- :pr:`9255`: ENH: add alternative option to confint_poisson
- :pr:`9262`: MAINT: Change future keyword argument
- :pr:`9270`: Add Pyodide support and CI jobs for `statsmodels`
- :pr:`9280`: ENH: Add optional parameters for summary_col to indicate FEs (rebased)
- :pr:`9285`: DOC: Replace `postive` by `positive`
- :pr:`9291`: REF: Remove numpy testing import from test runner
- :pr:`9292`: MAINT: Update requirements
- :pr:`9296`: Avoid random ordering in include_dirs lists
- :pr:`9299`: DOC: Generate docs for plot_ccf and plot_accf_grid
- :pr:`9309`: DOC: Add explanation of `typ` I II III of `anova_lm`
- :pr:`9310`: DOC: Fix documentation of `statsmodels.tsa.ar_model.AutoReg`
- :pr:`9311`: BUG: Ensure ZA does not overwrite
- :pr:`9312`: MAINT: Remove oldest-supported-numpy
- :pr:`9334`: ENH/BUG: Ensure array is owned
- :pr:`9336`: MAINT: Change how indices are compared
- :pr:`9341`: Bump actions/setup-node from 4.0.2 to 4.0.3
- :pr:`9343`: Fix OpenBLAS `pow_dd` unresolved symbol error, update Emscripten CI testing
- :pr:`9346`: DOC: Add citation file
- :pr:`9348`: DOC: Improve documentation of acf and plot_acf
- :pr:`9351`: STY: Accept 88 characters in linting
- :pr:`9354`: MAINT: Simplify and standardize setup
- :pr:`9356`: MAINT: Backport changes needed for 0.14.3 release
- :pr:`9358`: TST: Relax tolerance on test that fails for dynamic factor
- :pr:`9359`: MAINT: Run pyupgrade on 0.14 branch
- :pr:`9360`: MAINT: Run pyupgrade on main  branch
- :pr:`9361`: adjusting notation of error term in regression docs
- :pr:`9363`: DOC: Add release note for 0.14.3
- :pr:`9364`: DOC: Spelling
- :pr:`9365`: Backport of #9270: add Pyodide support and CI jobs for v0.14.x
- :pr:`9370`: Bump actions/setup-node from 4.0.3 to 4.0.4
- :pr:`9372`: Fix docstring formula display in SVAR class
- :pr:`9377`: DOC: Add release note for 0.14.4
- :pr:`9379`: DOC: Fix version number
- :pr:`9385`: BUG: Avoid modification in place
- :pr:`9386`: MAINT: Fix scalar assignment
- :pr:`9388`: ENH: changed np.round(mean_dff,2) -> mean_diff:.3g
- :pr:`9389`: feature/wilcoxon mann whitney sample size
- :pr:`9390`: BUG: Corect resid from UECM
- :pr:`9391`: DOC: Imroves docs for exponentialsmoothing and other places
- :pr:`9394`: BUG: Correct x and y label location in qqplot_2sample
- :pr:`9395`: MAINT: Replace deprecated Pandas append with concat in dynamic_factor_mq
- :pr:`9396`: BUG: Remove method setting in summary2 of genmod
- :pr:`9397`: DOC: Fix typo in previous fix
- :pr:`9398`: BUG: Ensure hessian is skipped
- :pr:`9399`: ENH: Add leybourne-mccabe test
- :pr:`9400`: BUG: Correct LLF for ETSModel
- :pr:`9401`: Feature/wilcoxon mann whitney sample size squashed
- :pr:`9407`: ENH: more reliable casting of pandas data
- :pr:`9411`: Bump actions/setup-node from 4.0.4 to 4.1.0
- :pr:`9413`: BUG: Ensure VAR can forecast with 0 lags
- :pr:`9422`: DOC: updated mediation tutorial documentation
- :pr:`9423`: ENH: Abstract formula engine
- :pr:`9424`: TST: Make test more resiliant
- :pr:`9439`: Dependencies consistency
- :pr:`9449`: CI: Update permissions
- :pr:`9453`: ENH: Add ruff support
- :pr:`9457`: BUG: Correct DatetimeIndex use
- :pr:`9458`: TST: Restore skip when no x13 available
- :pr:`9461`: BUG: Correct handleing of PeriodIndex in seasonal_decompose
- :pr:`9462`: DOC: Corrected a typo in chi^2
- :pr:`9467`: Update conf.py year
- :pr:`9468`: BUG: svar, A,B dtype, one parameter score shape, closes #9302
- :pr:`9470`: MAINT: Bump formulaic to 1.1.0
- :pr:`9471`: Fix formula eval depth in select models
- :pr:`9477`: DOC: Corrected typos in the Hurdle Count Model example
- :pr:`9483`: DOC: remove empty cell in tsa_arma_0.ipyb file
- :pr:`9484`: DOC: fixed ETS simple exponential smoothing equations
- :pr:`9487`: BUG/ENH: Tukeyhsd, fix unused variance, add Games-Howell
- :pr:`9492`: Bump actions/setup-node from 4.1.0 to 4.2.0
- :pr:`9498`: Modify x13_arima_analysis to produce seasonality fit diagnostics
- :pr:`9503`: TST: Relax tolerance on overly tight test
- :pr:`9510`: fix doc for extrapolate_trend and allow `period` as well
- :pr:`9518`: [ENH] Allow ARDL model trend 'ctt'
- :pr:`9524`: BUG: Fix bug in Runs.runs_test for the case of a single run yielding â€¦
- :pr:`9532`: DOC: fix duplicate words in weightstats
- :pr:`9535`: Bump actions/setup-node from 4.2.0 to 4.3.0
- :pr:`9541`: BUG: Correct spelling of pytest fixture
- :pr:`9543`: MAINT: Remove _lazywhere in favor of apply_where
- :pr:`9544`: ENH: add plotkwargs for qqplot_2samples()
- :pr:`9545`: MAINT: Improve handeling of missing mvndst
- :pr:`9546`: STY: Remove unused import
- :pr:`9547`: CI: Fix flaky test and add 3.13 jobs
- :pr:`9550`: Add optional raw spec parameter for x13_arima_analysis()
- :pr:`9551`: DOC: Check, fix and format some notebooks
- :pr:`9552`: DOC: Check, fix and format some notebooks
- :pr:`9553`: DOC: Fix statespace local linear
- :pr:`9554`: DOC: Format and fix up notebooks
- :pr:`9557`: Bump actions/setup-node from 4.3.0 to 4.4.0
- :pr:`9558`: DOC: Fixed typo in VARResults Attribute docstring : params -> coefs.
- :pr:`9561`: Fix Broken Link to Citation Paper of 2010 Conference
- :pr:`9568`: MAINT: Convert decimal for float to avoid future issue
- :pr:`9571`: ENH: medcouple n log n (see #9570)
- :pr:`9581`: BUG: make Binomial family more robust to corner case mu=0 , endog=0
- :pr:`9582`: ENH: Support for array-like and pandas-like data
- :pr:`9586`: MAINT: Remove lazywhere
- :pr:`9588`: DOC: Update supported Python versions
- :pr:`9591`: Rls 0 14 5 notes
- :pr:`9594`: MAINT: Forward port changes to Holt-Winters
- :pr:`9595`: TST: Fix warning catching
- :pr:`9596`: Xfail regularized problems
- :pr:`9597`: STY: Fix linting fails
- :pr:`9598`: Future fixes
- :pr:`9602`: MAINT: Prepare for pandas 3 strings
- :pr:`9607`: Commit (https://github.com/statsmodels/statsmodels/issues/9606)
- :pr:`9615`: MAINT: Wrap pandas deprecate_kwarg
- :pr:`9616`: Bump actions/checkout from 4 to 5
- :pr:`9617`: DOC: Fix minor issues in notebooks
- :pr:`9618`: Update pytest
- :pr:`9621`: DOC: Fix minor issues in notebooks and RST
- :pr:`9622`: Fix redundant heading in docs/README.md
- :pr:`9624`: Bump actions/setup-python from 5 to 6
- :pr:`9625`: Bump actions/setup-node from 4.4.0 to 5.0.0
- :pr:`9626`: DOC: Fix typo in maintainer_notes.rst (get â†’ git)
- :pr:`9630`: MAINT: Fix for deprecation warnings
- :pr:`9631`: MAINT: Remove dependence on npymath
- :pr:`9632`: SETUP: Further clean up on setup
- :pr:`9633`: MAINT: Update for recent changes
- :pr:`9634`: BLD: Explore using meson
- :pr:`9636`: Fix 'add_trend' error message to correctly specify which columns are constant.
- :pr:`9637`: TST: Report xfail for flaky test
- :pr:`9638`: CI: Close figures at the end of tests
- :pr:`9639`: TST: Fix test that fails in prerelease testing
- :pr:`9640`: Assert rasies
- :pr:`9641`: MAINT: Remove unused import
- :pr:`9642`: CL:N: Add Stacklevel and other quality issues
- :pr:`9643`: CLN: Implement rules that are close to passing
- :pr:`9644`: CLN: Remove some additional formatting issues
- :pr:`9646`: Fix import error
- :pr:`9647`: CLN: Remove some additional formatting issues
- :pr:`9648`: MAINT: Remove panding deprecation matrix usage
- :pr:`9649`: MAINT: Remove panding deprecation matrix usage
- :pr:`9650`: CLN: Fix linting for bugbear
- :pr:`9651`: Ruff tests
- :pr:`9652`: Bump pypa/cibuildwheel from 3.1.4 to 3.2.0
- :pr:`9656`: CI: Add 3.14 in GH actions
- :pr:`9660`: DOC: Fix Gamma loglike_obs docstring and clarify weights parameterizaâ€¦
- :pr:`9668`: Bump github/codeql-action from 3 to 4
- :pr:`9669`: Bump pypa/cibuildwheel from 3.2.0 to 3.2.1
- :pr:`9673`: BUG: Fix conversion of 1-d arrays to scalars
- :pr:`9683`: DOC: Fix issues affecting notebooks
- :pr:`9688`: BUG/DOC: fix state space model transition timing
- :pr:`9689`: MAINT: Remove feature deprecated in Pandas 3
- :pr:`9691`: ENH: Add no cross terms option to White's test for heteroscedasticity
- :pr:`9692`: Bump pypa/cibuildwheel from 3.2.1 to 3.3.0
- :pr:`9698`: Bump actions/checkout from 5 to 6
- :pr:`9700`: MAINT: Improve compatability with recent NumPy
- :pr:`9701`: DOC: Release note for 0.14.6
- :pr:`9709`: CI: add CPython 3.14t CI
- :pr:`9710`: STY: Use del obj.attr rather than delattr(obj, "attr")
- :pr:`9712`: MAINT: Obscure cow changes
- :pr:`9716`: TST: run nonparametric tests in parallel on CI
- :pr:`9717`: BLD: generate free-threading compatible cython modules
- :pr:`9718`: DOC: fix typo in example notebook
- :pr:`9720`: `PERF: Optimize VECM memory/speed by avoiding O(T^2) projection matrix`
- :pr:`9721`: MAINT: remove obsolete statsmodels.interface package (empty)
- :pr:`9722`: MAINT: lazy_apply patsy/pandas compatibility
- :pr:`9724`: Fixed some spelling, grammar, and punctuation on the theta model example notebook
- :pr:`9726`: TST: Add marker for high memory tests
- :pr:`9728`: BUG: Pass alpha to plot_predict
- :pr:`9729`: FIX: incorrect length comparison in endpoint transformation logic
- :pr:`9732`: CLN: Removed unused _partial_regression function Fixes #9731 The _parâ€¦
- :pr:`9735`: Bump pypa/cibuildwheel from 3.3.0 to 3.3.1
- :pr:`9736`: TST: Xfail test on Windows due to SciPy changes
- :pr:`9737`: REF: Remove dependence on global RandomState
- :pr:`9738`: BUG: FIX compilation errors in statespace/meson.build #9733
- :pr:`9739`: BUG: Fix patsy eval_env handling in FormulaManager and add parametrized reâ€¦
- :pr:`9742`: TST: Enable thread safe tests
- :pr:`9747`: BUG: raise error for invalid endog input in emplike.DescStat
- :pr:`9749`: docs: fix broken academic reference in anova.py
- :pr:`9750`: ENH: Add missing attributes from AutoReg
- :pr:`9755`: DOC: fixed import statement in api-structure page
- :pr:`9757`: fix: add informative error message when Hessian inversion fails in fit_regularized
- :pr:`9758`: fix: replace 4 bare except clauses with except Exception
- :pr:`9759`: Bump pypa/cibuildwheel from 3.3.1 to 3.4.0
- :pr:`9760`: Relax overly tight test tol
- :pr:`9761`: TST: Xfail bad test
- :pr:`9762`: CI: Add jinja2 for testing
- :pr:`9763`: MAINT: fix `compat.scipy.apply_where` for scipy-internal change
- :pr:`9764`: TST: Remove valid cases from exception check
- :pr:`9766`: DOC: improve docstrings in robust.norms
- :pr:`9767`: MAINT: use `get_lapack_funcs` for low-level LAPACK functions
- :pr:`9769`: CLN: Fix lint issues
- :pr:`9770`: TST: Attempt to isolate OSX failure
- :pr:`9771`: STY: Fix flake8 error
- :pr:`9772`: MAINT: Check that returned eigenvalues are real
- :pr:`9773`: BUG: Treat empty docstrings as None in Docstring class
- :pr:`9775`: TST: Skip failing tests on Win ARM64
- :pr:`9778`: BLD: ensure the `libm` C math library gets linked for all targets
- :pr:`9781`: Bump pypa/cibuildwheel from 3.4.0 to 3.4.1
- :pr:`9782`: CI: Remove joblib from freethreaded run
- :pr:`9783`: CI: Use site packages for free threaded tests
- :pr:`9784`: DOC: Fix Python interpreter example backslash newlines that rendered improperly
- :pr:`9786`: MAINT: Refactor monkey patch for patsy
- :pr:`9787`: ENH Added Seasonal-Diagnostic Plot to graphics.tsaplots
- :pr:`9788`: DOC: Add seasonal diagnostic plot to docs
- :pr:`9789`: TST: Relax tolerance and problematic test
- :pr:`9792`: ENH: Add ARIMA tutorial notebook example
- :pr:`9798`: ENH: make `tsa/statespace` Cython usage compatible with SciPy ILP64 builds
- :pr:`9800`: BUG: fix use_boxcox control flow in ExponentialSmoothing.fit (fixes #9797)
- :pr:`9802`: ENH: Add partial cross-correlation function (pccf)
- :pr:`9804`: ENH: Add Polars DataFrame support (Issue #9744)
- :pr:`9805`: BUG: honor MixedLM summary title
- :pr:`9809`: Rename README_l1.txt to L1_ADDITION.txt
- :pr:`9811`: ENH: Allow seasonal-differencing-only models with non-seasonal estimators (Issue #6159)
- :pr:`9812`: {BUG} Fix Issue #9793: Override resid property in UECMResults
- :pr:`9813`: DOC: correct PredictionResults.conf_int docstring
- :pr:`9814`: fix: avoid division by zero in estimate_location
- :pr:`9815`: ENH: graphics: Add add_ellipse and support passing x, y arrays to addâ€¦
- :pr:`9816`: ENH: tsa/vector_ar: Allow passing pre-calculated error bands to IRF plots
- :pr:`9819`: Enh/hannan rissanen order validation
- :pr:`9820`: ENH: Vendor `Appender` and `Substitution` docstring helpers from pandas
- :pr:`9822`: Update test notes with virtual environment activation steps
- :pr:`9823`: BUG: L-BFGS-B optimizer ignores disp=False, prints output unconditionally
- :pr:`9824`: BUG: Fix scale attribute and resid_pearson for fixed scale cov_type (#8190)
- :pr:`9825`: MAINT: Remove iprint for SciPy 1.18+
- :pr:`9826`: BUG/CLN: remove dead assignment to cov_p in GLM fit
- :pr:`9829`: BUG: pass ax parameter through to dot_plot in CombineResults.plot_forest
- :pr:`9830`: Fix GLMInfluence.hat_matrix_diag method name
- :pr:`9831`: ENH: Vendor `cache_readonly` and `deprecate_kwarg` from pandas private API
- :pr:`9832`: MAINT: drop removed scipy interp2d from TableDist (closes #8909)
- :pr:`9833`: MAINT: Future fixes
- :pr:`9834`: MAINT: Reduce future warnings
- :pr:`9835`: BUG: Fix VIF numerical instability by standardizing design matrix
- :pr:`9836`: DOC: Improve docstrings
- :pr:`9837`: MAINt: Update CIBW to 4.1.0
- :pr:`9838`: DOC: fix incorrect parameter names in deconvolve, powerdiscrepancy and VECMResults.predict docstrings
- :pr:`9839`: DOC: fix `freeman_tukey` formula rendering in powerdiscrepancy docstring
- :pr:`9840`: Improve text formatting for `macOS`
- :pr:`9842`: TST: ATtempt to avoid rare failures in thread-safe
- :pr:`9843`: CI : Pin github actions to full commit sha
- :pr:`9844`: BUG: skip summary diagnostics when slim=True
- :pr:`9845`: ENH: add fixed_params support to innovations_mle (Issue#6159)
- :pr:`9848`: DOC: fix typo
- :pr:`9849`: TST: Mark test as unsafe
- :pr:`9850`: DOC: fix typos
- :pr:`9852`: FIX: anova_lm silently returns NaN p-values when models are passed in reverse order
- :pr:`9853`: BUG: set k_exog_user on SVARResults so summary() works (GH#8025)
- :pr:`9854`: Improve `test_family` documentation
- :pr:`9855`: MAINT: run `isort` on codebase
- :pr:`9857`: TST: Relax tol on test that frequenctly fails
- :pr:`9858`: CI: Reduce the number of runs to improve performance in CI
- :pr:`9859`: Bump actions/checkout from 6 to 7
- :pr:`9861`: DOC: Change to pydata theme
- :pr:`9862`: BUG: fix Binomial.deriv() to return 1 - 2*mu/n (missing division by n)
- :pr:`9863`: DOC: Shorten word in title
- :pr:`9864`: DOC: Fix URL and notebooks
- :pr:`9865`: DOC: Fix origin in conf
- :pr:`9866`: BUG: record robust scale in RLM fit_history
- :pr:`9867`: MAINT: fix import sorting in test_weights
- :pr:`9870`: MAINT: link validation logic in Family._setlink
- :pr:`9873`: Fix typos in test_chisquare_prob docstring
- :pr:`9874`: [ENH] Add Jonckheere-Terpstra ordered trend test
- :pr:`9876`: DOC: improve math formulas in robust.norms docstrings
- :pr:`9877`: BUG: fix NegativeBinomial check for optional alpha
- :pr:`9878`: MAINT: Reduce direct use of np.random.func
- :pr:`9879`: MAINT: Remove direct use of np.random
- :pr:`9881`: [codex] DOC: document GLS other_results
- :pr:`9883`: MAINT: adapt to upcoming change in pd.freq
- :pr:`9884`: BUG: return nan from Power.solve_power when it fails to converge
- :pr:`9885`: ENH: report the last root-finder value in the solve_power ConvergenceWarning
- :pr:`9886`: fix: correct parameter names in docstrings (prob_infl, bin_edges, pred_kwds, param_nums, mu1_low)
- :pr:`9887`: fix DiscreteResults crash with full_output=0
- :pr:`9888`: BUG: Fix ccovf shape mismatch for different length arrays
- :pr:`9890`: DOC: fix Negative Binomial cumulant function in GLM families table
- :pr:`9892`: DOC: Following NumPy-style doc for Gamma log-likelihood
- :pr:`9893`: DOC: fix Gamma distribution notation in GLM families table
- :pr:`9894`: MAINT: fix typos in docstrings and comments
- :pr:`9895`: ENH: raise informative error for impossible one-sided solve_power cases
- :pr:`9896`: DOC: Fix failure in docs due to warning
- :pr:`9898`: ENH/BUG: add min_diag option to cov_nearest for zero or negative diagonal
- :pr:`9899`: BUG: describe/Description handles 0-row (empty) input gracefully (#9891)
- :pr:`9901`: Fix up random generation
- :pr:`9902`: BUG: Attach mlefit attributes to the results instance so they appear in dir()
- :pr:`9903`: BUG: Use exog centroid as center in rainbow test use_distance (#9103)
- :pr:`9904`: TST: Improve tests for thread safety
- :pr:`9905`: Fix a small issue in statsmodels (#9869)
- :pr:`9906`: BUG: filter unsupported kwargs in MixedLM.fit to prevent AttributeError
- :pr:`9907`: BUG: use rank-adjusted df in wald_test_terms for rank-deficient models
- :pr:`9908`: BUG: Do not pass hess to L-BFGS-B and TNC in _fit_minimize
- :pr:`9909`: BUG: Fix sison-glaz confint failure for small or sparse counts
- :pr:`9910`: TST: Improve thread safety of tests
- :pr:`9912`: CLN: Fix CodeQL detected minor issues
- :pr:`9913`: CI: Drop support for Python 3.9 in CI
- :pr:`9914`: DOC: add missing PoissonResults and NegativeBinomialPResults to discretemod autosummary (closes #9022)
- :pr:`9915`: BUG: ARDLResults.apply/append loses exog lag order
- :pr:`9916`: BUG: divide adjusted ccovf by the overlapping count, not len(x) - k
- :pr:`9919`: BUG: read the entropy integration limits from the kernel
- :pr:`9920`: BUG: populate _retain_cols in out_of_sample without a prior in_sample call
- :pr:`9923`: TST: Fix threaded failing test
- :pr:`9924`: BUG: Correct test to not use the singleton
- :pr:`9925`: BUG: Fix import when MPL not installed
- :pr:`9926`: Bump r-lib/actions/setup-pandoc from 2.12.0 to 2.12.1
- :pr:`9927`: Bump actions/setup-python from 6.2.0 to 7.0.0
- :pr:`9928`: Bump pypa/cibuildwheel from a0a973acdc9e7b7f8b04ac5c80e6883a5a102615 to 294735312765b09d24a2fbec22660ce817587d55
- :pr:`9929`: DOC: Fix many docstring issues in discrete
- :pr:`9930`: DOC: Fix many docstring issues in genmod
- :pr:`9931`: DOC: Fix many docstring issues in stats
- :pr:`9932`: CLN: Fix import order using isort
- :pr:`9933`: fix(grouputils): unify group_sums orientation and fix group_demean
- :pr:`9934`: DOC: Improve tsa docstrings ex. statespace
- :pr:`9935`: DOC: Improve base, compat and dataset docstrings
- :pr:`9936`: MAINT: Remove deprecations
- :pr:`9937`: DOC: Improve graphics docstrings
- :pr:`9938`: DOC: Improve imputation, multivariate and non-parametric docstrings
- :pr:`9939`: DOC: Update notebooks for deprecations
- :pr:`9940`: DOC Improve docstrings othermode, regression and robust
- :pr:`9941`: DOC: fix typos in docstrings, comments, and messages
- :pr:`9942`: DOC: Fix small issues found in docbuild
- :pr:`9943`: DOC: Fix emplike and duration
- :pr:`9944`: DOC: Fix treatment and gam docstrings
- :pr:`9945`: DOC: Fix docstring issues in tools
- :pr:`9946`: DOC: Fix some issues in statespace docstrings
- :pr:`9947`: REF: Move from random_state to rng
- :pr:`9948`: TST: Add marker for joblib
- :pr:`9949`: CI: Improve doc build reqs
- :pr:`9950`: ENH: Consistently use `rng` to move towards SPEC-007
- :pr:`9951`: DOC: Start release note for 0.15.0
- :pr:`9952`: DOC: SMall fixes for docs
- :pr:`9953`: MAINT: Bump to the actual minimum of 3.10
- :pr:`9954`: DOC: Final pass at doc fixes
- :pr:`9955`: DOC: Fix notebook and allow all to run
- :pr:`9957`: ENH: Add Hamilton filter (continued from 9872)
- :pr:`9958`: BUG: Fix removal of compat lstsq
- :pr:`9959`: CLN: Fix small lint issue in test
- :pr:`9960`: More doc fixes
- :pr:`9961`: DOC: Small doc fixes
- :pr:`9962`: DOC: Fix NegativeBinomialP.fit docstring
- :pr:`9963`: DOC: Fix title level in notebook and move ref
- :pr:`9967`: DOC: document that exog is matched by position for non-formula models
- :pr:`9969`: DOC: Remove sections from docstrings that do not render correctly
- :pr:`9972`: REF: Reduce mutability of models fit() methods
- :pr:`9973`: REF: Reduce genmod use of del
- :pr:`9974`: BUG: raise on non-2x2 tables in stats.mcnemar (#9485)
- :pr:`9976`: BUG: respect caller warning filters in discrete l1 fit_regularized (#9179)
- :pr:`9977`: REF: Remvoe anti-patterns in multivariate and robust
- :pr:`9978`: REF: Remove anti-patterns in tsa
- :pr:`9980`: REF: Remove anti-pattern use in stats and tools
- :pr:`9981`: MAINT: Move from isort to ruff
- :pr:`9982`: Bump actions/checkout from 7.0.0 to 7.0.1
- :pr:`9983`: Bump pypa/cibuildwheel from 4.1.0 to 4.1.1
- :pr:`9984`: REF: Extend the best practices to additional files
- :pr:`9985`: BUG: Reject None in string_like unless optional is True
- :pr:`9987`: BUG: Reject None in array_like unless optional is True
- :pr:`9988`: REF: Make use of pathlib
- :pr:`9989`: CLN: Clean examples
- :pr:`9990`: ENH: Improve nbgeneration
- :pr:`9991`: DOC: Add plot for hamilton_filter
- :pr:`9992`: BUG: Don't validate the specification when extending SARIMAX results
- :pr:`9993`: BUG: Fix score_test to return HolderTuple instead of plain tuple #9785
- :pr:`9994`: BUG: Select the correct axis in drop_missing
- :pr:`9995`: MAINT: Improve formula engine specification
- :pr:`9996`: docs: use HTTPS for MixedLM reference
- :pr:`9997`: DOC clarify add_constant prepend default
- :pr:`9998`: DOC clarify GLMGam out-of-sample prediction
- :pr:`9999`: DOC fix ANOVA example link
- :pr:`10000`: DOC list all GEE covariance structures
- :pr:`10001`: ENH: Add block jackknife estimator (addresses #9752)
- :pr:`10002`: BUG: Ensure AutoReg summary can run after calling remove data
- :pr:`10003`: TST: Add tests for summary-remove-data pattern
- :pr:`10005`: BUG: Report the correct accepted types in dict_like
- :pr:`10006`: DOC: Correct the recipr0 summary line
- :pr:`10007`: TST: Add tests for summary-remove-data pattern in regression
- :pr:`10008`: TST: Add tests for summary-remove-data pattern
- :pr:`10009`: Statespace summary remove data
- :pr:`10010`: BUG: clip wilson proportion_confint bounds to [0, 1]
- :pr:`10011`: DOC: fix discrete results parameters
- :pr:`10012`: BUG: sign_test raises an opaque error when all observations tie with mu0
- :pr:`10013`: BUG: multipletests raises ZeroDivisionError on an empty p-value array
- :pr:`10014`: BUG: maxabs and iqr raise on an empty input, unlike the other eval_measures
- :pr:`10015`: MAINT: Remove Deprecations and outdated code
- :pr:`10016`: ENH: Allow list of lags additional to maxlag
- :pr:`10017`: BUG: use the non-missing sample size for acf confint/qstat when NaNs are handled
- :pr:`10018`: MAINT: Remove debug code
- :pr:`10019`: MAINT: Archive unused statsmodels.sandbox files
- :pr:`10020`: BUG: Avoid divide by 0 in acf/acovf with explicit error
- :pr:`10021`: TST: Add test run for x13
- :pr:`10022`: MAINT: COrrect test on older SciPy
- :pr:`10023`: CLN: Remove unproductive __name__ == "__main__" code
- :pr:`10025`: ENH: Reduce variable output returns
- :pr:`10026`: MAINT: Address deprecations
- :pr:`10027`: More named tuple
- :pr:`10028`: DOC: Remove five documented parameters that are not in the signature
- :pr:`10029`: REF: Move variable return to NamedTuple
- :pr:`10030`: ENH: Add NamedTuples to remaining fixed-arity tsa.stattools functions
- :pr:`10031`: DOC: Add numpydoc parameters sections to NamedTuple result classes
- :pr:`10032`: MAINT: Small jobs prior to release
- :pr:`10033`: DOC: Improve docstrings and css
- :pr:`10034`: DOC: Update release note
- :pr:`10035`: ENH: More use of NamedTuple
- :pr:`10036`: DOC: Fix rst errors and update notebooks
- :pr:`10037`: DOC: General fixes
- :pr:`10038`: DOC: Fix minor typo ("Destribution" -> "Distribution")
- :pr:`10039`: TST: rename misleading WLS equivalence tests
- :pr:`10040`: DOC: General fixes
- :pr:`10041`: DOC: fix two defaults that the code does not have
- :pr:`10042`: Use self._ntop instead of literal 5 for categorical frequencies in Description
- :pr:`10043`: Fix smal bugs
- :pr:`10044`: Fix more small bugs
- :pr:`10045`: DOC: Add AI policy
- :pr:`10046`: DOC: fix typo in GLS example
- :pr:`10047`: DOC: clarify GLSAR rho argument
- :pr:`10048`: CI: Switch build that tests x13 to have coverage
- :pr:`10049`: ENH/TST: Deprecate parameter and test edge cases
- :pr:`10050`: MAINT: Improve issue and PR templates
- :pr:`10051`: CI: Change x13 binary installation
- :pr:`10052`: CI: Improve documentation generation
- :pr:`10053`: DOC: Add newly introduced functions to docs
- :pr:`10054`: CLN: Move to read_csv from genfromtxt
- :pr:`10055`: [ENH] Add Pesaran-Timmermann directional accuracy test
- :pr:`10056`: Fixups
- :pr:`10057`: DOC: add AR(p) notation to GLSAR.whiten
- :pr:`10058`: MAINT: Protect against pandas 4 changes
- :pr:`10060`: MAINT: Update PR template
- :pr:`10061`: DOC: Updates for recent robust norm docstrings
- :pr:`10062`: CLN: Remove whitespace
- :pr:`10063`: DOC: Standardized docstring changes
- :pr:`10064`: CI: Add lint-only GitHub workflow (ruff + flake8, Linux, Python 3.14)
- :pr:`10065`: More pandas 4 fixes
- :pr:`10066`: ENH: implementation of DM test
- :pr:`10067`: CLN: Fix small issues in jonckheere-terpstra
- :pr:`10068`: CLN: Fix lint issues
- :pr:`10069`: ENH: add p-value adjustments based on local false discovery rate
- :pr:`10070`: Bump actions/github-script from d746ffe35508b1917358783b479e04febd2b8f71 to 3a2844b7e9c422d3c10d287c895573f7108da1b3
- :pr:`10071`: Bump pypa/cibuildwheel from 4.1.1 to 4.2.0
- :pr:`10072`: MAINT/CLN: Remove Holder/HolderTuple in favor of documented classes
- :pr:`10074`: DOC: Remove warning from docs
- :pr:`10075`: BUG: Fix Jonckheere-Terpstra on Pyodide by casting np.repeat arg to intp size
- :pr:`10077`: MAINT: remove reduntant `method` validation in yule_walker
- :pr:`10078`: DOC: Add AGENTS.md and update CONTRIBUTING
- :pr:`10079`: Move README
- :pr:`10080`: DOC: Remove coveralls
- :pr:`10081`: BUG: Finish move from README.rst to README.md
- :pr:`10082`: CLN: Fix lint issues
- :pr:`9956`: docs(stats): clarify TukeyHSD reject and pvalues access
- :pr:`10076`: DOC: Improve documentation for yule_walker
- :pr:`10083`: TST: Test remaining edge cases in jonckheere_terpstra
- :pr:`10084`: BUG: Correct edge cases in n log n medcouple path
- :pr:`10085`: DOC: Update release note
- :pr:`10087`: ENH: Also check binaries with .exe
- :pr:`10088`: BUG: Check for positivity of eigval in condition number
- :pr:`10089`: BUG: fix MNLogit resid_response raising ValueError (closes #7096)
- :pr:`10090`: ENH: Make ndim more orthogonal to maxdim
- :pr:`10091`: Add LocalProjections estimator for impulse response functions (Jordà…)
- :pr:`10092`: ENH: Modify the approach to use dataclasses to limit unpack
- :pr:`10093`: REF: Move away from limited iter NamedTuple
- :pr:`10094`: MAINT: Restore accidental behavior change
- :pr:`10095`: TST: Add tests for limited iteration superclass
- :pr:`10096`: CLN: Standardize names in new objects
- :pr:`10097`: DOC: Reduce reference noise in sphinx
- :pr:`10098`: CLN/DOC: Simplify NamedTuple and dataclasses
- :pr:`10099`: DOC: Fix typo in WLS example notebook row labels
- :pr:`10100`: REF: Remove unused scipy import and cell from wls.ipynb
- :pr:`10101`: ENH: Implement L1 solver for GLM Extended #9430
- :pr:`10102`: PERF: Improve import performan in some cases
- :pr:`10103`: ENH: add crv3 cluster robust inference via the cluster jackknife for OLS/WLS
- :pr:`10104`: Docstring types regression
- :pr:`10105`: BUG: Forward missing kwarg from MixedLM.from_formula to superclass
- :pr:`10106`: BUG: Pivot the QR factorization in tools.matrix_rank
- :pr:`10107`: DOC: Standardized docstrings in tools
- :pr:`10108`: DOC: Standardized docstrings in robust
- :pr:`10110`: DOC: Standardized docstrings in stats
- :pr:`10111`: DOC: Standardized docstrings in othermod, treatment and multivariate
- :pr:`10112`: DOC: Standardized docstrings in base, datasets and compat
- :pr:`10113`: BUG: Fix numerous small bugs
- :pr:`10114`: DOC: Fix small remaining issues in regression
- :pr:`10115`: DOC: Fix small remaining issues around use of np and pd
- :pr:`10116`: DOC: Documentation cleaning pass for formula, graphics and imputation
- :pr:`10117`: DOC: Documentation cleaning pass core routines in tsa
- :pr:`10118`: DOC: Fix UECM
- :pr:`10119`: DOC: Clean docstrings in discrete, duration gam and genmod
- :pr:`10120`: BUG: Add missing self to update
- :pr:`10121`: DOC: Docstring clean in dist, emplike, iolib and mismodel
- :pr:`10122`: DOC: Replace broken OECD glossary links in endog_exog docs
- :pr:`10123`: DOC: Docstring clean in nonparametric
- :pr:`10124`: DOC: Docstring clean in vector_ar
- :pr:`10125`: DOC: Update agents to improve docstrings
- :pr:`10127`: DOC: Clean docstrings in statespace
- :pr:`10128`: DOC: Improve dataset docstrings
- :pr:`10129`: BUG: Rename variable to SUNACTIVITY
- :pr:`10130`: BUG: Correct distargs usage in scale_trimmed
- :pr:`10131`: BUG: Fix bug in line-style application
- :pr:`10132`: BUG: Enable percentile in _select_sigma
- :pr:`10133`: BUG: Fix factor reverse intent
- :pr:`10134`: BUG: Only initialize trend when required
- :pr:`10135`: BUG: Correct hess choice in betareg
- :pr:`10136`: BUG: Ensure gap size is correct in mosaic_plot
- :pr:`10137`: BUG: Ensure not implemented options raise
- :pr:`10138`: TST: Re-enable test
- :pr:`10139`: BUG: Fix bugs found in full scan
- :pr:`10140`: ENH: Warn users if exog is singular in *LS
- :pr:`10141`: BUG: Fix small bugs
- :pr:`10142`: TST: Change warning class on fit_collinear
- :pr:`10143`: BUG: Correct size of cancorr returns
- :pr:`10144`: TST: Change warning on WASM
- :pr:`10145`: DOC: Standard docstrings for rng
- :pr:`10146`: TST: Silence singular warnings
- :pr:`10147`: TST: Add tests for rng selector
- :pr:`10150`: TST: Cover results-class surface gaps
- :pr:`10151`: TST: Cover dead computational methods on live estimators
- :pr:`10152`: BUG: Fix OLSInfluence._ols_xnoti crashing on every call
- :pr:`10153`: TST: Cover margins and diagnostics gaps
- :pr:`10154`: BUG: Fix RLMDetSMM.fit crashing with its documented h=None default
- :pr:`10155`: TST: Verify NewsResults summary content, not just non-emptiness (Phase…)
- :pr:`10156`: MAINT: Deprecate estimator classes with no callers and no test coverage
- :pr:`10158`: CI: Disable failing conda run
- :pr:`10160`: CI: Re-enable conda with different cython
- :pr:`10161`: ENH: Enforce string like validation
- :pr:`10162`: CI: Revery cython for legacy conda test
- :pr:`10163`: BUG: Fix MICEData using observed-row index for predict_miss_kwds
- :pr:`10164`: BUG: Fix TreatmentEffectResults mislabeling every method as IPW
- :pr:`10165`: BUG: Guard against None zero_kwds in effectsize_2proportions
- :pr:`10166`: TST: Add marker for matplotlib tests
- :pr:`10167`: ENH: Add validation to from_string methods
- :pr:`10168`: CLN: Move non-core code our of package
- :pr:`10169`: DOC: Improve docstring for pacf
- :pr:`10170`: ENH: Simplify aliases
- :pr:`10171`: REF: Delegate ETS breakvar test to the shared implementation
- :pr:`10172`: BUG: fix SARIMAX time-varying regression with differencing in the state vector
- :pr:`10173`: ENH: Improve string checking
- :pr:`10174`: BUG: Add array_like for offset
- :pr:`10175`: BUG: Remove cache_readonly the presented parameter
- :pr:`10176`: BUG: Ensure array_like covnull is coerced
- :pr:`10177`: BUG: Correct bug in knot centereing
- :pr:`10178`: BUG: Ensure linepred is always available
- :pr:`7327`: BUG: Fix libsturng issue #6541
- :pr:`9297`: Update model.py --corrected wald test error for RegimeSwitchingmodels
- :pr:`9695`: Fix: cov_type in MixedLM.fit
- :pr:`9794`: fix: use normalized_cov_params as fallback when hessian inversion fails in GLM.fit
- :pr:`9979`: BUG: back-transform the univariate smoothed measurement disturbance
- :pr:`10179`: ENH/BUG: Use scipy.special.log_wright_bessel for the Tweedie log-likelihood
- :pr:`10180`: ENH: Add explicit target for removal of string aliases
- :pr:`10181`: CI: Reduce Linux jobs to speed up completion
- :pr:`10182`: ENH: Allow string type for groups in NominalGEE
- :pr:`10183`: DOC: Update the release notes
- :pr:`10184`: MAINT: Fix the sign when using Newton's method
- :pr:`10185`: BUG: Fix MNLogit score_test crash with exog_extra (GH#9273)
- :pr:`10186`: MAINT: Add scipy version check
- :pr:`10187`: BUG: pass transformed through to MarkovSwitching.hessian
- :pr:`10188`: TST: Avoid test where log_wright_bessel is not available
- :pr:`10189`: MAINT: Add coverage analysis tooling for the estimation API
- :pr:`10190`: BUG: Fix bad merge
- :pr:`10191`: BUG: support model-aware RLM scale callbacks
- :pr:`10192`: MAINT: Standardize on ruff
- :pr:`10193`: MAINT: Increase rule use from ruff
- :pr:`10194`: TST: Cover public API coverage gaps (batch: tools/stats/base/iolib)
- :pr:`10195`: BUG: Fix _BayesMixedGLM.fit silently returning None
- :pr:`10196`: BUG: Fix GLS.hessian_factor for 1d (heteroskedastic) sigma
- :pr:`10197`: BUG: Fix emplikeAFT.predict using endog instead of exog
- :pr:`10198`: BUG: Fix rvs_kernel ignoring rng for the Beta-kernel draws
- :pr:`10199`: TST: Add coverage for VARProcess/VARResults acorr methods
- :pr:`10200`: BUG: Fix Representation.initialize_components missing k_states arg
- :pr:`10201`: BUG: Fix miso_lfilter column selection for nvars != 2, 3
- :pr:`10202`: TST: Add skip on WASM for linalg error
- :pr:`10203`: BUG: Return NotImplementedError rather than wrong result in GLS.hessian_factor
- :pr:`10204`: ENH: Add fit_regularized to HurdleCountModel
- :pr:`10206`: MAINT: Address future changes in pandas
