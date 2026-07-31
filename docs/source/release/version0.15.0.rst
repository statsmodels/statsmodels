:orphan:

==============
Release 0.15.0
==============

Release summary
===============

This note covers all changes merged into ``main`` between the ``v0.15.0.dev0``
tag (2023-05-05) and the current development head (2026-07-22). It is a
working draft assembled directly from the git history rather than a final,
polished release announcement, and it does not yet follow the format of
previous ``versionX.Y.rst`` release notes.

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

Release statistics
-------------------

- **Pull requests merged**: 355
- **Non-merge commits**: 953
- **Contributors** (by git log author, unique names): 122
- **Time span**: 2023-05-05 through 2026-07-22

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
- ``ARDL`` models can now use a ``"ctt"`` trend. :pr:`9518`
- ``x13_arima_analysis`` gained seasonality fit diagnostics and an optional
  raw spec parameter. :pr:`9498`, :pr:`9550`

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

Platform and packaging compatibility
----------------------------------------

- Cython 3 compatibility, and compatibility of the ``tsa.statespace`` Cython
  code with SciPy ILP64 builds. :pr:`9078`, :pr:`9798`
- Experimental Pyodide/WebAssembly support and CI jobs. :pr:`9270`, :pr:`9343`
- Free-threaded (no-GIL) CPython compatibility work, including
  free-threading-compatible Cython modules and CI coverage. :pr:`9717`

Notable bug fixes
---------------------

A few of the more consequential correctness fixes in this release (see
*Bug Fixes* below for the full list):

- ``families.Binomial.deriv()`` was missing a division by ``n`` and returned
  an incorrect value; it now correctly returns ``1 - 2 * mu / n``. :pr:`9862`
- The log-likelihood computation for ``ETSModel`` was corrected. :pr:`9400`
- A state space model transition-timing bug was fixed. :pr:`9688`
- ``anova_lm`` silently returned ``NaN`` p-values when models were passed in
  reverse order. :pr:`9852`
- Numerical instability in VIF was fixed by standardizing the design matrix
  before computing it. :pr:`9835`


Breaking Changes and Deprecations
===================================

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

Minimum dependency versions raised
--------------------------------------

- NumPy: 1.18 -> 1.22.3
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

.. rubric:: Performance

- Optimize VECM memory/speed by avoiding an :math:`O(T^2)` projection matrix. :pr:`9720`


Notable Bug Fixes
====================

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
a pass over example notebooks to fix formatting and broken links.

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


Testing, Linting, and Maintenance
=====================================

A substantial amount of routine maintenance went into keeping the test
suite green against upstream changes in NumPy, SciPy, and pandas (including
pandas copy-on-write and preparation for pandas 3), adopting ``ruff`` for
linting in addition to ``flake8``, running ``isort``/``pyupgrade`` across
the codebase, relaxing overly tight test tolerances, and improving thread
safety of the test suite ahead of free-threaded CPython support. Selected
items:

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


Development summary and credits
===================================

Thanks to everyone who contributed code, documentation, bug reports, and
review to this release cycle. The following list of contributors is
generated from ``git log`` between ``v0.15.0.dev0`` and the current
development head, and may not be complete or fully deduplicated across
differently-configured git identities:

Aditi Juneja, Adrian Ross, Agriya Khetarpal, Alex Alborghetti, Andrés,
Andrés López, Anh Trinh, Aniket, Aniket Singh Yadav, Anselm Hahn, Antoine
Mayerowitz, Anuraag Pandhi, Artem Glebov, Ben, Benjamin Leff, Bortlesboat,
Caleb Lindgren, Chad Fulton, Christine P. Chai, Clément Fauchereau, Daan
Knoope, David Ivanov, Deshan, Dhairya Motta, Dhruvil Darji, Eden Rochman,
Elton Chang, Erich Morisse, Eugen Goebel, Evan Lyall, Evgeni Burovski,
FuturMix, Hadi Dayekh, Harish Bhavandla, IsaacP, IntegralIndefinida, Illia
Polovnikov, Iman, Jesse W. Collins, Jim Varanelli, Joey Scanga, Josef
Perktold, Joshua Markovic, Justin Mahlik, Kaif, Kevin Sheppard, Kevin
Gregory, Kumar Aditya, Lakshmi786, Luke J, Maciej Skorski, Manlai Amar, Marc
Bresson, Mathias Hauser, Maxime Gourguechon, Melissa Wu, Michał Górny,
Michel de Ruiter, Naimish Machchhar, Pranav Achar, Puneet Dixit, Rahul
Rathnavel K, Ralf Gommers, Rebecca N. Palmer, RoyS, Sebastian Pölsterl,
Shamus, Solaris-star, Sreekant Baheti, Tartopohm, Vedant Madane, Vikram
Kumar, Viktor, Vitaliy, Wali Reheman, Will Tirone, YangWu1227, Zbigniew
Jędrzejewski-Szmek, Zhengbo Wang, and many others.

These lists are automatically generated based on ``git log`` and may not be
complete.
