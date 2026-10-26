:orphan:

==============
Release 0.15.1
==============

Release summary
===============

Release 0.15.1 is a maintenance release of statsmodels 0.15. It collects the
bug fixes, stricter checks of function arguments and smaller enhancements that
were merged into ``main`` after the ``v0.15.0`` tag (2026-08-27). Most changes
correct results or error messages that were wrong or unhelpful in 0.15.0. Two
arguments are deprecated and continue to work with a ``FutureWarning``.

statsmodels is using github to store the updated documentation. Two versions
are available:

- `Stable <https://www.statsmodels.org/>`_, the latest release
- `Development <https://www.statsmodels.org/devel/>`_, the latest build of the main branch

Release Statistics
------------------

- **Pull requests merged**: 259
- **Time span**: 2026-08-27 through 2026-10-24

Breaking Changes and Deprecations
=================================

Deprecations
------------

- The ``return_df`` argument of
  :func:`~statsmodels.stats.diagnostic.acorr_ljungbox` has no effect, since
  the function always returns a ``DataFrame``. Passing it now raises a
  ``FutureWarning``, and it will be removed in statsmodels 0.17.0.
  :issue:`8968`, :pr:`10549`
- The ``linear`` argument of ``predict`` in
  :class:`~statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM`,
  :class:`~statsmodels.genmod.bayes_mixed_glm.PoissonBayesMixedGLM` and
  :class:`~statsmodels.genmod.bayes_mixed_glm.BayesMixedGLMResults` is
  deprecated in favor of ``which``, matching
  :meth:`~statsmodels.genmod.generalized_linear_model.GLM.predict`. Use
  ``which="linear"`` instead of ``linear=True`` and ``which="mean"`` instead
  of ``linear=False``. Passing ``linear`` now raises a ``FutureWarning``, and
  it will be removed in statsmodels 0.17.0. The value of ``which`` is case
  sensitive and ``exog`` can be a list. :issue:`8562`, :pr:`10554`,
  :pr:`10555`

Behavior changes
----------------

- ``OaxacaResults.summary()`` returns the summary as a string instead of
  printing it and returning ``None``. Use ``print(result.summary())`` to
  display it. :issue:`7733`, :pr:`10491`
- ``NegativeBinomial`` in ``statsmodels.discrete.discrete_model`` raises a
  ``TypeError`` if the ``alpha`` keyword is passed. The dispersion parameter
  is estimated jointly with the regression coefficients, and a value passed as
  ``alpha`` was silently ignored. :pr:`10427`
- The ``cov_type`` values ``"HC1"``, ``"HC2"`` and ``"HC3"`` are only
  implemented for linear regression models. For other models, for example GLM
  and the discrete models, HC0 was used silently while the results reported
  the requested ``cov_type``. Now a ``SpecificationWarning`` is issued and the
  results report ``cov_type="HC0"``. The warning is attributed to the calling
  code, and its message gives a ``scaling_factor`` that matches the
  heteroskedasticity robust covariance of Stata.
  :issue:`7798`, :pr:`10523`, :pr:`10580`
- ``acorr_ljungbox`` raises a ``ValueError`` if a requested lag is smaller
  than 1. A lag of 0 silently returned the statistic of the largest lag. Lags
  that are not smaller than the number of observations also raise a
  ``ValueError`` now instead of failing with an array broadcasting error. The
  error messages name the lag that was passed. :pr:`10481`, :pr:`10311`,
  :pr:`10561`
- With ``missing="drop"``, a ``MissingDataError`` is raised if all
  observations are dropped because of missing values. An unknown value of
  ``missing`` raises a ``ValueError``. :issue:`4417`, :pr:`10535`,
  :pr:`10407`
- The name of the column ``"Cont. Int. Hi."`` in the data frame returned by
  ``DiscreteMargins.summary_frame`` is corrected to ``"Conf. Int. Hi."``.
  :pr:`10506`
- Many functions in ``statsmodels.stats``, ``statsmodels.tsa`` and other
  modules now raise a ``ValueError`` for invalid arguments that were accepted
  before, see *Stricter Input Validation* below.

New Features and Enhancements
=============================

New functions and methods
-------------------------

- Add :func:`~statsmodels.stats.diagnostic.compare_vuong`, the Vuong test for
  non-nested likelihood models, and its result class ``VuongTestResult``. The
  test supports AIC and BIC corrections and one-sided alternatives. The
  function is available from ``statsmodels.stats.api``. :issue:`3897`,
  :pr:`10502`, :pr:`10560`
- Add :func:`~statsmodels.stats.proportion.confint_proportions_paired` for the
  confidence interval of the difference of two paired proportions, using
  Newcombe's method 10 (the default) or the Wald interval. :issue:`10374`,
  :pr:`10375`
- Add ``RollingRegressionResults.get_resid`` to compute the residuals of
  ``RollingOLS`` and ``RollingWLS``, optionally as one-step ahead prediction
  errors with ``out_of_sample=True``. :issue:`7521`, :pr:`10462`
- Add the keyword-only argument ``effect_group`` to ``TreatmentEffect.aipw``
  and ``TreatmentEffect.aipw_wls`` to estimate the effect on the treated
  (ATET) or on the controls (ATC) in addition to the average treatment effect
  (ATE), which remains the default. :issue:`10226`, :pr:`10233`
- Add the methods ``TreatmentEffect.overlap_summary`` and
  ``TreatmentEffect.balance_table``. They report the distribution of the
  propensity score by treatment group and the covariate balance before and
  after inverse probability weighting. :issue:`10234`, :pr:`10235`
- ``MNLogit`` has the documented ``names`` attribute that maps the column
  number of ``wendog`` to the name of the category of the endogenous variable.
  :issue:`8995`, :pr:`10486`, :pr:`10558`
- ``DynamicFactorMQ.fit_em(full_output=True)`` includes a ``converged`` entry
  in the EM return values, which distinguishes a converged fit from a fit that
  stopped at ``maxiter`` or was reverted. :issue:`10157`, :pr:`10214`
- ``statsmodels.tsa.tsatools.freq_to_period`` supports sub-daily frequencies,
  for example hourly data maps to 24 and ``"5min"`` to 288, if the multiplier
  divides the day. It also supports the business month, quarter and year
  frequencies (``"BME"``, ``"BQS"``, ``"BYE"``, ``"CBMS"`` and similar), which
  map to the same period as their calendar counterparts, including the older
  pandas spellings. :issue:`9619`, :issue:`9308`, :issue:`10525`,
  :pr:`10401`, :pr:`10528`, :pr:`10542`
- ``statsmodels.tsa.tsatools.add_lag`` accepts the label of a column of a
  pandas ``DataFrame`` as ``col``. :pr:`10302`

Performance
-----------

- ``RollingOLS`` and ``RollingWLS`` estimate the windows in batches using
  stacked linear algebra. The pull request reports speed-ups of 5 to 30 times.
  :issue:`8417`, :pr:`10533`
- The leave-one-observation-out measures of ``OLSInfluence`` (``dfbeta``,
  ``dfbetas``, ``dffits``, ``resid_studentized_external``, ``cov_ratio`` and
  ``summary_frame``) are computed from closed form updates instead of
  estimating one regression for each observation if the model is OLS or WLS
  and ``exog`` has full rank. :issue:`9009`, :pr:`10367`
- ``adfuller`` with ``autolag`` selects the lag length from a single QR
  decomposition and needs much less memory and time for large ``maxlag``.
  This also applies to ``coint`` and ``zivot_andrews``. :issue:`1849`,
  :pr:`10526`
- ``acorr_ljungbox`` computes the autocorrelations using the FFT, which is
  faster for long series. :pr:`10470`
- ``robust.scale.qn_scale`` is about 5 times faster for large samples.
  :pr:`10244`
- Lag order selection fits the nested sequence of models from a single
  blocked QR decomposition, shared by a new private helper. This applies to
  ``adfuller`` with ``autolag``, ``VAR.select_order``, ``pacf_ols``,
  ``pacf_yw`` (Levinson-Durbin) and ``ardl_select_order``. ``VAR.select_order``
  is 12 to 43 times faster, ``pacf_ols`` about 90 times faster for long series,
  and ``ardl_select_order`` evaluates thousands of candidate models in a
  fraction of a second instead of seconds. ``pacf_ols`` with
  ``efficient=True`` demeans the series first, which is more accurate when the
  mean is large relative to the variation. :issue:`2276`, :pr:`10571`

Bug Fixes
=========

Regression and linear models
----------------------------

- ``OLS.hessian(params, scale=None)``, the Hessian of the profile
  (concentrated) log-likelihood, was missing a factor of 2 in the term that
  involves ``X'X``. The results of ``OLS.fit_regularized`` with
  ``profile_scale=True`` change because they use this Hessian.
  :issue:`9469`, :pr:`10473`, :pr:`10556`
- ``OLSInfluence`` for the results of a ``WLS`` model used the unweighted
  design matrix and residuals for the hat matrix, the studentized residuals,
  Cook's distance, ``dffits`` and the leave-one-observation-out measures.
  These are now computed from the whitened data, which matches R's
  ``influence.measures`` for ``lm(..., weights=w)``. The weights are treated as
  precision weights. :issue:`9067`, :pr:`10476`, :pr:`10563`
- ``compare_lm_test`` with ``cov_type="cluster"`` returned a Lagrange
  multiplier statistic that was too small by a factor of ``nobs``.
  :issue:`10284`, :pr:`10286`
- ``get_prediction(weights=...)`` of regression results raised "weights has
  wrong shape" if the number of rows of ``exog`` was equal to the number of
  parameters, and did not check the length of ``weights`` against the number
  of rows otherwise. :issue:`10283`, :pr:`10285`
- ``get_prediction()`` called without ``exog`` now uses the offset and
  exposure of the model, as ``predict`` does, for GLM, GLMGam, the discrete
  count models and the binary models (Logit and Probit). Before, the offset
  and the exposure were ignored, so that the predicted mean, the standard
  errors, the intervals and the influence measures derived from them were
  wrong. ``get_prediction(offset=c)`` keeps the exposure of the model and
  ``get_prediction(exposure=c)`` keeps its offset. :pr:`10484`, :pr:`10562`
- ``fit(use_t=True)`` was ignored with ``cov_type="nonrobust"`` (the default)
  in the discrete models, which used the normal and chi-square distributions.
  :issue:`10307`, :pr:`10327`
- ``OLS.fit_regularized(L1_wt=0)`` returns wrapped results, so that the
  parameters are a pandas ``Series`` for pandas data. :issue:`4724`,
  :pr:`10262`
- The results of ``fit_constrained`` for a model created from a formula can be
  pickled. The formula is no longer passed to the transformed model.
  :issue:`9018`, :pr:`10279`
- ``RecursiveLS`` with constraints and a NumPy ``endog`` either raised an
  exception or silently dropped the constraints. :pr:`10282`
- ``recursive_olsresiduals`` no longer raises if the initial window is rank
  deficient, for example because of a duplicated row. The recursion starts
  from the pseudo-inverse. :issue:`8446`, :pr:`10493`
- ``linear_harvey_collier`` always dropped the first 3 recursive residuals.
  The default ``skip`` is now the number of regressors, which changes the
  result for models that do not have 3 regressors. :issue:`8446`,
  :pr:`10291`
- ``RollingOLS`` and ``RollingWLS`` with ``missing="skip"`` returned wrong
  estimates for the windows after a window with missing values. The number of
  missing values in each window was off by one and the cross-products were not
  recomputed after skipped windows. :pr:`10463`
- ``RollingOLS`` and ``RollingWLS`` with ``expanding=True`` raised an
  ``IndexError`` if ``window`` is larger than the number of observations. With
  ``expanding=False`` a ``ValueError`` is raised in this case. :issue:`9287`,
  :pr:`10402`, :pr:`10495`
- ``yule_walker`` raises a ``ValueError`` for a negative ``order`` instead of
  an ``IndexError``. :pr:`10346`

Generalized linear models and discrete models
---------------------------------------------

- ``predict`` preserves the observation index of a pandas Series for models
  with one predictor and additional parameters, such as ``NegativeBinomial``,
  ``GeneralizedPoisson``, ``ZeroInflatedPoisson``, ``BetaModel``,
  ``OrderedModel`` and ``MNLogit``. Previously, multiple observations raised
  an index length error and a single observation received an incorrect label.
  :pr:`10590`
- ``GEE`` used the number of columns of ``exog`` for ``df_model``,
  ``df_resid`` and the default ``ddof_scale``. These are now based on the rank
  of ``exog`` as in the other models, so that a singular design can no longer
  produce a negative ``df_resid`` or scale. :issue:`1928`, :pr:`10278`
- ``Logit.loglike`` and ``Logit.loglikeobs`` are computed with ``logaddexp``
  and no longer overflow for large absolute values of the linear predictor.
  :issue:`3923`, :pr:`10217`
- ``Probit.loglike``, ``loglikeobs``, ``score``, ``score_obs`` and
  ``hessian`` are computed in log space. The log-likelihood was clipped to a
  constant and the Hessian was ``nan`` for extreme values of the linear
  predictor. :issue:`10228`, :pr:`10229`
- ``HurdleCountModel.fit`` ignored the ``use_t`` argument. :pr:`10299`
- ``MNLogit.get_margeff`` with ``dummy=True`` or ``count=True`` failed or
  returned an incorrect covariance if the number of regressors differs from
  the number of outcome categories. :issue:`5488`, :pr:`10298`

Mixed models and imputation
---------------------------

- ``MixedLM`` returned wrong fixed effects, a scale that was too small, an
  infinite log-likelihood or failed with a singular matrix error if the random
  effects covariance or a variance component was at the boundary of the
  parameter space, i.e. a variance of 0. The degenerate directions are now
  dropped consistently in ``get_fe_params``, ``loglike``, ``score``,
  ``hessian`` and ``get_scale``, which is the limit as these variances go to
  zero. :issue:`10239`, :pr:`10242`, :pr:`10458`
- ``MixedLM`` with ``missing="drop"`` raised an ``IndexError`` because the
  rows with missing values were not dropped from all inputs, ``endog``,
  ``exog``, ``groups`` and ``exog_re``. If ``exog_vc`` is used and observations
  are dropped, a ``ValueError`` is raised because the variance component
  design matrices cannot be aligned. Drop the missing values before creating
  the model, or use ``from_formula``. :issue:`7978`, :pr:`10467`
- ``MixedLM.fit`` with ``do_cg=False`` raised an ``UnboundLocalError``.
  :issue:`7980`, :pr:`10328`
- ``MixedLM.fit_regularized`` failed for models with pandas data, for example
  from a formula: ``summary`` raised a ``KeyError`` and, with pandas 3,
  ``fit_regularized`` itself failed. :issue:`8902`, :pr:`10465`
- ``MICE.combine`` failed for results of models that do not have a
  ``_results`` attribute, for example those of ``PHReg``. :issue:`9136`,
  :pr:`10479`, :pr:`10557`

Time series analysis
--------------------

- ``range_unit_root_test`` documents the correct null hypothesis, a unit
  root, which is rejected for small statistics. The ``InterpolationWarning``
  reports the correct direction: it is raised if the statistic is outside the
  range of the tabulated critical values, which is 1% to 95%; it was also
  raised for statistics between the 90% and 95% critical values.
  :issue:`8978`, :pr:`10475`, :pr:`10568`
- ``UECMResults.bounds_test`` raised a ``TypeError`` for a model without lagged
  differences of the dependent variable (``lags=0``). :pr:`10578`
- :meth:`~statsmodels.tsa.arima_process.ArmaProcess.arma2ma` and ``arma2ar``
  raised an ``AttributeError`` if ``lags`` is not given. They default to
  ``nobs`` coefficients. :issue:`4464`, :pr:`10466`, :pr:`10569`
- ``acorr_ljungbox`` with ``auto_lag=True`` selected the wrong lag. The
  threshold used the lag-0 autocorrelation, which is always 1, so that the
  penalized criterion was never used, and the index of the maximum was
  returned instead of the lag. :issue:`6785`, :issue:`8338`, :issue:`8339`,
  :pr:`10236`
- ``adfuller`` with ``regression="n"`` and an even number of observations
  could choose a ``maxlag`` for which the regression fits exactly and the
  statistic is 0. This made ``coint`` return a statistic of 0 and a p-value of
  0.9859 for short samples with an even number of observations. The maximum
  lag now keeps one residual degree of freedom. :issue:`9375`, :pr:`10430`
- ``kpss`` raises a ``ValueError`` for a constant series and a
  ``MissingDataError`` for ``nan`` or infinite values. :issue:`8224`,
  :pr:`10432`, :pr:`10306`
- ``acorr_breusch_godfrey`` works for the results of ``AutoReg`` and ``ARDL``
  models. It failed with an ``AttributeError`` or a length mismatch, and the
  auxiliary regression omitted the lags and deterministic terms. The results
  are verified against R's ``lmtest::bgtest``. :issue:`9090`, :pr:`10400`
- The ``ARDL`` model defines ``k_constant``, so that ``rsquared`` and ``ess``
  of the results are correct if the model has a constant. :pr:`10389`
- ``UECMResults.apply`` and ``UECMResults.append`` return ``UECM`` results
  instead of ``ARDL`` results. :pr:`10293`
- ``ARIMA.fit(method="innovations_mle", start_params=...)`` raises a
  ``NotImplementedError`` that points to ``method="statespace"`` instead of a
  ``TypeError``. :issue:`8091`, :pr:`10482`
- ``SARIMAX.start_params`` failed with an ``IndexError`` for a series with a
  single observation. :issue:`9237`, :pr:`10472`
- The summary of ``SARIMAX`` printed the list of seasonal MA lags instead of
  the integer order if the seasonal MA order differs from the non-seasonal MA
  order. :issue:`9490`, :pr:`10231`
- ``ARIMA``, ``SARIMAX`` and ``VARMAX`` accept a polynomial ``trend`` that is
  a NumPy array. An array that is not one-dimensional with finite numeric
  values raises the ``ValueError`` that lists the valid trends. :issue:`9145`,
  :pr:`10365`, :pr:`10448`
- ``VARMAX.start_params`` failed for integer ``endog`` with a trend or
  ``exog``. :issue:`9014`, :pr:`10362`
- ``VARProcess.simulate_var`` for a stand-alone ``VARProcess`` counted the
  trend terms along the wrong dimension of ``coefs_exog`` and failed with an
  ``AttributeError``. A process with a time trend or exogenous regressors
  raises a ``ValueError`` that requires an offset. :issue:`8926`, :pr:`10446`
- The p-values of ``VECMResults`` (``pvalues_alpha``, ``pvalues_beta``,
  ``pvalues_gamma``, ``pvalues_det_coef`` and ``pvalues_det_coef_coint``) are
  computed with the survival function. They were exactly 0 for large
  t-values. :pr:`10445`
- Out-of-sample forecasts of ``ExponentialSmoothing`` with a seasonal
  component discarded the final in-sample update of the seasonal factors and
  used the factors from one step earlier. :issue:`9538`, :pr:`10230`
- For damped trends, the summary and the ``initial_trend`` entry of the
  parameters of the ``ExponentialSmoothing`` results reported a value derived
  from the damped initial trend instead of the estimated initial trend.
  :pr:`10294`
- ``ExponentialSmoothing`` accepts ``use_boxcox="log"`` in the constructor.
  The value was documented but raised a ``TypeError``. :issue:`9135`,
  :pr:`10361`
- ``MSTL.fit`` consumed the ``inner_iter`` and ``outer_iter`` entries of
  ``stl_kwargs``, so that a second call of ``fit`` used different settings.
  :pr:`10292`
- ``seasonal_plot`` did not sort the observations of each season, because the
  result of ``sort_index`` was discarded. :pr:`10275`
- ``lpol_fima`` and ``lpol_fiar`` returned ``nan`` for ``d = 0`` and
  coefficients with the wrong sign for ``d < 0``. They are computed with the
  recursion of the coefficients, which is valid for all ``d``. :pr:`10296`,
  :pr:`10366`
- ``companion_matrix`` no longer modifies the polynomial that is passed in,
  and accepts a tuple. :pr:`10289`
- ``dentonm`` raises a ``ValueError`` if the indicator has fewer observations
  than required by the benchmark and the frequency. The descriptions of
  ``indicator`` and ``benchmark`` in the docstring, low and high frequency,
  were swapped. :pr:`10303`
- Time series models ignore a ``PeriodIndex`` whose periods are not
  consecutive, i.e. one with gaps or repeated periods, and issue a
  ``ValueWarning``, as they already did for a ``DatetimeIndex`` without a
  frequency. Before, such an index was used, and forecasts were labelled with
  dates from inside the sample. Forecasting now raises an error unless an
  ``index`` is passed. Data with gaps that follow a calendar rule, such as
  trading days, can use a ``DatetimeIndex`` with a matching frequency (e.g.
  ``"B"`` or a ``CustomBusinessDay``); data with missing observations can be
  reindexed to consecutive periods. Consecutive periods with a frequency
  multiplier, such as ``"2Q"``, are still supported. For an ignored
  ``PeriodIndex`` the "Sample" line of the results summary shows integer
  positions, e.g. ``0 - 19``, and ``model.data.dates`` is ``None``, as for a
  ``DatetimeIndex`` without a frequency. :issue:`6247`, :issue:`8950`,
  :pr:`10572`, :pr:`10581`
- The asymptotic critical values and p-values of
  :meth:`~statsmodels.tsa.ardl.UECMResults.bounds_test` are looked up with the
  number of regressors, excluding the dependent variable, as in Pesaran, Shin
  and Smith (2001). They were those for one regressor too many, so the
  p-values were too small. Models with 10 regressors use the tables instead of
  simulation. :issue:`10546`, :pr:`10574`
- The statistic of :meth:`~statsmodels.tsa.ardl.UECMResults.bounds_test` was
  computed without the fixed regressors and the seasonal terms of the model,
  so that it did not depend on them. They are now included as controls, and
  the statistic matches R's ``ARDL::bounds_f_test``. A model that uses
  ``deterministic`` raises a ``NotImplementedError`` instead of ignoring the
  process. :pr:`10579`
- ``x13_arima_analysis`` reads a ``rawspec`` file as UTF-8 independent of the
  locale. Before, the file was decoded with the locale encoding, so that text
  with non-ASCII characters was garbled or failed, for example on Windows.
  A UTF-8 byte order mark is ignored, a file that is not valid UTF-8 raises a
  ``ValueError`` that says so, and ``rawspec`` accepts a ``pathlib.Path`` as
  documented. :pr:`10585`

Statistics and tests
--------------------

- The p-values of several tests were computed as one minus the cdf, which
  gives 0 or loses digits for large statistics. They now use the survival
  function. This affects ``survdiff``, the empirical likelihood ``ANOVA`` and
  ``OLSResults.el_test``, the GEE score tests, the chi-square tests of
  ``Table``, ``SquareTable`` and ``StratifiedTable``, the p-values of the UECM
  long-run coefficients and the bounds test of ``ARDL``, and the distance
  covariance test. :issue:`10274`, :pr:`10276`
- ``DescrStatsW``, ``CompareMeans`` and the functions that use them no longer
  overflow in the variance and in the standard errors of the mean difference
  for data with a large scale if the result is representable. :issue:`10252`,
  :pr:`10255`
- ``DescrStatsW`` rejects weights that are negative or not finite with a
  ``ValueError``. Non-finite weights gave ``nan`` or ignored the weight in
  ``quantile``. :pr:`10355`, :pr:`10371`
- ``StratifiedTable.test_null_odds(correction=True)`` applies the continuity
  correction of 0.5 only if ``|sum(a - E(a))| >= 0.5``, as ``mantelhaen.test``
  of R does. Before, ``min(0.5, |sum(a - E(a))|)`` was subtracted, which
  changed the statistic when the deviation is below 0.5. The statistic and the
  p-value of a table with zero variance are ``nan``, they were ``inf`` and 0.
  :pr:`10551`, :pr:`10583`
- The DerSimonian-Laird estimate of the between-study variance in
  ``combine_effects`` (``method_re="dl"`` or ``"chi2"``) is truncated at zero.
  A negative variance was used before, which could give a random effects mean
  outside the range of the effects. The ``h2`` and ``i2`` attributes of the
  results are at least 1 and 0, as in R's ``metafor``. They were below 1 and
  negative if ``q < k - 1``. ``combine_effects`` accepts lists and tuples.
  :issue:`10520`, :pr:`10522`, :pr:`10555`
- ``effectsize_2proportions`` applied the zero-cell continuity correction to
  all studies if any study had a zero cell. It is now applied only to the
  studies with a zero cell. With ``zero_correction="tac"``, the constants of
  the two samples were swapped and the sample sizes increased by the wrong
  amount, which gave a nonzero effect for a study without events in both
  samples if the sample sizes differ. Lists and tuples are accepted with
  ``zero_correction="tac"``. :issue:`10290`, :pr:`10301`, :pr:`10366`
- The Miettinen-Nurminen score test of the difference of two proportions
  (``method="score"`` and ``compare="diff"`` in ``test_proportions_2indep``
  and ``confint_proportions_2indep``) used a wrong linear coefficient in the
  constrained maximum likelihood estimate if the null ``value`` is not 0. The
  score test also returned ``nan`` for symmetric tables. The root finding for
  the confidence interval of the difference could start outside of the
  parameter space. :issue:`10269`, :pr:`10270`, :pr:`10366`
- ``power_equivalence_poisson_2indep`` and
  ``power_equivalence_neginb_2indep`` returned wrong results if ``nobs_ratio``
  is not 1. The ``ftotal`` null variance of
  ``power_equivalence_neginb_2indep`` and ``power_negbin_ratio_2indep`` is
  equal to the score variance if ``dispersion=0``. A ``nobs_ratio`` that is not
  positive and finite raises a ``ValueError`` in the two equivalence
  functions. :issue:`10519`, :pr:`10521`, :pr:`10555`
- ``effectsize_oneway`` with a scalar ``vars_`` and ``use_var="bf"`` returned
  an infinite effect size. :pr:`10455`
- The default ``value`` of
  :func:`~statsmodels.stats.rates.power_poisson_ratio_2indep` is now 1, the
  null hypothesis of equal rates. The previous default of 0 gave a power of 0,
  1 or ``nan``. The value must now be positive and finite.
  :func:`~statsmodels.stats.rates.power_negbin_ratio_2indep` has the same
  check; ``value=0`` gave a divide by zero warning, and ``value=inf`` was
  accepted by both functions. :issue:`10511`, :pr:`10512`, :pr:`10550`,
  :pr:`10580`
- :func:`~statsmodels.stats.proportion.power_binom_tost` returns a power of 0
  instead of a negative value if the rejection region of the exact test is
  empty. ``low`` and ``upp`` also accept list-like input, and a NaN ``p_alt``
  raises a ``ValueError``. :issue:`10507`, :pr:`10508`, :pr:`10580`
- ``proportions_chisquare`` raises a ``ValueError`` for a single sample
  without ``value``. The statistic was 0 and the p-value ``nan``.
  :pr:`10451`
- ``StratifiedTable.test_equal_odds`` (Breslow-Day and Tarone tests) returned
  ``nan`` or an inaccurate statistic when the pooled odds ratio is one or very
  close to one, and for tables with small fitted cells. The statistic is now
  computed accurately in these cases. :pr:`10540`
- ``gof_chisquare_discrete`` and ``gof_binning_discrete`` no longer restrict
  the support of the distribution to the range -1000 to 1000. The bins are
  chosen using the quantile function. The bins work for distributions for
  which the ``ppf`` of SciPy fails to converge, such as ``skellam`` with large
  parameters, and for edges of about 1e9 and larger. ``nsupp`` below 1 raises
  a ``ValueError``. :issue:`4359`, :pr:`10529`, :pr:`10555`
- ``rmspe`` truncated the percentage errors for integer input. :issue:`10509`,
  :pr:`10510`
- ``cov_cluster`` raised a ``ZeroDivisionError`` for a single cluster. It now
  raises a ``ValueError`` that requires at least 2 clusters. :pr:`10496`
- ``summary_col`` with ``fixed_effects`` placed the fixed effects indicator
  rows after the rows of ``info_dict``. They are placed after the parameter
  rows and before the model info rows. :issue:`9282`, :pr:`10218`
- ``corr_rank`` and ``corr_normal_scores`` return a 1 by 1 matrix for a single
  variable instead of a scalar. :pr:`10456`

Treatment effects
-----------------

- ``TreatmentEffect`` clipped the propensity score inconsistently. The point
  estimates used the unclipped propensity score, while the GMM moment
  conditions of ``ipw`` and ``aipw`` clipped at 0.01 and 0.99 and those of
  ``aipw_wls`` and ``ipw_ra`` clipped at 0.001 and 0.999. The new argument
  ``ps_bounds`` (default ``(0.001, 0.999)``) sets the bounds, which are now
  used for the point estimates and the inference of all methods. Results
  change if the estimated propensity score is outside of the bounds.
  :issue:`10222`, :pr:`10223`
- The GMM moment conditions of ``aipw_wls`` and ``ipw_ra`` assumed that the
  propensity score model has 6 parameters. They now use the number of
  parameters of the selection model. :pr:`10221`

Robust and multivariate statistics
----------------------------------

- ``CovDetMCD``, ``CovDetS`` and ``CovDetMM`` raised a ``ValueError``
  ("Percentiles must be in the range [0, 100]") if ``nobs < 2 * k_vars + 4``.
  :issue:`10241`, :pr:`10247`
- ``CovDetMCD``, ``CovDetS`` and ``CovDetMM`` failed for data with a single
  column, and ``cov_ogk`` and ``cov_tyler_regularized`` did not handle a
  single variable. :issue:`10300`, :pr:`10330`, :pr:`10456`
- ``CovDetMCD`` ranked the starting sets by the determinant, which overflows
  for moderately large ``k_vars``, and then selected the first starting set.
  The ranking uses the log determinant. The determinant used to normalize the
  covariance in ``cov_weighted`` (``weights_cov_denom="det"``), ``cov_tyler``
  (``normalize="det"``) and ``CovM`` is computed without overflow.
  :issue:`10295`, :issue:`10257`, :pr:`10329`, :pr:`10287`
- The deterministic starting sets of ``CovDetMCD``, ``CovDetS`` and
  ``CovDetMM`` were ranked with a wrong Mahalanobis distance, because
  ``mahalanobis`` was called with the mean in place of the covariance. The
  estimates can change slightly. :pr:`10457`
- ``CovDetS`` and ``CovDetMM`` ignored their ``breakdown_point`` argument and
  always used 0.5 for the tuning constant of the first-stage S-estimator.
  Values outside (0, 0.5] now raise a ``ValueError``, and small values in
  higher dimension no longer fail. :pr:`10468`, :pr:`10564`
- ``CovDetMCD`` raises a ``ValueError`` if the covariance of the selected
  subset is numerically singular, for example if at least ``h`` observations
  lie on a hyperplane, or if ``h`` is not larger than the number of variables.
  Before, the fit returned a result with a singular covariance.
  :issue:`10437`, :pr:`10464`, :pr:`10570`
- The Ledoit-Wolf starting estimate of the deterministic estimators omitted
  the shrinkage target, and divided by zero for a single variable.
  :issue:`10368`, :pr:`10378`, :pr:`10452`
- ``CovDetMCD``, ``CovDetS`` and ``CovDetMM`` raise a ``ValueError`` that
  names the column if the standardizing scale of a column is zero, i.e. more
  than half of the observations share one value. :issue:`10437`, :pr:`10461`
- ``robust.covariance.mahalanobis`` uses the pseudo-inverse if ``cov`` is
  numerically singular. Before, this was only done if the inverse failed, and
  a rank deficient matrix could return values of the order of 1e16 on some
  platforms. :pr:`10376`
- ``qn_scale`` returns ``nan`` if the data contain ``nan`` or infinite values.
  It returned a finite value or failed before. The underlying implementation
  no longer crashes for samples with 46341 or more observations.
  :pr:`10372`, :pr:`10244`
- ``PCA`` with ``gls=True`` applied the weights inverted: the data were
  divided by the square root of the weights instead of multiplied by it.
  :issue:`10256`, :pr:`10258`

Nonparametric
-------------

- The tricube kernel in ``statsmodels.nonparametric.kernels`` returns 0
  outside of its support, :math:`|u| > 1`, instead of the peak value. Results
  of :class:`~statsmodels.nonparametric.kernel_regression.KernelReg` and
  :class:`~statsmodels.nonparametric.kernel_regression.KernelCensoredReg`
  that use ``ckertype="tricube"`` change. Scalar arguments are accepted.
  :issue:`10541`, :pr:`10547`, :pr:`10550`
- ``pdf_kernel_asym`` and ``cdf_kernel_asym`` no longer raise
  ``ZeroDivisionError`` (array ``x``) or ``TypeError`` (scalar ``x``) when the
  sample has more than ``batch_size * 1000`` observations (10 000 with the
  default). Intermediate kernel arrays never exceed ``batch_size * 1000``
  elements, because the sample is split as well as the evaluation points.
  ``batch_size`` must be a number with ``batch_size * 1000 >= 1``, and a
  one-element ``x``, including a ``Series``, returns a scalar. :pr:`10577`,
  :pr:`10582`
- A scalar user-specified bandwidth is accepted by ``KDEMultivariate``,
  ``KDEMultivariateConditional``, ``KernelReg`` and ``KernelCensoredReg``.
  :issue:`4747`, :pr:`10272`
- The bandwidth of ``KDEMultivariate`` and ``KernelReg`` is validated. A
  user-specified bandwidth has to be positive, or non-negative for
  ``KernelReg`` where 0 is allowed for discrete variables, and an unknown
  bandwidth method raises a ``ValueError``. In ``KernelReg``, any string other
  than ``"cv_ls"`` selected the AIC bandwidth, including ``"normal_reference"``
  which now uses the normal reference rule of thumb. :pr:`10412`
- ``fast_linbin``, used by ``KDEUnivariate.fit(fft=True)``, wrote outside of
  its array for some inputs and dropped observations at the upper end of the
  grid. :issue:`9556`, :pr:`10232`

Duration models
---------------

- The Schoenfeld, score and martingale residuals of ``PHReg`` with
  ``ties="efron"`` used the Breslow approximation of the ties. They now use
  the approximation of the model and match R's ``survival::coxph``. The sum
  of the score residuals is the score of the partial likelihood, and the
  cluster robust covariance, which is built from the score residuals, was
  wrong with Efron ties. Observations that are not at risk at any failure time
  have a score residual of 0 (they were added to the risk set of the preceding
  failure time before). Observations that are not used no longer produce a
  ``nan`` robust covariance. :issue:`10288`, :pr:`10363`, :pr:`10377`

Other
-----

- ``missing="drop"`` with a boolean ``endog`` failed with "Data must be
  1-dimensional". :issue:`5445`, :pr:`10477`
- ``ExpandedNormal`` no longer modifies the array of cumulants that is passed
  in. :pr:`10281`

Stricter Input Validation
=========================

The following functions and methods now raise a ``ValueError`` for arguments
that are outside of their valid range. Depending on the function, invalid
values returned ``nan``, ``inf`` or a meaningless number, or failed with an
unrelated exception, for example a bare NumPy error or an ``IndexError``.
Functions that broadcast over an argument continue to accept arrays, and
reject arrays that contain an invalid value or ``nan``.

Confidence level and significance
---------------------------------

- ``alpha`` must be in the interval (0, 1) in: ``conf_int`` of the results of
  the models, of the prediction results and of the contrast results,
  ``zconfint``, the t and z confidence intervals of ``DescrStatsW`` and
  ``CompareMeans``, ``proportion_confint``, ``multinomial_proportions_confint``,
  ``confint_proportions_2indep``, the ``Table2x2`` confidence intervals,
  ``confint_poisson``, ``tolerance_int_poisson``,
  ``confint_quantile_poisson``, ``confint_mvmean_fromstats``,
  ``confint_noncentrality``, ``combine_effects``, ``tukeyhsd``, the
  confidence intervals of ``acf``, ``pacf``, ``ccf`` and ``pccf``,
  ``multipletests`` and ``fdrcorrection_twostage``. ``fdrcorrection`` accepts
  values in (0, 1]. ``signif`` of ``VARResults.test_whiteness`` has to be in
  (0, 1). The p-values that are passed to ``multipletests`` have to be in
  [0, 1]. :pr:`10391`, :pr:`10433`, :pr:`10392`, :pr:`10403`, :pr:`10380`,
  :pr:`10397`, :pr:`10399`, :pr:`10379`, :pr:`10413`, :pr:`10384`,
  :pr:`10415`, :pr:`10333`, :pr:`10390`, :pr:`10338`, :pr:`10310`,
  :pr:`10410`
- The power functions and classes of ``statsmodels.stats.power``
  (``TTestPower``, ``TTestIndPower``, ``NormalIndPower``, ``FTestPower``,
  ``FTestPowerF2``, ``FTestAnovaPower``, ``ttest_power``, ``normal_power``,
  ``normal_power_het``, ``ftest_anova_power``, ``ftest_power`` and
  ``ftest_power_f2``) check ``alpha``, and the F test classes check
  ``df_num`` and ``df_denom``. The two-sample Poisson and negative binomial
  power functions in ``statsmodels.stats.rates`` check ``alpha`` and
  ``nobs1``, ``chisquare_power`` checks ``nobs``, ``n_bins`` and ``alpha``,
  and ``samplesize_proportions_2indep_onetail`` checks ``alpha`` and
  ``power``. :pr:`10393`, :pr:`10447`, :pr:`10386`, :pr:`10385`,
  :pr:`10388`

Counts, sample sizes, weights and probabilities
-----------------------------------------------

- Counts must be non-negative and not larger than ``nobs``, and ``nobs`` and
  exposures must be positive, in: ``proportion_confint``, ``binom_test``,
  ``proportions_ztest``, ``proportions_chisquare``,
  ``confint_proportions_2indep``, ``effectsize_2proportions``,
  ``test_poisson``, ``test_poisson_2indep``, ``etest_poisson_2indep``,
  ``confint_poisson_2indep`` and the one-sample Poisson rate functions.
  ``value`` of ``proportions_ztest`` and ``proportions_chisquare`` has to be
  finite and in the valid range, and ``prop_var`` of ``proportions_ztest`` in
  (0, 1). :pr:`10380`, :pr:`10360`, :pr:`10358`, :pr:`10414`, :pr:`10345`,
  :pr:`10354`, :pr:`10373`, :pr:`10379`, :pr:`10388`, :pr:`10398`
- The proportions of ``proportion_effectsize`` have to be in [0, 1], the
  arguments ``diff`` and ``prop2`` of ``power_proportions_2indep`` and
  ``samplesize_proportions_2indep_onetail`` have to keep both proportions in
  [0, 1], and the probabilities of ``chisquare_effectsize`` have to be
  non-negative. :pr:`10383`, :pr:`10431`, :pr:`10425`, :pr:`10395`
- ``samplesize_confint_proportion`` checks its arguments: ``alpha`` has to be
  in (0, 1), ``half_length`` positive and ``proportion`` in [0, 1]. Invalid
  values, including nan in arrays, raise a ``ValueError``. They returned
  ``inf``, ``nan``, the sample size of ``abs(half_length)`` or a negative
  sample size. Lists are accepted. :pr:`10381`
- The dispersion of the Poisson and negative binomial power functions in
  ``statsmodels.stats.rates`` has to be non-negative. ``alpha`` of the
  ``NegativeBinomial`` link in ``statsmodels.genmod.families.links`` has to
  be positive. :pr:`10422`, :pr:`10426`, :pr:`10357`
- The cell counts of ``Table``, ``StratifiedTable`` and ``mcnemar`` have to be
  finite and non-negative. :pr:`10332`, :pr:`10350`, :pr:`10370`
- ``probs`` of ``DescrStatsW.quantile`` has to be in [0, 1] and ``ddof`` of
  ``ztest`` and ``zconfint`` has to be non-negative. :pr:`10404`,
  :pr:`10405`
- ``nobs`` has to be positive in ``test_cov_spherical``,
  ``test_cov_diagonal``, ``test_cov_blockdiagonal``, ``test_cov_oneway``,
  ``confint_mvmean_fromstats`` and ``confint_effectsize_oneway``, and
  ``effectsize_oneway`` requires non-negative ``vars_`` and positive ``nobs``.
  :pr:`10423`, :pr:`10433`, :pr:`10408`, :pr:`10356`
- The variances in ``combine_effects``, the diagonal of ``cov`` in
  ``cov2corr`` and the standard deviations in ``corr2cov`` have to be
  non-negative. :pr:`10415`, :pr:`10417`

Intervals, tuning parameters and shapes
---------------------------------------

- Equivalence intervals have to satisfy ``low < upp`` (``low <= upp`` for the
  Poisson functions) in the TOST methods of ``DescrStatsW`` and
  ``CompareMeans``, ``ttost_paired``, ``ztost``, ``binom_tost``,
  ``proportions_ztost``, ``binom_tost_reject_interval``,
  ``power_binom_tost``, ``power_ztost_prop``, ``tost_poisson_2indep``,
  ``nonequivalence_poisson_2indep`` and the Poisson and negative binomial
  equivalence power functions. ``equiv_margin`` of ``equivalence_oneway`` has
  to be positive and ``p_alt`` of ``power_binom_tost`` and
  ``power_ztost_prop`` has to be in [0, 1]. :pr:`10334`, :pr:`10382`,
  :pr:`10406`, :pr:`10411`, :pr:`10421`, :pr:`10420`
- ``c`` of ``mad`` and ``iqr`` has to be positive, the percentile bands of
  ``robust_kurtosis`` have to be ordered, ``scale`` of ``GLM.fit`` has to be
  positive, ``exposure`` of ``GLM.predict`` has to be positive and the length
  of ``offset`` of the binary discrete models has to match ``endog``.
  :pr:`10419`, :pr:`10416`, :pr:`10394`, :pr:`10428`, :pr:`10352`
- ``bw`` of ``KDEUnivariate.fit`` has to be positive, ``ncomp`` of ``PCA`` has
  to be positive, ``nlags`` of ``cov_hac`` and ``S_hac_simple`` has to be
  non-negative, ``degree`` of ``reset_ramsey`` has to be at least 2,
  ``exog_idx`` of ``variance_inflation_factor`` and ``axis`` of
  ``omni_normtest`` are checked, and ``local_fdr`` checks ``deg`` and
  ``nbins``. :pr:`10396`, :pr:`10341`, :pr:`10344`, :pr:`10343`,
  :pr:`10424`, :pr:`10347`, :pr:`10348`
- ``het_goldfeldquandt`` checks ``split`` and that both subsamples have
  residual degrees of freedom, ``compare_cox`` and ``compare_j`` check that
  the two models use the same number of observations, and ``fleiss_kappa``,
  ``cohens_kappa`` and ``to_table`` check the shape of their input.
  :pr:`10342`, :pr:`10369`, :pr:`10351`, :pr:`10314`, :pr:`10315`,
  :pr:`10316`
- The tuning constants of the robust norms in ``statsmodels.robust.norms`` are
  validated. ``t`` of ``HuberT``, ``a`` of ``RamsayE`` and ``AndrewWave``,
  ``c`` of ``TrimmedMean``, ``TukeyBiweight``, ``TukeyQuartic`` and
  ``StudentT``, and ``df`` of ``StudentT`` have to be positive, ``k`` of
  ``TukeyQuartic`` has to be a positive even integer, ``q`` of
  ``MQuantileNorm`` has to be in (0, 1), and the constants of ``Hampel`` have
  to be scalars with ``0 < a <= b <= c``. Zero, negative and ``nan`` values
  used to create norms with an all-zero or ``nan`` ``psi``, and array valued
  constants of ``Hampel`` failed in every method. The check applies to the
  constructor and to ``_set_tuning_param``, attributes that are assigned later
  are not checked. Arguments of the wrong type, for example ``None``, a string
  or a complex number, raise a ``TypeError``. :pr:`10418`

Time series
-----------

- ``nlags`` has to be non-negative in ``pacf``, ``kpss`` and
  ``levinson_durbin``, and smaller than the number of observations in ``acf``
  and ``acorr_breusch_godfrey``. ``ccf`` checks ``nlags`` against the number
  of observations, ``acovf`` requires a non-negative ``nlag``,
  ``levinson_durbin`` raises if fewer autocovariances than lags are available,
  ``nobs`` of ``arma_acovf`` has to be positive and ``nobs`` of ``q_stat``
  has to be larger than the number of lags. :pr:`10317`, :pr:`10319`,
  :pr:`10335`, :pr:`10337`, :pr:`10321`, :pr:`10318`, :pr:`10320`,
  :pr:`10297`, :pr:`10336`, :pr:`10409`
- ``max_ar`` and ``max_ma`` of ``arma_order_select_ic`` have to be
  non-negative, ``maxlags`` of ``VAR.fit`` and ``vecm.select_order`` has to be
  non-negative (and smaller than the number of observations for ``VAR.fit``),
  and ``steps`` of the VAR forecasts and ``periods`` of ``VARResults.irf``
  have to be non-negative. :pr:`10353`, :pr:`10322`, :pr:`10339`,
  :pr:`10429`
- ``period`` of ``seasonal_decompose`` and ``ThetaModel`` has to be at least
  2, ``order`` of ``detrend`` has to be non-negative, ``lamb`` of ``hpfilter``
  has to be positive, ``bkfilter`` requires ``low < high`` and more than
  ``2 * K`` observations, and ``cffilter`` requires ``low < high``.
  :pr:`10309`, :pr:`10359`, :pr:`10323`, :pr:`10305`, :pr:`10304`
- The smoothing and damping parameters of ``ExponentialSmoothing.fit`` have to
  be in [0, 1]. :pr:`10387`
