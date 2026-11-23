:orphan:

..
   New release-note bullets are added as fragments in docs/upcoming_changes/
   and moved into this file at release time with
   tools/releasing/collect_release_notes.py.

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
- Passing a row, an array with shape (1, n) or (1, ..., 1, n) with n > 1, as
  ``x`` to :func:`~statsmodels.tsa.filters.filtertools.recursive_filter` is
  deprecated, since ``x`` must be 1-D or an (n, 1) column. The row is still
  filtered as a 1-D series of n observations and returned with the shape of
  ``x``, and ``init`` now works with it instead of raising a ``ValueError``.
  Passing a row now raises a ``FutureWarning``, and it will raise a
  ``ValueError`` in statsmodels 0.17.0. Pass a 1-D array or an (n, 1) column
  instead. :pr:`10810`
- ``statsmodels.tsa.statespace.tools.set_mode``,
  ``statsmodels.tsa.tsatools.unintegrate`` and ``unintegrate_levels``, and
  ``statsmodels.tsa.vector_ar.output.normality_summary`` and
  ``hypothesis_test_table`` are deprecated. Nothing in statsmodels uses them
  and they are neither exported nor documented. Calling them now raises a
  ``FutureWarning``, and they will be removed after statsmodels 0.17.
  :pr:`10828`
- The ``statsmodels.tsa.arima_model`` module, whose ``ARMA`` and ``ARIMA``
  classes were removed in statsmodels 0.13 and only raise
  ``NotImplementedError``, will be removed after statsmodels 0.16. Use
  ``statsmodels.tsa.arima.model.ARIMA`` or
  ``statsmodels.tsa.statespace.sarimax.SARIMAX``. :pr:`10828`
- ``distance_indicators`` with ``x`` of more than one dimension issues a
  ``FutureWarning`` and returns the same result as before. After statsmodels
  0.17, a column will give the result of the 1-D series and other input will
  raise. :pr:`10858`
- A bool ``p`` in ``c_sja`` and ``c_sjt``, and so ``det_order=True`` in
  :func:`~statsmodels.tsa.vector_ar.vecm.coint_johansen`, is deprecated. It is
  still used as 0 or 1, with a ``FutureWarning``, and will raise after
  statsmodels 0.17. :pr:`10859`
- The ``return_results`` argument of :func:`~statsmodels.tsa.stattools.coint`
  is deprecated. It has no effect, issues a ``FutureWarning`` when it is not
  ``None``, and will be removed after statsmodels 0.17 is released.
  :pr:`10848`
- 2-D lag polynomials that the functions of ``statsmodels.tsa.arima_process``
  accepted, for example a column in ``arma_periodogram`` or a row in
  ``index2lpol``, are deprecated. They give the same result as before with a
  ``FutureWarning``, and will raise after statsmodels 0.17. :pr:`10861`
- ``arma_periodogram`` and ``ArmaProcess.periodogram`` divide by
  ``sqrt(2 * pi)`` instead of ``2 * pi``. A new ``normalization`` argument
  selects ``"2pi"`` (the spectral density) or ``"sqrt(2pi)"`` (the current
  result). Without it, a ``FutureWarning`` is issued, and the default will
  change to ``"2pi"`` after statsmodels 0.17. :pr:`10863`

Behavior changes
----------------

- ``MixedLM.score_sqrt`` is private. :pr:`10700`
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
- The ``DataFrame`` returned by ``predict`` of the results of ``MNLogit``, when
  ``exog`` is a ``DataFrame`` or the model was fit with a formula, has the
  categories of ``endog`` as columns, in the order of ``wendog`` with the
  reference category first, instead of 0, 1, ..., J - 1. The categories keep
  their type, so ``model.names`` and the labels in ``summary`` are strings but
  the columns of ``predict`` are, for example, integers for integer ``endog``.
  Predictions for an array ``exog`` are still arrays. :issue:`4804`,
  :pr:`10587`
- With ``missing="drop"``, a ``MissingDataError`` is raised if all
  observations are dropped because of missing values. An unknown value of
  ``missing`` raises a ``ValueError``. :issue:`4417`, :pr:`10535`,
  :pr:`10407`
- The name of the column ``"Cont. Int. Hi."`` in the data frame returned by
  ``DiscreteMargins.summary_frame`` is corrected to ``"Conf. Int. Hi."``.
  :pr:`10506`
- :func:`~statsmodels.stats.anova.anova_lm` raises a ``ValueError`` for a
  single model with a rank-deficient design matrix, for example a duplicated
  column, an empty cell of an interaction or more columns than observations.
  The Type I, II and III tables of such a model assigned arbitrary sums of
  squares, degrees of freedom and p-values to the redundant terms. The
  weighted design matrix is checked, so a factor level that only appears in
  observations with zero weight is also rejected. :issue:`5398`,
  :issue:`8506`, :pr:`10340`, :pr:`10622`
- :meth:`~statsmodels.genmod.generalized_linear_model.GLM.information` returns
  the positive definite expected (Fisher) information matrix ``X' W X / scale``
  as documented for ``LikelihoodModel.information``. It previously returned the
  negative of this matrix, which is the expected Hessian. This affects all
  families and links, also ``GLMGam``. The result is the expected information
  also for non-canonical links, where it differs from the observed information
  ``-model.hessian(params)``. To get the old value, use
  ``model.hessian(params, observed=False)``, or ``-model.information(params)``.
  :issue:`7826`, :pr:`10498`, :pr:`10628`
- A two-level categorical response, such as a bool, str or ``Categorical``
  column, in a formula for
  :class:`~statsmodels.genmod.generalized_linear_model.GLM` with the
  ``Binomial`` family now models the probability of the second level, as the
  ``glm`` function of R does. The expanded indicator columns of both levels were
  read as (successes, failures), so the first level was modeled and the signs
  of all coefficients were reversed. ``endog_names`` shows the modeled level.
  Other families, and a response with a single level in the data, raise a
  ``ValueError``. :issue:`10501`, :pr:`10503`, :pr:`10635`
- The bandwidths of discrete variables that are selected by
  :class:`~statsmodels.nonparametric.kernel_density.KDEMultivariate`,
  :class:`~statsmodels.nonparametric.kernel_density.KDEMultivariateConditional`
  and :class:`~statsmodels.nonparametric.kernel_regression.KernelReg`
  (``bw="normal_reference"``, ``"cv_ml"``, ``"cv_ls"`` and ``"aic"``) are kept
  in the range for which the kernels are defined: at most
  ``(levels - 1) / levels`` for an unordered variable and less than 1 for an
  ordered variable. The normal reference rule of thumb scaled with the numeric
  codes of a discrete variable, so that the density was negative, and the
  cross-validation returned bandwidths outside of the range. The
  cross-validation searches the best bandwidths inside the range, also in
  ``KernelReg`` where the bandwidths of discrete variables were not bounded.
  The default bandwidth still depends on the numeric codes.
  :issue:`3790`, :pr:`10515`, :pr:`10641`
- :func:`~statsmodels.graphics.functional.banddepth` with ``method="BD2"``
  computed the pairs of curves that contain a curve from the ranks of the
  curve at each point. This is wrong if curves cross or have equal values,
  since the same two curves have to bound the curve at all points. The depth
  now counts the pairs of curves that contain the curve at all sampled points,
  with values equal to a bounding curve counted as inside, so the depth of
  such data changes. Curves are not interpolated, so only the sampled values
  matter. ``data`` can be any array_like, for example a ``DataFrame``.
  :issue:`7105`, :pr:`10513`, :pr:`10638`
- :func:`~statsmodels.graphics.functional.banddepth` with ``method="MBD"``
  counts values equal to a band boundary as inside. Previously, tied values
  were assigned arbitrary ranks, so depths could depend on the input row order.
  The depths of data with tied values change.
  :pr:`10691`
- Many functions in ``statsmodels.stats``, ``statsmodels.tsa`` and other
  modules now raise a ``ValueError`` for invalid arguments that were accepted
  before, see *Stricter Input Validation* below.
- ``KDEUnivariate.evaluate`` with a kernel of bounded support (``"epa"``,
  ``"tri"``, ``"uni"``, ``"biw"``, ``"triw"``, ``"tric"``, ``"cos"`` and
  ``"cos2"``) returns one density per point, which is 0.0 where no
  observation is in the support of the kernel. It returned a scalar ``nan``
  for the whole array of points if one of them was outside the support of all
  observations. ``cdf`` and ``entropy``, and so ``sf`` and ``cumhazard``, of
  these kernels returned ``nan`` for most data. They now integrate the density
  over the support of the observations, in the units of the data, so that the
  ``cdf`` starts at 0 and ends at 1. :issue:`7195`, :pr:`10632`
- ``filter``, ``smooth`` and ``fit`` of state space models with a stationary
  initialization raise a ``LinAlgError`` if a parameter is fixed at a unit
  root, for example with ``fix_params``, instead of returning results with an
  ``llf`` of 0.0 that were computed from an initial state covariance with
  negative variances. Loglikelihoods at parameters with a valid stationary
  covariance are unchanged, but the fit of a very ill-conditioned model, whose
  optimizer used to survive negative covariances along its path, can end at a
  different optimum. For example, a ``SARIMAX`` model with ``order=(15, 0, 0)``
  and ``trend="c"`` for ``np.linspace(0, 2 * np.pi, 1000)`` ends at an ``llf``
  of 1667.85 instead of 6835.06.
  :issue:`5459`, :pr:`10747`
- ``powerdiscrepancy`` with ``lambd=0`` (``"loglikeratio"``, the default, which
  is the G-test) returns valid test statistics when observed counts contain
  zeros, evaluating :math:`x \ln(x)` to 0, instead of returning
  ``(nan, nan)``. :issue:`10703`, :pr:`10704`
- In-sample ``MNLogit`` results predictions with ``exog`` omitted return a
  ``DataFrame`` with the training row index and response categories as columns
  when the model was fit with pandas data (``endog`` or ``exog``) or with a
  formula, including a formula on a dict or a recarray (the index is then
  0, 1, ..., nobs - 1). Previously, these predictions were ndarrays. Models
  fit with NumPy data and no formula, and predictions with explicit NumPy
  ``exog``, still return ndarrays. Use ``np.asarray`` on the predictions if an
  ndarray is needed. :issue:`4541`, :pr:`10735`
- ``powerdiscrepancy`` (and the G-test) with ``lambd=0`` returns valid test
  statistics when observed counts contain zeros, evaluating :math:`x \ln(x)`
  to 0, instead of returning ``(nan, nan)``. :issue:`10703`, :pr:`10704`
- The bias correction (third) step of
  :func:`~statsmodels.tsa.arima.estimators.hannan_rissanen.hannan_rissanen`
  filters the second-step residuals with the inverse of the MA polynomial,
  :math:`\theta(B) W_t = Z_t`, as in Brockwell and Davis (2016), section
  5.1.4. Previously :math:`W_t` was a first-order approximation of this
  filter. **This changes the estimated coefficients** whenever the third step
  is applied, which is the default (``unbiased=None``) if the second-step
  estimates are stationary and invertible. The estimates are closer to the
  maximum likelihood estimates: for 50 simulated ARMA(2, 1) series with 1000
  observations the median distance to the estimates of
  ``ARIMA(..., trend="n").fit()`` fell from 0.083 to 0.020. This also changes
  the final estimates of ``ARIMA.fit(method="hannan_rissanen")`` and of
  :func:`~statsmodels.tsa.arima.estimators.gls.gls` with
  ``arma_estimator="hannan_rissanen"``. It changes the starting parameters of
  ``arma_estimator="innovations_mle"`` (``ARIMA.fit(method="innovations_mle")``
  and the default of ``gls``), but not their final estimates, which agree up
  to the tolerance of the optimizer. If the corrected parameters are
  non-stationary or non-invertible, the second-step estimates are returned
  with an ``EstimationWarning``. :issue:`6159`, :pr:`10712`
- :func:`~statsmodels.tsa.stattools.diebold_mariano_test` raises a
  ``ValueError`` for inputs for which it returned a meaningless result or
  raised another exception before. This is the case if the loss differential
  is constant up to rounding error, for example if the two forecasts are
  identical (``nan`` before, or a huge statistic with a p-value of 0 if the
  forecast errors are constant); if ``lags`` is not smaller than the number
  of observations, or ``horizon`` is not smaller than it when ``lags`` is
  not given or ``harvey_adj=True`` (a statistic that is not meaningful
  before, which is 0 with ``harvey_adj=True`` if ``horizon`` is the number of
  observations or one more); and if ``criterion`` does not return one loss
  per observation (a ``TypeError`` before for a scalar, otherwise a
  statistic from a loss series of the wrong length). The results that were
  returned for other input are unchanged. :issue:`10753`, :pr:`10755`
- :func:`~statsmodels.tsa.stattools.acf` and
  :func:`~statsmodels.tsa.stattools.pacf` return 1.0 for lag 0 and ``nan`` for
  lags 1 and higher when the input series is constant, for every method of
  ``pacf``. Before, ``acf`` returned ``nan`` at all lags if the mean of the
  series is exactly representable, and otherwise autocorrelations close to 1
  caused by the rounding error of the mean, for example 0.99 at lag 1 for 100
  observations of 101.37. ``pacf`` returned numbers or ``nan`` depending on
  the method and the constant. :func:`~statsmodels.tsa.stattools.ccf` returns
  ``nan`` for all lags when either series is constant. The ``qstat`` and
  ``pvalues`` of ``acf(..., qstat=True)`` and the statistics and p-values of
  :func:`~statsmodels.stats.diagnostic.acorr_ljungbox` are ``nan`` for a
  constant series. :issue:`10647`, :pr:`10701`
- Newton's method (``method="newton"``, which is the default of the
  discrete models and ``PHReg``) shortens a step that gives a
  log-likelihood that is not finite or lower. Fits that did not take such a
  step are unchanged. Fits from starting values for which the full steps
  diverge or need many iterations, for example ``Poisson`` with a large
  ``exog`` and zeros as starting values, now converge. If no step is found,
  the fit stops at the last parameters with ``converged`` ``False`` and a
  ``ConvergenceWarning``, where the parameters were ``nan`` and
  ``converged`` was ``True``. A step that is not a direction of descent, for
  example because the Hessian is not positive definite, is only taken if it
  decreases the log-likelihood, and the iteration stops if the change of the
  parameters of the full Newton step, not that of a shortened step, is at
  most ``tol``. ``GLM.fit`` with ``method="newton"`` shortens only a step
  that is not finite if the link is not the canonical link of the family.
  :pr:`10605`
- ``GLM.fit_regularized`` scaled the log-likelihood by the number of rows
  instead of the sum of the ``freq_weights``, for ``method="elastic_net"``,
  ``method="l1_slsqp"`` and the ridge fit with ``L1_wt=0``. With frequency
  weights this changed the effective penalty, so the estimates and the
  selected variables differed from the fit on the data with each row
  repeated according to its weight. The estimates of existing fits with
  ``freq_weights`` change. The estimates of fits without ``freq_weights`` are
  unchanged. :pr:`10705`
- ``GLM.fit_regularized`` with ``L1_wt=0`` (the ridge fit) used the scale
  estimated at the current parameters in the log-likelihood and the score,
  where the fits with ``L1_wt`` greater than 0 and ``method="l1_slsqp"`` use a
  scale of 1. For families with an estimated scale, for example ``Gaussian``,
  ``Gamma`` and ``Tweedie``, the estimates differed from the solution of the
  penalized problem, for example from the closed-form ridge estimate of the
  linear model, and the fit often warned that the optimization may have
  failed. The ridge fit now uses a scale of 1, so that its estimates change
  for these families, by much if the estimated scale is far from 1. The
  estimates of ``Binomial``, ``Poisson`` and ``NegativeBinomial``, which have
  a fixed scale, are unchanged. Multiplying ``alpha`` by the scale estimated
  at the estimates gives the penalty of earlier versions, where that fit
  converged; the example of ``GLM.fit_regularized`` shows how. :pr:`10776`
- The marginal effects of ``GEE`` results fit with ``use_t=True`` use the t
  distribution with ``df_resid_inference`` degrees of freedom, as the
  parameters do. Their p-values and confidence intervals change, and the
  ``z`` and ``Pr(>|z|)`` columns of ``summary_frame`` are named ``t`` and
  ``Pr(>|t|)``. Results with the default ``use_t=False`` are unchanged.
  :pr:`10792`
- ``ExponentialSmoothing`` in ``statsmodels.tsa.holtwinters`` with
  ``trend="mul"``, ``damped_trend=True`` and ``seasonal="add"`` estimated its
  parameters by minimizing residuals computed with ``l * phi * b`` instead of
  ``l * b**phi``, while the fitted values, the reported ``sse`` and the
  forecasts use ``l * b**phi``. The estimates of this model now minimize the
  reported ``sse``, so its parameters, fitted values and forecasts change, for
  example the ``sse`` of a fit to the first 240 months of the co2 data falls
  from 590.8 to 27.0. Results of all other trend, damping and seasonal
  combinations are unchanged. :pr:`10829`
- :func:`~statsmodels.tsa.filters.filtertools.recursive_filter` filters an
  (n, 1) column, such as a one-column ``DataFrame``, along its rows and
  returns the values of the 1-D series. Before, no recursion was applied to a
  column, which was returned unchanged without an error, and ``init`` raised a
  ``ValueError``. Input with more than one row and more than one column, which
  was filtered across the columns of each row, now raises a ``ValueError``. So
  does input with more than two dimensions that is neither a row with shape
  (1, ..., 1, n) nor a single observation with shape (1, ..., 1), such as an
  (n, 1, 1) array, which was returned unchanged. Results for 1-D input are
  unchanged. :pr:`10810`
- The ``lags`` arguments of the ``ArmaProcess`` methods ``acovf``, ``acf``,
  ``pacf``, ``periodogram``, ``impulse_response``, ``arma2ma`` and
  ``arma2ar`` raise a ``ValueError`` for values below 1. Before, ``lags=0``
  was taken as the default and returned ``nobs`` values, and a negative
  value gave a numpy error or an empty result. An integral float is now
  accepted by all of them. :pr:`10835`
- ``pacf_burg`` (and ``pacf`` with ``method="burg"``), ``kpss`` with
  ``regression="c"``, ``range_unit_root_test``, ``pccf`` and ``q_stat`` treat
  an (nobs, 1) column, such as a one-column ``DataFrame``, as the 1-D series
  instead of raising, and raise a ``ValueError`` for input with more than one
  column. ``q_stat`` silently took a (1, k) row as k values at lag 1 and gave
  wrong statistics; it now raises. Results for 1-D input are unchanged.
  :pr:`10837`
- :func:`~statsmodels.tsa.stattools.ccf` and
  :func:`~statsmodels.tsa.stattools.ccovf` treat an (nobs, 1) column, such as
  a one-column ``DataFrame``, as the 1-D series. Before, two columns gave a
  meaningless 2-D result without an error, for example an (nlags, nobs) array
  from ``ccf``, and a column next to a 1-D series raised an error from SciPy.
  Input with more than one column now raises a ``ValueError``. Results for
  1-D input are unchanged. :pr:`10811`
- :func:`~statsmodels.tsa.stattools.pccf` with ``method="yw"`` or ``"ywm"``
  (the default) no longer depends on the scales of ``x`` and ``y``: rescaling
  one series, for example by ``1e8``, made every lag ``nan``. Series that are
  collinear to working precision give ``nan`` for every lag, as exactly
  collinear series did, instead of inaccurate values. When the recursion
  reaches a singular prediction error covariance matrix, the later lags are
  ``nan`` on every platform; before, they depended on rounding, and the lag
  computed just before could be dropped. Results for other input change only
  by rounding. :pr:`10843`
- When ``den`` is longer than ``num``, ``deconvolve`` returns an empty ndarray
  instead of a list as the quotient, and a float (or complex) copy of ``num``
  as the remainder instead of ``num`` itself. The values are unchanged.
  :pr:`10864`
- ``arma_pacf``, ``arma_impulse_response``, ``arma2ma`` and ``arma2ar`` raise
  a ``ValueError`` for ``lags`` below 1, as the ``ArmaProcess`` methods do since
  :pr:`10835`, instead of NumPy errors, and ``arma_periodogram`` raises for an
  integer ``worN`` below 1 instead of returning empty arrays.
  ``ArmaProcess.from_estimation`` no longer replaces ``nobs=0`` with the number
  of observations of the results. :pr:`10860`
- ``ArmaProcess.generate_sample`` uses the ``nobs`` of the process as the
  default sample size, as documented, instead of 100. Processes created with
  another ``nobs``, including those from ``ArmaProcess.from_estimation``, now
  generate ``nobs`` values when ``nsample`` is not given. :pr:`10862`
- ``levinson_durbin`` with ``isacov=True`` and
  :func:`~statsmodels.tsa.stattools.diebold_mariano_test` treat an (nobs, 1)
  column, such as a one-column ``DataFrame``, as the 1-D series instead of
  raising, and raise a ``ValueError`` for input with more than one column.
  Results for 1-D input are unchanged. :pr:`10849`
- The default ``nlags`` of :func:`~statsmodels.tsa.stattools.acf` with
  ``missing="drop"`` or ``"conservative"`` counts only the non-missing
  observations. It counted the NaNs, so it could return more lags, or raise
  when ``nlags`` was not given. The values at each lag are unchanged.
  :pr:`10847`

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
- Add ``MultiComparison.dunnett_t3`` for Dunnett's T3 test of all pairwise
  comparisons of means with unequal variances. The p-values and the
  simultaneous confidence intervals use the studentized maximum modulus
  distribution with the Welch degrees of freedom of each pair. The
  ``plot_simultaneous`` method of the results raises ``NotImplementedError``.
  :issue:`9472`, :pr:`10469`
- Add the ``one_sided`` argument to
  :func:`~statsmodels.tsa.filters.hp_filter.hpfilter`. With
  ``one_sided=True``, the trend at time t only depends on the observations up
  to and including t. It is computed with the Kalman filter of the state space
  form of the HP filter. :issue:`6947`, :pr:`10588`
- Add :func:`~statsmodels.stats.multicollinearity.vif`,
  :func:`~statsmodels.stats.multicollinearity.vif_ridge` and
  :func:`~statsmodels.stats.multicollinearity.collinear_index` for the
  diagnosis of multicollinearity. They are available from
  ``statsmodels.stats.api``. ``vif`` computes the variance inflation factors of
  all columns of a data matrix, or of a covariance or correlation matrix, with
  one eigenvalue decomposition. It is about 90 times faster than
  ``variance_inflation_factor`` called for each column for 20 variables and
  1000 observations, about 300 times faster for 60 variables, and returns a
  ``Series`` for a ``DataFrame``. :issue:`1669`, :pr:`2380`, :pr:`7711`,
  :pr:`10623`
- Add :func:`~statsmodels.stats.inter_rater.krippendorff_alpha`, Krippendorff's
  alpha for the agreement of any number of raters, with missing ratings, for
  nominal, ordinal, interval and ratio data. It is available from
  ``statsmodels.stats.api``. :pr:`10738`
- Add the arguments ``alternative`` and ``kernel`` to
  :func:`~statsmodels.tsa.stattools.diebold_mariano_test`. ``alternative``
  selects a one-sided test, ``"larger"`` if ``forecast_b`` is more accurate
  or ``"smaller"`` if ``forecast_a`` is. ``kernel="uniform"`` estimates the
  long-run variance with equal weights up to ``lags``, as in Diebold and
  Mariano (1995) and Harvey et al. (1997), and uses ``horizon - 1`` lags by
  default; a ``ValueError`` is raised if its estimate of the variance is not
  positive. With ``harvey_adj=True`` this gives the results of ``dm.test`` in
  the R package forecast. The results of the default arguments are
  unchanged.
  :issue:`10753`, :pr:`10755`
- Add :func:`~statsmodels.tsa.stattools.clark_west_test`, the test of Clark
  and West (2007) of equal predictive accuracy of nested models, and its
  result class ``ClarkWestResult``. It adjusts the mean squared prediction
  error of the larger model for the noise from estimating parameters that
  are zero under the null, which biases the comparison of the unadjusted
  errors, for example with ``diebold_mariano_test``, against the larger
  model. The function is available from ``statsmodels.tsa.api``.
  :issue:`10754`, :pr:`10756`
- Add the option ``step_halving`` to ``fit(method="newton")``. It is
  ``True`` to halve a step, up to 30 times, that gives a log-likelihood
  that is not finite or lower, ``"finite"`` to halve only a step that is
  not finite, and ``False`` for the full Newton step. The default is
  ``True``, and ``GLM`` uses ``"finite"`` if the link is not the canonical
  link of the family. :pr:`10605`
- Add the option ``use_t`` to ``GEE.fit``, ``OrdinalGEE.fit`` and
  ``NominalGEE.fit``. With ``use_t=True``, p-values, confidence intervals,
  ``t_test`` and ``f_test`` use the t distribution with the number of groups
  minus the number of parameters as degrees of freedom. :issue:`8610`,
  :pr:`10781`
- Add ``cov_type="kauermann_carroll"`` to ``GEE.fit``, the Kauermann-Carroll
  small-sample correction of the robust covariance. It scales the residuals
  of each group by the inverse square root of one minus the leverage, where
  ``"bias_reduced"`` (Mancl-DeRouen) uses the inverse. :issue:`1183`,
  :pr:`10781`
- Add ``cov_type="fay_graubard"`` and ``cov_type="morel_bokossa_neerchal"``
  to ``GEE.fit``, the small-sample corrections of the robust covariance of
  Fay and Graubard (2001) and Morel, Bokossa and Neerchal (2003), with the
  constants ``b = 0.75``, ``d = 2`` and ``r = 1``. :issue:`1183`,
  :pr:`10794`

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
- ``KDEMultivariateConditional.cdf`` and ``KDEMultivariateConditional.imse``
  evaluate the kernel of the exogenous variables only once for each
  observation. The bandwidth selection with ``bw="cv_ls"`` is about 2 times
  faster with one continuous exogenous variable and about 3.5 to 4 times
  faster with two exogenous variables, and ``cdf`` is about 1.5 to 2 times
  faster. :issue:`6076`, :pr:`10626`
- Lag order selection fits the nested sequence of models from a single
  blocked QR decomposition, shared by a new private helper. This applies to
  ``adfuller`` with ``autolag``, ``VAR.select_order``, ``pacf_ols``,
  ``pacf_yw`` (Levinson-Durbin) and ``ardl_select_order``. ``VAR.select_order``
  is 12 to 43 times faster, ``pacf_ols`` about 90 times faster for long series,
  and ``ardl_select_order`` evaluates thousands of candidate models in a
  fraction of a second instead of seconds. ``pacf_ols`` with
  ``efficient=True`` demeans the series first, which is more accurate when the
  mean is large relative to the variation. :issue:`2276`, :pr:`10571`
- ``zivot_andrews`` computes the regressions of all candidate break periods
  from one QR decomposition of the regressors that do not depend on the
  break, instead of fitting one regression per period. The cost grows
  linearly instead of quadratically with the number of observations, so the
  test is about 25 to 65 times faster for 2,000 observations and 50 to 140
  times faster for 5,000. The statistics are more accurate when the
  regressors are close to collinear, so for such series the statistic can
  change in the later digits. The new values agree with R (urca) and with
  high-precision arithmetic. :issue:`10601`, :pr:`10602`, :pr:`10624`
- ``OLS``, ``WLS`` and ``GLS`` have the new fit option ``method="chol"``,
  which solves the normal equations with a Cholesky factorization of the
  k x k cross product matrix of the whitened regressors instead of computing
  the pseudoinverse or the QR factorization of the n x k design matrix. For
  3,000,000 observations and 10 regressors, ``fit`` is about 20 times faster
  than with the default ``method="pinv"``, and creating and fitting the model
  needs more than 10 times less peak memory: the fit itself allocates no
  array of the size of the design matrix, while ``"pinv"`` and ``"qr"`` need
  about three times its size. The method squares the condition number of
  the design with unit-norm columns and is less accurate than ``"pinv"`` and
  ``"qr"`` on ill-conditioned designs, for example with an uncentered
  regressor with a large mean next to the constant; it should only be used
  when speed and memory usage matter more than numerical accuracy, see the
  Notes of ``fit``. Collinear regressors by the tolerance of R's ``lm`` fall
  back to the pseudoinverse with a ``SingularMatrixWarning``. The
  pseudoinverse ``pinv_wexog`` is only computed when a covariance estimator
  or an influence measure needs it.
  :issue:`9317`, :pr:`10679`
- ``get_margeff`` with the default ``at="overall"`` computes the delta method
  Jacobian of the average marginal effects directly instead of averaging the
  Jacobian at each observation. Memory use no longer grows with the number of
  observations times the number of marginal effects times the number of
  parameters. For ``MNLogit`` with 4 categories, 28 regressors and 20,000
  observations the peak memory drops from 3 GB to 0.09 GB. This also applies
  to ``get_margeff`` of ``GLM`` and ``GEE``. :issue:`4158`, :pr:`10722`

Bug Fixes
=========

Regression and linear models
----------------------------

- ``OLS.fit`` did not detect an exactly rank-deficient design, for example a
  duplicated constant, from about 1,000 observations on. The smallest singular
  value of such a design is a few multiples of the machine precision times the
  largest, independent of the number of observations, and was above the fixed
  cutoff. No ``SingularMatrixWarning`` was issued and the parameters were of
  the order of 1e12. The rank is now the number of singular values above
  ``max(nobs, k) * eps`` times the largest, the rule of
  ``numpy.linalg.matrix_rank``. This is used for the rank and for the
  pseudoinverse in both ``method="pinv"`` and ``method="qr"``. Designs with a
  condition number above about ``1 / (nobs * eps)`` are now treated as
  rank-deficient and the minimum-norm solution is returned. :pr:`10714`
- ``OLS.fit(method="qr")`` computed the covariance of the parameters as
  ``inv(R'R)``, which squares the condition number of the design, and solved
  ``R b = Q'y``, which has no solution for a rank-deficient design. It now uses
  the SVD of ``R``, so the covariance has the accuracy of the default
  ``method="pinv"`` and a rank-deficient design returns the minimum-norm
  solution with a ``SingularMatrixWarning``. The ``pinv_wexog`` attribute is
  computed from the same factorization instead of a second SVD. :pr:`10714`
- ``OLS.hessian(params, scale=None)``, the Hessian of the profile
  (concentrated) log-likelihood, was missing a factor of 2 in the term that
  involves ``X'X``. The results of ``OLS.fit_regularized`` with
  ``profile_scale=True`` change because they use this Hessian.
  :issue:`9469`, :pr:`10473`, :pr:`10556`
- ``GLSAR.hessian_factor`` raises a ``NotImplementedError``. It inherited
  ``GLS.hessian_factor``, which returned one value for each of the ``nobs``
  observations, but the whitened design of ``GLSAR`` has ``nobs - order`` rows
  that mix neighboring observations, so the values did not reproduce its
  cross-product. :pr:`10675`
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
- ``OLS``, ``WLS``, ``GLS``, ``GLSAR`` and ``QuantReg`` raise a
  ``MissingDataError`` when ``endog``, ``exog``, ``weights`` or ``sigma``
  contain ``inf`` or ``nan`` after missing values are handled, so that
  ``missing="drop"`` still removes ``nan`` but not ``inf``. Before, only
  ``nan`` and ``inf`` in ``exog`` were detected, and only if ``hasconst`` was
  not ``False``. ``-inf`` in ``exog``, and non-finite values of the other
  arrays, raised a ``LinAlgError`` in ``fit`` or gave parameters that are all
  ``nan``. :issue:`1642`, :issue:`3468`, :pr:`4969`
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
- ``forecast``, ``get_forecast``, ``predict`` and ``get_prediction`` of the
  ``RecursiveLS`` results accept the ``exog`` values of the new periods for
  out-of-sample predictions. Before, they raised a ``ValueError`` unless a
  3-dimensional ``design`` matrix was given. :issue:`9708`, :pr:`10651`
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
- ``get_prediction`` of regression, GLM and discrete results with
  ``use_t=True`` and ``cov_type="cluster"`` (also two-way clustering),
  ``"hac-panel"`` or ``"hac-groupsum"``, and for linear models
  ``"cluster-crv3"`` or ``"cluster-jk"``, used the t distribution with
  ``df_resid`` degrees of freedom instead of ``n_groups - 1`` (the smaller
  number of groups minus one for two-way clustering) as in ``conf_int`` and
  ``t_test``, so the confidence interval of the mean was too narrow. Nothing
  changes with ``df_correction=False`` or ``use_t=False``. The prediction
  interval for a new observation (``obs_ci``) now uses the same degrees of
  freedom, which is conservative. :issue:`10743`, :pr:`10744`

Generalized linear models and discrete models
---------------------------------------------

- ``score_test`` with a one-dimensional ``exog_extra`` for the discrete
  models and the ``GLM`` result method raised an ``IndexError`` in 0.15.0. A
  one-dimensional ``exog_extra`` is a single additional variable, as before.
  :pr:`10609`
- ``score_test`` with ``exog_extra`` used the number of columns of
  ``exog_extra`` as the number of constraints. If some of the columns are
  linear combinations of other columns or of the model's ``exog``, for
  example the linear term that a spline basis contains, the degrees of
  freedom were too large, so the p-value was too large and the test
  conservative, or the test raised a ``LinAlgError`` for exactly redundant
  columns. The number of constraints is now the rank of ``exog_extra``
  after projecting out ``exog``, and the statistic is computed from a basis
  of the independent part. If all columns of ``exog_extra`` are in the span
  of ``exog``, a ``ValueError`` is raised. An ``exog_extra`` with ``nan`` or
  infinite values or with a different number of rows than the model's
  ``exog`` also raises a ``ValueError``. This applies to
  ``GLM.score_test``, the ``score_test`` method of ``GLM`` results and of
  the discrete models. :issue:`5364`, :pr:`10680`
- ``predict`` preserves the observation index of a pandas Series for models
  with one predictor and additional parameters, such as ``NegativeBinomial``,
  ``GeneralizedPoisson``, ``ZeroInflatedPoisson``, ``BetaModel``,
  ``OrderedModel`` and ``MNLogit``. Previously, multiple observations raised
  an index length error and a single observation received an incorrect label.
  :pr:`10590`
- The analytic ``hessian`` of ``ZeroInflatedPoisson`` with logit inflation,
  the default, set the cross derivatives between the inflation and the count
  parameters to zero, and the formula for them was also incorrect. The
  standard errors were too small by about 10% for the two intercepts in
  a simulated example with 3000 observations, and by up to 0.6% for the
  RandHIE data. ``bse``, ``cov_params``, ``conf_int`` and the p-values of the
  results change accordingly, and ``bse`` now agrees with the standard errors
  from a numerical Hessian. :pr:`10723`
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
- ``MNLogit.from_formula`` raised a ``ValueError`` if the response is a str,
  bool or categorical variable. The response can now be a single categorical
  variable, which is treated like the corresponding ``endog`` of the array
  API, so that the categories label the equations. Any other response with
  multiple columns still raises a ``ValueError``. ``Logit`` and ``Probit``
  with such a response in a formula are not changed. :issue:`9475`,
  :pr:`10639`
- ``fit`` of ``ConditionalLogit``, ``ConditionalPoisson`` and
  ``ConditionalMNLogit`` ignored the keyword arguments of the optimizer,
  ``callback``, ``retall`` and ``warn_convergence``. They are now passed on,
  and the results of ``ConditionalLogit`` and ``ConditionalPoisson`` have
  ``mle_retvals`` and ``mle_settings``. A ``cov_type`` other than
  ``"nonrobust"`` was ignored and now raises a ``NotImplementedError``.
  :issue:`8849`, :pr:`10630`
- The estimation of the autocorrelation of
  :class:`~statsmodels.genmod.cov_struct.Autoregressive` with ``grid=False``
  raised a ``ValueError`` ("unable to find right bracket") if the estimate was
  larger than about 0.75, and failed in ``scipy.optimize.brent`` if the
  minimum was exactly at a point of the search. The estimate is now found for
  values up to within about 1e-6 of one. A larger estimate still raises the
  ``ValueError``. A zero estimate is a ``float``. :issue:`8122`, :pr:`10636`
- The Cox-Snell ``GLMResults.pseudo_rsquared``, which is also shown in
  ``summary``, used the number of rows instead of the sum of the
  ``freq_weights``. With frequency weights the value differed from the value
  for the data with each row repeated according to its weight. :pr:`10684`
- The ``score`` of ``TruncatedLFGeneralizedPoisson`` and
  ``TruncatedLFNegativeBinomialP`` was incorrect for every ``p`` except
  ``p=1`` of the generalized Poisson and ``p=2`` of the negative binomial
  model. The part of the score that comes from the probability of the
  truncated counts used the default ``p`` of the untruncated model. This
  includes the default ``p=2`` of ``TruncatedLFGeneralizedPoisson`` and
  ``HurdleCountModel`` with ``dist="negbin"`` and ``p`` other than 2.
  ``fit`` stopped before reaching the maximum of the log-likelihood with a
  ``ConvergenceWarning``, so the fitted ``params``, ``bse`` and ``llf`` of
  these models change for valid input. For example, the ``alpha`` of the
  default ``TruncatedLFGeneralizedPoisson`` changes from 0.099 to 0.219 and
  ``llf`` from -3582.06 to -3554.15 for 3000 simulated negative binomial
  counts. The results are unchanged for ``p=1`` of the generalized Poisson
  and ``p=2`` of the negative binomial model, and they change for every
  other ``p``. :pr:`10736`
- The ``hessian`` and ``hessian_factor`` of ``NegativeBinomialP`` had an
  incorrect cross derivative of the regression parameters and ``alpha`` if
  ``p`` is not 1 or 2. ``bse``, ``cov_params``, ``conf_int``, the p-values and
  the robust (``cov_type="HC0"``) covariance of the results change for these
  values of ``p``, and ``bse`` now agrees with the standard errors from a
  numerical Hessian. In 20 simulated samples of 3000 counts, the standard
  errors of the constant and of ``alpha`` were too large by a factor of 1.2 to
  2.2 for ``p=1.5`` and by 1.1 to 1.3 for ``p=1.8``, while they were within 8%
  for ``p=1.2``. ``bse`` was ``nan`` for ``p=2.5`` and ``p=3``.
  ``fit(method="newton")``, which uses the Hessian, now converges for these
  values of ``p``. In one sample of 1000 counts it stopped after 100
  iterations without convergence for ``p=1.5``, ``2.5`` and ``3``, and it had
  a ``nan`` log-likelihood for ``p=1.8``. ``score_test`` with a tuple
  ``exog_extra`` also changes. The results for ``p=1`` and ``p=2`` are not
  changed. :pr:`10737`
- ``loglike``, ``loglikeobs``, ``score`` and ``hessian`` of
  ``NegativeBinomial`` with ``loglike_method="nb2"`` or ``"nb1"`` used the
  last element of ``params`` as ``log(alpha)`` instead of ``alpha`` in a model
  instance that was not fit, and in the model of results that were pickled
  and unpickled. For example, ``NegativeBinomial(endog, exog).loglike(params)``
  differed from ``llf`` of the results with these ``params``, -260.73 instead
  of -249.90 for 150 simulated counts, and ``score`` was not zero at the
  maximum likelihood estimate. Results that were pickled before ``llf`` was
  first used computed it from the unpickled model, so ``llf``, ``aic``,
  ``bic``, ``llr``, ``prsquared`` and ``summary`` were incorrect after
  unpickling; in the example, ``llr`` was -9.99 instead of 11.65. ``fit``
  and the results computed by ``fit`` are not changed.
- ``MNLogit.get_margeff`` with ``method="eyex"`` or ``method="dyex"`` raised
  a ``ValueError`` (array broadcasting) if ``at`` is ``"overall"`` or
  ``"all"``, and whenever the number of regressors differs from the number of
  outcome categories. If the two numbers are equal and ``at`` is ``"mean"``,
  ``"median"`` or ``"zero"``, it returned incorrect marginal effects and
  standard errors. :pr:`10746`
- The IRLS fit of ``GLM`` from ``start_params`` or starting values of the
  mean for which the first step overflows raised a ``ValueError`` about the
  weights. A step that gives a mean, weights or a deviance that is not
  finite is now halved towards the last parameters, up to 30 times, and for
  the canonical link of the family a step that increases the deviance is
  halved as well. A ``ValueError`` that names ``start_params`` is raised if
  the mean or the weights at ``start_params`` are not finite. A Newton fit
  from such starting values returned ``nan`` parameters with ``converged``
  ``True``. :issue:`5394`, :pr:`10605`
- ``GLMGamResults.predict`` and ``get_prediction`` no longer apply the
  model's formula a second time when ``exog`` is a design matrix created from
  the fitted model's ``design_info`` or model specification. Before, the fitted
  exog frame could not be reused for any linear term that is not a plain
  column, for example ``np.log(x1)``, ``I(x1**2)``, an interaction with a
  factor, or ``C(g)``. An ``exog`` that carries the ``design_info`` of another
  design used to be re-evaluated by the formula and now raises a
  ``ValueError``. :issue:`9562` :pr:`10752`
- ``cov_type`` of ``GEE.fit``, ``OrdinalGEE.fit`` and ``NominalGEE.fit`` is
  case-insensitive. A value such as ``"Bias_Reduced"`` raised an
  ``UnboundLocalError``. If the expansion of the ``"bias_reduced"`` or
  ``"kauermann_carroll"`` covariance of a constrained fit fails, ``fit`` now
  returns ``None`` with a ``ConvergenceWarning``. It used to return results
  without that covariance, which raised a ``ValueError`` when it was used.
  :pr:`10791`
- ``GEE.fit`` with ``cov_type="bias_reduced"`` or ``"kauermann_carroll"``
  raises a ``ValueError`` that names the group if a group has leverage 1, for
  example a single observation with its own dummy variable. It raised a bare
  ``LinAlgError``, or for some families returned a meaningless covariance.
  The same ``ValueError`` is raised if the covariance of the parameters is not
  finite, where the standard errors were ``NaN`` or a ``LinAlgError`` was
  raised. :pr:`10792`
- ``summary`` of the marginal effects of ``GEE`` results raised a
  ``ValueError``. :pr:`10792`
- ``exposure`` is used correctly in ``TruncatedLFPoisson``,
  ``TruncatedLFNegativeBinomialP`` and ``TruncatedLFGeneralizedPoisson``, and
  in ``predict(which="prob")`` of ``ZeroInflatedPoisson``,
  ``ZeroInflatedNegativeBinomialP`` and ``ZeroInflatedGeneralizedPoisson``.
  The log of the exposure was taken twice, which gave a ``nan``
  log-likelihood or wrong parameters if an exposure is smaller than 1 and
  raised a ``ValueError`` for an array exposure in the negative binomial and
  generalized Poisson versions. ``ZeroInflatedPoisson`` ignored the exposure
  in ``predict(which="prob")``. A model with ``exposure=e`` now gives the
  same results as with ``offset=np.log(e)``.
  ``predict`` of ``NegativeBinomialP`` and ``GeneralizedPoisson`` always takes
  the log of an ``exposure`` that is passed, as ``Poisson`` does, so an
  explicit ``exposure=0`` now gives ``log(0)`` instead of being treated as no
  exposure. :issue:`8842`, :pr:`10505`

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
- ``MixedLM.fit`` could stop below the maximum of the log-likelihood, report that
  it had not converged, or end at a variance of 0, because ``MixedLM.score``
  dropped the signs of the packed square-root parameters and so was not the
  derivative at the vector that the optimizer holds. These fits now reach the
  maximum, so estimates, standard errors and log-likelihoods of existing
  models can change. ``MixedLM.score`` with a packed vector that has negative
  entries now returns the derivative at that vector. In 72 simulated models
  fitted with the default options, 16 ended more than 1e-4 below the maximum
  found by lme4 and 12 reported no convergence, and none does now.
  :pr:`10700`
- ``MixedLMResults.bse_re`` raised a ``ValueError`` for models with variance
  components and pandas data, for example from a formula. :pr:`10702`
- ``MICE.combine`` failed for results of models that do not have a
  ``_results`` attribute, for example those of ``PHReg``. :issue:`9136`,
  :pr:`10479`, :pr:`10557`
- ``MICEData`` no longer resets the index of the data. ``MICEData.data``, and
  so the data returned by ``next_sample`` and passed to ``history_callback``,
  keeps the index of the input, which can contain duplicate labels, so that
  imputed values can be matched to the rows of the input. Rows in which all
  variables are missing are still dropped before imputation.
  :issue:`6532`, :pr:`10499`, :pr:`10629`
- ``MICEData`` issues a ``ValueWarning`` when it drops rows in which all
  variables are missing. These rows were dropped without a warning.
  :issue:`6532`, :pr:`10777`

Time series analysis
--------------------

- The ARIMA GLS estimator now uses the complete AR and MA lag polynomials
  when whitening the regression data. Seasonal terms were previously
  omitted, and nonconsecutive lags were treated as consecutive, producing
  incorrect regression coefficient estimates. This affects
  ``statsmodels.tsa.arima.estimators.gls.gls`` and ``ARIMA.fit`` for a model
  with ``exog`` or a trend that is estimated with a ``method`` other than
  ``"statespace"``, or with ``method="statespace"`` and ``gls=True``. The
  estimates of such models with seasonal terms or nonconsecutive lags change.
  :pr:`10713`
- ``ARDL.predict`` with a ``start`` after the first out-of-sample observation
  returned wrong forecasts for a model with ``exog``, ``fixed`` or seasonal
  terms, and raised an ``IndexError`` if ``dynamic`` was used. The forecasts
  between the end of the sample and ``start`` are now computed first.
  ``ARDLResults.get_prediction`` raised a ``ValueError`` for such a ``start``.
  :issue:`9699`, :pr:`10664`, :pr:`10665`
- ``MLEModel.loglike`` returns ``-inf`` instead of raising a ``LinAlgError``
  if the states are initialized as stationary and the transition matrix has a
  unit root, so that the stationary covariance of the initial state does not
  exist. A derivative free optimizer, such as ``method="powell"``, can try
  parameters that are transformed to the boundary of the stationary region.
  The fit of the dynamic factor model in the example notebook failed with this
  error on some platforms. :pr:`10611`
- ``range_unit_root_test`` documents the correct null hypothesis, a unit
  root, which is rejected for small statistics. The ``InterpolationWarning``
  reports the correct direction: it is raised if the statistic is outside the
  range of the tabulated critical values, which is 1% to 95%; it was also
  raised for statistics between the 90% and 95% critical values.
  :issue:`8978`, :pr:`10475`, :pr:`10568`
- ``zivot_andrews`` with ``maxlag=0`` and ``autolag=None`` used the default
  number of lags, 12 * (nobs / 100) ** 0.25, instead of no lagged differences.
  A negative ``maxlag`` raises a ``ValueError``. :pr:`10627`
- ``AutoReg.predict`` with ``dynamic`` raised an ``IndexError`` if ``start``
  was after the first out-of-sample observation. If ``dynamic`` was after the
  end of the sample, the lags were ignored in the forecasts before it.
  ``AutoRegResults.get_prediction`` raised a ``ValueError`` if ``start`` was
  after the first out-of-sample observation. The forecasts between the end of
  the sample and ``start`` are now computed first. :pr:`10666`
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
- ``ETSModel`` with ``initialization_method="estimated"`` (the default) and a
  seasonal component estimates ``seasonal_periods - 1`` initial seasonal
  states and computes the last one so that the initial seasonals sum to 0
  (additive seasonality) or to ``seasonal_periods`` (multiplicative
  seasonality), as R's ``forecast::ets`` does. Before, the last initial
  seasonal was fixed to 0 or 1 and the heuristic start values were shifted or
  divided by it. With multiplicative seasonality and a small last heuristic
  seasonal this moved the start far from the heuristic initial states, and the
  fit ended in a much worse optimum than ``initialization_method="heuristic"``
  with negative forecasts for a positive series. ``start_params`` also changed
  the ``initial_seasonal`` attribute of the model in place. The results of
  seasonal fits with estimated initial states change: the optimum, the
  ``initial_level`` and the initial seasonals. For example, the damped
  additive model of the ``austourists`` data in the ETS example notebook has
  an initial level of 24.23 instead of 29.45 (R: 24.01). ``aic``, ``aicc``
  and ``bic`` of a given fit are computed as before, because ``df_model``
  still counts all initial seasonals. The covariance of the parameters is
  computed for the free initial seasonals and mapped to the computed one.
  Before, the Hessian of all initial states was singular, so the standard
  errors of the initial level and the initial seasonals were ``nan`` or
  arbitrarily large, for example in the summary of the notebook example. If
  ``initial_seasonal.<m-1>`` is fixed or bounded, all initial seasonals are
  free parameters, as before. A fixed ``initial_level`` is a restriction, as
  the initial seasonals still sum to 0 or m, while before the last one was
  fixed to 0 or 1. For example, with ``fix_params({"initial_level": 30})``
  the additive model with an additive trend of the ``austourists`` data has
  a log-likelihood of -155.84 instead of -150.51. :issue:`8551`, :pr:`10710`
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
- ``x13_arima_analysis`` and ``x13_arima_select_order`` remove the output files
  written by X-12/X-13 from the temporary directory. Before, each call left
  files such as ``.out``, ``.err``, ``.log``, ``.d11``, ``.d12`` and ``.d13``
  behind. :issue:`8277`, :pr:`10674`
- ``coint_johansen``, ``select_coint_rank`` and the error bands of the
  impulse responses of a ``VAR`` returned complex eigenvalues and eigenvectors
  with NumPy 2.5 and later, which ``np.linalg.eig`` returns as complex arrays
  even if all the eigenvalues are real, and ``coint_johansen`` warned with a
  ``ComplexWarning``. The results are real again. :pr:`10613`
- ``coint_johansen`` computes the eigenvalues and eigenvectors from a QR
  decomposition and a singular value decomposition (Doornik and O'Brien,
  2002) instead of ``np.linalg.eig`` of the product of the moment matrices.
  The eigenvalues are real and in [0, 1), and the eigenvalues and statistics
  are accurate for nearly cointegrated data, where the old code returned
  eigenvalues larger than 1 or negative, or raised a ``LinAlgError``. It
  raises a ``LinAlgError`` if the lagged differences, levels and differences
  are linearly dependent, which also covers collinear columns that returned
  meaningless numbers before, and a ``ValueError`` if there are fewer
  observations than columns of the regression. :pr:`10471`, :pr:`10637`
- ``smoothed_state_autocovariance`` does not warn that the univariate filter
  diagonalized the observation covariance when it extends the model beyond
  the sample, since it does not use the disturbance covariances. :pr:`10613`
- ``arma_order_select_ic`` no longer warns about the starting values of the
  candidate models that fall back to zeros. :pr:`10613`
- ``hamilton_filter`` and the starting values of ``DynamicFactorMQ`` no longer
  warn about the unused scale of a regression without residual degrees of
  freedom. :pr:`10613`
- ``ThetaModelResults.plot_predict`` with ``alpha=None`` no longer adds a
  legend without labels, which warned. :pr:`10613`
- ``SVAR.sirf_errband_mc`` and ``IRAnalysis.errband_mc`` with ``svar=True``
  show one ``ConvergenceWarning`` with the number of replications whose
  maximum likelihood fit did not converge, instead of one warning for each of
  them. :pr:`10613`
- At a unit root the stationary initialization could return an initial state
  covariance with negative variances instead of raising, for example
  ``-4.99e291`` for ``SARIMAX(np.zeros(10)).filter([1, 1])``. The filter then
  reported a loglikelihood of exactly 0, above that of any stationary model.
  This is now detected: ``filter`` raises a ``LinAlgError`` saying that the
  model is not stationary to numerical precision, and ``MLEModel.loglike``
  returns ``-inf``, as for the unit roots that already raised (:pr:`10611`).
  Unit roots of this kind are rejected, and explosive roots farther from the
  unit circle are not changed: for an AR(1) model the check applies up to a
  coefficient of about ``1 + 6e-5``, while coefficients of ``1 + 1e-4`` or
  ``1.2`` still give a negative initial variance and an ``llf`` of 0.0.
  :issue:`5459`, :pr:`10747`
- ``IRAnalysis.plot`` and ``IRAnalysis.plot_cum_effects`` ignored the
  argument ``plot_params``. It is now applied to the response curves, so that
  their color, line width and other Matplotlib ``Line2D`` properties can be
  set. Error bands and horizontal reference lines keep their default styles.
  A key of ``plot_params`` that is not a ``Line2D`` property, for example
  ``"fontsize"``, was ignored before and now raises an ``AttributeError``.
  :issue:`380`, :pr:`10720`
- ``KalmanFilter.loglike`` of a state space model with exact diffuse
  initialization and a concentrated scale (``filter_concentrated``) is now
  equal to ``loglikeobs().sum()``, to ``llf`` and to the value with
  ``fixed_scale``, and it no longer depends on ``conserve_memory``. The
  concentrated log-likelihood left out the finite innovations of the diffuse
  periods, and with ``conserve_memory`` options that discard the
  log-likelihood it ignored ``loglikelihood_burn`` in exact diffuse models. The
  value of ``loglike``, which is the objective of ``fit``, and so the fitted
  parameters change for models in which the diffuse periods contain finite
  innovations, for example several observed series with fewer diffuse states,
  also if some observations are missing, or a diffuse state that is observed
  only later in the sample. Models without such innovations, for example a
  single series with a local level or local linear trend, also with missing
  values, and ``SARIMAX`` with ``concentrate_scale=True`` and exact diffuse
  initialization, are unchanged. :issue:`10741`, :pr:`10742`
- ``MarkovRegression``, ``MarkovAutoregression`` and ``MarkovSwitching`` now
  raise a ``MissingDataError`` if ``exog_tvtp`` contains NaN or inf. Before,
  such values were accepted and ``fit`` returned a log-likelihood of NaN
  without an error. The ``missing`` option does not apply to ``exog_tvtp``.
  :pr:`10766`
- ``ETSModel.loglike``, ``ETSModel.smooth`` and ``ETSModel.fit`` with
  ``start_params`` raised an error for parameters that are not float64 in
  native byte order, for example a big-endian, ``float32`` or integer array.
  :issue:`10770`, :pr:`10795`
- ``VARMAX.start_params``, and so ``VARMAX.fit``, raised a ``ValueError`` for
  big-endian ``endog``. :issue:`10770`, :pr:`10795`
- ``reorder_missing_matrix``, ``reorder_missing_vector``,
  ``copy_missing_matrix``, ``copy_missing_vector``, ``copy_index_matrix`` and
  ``copy_index_vector`` in ``statsmodels.tsa.statespace.tools`` raised a
  ``ValueError`` for big-endian arrays when ``inplace=False``, and for a
  ``missing`` or ``index`` mask of ``bool`` or ``int64`` dtype although the
  masks are documented as bool. They now accept big-endian arrays, and masks of
  bool or any integer dtype, where a nonzero value counts as True. With
  ``inplace=True`` a big-endian array that is changed in place still raises a
  ``ValueError``. :issue:`10770`, :pr:`10795`
- ``VAR.fit`` now raises a ``SingularMatrixWarning`` if the lagged regressors
  are perfectly collinear, matching the warning already raised by ``OLS`` for
  a rank-deficient design matrix, using the same ``nobs``-scaled rank rule
  (:pr:`10714`). The same cutoff now chooses the minimum-norm solution, so
  the coefficients of such a design are no longer of the order of 1e11, as
  they could be with the earlier cutoff of 1e-15.
  ``VAR.select_order`` collects this check across all lag
  orders it evaluates internally and raises at most one warning per call
  instead of one per lag order. :issue:`10646`, :pr:`10694`
- ``MLEModel.score`` with ``transformed=False``, which ``fit`` uses, applied
  the Jacobian of the parameter transformation instead of its transpose, and
  did not zero the rows of fixed parameters. For models whose Jacobian is not
  symmetric, such as SARIMAX with an AR order of two or more and VARMAX, the
  solvers that use the score (``bfgs``, ``lbfgs`` with ``optim_score``,
  ``basinhopping``, ``minimize`` with a gradient method) got a wrong gradient
  away from the optimum and often stopped short of it, for example with a
  ``ConvergenceWarning``. :pr:`10818`
- ``fit(method="newton")`` of state space models and of Markov switching
  models used the score and the Hessian with respect to the constrained
  parameters, evaluated at the values of the unconstrained ones, and for
  state space models a Newton step that was ``nobs`` times too long, so that
  it often stopped with a ``ConvergenceWarning`` short of the optimum.
  Newton's method passes its extra arguments to the score and the Hessian,
  and ``MLEModel.hessian`` with ``transformed=False`` gives the Hessian with
  respect to the unconstrained parameters (by the chain rule) instead of
  raising or returning derivatives with respect to the constrained ones.
  ``MLEModel.hessian`` has a new keyword-only argument ``average``.
  ``fit(method="ncg")`` of state space models no longer raises. :pr:`10817`
- ``coint_johansen`` and ``select_coint_rank`` with ``k_ar_diff=0`` used the
  levels ``y_t`` instead of the lagged levels ``y_{t-1}``, so the eigenvalues,
  the trace and maximum eigenvalue statistics and the eigenvectors were wrong
  and the selected cointegration rank could change. With ``det_order=0``, on
  the log of real GDP, consumption and investment in ``macrodata`` the trace
  statistic for rank 0 was 39.4 instead of 29.6, which selected rank 1 instead
  of 0. For a ``det_order`` of -1 and 0 the statistics now equal the
  likelihood ratio statistics of ``VECM``. Results with ``k_ar_diff`` of 1 or
  more do not change. :issue:`7459`, :pr:`10807`
- ``DeterministicProcess.out_of_sample`` without any deterministic terms
  returned the in-sample rows on the in-sample index instead of ``steps``
  rows on the forecast index, and ``DeterministicProcess.range`` beyond the
  sample returned the wrong rows in that case. :pr:`10834`
- ``arma_order_select_ic`` accepts ``ic`` as any sequence of str, such as an
  ndarray or an ``Index``, as documented, instead of only a list or tuple.
  :pr:`10840`
- ``grangercausalitytests`` with an empty ``maxlag`` raised a NumPy error
  instead of its own ``ValueError``. :pr:`10839`
- ``breakvar_heteroskedasticity_test`` raised an ``UnboundLocalError`` for a
  ``subset_length`` that is a NumPy integer or a whole-number float, such as
  ``np.int64(40)`` or ``40.0``, and for invalid values. It accepts every
  numeric type and raises a ``ValueError`` for invalid values. :pr:`10838`
- ``add_trend`` with ``trend="n"`` returned a list unchanged and raised an
  ``AttributeError`` for a tuple. It returns an ndarray, like the other
  trends. :pr:`10833`
- ``_Var.predict`` (deprecated) returned neither the fitted values nor the
  residuals, with the wrong number of rows unless there was one lag, and kept
  returning the first result after a refit. It returns the fitted values.
  :pr:`10822`
- ``varfilter`` in ``statsmodels.tsa.varma_process`` with 3-d lag matrices
  used the variables in reverse order and returned the wrong number of rows
  unless there were two lags, and a 2-d filter with one lag and several
  variables was taken as a single lag polynomial and returned one column. It
  now computes the documented filter. :pr:`10824`
- ``TsaDescriptive.periodogram`` and ``TsaDescriptive.plot4`` (deprecated)
  raised an ``AttributeError`` since 0.12.0, because they called a function
  that was removed then. They compute the periodogram again, as before 0.12.0.
  :pr:`10821`
- ``_Var.covmat`` (deprecated) scaled the covariance of the coefficients by
  the residual sum of squares instead of the residual variance, so it was too
  large by the residual degrees of freedom. :pr:`10823`
- A state space model that was pickled or deep-copied after it had been used
  ignored new parameters once restored: ``loglike``, ``filter``, ``smooth``
  and ``fit`` used the system matrices of the last evaluation before
  pickling. This also affected results saved with ``save`` and read with
  ``load``, whose ``model.loglike`` returned the fitted log-likelihood for
  any parameters and whose model, refit, returned the starting parameters.
  Pickles written by earlier versions load correctly. :issue:`3290`,
  :pr:`10816`
- ``grangercausalitytests`` with a one-element ndarray or ``Series`` ``maxlag``,
  such as ``np.array([3])``, ran the tests for every lag up to that value
  instead of only that lag, as a list does. The "Insufficient observations"
  error gave a maximum allowable lag that was one too small in most cases, and
  ``x`` without exactly two columns raised an unrelated error; it now raises a
  ``ValueError`` that says so. :pr:`10846`
- ``constrain_stationary_multivariate`` with float32 or complex64 input and
  ``transform_variance=False`` raised ``ValueError: Buffer dtype mismatch`` on
  NumPy < 2 when ``order + k_endog`` is at least 4, because an intermediate
  array was promoted to double precision. :pr:`10857`
- ``fit(method="cg")`` did not pass its extra arguments to the objective and
  the score. State space models and Markov switching models pass
  ``transformed=False`` this way, so the conjugate gradient solver
  optimized the log-likelihood with the parameters taken as constrained and
  stopped far from the optimum, often reporting convergence, for example
  after one iteration at an llf of -698.8 where the optimum is -456.0.
  Results of models that do not use extra arguments are unchanged.
  :pr:`10819`
- ``index2lpol`` raised an ``IndexError`` for a tuple ``index``, and ``ar2arma``
  raised a ``TypeError`` for a tuple ``ar_des``. :pr:`10858`
- ``c_sja`` and ``c_sjt`` raised unrelated errors for a non-integer ``p`` and
  returned a view of their table, so changing a result changed later calls.
  A non-integer ``p`` raises a ``ValueError``, and a copy is returned.
  :pr:`10859`
- ``CalendarTimeTrend`` returned its terms on a ``PeriodIndex`` for a
  ``DatetimeIndex``, so a ``DeterministicProcess`` that combined it with other
  terms gave misaligned frames with NaN, and on a ``PeriodIndex`` with a
  frequency other than ``freq`` it counted time in the wrong unit.
  :pr:`10853`
- ``DeterministicProcess.range`` ignored the step of a ``RangeIndex`` whose
  step is not 1, so it returned the wrong rows or raised a ``KeyError``.
  :pr:`10852`
- ``coint`` returned ``critical_values`` as a list for ``trend="n"``. It is an
  ndarray of NaN values, as for the other trends. :pr:`10848`
- ``coint`` returned ``critical_values`` as a list for ``trend="n"``. It is now
  an ndarray, as it is for the other trends. :pr:`10848`
- Lag polynomials with more than one dimension in the functions of
  ``statsmodels.tsa.arima_process`` and in ``ArmaProcess`` raised NumPy or
  SciPy errors that did not name the argument. They raise a ``ValueError``
  that does. :pr:`10861`
- :func:`~statsmodels.tsa.filters.filtertools.recursive_filter` raised NumPy or
  SciPy errors for an ``ar_coeff`` or ``init`` given as an (n, 1) column, and
  a misleading length error for a row ``init``. A column is accepted, and
  other 2-d input raises a ``ValueError`` that names the argument.
  :pr:`10872`

Statistics and tests
--------------------

- ``statsmodels.stats.power`` validates that a root returned by a solver
  attains the requested power. Failed solutions now return ``NaN`` with a
  ``ConvergenceWarning`` instead of returning a misleading finite result or
  raising an error, also when solving for ``alpha``, which now finds
  significance levels far below 1e-5. Noncentral-t CDF calculations retry a
  reflected tail when CDFLIB returns ``NaN`` and use the limit 0 or 1 if that
  fails too, preserving already finite values. This can turn previously missing
  t-test powers into finite results; it does not guarantee relative accuracy
  in extremely small tails. :pr:`10552`
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
- The score confidence interval of the risk ratio
  (``method="score"`` and ``compare="ratio"`` in
  :func:`~statsmodels.stats.proportion.confint_proportions_2indep`) was wrong,
  and sometimes complex, if a count is equal to ``nobs``, for example 10 of 10
  events in one sample. The interval often did not contain the estimate. The
  limits are now found by inverting the score test, with a bracket that also
  works for ``alpha`` of 0.5 and larger. If both counts are equal to ``nobs``
  the limits have a closed form. :issue:`10538`, :pr:`10539`, :pr:`10640`
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
- :func:`~statsmodels.stats.proportion.binom_test_reject_interval` accepts
  array_like ``value``, ``nobs`` and ``alpha`` and returns integer bounds,
  also for the unused bound of the one-sided alternatives, which is broadcast
  to the shape of the result. Scalar input still returns Python integers.
  ``value`` has to be in [0, 1], ``alpha`` in (0, 1) and ``nobs`` a positive
  integer, otherwise a ``ValueError`` is raised. Array input raised a
  ``TypeError``, and invalid values gave a ``ValueError`` about converting
  ``nan`` to an integer or meaningless bounds. The docstring states that the
  two-sided region is equal-tailed and not the region of ``binom_test``.
  :issue:`9238`, :pr:`10488`, :pr:`10634`
- ``TTestIndPower.power`` and ``TTestIndPower.solve_power`` accept ``ratio=0``,
  which gives the power of the one-sample t-test with ``nobs1`` observations;
  it raised a ``ZeroDivisionError`` or gave the size of the test for NumPy
  scalars. ``ratio`` can be array_like, and a negative or NaN ``ratio`` raises
  a ``ValueError``; ``solve_power`` returned ``nan`` with a convergence warning
  for a negative ``ratio``. :pr:`10349`, :pr:`10584`
- ``TTestPower.solve_power`` and ``TTestIndPower.solve_power`` raise a
  ``ValueError`` for an invalid ``alternative``, for example ``"one-sided"``.
  They returned ``nan`` with a ``ConvergenceWarning`` if solving for the
  sample size. The spellings are case sensitive, as in ``power``, and the
  deprecated alias ``"2s"`` continues to work with a ``FutureWarning``.
  :issue:`9378`, :pr:`10483`, :pr:`10631`
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
- ``Description`` no longer calls ``scipy.stats.mode`` for a column without
  observations, which warns in SciPy 1.14 and later. :pr:`10613`
- :func:`~statsmodels.stats.rates.confint_poisson` with ``method="midp-c"``
  returned an interval of width zero at the upper limit for a count of 0. The
  limits are now ``0`` and ``-log(alpha) / nobs``. The numerical inversion
  checks that ``count`` and ``nobs`` have a single element, and raises a
  ``ValueError`` otherwise. :issue:`10536`, :pr:`10537`, :pr:`10633`
- ``confint_poisson_2indep`` with ``method="sqrtcc"`` and ``compare="ratio"``
  ignored ``exposure1`` and ``exposure2`` and returned an interval for the ratio
  of the counts instead of the ratio of the rates. It also used the critical
  value instead of its square in the Fieller discriminant. The bounds are now
  rescaled by ``exposure2 / exposure1`` and use the squared critical value. For
  small counts the upper limit was the square of a negative root instead of
  ``inf`` and the lower limit was positive instead of 0. :pr:`10660`
- ``mcnemar`` with ``exact=False`` and ``correction=True`` applied the
  continuity correction also when the two discordant counts are equal, which
  gave a positive statistic, and ``inf`` with a p-value of 0 if both counts
  are 0. As in R's ``mcnemar.test``, the correction is now only applied if the
  counts differ, so the statistic and the p-value are ``nan`` if both counts
  are 0. :issue:`10670`, :pr:`10671`
- ``TTestIndPower.solve_power`` with ``ratio=None`` raised a ``TypeError``
  from SciPy if no ratio attains the requested power. It returns ``nan`` with
  a ``ConvergenceWarning``, as documented. :issue:`10672`, :pr:`10673`
- ``cohens_kappa`` returns the same results for all tables in which a rater
  uses a single category: kappa is 0, its standard errors and confidence
  limits are 0, and the z statistic and the p-values are nan. If both raters
  use the same single category, all statistics are nan. Before, the results
  depended on rounding errors, and NumPy emitted a ``RuntimeWarning``.
  :issue:`5530`, :pr:`10709`
- ``eval_measures.iqr`` returned 0 if ``x1`` had length 1 along ``axis`` and
  was broadcast against a longer ``x2``, raised an ``IndexError`` for a scalar
  ``x1``, and raised a ``ValueError`` with ``axis=None`` for inputs of
  different shapes. The quartiles are now those of the broadcast difference.
  :issue:`10757`, :pr:`10758`

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
- ``Factor`` with ``method="ml"`` and ``PHD`` returned complex loadings and
  directions with NumPy 2.5 and later, which ``np.linalg.eig`` returns as
  complex arrays even if all the eigenvalues are real. The results are real
  again. :pr:`10613`

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
- ``kdensityfft`` raised a ``ValueError`` for data that is not float64 in
  native byte order, for example a big-endian, ``float32`` or integer array.
  :issue:`10770`, :pr:`10795`

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
- ``SurvfuncRight`` with ``exog`` and integer times stored an arbitrary
  integer instead of ``nan`` in ``surv_prob_se``, ``n_risk`` and ``n_events``,
  and warned about casting ``nan`` to an integer. They are ``nan`` as floats.
  :pr:`10613`
- ``SurvfuncRight.quantile`` no longer depends on floating point rounding when
  the estimated survival probability equals ``1 - p``. For example, the median
  of the 12 distinct uncensored times 1, ..., 12 is 7 instead of 6. At such a
  tie the next event time is returned, unlike R, which returns the midpoint of
  the two event times. :pr:`10648`
- ``SurvfuncRight.simultaneous_cb`` used the number at risk at each time
  instead of the total sample size ``n`` in the Hall-Wellner band of Hall and
  Wellner (1980) and Klein and Moeschberger (2003), eqs. 4.4.4-4.4.5, and with
  ``freq_weights`` it used Tsiatis' variance instead of the Greenwood
  variance. The band was too wide at early times and too narrow in the tail.
  All limits change. They agree with the R package ``km.ci`` at
  ``k = 1.3581``, and with ``freq_weights`` they equal the band of the data
  with each row repeated by its weight. :pr:`10649`
- ``PHReg`` warns with a ``ValueWarning`` for unknown keyword arguments. Before, they
  were ignored silently, for example ``weights``, which ``PHReg`` does not support,
  and ``event``, which is the name of ``status`` in ``lifelines``. :issue:`7741`,
  :pr:`10650`

Other
-----

- ``missing="drop"`` with a boolean ``endog`` failed with "Data must be
  1-dimensional". :issue:`5445`, :pr:`10477`
- ``ExpandedNormal`` no longer modifies the array of cumulants that is passed
  in. :pr:`10281`
- ``DistributedModel`` failed with ``GEE`` because the model for the results
  was created without ``groups``. The default debiased estimation method needs
  ``fit_regularized`` with ``alpha``, which ``GEE`` does not have, and now
  raises a ``NotImplementedError`` before any data is fit. :issue:`10778`,
  :pr:`10780`
- ``zipoisson.ppf`` and ``zinegbin.ppf`` returned -1 when ``q`` equals the
  zero inflation ``w``. They return 0. :issue:`10841`, :pr:`10842`

Documentation
=============

- Clarify that ``hessian_factor`` in ``OLS``, ``WLS`` and ``GLS``
  returns unscaled least-squares weights and ignores its compatibility
  arguments. Numerical behavior is unchanged. :issue:`8653`, :pr:`10675`
- The ``news`` docstrings of ``MLEResults`` and ``DynamicFactorMQResults``
  say that two of ``start``, ``end`` and ``periods`` must be given, and that
  the default impact date is the last observation of the previous dataset.
  :issue:`10692`, :pr:`10693`
- Clarify how the ARIMA GLS estimator handles constants, deterministic
  regressors, and trends in integrated models. Numerical behavior is
  unchanged. :issue:`6159`, :pr:`10711`
- Clarify that ``mu0`` of ``sign_test`` is the median under the null
  hypothesis, and that the sample median is not a useful value for it when
  the data have no ties. Numerical behavior is unchanged. :issue:`4101`,
  :pr:`10721`
- Fix typos in the "``endog``, ``exog``, what's that?", "Generalized
  Estimating Equations" and "Getting started" pages.
- Add an example to ``cohens_kappa`` showing that kappa depends on class
  prevalence: with the same rater accuracy and raw agreement, kappa is much
  smaller when one class is rare. :issue:`9680`, :pr:`10806`
- Correct docstrings in ``statsmodels.tsa.stattools``: the default ``nlags``
  and result shape of ``pacf_ols``, the result shape of ``pacf_yw``, the
  ``"legacy"`` lags of ``kpss``, the default ``maxlag`` and ``icbest`` of
  ``adfuller``, the default ``maxlag`` of ``zivot_andrews``, the unused
  ``return_results`` of ``coint`` and its critical values for ``trend="n"``,
  the two columns of ``x`` and ``addconst`` in ``grangercausalitytests``, and
  the fit advice of ``arma_order_select_ic``, which suggested a method that
  makes every fit fail. :pr:`10836`
- Correct docstrings in ``statsmodels.tsa.tsatools``, ``arima_process``,
  ``deterministic`` and the BDS test: ``detrend`` returns pandas objects for
  pandas input, the parameters of ``DeterministicProcess.in_sample`` and
  ``out_of_sample`` (which listed a non-existent ``index``), the order
  constraint of ``Fourier`` and ``CalendarFourier``, return types of
  ``bds``, ``distance_indicators`` and ``correlation_sums``, and examples
  that failed on pandas 3. :pr:`10832`
- Correct the docstrings of the deprecated ``TsaDescriptive``,
  ``TSMLEModel`` and ``_Var``: the accepted data types, the ``order`` of
  ``TsaDescriptive.fit`` (p, d, q) and its return type, the shapes of the
  ``_Var`` coefficient arrays, and a runnable ``_Var`` example in place of one
  that used undefined names. :pr:`10815`

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

.. include:: _upcoming_changes.rst
