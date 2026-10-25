# Diagnostics and inference

Diagnostics are evidence about assumptions and influential observations, not
automatic verdicts. Use them with subject-matter knowledge, plots, alternative
specifications, and the model's response scale.

The code blocks below are fragments and patterns for an existing fitted model
and data, not independently executable examples. Names such as `sm`, `y`, `X`,
`groups`, and `results` refer to objects already defined in the surrounding
analysis.

## Residuals and fitted values

Start with the quantities exposed by the fitted result, commonly
`results.resid`, `results.fittedvalues`, and, for GLMs, model-specific residuals
such as `results.resid_pearson` or `results.resid_deviance`. Plot residuals
against fitted values and important predictors, and consider scale-location or
response-specific plots. Patterns can indicate nonlinearity, omitted terms,
unequal variance, dependence, or an unsuitable link. A residual plot that
looks acceptable cannot validate all assumptions.

For a linear model, `results.get_influence()` returns an influence result object
with measures such as leverage and Cook's distance. For example:

```python
influence = results.get_influence()
leverage = influence.hat_matrix_diag
cooks_distance = influence.cooks_distance
```

`cooks_distance` is a pair `(distance, p-value)`: the Cook's distances and
the p-values from an F reference distribution for each distance.
`influence.summary_frame()` provides a tabular summary for OLS
influence measures. Other result classes can expose different influence APIs;
check the fitted result class rather than assuming OLS methods transfer.

Large leverage, a large Cook's distance, or a small p-value from an outlier
procedure identifies an observation for investigation. It does not prove that
the observation is erroneous or that it should be removed. Check data quality,
measurement, sampling, and the effect of defensible alternative analyses.

## Heteroskedasticity, autocorrelation, and collinearity

For an OLS residual, common diagnostic functions include:

```python
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_breuschpagan,
    het_white,
)
from statsmodels.stats.stattools import durbin_watson

dw = durbin_watson(results.resid)
bp_lm, bp_lm_pvalue, bp_f, bp_f_pvalue = het_breuschpagan(
    results.resid, results.model.exog
)
white_lm, white_lm_pvalue, white_f, white_f_pvalue = het_white(
    results.resid, results.model.exog
)
ljung_box = acorr_ljungbox(results.resid, lags=[1, 4])
```

`durbin_watson` returns a statistic; `het_breuschpagan` and `het_white`
return tuples of statistics and p-values; `acorr_ljungbox` always returns a
DataFrame with columns `lb_stat` and `lb_pvalue`. `het_breuschpagan` and
`het_white` use the supplied regressors in an auxiliary regression and assume
that they include a constant, as `results.model.exog` does when the model has
one. `durbin_watson` and `acorr_ljungbox` are only meaningful when the
observations are in time order. If the residuals come from a model that
includes lagged dependent variables or estimated ARMA terms, set `model_df` in
`acorr_ljungbox` to the number of estimated parameters that use up degrees of
freedom. These tests depend on the supplied residuals, regressors, lag choices, and
assumptions. A significant result is a prompt to investigate, not a complete
diagnosis; a nonsignificant result does not prove the assumption.

Several diagnostic functions can return named result objects when requested.
`acorr_lm`, `het_arch`, `acorr_breusch_godfrey`, and `het_goldfeldquandt` still
return a plain tuple by default and emit a `FutureWarning` when
`result_object` is not given, so pass `result_object=True` explicitly:

```python
from statsmodels.stats.diagnostic import (
    acorr_breusch_godfrey,
    het_arch,
    het_goldfeldquandt,
)

bg = acorr_breusch_godfrey(results, nlags=2, result_object=True)
arch = het_arch(results.resid, result_object=True)
gq = het_goldfeldquandt(
    results.model.endog, results.model.exog, result_object=True
)
```

`acorr_lm`, `het_arch`, and `acorr_breusch_godfrey` then return an
`LMTestResult` with fields `lm`, `lmpval`, `fval`, `fpval`, and `res_store`
(for example `bg.lmpval`). `het_goldfeldquandt` returns a
`GoldfeldQuandtResult` with fields `fval`, `pval`, `ordering`, and `res_store`.
Use these named fields rather than positional tuple conventions, and check each
function's current docstring for its default return mode.

Variance inflation factor is computed one design column at a time:

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_for_x1 = variance_inflation_factor(results.model.exog, 1)
```

VIF is a collinearity diagnostic for the supplied design, not a universal
cutoff or a test that a model is invalid. Include or exclude the constant
consistently and interpret indicator encodings with care.

## Covariance choices

Covariance choice follows the sampling and error structure, not the desired
p-value. Common supported patterns for result classes that use the shared
robust covariance machinery include:

```python
ols_hc3 = sm.OLS(y, X).fit(cov_type="HC3")
ols_cluster = sm.OLS(y, X).fit(
    cov_type="cluster", cov_kwds={"groups": groups}
)
ols_hac = sm.OLS(y, X).fit(
    cov_type="HAC", cov_kwds={"maxlags": 2}
)
```

`GLM.fit` has its own explicit covariance configuration:

```python
glm_robust = sm.GLM(y, X, family=sm.families.Poisson()).fit(
    cov_type="HC0", cov_kwds={}, use_t=False
)
```

For GLM, `HC1` through `HC3` are not implemented and fall back to `HC0` with a
`SpecificationWarning`, so request `HC0` explicitly.

The available covariance types and keywords vary by result/model class.
Check that class's `fit` documentation before using cluster, HAC, panel, or
other covariance types. `HC0` through `HC3` address heteroskedasticity in
specified settings; `HAC` needs an appropriate time ordering and lag choice;
cluster covariance needs a defensible cluster definition and enough clusters.
Robust covariance changes standard errors, test statistics, confidence
intervals, and often p-values. It does not repair misspecification of the
mean, link, likelihood, omitted variables, functional form, or dependence
structure.

## Confidence intervals and hypothesis tests

For fitted parameters, common result APIs are:

```python
parameter_ci = results.conf_int(alpha=0.05)
single_test = results.t_test("x1 = 0")
joint_test = results.f_test("x1 = 0, x2 = 0")
```

`conf_int()` returns intervals for parameters. `t_test`, `f_test`, and
`wald_test` operate on restrictions and return test-result objects whose
attributes depend on the test. Confirm parameter names and the covariance used
by the result before interpreting the output. Nonlinear coefficients,
transformed responses, multiple testing, weak identification, and boundary
parameters need additional care beyond reading a default p-value.

For discrete and GLM models, coefficient changes are on the model's linear
predictor scale unless transformed. `results.get_margeff()` returns a
model-specific marginal-effects result for models that support it. The
terminology is marginal effects, not a generic guarantee that coefficients
are marginal effects; choose `at`, `method`, and any discrete/count handling
for the estimand and inspect the returned object.

## Prediction intervals and prediction inference

Use the fitted result's prediction API for new observations:

```python
prediction = results.get_prediction(new_exog)
prediction_frame = prediction.summary_frame()
prediction_intervals = prediction.conf_int()
```

For formula fits, pass a data frame with the original variable names and use
the formula transformation behavior. For array fits, pass a design matrix
with the same encoding and intercept columns as the fitted model. Available
statistics and interval columns depend on the result class. OLS prediction
results can distinguish uncertainty for the mean response from uncertainty
for a new observation; discrete and GLM prediction results use model-specific
response statistics. Read the returned frame and the model's prediction
docstring rather than assuming that every model offers an observation interval.

Parameter confidence intervals quantify uncertainty about coefficients under
the fitted model. Prediction intervals quantify uncertainty about a predicted
response or mean at specified covariates. Neither automatically establishes
causal effects, and neither protects against extrapolation or misspecification.
