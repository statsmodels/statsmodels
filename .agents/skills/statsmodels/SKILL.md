---
name: statsmodels
description: Guide practical statistical modeling with statsmodels, including model choice, formulas and design matrices, fitting, diagnostics, inference, model comparison, and prediction. Use when analyzing data with statsmodels or when choosing, fitting, diagnosing, predicting with, or interpreting a statsmodels model or result.
---

# statsmodels user workflows

Use this skill for practical statistical modeling with the APIs in the current
statsmodels repository. Keep the work focused on model specification, fitting,
inference, diagnostics, and prediction. Do not add repository contribution or
development instructions to the user-facing workflow.

## Start with the data and the question

Before choosing a model, establish:

- the outcome's support and meaning: continuous, binary, count, nominal
  categorical, or ordered categorical;
- whether observations are plausibly IID, grouped, clustered, repeated, or
  time ordered;
- whether the goal is parameter inference, a comparison or hypothesis test,
  prediction, or a combination;
- which variables are numeric, categorical, exposure or offset variables, and
  which rows are eligible for analysis.

Choose a model whose likelihood, mean-variance relationship, dependence
structure, and estimand match those decisions. A model that converges can
still be inappropriate for the data or question.

## Route common model families

| Situation | Starting points |
| --- | --- |
| Continuous response with a mean relationship | `OLS`; consider `WLS` or `GLS` when the weighting or covariance structure is part of the specification. |
| Binary response | `Logit`, `Probit`, or `GLM` with a `Binomial` family. |
| Count response | `Poisson`, `NegativeBinomial`, or a count-family `GLM` when the count distribution, mean-variance relationship, and exposure assumptions are appropriate. |
| Other non-Gaussian response | Choose a model or `GLM` family and link that match the response support and likelihood; do not default to count models for an arbitrary non-Gaussian outcome. |
| Nominal response with more than two categories | `MNLogit`. |
| Ordered categorical response | `OrderedModel`, with ordered levels and no intercept in the final design. |
| Binary grouped response with group-specific nuisance intercepts | `ConditionalLogit`, with a required `groups` argument and no intercept. |
| Repeated or clustered observations | Consider `MixedLM`, `GEE`, or a model with an appropriate cluster covariance, depending on the estimand and dependence assumptions. |
| Time-ordered observations or forecasting | Use a suitable model from `statsmodels.tsa` and validate in time order. |

This is a routing guide, not a substitute for checking the model's
assumptions and current documentation.

## Specify the model deliberately

Formula APIs, such as `statsmodels.formula.api.ols`, use statsmodels' R-style
formula interface, supported by formula engines including Patsy and Formulaic,
and normally include an intercept. Array-based constructors, such as
`statsmodels.api.OLS`, require the design matrix you intend to fit; add a
constant explicitly with `statsmodels.api.add_constant` when it belongs in the
model. Do not assume that an intercept is always appropriate.

Intercept rules are model-specific:

- `OrderedModel` requires no explicit or implicit constant in its final design
  matrix. A constant is not separately identified from its thresholds, and a
  remaining constant raises `ValueError`. Formula handling has special cases,
  especially with categorical terms, so inspect the resulting design matrix.
- `ConditionalLogit` also rejects an intercept. Omit it from an array design;
  with a formula, use a no-intercept specification such as `y ~ 0 + x1 + x2`
  and pass `groups=...`. Group-specific intercepts are conditioned out and
  are not estimated as ordinary parameters.

Treat missing data as part of the specification. Inspect missingness before
fitting, choose explicit handling such as `missing="drop"` or
`missing="raise"` where the model supports it, and record which observations
were analyzed. Do not silently compare models fit to different rows.

## Use the modeling workflow

Follow:

`data -> specification -> fit -> inspect results -> diagnostics -> prediction/inference`

After fitting, check convergence and the fitted model's parameterization
before interpreting `params`, `bse`, `pvalues`, `conf_int()`, or `summary()`.
Use diagnostics to investigate residual structure, influential observations,
heteroskedasticity, autocorrelation, collinearity, and other failures relevant
to the model. Robust covariance changes estimated uncertainty under specified
dependence or variance conditions; it does not repair a misspecified mean,
likelihood, link, functional form, or dependence structure.

For prediction, use the fitted result object's `predict(...)` or
`get_prediction(...)` API as appropriate. Formula-fitted results generally
accept new data in the original variable form when `transform=True`; array
fitted results require a compatible design matrix, including its constant.
Prediction intervals and confidence intervals for predictions are not the
same as confidence intervals for parameters.

Read the relevant reference before giving detailed guidance:

- [Quick start](references/quick_start.md) for runnable OLS, Logit, and GLM
  examples.
- [Model selection and formulas](references/model_selection_and_formula.md)
  for specification, comparison, missing-data, and validation decisions.
- [Diagnostics and inference](references/diagnostics_and_inference.md) for
  result objects, covariance choices, tests, and interpretation limits.

Use APIs present in the current repository and avoid version-pinned external
metadata. When a model-specific option or result is uncertain, inspect that
model's current docstring or source before stating that it is supported.
