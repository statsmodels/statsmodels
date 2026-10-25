# Model selection and formulas

## Select from the data-generating structure and question

Start with the outcome and the dependence structure, then ask what is being
estimated or predicted. A continuous response may motivate `OLS`, `WLS`, or
`GLS`; a binary response may motivate `Logit`, `Probit`, or a binomial `GLM`;
counts may motivate `Poisson`, `NegativeBinomial`, or a count-family `GLM`;
nominal categories may motivate `MNLogit`; and ordered categories may motivate
`OrderedModel`. These choices encode different response distributions and
interpretations.

If observations share subjects, locations, firms, groups, or time periods,
account for that structure through an appropriate model or covariance. Common
starting points include `MixedLM`, `GEE`, grouped or clustered covariance, and
models in `statsmodels.tsa`. A cluster-robust covariance is not a replacement
for a model whose conditional mean or likelihood is wrong.

Keep prediction and inference goals distinct while allowing either ecosystem
to support either goal. Prediction emphasizes out-of-sample performance and
leakage-resistant validation; inference emphasizes an estimand, assumptions,
uncertainty, and a defensible data-generating interpretation. The choice is
about the task and assumptions, not a rule that assigns prediction to one
library and inference to another.

## Formula and array specifications

Formula interfaces are available through `statsmodels.formula.api` and through
model `from_formula` methods. Statsmodels' R-style formula interface can express
transformations and categorical variables; formula processing is supported by
engines including Patsy and Formulaic, for example:

```python
import statsmodels.formula.api as smf

results = smf.ols("y ~ x1 + x2 + C(region) + I(x1 ** 2)", data=data).fit()
```

Formula specifications normally include an intercept. Use `0 +` or `- 1`
when a no-intercept formula is appropriate, but verify the resulting matrix,
especially with categorical terms. Formula-fitted result objects generally
accept new observations in their original data form through
`get_prediction(new_data, transform=True)`.

Array constructors such as `sm.OLS(endog, exog)` and `sm.Logit(endog, exog)` do
not add an intercept. Use `sm.add_constant(exog)` when needed, and reproduce
the same columns, encoding, and order for new data and for comparisons.

## Intercepts are model-specific

Do not treat the formula default as a universal modeling rule.

`OrderedModel` requires no explicit or implicit constant in the final design
matrix. Its threshold parameters already determine category cut points, and a
constant is not separately identified from them. The constructor raises
`ValueError` if a constant remains. Its `from_formula` implementation removes
an explicit `Intercept`, but categorical terms and implicit constants can make
formula behavior subtle; inspect `model.exog`, `model.exog_names`, and
`model.k_constant` after construction. An intercept-free numerical formula can
be straightforward, while categorical designs require particular care.

`ConditionalLogit` also rejects an intercept. Its array `exog` must omit one.
Its formula interface requires the grouped `groups` argument and warns when
the formula does not contain a no-intercept marker. Use a specification such as
`y ~ 0 + x1 + x2`, then verify the constructed design. Group-specific
intercepts are conditioned out, so they are not returned as ordinary
coefficient estimates.

## Compare models carefully

Use `results.aic` or `results.bic` only when the likelihood values are
meaningfully comparable. Normally this means using the same observed response
and analyzed observations, compatible handling of missing data, weights,
exposure, and offsets, and likelihoods defined on a comparable observed-data
basis and scale. Different likelihood families are not automatically excluded,
but their fit statistics are not meaningful to compare when these conditions
fail. Lower AIC or BIC is not proof that a model is substantively correct, and
these criteria do not generally become comparable merely because two result
objects have the same number of rows.

A likelihood-ratio test compares a restricted model with a less restricted
model when the null model is nested in the alternative and the fits use the
same observations under a compatible likelihood formulation. Standard
chi-square reference behavior also relies on regularity conditions. Parameters on boundaries, unidentified
parameters under the null, nonstandard nuisance parameters, or small samples
can invalidate the usual approximation. AIC/BIC and likelihood-ratio tests are
not substitutes for checking the specification.

Missing-data handling must be aligned before model comparison. If one formula
or model drops a row because of an additional variable, refit the other model
on the same eligible observations before comparing fit statistics or testing
nested hypotheses. Make the missing-data policy explicit and retain the row
selection used for the analysis.

## Validate according to dependence

- For plausibly IID observations, a held-out split or cross-validation can be
  reasonable when preprocessing is learned inside each training fold.
- For grouped or clustered observations, split by the grouping unit when the
  goal is generalization to new units. Do not let records from one unit cross
  train and validation sets without a clear reason.
- For panel or repeated observations, preserve the intended unit and time
  structure and distinguish inference for existing units from prediction for
  new units.
- For time series, train on earlier periods and validate on later periods;
  avoid using future information in transformations, features, or missing-data
  decisions.

Validation design should match the deployment or scientific target. It does
not by itself establish causal validity or correct inferential uncertainty.
