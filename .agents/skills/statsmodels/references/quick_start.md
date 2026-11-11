# Quick start

These examples are intentionally independent. Each block defines its imports,
data, model, fit, new-data or new-design input, and prediction call. The
examples show common workflows; they do not make intercepts universal.

## OLS with a formula

Formula syntax creates the design matrix and normally includes an intercept.
Pass new observations with the original variable names so the fitted formula
can transform them.

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

rng = np.random.default_rng(123)
data = pd.DataFrame({
    "x1": np.linspace(0, 5, 40),
    "x2": rng.normal(size=40),
})
data["y"] = 1.5 + 2.0 * data["x1"] - 0.5 * data["x2"] + rng.normal(
    scale=0.5, size=len(data)
)

model = smf.ols("y ~ x1 + x2", data=data)
results = model.fit()

new_data = pd.DataFrame({"x1": [1.0, 3.0], "x2": [0.0, 1.0]})
prediction = results.get_prediction(new_data).summary_frame()
print(results.params)
print(prediction)
```

## OLS with an array design

Array-based APIs do not add a constant. Add one explicitly if the model has an
intercept, and construct new designs with the same columns and order.

```python
import numpy as np
import statsmodels.api as sm

rng = np.random.default_rng(123)
x = rng.normal(size=(40, 2))
y = 1.5 + x @ np.array([2.0, -0.5]) + rng.normal(scale=0.5, size=40)
X = sm.add_constant(x, has_constant="add")

model = sm.OLS(y, X)
results = model.fit()

new_x = np.array([[1.0, 0.0], [0.0, 1.0]])
new_X = sm.add_constant(new_x, has_constant="add")
prediction = results.get_prediction(new_X).summary_frame()
print(results.params)
print(prediction)
```

## Binary response with `Logit`

`Logit` expects a binary response, commonly coded as 0 and 1. This array
example includes a constant explicitly. For a formula, use
`statsmodels.formula.api.logit` and provide new data in the formula's original
variables.

```python
import numpy as np
import statsmodels.api as sm

rng = np.random.default_rng(123)
x = rng.normal(size=80)
probability = 1 / (1 + np.exp(-(0.25 + 1.1 * x)))
y = rng.binomial(1, probability)
X = sm.add_constant(x, has_constant="add")

model = sm.Logit(y, X)
results = model.fit(disp=False)

new_x = np.array([[-1.0], [0.0], [1.0]])
new_X = sm.add_constant(new_x, has_constant="add")
prediction = results.get_prediction(new_X).summary_frame()
print(results.params)
print(prediction)
```

## GLM with a Poisson family and log link

A Poisson GLM is a starting point for count data when its mean-variance and
exposure assumptions are appropriate. The family and link are explicit here;
choose them from the outcome and design rather than treating GLM as a generic
replacement for every regression.

```python
import numpy as np
import statsmodels.api as sm

rng = np.random.default_rng(123)
x = rng.normal(size=80)
mean = np.exp(0.4 + 0.6 * x)
y = rng.poisson(mean)
X = sm.add_constant(x, has_constant="add")

family = sm.families.Poisson(link=sm.families.links.Log())
model = sm.GLM(y, X, family=family)
results = model.fit()

new_x = np.array([[-1.0], [0.0], [1.0]])
new_X = sm.add_constant(new_x, has_constant="add")
prediction = results.get_prediction(new_X).summary_frame()
print(results.params)
print(prediction)
```

## Inspect fitted results

The common result attributes below are estimates and model-based uncertainty
summaries. Their interpretation depends on the fitted model, covariance
choice, and any transformations in the specification.

```python
import numpy as np
import statsmodels.api as sm

rng = np.random.default_rng(123)
x = rng.normal(size=50)
y = 2.0 + 1.25 * x + rng.normal(size=50)
X = sm.add_constant(x, has_constant="add")
results = sm.OLS(y, X).fit()

print(results.params)
print(results.bse)
print(results.pvalues)
print(results.conf_int())
print(results.summary())
```

`params`, `bse`, `pvalues`, and `conf_int()` refer to parameters. A result
object can also expose model-specific quantities such as `resid`, `fittedvalues`,
`aic`, `bic`, or `get_margeff()`. Check the model's result class before
assuming every attribute is available or has the same interpretation.
