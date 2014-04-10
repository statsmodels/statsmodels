
## Ordinary Least Squares

from __future__ import print_function
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std

np.random.seed(9876789)


# ## OLS estimation
# 
# Artificial data:

nsample = 100
x = np.linspace(0, 10, 100)
X = np.column_stack((x, x**2))
beta = np.array([1, 0.1, 10])
e = np.random.normal(size=nsample)


# Our model needs an intercept so we add a column of 1s:

X = sm.add_constant(X)
y = np.dot(X, beta) + e


# Inspect data:

X = sm.add_constant(X)
y = np.dot(X, beta) + e


# Fit and summary:

model = sm.OLS(y, X)
results = model.fit()
print(results.summary())


# Quantities of interest can be extracted directly from the fitted model. Type ``dir(results)`` for a full list. Here are some examples:  

print('Parameters: ', results.params)
print('R2: ', results.rsquared)


# ## OLS non-linear curve but linear in parameters
# 
# We simulate artificial data with a non-linear relationship between x and y:

nsample = 50
sig = 0.5
x = np.linspace(0, 20, nsample)
X = np.column_stack((x, np.sin(x), (x-5)**2, np.ones(nsample)))
beta = [0.5, 0.5, -0.02, 5.]

y_true = np.dot(X, beta)
y = y_true + sig * np.random.normal(size=nsample)


# Fit and summary:

res = sm.OLS(y, X).fit()
print(res.summary())


# Extract other quantities of interest:

print('Parameters: ', res.params)
print('Standard errors: ', res.bse)
print('Predicted values: ', res.predict())


# Draw a plot to compare the true relationship to OLS predictions. Confidence intervals around the predictions are built using the ``wls_prediction_std`` command.

prstd, iv_l, iv_u = wls_prediction_std(res)

fig, ax = plt.subplots()

ax.plot(x, y, 'o', label="data")
ax.plot(x, y_true, 'b-', label="True")
ax.plot(x, res.fittedvalues, 'r--.', label="OLS")
ax.plot(x, iv_u, 'r--')
ax.plot(x, iv_l, 'r--')
ax.legend(loc='best');


# ## OLS with dummy variables
# 
# We generate some artificial data. There are 3 groups which will be modelled using dummy variables. Group 0 is the omitted/benchmark category.

nsample = 50
groups = np.zeros(nsample, int)
groups[20:40] = 1
groups[40:] = 2
#dummy = (groups[:,None] == np.unique(groups)).astype(float)

dummy = sm.categorical(groups, drop=True)
x = np.linspace(0, 20, nsample)
# drop reference category
X = np.column_stack((x, dummy[:,1:]))
X = sm.add_constant(X, prepend=False)

beta = [1., 3, -3, 10]
y_true = np.dot(X, beta)
e = np.random.normal(size=nsample)
y = y_true + e


# Inspect the data:

print(X[:5,:])
print(y[:5])
print(groups)
print(dummy[:5,:])


# Fit and summary:

res2 = sm.OLS(y, X).fit()
print(res.summary())


# Draw a plot to compare the true relationship to OLS predictions:

prstd, iv_l, iv_u = wls_prediction_std(res2)

fig, ax = plt.subplots()

ax.plot(x, y, 'o', label="Data")
ax.plot(x, y_true, 'b-', label="True")
ax.plot(x, res2.fittedvalues, 'r--.', label="Predicted")
ax.plot(x, iv_u, 'r--')
ax.plot(x, iv_l, 'r--')
ax.legend(loc="best")


# ## Joint hypothesis test
# 
# ### F test
# 
# We want to test the hypothesis that both coefficients on the dummy variables are equal to zero, that is, $R \times \beta = 0$. An F test leads us to strongly reject the null hypothesis of identical constant in the 3 groups:

R = [[0, 1, 0, 0], [0, 0, 1, 0]]
print(np.array(R))
print(res2.f_test(R))


# You can also use formula-like syntax to test hypotheses

print(res2.f_test("x2 = x3 = 0"))


# ### Small group effects
# 
# If we generate artificial data with smaller group effects, the T test can no longer reject the Null hypothesis: 

beta = [1., 0.3, -0.0, 10]
y_true = np.dot(X, beta)
y = y_true + np.random.normal(size=nsample)

res3 = sm.OLS(y, X).fit()


print(res3.f_test(R))


print(res3.f_test("x2 = x3 = 0"))


# ### Multicollinearity
# 
# The Longley dataset is well known to have high multicollinearity. That is, the exogenous predictors are highly correlated. This is problematic because it can affect the stability of our coefficient estimates as we make minor changes to model specification. 

from statsmodels.datasets.longley import load_pandas
y = load_pandas().endog
X = load_pandas().exog
X = sm.add_constant(X)


# Fit and summary:

ols_model = sm.OLS(y, X)
ols_results = ols_model.fit()
print(ols_results.summary())


# #### Condition number
# 
# One way to assess multicollinearity is to compute the condition number. Values over 20 are worrisome (see Greene 4.9). The first step is to normalize the independent variables to have unit length: 

for i, name in enumerate(X):
    if name == "const":
        continue
    norm_x[:,i] = X[name]/np.linalg.norm(X[name])
norm_xtx = np.dot(norm_x.T,norm_x)


# Then, we take the square root of the ratio of the biggest to the smallest eigen values. 

eigs = np.linalg.eigvals(norm_xtx)
condition_number = np.sqrt(eigs.max() / eigs.min())
print(condition_number)


# #### Dropping an observation
# 
# Greene also points out that dropping a single observation can have a dramatic effect on the coefficient estimates: 

ols_results2 = sm.OLS(y.ix[:14], X.ix[:14]).fit()
print("Percentage change %4.2f%%\n"*7 % tuple([i for i in (ols_results2.params - ols_results.params)/ols_results.params*100]))


# We can also look at formal statistics for this such as the DFBETAS -- a standardized measure of how much each coefficient changes when that observation is left out.

infl = ols_results.get_influence()


# In general we may consider DBETAS in absolute value greater than $2/\sqrt{N}$ to be influential observations

2./len(X)**.5


print(infl.summary_frame().filter(regex="dfb"))

