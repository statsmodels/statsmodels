
## Robust Linear Models

from __future__ import print_function
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std


# ## Estimation
# 
# Load data:

data = sm.datasets.stackloss.load()
data.exog = sm.add_constant(data.exog)


# Huber's T norm with the (default) median absolute deviation scaling

huber_t = sm.RLM(data.endog, data.exog, M=sm.robust.norms.HuberT())
hub_results = huber_t.fit()
print(hub_results.params)
print(hub_results.bse)
print(hub_results.summary(yname='y',
            xname=['var_%d' % i for i in range(len(hub_results.params))]))


# Huber's T norm with 'H2' covariance matrix

hub_results2 = huber_t.fit(cov="H2")
print(hub_results2.params)
print(hub_results2.bse)


# Andrew's Wave norm with Huber's Proposal 2 scaling and 'H3' covariance matrix

andrew_mod = sm.RLM(data.endog, data.exog, M=sm.robust.norms.AndrewWave())
andrew_results = andrew_mod.fit(scale_est=sm.robust.scale.HuberScale(), cov="H3")
print('Parameters: ', andrew_results.params)


# See ``help(sm.RLM.fit)`` for more options and ``module sm.robust.scale`` for scale options
# 
# ## Comparing OLS and RLM
# 
# Artificial data with outliers:

nsample = 50
x1 = np.linspace(0, 20, nsample)
X = np.column_stack((x1, (x1-5)**2))
X = sm.add_constant(X)
sig = 0.3   # smaller error variance makes OLS<->RLM contrast bigger
beta = [5, 0.5, -0.0]
y_true2 = np.dot(X, beta)
y2 = y_true2 + sig*1. * np.random.normal(size=nsample)
y2[[39,41,43,45,48]] -= 5   # add some outliers (10% of nsample)


# ### Example 1: quadratic function with linear truth
# 
# Note that the quadratic term in OLS regression will capture outlier effects. 

res = sm.OLS(y2, X).fit()
print(res.params)
print(res.bse)
print(res.predict())


# Estimate RLM:

resrlm = sm.RLM(y2, X).fit()
print(resrlm.params)
print(resrlm.bse)


# Draw a plot to compare OLS estimates to the robust estimates:

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.plot(x1, y2, 'o',label="data")
ax.plot(x1, y_true2, 'b-', label="True")
prstd, iv_l, iv_u = wls_prediction_std(res)
ax.plot(x1, res.fittedvalues, 'r-', label="OLS")
ax.plot(x1, iv_u, 'r--')
ax.plot(x1, iv_l, 'r--')
ax.plot(x1, resrlm.fittedvalues, 'g.-', label="RLM")
ax.legend(loc="best")


# ### Example 2: linear function with linear truth
# 
# Fit a new OLS model using only the linear term and the constant:

X2 = X[:,[0,1]] 
res2 = sm.OLS(y2, X2).fit()
print(res2.params)
print(res2.bse)


# Estimate RLM:

resrlm2 = sm.RLM(y2, X2).fit()
print(resrlm2.params)
print(resrlm2.bse)


# Draw a plot to compare OLS estimates to the robust estimates:

prstd, iv_l, iv_u = wls_prediction_std(res2)

fig, ax = plt.subplots()
ax.plot(x1, y2, 'o', label="data")
ax.plot(x1, y_true2, 'b-', label="True")
ax.plot(x1, res2.fittedvalues, 'r-', label="OLS")
ax.plot(x1, iv_u, 'r--')
ax.plot(x1, iv_l, 'r--')
ax.plot(x1, resrlm2.fittedvalues, 'g.-', label="RLM")
ax.legend(loc="best")

