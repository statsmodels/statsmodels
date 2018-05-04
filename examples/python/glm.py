
## Generalized Linear Models
from __future__ import print_function
import numpy as np
import statsmodels.api as sm
from scipy import stats
from matplotlib import pyplot as plt


# ## GLM: Binomial response data
#
# ### Load data
#
#  In this example, we use the Star98 dataset which was taken with permission
#  from Jeff Gill (2000) Generalized linear models: A unified approach. Codebook
#  information can be obtained by typing:

print(sm.datasets.star98.NOTE)


# Load the data and add a constant to the exogenous (independent) variables:

data = sm.datasets.star98.load()
data.exog = sm.add_constant(data.exog, prepend=False)


#  The dependent variable is N by 2 (Success: NABOVE, Failure: NBELOW):

print(data.endog[:5,:])


#  The independent variables include all the other variables described above, as
#  well as the interaction terms:

print(data.exog[:2,:])


# ### Fit and summary

glm_binom = sm.GLM(data.endog, data.exog, family=sm.families.Binomial())
res = glm_binom.fit()
print(res.summary())


# ### Quantities of interest

print('Total number of trials:',  data.endog[0].sum())
print('Parameters: ', res.params)
print('T-values: ', res.tvalues)


# First differences: We hold all explanatory variables constant at their means and manipulate the percentage of low income households to assess its impact on the response variables:

means = data.exog.mean(axis=0)
means25 = means.copy()
means25[0] = stats.scoreatpercentile(data.exog[:,0], 25)
means75 = means.copy()
means75[0] = lowinc_75per = stats.scoreatpercentile(data.exog[:,0], 75)
resp_25 = res.predict(means25)
resp_75 = res.predict(means75)
diff = resp_75 - resp_25


# The interquartile first difference for the percentage of low income households in a school district is:

print("%2.4f%%" % (diff*100))


# ### Plots
#
#  We extract information that will be used to draw some interesting plots:

nobs = res.nobs
y = data.endog[:,0]/data.endog.sum(1)
yhat = res.mu


# Plot yhat vs y:

from statsmodels.graphics.api import abline_plot


fig, ax = plt.subplots()
ax.scatter(yhat, y)
line_fit = sm.OLS(y, sm.add_constant(yhat, prepend=True)).fit()
abline_plot(model_results=line_fit, ax=ax)


ax.set_title('Model Fit Plot')
ax.set_ylabel('Observed values')
ax.set_xlabel('Fitted values');


# Plot yhat vs. Pearson residuals:

fig, ax = plt.subplots()

ax.scatter(yhat, res.resid_pearson)
ax.hlines(0, 0, 1)
ax.set_xlim(0, 1)
ax.set_title('Residual Dependence Plot')
ax.set_ylabel('Pearson Residuals')
ax.set_xlabel('Fitted values')


# Histogram of standardized deviance residuals:

from scipy import stats

fig, ax = plt.subplots()

resid = res.resid_deviance.copy()
resid_std = stats.zscore(resid)
ax.hist(resid_std, bins=25)
ax.set_title('Histogram of standardized deviance residuals');


# QQ Plot of Deviance Residuals:

from statsmodels import graphics
graphics.gofplots.qqplot(resid, line='r')


# ## GLM: Gamma for proportional count response
#
# ### Load data
#
#  In the example above, we printed the ``NOTE`` attribute to learn about the
#  Star98 dataset. Statsmodels datasets ships with other useful information. For
#  example:

print(sm.datasets.scotland.DESCRLONG)


#  Load the data and add a constant to the exogenous variables:

data2 = sm.datasets.scotland.load()
data2.exog = sm.add_constant(data2.exog, prepend=False)
print(data2.exog[:5,:])
print(data2.endog[:5])


# ### Fit and summary

glm_gamma = sm.GLM(data2.endog, data2.exog, family=sm.families.Gamma())
glm_results = glm_gamma.fit()
print(glm_results.summary())


# ## GLM: Gaussian distribution with a noncanonical link
#
# ### Artificial data

nobs2 = 100
x = np.arange(nobs2)
np.random.seed(54321)
X = np.column_stack((x,x**2))
X = sm.add_constant(X, prepend=False)
lny = np.exp(-(.03*x + .0001*x**2 - 1.0)) + .001 * np.random.rand(nobs2)


# ### Fit and summary

gauss_log = sm.GLM(lny, X, family=sm.families.Gaussian(sm.families.links.log()))
gauss_log_results = gauss_log.fit()
print(gauss_log_results.summary())
