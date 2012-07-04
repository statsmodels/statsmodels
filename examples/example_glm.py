'''Generalized Linear Models
'''
import numpy as np
import statsmodels.api as sm
from scipy import stats
from matplotlib import pyplot as plt

#GLM: Binomial response data
#---------------------------

#Load data
#^^^^^^^^^
# In this example, we use the Star98 dataset which was taken with permission
# from Jeff Gill (2000) Generalized linear models: A unified approach. Codebook
# information can be obtained by typing: 
print sm.datasets.star98.NOTE

# Load the data and add a constant to the exogenous (independent) variables:
data = sm.datasets.star98.load()
data.exog = sm.add_constant(data.exog)

# The dependent variable is N by 2 (Success: NABOVE, Failure: NBELOW): 
print data.endog[:5,:]

# The independent variables include all the other variables described above, as
# well as the interaction terms:
print data.exog[:2,:]

#Fit and summary
#^^^^^^^^^^^^^^^
glm_binom = sm.GLM(data.endog, data.exog, family=sm.families.Binomial())
binom_results = glm_binom.fit()
print binom_results.summary()

#Quantities of interest
#^^^^^^^^^^^^^^^^^^^^^^
# Total number of trials: 
print data.endog[0].sum()

# Parameter estimates:
print binom_results.params

# The corresponding t-values:
print binom_results.tvalues

# First differences: We hold all explanatory variables constant at their means
# and manipulate the percentage of low income households to assess its impact
# on the response variables: 
means = data.exog.mean(axis=0)
means25 = means.copy()
means25[0] = stats.scoreatpercentile(data.exog[:,0], 25)
means75 = means.copy()
means75[0] = lowinc_75per = stats.scoreatpercentile(data.exog[:,0], 75)
resp_25 = binom_results.predict(means25)
resp_75 = binom_results.predict(means75)
diff = resp_75 - resp_25

# The interquartile first difference for the percentage of low income
# households in a school district is:
print "%2.4f%%" % (diff*100)

#Plots
#^^^^^

# We extract information that will be used to draw some interesting plots: 
nobs = binom_results.nobs
y = data.endog[:,0]/data.endog.sum(1)
yhat = binom_results.mu

# Plot yhat vs y:
plt.figure()
plt.scatter(yhat, y)
line_fit = sm.OLS(y, sm.add_constant(yhat)).fit().params
fit = lambda x: line_fit[1]+line_fit[0]*x # better way in scipy?
plt.plot(np.linspace(0,1,nobs), fit(np.linspace(0,1,nobs)))
plt.title('Model Fit Plot')
plt.ylabel('Observed values')
#@savefig glm_fitted.png
plt.xlabel('Fitted values')

# Plot yhat vs. Pearson residuals:
plt.figure()
plt.scatter(yhat, binom_results.resid_pearson)
plt.plot([0.0, 1.0],[0.0, 0.0], 'k-')
plt.title('Residual Dependence Plot')
plt.ylabel('Pearson Residuals')
#@savefig glm_resids.png
plt.xlabel('Fitted values')

#Histogram of standardized deviance residuals
plt.figure()
res = binom_results.resid_deviance.copy()
stdres = (res - res.mean())/res.std()
plt.hist(stdres, bins=25)
#@savefig glm_hist_res.png
plt.title('Histogram of standardized deviance residuals')

#QQ Plot of Deviance Residuals
plt.figure()
res.sort()
p = np.linspace(0 + 1./(nobs-1), 1-1./(nobs-1), nobs)
quants = np.zeros_like(res)
for i in range(nobs):
    quants[i] = stats.scoreatpercentile(res, p[i]*100)
mu = res.mean()
sigma = res.std()
y = stats.norm.ppf(p, loc=mu, scale=sigma)
plt.scatter(y, quants)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--')
plt.title('Normal - Quantile Plot')
plt.ylabel('Deviance Residuals Quantiles')
plt.xlabel('Quantiles of N(0,1)')

from statsmodels import graphics
#@savefig glm_qqplot.png
img = graphics.gofplots.qqplot(res, line='r')

#GLM: Gamma for proportional count response
#------------------------------------------
#Load data
#^^^^^^^^^
# In the example above, we printed the ``NOTE`` attribute to learn about the
# Star98 dataset. Statsmodels datasets ships with other useful information. For
# example: 
print sm.datasets.scotland.DESCRLONG

# Load the data and add a constant to the exogenous variables:
data2 = sm.datasets.scotland.load()
data2.exog = sm.add_constant(data2.exog)
print data2.exog[:5,:]
print data2.endog[:5]

#Fit and summary
#^^^^^^^^^^^^^^^
glm_gamma = sm.GLM(data2.endog, data2.exog, family=sm.families.Gamma())
glm_results = glm_gamma.fit()
print glm_results.summary()

#GLM: Gaussian distribution with a noncanonical link
#---------------------------------------------------
#Artificial data
#^^^^^^^^^^^^^^^
nobs2 = 100
x = np.arange(nobs2)
np.random.seed(54321)
X = np.column_stack((x,x**2))
X = sm.add_constant(X)
lny = np.exp(-(.03*x + .0001*x**2 - 1.0)) + .001 * np.random.rand(nobs2)

#Fit and summary
#^^^^^^^^^^^^^^^
gauss_log = sm.GLM(lny, X, family=sm.families.Gaussian(sm.families.links.log))
gauss_log_results = gauss_log.fit()
print gauss_log_results.summary()


