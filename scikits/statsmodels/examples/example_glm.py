'''Examples: scikits.statsmodels.GLM
'''
import numpy as np
import scikits.statsmodels as sm
from scipy import stats
import pylab

### Example for using GLM on binomial response data
### the input response vector in this case is N by 2 (success, failure)
# This data is taken with permission from
# Jeff Gill (2000) Generalized linear models: A unified approach
# The dataset can be described by uncommenting

# print sm.datasets.star98.DESCRLONG

# The response variable is
# (# of students above the math national median, # of students below)

# The explanatory variables are (in column order)
# The proportion of low income families "LOWINC"
# The proportions of minority students,"PERASIAN","PERBLACK","PERHISP"
# The percentage of minority teachers "PERMINTE",
# The median teacher salary including benefits in 1000s "AVSALK"
# The mean teacher experience in years "AVYRSEXP",
# The per-pupil expenditures in thousands "PERSPENK"
# The parent-teacher ratio "PTRATIO"
# The percent of students taking college credit courses "PCTAF",
# The percentage of charter schools in the districut "PCTCHRT"
# The percent of schools in the district operating year round "PCTYRRND"
# The following are interaction terms "PERMINTE_AVYRSEXP","PERMINTE_AVSAL",
# "AVYRSEXP_AVSAL","PERSPEN_PTRATIO","PERSPEN_PCTAF","PTRATIO_PCTAF",
# "PERMINTE_AVYRSEXP_AVSAL","PERSPEN_PTRATIO_PCTAF"

data = sm.datasets.star98.Load()
data.exog = sm.add_constant(data.exog)

print """The response variable is (success, failure).  Eg., the first
observation is """, data.endog[0]
print"""Giving a total number of trials for this observation of
""", data.endog[0].sum()

glm_binom = sm.GLM(data.endog, data.exog, family=sm.family.Binomial())

### In order to fit this model, you must (for now) specify the number of
### trials per observation ie., success + failure
### This is the only time the data_weights argument should be used.

trials = data.endog.sum(axis=1)
binom_results = glm_binom.fit(data_weights=trials)
print """The fitted values are
""", binom_results.params
print """The corresponding t-values are
""", binom_results.t()

# It is common in GLMs with interactions to compare first differences.
# We are interested in the difference of the impact of the explanatory variable
# on the response variable.  This example uses interquartile differences for
# the percentage of low income households while holding the other values
# constant at their mean.


means = data.exog.mean(axis=0)
means25 = means.copy()
means25[0] = stats.scoreatpercentile(data.exog[:,0], 25)
means75 = means.copy()
means75[0] = lowinc_75per = stats.scoreatpercentile(data.exog[:,0], 75)
resp_25 = glm_binom.family.fitted(np.inner(means25, binom_results.params))
resp_75 = glm_binom.family.fitted(np.inner(means75, binom_results.params))
diff = resp_75 - resp_25
print """The interquartile first difference for the percentage of low income
households in a school district is %2.4f %%""" % (diff*100)

means0 = means.copy()
means100 = means.copy()
means0[0] = data.exog[:,0].min()
means100[0] = data.exog[:,0].max()
resp_0 = glm_binom.family.fitted(np.inner(means0, binom_results.params))
resp_100 = glm_binom.family.fitted(np.inner(means100, binom_results.params))
diff_full = resp_100 - resp_0
print """The full range difference is %2.4f %%""" % (diff_full*100)

nobs = binom_results.nobs
y = data.endog[:,0]/trials
yhat = binom_results.mu

pylab.scatter(yhat, y)
line_fit = sm.OLS(y, sm.add_constant(yhat)).fit().params
fit = lambda x: line_fit[1]+line_fit[0]*x # better way in scipy?
pylab.plot(np.linspace(0,1,nobs), fit(np.linspace(0,1,nobs)))
pylab.title('Model Fit Plot')
pylab.ylabel('Observed values')
pylab.xlabel('Fitted values')
pylab.show()
pylab.clf()

pylab.scatter(yhat, binom_results.resid_pearson)
#pylab.plot(np.linspace(yhat.min()-.05, yhat.max()+.05,nobs), np.zeros_like(y))
pylab.plot([0.0, 1.0],[0.0, 0.0], 'k-')
pylab.title('Residual Dependence Plot')
pylab.ylabel('Pearson Residuals')
pylab.xlabel('Fitted values')
pylab.show()
pylab.clf()

res = binom_results.resid_deviance
stdres = (res - res.mean())/res.std()
pylab.hist(stdres, bins=25)
pylab.title('Histogram of standardized deviance residuals')
pylab.show()


### Example for using GLM Gamma for a proportional count response
# Brief description of the data and design
# print sm.datasets.scotland.DESCRLONG
data2 = sm.datasets.scotland.Load()
data2.exog = sm.add_constant(data2.exog)
glm_gamma = sm.GLM(data2.endog, data2.exog, family=sm.family.Gamma())
glm_results = glm_gamma.fit()

### Example for Gaussian distribution with a noncanonical link
nobs = 100
x = np.arange(nobs)
np.random.seed(54321)
X = np.column_stack((x,x**2))
X = sm.add_constant(X)
lny = np.exp(-(.03*x + .0001*x**2 - 1.0)) + .001 * np.random.rand(nobs)
gauss_log = sm.GLM(lny, X, family=sm.family.Gaussian(sm.family.links.log))
gauss_log_results = gauss_log.fit()
