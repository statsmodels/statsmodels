'''Generalized Linear Models
'''
import numpy as np
import statsmodels.api as sm
from scipy import stats
from matplotlib import pyplot as plt

#Example for using GLM on binomial response data
#the input response vector in this case is N by 2 (success, failure)
#This data is taken with permission from
#Jeff Gill (2000) Generalized linear models: A unified approach

#The response variable is
#(of students above the math national median, # of students below)

#| The explanatory variables are (in column order)
#| The proportion of low income families "LOWINC"
#| The proportions of minority students,"PERASIAN","PERBLACK","PERHISP"
#| The percentage of minority teachers "PERMINTE",
#| The median teacher salary including benefits in 1000s "AVSALK"
#| The mean teacher experience in years "AVYRSEXP",
#| The per-pupil expenditures in thousands "PERSPENK"
#| The pupil-teacher ratio "PTRATIO"
#| The percent of students taking college credit courses "PCTAF",
#| The percentage of charter schools in the districut "PCTCHRT"
#| The percent of schools in the district operating year round "PCTYRRND"
#| The following are interaction terms "PERMINTE_AVYRSEXP","PERMINTE_AVSAL",
#| "AVYRSEXP_AVSAL","PERSPEN_PTRATIO","PERSPEN_PCTAF","PTRATIO_PCTAF",
#| "PERMINTE_AVYRSEXP_AVSAL","PERSPEN_PTRATIO_PCTAF"

data = sm.datasets.star98.load()
data.exog = sm.add_constant(data.exog)

#The response variable is (success, failure).  Eg., the first
#observation is
print data.endog[0]
#Giving a total number of trials for this observation of
#print data.endog[0].sum()

glm_binom = sm.GLM(data.endog, data.exog, family=sm.families.Binomial())

binom_results = glm_binom.fit()
#The fitted values are
print binom_results.params
#The corresponding t-values are
print binom_results.tvalues

#It is common in GLMs with interactions to compare first differences.
#We are interested in the difference of the impact of the explanatory variable
#on the response variable.  This example uses interquartile differences for
#the percentage of low income households while holding the other values
#constant at their mean.

means = data.exog.mean(axis=0)
means25 = means.copy()
means25[0] = stats.scoreatpercentile(data.exog[:,0], 25)
means75 = means.copy()
means75[0] = lowinc_75per = stats.scoreatpercentile(data.exog[:,0], 75)
resp_25 = binom_results.predict(means25)
resp_75 = binom_results.predict(means75)
diff = resp_75 - resp_25
#.. print """The interquartile first difference for the percentage of low income
#.. households in a school district is %2.4f %%""" % (diff*100)
#The interquartile first difference for the percentage of low income
#households in a school district is
print diff*100

means0 = means.copy()
means100 = means.copy()
means0[0] = data.exog[:,0].min()
means100[0] = data.exog[:,0].max()
resp_0 = binom_results.predict(means0)
resp_100 = binom_results.predict(means100)
diff_full = resp_100 - resp_0
print """The full range difference is %2.4f %%""" % (diff_full*100)

nobs = binom_results.nobs
y = data.endog[:,0]/data.endog.sum(1)
yhat = binom_results.mu

#Plot of yhat vs y
plt.figure()
plt.scatter(yhat, y)
line_fit = sm.OLS(y, sm.add_constant(yhat)).fit().params
fit = lambda x: line_fit[1]+line_fit[0]*x # better way in scipy?
plt.plot(np.linspace(0,1,nobs), fit(np.linspace(0,1,nobs)))
plt.title('Model Fit Plot')
plt.ylabel('Observed values')
#@savefig glm_fitted.png
plt.xlabel('Fitted values')

#Plot of yhat vs. Pearson residuals
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

#.. plt.show()
#.. plt.close('all')


#Example for using GLM Gamma for a proportional count response
#Brief description of the data and design
print sm.datasets.scotland.DESCRLONG
data2 = sm.datasets.scotland.load()
data2.exog = sm.add_constant(data2.exog)
glm_gamma = sm.GLM(data2.endog, data2.exog, family=sm.families.Gamma())
glm_results = glm_gamma.fit()

##Example for Gaussian distribution with a noncanonical link
nobs2 = 100
x = np.arange(nobs2)
np.random.seed(54321)
X = np.column_stack((x,x**2))
X = sm.add_constant(X)
lny = np.exp(-(.03*x + .0001*x**2 - 1.0)) + .001 * np.random.rand(nobs2)
gauss_log = sm.GLM(lny, X, family=sm.families.Gaussian(sm.families.links.log))
gauss_log_results = gauss_log.fit()

#check summary
print binom_results.summary()
