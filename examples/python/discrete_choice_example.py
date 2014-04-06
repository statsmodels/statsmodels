
## Discrete Choice Models

### Fair's Affair data

# A survey of women only was conducted in 1974 by *Redbook* asking about extramarital affairs.
from __future__ import print_function
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import logit, probit, poisson, ols


print(sm.datasets.fair.SOURCE)


print(sm.datasets.fair.NOTE)


dta = sm.datasets.fair.load_pandas().data


dta['affair'] = (dta['affairs'] > 0).astype(float)
print(dta.head(10))


print(dta.describe())


affair_mod = logit("affair ~ occupation + educ + occupation_husb" 
                   "+ rate_marriage + age + yrs_married + children"
                   " + religious", dta).fit()


print(affair_mod.summary())


# How well are we predicting?

affair_mod.pred_table()


# The coefficients of the discrete choice model do not tell us much. What we're after is marginal effects.

mfx = affair_mod.get_margeff()
print(mfx.summary())


respondent1000 = dta.ix[1000]
print(respondent1000)


resp = dict(zip(range(1,9), respondent1000[["occupation", "educ", 
                                            "occupation_husb", "rate_marriage", 
                                            "age", "yrs_married", "children", 
                                            "religious"]].tolist()))
resp.update({0 : 1})
print(resp)


mfx = affair_mod.get_margeff(atexog=resp)
print(mfx.summary())


affair_mod.predict(respondent1000)


affair_mod.fittedvalues[1000]


affair_mod.model.cdf(affair_mod.fittedvalues[1000])


# The "correct" model here is likely the Tobit model. We have an work in progress branch "tobit-model" on github, if anyone is interested in censored regression models.

#### Exercise: Logit vs Probit

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
support = np.linspace(-6, 6, 1000)
ax.plot(support, stats.logistic.cdf(support), 'r-', label='Logistic')
ax.plot(support, stats.norm.cdf(support), label='Probit')
ax.legend();


fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
support = np.linspace(-6, 6, 1000)
ax.plot(support, stats.logistic.pdf(support), 'r-', label='Logistic')
ax.plot(support, stats.norm.pdf(support), label='Probit')
ax.legend();


# Compare the estimates of the Logit Fair model above to a Probit model. Does the prediction table look better? Much difference in marginal effects?

#### Genarlized Linear Model Example

print(sm.datasets.star98.SOURCE)


print(sm.datasets.star98.DESCRLONG)


print(sm.datasets.star98.NOTE)


dta = sm.datasets.star98.load_pandas().data
print(dta.columns)


print(dta[['NABOVE', 'NBELOW', 'LOWINC', 'PERASIAN', 'PERBLACK', 'PERHISP', 'PERMINTE']].head(10))


print(dta[['AVYRSEXP', 'AVSALK', 'PERSPENK', 'PTRATIO', 'PCTAF', 'PCTCHRT', 'PCTYRRND']].head(10))


formula = 'NABOVE + NBELOW ~ LOWINC + PERASIAN + PERBLACK + PERHISP + PCTCHRT '
formula += '+ PCTYRRND + PERMINTE*AVYRSEXP*AVSALK + PERSPENK*PTRATIO*PCTAF'


##### Aside: Binomial distribution

# Toss a six-sided die 5 times, what's the probability of exactly 2 fours?

stats.binom(5, 1./6).pmf(2)


from scipy.misc import comb
comb(5,2) * (1/6.)**2 * (5/6.)**3


from statsmodels.formula.api import glm
glm_mod = glm(formula, dta, family=sm.families.Binomial()).fit()


print(glm_mod.summary())


# The number of trials 

glm_mod.model.data.orig_endog.sum(1)


glm_mod.fittedvalues * glm_mod.model.data.orig_endog.sum(1)


# First differences: We hold all explanatory variables constant at their means and manipulate the percentage of low income households to assess its impact
# on the response variables:

exog = glm_mod.model.data.orig_exog # get the dataframe


means25 = exog.mean()
print(means25)


means25['LOWINC'] = exog['LOWINC'].quantile(.25)
print(means25)


means75 = exog.mean()
means75['LOWINC'] = exog['LOWINC'].quantile(.75)
print(means75)


resp25 = glm_mod.predict(means25)
resp75 = glm_mod.predict(means75)
diff = resp75 - resp25


# The interquartile first difference for the percentage of low income households in a school district is:

print("%2.4f%%" % (diff[0]*100))


nobs = glm_mod.nobs
y = glm_mod.model.endog
yhat = glm_mod.mu


from statsmodels.graphics.api import abline_plot
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, ylabel='Observed Values', xlabel='Fitted Values')
ax.scatter(yhat, y)
y_vs_yhat = sm.OLS(y, sm.add_constant(yhat, prepend=True)).fit()
fig = abline_plot(model_results=y_vs_yhat, ax=ax)


##### Plot fitted values vs Pearson residuals

# Pearson residuals are defined to be 
# 
# $$\frac{(y - \mu)}{\sqrt{(var(\mu))}}$$
# 
# where var is typically determined by the family. E.g., binomial variance is $np(1 - p)$

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, title='Residual Dependence Plot', xlabel='Fitted Values',
                          ylabel='Pearson Residuals')
ax.scatter(yhat, stats.zscore(glm_mod.resid_pearson))
ax.axis('tight')
ax.plot([0.0, 1.0],[0.0, 0.0], 'k-');


##### Histogram of standardized deviance residuals with Kernel Density Estimate overlayed

# The definition of the deviance residuals depends on the family. For the Binomial distribution this is 
# 
# $$r_{dev} = sign\(Y-\mu\)*\sqrt{2n(Y\log\frac{Y}{\mu}+(1-Y)\log\frac{(1-Y)}{(1-\mu)}}$$
# 
# They can be used to detect ill-fitting covariates

resid = glm_mod.resid_deviance
resid_std = stats.zscore(resid) 
kde_resid = sm.nonparametric.KDEUnivariate(resid_std)
kde_resid.fit()


fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, title="Standardized Deviance Residuals")
ax.hist(resid_std, bins=25, normed=True);
ax.plot(kde_resid.support, kde_resid.density, 'r');


##### QQ-plot of deviance residuals

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
fig = sm.graphics.qqplot(resid, line='r', ax=ax)

