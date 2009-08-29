'''Examples: scikits.statsmodels.GLM
'''
import numpy as np
import scikits.statsmodels as sm

### Example for using GLM on binomial response data
### the input response vector is specified as (success, failure)

data = sm.datasets.star98.Load()
data.exog = sm.add_constant(data.exog)

print """The response variable is (success, failure).  Eg., the first
observation is """, data.endog[0]

glm_binom = sm.GLM(data.endog, data.exog, family=sm.family.Binomial())

### In order to fit this model, you must (for now) specify the number of
### trials per observation ie., success + failure
### This is the only time the data_weights argument should be used.

trials = data.endog.sum(axis=1)
binom_results = glm_binom.fit(data_weights = trials)

### Example for using GLM Gamma
data2 = sm.datasets.scotvote.Load()
data2.exog = sm.add_constant(data2.exog)
glm_gamma = sm.GLM(data.endog, data.exog, family=sm.family.Gamma())
glm_results = glm_gamma.fit()

### Example for Gaussian link with a noncanonical link
nobs = 100
x = np.arange(nobs)
np.random.seed(54321)
X = np.column_stack(x,x**2)
X = sm.add_constant(X)
lny = np.exp(-(.03*x + .0001*x**2 - 1.0)) + .001 * np.random.rand(nobs)
gauss_log = sm.GLM(lny, X, family=sm.family.Gaussian(sm.family.links.log))
gauss_log_results = gauss_log.fit()
