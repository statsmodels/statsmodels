
## Discrete Choice Models Overview
from __future__ import print_function
import numpy as np
import statsmodels.api as sm


# ## Data
# 
# Load data from Spector and Mazzeo (1980). Examples follow Greene's Econometric Analysis Ch. 21 (5th Edition).

spector_data = sm.datasets.spector.load()
spector_data.exog = sm.add_constant(spector_data.exog, prepend=False)


# Inspect the data:

print(spector_data.exog[:5,:])
print(spector_data.endog[:5])


# ## Linear Probability Model (OLS)

lpm_mod = sm.OLS(spector_data.endog, spector_data.exog)
lpm_res = lpm_mod.fit()
print('Parameters: ', lpm_res.params[:-1])


# ## Logit Model

logit_mod = sm.Logit(spector_data.endog, spector_data.exog)
logit_res = logit_mod.fit(disp=0)
print('Parameters: ', logit_res.params)


# Marginal Effects

margeff = logit_res.get_margeff()
print(margeff.summary())


# As in all the discrete data models presented below, we can print(a nice summary of results:

print(logit_res.summary())


# ## Probit Model 

probit_mod = sm.Probit(spector_data.endog, spector_data.exog)
probit_res = probit_mod.fit()
probit_margeff = probit_res.get_margeff()
print('Parameters: ', probit_res.params)
print('Marginal effects: ')
print(probit_margeff.summary())


# ## Multinomial Logit

# Load data from the American National Election Studies:

anes_data = sm.datasets.anes96.load()
anes_exog = anes_data.exog
anes_exog = sm.add_constant(anes_exog, prepend=False)


# Inspect the data:

print(anes_data.exog[:5,:])
print(anes_data.endog[:5])


# Fit MNL model:

mlogit_mod = sm.MNLogit(anes_data.endog, anes_exog)
mlogit_res = mlogit_mod.fit()
print(mlogit_res.params)


# ## Poisson
# 
# Load the Rand data. Note that this example is similar to Cameron and Trivedi's `Microeconometrics` Table 20.5, but it is slightly different because of minor changes in the data. 

rand_data = sm.datasets.randhie.load()
rand_exog = rand_data.exog.view(float, type=np.ndarray).reshape(len(rand_data.exog), -1)
rand_exog = sm.add_constant(rand_exog, prepend=False)


# Fit Poisson model: 

poisson_mod = sm.Poisson(rand_data.endog, rand_exog)
poisson_res = poisson_mod.fit(method="newton")
print(poisson_res.summary())


# ## Negative Binomial
# 
# The negative binomial model gives slightly different results. 

mod_nbin = sm.NegativeBinomial(rand_data.endog, rand_exog)
res_nbin = mod_nbin.fit(disp=False)
print(res_nbin.summary())


# ## Alternative solvers
# 
# The default method for fitting discrete data MLE models is Newton-Raphson. You can use other solvers by using the ``method`` argument: 

mlogit_res = mlogit_mod.fit(method='bfgs', maxiter=100)
print(mlogit_res.summary())

