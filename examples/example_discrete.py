"""Discrete Data Models
"""

import numpy as np
import statsmodels.api as sm

# Load data from Spector and Mazzeo (1980). Examples follow Greene's
# Econometric Analysis Ch. 21 (5th Edition).
spector_data = sm.datasets.spector.load()
spector_data.exog = sm.add_constant(spector_data.exog, prepend=False)

# Inspect the data:
spector_data.exog[:5,:]
spector_data.endog[:5]

# Linear Probability Model (OLS)
#-------------------------------
lpm_mod = sm.OLS(spector_data.endog, spector_data.exog)
lpm_res = lpm_mod.fit()
print lpm_res.params[:-1]

#Logit Model
#-----------
logit_mod = sm.Logit(spector_data.endog, spector_data.exog)
logit_res = logit_mod.fit()
print logit_res.params
logit_margeff = logit_res.get_margeff(method='dydx', at='overall')
print logit_margeff.summary()

#l1 regularized logit
#-----------
alpha = 0.1 * len(spector_data.endog) * np.ones(spector_data.exog.shape[1])
alpha[-1] = 0
logit_l1_res = logit_mod.fit_regularized(method='l1', alpha=alpha)
print logit_l1_res.summary()


# As in all the discrete data models presented below, we can print a nice
# summary of results:
print logit_res.summary()

#Probit Model
#------------
probit_mod = sm.Probit(spector_data.endog, spector_data.exog)
probit_res = probit_mod.fit()
print probit_res.params
probit_margeff = probit_res.get_margeff()
print probit_margeff.summary()

#Multinomial Logit
#-----------------

# Load data from the American National Election Studies:
anes_data = sm.datasets.anes96.load()
anes_exog = anes_data.exog
anes_exog = sm.add_constant(anes_exog, prepend=False)

# Inspect the data:
anes_data.exog[:5,:]
anes_data.endog[:5]

# Fit MNL model
mlogit_mod = sm.MNLogit(anes_data.endog, anes_exog)
mlogit_res = mlogit_mod.fit()
print mlogit_res.params
mlogit_margeff = mlogit_res.get_margeff()
print mlogit_margeff.summary()

#l1 regularized Multinomial Logit
#-----------------
alpha = 10 * np.ones((mlogit_mod.J - 1, mlogit_mod.K))
alpha[-1,:] = 0
mlogit_mod2 = sm.MNLogit(anes_data.endog, anes_exog)
mlogit_l1_res = mlogit_mod2.fit_regularized(method='l1', alpha=alpha)
print mlogit_l1_res.summary()

#Poisson model
#-------------

# Load the Rand data. Note that this example is similar to Cameron and
# Trivedi's `Microeconometrics` Table 20.5, but it is slightly different
# because of minor changes in the data.
rand_data = sm.datasets.randhie.load()
rand_exog = rand_data.exog.view(float).reshape(len(rand_data.exog), -1)
rand_exog = sm.add_constant(rand_exog, prepend=False)

# Fit Poisson model:
poisson_mod = sm.Poisson(rand_data.endog, rand_exog)
poisson_res = poisson_mod.fit(method="newton")
print poisson_res.summary()
poisson_margeff = poisson_res.get_margeff()
print poisson_margeff.summary()

# l1 regularized Poisson model
poisson_mod2 = sm.Poisson(rand_data.endog, rand_exog)
alpha = 0.1 * len(rand_data.endog) * np.ones(rand_exog.shape[1])
alpha[-1] = 0
poisson_l1_res = poisson_mod2.fit_regularized(method='l1', alpha=alpha)

#Alternative solvers
#-------------------

# The default method for fitting discrete data MLE models is Newton-Raphson.
# You can use other solvers by using the ``method`` argument:
mlogit_res = mlogit_mod.fit(method='bfgs', maxiter=500)

#.. The below needs a lot of iterations to get it right?
#.. TODO: Add a technical note on algorithms
#.. mlogit_res = mlogit_mod.fit(method='ncg') # this takes forever


