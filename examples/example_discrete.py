"""Discrete Data Models
"""

import numpy as np
import statsmodels.api as sm

# Load data from Spector and Mazzeo (1980). Examples follow Greene's
# Econometric Analysis Ch. 21 (5th Edition).
spector_data = sm.datasets.spector.load()
spector_data.exog = sm.add_constant(spector_data.exog, prepend=False)

# Linear Probability Model (OLS)
--------------------------------
lpm_mod = sm.OLS(spector_data.endog, spector_data.exog)
lpm_res = lpm_mod.fit()
print lpm_res.params[:-1]

#Logit Model
#-----------
logit_mod = sm.Logit(spector_data.endog, spector_data.exog)
logit_res = logit_mod.fit()
print logit_res.params
print logit_res.margeff()

# As in all the discrete data models presented below, we can print a nice
# summary of results:
print logit_res.summary()

#Probit Model 
#------------
probit_mod = sm.Probit(spector_data.endog, spector_data.exog)
probit_res = probit_mod.fit()
print probit_res.params
print probit_res.margeff()

#Multinomial Logit
#-----------------

# Load data from the American National Election Studies:
anes_data = sm.datasets.anes96.load()
anes_exog = anes_data.exog
anes_exog[:,0] = np.log(anes_exog[:,0] + .1)
anes_exog = np.column_stack((anes_exog[:,0],anes_exog[:,2],anes_exog[:,5:8]))
anes_exog = sm.add_constant(anes_exog, prepend=False)

# Fit MNL model
mlogit_mod = sm.MNLogit(anes_data.endog, anes_exog)
mlogit_res = mlogit_mod.fit()

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

#Alternative solvers
#-------------------

# The default method for the fit in discrete data MLE models is Newton-Raphson.
# You can use other solvers by using the ``method`` argument: 
mlogit_res = mlogit_mod.fit(method='bfgs', maxiter=100)

#.. The below needs a lot of iterations to get it right?
#.. TODO: Add a technical note on algorithms
#.. mlogit_res = mlogit_mod.fit(method='ncg') # this takes forever


