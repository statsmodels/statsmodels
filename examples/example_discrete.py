"""Discrete Models
"""

import numpy as np
import statsmodels.api as sm

# Load the data from Spector and Mazzeo (1980). Examples follow Greene's
# Econometric Analysis Ch. 21 (5th Edition).
spector_data = sm.datasets.spector.load()
spector_data.exog = sm.add_constant(spector_data.exog, prepend=False)

# Linear Probability Model using OLS
lpm_mod = sm.OLS(spector_data.endog, spector_data.exog)
lpm_res = lpm_mod.fit()

# Logit Model
logit_mod = sm.Logit(spector_data.endog, spector_data.exog)
logit_res = logit_mod.fit()

# Probit Model
probit_mod = sm.Probit(spector_data.endog, spector_data.exog)
probit_res = probit_mod.fit()

# This example is based on Greene Table 21.1 5th Edition
# Linear Model Parameters
print lpm_res.params
# Logit Model Parameters
print logit_res.params
# Probit Model Parameters
print probit_res.params
#.. print "Typo in Greene for Weibull, replaced with logWeibull or Gumbel"
#.. print "(Tentatively) Weibull Model"
#.. print weibull_res.params

# Linear Model Parameters
print lpm_res.params[:-1]
# Logit Model Marginal Effects
print logit_res.margeff()
# Probit Model Marginal Effects
print probit_res.margeff()

# Multinomial Logit Example using American National Election Studies Data
anes_data = sm.datasets.anes96.load()
anes_exog = anes_data.exog
anes_exog[:,0] = np.log(anes_exog[:,0] + .1)
anes_exog = np.column_stack((anes_exog[:,0],anes_exog[:,2],anes_exog[:,5:8]))
anes_exog = sm.add_constant(anes_exog, prepend=False)
mlogit_mod = sm.MNLogit(anes_data.endog, anes_exog)
mlogit_res = mlogit_mod.fit()

# The default method for the fit is Newton-Raphson
# However, you can use other solvers
mlogit_res = mlogit_mod.fit(method='bfgs', maxiter=100)
#.. The below needs a lot of iterations to get it right?
#.. TODO: Add a technical note on algorithms
#.. mlogit_res = mlogit_mod.fit(method='ncg') # this takes forever

# Poisson model
# This is similar to Cameron and Trivedi's `Microeconometrics` Table 20.5;
# however, the data differs slightly from theirs
rand_data = sm.datasets.randhie.load()
rand_exog = rand_data.exog.view(float).reshape(len(rand_data.exog), -1)
rand_exog = sm.add_constant(rand_exog, prepend=False)
poisson_mod = sm.Poisson(rand_data.endog, rand_exog)
poisson_res = poisson_mod.fit(method="newton")

print poisson_res.summary()
