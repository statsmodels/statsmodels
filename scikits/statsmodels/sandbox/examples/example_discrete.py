"""Example: scikits.statsmodels.sandbox.discretemod
"""

import numpy as np
from scikits.statsmodels.sandbox.discretemod import *
import scikits.statsmodels as sm

# Load the data from Spector and Mazzeo (1980)
# Examples follow Greene's Econometric Analysis Ch. 21 (5th Edition).
spector_data = sm.datasets.spector.Load()
spector_data.exog = sm.add_constant(spector_data.exog)

# Linear Probability Model using OLS
lpm_mod = sm.OLS(spector_data.endog,spector_data.exog)
lmp_res = lpm_mod.fit()

# Logit Model
logit_mod = Logit(spector_data.endog, spector_data.exog)
logit_res = logit_mod.fit()

# Probit Model
probit_mod = Probit(spector_data.endog, spector_data.exog)
probit_res = probit_mod.fit()

print "This example is based on Greene Table 21.1 5th Edition"
print "Linear Model"
print lmp_res.params
print "Logit Model"
print logit_res.params
print "Probit Model"
print probit_res.params
#print "Typo in Greene for Weibull, replaced with logWeibull or Gumbel"
#print "(Tentatively) Weibull Model"
#print weibull_res.params

anes_data = sm.datasets.anes96.Load()
anes_exog = anes_data.exog
anes_exog[:,0] = np.log(anes_exog[:,0] + .1)
anes_exog = np.column_stack((anes_exog[:,0],anes_exog[:,2],anes_exog[:,5:8]))
anes_exog = sm.add_constant(anes_exog)
mlogit_mod = MNLogit(anes_data.endog, anes_exog)
mlogit_res = mlogit_mod.fit()

# The default method for the fit is Newton-Raphson
# However, you can use other solvers
mlogit_res = mlogit_mod.fit(method='bfgs')
mlogit_res = mlogit_mod.fit(method='ncg') # this takes forever

# Example from http://www.ats.ucla.edu/stat/r/dae/mlogit.htm
#mlog_data = np.genfromtxt('http://www.ats.ucla.edu/stat/r/dae/mlogit.csv',
#        delimiter=',', names=True)
#mlog_endog = mlog_data['brand']
#mlog_exog = mlog_data[['female','age']].view(float).reshape(-1,2)
#mlog_exog = sm.add_constant(mlog_exog, prepend=True)
#mlog_mod = MNLogit(mlog_endog, mlog_exog)
#mlog_res = mlog_mod.fit(method='newton')
#marr = np.array([[22.721396, 10.946741],[-.465941,.057873],
#        [-.685908,-.317702]])
# The above are the results from R using Brand 3 as base outcome
#marr = np.array([[-11.77466, -22.7214],[.5238143, .4659414],
#        [.3682065, .6859082]])
# The above results are from Stata using Brand 1 as base outcome
# we match these, but will provide a baseoutcome option eventually.
