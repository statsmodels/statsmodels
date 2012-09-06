"""
You can fit your LikelihoodModel using l1 regularization by changing
    the method argument and adding an argument alpha.  See code for 
    details.

The Story
---------
The maximum likelihood (ML) solution works well when the number of data 
points is large and the noise is small.  When the ML solution starts 
"breaking", the regularized solution should do better.

The l1 Solvers
--------------
The solvers are slower than standard Newton, and sometimes have 
    convergence issues Nonetheless, the final solution makes sense and 
    is often better than the ML solution.
The standard l1 solver is fmin_slsqp and is included with scipy.  It 
    sometimes has trouble verifying convergence when the data size is 
    large.
The l1_cvxopt_cp solver is part of CVXOPT and this package needs to be 
    installed separately.  It works well even for larger data sizes.
"""
import statsmodels.api as sm
import numpy as np


# Load the data from Spector and Mazzeo (1980)
spector_data = sm.datasets.spector.load()
spector_data.exog = sm.add_constant(spector_data.exog, prepend=True)
N = len(spector_data.endog)
K = spector_data.exog.shape[1]

### Logit Model
logit_mod = sm.Logit(spector_data.endog, spector_data.exog)
## Standard logistic regression
logit_res = logit_mod.fit()
## Regularized regression
# Set the reularization parameter to something reasonable
alpha = 0.05 * N * np.ones(K)
# Use l1, which solves via a built-in (scipy.optimize) solver
logit_l1_res = logit_mod.fit(method='l1', alpha=alpha, trim_params=True)
# Use l1_cvxopt_cp, which solves with a CVXOPT solver
logit_l1_cvxopt_res = logit_mod.fit(method='l1_cvxopt_cp', alpha=alpha)
## Print results
print logit_res.summary()
print logit_l1_res.summary()
print logit_l1_cvxopt_res.summary()

### Multinomial Logit Example using American National Election Studies Data
anes_data = sm.datasets.anes96.load()
anes_exog = anes_data.exog
anes_exog[:,0] = np.log(anes_exog[:,0] + .1)
anes_exog = np.column_stack((anes_exog[:,0],anes_exog[:,2],anes_exog[:,5:8]))
anes_exog = sm.add_constant(anes_exog, prepend=False)
mlogit_mod = sm.MNLogit(anes_data.endog, anes_exog)
mlogit_res = mlogit_mod.fit()
## Set the regularization parameter.  
alpha = 10 * np.ones((mlogit_mod.J - 1, mlogit_mod.K))
# Don't regularize the constant
alpha[-1,:] = 0
mlogit_l1_res = mlogit_mod.fit(method='l1', alpha=alpha, trim_params=True)
## Print results
print mlogit_res.summary()
print mlogit_l1_res.summary()
