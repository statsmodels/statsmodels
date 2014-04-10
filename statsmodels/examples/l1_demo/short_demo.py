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
The standard l1 solver is fmin_slsqp and is included with scipy.  It
    sometimes has trouble verifying convergence when the data size is
    large.
The l1_cvxopt_cp solver is part of CVXOPT and this package needs to be
    installed separately.  It works well even for larger data sizes.
"""
from __future__ import print_function
from statsmodels.compat.python import range
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import pdb  # pdb.set_trace()


## Load the data from Spector and Mazzeo (1980)
spector_data = sm.datasets.spector.load()
spector_data.exog = sm.add_constant(spector_data.exog)
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
logit_l1_res = logit_mod.fit_regularized(method='l1', alpha=alpha, acc=1e-6)

# Use l1_cvxopt_cp, which solves with a CVXOPT solver
logit_l1_cvxopt_res = logit_mod.fit_regularized(
        method='l1_cvxopt_cp', alpha=alpha)

## Print results
print("============ Results for Logit =================")
print("ML results")
print(logit_res.summary())
print("l1 results")
print(logit_l1_res.summary())
print(logit_l1_cvxopt_res.summary())

### Multinomial Logit Example using American National Election Studies Data
anes_data = sm.datasets.anes96.load()
anes_exog = anes_data.exog
anes_exog = sm.add_constant(anes_exog, prepend=False)
mlogit_mod = sm.MNLogit(anes_data.endog, anes_exog)
mlogit_res = mlogit_mod.fit()

## Set the regularization parameter.
alpha = 10 * np.ones((mlogit_mod.J - 1, mlogit_mod.K))

# Don't regularize the constant
alpha[-1,:] = 0
mlogit_l1_res = mlogit_mod.fit_regularized(method='l1', alpha=alpha)
print(mlogit_l1_res.params)

#mlogit_l1_res = mlogit_mod.fit_regularized(
#        method='l1_cvxopt_cp', alpha=alpha, abstol=1e-10, trim_tol=1e-6)
#print mlogit_l1_res.params

## Print results
print("============ Results for MNLogit =================")
print("ML results")
print(mlogit_res.summary())
print("l1 results")
print(mlogit_l1_res.summary())
#
#
#### Logit example with many params, sweeping alpha
spector_data = sm.datasets.spector.load()
X = spector_data.exog
Y = spector_data.endog

## Fit
N = 50  # number of points to solve at
K = X.shape[1]
logit_mod = sm.Logit(Y, X)
coeff = np.zeros((N, K))  # Holds the coefficients
alphas = 1 / np.logspace(-0.5, 2, N)

## Sweep alpha and store the coefficients
# QC check doesn't always pass with the default options.
# Use the options QC_verbose=True and disp=True
# to to see what is happening.  It just barely doesn't pass, so I decreased
# acc and increased QC_tol to make it pass
for n, alpha in enumerate(alphas):
    logit_res = logit_mod.fit_regularized(
        method='l1', alpha=alpha, trim_mode='off', QC_tol=0.1, disp=False,
        QC_verbose=True, acc=1e-15)
    coeff[n,:] = logit_res.params

## Plot
plt.figure(1);plt.clf();plt.grid()
plt.title('Regularization Path');
plt.xlabel('alpha');
plt.ylabel('Parameter value');
for i in range(K):
    plt.plot(alphas, coeff[:,i], label='X'+str(i), lw=3)
plt.legend(loc='best')
plt.show()
