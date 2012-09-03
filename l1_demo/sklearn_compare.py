"""
For comparison with sklearn.linear_model.LogisticRegression

Computes a regularzation path with both packages.  The coefficient values in
    either path are related by a constant.  We find the reparameterization of
    the statsmodels path that makes the paths match up.
"""
from sklearn import linear_model
from sklearn import datasets
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import pdb

#### Load data
## The Spector and Mazzeo (1980) data from statsmodels
spector_data = sm.datasets.spector.load()
X = spector_data.exog
Y = spector_data.endog

#### Fit and plot results
N = 200  # number of points to solve at

## Statsmodels
logit_mod = sm.Logit(Y, X)
sm_coeff = np.zeros((N, 3))  # Holds the coefficients
#for n, alpha in enumerate(np.linspace(0, 3, N)):
alphas = 1 / np.logspace(-1, 2, N)
for n, alpha in enumerate(alphas):
    logit_res = logit_mod.fit(method='l1', alpha=alpha)
    sm_coeff[n,:] = logit_res.params
# The sm_coeff order needs to be reversed to match up with sk_coeff
sm_coeff = sm_coeff[::-1, :]

## Sklearn
sk_coeff = np.zeros((N, 3))
for n, C in enumerate(np.logspace(-0.45, 2, N)):
    clf = linear_model.LogisticRegression(
            C=C, penalty='l1', fit_intercept=False)
    clf.fit(X, Y)
    sk_coeff[n, :] = clf.coef_

## Get the reparametrization of sm_coeff that makes the paths equal
# Do this by finding one single re-parameterization of the second coefficient
# that makes the path for the second coefficient (almost) identical.  This
# same parameterization will work for the other two coefficients since the
# the regularization coefficients (in sk and sm) are related by a constant.
#
# X2 is chosen since this coefficient becomes non-zero before the other two.
sk_X2 = sk_coeff[:,2]
sm_X2 = sm_coeff[:,2]
s = np.zeros(N)
s = np.searchsorted(sk_X2, sm_X2)

## Plot
plt.figure(1);plt.clf();plt.grid()
plt.xlabel('Index in sklearn simulation')
plt.ylabel('Coefficient value')
plt.title('Regularization Paths')
colors = ['b', 'r', 'k']
for coeff, name in [(sm_coeff, 'sm'), (sk_coeff, 'sk')]:
    if name == 'sk':
        ltype = 'x'  # linetype
        t = range(N)  # The 'time' parameter
    else:
        ltype = 'o'
        t = s
    for i in xrange(3):
        plt.plot(t, coeff[:,i], ltype+colors[i], label=name+'-X'+str(i))
plt.legend(loc='best')
