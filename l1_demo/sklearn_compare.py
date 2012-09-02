"""
For comparison with sklearn.linear_model.LogisticRegression
"""
from sklearn import linear_model
from sklearn import datasets
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

data_to_use = 'spector'

#### Load data
## The iris dataset from sklearn is perfectly separating...
## so not good for this comparison

## The Spector and Mazzeo (1980) data from statsmodels
## The Spector data gives very different results for statsmodels/sklearn
if data_to_use == 'spector':
    spector_data = sm.datasets.spector.load()
    X = spector_data.exog
    Y = spector_data.endog

## The Digits data from sklearn 
## The Digits data gives very different results for statsmodels/sklearn
if data_to_use == 'digits':
    digits = datasets.load_digits()
    X = digits.data
    Y = digits.target
    ## We are doing binary logistic regression, so don't include all targets
    X = X[Y<2]
    ## Keeping all of (this subset of) X leads to a singular hessian...so take
    ## some out.
    sigma = np.dot(X.T, X)
    nonconst_inds = np.nonzero(sigma.diagonal() > 0)[0]
    X = X[:, nonconst_inds[1:10]]
    Y = Y[Y<2]


#### Fit and print results
## Statsmodels
logit_mod = sm.Logit(Y, X)
## Standard logistic regression
logit_res = logit_mod.fit(method='newton', tol=1e-6)
print "\nStatsmodels coefficients"
print logit_res.params

## Sklearn
clf = linear_model.LogisticRegression(C=100000.0, penalty='l1', tol=1e-6)
clf.fit(X, Y)
print "\nsklearn coefficients"
print clf.coef_
