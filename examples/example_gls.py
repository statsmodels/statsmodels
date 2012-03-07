"""Generalized Least Squares
"""

import statsmodels.api as sm
import numpy as np
data = sm.datasets.longley.load()
data.exog = sm.add_constant(data.exog)

# The Longley dataset is a time series dataset
# Let's assume that the data is heteroskedastic and that we know
# the nature of the heteroskedasticity.  We can then define
# `sigma` and use it to give us a GLS model

# First we will obtain the residuals from an OLS fit

ols_resid = sm.OLS(data.endog, data.exog).fit().resid

# Assume that the error terms follow an AR(1) process with a trend
# resid[i] = beta_0 + rho*resid[i-1] + e[i]
# where e ~ N(0,some_sigma**2)
# and that rho is simply the correlation of the residuals
# a consistent estimator for rho is to regress the residuals
# on the lagged residuals

resid_fit = sm.OLS(ols_resid[1:], sm.add_constant(ols_resid[:-1])).fit()
print resid_fit.tvalues[0]
print resid_fit.pvalues[0]
# While we don't have strong evidence that the errors follow an AR(1)
# process we continue

rho = resid_fit.params[0]

# As we know, an AR(1) process means that near-neighbors have a stronger
# relation so we can give this structure by using a toeplitz matrix

from scipy.linalg import toeplitz

toeplitz(range(5))
#.. array([[0, 1, 2, 3, 4],
#..       [1, 0, 1, 2, 3],
#..       [2, 1, 0, 1, 2],
#..       [3, 2, 1, 0, 1],
#..       [4, 3, 2, 1, 0]])

order = toeplitz(range(len(ols_resid)))

# so that our error covariance structure is actually rho**order
# which defines an autocorrelation structure

sigma = rho**order

gls_model = sm.GLS(data.endog, data.exog, sigma=sigma)
gls_results = gls_model.fit()

# of course, the exact rho in this instance is not known so it
# it might make more sense to use feasible gls, which currently only
# has experimental support

# We can use the GLSAR model with one lag, to get to a similar result

glsar_model = sm.GLSAR(data.endog, data.exog, 1)
glsar_results = glsar_model.iterative_fit(1)

# comparing gls and glsar results, we see that there are some small
# differences in the parameter estimates and the resulting standard
# errors of the parameter estimate. This might be do to the numerical
# differences in the algorithm, e.g. the treatment of initial conditions,
# because of the small number of observations in the longley dataset.
print gls_results.params
print glsar_results.params
print gls_results.bse
print glsar_results.bse

