'''Example: Principal Component Regression

* simulate model with 2 factors and 4 explanatory variables
* use pca to extract factors from data,
* run OLS on factors,
* use information criteria to choose "best" model

Warning: pca sorts factors by explaining variance in explanatory variables,
which are not necessarily the most important factors for explaining the
endogenous variable.

# try out partial correlation for dropping (or adding) factors
# get algorithm for partial least squares as an alternative to PCR

'''


import numpy as np
from numpy.testing import assert_array_almost_equal
import scikits.statsmodels as sm
from scikits.statsmodels.sandbox.tools import pca


# Example: principal component regression
nobs = 1000
f0 = np.c_[np.random.normal(size=(nobs,2)), np.ones((nobs,1))]
f2xcoef = np.c_[np.repeat(np.eye(2),2,0),np.arange(4)[::-1]].T
f2xcoef = np.array([[ 1.,  1.,  0.,  0.],
                    [ 0.,  0.,  1.,  1.],
                    [ 3.,  2.,  1.,  0.]])
f2xcoef = np.array([[ 0.1,  3.,  1.,    0.],
                    [ 0.,  0.,  1.5,   0.1],
                    [ 3.,  2.,  1.,    0.]])
x0 = np.dot(f0, f2xcoef)
x0 += 0.1*np.random.normal(size=x0.shape)
ytrue = np.dot(f0,[1., 1., 1.])
y0 = ytrue + 0.1*np.random.normal(size=ytrue.shape)

xred, fact, eva, eve  = pca(x0, keepdim=0)
print eve
print fact[:5]
print f0[:5]

import scikits.statsmodels as sm

res = sm.OLS(y0, sm.add_constant(x0)).fit()
print 'OLS on original data'
print res.params
print res.aic
print res.rsquared

#print 'OLS on Factors'
#for k in range(x0.shape[1]):
#    xred, fact, eva, eve  = pca(x0, keepdim=k, normalize=1)
#    fact_wconst = sm.add_constant(fact)
#    res = sm.OLS(y0, fact_wconst).fit()
#    print 'k =', k
#    print res.params
#    print 'aic:  ', res.aic
#    print 'bic:  ', res.bic
#    print 'llf:  ', res.llf
#    print 'R2    ', res.rsquared
#    print 'R2 adj', res.rsquared_adj

print 'OLS on Factors'
results = []
xred, fact, eva, eve  = pca(x0, keepdim=0, normalize=1)
for k in range(0, x0.shape[1]+1):
    #xred, fact, eva, eve  = pca(x0, keepdim=k, normalize=1)
    # this is faster and same result
    fact_wconst = sm.add_constant(fact[:,:k])
    res = sm.OLS(y0, fact_wconst).fit()
    print 'k =', k
    print res.params
    print 'aic:  ', res.aic
    print 'bic:  ', res.bic
    print 'llf:  ', res.llf
    print 'R2    ', res.rsquared
    print 'R2 adj', res.rsquared_adj
    results.append([k, res.aic, res.bic, res.rsquared_adj])

results = np.array(results)
print results
print 'best result for k, by AIC, BIC, R2_adj'
print np.r_[(np.argmin(results[:,1:3],0), np.argmax(results[:,3],0))]



