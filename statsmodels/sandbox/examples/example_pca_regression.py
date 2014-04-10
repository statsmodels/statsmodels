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
import statsmodels.api as sm
from statsmodels.sandbox.tools import pca
from statsmodels.sandbox.tools.cross_val import LeaveOneOut


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
print(eve)
print(fact[:5])
print(f0[:5])

import statsmodels.api as sm

res = sm.OLS(y0, sm.add_constant(x0, prepend=False)).fit()
print('OLS on original data')
print(res.params)
print(res.aic)
print(res.rsquared)

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

print('OLS on Factors')
results = []
xred, fact, eva, eve  = pca(x0, keepdim=0, normalize=1)
for k in range(0, x0.shape[1]+1):
    #xred, fact, eva, eve  = pca(x0, keepdim=k, normalize=1)
    # this is faster and same result
    fact_wconst = sm.add_constant(fact[:,:k], prepend=False)
    res = sm.OLS(y0, fact_wconst).fit()
##    print 'k =', k
##    print res.params
##    print 'aic:  ', res.aic
##    print 'bic:  ', res.bic
##    print 'llf:  ', res.llf
##    print 'R2    ', res.rsquared
##    print 'R2 adj', res.rsquared_adj
    prederr2 = 0.
    for inidx, outidx in LeaveOneOut(len(y0)):
        resl1o = sm.OLS(y0[inidx], fact_wconst[inidx,:]).fit()
        #print data.endog[outidx], res.model.predict(data.exog[outidx,:]),
        prederr2 += (y0[outidx] - resl1o.predict(fact_wconst[outidx,:]))**2.
    results.append([k, res.aic, res.bic, res.rsquared_adj, prederr2])

results = np.array(results)
print(results)
print('best result for k, by AIC, BIC, R2_adj, L1O')
print(np.r_[(np.argmin(results[:,1:3],0), np.argmax(results[:,3],0),
             np.argmin(results[:,-1],0))])

from statsmodels.iolib.table import (SimpleTable, default_txt_fmt,
                        default_latex_fmt, default_html_fmt)

headers = 'k, AIC, BIC, R2_adj, L1O'.split(', ')
numformat = ['%6d'] + ['%10.3f']*4 #'%10.4f'
txt_fmt1 = dict(data_fmts = numformat)
tabl = SimpleTable(results, headers, None, txt_fmt=txt_fmt1)

print("PCA regression on simulated data,")
print("DGP: 2 factors and 4 explanatory variables")
print(tabl)
print("Notes: k is number of components of PCA,")
print("       constant is added additionally")
print("       k=0 means regression on constant only")
print("       L1O: sum of squared prediction errors for leave-one-out")



