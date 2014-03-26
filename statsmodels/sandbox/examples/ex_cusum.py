# -*- coding: utf-8 -*-
"""
Created on Fri Apr 02 11:41:25 2010

Author: josef-pktd
"""


import numpy as np
from scipy import stats
from numpy.testing import assert_almost_equal
import statsmodels.api as sm
from statsmodels.sandbox.regression.onewaygls import OneWayLS
from statsmodels.stats.diagnostic import recursive_olsresiduals
from statsmodels.sandbox.stats.diagnostic import _recursive_olsresiduals2 as recursive_olsresiduals2

#examples from ex_onewaygls.py
#choose example
#--------------
example = ['null', 'smalldiff', 'mediumdiff', 'largediff'][1]
example_size = [20, 100][1]
example_groups = ['2', '2-2'][1]
#'2-2': 4 groups,
#       groups 0 and 1 and groups 2 and 3 have identical parameters in DGP

#generate example
#----------------
#np.random.seed(87654589)
nobs = example_size
x1 = 0.1+np.random.randn(nobs)
y1 = 10 + 15*x1 + 2*np.random.randn(nobs)

x1 = sm.add_constant(x1, prepend=False)
#assert_almost_equal(x1, np.vander(x1[:,0],2), 16)
#res1 = sm.OLS(y1, x1).fit()
#print res1.params
#print np.polyfit(x1[:,0], y1, 1)
#assert_almost_equal(res1.params, np.polyfit(x1[:,0], y1, 1), 14)
#print res1.summary(xname=['x1','const1'])

#regression 2
x2 = 0.1+np.random.randn(nobs)
if example == 'null':
    y2 = 10 + 15*x2 + 2*np.random.randn(nobs)  # if H0 is true
elif example == 'smalldiff':
    y2 = 11 + 16*x2 + 2*np.random.randn(nobs)
elif example == 'mediumdiff':
    y2 = 12 + 16*x2 + 2*np.random.randn(nobs)
else:
    y2 = 19 + 17*x2 + 2*np.random.randn(nobs)

x2 = sm.add_constant(x2, prepend=False)

# stack
x = np.concatenate((x1,x2),0)
y = np.concatenate((y1,y2))
if example_groups == '2':
    groupind = (np.arange(2*nobs)>nobs-1).astype(int)
else:
    groupind = np.mod(np.arange(2*nobs),4)
    groupind.sort()
#x = np.column_stack((x,x*groupind[:,None]))

res1 = sm.OLS(y, x).fit()
skip = 8

rresid, rparams, rypred, rresid_standardized, rresid_scaled, rcusum, rcusumci = \
            recursive_olsresiduals(res1, skip)
print(rcusum)
print(rresid_scaled[skip-1:])

assert_almost_equal(rparams[-1], res1.params)

import matplotlib.pyplot as plt
plt.plot(rcusum)
plt.plot(rcusumci[0])
plt.plot(rcusumci[1])
plt.figure()
plt.plot(rresid)
plt.plot(np.abs(rresid))

print('cusum test reject:')
print(((rcusum[1:]>rcusumci[1])|(rcusum[1:]<rcusumci[0])).any())

rresid2, rparams2, rypred2, rresid_standardized2, rresid_scaled2, rcusum2, rcusumci2 = \
            recursive_olsresiduals2(res1, skip)
#assert_almost_equal(rparams[skip+1:], rparams2[skip:-1],13)
assert_almost_equal(rparams[skip:], rparams2[skip:],13)
#np.c_[rparams[skip+1:], rparams2[skip:-1]]
#plt.show()

####################  Example break test
#import statsmodels.sandbox.tools.stattools
from statsmodels.sandbox.stats.diagnostic import breaks_hansen, \
        breaks_cusumolsresid#, breaks_cusum
H, crit95, ft, s = breaks_hansen(res1)
print(H)
print(crit95)

supb, pval, crit = breaks_cusumolsresid(res1.resid)
print(supb, pval, crit)

##check whether this works directly: Ploberger/Kramer framing of standard cusum
##no, it's different, there is another denominator
#print breaks_cusumolsresid(rresid[skip:])
#this function is still completely wrong, cut and paste doesn't apply
#print breaks_cusum(rresid[skip:])



