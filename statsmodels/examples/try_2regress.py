# -*- coding: utf-8 -*-
"""F test for null hypothesis that coefficients in two regressions are the same

see discussion in http://mail.scipy.org/pipermail/scipy-user/2010-March/024851.html

Created on Thu Mar 25 22:56:45 2010
Author: josef-pktd
"""

from __future__ import print_function
import numpy as np
from numpy.testing import assert_almost_equal
import statsmodels.api as sm

np.random.seed(87654589)

nobs = 10 #100
x1 = np.random.randn(nobs)
y1 = 10 + 15*x1 + 2*np.random.randn(nobs)

x1 = sm.add_constant(x1, prepend=False)
assert_almost_equal(x1, np.vander(x1[:,0],2), 16)
res1 = sm.OLS(y1, x1).fit()
print(res1.params)
print(np.polyfit(x1[:,0], y1, 1))
assert_almost_equal(res1.params, np.polyfit(x1[:,0], y1, 1), 14)
print(res1.summary(xname=['x1','const1']))

#regression 2
x2 = np.random.randn(nobs)
y2 = 19 + 17*x2 + 2*np.random.randn(nobs)
#y2 = 10 + 15*x2 + 2*np.random.randn(nobs)  # if H0 is true

x2 = sm.add_constant(x2, prepend=False)
assert_almost_equal(x2, np.vander(x2[:,0],2), 16)

res2 = sm.OLS(y2, x2).fit()
print(res2.params)
print(np.polyfit(x2[:,0], y2, 1))
assert_almost_equal(res2.params, np.polyfit(x2[:,0], y2, 1), 14)
print(res2.summary(xname=['x2','const2']))


# joint regression

x = np.concatenate((x1,x2),0)
y = np.concatenate((y1,y2))
dummy = np.arange(2*nobs)>nobs-1
x = np.column_stack((x,x*dummy[:,None]))

res = sm.OLS(y, x).fit()
print(res.summary(xname=['x','const','x2','const2']))

print('\nF test for equal coefficients in 2 regression equations')
#effect of dummy times second regression is zero
#is equivalent to 3rd and 4th coefficient are both zero
print(res.f_test([[0,0,1,0],[0,0,0,1]]))

print('\nchecking coefficients individual versus joint')
print(res1.params, res2.params)
print(res.params[:2], res.params[:2]+res.params[2:])
assert_almost_equal(res1.params, res.params[:2], 13)
assert_almost_equal(res2.params, res.params[:2]+res.params[2:], 13)
