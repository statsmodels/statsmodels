# -*- coding: utf-8 -*-
"""Cluster Robust Standard Errors with Two Clusters

Created on Sat Dec 17 08:39:16 2011

Author: Josef Perktold
"""
from statsmodels.compat.python import urlretrieve
import numpy as np
from numpy.testing import assert_almost_equal

import statsmodels.api as sm

import statsmodels.stats.sandwich_covariance as sw

#requires Petersen's test_data
#http://www.kellogg.northwestern.edu/faculty/petersen/htm/papers/se/test_data.txt
try:
    pet = np.genfromtxt("test_data.txt")
    print('using local file')
except IOError:
    urlretrieve('http://www.kellogg.northwestern.edu/faculty/petersen/htm/papers/se/test_data.txt',
                       'test_data.txt')
    print('downloading file')
    pet = np.genfromtxt("test_data.txt")


endog = pet[:,-1]
group = pet[:,0].astype(int)
time = pet[:,1].astype(int)
exog = sm.add_constant(pet[:,2])
res = sm.OLS(endog, exog).fit()

cov01, covg, covt = sw.cov_cluster_2groups(res, group, group2=time)

#Reference number from Petersen
#http://www.kellogg.northwestern.edu/faculty/petersen/htm/papers/se/test_data.htm

bse_petw = [0.0284, 0.0284]
bse_pet0 = [0.0670, 0.0506]
bse_pet1 = [0.0234, 0.0334]  #year
bse_pet01 = [0.0651, 0.0536]  #firm and year

bse_0 = sw.se_cov(covg)
bse_1 = sw.se_cov(covt)
bse_01 = sw.se_cov(cov01)

print('OLS            ', res.bse)
print('het HC0        ', res.HC0_se, bse_petw - res.HC0_se)
print('het firm       ', bse_0, bse_0 - bse_pet0)
print('het year       ', bse_1, bse_1 - bse_pet1)
print('het firm & year', bse_01, bse_01 - bse_pet01)

print('relative difference standard error het firm & year to OLS')
print('               ', bse_01 / res.bse)

#From the last line we see that the cluster and year robust standard errors
#are approximately twice those of OLS

assert_almost_equal(bse_petw, res.HC0_se, decimal=4)
assert_almost_equal(bse_0, bse_pet0, decimal=4)
assert_almost_equal(bse_1, bse_pet1, decimal=4)
assert_almost_equal(bse_01, bse_pet01, decimal=4)
