# -*- coding: utf-8 -*-
"""

Created on Sat Dec 17 08:39:16 2011

Author: Josef Perktold
"""
import numpy as np
from numpy.testing import assert_almost_equal

import statsmodels.api as sm

import statsmodels.sandbox.panel.sandwich_covariance as sw
import statsmodels.sandbox.panel.sandwich_covariance_generic as swg

#requires Petersen's test_data
#http://www.kellogg.northwestern.edu/faculty/petersen/htm/papers/se/test_data.txt
pet = np.genfromtxt("test_data.txt")
endog = pet[:,-1]
group = pet[:,0].astype(int)
time = pet[:,1].astype(int)
exog = sm.add_constant(pet[:,2], prepend=True)
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
print res.HC0_se, bse_petw - res.HC0_se
print bse_0, bse_0 - bse_pet0
print bse_1, bse_1 - bse_pet1
print bse_01, bse_01 - bse_pet01
assert_almost_equal(bse_petw, res.HC0_se, decimal=4)
assert_almost_equal(bse_0, bse_pet0, decimal=4)
assert_almost_equal(bse_1, bse_pet1, decimal=4)
assert_almost_equal(bse_01, bse_pet01, decimal=4)
