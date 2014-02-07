# -*- coding: utf-8 -*-
"""

Created on Fri Jun 08 16:13:47 2012

Author: Josef Perktold
"""

import numpy as np
from scipy import stats
from statsmodels.sandbox.distributions.gof_one_sample import (
                      GOFNormal, GOFExpon)

rvs = stats.expon.rvs(loc=0, scale=5, size=200)
ge = GOFExpon(rvs)
print ge.get_test()  #default is 'a2'
print ge.get_test('w2')
print ge.get_test('u2')
rvsn = np.random.randn(50)
print GOFExpon(rvsn**2).get_test()

#rvsm = (np.random.randn(50, 2) * [1, 2]).sum(1)
rvsm = (np.random.randn(50, 2) * [1, 2]).ravel()
gn = GOFNormal(rvsm)
print gn.get_test()
from statsmodels.stats import diagnostic as dia
print dia.normal_ad(rvsm)
print gn.get_test('d')
print dia.lillifors(rvsm)


