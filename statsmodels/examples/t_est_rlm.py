# -*- coding: utf-8 -*-
"""
Example from robust test_rlm, fails on Mac

Created on Sun Mar 27 14:36:40 2011

"""

from __future__ import print_function
import numpy as np
import statsmodels.api as sm
RLM = sm.RLM

DECIMAL_4 = 4
DECIMAL_3 = 3
DECIMAL_2 = 2
DECIMAL_1 = 1

from statsmodels.datasets.stackloss import load
data = load()   # class attributes for subclasses
data.exog = sm.add_constant(data.exog, prepend=False)

decimal_standarderrors = DECIMAL_1
decimal_scale = DECIMAL_3

results = RLM(data.endog, data.exog,\
            M=sm.robust.norms.HuberT()).fit()   # default M
h2 = RLM(data.endog, data.exog,\
            M=sm.robust.norms.HuberT()).fit(cov="H2").bcov_scaled
h3 = RLM(data.endog, data.exog,\
            M=sm.robust.norms.HuberT()).fit(cov="H3").bcov_scaled


from statsmodels.robust.tests.results.results_rlm import Huber
res2 = Huber()

print("res2.h1")
print(res2.h1)
print("results.bcov_scaled")
print(results.bcov_scaled)
print("res2.h1 - results.bcov_scaled")
print(res2.h1 - results.bcov_scaled)

from numpy.testing import assert_almost_equal
assert_almost_equal(res2.h1, results.bcov_scaled, 4)
