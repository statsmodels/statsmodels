# -*- coding: utf-8 -*-
"""examples for sandwich estimators of covariance

Author: Josef Perktold

"""

import numpy as np
from numpy.testing import assert_almost_equal

import scikits.statsmodels.api as sm

import scikits.statsmodels.sandbox.panel.sandwich_covariance as sw
import scikits.statsmodels.sandbox.panel.sandwich_covariance_generic as swg


nobs = 100
kvars = 4 #including constant
x = np.random.randn(nobs, kvars-1)
exog = sm.add_constant(x, prepend=True)
params_true = np.ones(kvars)
y_true = np.dot(exog, params_true)
sigma = 0.1 + np.exp(exog[:,-1])
endog = y_true + sigma * np.random.randn(nobs)

self = sm.OLS(endog, exog).fit()

print self.HC3_se
print sw.se_cov(sw.cov_HC3(self))
#test standalone refactoring
assert_almost_equal(sw.se_cov(sw.cov_HC0(self)), self.HC0_se, 15)
assert_almost_equal(sw.se_cov(sw.cov_HC1(self)), self.HC1_se, 15)
assert_almost_equal(sw.se_cov(sw.cov_HC2(self)), self.HC2_se, 15)
assert_almost_equal(sw.se_cov(sw.cov_HC3(self)), self.HC3_se, 15)
print self.HC0_se
print sw.cov_hac_simple(self, nlags=0)[1]
#test White as HAC with nlags=0, same as nlags=1 ?
assert_almost_equal(sw.cov_hac_simple(self, nlags=0)[1], self.HC0_se, 15)
print sw.cov_white_simple(self)[1]
#test White
assert_almost_equal(sw.cov_white_simple(self)[1], self.HC0_se, 15)


groups = np.repeat(np.arange(5), 20)

idx = np.nonzero(np.diff(groups))[0].tolist()
groupidx = zip([0]+idx, idx+[len(groups)])
ngroups = len(groupidx)

print sw.cov_crosssection2(self, groups)[1]
#two strange looking corner cases BUG?
print sw.cov_crosssection2(self, np.ones(len(endog), int))[1]
print sw.cov_crosssection(self, np.arange(len(endog)))[1]