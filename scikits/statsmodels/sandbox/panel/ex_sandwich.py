# -*- coding: utf-8 -*-
"""examples for sandwich estimators of covariance

Author: Josef Perktold

"""

import numpy as np

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

groups = np.repeat(np.arange(5), 20)

idx = np.nonzero(np.diff(groups))[0].tolist()
groupidx = zip([0]+idx, idx+[len(groups)])
ngroups = len(groupidx)
