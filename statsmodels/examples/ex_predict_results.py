# -*- coding: utf-8 -*-
"""
Created on Sat Dec 20 12:01:13 2014

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from statsmodels.regression.linear_model import OLS, WLS

from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.regression._prediction import get_prediction


# from example wls.py

nsample = 50
x = np.linspace(0, 20, nsample)
X = np.column_stack((x, (x - 5)**2))
from statsmodels.tools.tools import add_constant
X = add_constant(X)
beta = [5., 0.5, -0.01]
sig = 0.5
w = np.ones(nsample)
w[nsample * 6/10:] = 3
y_true = np.dot(X, beta)
e = np.random.normal(size=nsample)
y = y_true + sig * w * e
X = X[:,[0,1]]


# ### WLS knowing the true variance ratio of heteroscedasticity

mod_wls = WLS(y, X, weights=1./w)
res_wls = mod_wls.fit()



prstd, iv_l, iv_u = wls_prediction_std(res_wls)
pred_res = get_prediction(res_wls)
ci = pred_res.conf_int(obs=True)

from numpy.testing import assert_allclose
assert_allclose(pred_res.se_obs, prstd, rtol=1e-13)
assert_allclose(ci, np.column_stack((iv_l, iv_u)), rtol=1e-13)

print pred_res.summary_frame().head()

pred_res2 = res_wls.get_prediction()
ci2 = pred_res2.conf_int(obs=True)

from numpy.testing import assert_allclose
assert_allclose(pred_res2.se_obs, prstd, rtol=1e-13)
assert_allclose(ci2, np.column_stack((iv_l, iv_u)), rtol=1e-13)

print pred_res2.summary_frame().head()
