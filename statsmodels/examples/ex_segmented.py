# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 21:32:33 2015

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
import matplotlib.pyplot as plt

from statsmodels.regression.linear_model import OLS
from statsmodels.base._segmented import Segmented

#########  example
nobs = 500
bp_true = -0.5
sig_e = 0.1

np.random.seed(9999)

x01 = np.sort(np.random.uniform(-1.99, 0.9, size=nobs))
#x0z = np.sign(x01 % 2 - 1).cumsum()
x0z = np.abs(np.exp(x01 +0.5) % 2 - 1)
beta_diff = 0.1
beta = [1, 0, beta_diff]
#exog0 = np.column_stack((np.ones(nobs), x0, x0s2))
#exog0 = np.array((np.ones(nobs), x0, x0s2)).T
y_true = x0z #exog0.dot(beta)
y = y_true + sig_e * np.random.randn(nobs)

#res_oracle = OLS(y, exog0).fit()

mod_base1 = OLS(y, np.column_stack((np.ones(nobs), x01)))
# use only part of sample
sl = slice(0, nobs//2, None)
sl = slice(None, None, None)
mod_base2 = OLS(y[sl], np.column_stack((np.ones(nobs), x01, x01))[sl,:])

#res_fitted2 = segmented(mod_base2, 1, k_segments=1)

q = np.percentile(x01, [25, 60, 85])

mod_base2 = OLS(mod_base2.endog, np.column_stack((np.ones(nobs), x01, np.maximum(x01 - q[0], 0), np.maximum(x01 - q[1], 0),
                                                  np.maximum(x01 - q[2], 0)  )))

res_base = mod_base2.fit()

seg = Segmented(mod_base2, x01, [2, 3, 4])
q = np.percentile(x01, [10, 25, 60, 85, 90])
seg._fit_all(q, maxiter=1)
res_fitted2 = seg.get_results()

seg = Segmented(mod_base2, x01, [2, 3, 4])
q = np.percentile(x01, [10, 25, 60, 85, 90])
seg._fit_all(q, maxiter=10)
res_fitted_it2 = seg.get_results()

seg_p1, r = seg.add_knot(maxiter=3)
res_fitted_p1 = seg_p1.get_results()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(x01, y, '.', alpha=0.5)
ax.plot(x01, y_true, '-', lw=2)
#ax.plot(x01, x0z, '-', lw=2)
ax.plot(x01[sl], y_true, '-', lw=2, label='true', color='b')
ax.plot(x01[sl], res_base.fittedvalues, '--', lw=2, label='start', alpha=0.5)
ax.plot(x01[sl], res_fitted2.fittedvalues, '-', lw=2, label='best-1')
ax.plot(x01[sl], res_fitted_it2.fittedvalues, '-', lw=2, label='best-iter')
ax.plot(x01[sl], res_fitted_p1.fittedvalues, '-', lw=2, color='r', label='best-add')
ax.vlines(res_fitted_it2.knot_locations, *ax.get_ylim())
ax.legend(loc='lower left')
ax.set_title('Optimal Knot Selection')

mod_base0 = OLS(y, np.ones(nobs))
segn = Segmented.from_model(mod_base0, x01, k_knots=3, degree=1)
