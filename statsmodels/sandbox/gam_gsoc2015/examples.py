__author__ = 'Luca Puggini: <lucapuggio@gmail.com>'

import numpy as np
import matplotlib.pyplot as plt

sigmoid = np.vectorize(lambda x: 1.0/(1.0 + np.exp(-x)))

from statsmodels.sandbox.gam_gsoc2015.smooth_basis import BSplines
from statsmodels.sandbox.gam_gsoc2015.gam import GLMGam

n = 1000
x = np.random.uniform(0, 1, (n, 2))
x = x - x.mean()
y = x[:, 0] * x[:, 0] + np.random.normal(0, .01, n)
y -= y.mean()

bsplines = BSplines(x, degree=[3] * 2, df=[10]*2)

alpha = 0.001
glm_gam = GLMGam(y, bsplines, alpha=alpha)
res_glm_gam = glm_gam.fit(method='bfgs', max_start_irls=0,
                            disp=0, maxiter=5000, maxfun=5000)


y_est = res_glm_gam.predict(bsplines.basis_)
y_partial_est, se = res_glm_gam.partial_values(bsplines, 0)

plt.plot(y_est, y_partial_est, '.')
plt.show()