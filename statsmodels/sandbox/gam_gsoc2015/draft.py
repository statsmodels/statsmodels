''' This file contains draft of code. Do not look at it '''
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from smooth_basis import make_poly_basis, make_bsplines_basis
from gam import GamPenalty, LogitGam, GLMGam, MultivariateGamPenalty
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLMResults, GLM
from gam import PenalizedMixin, GLMGam, GLMGAMResults



degree = 4
df = 10

n = 200
x = np.linspace(-1, 1, n)
y = x * x * x - x + np.random.normal(0, .1, n)

basis, der_basis, der2_basis = make_poly_basis(x, degree)
cov_der2 = np.dot(der2_basis.T, der2_basis)


x_new = x[:n/2]
y_new = x_new**3 - x_new
new_basis, _, _ = make_poly_basis(x_new, degree, intercept=True)



alpha = 0.1
plt.figure()
gp = GamPenalty(alpha=alpha, cov_der2=cov_der2, der2=der2_basis)
glm_gam = GLMGam(y, basis, penal=gp)
res_glm_gam = glm_gam.fit(method='nm', max_start_irls=0,
                          disp=1, maxiter=5000, maxfun=5000)



plt.subplot(2, 1, 1)
res_glm_gam.plot_predict(x, basis)
plt.plot(x, y, '.')
plt.ylim(-1, 1)
plt.subplot(2, 1, 2)
res_glm_gam.plot_predict(x_new, new_basis)
#plt.plot(x_new, np.dot(new_basis, res_glm_gam.params))
plt.plot(x_new, y_new, '.')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.legend()
plt.show()
