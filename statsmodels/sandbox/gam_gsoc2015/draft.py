''' This file contains draft of code. Do not look at it '''
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import smooth_basis
from smooth_basis import make_poly_basis, make_bsplines_basis
from gam import GamPenalty, LogitGam, GLMGam, MultivariateGamPenalty
#from statsmodels.genmod.generalized_linear_model import GLMResults, GLM
#from gam import PenalizedMixin, GLMGam, GLMGAMResults



degree = 4
df = 10

n = 200
x = np.linspace(-1, 1, n)
y = x * x * x - x + np.random.normal(0, .1, n)
#y = np.random.normal(0, 1, n)

basis, der_basis, der2_basis = make_poly_basis(x, degree)
cov_der2 = np.dot(der2_basis.T, der2_basis)



alpha = 0.1
gp = GamPenalty(alpha=alpha, cov_der2=cov_der2, der2=der2_basis)
glm_gam = GLMGam(y, basis, penal=gp)
res_glm_gam = glm_gam.fit(method='nm', max_start_irls=0,
                          disp=1, maxiter=5000, maxfun=5000)


print('t=', res_glm_gam.significance_test(basis))