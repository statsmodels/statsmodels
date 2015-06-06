import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from smooth_basis import make_poly_basis, make_bsplines_basis
from gam import PenalizedMixin, GamPenalty
from statsmodels.api import GLM
from statsmodels.discrete.discrete_model import Logit


class LogitGam(PenalizedMixin, Logit):
    pass


class GLMGam(PenalizedMixin, GLM):
    pass


n = 100

# make the data
x = np.linspace(-10, 10, n)
y = 1/(1 + np.exp(-x*x))
mu = y.mean()
y[y > mu] = 1
y[y < mu] = 0

## make the splines basis ##
df = 10
degree = 5
basis, der_basis, der2_basis = make_bsplines_basis(x, df, degree)
cov_der2 = np.dot(der2_basis.T, der2_basis)


## train the gam logit model ##
alpha = 10
params0 = np.random.normal(0, 1, df)
gp = GamPenalty(wts=1, alpha=alpha, cov_der2=cov_der2, der2=der2_basis)
g = LogitGam(y, basis, penal=gp)
res_g = g.fit()

plt.plot(x, np.dot(basis, res_g.params))
plt.show()


######## GAM GLM  ##################

# y is continuous
y = x * x + np.random.normal(0, 1, n)

# despite the large alpha we don't see a penalization
alpha = 1000000000000000000000

# train the model
gp = GamPenalty(alpha=alpha, cov_der2=cov_der2, der2=der2_basis)
glm_gam = GLMGam(y, basis, penal = gp)
res_glm_gam = glm_gam.fit()

plt.plot(x, np.dot(basis, res_glm_gam.params))
plt.show()
