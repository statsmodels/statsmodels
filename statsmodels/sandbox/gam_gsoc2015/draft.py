import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from statsmodels.sandbox.gam_gsoc2015.gam import MultivariateGamPenalty, GLMGam, UnivariateGamPenalty
from statsmodels.sandbox.gam_gsoc2015.smooth_basis import (CubicSplines, PolynomialSmoother,
                                                           UnivariatePolynomialSmoother, UnivariateBSplines, CubicCyclicSplines)
from patsy import dmatrix
from statsmodels.sandbox.gam_gsoc2015.smooth_basis import _get_all_sorted_knots
n = 100
x1 = np.random.uniform(-1, 1, n)
x2 = np.random.normal(0, 1, n)
df = 3

y = x1 * x2 + x1 + np.random.normal(0, .1, n)
dm = dmatrix("te(cc(x1, df), cc(x2, df)) - 1", {"x1": x1.ravel(), "x2": x2.ravel(), "df": df})

cc1 = CubicCyclicSplines(x1, df=df)
cc2 = CubicCyclicSplines(x2, df=df)

s = sp.linalg.block_diag(*[cc1.s, cc2.s])
print(s.shape, dm.shape)

gam = GLMGam(y, x1)
alphas = [1, 1]
gam_res = gam._fit_pirls(y, dm, s, alpha=alphas)

plt.plot(x1, y, '.')
plt.show()

print(y.shape)
print(s[0].shape, s[1].shape)