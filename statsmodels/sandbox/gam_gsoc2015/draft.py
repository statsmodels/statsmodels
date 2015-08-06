import numpy as np
import matplotlib.pyplot as plt
from statsmodels.sandbox.gam_gsoc2015.gam import MultivariateGamPenalty, GLMGam
from statsmodels.sandbox.gam_gsoc2015.smooth_basis import CubicSplines, PolynomialSmoother

n = 500
x = np.random.uniform(-1, 1, n)

y = 10*x**3 - 10*x + np.random.normal(0, 1, n)

y -= y.mean()
cs = CubicSplines(x, 10).fit()

# required only to initialize the gam. they have no influence on the result.
dummy_smoother = PolynomialSmoother(x, [2])
gp = MultivariateGamPenalty(dummy_smoother, alphas=[0])
#

gam = GLMGam(y, cs.xs, penal=gp)

start_params = np.ones(shape=(cs.xs.shape[1],))
weights = np.array([0.5] * n)
gam_res = gam._fit_pirls_version2(y=y, spl_x=cs.xs, spl_s=cs.s, alpha=0, start_params=start_params, weights=weights)

print(cs.xs.shape, gam_res.params.shape)
y_est = np.dot(cs.xs, gam_res.params.T)


plt.plot(x, y, '.')
plt.plot(x, y_est, '.')

plt.show()
#