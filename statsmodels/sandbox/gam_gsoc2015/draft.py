import numpy as np
import matplotlib.pyplot as plt
from statsmodels.sandbox.gam_gsoc2015.gam import MultivariateGamPenalty, GLMGam, UnivariateGamPenalty
from statsmodels.sandbox.gam_gsoc2015.smooth_basis import (CubicSplines, PolynomialSmoother,
                                                           UnivariatePolynomialSmoother, UnivariateBSplines)

n = 500
x = np.random.uniform(-1, 1, n)

y = 10*x**3 - 10*x + np.random.normal(0, 1, n)

y -= y.mean()

# required only to initialize the gam. they have no influence on the result.
poly = UnivariateBSplines(x, df=10, degree=4)
gp = UnivariateGamPenalty(poly, alpha=0)
gam = GLMGam(y, poly.basis_, penal=gp)

for i, alpha in enumerate([0, .1, .2, .4]):
    gam_res = gam._fit_pirls_version2(y=y, spl_x=poly.basis_, spl_s=poly.cov_der2_, alpha=alpha)

    y_est = np.dot(poly.basis_, gam_res.params.T)

    plt.subplot(2, 2, i+1)
    plt.plot(x, y, '.')
    plt.plot(x, y_est, '.')

plt.show()


