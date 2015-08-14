
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from statsmodels.sandbox.gam_gsoc2015.gam import MultivariateGamPenalty, GLMGam, UnivariateGamPenalty
from statsmodels.sandbox.gam_gsoc2015.smooth_basis import (UnivariateCubicSplines, PolynomialSmoother,
                                                           UnivariatePolynomialSmoother, UnivariateBSplines,
                                                           UnivariateCubicCyclicSplines, BSplines, CubicSplines)
from patsy import dmatrix
from statsmodels.sandbox.gam_gsoc2015.smooth_basis import _get_all_sorted_knots


np.random.seed(0)
# Multivariate GAM PIRLS
n = 500
x = np.random.uniform(-1, 1, n)

y = 10*x**3 - 10*x + np.random.normal(0, 1, n)

y -= y.mean()
cs = CubicSplines(x, [10])

# required only to initialize the gam. they have no influence on the result.
for i, alpha in enumerate([0, 1, 5, 10]):

    gam = GLMGam(y, cs, alpha=alpha)
    gam_res = gam.fit()
    y_est = gam_res.predict(cs.basis_)

    plt.subplot(2, 2, i+1)
    plt.plot(x, y, '.')
    plt.plot(x, y_est, '.')
    plt.title('alpha=' + str(alpha))

plt.tight_layout()
plt.show()