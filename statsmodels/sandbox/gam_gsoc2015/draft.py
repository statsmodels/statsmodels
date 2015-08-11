import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from statsmodels.sandbox.gam_gsoc2015.gam import MultivariateGamPenalty, GLMGam, UnivariateGamPenalty
from statsmodels.sandbox.gam_gsoc2015.smooth_basis import (CubicSplines, PolynomialSmoother,
                                                           UnivariatePolynomialSmoother, UnivariateBSplines,
                                                           CubicCyclicSplines, CubicRegressionSplines)
from patsy import dmatrix
from statsmodels.sandbox.gam_gsoc2015.smooth_basis import _get_all_sorted_knots

np.random.seed(0)
n = 300
x1 = np.linspace(-5, 5, n)

y = x1*x1*x1 - x1*x1 + np.random.normal(0, 10, n)
y -= y.mean()


df = 30
cr = CubicRegressionSplines(x1, df=df)
cc = CubicCyclicSplines(x1, df=df)
cs = CubicSplines(x1, df=df)

y_est_cs_list = []
for i, alpha in enumerate([0, 5, 10, 30]):
    gam = GLMGam(y, x1)
    gam_res_cr = gam._fit_pirls(y, cr.basis_, cr.s, alpha=alpha)
    gam_res_cc = gam._fit_pirls(y, cc.basis_, cc.s, alpha=alpha)
    gam_res_cs = gam._fit_pirls(y, cs.basis_, cs.s, alpha=alpha)

    y_est_cc = np.dot(cc.basis_, gam_res_cc.params)
    y_est_cr = np.dot(cr.basis_, gam_res_cr.params)
    y_est_cs = np.dot(cs.basis_, gam_res_cs.params)

    y_est_cr -= y_est_cr.mean()
    y_est_cs -= y_est_cs.mean()
    y_est_cc -= y_est_cc.mean()

    y_est_cs_list.append(y_est_cs)

    plt.subplot(2, 2, i+1)
    plt.title('alpha=' + str(alpha))
    plt.plot(x1, y, '.', c='green')
    plt.plot(x1, y_est_cc, c='blue', label='cc')
    plt.plot(x1, y_est_cr, label='cr')
    #plt.plot(x1, y_est_cs, label='cs')
    plt.legend(loc='best')
plt.show()

