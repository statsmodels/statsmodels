import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from statsmodels.sandbox.gam_gsoc2015.gam import MultivariateGamPenalty, GLMGam, UnivariateGamPenalty, OLD_GLMGam
from statsmodels.sandbox.gam_gsoc2015.smooth_basis import (UnivariateCubicSplines, PolynomialSmoother,
                                                           UnivariatePolynomialSmoother, UnivariateBSplines,
                                                           UnivariateCubicCyclicSplines, CubicRegressionSplines, BSplines)
from patsy import dmatrix
from statsmodels.sandbox.gam_gsoc2015.smooth_basis import _get_all_sorted_knots

np.random.seed(0)
n = 300
x1 = np.linspace(-5, 5, n)

y = x1*x1*x1 - x1*x1 + np.random.normal(0, 10, n)
y -= y.mean()



gp = MultivariateGamPenalty(multivariate_smoother=bs, alphas=alphas)
gam = GLMGam