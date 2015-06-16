from smooth_basis import make_poly_basis
from gam import GamPenalty, LogitGam, GLMGam
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import minimize
import pandas as pd
from statsmodels.genmod.families.family import Gaussian
from numpy.linalg import norm

sigmoid = np.vectorize(lambda x: 1.0/(1.0 + np.exp(-x)))


def sample_data():
    """
    A polynomial of degree 4
    poly -> ax^4 + bx^3 + cx^2 + dx + e
    second der -> 12ax^2 + 6bx + 2c
    integral from -1 to 1 of second der^2 -> (288 a^2)/5 + 32 a c + 8 (3 b^2 + c^2)
    the gradient of the integral is -> [576*a/5 + 32 * c, 48*b, 32*a + 16*c, 0, 0]
    :return:
    """
    n = 10000
    x = np.linspace(-1, 1, n)
    y = 2 * x**3 - x

    degree = 4
    basis, der_basis, der2 = make_poly_basis(x, degree)
    cov_der2 = np.dot(der2.T, der2)

    return x, y, basis, cov_der2, der2

def integral(params):
    e, d, c, b, a = params
    itg = (288 * a**2)/5 + (32 * a * c) + 8 * (3 * b**2 + c**2)
    itg /= 2
    return itg

def grad(params):
    e, d, c, b, a = params
    grd = np.array([576*a/5 + 32 * c, 48*b, 32*a + 16*c, 0, 0])
    grd = grd[::-1]
    return grd / 2

def cost_function(params, basis, y, alpha):

    # this should be the MSE or log likelihood value
    lin_pred = np.dot(basis, params)
    gaussian = Gaussian()
    expval = gaussian.link.inverse(lin_pred)
    loglike = gaussian.loglike(expval, y)

    # this is the vale of the GAM penalty. For the example polynomial
    itg = integral(params)

    # return the cost function of the GAM for the given polynomial
    return - loglike + alpha * itg, -loglike, itg

def test_gam_penalty():

    x, y, basis, cov_der2, der2 = sample_data()

    alpha = 1
    gp = GamPenalty(alpha=alpha, cov_der2=cov_der2, der2=der2)

    for i in range(10):
        params = np.random.randint(-2, 2, 5)
        gp_score = gp.func(params)
        itg = integral(params)
        assert norm(itg - gp_score) < 1, print(gp_score, itg, params)

def test_gam_gradient():
    """
    test the gam gradient for the example polynomial
    :return:
    """
    x, y, basis, cov_der2, der2 = sample_data()

    alpha = 1
    gp = GamPenalty(alpha=alpha, cov_der2=cov_der2, der2=der2)

    for i in range(10):
        params = np.random.randint(-2, 2, 5)
        gam_grad = gp.grad(params)
        grd = grad(params)
        err = norm(gam_grad - grd) / 5
        assert err < 1, 'the gradients are not matching'

    return

def test_gam_optimization():
    x, y, basis, cov_der2, der2 = sample_data()

    alpha = 1
    params = np.random.randint(-2, 2, 5)
    cost, err, itg = cost_function(params, basis, y, alpha)

    gp = GamPenalty(alpha=alpha, cov_der2=cov_der2, der2=der2)
    gam_itg = gp.func(params)
    glm_gam = GLMGam(y, basis, penal = gp)
    res_glm_gam = glm_gam.fit()
    gam_loglike = glm_gam.loglike(params)

    print('the values obtained by cost func=', cost, err, itg)
    print('the values from gam=', gam_loglike, gam_itg )

    return


test_gam_penalty()
test_gam_gradient()
test_gam_optimization()
