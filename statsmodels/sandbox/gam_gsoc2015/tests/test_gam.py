from smooth_basis import make_poly_basis, make_bsplines_basis
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

def hessian(params):
    hess = np.array([[576/5, 0, 32,  0, 0],
                     [0, 48,  0,  0, 0],
                     [32,  0,  0,  0, 0],
                     [0,  0,  0,  0, 0],
                     [0,  0,  0,  0, 0]])
    return hess / 2


def cost_function(params, basis, y, alpha):

    # this should be the MSE or log likelihood value
    lin_pred = np.dot(basis, params)
    gaussian = Gaussian()
    expval = gaussian.link.inverse(lin_pred)
    loglike = gaussian.loglike(expval, y)

    # this is the vale of the GAM penalty. For the example polynomial
    itg = integral(params)

    # return the cost function of the GAM for the given polynomial
    return loglike +  alpha * itg, loglike, itg


def test_gam_penalty():
    """
    test the func method of the gam penalty
    :return:
    """
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


def test_gam_hessian():
    """
    test the deriv2 method of the gam penalty
    :return:
    """
    x, y, basis, cov_der2, der2 = sample_data()
    alpha = 1
    gp = GamPenalty(alpha=alpha, cov_der2=cov_der2, der2=der2)

    for i in range(10):
        params = np.random.randint(-2, 2, 5)
        gam_der2 = gp.deriv2(params)
        hess = hessian(params)
        hess = np.flipud(hess)
        hess = np.fliplr(hess)
        #print(hess - gam_der2)
        assert norm(hess - gam_der2)/25 < 1, 'error in the hessian of the GAM. Err=' + str(norm(hess - gam_der2)/25)
    return


def test_approximation():
    x, y, basis, cov_der2, der2 = sample_data()
    alpha = 0
    for i in range(10):
        params = np.random.randint(-2, 2, 5)
        cost, err, itg = cost_function(params, basis, y, alpha)
        gp = GamPenalty(alpha=alpha, cov_der2=cov_der2, der2=der2)
        gam_itg = gp.func(params)
        glm_gam = GLMGam(y, basis, penal = gp)
        res_glm_gam = glm_gam.fit(maxiter=1) # TODO: can this fit be removed? It is useless. We just need the log likelihood
        gam_loglike = glm_gam.loglike(params)
        assert norm(gam_loglike - err) < 1.e-10, 'erron in the MSE part of the cost function'
    return


def test_gam_optimization():

    x, y, _, _, _ = sample_data()
    df = 10
    degree = 5
    basis, der_basis, der2 = make_bsplines_basis(x, df=df, degree=degree)

    print(basis.mean(axis=0), y.mean())

    cov_der2 = np.dot(der2.T, der2)

    alpha = 0.0819
    gp = GamPenalty(alpha=alpha, cov_der2=cov_der2, der2=der2)

    glm_gam = GLMGam(y, basis, penal=gp)
    res_glm_gam = glm_gam.fit(maxiter=10000)

    x_new = np.linspace(-1, 1, 100)
    y_new = 2 * x_new**3 - x_new

    new_basis, _, _ = make_bsplines_basis(x_new, df=df, degree=degree)

    plt.subplot(2, 1, 1)
    plt.plot(x, y, '.')
    plt.plot(x, np.dot(basis, res_glm_gam.params))
    plt.subplot(2, 1, 2)
    plt.plot(x_new, y_new, '.')
    plt.plot(x_new, np.dot(new_basis, res_glm_gam.params))

    plt.show()
    return

# these tests are fine.
test_gam_penalty()
test_gam_gradient()
test_approximation()
test_gam_hessian()

#test_gam_optimization()
