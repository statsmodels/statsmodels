
import os

from statsmodels.sandbox.gam_gsoc2015.smooth_basis import make_poly_basis, make_bsplines_basis
from statsmodels.sandbox.gam_gsoc2015.gam import GamPenalty, GLMGam, MultivariateGamPenalty, LogitGam
import numpy as np
import pandas as pd
from statsmodels.genmod.families.family import Gaussian
from numpy.linalg import norm
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from numpy.testing import assert_allclose

sigmoid = np.vectorize(lambda x: 1.0/ (1.0 + np.exp(-x)))


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
    y = 2 * x ** 3 - x

    degree = 4
    basis, der_basis, der2 = make_poly_basis(x, degree)
    cov_der2 = np.dot(der2.T, der2)

    return x, y, basis, cov_der2, der2


def integral(params):
    e, d, c, b, a = params
    itg = (288 * a ** 2) / 5 + (32 * a * c) + 8 * (3 * b ** 2 + c ** 2)
    itg /= 2
    return itg


def grad(params):
    e, d, c, b, a = params
    grd = np.array([576 * a / 5 + 32 * c, 48 * b, 32 * a + 16 * c, 0, 0])
    grd = grd[::-1]
    return grd / 2


def hessian(params):
    hess = np.array([[576 / 5, 0, 32, 0, 0],
                     [0, 48, 0, 0, 0],
                     [32, 0, 16, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]])
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
    return loglike + alpha * itg, loglike, itg


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
        assert_allclose(gp_score, itg, atol=1.e-1)


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
        assert_allclose(gam_grad, grd, rtol=1.e-2, atol=1.e-10)

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
        assert_allclose(gam_der2, hess, atol=1.e-13, rtol=1.e-3)
    return


def test_approximation():
    x, y, basis, cov_der2, der2 = sample_data()
    alpha = 0
    for i in range(10):
        params = np.random.randint(-2, 2, 5)
        cost, err, itg = cost_function(params, basis, y, alpha)
        gp = GamPenalty(alpha=alpha, cov_der2=cov_der2, der2=der2)
        gam_itg = gp.func(params)
        glm_gam = GLMGam(y, basis, penal=gp)
        res_glm_gam = glm_gam.fit(
            maxiter=1)  # TODO: can this fit be removed? It is useless. We just need the log likelihood
        gam_loglike = glm_gam.loglike(params)
        assert_allclose(gam_loglike, err)
    return


def test_gam_glm():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cur_dir, "results", "prediction_from_mgcv.csv")
    data_from_r = pd.read_csv(file_path)
    # dataset used to train the R model
    x = data_from_r.x
    y = data_from_r.y

    df = 10
    degree = 5
    basis, der_basis, der2 = make_bsplines_basis(x, df=df, degree=degree)
    cov_der2 = np.dot(der2.T, der2)

    # y_mgcv is obtained from R with the following code
    # g = gam(y~s(x, k = 10, bs = "cr"), data = data, scale = 80)
    y_mgcv = data_from_r.y_est

    alpha = 0.045
    gp = GamPenalty(alpha=alpha, cov_der2=cov_der2, der2=der2)
    glm_gam = GLMGam(y, basis, penal=gp)
    res_glm_gam = glm_gam.fit(maxiter=10000)
    y_gam = np.dot(basis, res_glm_gam.params)

    plt.plot(x, y_gam, label='gam')
    plt.plot(x, y_mgcv, label='mgcv')
    plt.plot(x, y, '.', label='y')
    plt.legend()
    plt.show()

    assert_allclose(y_gam, y_mgcv, atol=1.e-1)
    return


def test_gam_discrete():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cur_dir, "results", "prediction_from_mgcv.csv")
    data_from_r = pd.read_csv(file_path)
    # dataset used to train the R model
    x = data_from_r.x
    y = data_from_r.ybin

    df = 10
    degree = 5
    basis, der_basis, der2 = make_bsplines_basis(x, df=df, degree=degree)
    cov_der2 = np.dot(der2.T, der2)

    # y_mgcv is obtained from R with the following code
    # g = gam(y~s(x, k = 10, bs = "cr"), data = data, scale = 80)
    y_mgcv = data_from_r.ybin_est

    alpha = 0.00002
    gp = GamPenalty(alpha=alpha, cov_der2=cov_der2, der2=der2)
    lg_gam = LogitGam(y, basis, penal=gp)
    res_lg_gam = lg_gam.fit(maxiter=10000)
    y_gam = np.dot(basis, res_lg_gam.params)
    y_gam = sigmoid(y_gam)
    y_mgcv = sigmoid(y_mgcv)

    # plt.plot(x, y_gam, label='gam')
    # plt.plot(x, y_mgcv, label='mgcv')
    # plt.plot(x, y, '.', label='y')
    # plt.ylim(-0.4, 1.4)
    # plt.legend()
    # plt.show()

    assert_allclose(y_gam, y_mgcv, rtol=1.e-10, atol=1.e-1)

    return


def sample_multivariate_data():
    n = 1000
    x1 = np.linspace(-1, 1, n)
    x2 = np.linspace(-10, 10, n)
    y = x1 * x1 * x1 + x2 + np.random.normal(0, 0.01, n)
    degree1 = 4
    degree2 = 3
    basis1, der_basis1, der2_basis1 = make_poly_basis(x1, degree1, intercept=False)
    basis2, der_basis2, der2_basis2 = make_poly_basis(x2, degree2, intercept=False)
    cov_der2_1 = np.dot(der2_basis1.T, der2_basis1)
    cov_der2_2 = np.dot(der2_basis2.T, der2_basis2)

    basis = np.hstack([basis1, basis2])
    # der_basis = [der_basis1, der_basis2]
    der2_basis = [der2_basis1, der2_basis2]
    cov_der2 = [cov_der2_1,
                cov_der2_2]

    return x1, x2, y, basis, cov_der2, der2_basis, basis1, cov_der2_1, der2_basis1, basis2, cov_der2_2, der2_basis2


def test_multivariate_penalty():
    alphas = [1, 2]
    wts = [1, 1]
    x1, x2, y, basis, cov_der2, der2, basis1, cov_der2_1, der2_1, basis2, cov_der2_2, der2_2 = sample_multivariate_data()

    p = basis.shape[1]
    p1 = basis1.shape[1]
    p2 = basis2.shape[1]

    gp1 = GamPenalty(alpha=alphas[0], cov_der2=cov_der2_1, der2=der2_1)
    gp2 = GamPenalty(alpha=alphas[1], cov_der2=cov_der2_2, der2=der2_2)
    mgp = MultivariateGamPenalty(wts=wts, alphas=alphas, cov_der2=cov_der2,
                                 der2=der2)

    for i in range(10):
        params1 = np.random.randint(-3, 3, p1)
        params2 = np.random.randint(-3, 3, p2)
        params = np.concatenate([params1, params2])
        c1 = gp1.func(params1)
        c2 = gp2.func(params2)
        c = mgp.func(params)
        assert_allclose(c, c1 + c2, atol=1.e-10, rtol=1.e-10)

        d1 = gp1.grad(params1)
        d2 = gp2.grad(params2)
        d12 = np.concatenate([d1, d2])
        d = mgp.grad(params)
        assert_allclose(d, d12)

        h1 = gp1.deriv2(params1)
        h2 = gp2.deriv2(params2)
        h12 = block_diag(h1, h2)
        h = mgp.deriv2(params)
        assert_allclose(h, h12)

    return


def test_gam_significance():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cur_dir, "results", "prediction_from_mgcv.csv")
    data_from_r = pd.read_csv(file_path)
    # dataset used to train the R model
    x = data_from_r.x
    y = data_from_r.y

    df = 10
    degree = 6
    basis, der_basis, der2 = make_bsplines_basis(x, df=df, degree=degree)
    cov_der2 = np.dot(der2.T, der2)

    alpha = 0.045
    gp = GamPenalty(alpha=alpha, cov_der2=cov_der2, der2=der2)
    glm_gam = GLMGam(y, basis, penal=gp)
    res_glm_gam = glm_gam.fit(maxiter=10000)

    t, pvalues, rank = res_glm_gam.significance_test(basis)
    t_from_mgcv = 8.21  # these are the Chi.sq value and p values obtained from MGCV in R with the function summary(g)
    pvalues_from_mgcv = 0.0861

    assert_allclose(t, t_from_mgcv, atol=1.e-16, rtol=1.e-01)

    # TODO: it should be possible to extract the rank from MGCV but I do not know how. Maybe it is the value Ref.df=4.038
    # assert_allclose(rank, ???)

    # TODO: this test is not passed. The error is probably due to the way in which the rank is computed. If rank is replaced by 4 then the test is passed
    # assert_allclose(pvalues, pvalues_from_mgcv, atol=1.e-16, rtol=1.e-01)

    return


'''
test_gam_gradient()
test_gam_hessian()
test_gam_discrete()
test_multivariate_penalty()
test_approximation()
test_gam_glm()
test_gam_penalty()
test_gam_significance()
'''
