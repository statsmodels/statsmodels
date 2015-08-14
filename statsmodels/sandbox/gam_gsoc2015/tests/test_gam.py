
from __future__ import division

__author__ = 'Luca Puggini: <lucapuggio@gmail.com>'
__date__ = '08/07/15'


import os
from statsmodels.sandbox.gam_gsoc2015.smooth_basis import (make_poly_basis, make_bsplines_basis,
                                                           UnivariatePolynomialSmoother, PolynomialSmoother,
                                                           UnivariateBSplines, BSplines, UnivariateGenericSmoother,
                                                           GenericSmoothers, UnivariateCubicSplines,
                                                           UnivariateCubicCyclicSplines,
                                                           CubicSplines, CyclicCubicSplines)
from statsmodels.sandbox.gam_gsoc2015.gam import (GLMGam, LogitGam, make_augmented_matrix, get_sqrt,
                                                  penalized_wls)
from statsmodels.sandbox.gam_gsoc2015.gam_cross_validation.gam_cross_validation import (
                                                                                        UnivariateGamCVPath,
                                                                                        MultivariateGAMCV,
                                                                                        MultivariateGAMCVPath,
                                                                                        _split_train_test_smoothers)
from statsmodels.sandbox.gam_gsoc2015.gam_penalties import UnivariateGamPenalty, MultivariateGamPenalty
from statsmodels.sandbox.gam_gsoc2015.gam_cross_validation.cross_validators import KFold
import numpy as np
import pandas as pd
from statsmodels.genmod.families.family import Gaussian
from numpy.linalg import norm
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from numpy.testing import assert_allclose
import scipy as sp
from statsmodels.genmod.generalized_linear_model import lm


sigmoid = np.vectorize(lambda x: 1.0/ (1.0 + np.exp(-x)))


def univariate_sample_data():
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

    degree = [4]
    pol = PolynomialSmoother(x, degree)

    return pol, y


def integral(params):
    d, c, b, a = params
    itg = (288 * a ** 2) / 5 + (32 * a * c) + 8 * (3 * b ** 2 + c ** 2)
    itg /= 2
    return itg


def grad(params):
    d, c, b, a = params
    grd = np.array([576 * a / 5 + 32 * c, 48 * b, 32 * a + 16 * c, 0])
    grd = grd[::-1]
    return grd / 2


def hessian(params):
    hess = np.array([[576 / 5, 0, 32, 0],
                     [0, 48, 0, 0],
                     [32, 0, 16, 0],
                     [0, 0, 0, 0]
                     ])
    return hess / 2


def cost_function(params, pol, y, alpha):

    # this should be the MSE or log likelihood value
    lin_pred = np.dot(pol.basis_, params)
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
    pol, y = univariate_sample_data()
    univ_pol = pol.smoothers_[0]
    alpha = 1
    gp = UnivariateGamPenalty(alpha=alpha, univariate_smoother=univ_pol)

    for i in range(10):
        params = np.random.randint(-2, 2, 4)
        gp_score = gp.func(params)
        itg = integral(params)
        assert_allclose(gp_score, itg, atol=1.e-1)


def test_gam_gradient():
    """
    test the gam gradient for the example polynomial
    :return:
    """
    pol, y = univariate_sample_data()

    alpha = 1
    gp = UnivariateGamPenalty(alpha=alpha, univariate_smoother=pol.smoothers_[0])
    for i in range(10):
        params = np.random.randint(-2, 2, 4)
        params = np.array([1, 1, 1, 1])
        gam_grad = gp.grad(params)
        grd = grad(params)
        assert_allclose(gam_grad, grd, rtol=1.e-2, atol=1.e-2)

    return


def test_gam_hessian():
    """
    test the deriv2 method of the gam penalty
    :return:
    """
    pol, y = univariate_sample_data()
    univ_pol = pol.smoothers_[0]
    alpha = 1
    gp = UnivariateGamPenalty(alpha=alpha, univariate_smoother=univ_pol)

    for i in range(10):
        params = np.random.randint(-2, 2, 5)
        gam_der2 = gp.deriv2(params)
        hess = hessian(params)
        hess = np.flipud(hess)
        hess = np.fliplr(hess)
        assert_allclose(gam_der2, hess, atol=1.e-13, rtol=1.e-3)
    return


def test_approximation():
    poly, y = univariate_sample_data()
    alpha = 0
    for i in range(10):
        params = np.random.randint(-2, 2, 4)
        cost, err, itg = cost_function(params, poly, y, alpha)
        glm_gam = GLMGam(y, poly, alpha=alpha)
        res_glm_gam = glm_gam.fit(maxiter=1)  # TODO: can this fit be removed? It is useless. We just need the log likelihood
        gam_loglike = glm_gam.loglike(params)
        assert_allclose(gam_loglike, err)
    return


def test_gam_glm():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cur_dir, "results", "prediction_from_mgcv.csv")
    data_from_r = pd.read_csv(file_path)
    # dataset used to train the R model
    x = data_from_r.x.as_matrix()
    y = data_from_r.y.as_matrix()

    df = [10]
    degree = [5]
    bsplines = BSplines(x, degree=degree, df=df)
    # y_mgcv is obtained from R with the following code
    # g = gam(y~s(x, k = 10, bs = "cr"), data = data, scale = 80)
    y_mgcv = data_from_r.y_est

    # alpha = 1000
    alpha = 0.03

    glm_gam = GLMGam(y, bsplines, alpha=alpha)
    res_glm_gam = glm_gam.fit(method='bfgs', max_start_irls=0,
                              disp=1, maxiter=10000, maxfun=5000)

    glm_gam = GLMGam(y, bsplines, alpha=alpha)

    res_glm_gam = glm_gam.fit(method='bfgs', max_start_irls=0,
                              disp=1, maxiter=10000, maxfun=5000)
    y_gam = np.dot(bsplines.basis_, res_glm_gam.params)

    # plt.plot(x, y_gam, '.', label='gam')
    # plt.plot(x, y_mgcv, '.', label='mgcv')
    # plt.plot(x, y, '.', label='y')
    # plt.legend()
    # plt.show()

    assert_allclose(y_gam, y_mgcv, atol=1.e-1)
    return


def test_gam_discrete():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cur_dir, "results", "prediction_from_mgcv.csv")
    data_from_r = pd.read_csv(file_path)
    # dataset used to train the R model
    x = data_from_r.x.as_matrix()
    y = data_from_r.ybin.as_matrix()

    df = [10]
    degree = [5]
    bsplines = BSplines(x, degree=degree, df=df)

    # y_mgcv is obtained from R with the following code
    # g = gam(y~s(x, k = 10, bs = "cr"), data = data, scale = 80)
    y_mgcv = data_from_r.ybin_est

    alpha = 0.00002
    # gp = UnivariateGamPenalty(alpha=alpha, univariate_smoother=bsplines)
    # lg_gam = LogitGam(y, bsplines.basis_, penal=gp)
    #
    lg_gam = LogitGam(y, bsplines, alpha=alpha)
    res_lg_gam = lg_gam.fit(maxiter=10000)
    y_gam = np.dot(bsplines.basis_, res_lg_gam.params)
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


def multivariate_sample_data():
    n = 1000
    x1 = np.linspace(-1, 1, n)
    x2 = np.linspace(-10, 10, n)
    x = np.vstack([x1, x2]).T

    y = x1 * x1 * x1 + x2 + np.random.normal(0, 0.01, n)
    degree1 = 4
    degree2 = 3
    degrees = [degree1, degree2]
    pol = PolynomialSmoother(x, degrees)
    return x, y, pol


def test_multivariate_penalty():
    alphas = [1, 2]
    wts = [1, 1]
    x, y, pol = multivariate_sample_data()

    univ_pol1 = UnivariatePolynomialSmoother(x[:, 0], degree=pol.degrees[0])
    univ_pol2 = UnivariatePolynomialSmoother(x[:, 1], degree=pol.degrees[1])

    gp1 = UnivariateGamPenalty(alpha=alphas[0], univariate_smoother=univ_pol1)
    gp2 = UnivariateGamPenalty(alpha=alphas[1], univariate_smoother=univ_pol2)
    mgp = MultivariateGamPenalty(multivariate_smoother=pol, alpha=alphas, wts=wts)

    for i in range(10):
        params1 = np.random.randint(-3, 3, pol.smoothers_[0].dim_basis)
        params2 = np.random.randint(-3, 3, pol.smoothers_[1].dim_basis)
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


# def test_gam_glm_significance():
#     cur_dir = os.path.dirname(os.path.abspath(__file__))
#     file_path = os.path.join(cur_dir, "results", "prediction_from_mgcv.csv")
#     data_from_r = pd.read_csv(file_path)
#     # dataset used to train the R model
#     x = data_from_r.x.as_matrix()
#     y = data_from_r.y.as_matrix()
#
#     df = [10]
#     degree = [6]
#     bspline = BSplines(x, degree=degree, df=df)
#
#     alpha = 0.045
#     glm_gam = GLMGam(y, bspline, alpha=alpha)
#     res_glm_gam = glm_gam.fit(maxiter=10000)#, method='IRLS')
#
#     t, pvalues, rank = res_glm_gam.significance_test(bspline.basis_)
#     t_from_mgcv = 8.141  # these are the Chi.sq value and p values obtained from MGCV in R with the function summary(g)
#     pvalues_from_mgcv = 0.0864
#     rank_from_mgcv = 3.997
#
#     #assert_allclose(t, t_from_mgcv, atol=1.e-16, rtol=1.e-01)
#
#     # TODO: it should be possible to extract the rank from MGCV but I do not know how. Maybe it is the value Ref.df=4.038
#     #assert_allclose(rank, rank_from_mgcv)
#
#     # TODO: this test is not passed. The error is probably due to the way in which the rank is computed. If rank is replaced by 4 then the test is passed
#     #assert_allclose(pvalues, pvalues_from_mgcv, atol=1.e-16, rtol=1.e-01)
#
#     return


def test_partial_values():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cur_dir, "results", "prediction_from_mgcv.csv")

    data_from_r = pd.read_csv(file_path)

    # dataset used to train the R model
    x = data_from_r.x.as_matrix()
    y = data_from_r.y.as_matrix()
    se_from_mgcv = data_from_r.y_est_se
    df = [10]
    degree = [6]
    bsplines = BSplines(x, degree=degree, df=df)

    alpha = 0.025
    glm_gam = GLMGam(y, bsplines, alpha=alpha)
    res_glm_gam = glm_gam.fit(maxiter=10000, method='bfgs') # TODO: if IRLS is used res_glm_gam has not partial_values.

    univ_bsplines = bsplines.smoothers_[0]
    hat_y, se = res_glm_gam.partial_values(bsplines, mask=np.array([True]*univ_bsplines.dim_basis))

    assert_allclose(se, se_from_mgcv, rtol=0, atol=0.008)

    return


def test_partial_plot():
    # TODO: No test is performed.
    # Generate a plot to visualize analyze the result.

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cur_dir, "results", "prediction_from_mgcv.csv")

    data_from_r = pd.read_csv(file_path)

    # dataset used to train the R model
    x = data_from_r.x.as_matrix()
    y = data_from_r.y.as_matrix()
    se_from_mgcv = data_from_r.y_est_se
    df = [10]
    degree = [6]
    bsplines = BSplines(x, degree=degree, df=df)


    alpha = 0.03
    glm_gam = GLMGam(y, bsplines, alpha=alpha)
    res_glm_gam = glm_gam.fit(maxiter=10000, method='bfgs')

    # Uncomment to visualize the plot
    res_glm_gam.plot_partial(bsplines)
    plt.plot(x, y, '.')
    plt.show()

    return


def test_univariate_gam_cv_kfolds():

    def sample_metric(y1, y2):

        return np.linalg.norm(y1 - y2)/len(y1)

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cur_dir, "results", "prediction_from_mgcv.csv")

    data_from_r = pd.read_csv(file_path)

    # dataset used to train the R model
    x = data_from_r.x
    y = data_from_r.y
    se_from_mgcv = data_from_r.y_est_se
    df = 10
    degree = 6
    univ_bsplines = UnivariateBSplines(x, degree=degree, df=df)

    gam = GLMGam
    alphas = np.linspace(0, .02, 2)
    k = 10
    cv = KFold(k_folds=k)
    gam_cv = UnivariateGamCVPath(univariate_smoother=univ_bsplines, alphas=alphas, gam=gam, cost=sample_metric,
                                 y=y, cv=cv).fit()

    gp = UnivariateGamPenalty(alpha=gam_cv.alpha_cv_, univariate_smoother=univ_bsplines)
    glm_gam = GLMGam(y, univ_bsplines.basis_, penal=gp)
    res_glm_gam = glm_gam.fit(maxiter=10000)#, method='IRLS')
    y_est = res_glm_gam.predict(univ_bsplines.basis_)

    # The test is done with the result obtained with GCV and not KFOLDS CV.
    # This is because MGCV does not support KFOLD CV
    assert_allclose(data_from_r.y_mgcv_gcv, y_est, atol=1.e-1, rtol=1.e-1)

    return


def test_generic_smoother():

    x, y, poly = multivariate_sample_data()
    alphas = [0.4, 0.7]
    wts = [1, 1]

    gs = GenericSmoothers(poly.x, poly.smoothers_)
    gam_gs = GLMGam(y, gs, alpha=alphas)
    gam_gs_res = gam_gs.fit()

    gam_poly = GLMGam(y, poly, alpha=alphas)
    gam_poly_res = gam_poly.fit()

    assert_allclose(gam_gs_res.params, gam_poly_res.params)

    return


def test_multivariate_gam_1d_data():

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cur_dir, "results", "prediction_from_mgcv.csv")
    data_from_r = pd.read_csv(file_path)
    # dataset used to train the R model
    x = data_from_r.x.as_matrix()
    y = data_from_r.y

    df = [10]
    degree = [5]
    bsplines = BSplines(x, degree=degree, df=df)
    # y_mgcv is obtained from R with the following code
    # g = gam(y~s(x, k = 10, bs = "cr"), data = data, scale = 80)
    y_mgcv = data_from_r.y_est

    alpha = [0.0251]
    gp = MultivariateGamPenalty(bsplines, alpha=alpha)

    glm_gam = GLMGam(y, bsplines, alpha=alpha)
    res_glm_gam = glm_gam.fit(method='nm', max_start_irls=0,
                              disp=1, maxiter=10000, maxfun=5000)
    y_gam = np.dot(bsplines.basis_, res_glm_gam.params)

    # plt.plot(x, y_gam, '.', label='gam')
    # plt.plot(x, y_mgcv, '.', label='mgcv')
    # plt.plot(x, y, '.', label='y')
    # plt.legend()
    # plt.show()

    assert_allclose(y_gam, y_mgcv, atol=8.e-2)
    return


def test_multivariate_gam_cv():
    # no test is performed. It only checks that there isn't any runtime error

    def cost(x1, x2):
        return np.linalg.norm(x1 - x2) / len(x1)

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cur_dir, "results", "prediction_from_mgcv.csv")
    data_from_r = pd.read_csv(file_path)
    # dataset used to train the R model
    x = data_from_r.x.as_matrix()
    y = data_from_r.y.as_matrix()

    df = [10]
    degree = [5]
    bsplines = BSplines(x, degree=degree, df=df)
    # y_mgcv is obtained from R with the following code
    # g = gam(y~s(x, k = 10, bs = "cr"), data = data, scale = 80)

    alphas = [0.0251]
    alphas = [2]
    cv = KFold(3)

    gp = MultivariateGamPenalty(bsplines, alpha=alphas)
    gam_cv = MultivariateGAMCV(smoothers=bsplines, alphas=alphas, gam=GLMGam, cost=cost, y=y, cv=cv)
    gam_cv_res = gam_cv.fit()

    return


def test_multivariate_gam_cv_path():

    def sample_metric(y1, y2):

        return np.linalg.norm(y1 - y2)/len(y1)

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cur_dir, "results", "prediction_from_mgcv.csv")

    data_from_r = pd.read_csv(file_path)

    # dataset used to train the R model
    x = data_from_r.x.as_matrix()
    y = data_from_r.y.as_matrix()
    se_from_mgcv = data_from_r.y_est_se
    y_mgcv = data_from_r.y_mgcv_gcv

    df = [10]
    degree = [6]

    bsplines = BSplines(x, degree=degree, df=df)

    gam = GLMGam
    alphas = [np.linspace(0, 2, 10)]
    k = 3
    cv = KFold(k_folds=k, shuffle=True)

    # TODO: penal=?
    gam_cv = MultivariateGAMCVPath(smoothers=bsplines, alphas=alphas, gam=gam, cost=sample_metric, y=y, cv=cv)
    gam_cv_res = gam_cv.fit()

    glm_gam = GLMGam(y, bsplines, alpha=gam_cv.alpha_cv_)
    res_glm_gam = glm_gam.fit(method='irls', max_start_irls=0,
                              disp=1, maxiter=10000, maxfun=5000)
    y_est = res_glm_gam.predict(bsplines.basis_)

    # plt.plot(x, y, '.', label='y')
    # plt.plot(x, y_est, '.', label='y est')
    # plt.plot(x, y_mgcv, '.', label='y mgcv')
    # plt.legend()
    # plt.show()
    
    # The test is done with the result obtained with GCV and not KFOLDS CV.
    # This is because MGCV does not support KFOLD CV
    assert_allclose(data_from_r.y_mgcv_gcv, y_est, atol=1.e-1, rtol=1.e-1)

    return


def test_train_test_smoothers():

    n = 6
    x = np.zeros(shape=(n, 2))
    x[:, 0] = range(6)
    x[:, 1] = range(6, 12)
    poly = PolynomialSmoother(x, degrees=[3, 3])
    train_index = list(range(3))
    test_index = list(range(3, 6))
    train_smoother, test_smoother = _split_train_test_smoothers(poly.x, poly, train_index, test_index)

    expected_train_basis = [[0., 0., 0., 6., 36., 216.],
                            [1., 1., 1., 7., 49., 343.],
                            [2., 4., 8., 8., 64., 512.]]
    assert_allclose(train_smoother.basis_, expected_train_basis)

    expected_test_basis = [[3., 9., 27., 9., 81., 729.],
                           [4., 16., 64., 10., 100., 1000.],
                           [5., 25., 125., 11., 121., 1331.]]
    assert_allclose(test_smoother.basis_, expected_test_basis)

    return


def test_get_sqrt():
    n = 1000
    x = np.random.normal(0, 1, (n,3))
    x2 = np.dot(x.T, x)

    sqrt_x2 = get_sqrt(x2)

    x2_reconstruction = np.dot(sqrt_x2, sqrt_x2)
    assert_allclose(x2_reconstruction, x2)

    return


def test_make_augmented_matrix():

    n = 500
    x = np.random.uniform(-1, 1, (n, 3))
    s = np.dot(x.T, x)
    y = np.array(list(range(n)))
    w = np.random.uniform(0, 1, n)
    n_samples, n_columns = x.shape

    alpha = 0
    aug_x, aug_y, aug_w = make_augmented_matrix(x, y, s, w, alpha)
    expected_aug_x = np.vstack([x, np.zeros(shape=(n_columns, n_columns))])
    assert_allclose(aug_x, expected_aug_x)
    expected_aug_y = np.zeros(shape=(n_samples + n_columns, ))
    expected_aug_y[:n_samples] = y
    assert_allclose(aug_y, expected_aug_y)
    expected_aug_w = np.array([np.sqrt(i) for i in w] + [1] * n_columns)
    assert_allclose(aug_w, expected_aug_w)

    from statsmodels.sandbox.gam_gsoc2015.gam import get_sqrt
    alpha = 1
    aug_x, aug_y, aug_w = make_augmented_matrix(x, y, s, w, alpha)
    rs = get_sqrt(alpha * s)
    # rs = sp.linalg.cholesky(alpha * s)
    assert_allclose(np.dot(rs.T, rs), alpha*s)
    x1 = np.vstack([x, rs])  # augmented x
    expected_aug_x = np.vstack([x, rs])
    assert_allclose(aug_x, expected_aug_x)
    expected_aug_y = np.zeros(shape=(n_samples + n_columns, ))
    expected_aug_y[:n_samples] = y
    assert_allclose(aug_y, expected_aug_y)
    expected_aug_w = np.array([np.sqrt(i) for i in w] + [1] * n_columns)
    assert_allclose(aug_w, expected_aug_w)

    return


def test_penalized_wls():

    n = 20
    p = 3
    x = np.random.normal(0, 1, (n, 3))
    y = x[:, 1] - x[:, 2] + np.random.normal(0, .1, n)
    y -= y.mean()

    weights = np.ones(shape=(n,))
    s = np.random.normal(0, 1, (p, p))

    pen_wls_res = penalized_wls(x, y, s, weights, alpha=0)
    ls_res = lm.OLS(y, x).fit()

    assert_allclose(ls_res.params, pen_wls_res.params)

    return


def test_cyclic_cubic_splines():

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cur_dir, "results", "cubic_cyclic_splines_from_mgcv.csv")
    data_from_r = pd.read_csv(file_path)

    x = data_from_r[['x0', 'x2']].as_matrix()
    y = data_from_r['y'].as_matrix()
    y_est_mgcv = data_from_r[['y_est']].as_matrix()
    s_mgcv = data_from_r[['s(x0)', 's(x2)']].as_matrix()

    dfs = [10, 10]
    ccs = CyclicCubicSplines(x, df=dfs)
    alpha = [0.05, 0.0005]

    gam = GLMGam(y, ccs, alpha=alpha)
    #gam_res = gam._fit_pirls(y, ccs, alpha=alpha)
    gam_res = gam.fit(method='pirls')

    s0 = np.dot(ccs.basis_[:, ccs.mask[0]],
                gam_res.params[ccs.mask[0]])
    s0 -= s0.mean() # TODO: Mean has to be removed

    s1 = np.dot(ccs.basis_[:, ccs.mask[1]],
                gam_res.params[ccs.mask[1]])
    s1 -= s1.mean() # TODO: Mean has to be removed

    # plt.subplot(2, 1, 1)
    # plt.plot(x[:, 0], s0, '.', label='s0')
    # plt.plot(x[:, 0], s_mgcv[:, 0], '.', label='s0_mgcv')
    # plt.legend(loc='best')
    #
    # plt.subplot(2, 1, 2)
    # plt.plot(x[:, 1], s1, '.', label='s1_est')
    # plt.plot(x[:, 1], s_mgcv[:, 1], '.', label='s1_mgcv')
    # plt.legend(loc='best')
    # plt.show()

    assert_allclose(s0, s_mgcv[:, 0], atol=0.02)
    assert_allclose(s1, s_mgcv[:, 1], atol=0.33)

    return


def test_multivariate_cubic_splines():

    np.random.seed(0)
    from statsmodels.sandbox.gam_gsoc2015.smooth_basis import CubicSplines

    n = 500
    x1 = np.linspace(-3, 3, n)
    x2 = np.linspace(0, 1, n)

    x = np.vstack([x1, x2]).T
    y1 = np.sin(x1)/x1
    y2 = x2*x2
    y0 = y1 + y2
    y = y0 + np.random.normal(0, .3, n)
    y -= y.mean()
    y0 -= y0.mean()

    alphas = [1.5] * 2
    cs = CubicSplines(x, df=[10, 10])

    gam = GLMGam(y, cs, alpha=alphas)
    #gam_res = gam.fit(y, cs, alpha=alphas, method='pirls')
    gam_res = gam.fit()


    y_est = np.dot(cs.basis_, gam_res.params)
    y_est -= y_est.mean()

    # cut the tails
    index = list(range(50, n-50))
    y_est = y_est[index]
    y0 = y0[index]
    y = y[index]

    # plt.plot(y_est, label='y est')
    # plt.plot(y0, label='y0')
    # plt.plot(y, '.', label='y')
    # plt.legend(loc='best')
    # plt.show()

    assert_allclose(y_est, y0, atol=0.04)

    return


# test_gam_glm_significance() # TODO1: not implemented
# test_approximation() # Computationally demanding
# test_gam_hessian()
# test_gam_gradient()
# test_gam_discrete()
# test_multivariate_penalty()
# test_gam_glm()
# test_gam_penalty()
# test_partial_plot()
# test_partial_values()
# test_generic_smoother()
# test_multivariate_gam_1d_data()
# test_multivariate_gam_cv()
# test_multivariate_gam_cv_path()
# test_train_test_smoothers()
# test_make_augmented_matrix()
# test_penalized_wls()
# test_cyclic_cubic_splines()
# test_multivariate_cubic_splines()
# test_get_sqrt()
