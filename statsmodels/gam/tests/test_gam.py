"""
unit test for GAM

Author: Luca Puggini

Created on 08/07/2015
"""

from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.linalg import block_diag

from statsmodels.gam.gam_cross_validation.cross_validators import KFold
from statsmodels.gam.gam_cross_validation.gam_cross_validation import (
    MultivariateGAMCV,
    MultivariateGAMCVPath,
    _split_train_test_smoothers,
)
from statsmodels.gam.gam_penalties import MultivariateGamPenalty, UnivariateGamPenalty
from statsmodels.gam.generalized_additive_model import (
    GLMGam,
    LogitGam,
    make_augmented_matrix,
    penalized_wls,
)
from statsmodels.gam.smooth_basis import (
    BSplines,
    CyclicCubicSplines,
    GenericSmoothers,
    PolynomialSmoother,
    UnivariateCubicSplines,
    UnivariatePolynomialSmoother,
)
from statsmodels.genmod.families.family import Gaussian
from statsmodels.genmod.generalized_linear_model import GLM, lm
from statsmodels.tools.linalg import matrix_sqrt

sigmoid = np.vectorize(lambda x: 1.0 / (1.0 + np.exp(-x)))


def polynomial_sample_data():
    """A polynomial of degree 4

    poly = ax^4 + bx^3 + cx^2 + dx + e
    second der = 12ax^2 + 6bx + 2c
    integral from -1 to 1 of second der^2 is
        (288 a^2)/5 + 32 a c + 8 (3 b^2 + c^2)
    the gradient of the integral is der
        [576*a/5 + 32 * c, 48*b, 32*a + 16*c, 0, 0]

    Returns
    -------
    poly : smoother instance
    y : ndarray
        generated function values, demeaned
    """
    n = 10000
    x = np.linspace(-1, 1, n)
    y = 2 * x**3 - x
    y -= y.mean()
    degree = [4]
    pol = PolynomialSmoother(x, degree)
    return (pol, y)


def integral(params):
    d, c, b, a = params
    itg = 288 * a**2 / 5 + 32 * a * c + 8 * (3 * b**2 + c**2)
    itg /= 2
    return itg


def grad(params):
    d, c, b, a = params
    grd = np.array([576 * a / 5 + 32 * c, 48 * b, 32 * a + 16 * c, 0])
    grd = grd[::-1]
    return grd / 2


def hessian(params):
    hess = np.array([[576 / 5, 0, 32, 0], [0, 48, 0, 0], [32, 0, 16, 0], [0, 0, 0, 0]])
    return hess / 2


def cost_function(params, pol, y, alpha):
    lin_pred = np.dot(pol.basis, params)
    gaussian = Gaussian()
    expval = gaussian.link.inverse(lin_pred)
    loglike = gaussian.loglike(y, expval)
    itg = integral(params)
    return (loglike - alpha * itg, loglike, itg)


def test_gam_penalty():
    """
    test the func method of the gam penalty
    :return:
    """
    rs = np.random.RandomState(9993431)
    pol, y = polynomial_sample_data()
    univ_pol = pol.smoothers[0]
    alpha = 1
    gp = UnivariateGamPenalty(alpha=alpha, univariate_smoother=univ_pol)
    for _ in range(10):
        params = rs.randint(-2, 2, 4)
        gp_score = gp.func(params)
        itg = integral(params)
        assert_allclose(gp_score, itg, atol=0.1)


def test_gam_gradient():
    pol, y = polynomial_sample_data()
    alpha = 1
    smoother = pol.smoothers[0]
    gp = UnivariateGamPenalty(alpha=alpha, univariate_smoother=smoother)
    for _ in range(10):
        params = np.array([1, 1, 1, 1])
        gam_grad = gp.deriv(params)
        grd = grad(params)
        assert_allclose(gam_grad, grd, rtol=0.01, atol=0.01)


def test_gam_hessian():
    rs = np.random.RandomState(1)
    pol, y = polynomial_sample_data()
    univ_pol = pol.smoothers[0]
    alpha = 1
    gp = UnivariateGamPenalty(alpha=alpha, univariate_smoother=univ_pol)
    for _ in range(10):
        params = rs.randint(-2, 2, 5)
        gam_der2 = gp.deriv2(params)
        hess = hessian(params)
        hess = np.flipud(hess)
        hess = np.fliplr(hess)
        assert_allclose(gam_der2, hess, atol=1e-13, rtol=0.001)


def test_approximation():
    rs = np.random.RandomState(1)
    poly, y = polynomial_sample_data()
    alpha = 1
    for _ in range(10):
        params = rs.uniform(-1, 1, 4)
        cost, err, itg = cost_function(params, poly, y, alpha)
        glm_gam = GLMGam(y, smoother=poly, alpha=alpha)
        gam_loglike = glm_gam.loglike(params, scale=1, pen_weight=1)
        assert_allclose(err - itg, cost, rtol=1e-10)
        assert_allclose(gam_loglike, cost, rtol=0.1)


def test_gam_glm():
    cur_dir = Path(__file__).resolve().parent
    file_path = Path(cur_dir).joinpath("results", "prediction_from_mgcv.csv")
    data_from_r = pd.read_csv(file_path)
    x = data_from_r.x.values
    y = data_from_r.y.values
    df = [10]
    degree = [3]
    bsplines = BSplines(x, degree=degree, df=df, include_intercept=True)
    y_mgcv = np.asarray(data_from_r.y_est)
    alpha = 0.1
    glm_gam = GLMGam(y, smoother=bsplines, alpha=alpha)
    res_glm_gam = glm_gam.fit(method="bfgs", max_start_irls=0, disp=1, maxiter=10000)
    y_gam0 = np.dot(bsplines.basis, res_glm_gam.params)
    y_gam = np.asarray(res_glm_gam.fittedvalues)
    assert_allclose(y_gam, y_gam0, rtol=1e-10)
    assert_allclose(y_gam, y_mgcv, atol=0.01)


def test_gam_discrete():
    cur_dir = Path(__file__).resolve().parent
    file_path = Path(cur_dir).joinpath("results", "prediction_from_mgcv.csv")
    data_from_r = pd.read_csv(file_path)
    x = data_from_r.x.values
    y = data_from_r.ybin.values
    df = [10]
    degree = [5]
    bsplines = BSplines(x, degree=degree, df=df, include_intercept=True)
    y_mgcv = data_from_r.ybin_est
    alpha = 2e-05
    lg_gam = LogitGam(y, bsplines, alpha=alpha)
    res_lg_gam = lg_gam.fit(maxiter=10000)
    y_gam = np.dot(bsplines.basis, res_lg_gam.params)
    y_gam = sigmoid(y_gam)
    y_mgcv = sigmoid(y_mgcv)
    assert_allclose(y_gam, y_mgcv, rtol=1e-10, atol=0.1)


def multivariate_sample_data(seed=1):
    n = 1000
    x1 = np.linspace(-1, 1, n)
    x2 = np.linspace(-10, 10, n)
    x = np.vstack([x1, x2]).T
    rs = np.random.RandomState(seed)
    y = x1 * x1 * x1 + x2 + rs.normal(0, 0.01, n)
    degree1 = 4
    degree2 = 3
    degrees = [degree1, degree2]
    pol = PolynomialSmoother(x, degrees)
    return (x, y, pol)


def test_multivariate_penalty():
    alphas = [1, 2]
    weights = [1, 1]
    rs = np.random.RandomState(1)
    x, y, pol = multivariate_sample_data()
    univ_pol1 = UnivariatePolynomialSmoother(x[:, 0], degree=pol.degrees[0])
    univ_pol2 = UnivariatePolynomialSmoother(x[:, 1], degree=pol.degrees[1])
    gp1 = UnivariateGamPenalty(alpha=alphas[0], univariate_smoother=univ_pol1)
    gp2 = UnivariateGamPenalty(alpha=alphas[1], univariate_smoother=univ_pol2)
    with pytest.warns(UserWarning, match="weights is currently ignored"):
        mgp = MultivariateGamPenalty(
            multivariate_smoother=pol, alpha=alphas, weights=weights
        )
    for _ in range(10):
        params1 = rs.randint(-3, 3, pol.smoothers[0].dim_basis)
        params2 = rs.randint(-3, 3, pol.smoothers[1].dim_basis)
        params = np.concatenate([params1, params2])
        c1 = gp1.func(params1)
        c2 = gp2.func(params2)
        c = mgp.func(params)
        assert_allclose(c, c1 + c2, atol=1e-10, rtol=1e-10)
        d1 = gp1.deriv(params1)
        d2 = gp2.deriv(params2)
        d12 = np.concatenate([d1, d2])
        d = mgp.deriv(params)
        assert_allclose(d, d12)
        h1 = gp1.deriv2(params1)
        h2 = gp2.deriv2(params2)
        h12 = block_diag(h1, h2)
        h = mgp.deriv2(params)
        assert_allclose(h, h12)


def test_generic_smoother():
    x, y, poly = multivariate_sample_data()
    alphas = [0.4, 0.7]
    weights = [1, 1]
    gs = GenericSmoothers(poly.x, poly.smoothers)
    gam_gs = GLMGam(y, smoother=gs, alpha=alphas)
    gam_gs_res = gam_gs.fit()
    gam_poly = GLMGam(y, smoother=poly, alpha=alphas)
    gam_poly_res = gam_poly.fit()
    assert_allclose(gam_gs_res.params, gam_poly_res.params)


def test_multivariate_gam_1d_data():
    cur_dir = Path(__file__).resolve().parent
    file_path = Path(cur_dir).joinpath("results", "prediction_from_mgcv.csv")
    data_from_r = pd.read_csv(file_path)
    x = data_from_r.x.values
    y = data_from_r.y
    df = [10]
    degree = [3]
    bsplines = BSplines(x, degree=degree, df=df)
    y_mgcv = data_from_r.y_est
    alpha = [0.0168 * 0.0251 / 2 * 500]
    _ = MultivariateGamPenalty(bsplines, alpha=alpha)
    glm_gam = GLMGam(y, exog=np.ones((len(y), 1)), smoother=bsplines, alpha=alpha)
    res_glm_gam = glm_gam.fit(method="pirls", max_start_irls=0, disp=1, maxiter=10000)
    y_gam = res_glm_gam.fittedvalues
    assert_allclose(y_gam, y_mgcv, atol=0.01)


def test_multivariate_gam_cv():

    def cost(x1, x2):
        return np.linalg.norm(x1 - x2) / len(x1)

    cur_dir = Path(__file__).resolve().parent
    file_path = Path(cur_dir).joinpath("results", "prediction_from_mgcv.csv")
    data_from_r = pd.read_csv(file_path)
    x = data_from_r.x.values
    y = data_from_r.y.values
    df = [10]
    degree = [5]
    bsplines = BSplines(x, degree=degree, df=df)
    alphas = [0.0251]
    alphas = [2]
    cv = KFold(3)
    gp = MultivariateGamPenalty(bsplines, alpha=alphas)
    gam_cv = MultivariateGAMCV(
        smoother=bsplines,
        alphas=alphas,
        gam=GLMGam,
        cost=cost,
        endog=y,
        exog=None,
        cv_iterator=cv,
    )
    gam_cv_res = gam_cv.fit()


def test_multivariate_gam_cv_path():

    def sample_metric(y1, y2):
        return np.linalg.norm(y1 - y2) / len(y1)

    cur_dir = Path(__file__).resolve().parent
    file_path = Path(cur_dir).joinpath("results", "prediction_from_mgcv.csv")
    data_from_r = pd.read_csv(file_path)
    x = data_from_r.x.values
    y = data_from_r.y.values
    se_from_mgcv = data_from_r.y_est_se
    y_mgcv = data_from_r.y_mgcv_gcv
    df = [10]
    degree = [6]
    bsplines = BSplines(x, degree=degree, df=df, include_intercept=True)
    gam = GLMGam
    alphas = [np.linspace(0, 2, 10)]
    k = 3
    rs = np.random.RandomState(4212121)
    cv = KFold(k_folds=k, shuffle=True, rng=rs)
    gam_cv = MultivariateGAMCVPath(
        smoother=bsplines,
        alphas=alphas,
        gam=gam,
        cost=sample_metric,
        endog=y,
        exog=None,
        cv_iterator=cv,
    )
    gam_cv_res = gam_cv.fit()
    glm_gam = GLMGam(y, smoother=bsplines, alpha=gam_cv.alpha_cv)
    res_glm_gam = glm_gam.fit(method="irls", max_start_irls=0, disp=1, maxiter=10000)
    y_est = res_glm_gam.predict(bsplines.basis)
    assert_allclose(data_from_r.y_mgcv_gcv, y_est, atol=0.1, rtol=0.1)
    rs = np.random.RandomState(123)
    alpha_cv, res_cv = glm_gam.select_penweight_kfold(alphas=alphas, k_folds=3, rng=rs)
    assert_allclose(alpha_cv, gam_cv.alpha_cv, rtol=1e-12)


def test_train_test_smoothers():
    n = 6
    x = np.zeros(shape=(n, 2))
    x[:, 0] = range(6)
    x[:, 1] = range(6, 12)
    poly = PolynomialSmoother(x, degrees=[3, 3])
    train_index = list(range(3))
    test_index = list(range(3, 6))
    train_smoother, test_smoother = _split_train_test_smoothers(
        poly.x, poly, train_index, test_index
    )
    expected_train_basis = [
        [0.0, 0.0, 0.0, 6.0, 36.0, 216.0],
        [1.0, 1.0, 1.0, 7.0, 49.0, 343.0],
        [2.0, 4.0, 8.0, 8.0, 64.0, 512.0],
    ]
    assert_allclose(train_smoother.basis, expected_train_basis)
    expected_test_basis = [
        [3.0, 9.0, 27.0, 9.0, 81.0, 729.0],
        [4.0, 16.0, 64.0, 10.0, 100.0, 1000.0],
        [5.0, 25.0, 125.0, 11.0, 121.0, 1331.0],
    ]
    assert_allclose(test_smoother.basis, expected_test_basis)


def test_get_sqrt():
    n = 1000
    rs = np.random.RandomState(1)
    x = rs.normal(0, 1, (n, 3))
    x2 = np.dot(x.T, x)
    sqrt_x2 = matrix_sqrt(x2)
    x2_reconstruction = np.dot(sqrt_x2.T, sqrt_x2)
    assert_allclose(x2_reconstruction, x2)


def test_make_augmented_matrix():
    rs = np.random.RandomState(1)
    n = 500
    x = rs.uniform(-1, 1, (n, 3))
    s = np.dot(x.T, x)
    y = np.array(list(range(n)))
    w = rs.uniform(0, 1, n)
    nobs, n_columns = x.shape
    alpha = 0
    aug_y, aug_x, aug_w = make_augmented_matrix(y, x, alpha * s, w)
    expected_aug_x = x
    assert_allclose(aug_x, expected_aug_x)
    expected_aug_y = y
    expected_aug_y[:nobs] = y
    assert_allclose(aug_y, expected_aug_y)
    expected_aug_w = w
    assert_allclose(aug_w, expected_aug_w)
    alpha = 1
    aug_y, aug_x, aug_w = make_augmented_matrix(y, x, s, w)
    rs = matrix_sqrt(alpha * s)
    assert_allclose(np.dot(rs.T, rs), alpha * s)
    expected_aug_x = np.vstack([x, rs])
    assert_allclose(aug_x, expected_aug_x)
    expected_aug_y = np.zeros(shape=(nobs + n_columns,))
    expected_aug_y[:nobs] = y
    assert_allclose(aug_y, expected_aug_y)
    expected_aug_w = np.concatenate((w, [1] * n_columns), axis=0)
    assert_allclose(aug_w, expected_aug_w)


def test_penalized_wls():
    rs = np.random.RandomState(1)
    n = 20
    p = 3
    x = rs.normal(0, 1, (n, 3))
    y = x[:, 1] - x[:, 2] + rs.normal(0, 0.1, n)
    y -= y.mean()
    weights = np.ones(shape=(n,))
    s = rs.normal(0, 1, (p, p))
    pen_wls_res = penalized_wls(y, x, 0 * s, weights)
    ls_res = lm.OLS(y, x).fit()
    assert_allclose(ls_res.params, pen_wls_res.params)


def test_cyclic_cubic_splines():
    cur_dir = Path(__file__).resolve().parent
    file_path = Path(cur_dir).joinpath("results", "cubic_cyclic_splines_from_mgcv.csv")
    data_from_r = pd.read_csv(file_path)
    x = data_from_r[["x0", "x2"]].values
    y = data_from_r["y"].values
    y_est_mgcv = data_from_r[["y_est"]].values
    s_mgcv = data_from_r[["s(x0)", "s(x2)"]].values
    dfs = [10, 10]
    ccs = CyclicCubicSplines(x, df=dfs)
    alpha = [0.05 / 2, 0.0005 / 2]
    gam = GLMGam(y, smoother=ccs, alpha=alpha)
    gam_res = gam.fit(method="pirls")
    s0 = np.dot(ccs.basis[:, ccs.mask[0]], gam_res.params[ccs.mask[0]])
    s0 -= s0.mean()
    s1 = np.dot(ccs.basis[:, ccs.mask[1]], gam_res.params[ccs.mask[1]])
    s1 -= s1.mean()
    assert_allclose(s0, s_mgcv[:, 0], atol=0.02)
    assert_allclose(s1, s_mgcv[:, 1], atol=0.33)


def test_multivariate_cubic_splines():
    rs = np.random.RandomState(0)
    from statsmodels.gam.smooth_basis import CubicSplines

    n = 500
    x1 = np.linspace(-3, 3, n)
    x2 = np.linspace(0, 1, n) ** 2
    x = np.vstack([x1, x2]).T
    y1 = np.sin(x1) / x1
    y2 = x2 * x2
    y0 = y1 + y2
    y = y0 + rs.normal(0, 0.3 / 2, n)
    y -= y.mean()
    y0 -= y0.mean()
    alphas = [0.001, 0.001]
    cs = CubicSplines(x, df=[10, 10], constraints="center")
    gam = GLMGam(y, exog=np.ones((n, 1)), smoother=cs, alpha=alphas)
    gam_res = gam.fit(method="pirls")
    y_est = gam_res.fittedvalues
    y_est -= y_est.mean()
    index = list(range(50, n - 50))
    y_est = y_est[index]
    y0 = y0[index]
    y = y[index]
    assert_allclose(y_est, y0, atol=0.04)


def test_glm_pirls_compatibility():
    rs = np.random.RandomState(0)
    n = 500
    x1 = np.linspace(-3, 3, n)
    x2 = rs.rand(n)
    x = np.vstack([x1, x2]).T
    y1 = np.sin(x1) / x1
    y2 = x2 * x2
    y0 = y1 + y2
    y = y0 + rs.normal(0, 0.3, n)
    y -= y.mean()
    y0 -= y0.mean()
    alphas = [5.75] * 2
    alphas_glm = [1.2] * 2
    cs = BSplines(x, df=[10, 10], degree=[3, 3], constraints="center")
    gam_pirls = GLMGam(y, smoother=cs, alpha=alphas)
    gam_glm = GLMGam(y, smoother=cs, alpha=alphas)
    gam_res_glm = gam_glm.fit(method="nm", max_start_irls=0, disp=1, maxiter=20000)
    gam_res_glm = gam_glm.fit(
        start_params=gam_res_glm.params,
        method="bfgs",
        max_start_irls=0,
        disp=1,
        maxiter=20000,
    )
    gam_res_pirls = gam_pirls.fit()
    y_est_glm = np.dot(cs.basis, gam_res_glm.params)
    y_est_glm -= y_est_glm.mean()
    y_est_pirls = np.dot(cs.basis, gam_res_pirls.params)
    y_est_pirls -= y_est_pirls.mean()
    assert_allclose(gam_res_glm.params, gam_res_pirls.params, atol=5e-05, rtol=5e-05)
    assert_allclose(y_est_glm, y_est_pirls, atol=5e-05)


def test_zero_penalty():
    x, y, poly = multivariate_sample_data()
    alphas = [0, 0]
    gam_gs = GLMGam(y, smoother=poly, alpha=alphas)
    gam_gs_res = gam_gs.fit()
    y_est_gam = gam_gs_res.predict()
    glm = GLM(y, poly.basis).fit()
    y_est = glm.predict()
    assert_allclose(y_est, y_est_gam)


def test_spl_s():
    spl_s_R = [
        [0, 0, 0.0, 0.0, 0.0, 0.0],
        [0, 0, 0.0, 0.0, 0.0, 0.0],
        [0, 0, 0.0014, 0.0002, -0.001133333, -0.001],
        [0, 0, 0.0002, 0.002733333, 0.001666667, -0.001133333],
        [0, 0, -0.001133333, 0.001666667, 0.002733333, 0.0002],
        [0, 0, -0.001, -0.001133333, 0.0002, 0.0014],
    ]
    rs = np.random.RandomState(1)
    x = rs.normal(0, 1, 10)
    xk = np.array([0.2, 0.4, 0.6, 0.8])
    cs = UnivariateCubicSplines(x, df=4)
    cs.knots = xk
    spl_s = cs._splines_s()
    assert_allclose(spl_s_R, spl_s, atol=4e-10)


def test_partial_values2():
    rs = np.random.RandomState(0)
    n = 1000
    x = rs.uniform(0, 1, (n, 2))
    x = x - x.mean()
    y = x[:, 0] * x[:, 0] + rs.normal(0, 0.01, n)
    y -= y.mean()
    alpha = 0.0
    bsplines = BSplines(x, degree=[3] * 2, df=[10] * 2, include_intercept=[True, False])
    glm_gam = GLMGam(y, smoother=bsplines, alpha=alpha)
    res_glm_gam = glm_gam.fit(method="pirls", max_start_irls=0, disp=0, maxiter=5000)
    glm = GLM(y, bsplines.basis)
    ex = np.column_stack(
        (bsplines.smoothers[0].basis, np.zeros_like(bsplines.smoothers[1].basis))
    )
    y_est = res_glm_gam.predict(ex, transform=False)
    y_partial_est, se = res_glm_gam.partial_values(0)
    assert_allclose(y_est, y_partial_est, atol=0.05)
    assert se.min() < 100


def test_partial_values():
    cur_dir = Path(__file__).resolve().parent
    file_path = Path(cur_dir).joinpath("results", "prediction_from_mgcv.csv")
    data_from_r = pd.read_csv(file_path)
    x = data_from_r.x.values
    y = data_from_r.y.values
    se_from_mgcv = data_from_r.y_est_se
    df = [10]
    degree = [6]
    bsplines = BSplines(x, degree=degree, df=df, include_intercept=True)
    alpha = 0.025 / 115 * 500
    glm_gam = GLMGam(y, smoother=bsplines, alpha=alpha)
    res_glm_gam = glm_gam.fit(maxiter=10000, method="bfgs")
    univ_bsplines = bsplines.smoothers[0]
    hat_y, se = res_glm_gam.partial_values(0)
    assert_allclose(hat_y, data_from_r["y_est"], rtol=0, atol=0.008)
    bug_fact = np.sqrt(res_glm_gam.scale) * 0.976
    assert_allclose(se, se_from_mgcv * bug_fact, rtol=0, atol=0.008)


@pytest.mark.thread_unsafe(reason="Uses matplotlib")
@pytest.mark.matplotlib
def test_partial_plot(close_figures):
    cur_dir = Path(__file__).resolve().parent
    file_path = Path(cur_dir).joinpath("results", "prediction_from_mgcv.csv")
    data_from_r = pd.read_csv(file_path)
    x = data_from_r.x.values
    y = data_from_r.y.values
    se_from_mgcv = data_from_r.y_est_se
    df = [10]
    degree = [6]
    bsplines = BSplines(x, degree=degree, df=df)
    alpha = 0.03
    glm_gam = GLMGam(y, smoother=bsplines, alpha=alpha)
    res_glm_gam = glm_gam.fit(maxiter=10000, method="bfgs")
    fig = res_glm_gam.plot_partial(0)
    xp, yp = fig.axes[0].get_children()[0].get_data()
    sort_idx = np.argsort(x)
    hat_y, se = res_glm_gam.partial_values(0)
    assert_allclose(xp, x[sort_idx])
    assert_allclose(yp, hat_y[sort_idx])


def test_cov_params():
    rs = np.random.RandomState(0)
    n = 1000
    x = rs.uniform(0, 1, (n, 2))
    x = x - x.mean()
    y = x[:, 0] * x[:, 0] + rs.normal(0, 0.01, n)
    y -= y.mean()
    bsplines = BSplines(x, degree=[3] * 2, df=[10] * 2, constraints="center")
    alpha = [0, 0]
    glm_gam = GLMGam(y, smoother=bsplines, alpha=alpha)
    res_glm_gam = glm_gam.fit(method="pirls", max_start_irls=0, disp=0, maxiter=5000)
    glm = GLM(y, bsplines.basis)
    res_glm = glm.fit()
    assert_allclose(res_glm.cov_params(), res_glm_gam.cov_params(), rtol=0.0025)
    alpha = 1e-13
    glm_gam = GLMGam(y, smoother=bsplines, alpha=alpha)
    res_glm_gam = glm_gam.fit(method="pirls", max_start_irls=0, disp=0, maxiter=5000)
    assert_allclose(res_glm.cov_params(), res_glm_gam.cov_params(), atol=1e-10)
    res_glm_gam = glm_gam.fit(method="bfgs", max_start_irls=0, disp=0, maxiter=5000)
    assert_allclose(
        res_glm.cov_params(), res_glm_gam.cov_params(), rtol=0.0001, atol=1e-08
    )
