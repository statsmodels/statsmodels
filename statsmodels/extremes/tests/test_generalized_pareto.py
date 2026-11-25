"""
Tests for the generalized Pareto regression

Reference values are computed in the tests from independent code paths:

- ``scipy.stats.genpareto.fit`` with ``floc=0`` for the intercept-only model
  and, per group, for a model whose scale and shape both depend on one
  binary covariate (the likelihood then separates by group);
- a negative loglikelihood written with ``scipy.stats.genpareto.logpdf``,
  minimized with Nelder-Mead and differentiated numerically, for models
  with continuous covariates.

``test_matches_r`` compares with R's ismev and extRemes on a stored dataset;
see ``results/results_gpd_r.py``.
"""

from pathlib import Path
import pickle

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
import pytest
from scipy import optimize, stats

from statsmodels.extremes.api import GeneralizedParetoModel
from statsmodels.extremes.generalized_pareto import (
    _COEF_G,
    _COEF_H,
    _COEF_K,
    _pwm_genpareto,
    _series,
)
from statsmodels.extremes.tests.results import results_gpd_r
from statsmodels.tools.numdiff import approx_fprime, approx_hess
from statsmodels.tools.sm_exceptions import EstimationWarning


def _tight_fmin(func, x0, args=(), disp=0):
    return optimize.fmin(
        func,
        x0,
        args=args,
        # ftol much below 1e-12 is under the resolution of a loglikelihood
        # of order 1e3 and makes fmin run to maxfun with the same result.
        xtol=1e-10,
        ftol=1e-12,
        maxiter=20000,
        maxfun=40000,
        disp=disp,
    )


def _scipy_fit(y):
    """MLE (shape, scale) from scipy with the location fixed at zero."""
    shape, _, scale = stats.genpareto.fit(y, floc=0, optimizer=_tight_fmin)
    return shape, scale


def _scipy_cov_logscale_shape(y, shape, scale):
    """Inverse observed information of (log scale, shape) from scipy logpdf."""

    def llf(p):
        return stats.genpareto.logpdf(y, p[1], scale=np.exp(p[0])).sum()

    hess = approx_hess(np.array([np.log(scale), shape]), llf)
    return np.linalg.inv(-hess)


def _nllf_reference(params, y, x, z):
    """Negative loglikelihood written with scipy, independent of the model."""
    k = x.shape[1]
    scale = np.exp(x @ params[:k])
    shape = z @ params[k:]
    llf = stats.genpareto.logpdf(y, shape, scale=scale)
    if not np.all(np.isfinite(llf)):
        return np.inf
    return -llf.sum()


def _simulate(rng, nobs, params_scale, params_shape, x, z):
    scale = np.exp(x @ params_scale)
    shape = z @ params_shape
    return stats.genpareto.rvs(shape, scale=scale, random_state=rng)


@pytest.mark.parametrize("shape_true", [-0.3, 0.0, 0.2, 0.6])
def test_intercept_only_matches_scipy(shape_true):
    rng = np.random.default_rng(1234)
    y = stats.genpareto.rvs(shape_true, scale=2.0, size=400, random_state=rng)
    res = GeneralizedParetoModel(y).fit()
    shape, scale = _scipy_fit(y)
    assert_allclose(res.params, [np.log(scale), shape], rtol=1e-5, atol=1e-7)
    # atol: the shape can be close to zero
    llf_scipy = stats.genpareto.logpdf(y, shape, scale=scale).sum()
    assert_allclose(res.llf, llf_scipy, rtol=1e-10)
    cov = _scipy_cov_logscale_shape(y, shape, scale)
    assert_allclose(res.bse, np.sqrt(np.diag(cov)), rtol=1e-4)
    assert_array_equal(res.model.exog_names, ["const", "shape"])


def test_binary_covariate_on_scale_and_shape_matches_groupwise_scipy():
    # With exog = exog_shape = [1, d] for a group dummy d, the likelihood
    # separates into one GPD per group, so scipy's fit on each group is an
    # exact reference for all parameters and their standard errors.
    rng = np.random.default_rng(987)
    n0, n1 = 300, 450
    y0 = stats.genpareto.rvs(0.1, scale=1.5, size=n0, random_state=rng)
    y1 = stats.genpareto.rvs(0.35, scale=4.0, size=n1, random_state=rng)
    y = np.concatenate((y0, y1))
    d = np.r_[np.zeros(n0), np.ones(n1)]
    x = np.column_stack((np.ones_like(d), d))
    res = GeneralizedParetoModel(y, x, exog_shape=x).fit()

    shape0, scale0 = _scipy_fit(y0)
    shape1, scale1 = _scipy_fit(y1)
    expected = [
        np.log(scale0),
        np.log(scale1) - np.log(scale0),
        shape0,
        shape1 - shape0,
    ]
    assert_allclose(res.params, expected, rtol=1e-5)

    cov0 = _scipy_cov_logscale_shape(y0, shape0, scale0)
    cov1 = _scipy_cov_logscale_shape(y1, shape1, scale1)
    # params = A @ (log s0, xi0, log s1, xi1)
    a = np.array([[1, 0, 0, 0], [-1, 0, 1, 0], [0, 1, 0, 0], [0, -1, 0, 1]], float)
    cov_groups = np.zeros((4, 4))
    cov_groups[:2, :2] = cov0
    cov_groups[2:, 2:] = cov1
    cov = a @ cov_groups @ a.T
    assert_allclose(res.cov_params(), cov, rtol=1e-4, atol=1e-10)
    # atol: an exact zero is not expected; the covariance of the
    # intercepts across groups is of order 1e-3


@pytest.fixture(scope="module")
def covariate_data():
    rng = np.random.default_rng(20261125)
    nobs = 600
    x1 = rng.normal(size=nobs)
    x2 = rng.uniform(size=nobs)
    x = np.column_stack((np.ones(nobs), x1, x2))
    z = np.column_stack((np.ones(nobs), x2))
    y = _simulate(rng, nobs, np.array([0.3, 0.5, -0.4]), np.array([0.1, 0.15]), x, z)
    return y, x, z


@pytest.mark.parametrize("shape_cov", [False, True])
def test_covariates_match_reference_likelihood(covariate_data, shape_cov):
    y, x, z = covariate_data
    z = z if shape_cov else z[:, :1]
    res = GeneralizedParetoModel(y, x, exog_shape=z).fit()

    start = np.r_[np.log(y.mean()), np.zeros(x.shape[1] - 1 + z.shape[1])]
    ref = optimize.minimize(
        _nllf_reference,
        start,
        args=(y, x, z),
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 50000, "maxfev": 50000},
    )
    assert ref.success
    assert_allclose(res.params, ref.x, rtol=1e-5, atol=1e-7)
    # atol: a coefficient can be close to zero
    assert_allclose(-res.llf, ref.fun, rtol=1e-10)

    hess_ref = -approx_hess(ref.x, _nllf_reference, args=(y, x, z))
    cov_ref = np.linalg.inv(-hess_ref)
    assert_allclose(res.bse, np.sqrt(np.diag(cov_ref)), rtol=1e-4)


@pytest.mark.parametrize(
    "params",
    [
        [0.2, 0.4, -0.3, 0.12, 0.2],
        [0.2, 0.4, -0.3, -0.2, 0.05],
        [0.2, 0.4, -0.3, 1e-10, 0.0],
        [0.2, 0.4, -0.3, -1e-7, 0.0],
        [0.2, 0.4, -0.3, 0.0, 0.0],
    ],
)
def test_score_hessian_against_numerical(covariate_data, params):
    y, x, z = covariate_data
    mod = GeneralizedParetoModel(y, x, exog_shape=z)
    params = np.asarray(params)
    llf_ref = -_nllf_reference(params, y, x, z)
    assert_allclose(mod.loglike(params), llf_ref, rtol=1e-12)

    def llf(p):
        return -_nllf_reference(p, y, x, z)

    score_num = approx_fprime(params, llf, centered=True)
    assert_allclose(mod.score(params), score_num, rtol=1e-6, atol=1e-5)
    # atol: centered differences of a sum of 600 terms
    hess_num = approx_hess(params, llf)
    assert_allclose(mod.hessian(params), hess_num, rtol=1e-5, atol=1e-3)
    # atol: second differences of the loglikelihood

    score_obs = mod.score_obs(params)
    assert_allclose(score_obs.sum(0), mod.score(params), rtol=1e-12)


def test_near_zero_shape_is_continuous():
    # loglike, score and Hessian factors are continuous through shape 0 and
    # agree with the exponential limits there.
    y = np.array([0.01, 0.3, 1.0, 2.5, 7.0, 20.0])
    t = y / 2.0
    exog = np.ones((len(y), 1))
    mod = GeneralizedParetoModel(y, exog)
    for shape in [-1e-12, 0.0, 1e-12]:
        params = np.array([np.log(2.0), shape])
        assert_allclose(
            mod.loglikeobs(params),
            stats.expon.logpdf(y, scale=2.0),
            rtol=1e-11,
        )
        sf_scale, sf_shape = mod.score_factor(params)
        assert_allclose(sf_scale, t - 1, rtol=1e-10)
        assert_allclose(sf_shape, t**2 / 2 - t, rtol=1e-10)
        h11, h12, h22 = mod.hessian_factor(params)
        assert_allclose(h11, -t, rtol=1e-10)
        assert_allclose(h12, -t * (t - 1), rtol=1e-10)
        assert_allclose(h22, t**2 - 2 * t**3 / 3, rtol=1e-10)

    # a shape small enough that w = shape * t stays in the series range,
    # compared with scipy's closed form
    for shape in [1e-4, -1e-4, 0.03]:
        params = np.array([np.log(2.0), shape])
        assert_allclose(
            mod.loglikeobs(params),
            stats.genpareto.logpdf(y, shape, scale=2.0),
            rtol=1e-12,
        )


@pytest.mark.parametrize(
    "coef, closed_form, limit",
    [
        (_COEF_H, lambda w: np.log1p(w) / w, 1.0),
        (_COEF_G, lambda w: (np.log1p(w) - w / (1 + w)) / w**2, 0.5),
        (
            _COEF_K,
            lambda w: (-2 * np.log1p(w) + 2 * w / (1 + w) + w**2 / (1 + w) ** 2) / w**3,
            -2 / 3,
        ),
    ],
)
def test_series_matches_closed_form(coef, closed_form, limit):
    # Near the switch point |w| = 0.1 both forms are accurate, the closed
    # forms to about eps / w**2.
    w = np.array([-0.1, -0.07, -0.05, 0.05, 0.07, 0.1])
    assert_allclose(_series(w, coef), closed_form(w), rtol=1e-12)
    assert_allclose(_series(np.zeros(1), coef), [limit], rtol=1e-15)


def test_out_of_support():
    rng = np.random.default_rng(55)
    y = stats.genpareto.rvs(-0.3, scale=1.0, size=300, random_state=rng)
    mod = GeneralizedParetoModel(y)
    # max(y) is close to the end point 1 / 0.3, shape -0.5 puts it outside
    bad = np.array([0.0, -0.5])
    assert np.isneginf(mod.loglike(bad))
    assert np.isnan(mod.score(bad)).all()
    with pytest.raises(ValueError, match="outside of the support"):
        mod.fit(start_params=bad)
    with pytest.raises(ValueError, match="start_params must have length"):
        mod.fit(start_params=[0.0])
    res = mod.fit(start_params=[0.0, 0.0])
    shape, scale = _scipy_fit(y)
    assert_allclose(res.params, [np.log(scale), shape], rtol=1e-5)


def test_default_start_falls_back_into_support():
    # Uniform data are generalized Pareto with shape -1. The moment estimate
    # of the shape is clipped to -0.45, which puts the added exceedance 2.5
    # outside of the support, so the start uses shape 0 instead.
    rng = np.random.default_rng(0)
    y = np.r_[rng.uniform(size=200), 2.5]
    mod = GeneralizedParetoModel(y)
    shape_pwm, scale_pwm = _pwm_genpareto(y)
    assert 1 - 0.45 * 2.5 / scale_pwm < 0
    sp = mod._start_params()
    assert sp[1] == 0.0
    assert np.isfinite(mod.loglike(sp))
    res = mod.fit()
    shape, scale = _scipy_fit(y)
    assert_allclose(res.params, [np.log(scale), shape], rtol=1e-5)


def test_warns_below_minus_half():
    rng = np.random.default_rng(3)
    y = stats.genpareto.rvs(-0.7, scale=1.0, size=400, random_state=rng)
    with pytest.warns(EstimationWarning, match="below -0.5"):
        res = GeneralizedParetoModel(y).fit()
    assert res.params[1] < -0.5


def test_predict_and_distribution(covariate_data):
    y, x, z = covariate_data
    res = GeneralizedParetoModel(y, x, exog_shape=z).fit()
    params = res.params
    scale = np.exp(x @ params[:3])
    shape = z @ params[3:]
    assert_allclose(res.predict(which="linear"), x @ params[:3], rtol=1e-13)
    assert_allclose(res.predict(which="scale"), scale, rtol=1e-13)
    assert_allclose(res.fitted_scale, scale, rtol=1e-13)
    assert_allclose(res.predict(which="shape"), shape, rtol=1e-13)
    assert_allclose(res.fitted_shape, shape, rtol=1e-13)
    assert_allclose(res.predict(), stats.genpareto.mean(shape, scale=scale), rtol=1e-12)

    x_new = x[:5]
    z_new = z[:5]
    assert_allclose(
        res.predict(exog=x_new, exog_shape=z_new, which="scale"),
        scale[:5],
        rtol=1e-13,
    )
    distr = res.get_distribution(exog=x_new, exog_shape=z_new)
    assert_allclose(distr.logpdf(y[:5]), res.model.loglikeobs(params)[:5], rtol=1e-12)

    # residuals are standard exponential under the model
    resid = res.resid_exponential
    expected = np.log1p(shape * y / scale) / shape
    assert_allclose(resid, expected, rtol=1e-12)

    with pytest.raises(ValueError, match="which = var"):
        res.predict(which="var")


def test_predict_new_exog_constant_shape_and_infinite_mean():
    mod = GeneralizedParetoModel(
        np.array([0.5, 1.0, 2.0, 4.0]), np.column_stack((np.ones(4), np.arange(4)))
    )
    params = np.array([0.1, 0.2, 1.2])
    pred = mod.predict(params, exog=[[1.0, 10.0]], which="scale")
    assert_allclose(pred, [np.exp(0.1 + 2.0)], rtol=1e-13)
    assert np.isinf(mod.predict(params)).all()
    pred_shape = mod.predict(
        params, exog=[[1.0, 10.0]], exog_shape=[1.0], which="shape"
    )
    assert_allclose(pred_shape, [1.2], rtol=1e-13)


def test_pandas_names_and_summary():
    rng = np.random.default_rng(11)
    nobs = 300
    df = pd.DataFrame({"const": np.ones(nobs), "wind": rng.normal(size=nobs)})
    y = pd.Series(
        stats.genpareto.rvs(
            0.2, scale=np.exp(0.2 + 0.4 * df["wind"]), random_state=rng
        ),
        name="loss",
    )
    res = GeneralizedParetoModel(y, df, exog_shape=df[["const"]]).fit()
    assert list(res.params.index) == ["const", "wind", "shape-const"]
    smry = str(res.summary())
    assert "GeneralizedParetoModel" in smry
    assert "shape-const" in smry
    assert "loss" in smry
    assert list(res.bse.index) == ["const", "wind", "shape-const"]


def test_invalid_endog():
    with pytest.raises(ValueError, match="nonnegative"):
        GeneralizedParetoModel(np.array([1.0, -0.5, 2.0]))
    with pytest.raises(ValueError, match="nonnegative"):
        GeneralizedParetoModel(np.array([1.0, np.nan, 2.0]))
    with pytest.raises(ValueError, match="at least one positive"):
        GeneralizedParetoModel(np.zeros(4))
    with pytest.raises(ValueError, match="1-dimensional"):
        GeneralizedParetoModel(np.ones((3, 2)))
    with pytest.raises(ValueError, match="same number of rows"):
        GeneralizedParetoModel(np.ones(4), exog_shape=np.ones(3))


def test_pwm_degenerate_sample():
    assert _pwm_genpareto(np.zeros(5)) == (0.0, 0.0)
    shape, scale = _pwm_genpareto(
        stats.genpareto.rvs(
            0.2, scale=3.0, size=20000, random_state=np.random.default_rng(0)
        )
    )
    assert_allclose([shape, scale], [0.2, 3.0], rtol=0.05)


@pytest.mark.parametrize("fit_kwds", [{"refine": False}, {"method": "newton"}])
def test_fit_without_refinement(fit_kwds):
    # The plain BFGS fit and a pure Newton fit reach the same MLE as the
    # default (BFGS then Newton), to optimizer tolerance.
    rng = np.random.default_rng(12)
    y = stats.genpareto.rvs(0.15, scale=2.0, size=800, random_state=rng)
    mod = GeneralizedParetoModel(y)
    res_default = mod.fit()
    res = mod.fit(**fit_kwds)
    assert res.mle_retvals["converged"]
    assert_allclose(res.params, res_default.params, rtol=1e-3, atol=1e-4)
    assert_allclose(res.llf, res_default.llf, rtol=1e-8)
    assert res.llf <= res_default.llf + 1e-10


def test_init_keys_and_exogs():
    rng = np.random.default_rng(13)
    nobs = 50
    x = np.column_stack([np.ones(nobs), rng.normal(size=nobs)])
    y = stats.genpareto.rvs(0.1, size=nobs, random_state=rng)
    mod = GeneralizedParetoModel(y, x, exog_shape=x)
    assert "exog_shape" in mod._init_keys
    exog, exog_shape = mod._get_exogs()
    assert_allclose(exog, x)
    assert_allclose(exog_shape, x)


def test_remove_data_and_pickle():
    rng = np.random.default_rng(14)
    nobs = 400
    x = np.column_stack([np.ones(nobs), rng.normal(size=nobs)])
    y = stats.genpareto.rvs(0.2, scale=np.exp(0.3 * x[:, 1]), random_state=rng)
    res = GeneralizedParetoModel(y, x, exog_shape=x).fit()
    pred = res.predict(exog=x[:3], exog_shape=x[:3])
    bse = res.bse.copy()

    res.remove_data()
    assert res.model.exog is None
    assert res.model.exog_shape is None
    res2 = pickle.loads(pickle.dumps(res))
    assert_allclose(res2.bse, bse)
    assert_allclose(res2.predict(exog=x[:3], exog_shape=x[:3]), pred)


@pytest.mark.parametrize(
    "ref_name, shape_cov, rtol_params",
    [
        ("ismev_shape_const", False, 1e-4),
        ("ismev_shape_x2", True, 1e-4),
        # fevd's optimizer stops earlier (its nllh is 1.5e-7 higher)
        ("extremes_shape_x2", True, 1e-3),
    ],
)
def test_matches_r(ref_name, shape_cov, rtol_params):
    # Reference values from R ismev::gpd.fit and extRemes::fevd, see
    # results/results_gpd_r.py for the R code.
    ref = getattr(results_gpd_r, ref_name)
    path = Path(__file__).resolve().parent / "results" / "gpd_covariates.csv"
    data = pd.read_csv(path)
    nobs = len(data)
    x = np.column_stack((np.ones(nobs), data["x1"], data["x2"]))
    z = np.column_stack((np.ones(nobs), data["x2"]))
    z = z if shape_cov else z[:, :1]
    res = GeneralizedParetoModel(data["y"].to_numpy(), x, exog_shape=z).fit()
    assert_allclose(-res.llf, ref.nllh, rtol=1e-9)
    # R's optimizers stop slightly short of the maximum, never beyond it
    assert -res.llf <= ref.nllh + 1e-9
    assert_allclose(res.params, ref.params, rtol=rtol_params)
    # R's SEs come from optim's finite-difference Hessian
    assert_allclose(res.bse, ref.bse, rtol=1e-3)
