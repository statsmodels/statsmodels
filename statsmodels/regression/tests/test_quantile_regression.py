"""Tests for quantile regression models."""

from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pandas as pd
import pytest
import scipy.optimize
import scipy.stats

import statsmodels.api as sm
from statsmodels.formula._manager import FormulaManager
from statsmodels.iolib.summary import Summary
from statsmodels.regression.quantile_regression import (
    QuantReg,
    _max_step,
    _normal_solver,
)
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    IterationLimitWarning,
)

from .results.results_quantile_regression import (
    Rquantreg,
    biweight_bofinger,
    biweight_chamberlain,
    biweight_hsheather,
    cosine_bofinger,
    cosine_chamberlain,
    cosine_hsheather,
    epan2_bofinger,
    epan2_chamberlain,
    epan2_hsheather,
    epanechnikov_hsheather_q75,
    gaussian_bofinger,
    gaussian_chamberlain,
    gaussian_hsheather,
    parzen_bofinger,
    parzen_chamberlain,
    parzen_hsheather,
)

idx = ["income", "Intercept"]


class CheckModelResultsMixin:
    def test_params(self):
        assert_allclose(
            np.ravel(self.res1.params.loc[idx]), self.res2.table[:, 0], rtol=1e-3
        )

    def test_bse(self):
        assert_equal(self.res1.scale, 1)
        assert_allclose(
            np.ravel(self.res1.bse.loc[idx]), self.res2.table[:, 1], rtol=1e-3
        )

    def test_tvalues(self):
        assert_allclose(
            np.ravel(self.res1.tvalues.loc[idx]), self.res2.table[:, 2], rtol=1e-2
        )

    def test_pvalues(self):
        pvals_stata = scipy.stats.t.sf(self.res2.table[:, 2], self.res2.df_r)
        assert_allclose(np.ravel(self.res1.pvalues.loc[idx]), pvals_stata, rtol=1.1)

        # test that we use the t distribution for the p-values
        pvals_t = scipy.stats.t.sf(self.res1.tvalues, self.res2.df_r) * 2
        assert_allclose(np.ravel(self.res1.pvalues), pvals_t, rtol=1e-9, atol=1e-10)

    def test_conf_int(self):
        assert_allclose(
            self.res1.conf_int().loc[idx], self.res2.table[:, -2:], rtol=1e-3
        )

    def test_nobs(self):
        assert_allclose(self.res1.nobs, self.res2.N, rtol=1e-3)

    def test_df_model(self):
        assert_allclose(self.res1.df_model, self.res2.df_m, rtol=1e-3)

    def test_df_resid(self):
        assert_allclose(self.res1.df_resid, self.res2.df_r, rtol=1e-3)

    def test_prsquared(self):
        assert_allclose(self.res1.prsquared, self.res2.psrsquared, rtol=1e-3)

    def test_sparsity(self):
        assert_allclose(np.array(self.res1.sparsity), self.res2.sparsity, rtol=1e-3)

    def test_bandwidth(self):
        assert_allclose(np.array(self.res1.bandwidth), self.res2.kbwidth, rtol=1e-3)


d = {
    ("biw", "bofinger"): biweight_bofinger,
    ("biw", "chamberlain"): biweight_chamberlain,
    ("biw", "hsheather"): biweight_hsheather,
    ("cos", "bofinger"): cosine_bofinger,
    ("cos", "chamberlain"): cosine_chamberlain,
    ("cos", "hsheather"): cosine_hsheather,
    ("gau", "bofinger"): gaussian_bofinger,
    ("gau", "chamberlain"): gaussian_chamberlain,
    ("gau", "hsheather"): gaussian_hsheather,
    ("par", "bofinger"): parzen_bofinger,
    ("par", "chamberlain"): parzen_chamberlain,
    ("par", "hsheather"): parzen_hsheather,
    # ('rec','bofinger'): rectangle_bofinger,
    # ('rec','chamberlain'): rectangle_chamberlain,
    # ('rec','hsheather'): rectangle_hsheather,
    # ('tri','bofinger'): triangle_bofinger,
    # ('tri','chamberlain'): triangle_chamberlain,
    # ('tri','hsheather'): triangle_hsheather,
    ("epa", "bofinger"): epan2_bofinger,
    ("epa", "chamberlain"): epan2_chamberlain,
    ("epa", "hsheather"): epan2_hsheather,
    # ('epa2', 'bofinger'): epan2_bofinger,
    # ('epa2', 'chamberlain'): epan2_chamberlain,
    # ('epa2', 'hsheather'): epan2_hsheather
}


def setup_fun(kernel="gau", bandwidth="bofinger"):
    data = sm.datasets.engel.load_pandas().data

    mgr = FormulaManager()
    y, X = mgr.get_matrices("foodexp ~ income", data)
    statsm = QuantReg(y, X).fit(vcov="iid", kernel=kernel, bandwidth=bandwidth)
    stata = d[(kernel, bandwidth)]
    return statsm, stata


def test_fitted_residuals():
    data = sm.datasets.engel.load_pandas().data

    mgr = FormulaManager()
    y, X = mgr.get_matrices("foodexp ~ income", data)

    res = QuantReg(y, X).fit(q=0.1)
    # Note: maxabs relative error with fitted is 1.789e-09
    assert_almost_equal(np.array(res.fittedvalues), Rquantreg.fittedvalues, 5)
    assert_almost_equal(np.array(res.predict()), Rquantreg.fittedvalues, 5)
    assert_almost_equal(np.array(res.resid), Rquantreg.residuals, 5)


class TestEpanechnikovHsheatherQ75(CheckModelResultsMixin):
    # Vincent Arel-Bundock also spot-checked q=.1
    @classmethod
    def setup_class(cls):
        data = sm.datasets.engel.load_pandas().data
        mgr = FormulaManager()
        y, X = mgr.get_matrices("foodexp ~ income", data)
        cls.res1 = QuantReg(y, X).fit(
            q=0.75, vcov="iid", kernel="epa", bandwidth="hsheather"
        )
        cls.res2 = epanechnikov_hsheather_q75


class TestEpanechnikovBofinger(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        cls.res1, cls.res2 = setup_fun("epa", "bofinger")


class TestEpanechnikovChamberlain(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        cls.res1, cls.res2 = setup_fun("epa", "chamberlain")


class TestEpanechnikovHsheather(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        cls.res1, cls.res2 = setup_fun("epa", "hsheather")


class TestGaussianBofinger(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        cls.res1, cls.res2 = setup_fun("gau", "bofinger")


class TestGaussianChamberlain(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        cls.res1, cls.res2 = setup_fun("gau", "chamberlain")


class TestGaussianHsheather(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        cls.res1, cls.res2 = setup_fun("gau", "hsheather")


class TestBiweightBofinger(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        cls.res1, cls.res2 = setup_fun("biw", "bofinger")


class TestBiweightChamberlain(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        cls.res1, cls.res2 = setup_fun("biw", "chamberlain")


class TestBiweightHsheather(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        cls.res1, cls.res2 = setup_fun("biw", "hsheather")


class TestCosineBofinger(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        cls.res1, cls.res2 = setup_fun("cos", "bofinger")


class TestCosineChamberlain(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        cls.res1, cls.res2 = setup_fun("cos", "chamberlain")


class TestCosineHsheather(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        cls.res1, cls.res2 = setup_fun("cos", "hsheather")


class TestParzeneBofinger(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        cls.res1, cls.res2 = setup_fun("par", "bofinger")


class TestParzeneChamberlain(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        cls.res1, cls.res2 = setup_fun("par", "chamberlain")


class TestParzeneHsheather(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        cls.res1, cls.res2 = setup_fun("par", "hsheather")


# class TestTriangleBofinger(CheckModelResultsMixin):
#    @classmethod
#    def setup_class(cls):
#        cls.res1, cls.res2 = setup_fun('tri', 'bofinger')

# class TestTriangleChamberlain(CheckModelResultsMixin):
#    @classmethod
#    def setup_class(cls):
#        cls.res1, cls.res2 = setup_fun('tri', 'chamberlain')

# class TestTriangleHsheather(CheckModelResultsMixin):
#    @classmethod
#    def setup_class(cls):
#        cls.res1, cls.res2 = setup_fun('tri', 'hsheather')


def test_zero_resid():
    # smoke and regression tests

    X = np.array([[1, 0], [0, 1], [0, 2.1], [0, 3.1]], dtype=np.float64)
    y = np.array([0, 1, 2, 3], dtype=np.float64)

    res = QuantReg(y, X).fit(0.5, bandwidth="chamberlain")  # 'bofinger')
    res.summary()

    assert_allclose(res.params, np.array([0.0, 0.96774163]), rtol=1e-4, atol=1e-20)
    assert_allclose(res.bse, np.array([0.0447576, 0.01154867]), rtol=1e-4, atol=1e-20)
    assert_allclose(
        res.resid,
        np.array([0.0, 3.22583680e-02, -3.22574272e-02, 9.40732912e-07]),
        rtol=1e-4,
        atol=1e-20,
    )

    X = np.array([[1, 0], [0.1, 1], [0, 2.1], [0, 3.1]], dtype=np.float64)
    y = np.array([0, 1, 2, 3], dtype=np.float64)

    res = QuantReg(y, X).fit(0.5, bandwidth="chamberlain")
    res.summary()

    assert_allclose(
        res.params, np.array([9.99982796e-08, 9.67741630e-01]), rtol=1e-4, atol=1e-20
    )
    assert_allclose(res.bse, np.array([0.04455029, 0.01155251]), rtol=1e-4, atol=1e-20)
    assert_allclose(
        res.resid,
        np.array([-9.99982796e-08, 3.22583598e-02, -3.22574234e-02, 9.46361860e-07]),
        rtol=1e-4,
        atol=1e-20,
    )


def test_use_t_summary():
    X = np.array([[1, 0], [0, 1], [0, 2.1], [0, 3.1]], dtype=np.float64)
    y = np.array([0, 1, 2, 3], dtype=np.float64)

    res = QuantReg(y, X).fit(0.5, bandwidth="chamberlain", use_t=True)
    summ = res.summary()
    assert "P>|t|" in str(summ)
    assert "P>|z|" not in str(summ)


def test_alpha_summary():
    X = np.array([[1, 0], [0, 1], [0, 2.1], [0, 3.1]], dtype=np.float64)
    y = np.array([0, 1, 2, 3], dtype=np.float64)

    res = QuantReg(y, X).fit(0.5, bandwidth="chamberlain", use_t=True)
    summ_20 = res.summary(alpha=0.2)
    assert "[0.025      0.975]" not in str(summ_20)
    assert "[0.1        0.9]" in str(summ_20)


def test_remove_data():
    X = np.array([[1, 0], [0, 1], [0, 2.1], [0, 3.1]], dtype=np.float64)
    y = np.array([0, 1, 2, 3], dtype=np.float64)

    res = QuantReg(y, X).fit(0.5)
    res.remove_data()


def test_collinear_matrix():
    X = np.array(
        [[1, 0, 0.5], [1, 0, 0.8], [1, 0, 1.5], [1, 0, 0.25]], dtype=np.float64
    )
    y = np.array([0, 1, 2, 3], dtype=np.float64)

    res_collinear = QuantReg(y, X).fit(0.5)
    assert len(res_collinear.params) == X.shape[1]


def test_nontrivial_singular_matrix():
    rs = np.random.RandomState(2382109)
    x_one = rs.random(1000)
    x_two = rs.random(1000) * 10
    x_three = rs.random(1000)
    intercept = np.ones(1000)

    y = rs.random(1000) * 5
    X = np.column_stack((intercept, x_one, x_two, x_three, x_one))

    assert np.linalg.matrix_rank(X) < X.shape[1]
    res_singular = QuantReg(y, X).fit(0.5)
    assert len(res_singular.params) == X.shape[1]
    assert np.linalg.matrix_rank(res_singular.cov_params()) == X.shape[1] - 1

    # prediction is correct even with singular exog
    res_ns = QuantReg(y, X[:, :-1]).fit(0.5)
    assert_allclose(res_singular.predict(X), res_ns.predict(X[:, :-1]), rtol=0.01)


def test_summary_after_remove_data():
    # summary() must still work after remove_data() has been called
    data = sm.datasets.engel.load_pandas().data
    mgr = FormulaManager()
    y, X = mgr.get_matrices("foodexp ~ income", data)
    res = QuantReg(y, X).fit(q=0.75, vcov="iid", kernel="epa", bandwidth="hsheather")

    assert isinstance(res.summary(), Summary)
    res.remove_data()
    assert isinstance(res.summary(), Summary)


def test_scale_is_a_property():
    # GH: QuantRegResults.scale had its @cache_readonly commented out, so
    # it was a plain method: `res.scale` returned a bound method instead
    # of a float.
    X = np.array([[1, 0], [0, 1], [0, 2.1], [0, 3.1]], dtype=np.float64)
    y = np.array([0, 1, 2, 3], dtype=np.float64)
    res = QuantReg(y, X).fit(0.5, bandwidth="chamberlain")
    assert res.scale == 1.0


def test_ols_criteria_are_nan_not_ols_values():
    # QuantRegResults deliberately overrides the OLS-specific information
    # criteria/goodness-of-fit measures with nan, since they don't apply
    # to a pinball-loss fit -- guard against one of the overrides being
    # dropped and silently falling through to RegressionResults' formulas.
    rs = np.random.RandomState(918273)
    n = 60
    X = sm.add_constant(rs.standard_normal((n, 2)))
    y = X @ [1.0, 0.5, -0.5] + rs.standard_normal(n)
    res = QuantReg(y, X).fit(0.5)

    for name in (
            "bic",
            "aic",
            "llf",
            "rsquared",
            "rsquared_adj",
            "mse",
            "mse_model",
            "mse_total",
            "centered_tss",
            "uncentered_tss"
    ):
        assert np.isnan(getattr(res, name)), name


def test_prsquared_improves_with_informative_regressors():
    rs = np.random.RandomState(19)
    n = 200
    x = rs.standard_normal(n)
    y = 2.0 * x + rs.standard_normal(n) * 0.5

    exog_null = sm.add_constant(np.zeros(n))[:, :1]
    res_null = QuantReg(y, exog_null).fit(0.5)
    # an intercept-only fit reproduces the unconditional quantile, so
    # weighted absolute deviations match the "null" comparator exactly
    assert_allclose(res_null.prsquared, 0.0, atol=1e-10)

    exog_full = sm.add_constant(x)
    res_full = QuantReg(y, exog_full).fit(0.5)
    assert 0 < res_full.prsquared <= 1
    assert res_full.prsquared > res_null.prsquared


def test_fit_invalid_options_raise():
    X = np.array([[1, 0], [0, 1], [0, 2.1], [0, 3.1]], dtype=np.float64)
    y = np.array([0, 1, 2, 3], dtype=np.float64)
    mod = QuantReg(y, X)

    with pytest.raises(ValueError, match="kernel"):
        mod.fit(0.5, kernel="not-a-kernel")
    with pytest.raises(ValueError, match="bandwidth"):
        mod.fit(0.5, bandwidth="not-a-bandwidth")
    with pytest.raises(ValueError, match="vcov"):
        mod.fit(0.5, vcov="not-a-vcov")

    # case is not folded: this file's existing validation never was, and
    # string_like(..., lower=False) preserves that
    with pytest.raises(ValueError, match="vcov"):
        mod.fit(0.5, vcov="ROBUST")


# Tests of the Frisch-Newton interior point solver, method="fn"

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _load_fn_data(name):
    if name == "engel":
        data = sm.datasets.engel.load_pandas().data
        return data["foodexp"].to_numpy(), sm.add_constant(data["income"].to_numpy())
    data = pd.read_csv(RESULTS_DIR / "quantreg_sim.csv")
    exog = sm.add_constant(data[["x1", "x2", "x3", "x4", "x5"]].to_numpy())
    return data["y"].to_numpy(), exog


def _r_fn_results(name, method, q):
    # Computed by results/results_quantreg_fn.R with rq of R quantreg 6.1
    res = pd.read_csv(RESULTS_DIR / "results_quantreg_fn.csv")
    res = res[(res["dataset"] == name) & (res["method"] == method)]
    res = res[np.isclose(res["q"], q)].sort_values("index")
    return res["value"].to_numpy(), res["objective"].iloc[0]


def _check_loss(resid, q):
    return np.sum(resid * (q - (resid < 0)))


FN_QUANTILES = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]


@pytest.mark.parametrize("name", ["engel", "sim"])
@pytest.mark.parametrize("q", FN_QUANTILES)
def test_fn_matches_r(name, q):
    endog, exog = _load_fn_data(name)
    res = QuantReg(endog, exog).fit(q, method="fn")
    params_fn, objective_fn = _r_fn_results(name, "fn", q)
    params_br, objective_br = _r_fn_results(name, "br", q)
    # R uses the same algorithm and the same tolerance on the duality gap,
    # the iterates agree to rounding error.
    assert_allclose(res.params, params_fn, rtol=1e-8)
    # The exact simplex solution differs from both interior point solutions
    # by the remaining duality gap.
    assert_allclose(res.params, params_br, rtol=1e-6)
    # the duality gap, below p_tol=1e-6, bounds the excess objective
    assert_allclose(_check_loss(res.resid, q), objective_br, rtol=0, atol=1e-6)
    assert res.history["gap"][-1] < 1e-6
    assert res.iterations < 50


@pytest.mark.parametrize("q", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_fn_matches_irls(q):
    endog, exog = _load_fn_data("sim")
    mod = QuantReg(endog, exog)
    res_irls = mod.fit(q)
    res_fn = mod.fit(q, method="fn")
    # IRLS stops when the parameters change by less than p_tol=1e-6, which
    # can leave it a few 1e-3 away from the exact solution, depending on the
    # platform. The comparison with R above checks the interior point solution.
    assert_allclose(res_fn.params, res_irls.params, rtol=1e-2, atol=1e-2)
    # The iid covariance depends on the residuals only through the kernel
    # density estimate. The robust covariance also depends on the signs of
    # the residuals that are zero at the solution, which are rounding noise.
    res_irls = mod.fit(q, vcov="iid")
    res_fn = mod.fit(q, vcov="iid", method="fn")
    assert_allclose(res_fn.bse, res_irls.bse, rtol=1e-3)
    # fn is at least as good as irls in terms of the objective
    assert _check_loss(res_fn.resid, q) <= _check_loss(res_irls.resid, q)


class TestEpanechnikovHsheatherFn(CheckModelResultsMixin):
    # Stata results, the interior point solution is the same vertex
    @classmethod
    def setup_class(cls):
        data = sm.datasets.engel.load_pandas().data
        mgr = FormulaManager()
        y, X = mgr.get_matrices("foodexp ~ income", data)
        cls.res1 = QuantReg(y, X).fit(
            vcov="iid", kernel="epa", bandwidth="hsheather", method="fn"
        )
        cls.res2 = epan2_hsheather


def _linprog_objective(endog, exog, q):
    # Primal linear program min q 1'u + (1 - q) 1'v, exog b + u - v = endog,
    # solved by HiGHS as an independent check of the optimal value.
    nobs, k = exog.shape
    c = np.concatenate([np.zeros(k), q * np.ones(nobs), (1 - q) * np.ones(nobs)])
    a_eq = np.hstack([exog, np.eye(nobs), -np.eye(nobs)])
    bounds = [(None, None)] * k + [(0, None)] * (2 * nobs)
    res = scipy.optimize.linprog(c, A_eq=a_eq, b_eq=endog, bounds=bounds)
    assert res.status == 0
    return res.fun


@pytest.mark.parametrize("q", [0.001, 0.01, 0.99, 0.999])
def test_fn_extreme_quantiles(q):
    endog, exog = _load_fn_data("engel")
    res = QuantReg(endog, exog).fit(q, method="fn")
    assert_allclose(
        _check_loss(res.resid, q), _linprog_objective(endog, exog, q), rtol=1e-8
    )


@pytest.mark.parametrize("q", [0.25, 0.5])
def test_fn_ties(q):
    # integer response and binary regressors give many tied residuals
    rng = np.random.default_rng(0)
    exog = np.column_stack([np.ones(300), rng.integers(0, 2, size=(300, 2))])
    endog = np.round(exog @ [1.0, 2.0, -1.0] + rng.standard_t(3, size=300))
    res = QuantReg(endog, exog).fit(q, method="fn")
    assert_allclose(
        _check_loss(res.resid, q), _linprog_objective(endog, exog, q), rtol=1e-8
    )


@pytest.mark.parametrize("q", [0.1, 0.5])
@pytest.mark.parametrize("scale", [1e-10, 1e-3, 1e6])
def test_fn_scale_equivariance(scale, q):
    # The quantile regression estimator is equivariant to scaling endog. For
    # endog far below one the absolute default p_tol would already hold at
    # the least squares start.
    endog, exog = _load_fn_data("engel")
    res = QuantReg(endog, exog).fit(q, method="fn")
    res_scaled = QuantReg(scale * endog, exog).fit(q, method="fn")
    assert res_scaled.iterations > 0
    assert_allclose(res_scaled.params, scale * res.params, rtol=1e-6)


@pytest.mark.filterwarnings(
    "ignore::statsmodels.tools.sm_exceptions.ConvergenceWarning"
)
def test_fn_nearly_collinear():
    # Ill conditioned but full rank design, the normal equations lose
    # positive definiteness near the solution. The duality gap may stop short
    # of p_tol, which warns.
    rng = np.random.default_rng(1)
    nobs, k, q = 1000, 10, 0.001
    exog = np.column_stack([np.ones(nobs), rng.standard_normal((nobs, k - 1))])
    exog[:, 2] = exog[:, 1] + 1e-7 * rng.standard_normal(nobs)
    endog = exog @ rng.standard_normal(k) + rng.standard_t(3, nobs)
    assert np.linalg.matrix_rank(exog) == k
    res = QuantReg(endog, exog).fit(q, method="fn")
    assert_allclose(
        _check_loss(res.resid, q), _linprog_objective(endog, exog, q), rtol=1e-7
    )


def test_fn_unattainable_tolerance():
    endog, exog = _load_fn_data("engel")
    with pytest.warns(ConvergenceWarning, match="duality gap"):
        res = QuantReg(endog, exog).fit(0.5, method="fn", p_tol=0)
    assert np.all(np.isfinite(res.params))
    assert_allclose(
        _check_loss(res.resid, 0.5),
        _linprog_objective(endog, exog, 0.5),
        rtol=1e-10,
    )


def test_fn_nonunique_solution():
    # every value in [5, 6] is a median of 1, ..., 10
    endog = np.arange(1.0, 11.0)
    exog = np.ones((10, 1))
    res = QuantReg(endog, exog).fit(0.5, method="fn")
    assert 5 <= res.params[0] <= 6
    # sum of |y - 5.5| / 2 = 12.5 for every median
    assert_allclose(_check_loss(res.resid, 0.5), 12.5, rtol=1e-10)


def test_fn_exact_fit():
    # All least squares residuals are zero, the starting point has to move
    # the dual slacks away from zero.
    exog = sm.add_constant(np.arange(10.0))
    endog = exog @ np.array([1.0, 2.0])
    res = QuantReg(endog, exog).fit(0.3, method="fn")
    assert np.all(np.isfinite(res.params))
    assert_allclose(res.params, [1.0, 2.0], rtol=1e-6)


def test_fn_collinear():
    rs = np.random.RandomState(2382109)
    x_one = rs.random(1000)
    x_two = rs.random(1000) * 10
    x_three = rs.random(1000)
    intercept = np.ones(1000)
    y = rs.random(1000) * 5
    X = np.column_stack((intercept, x_one, x_two, x_three, x_one))

    res = QuantReg(y, X).fit(0.5, method="fn")
    res_ns = QuantReg(y, X[:, :-1]).fit(0.5, method="fn")
    res_irls = QuantReg(y, X).fit(0.5)
    assert_allclose(res.fittedvalues, res_ns.fittedvalues, rtol=1e-10)
    # minimum norm solution: the duplicated column splits the coefficient
    assert_allclose(res.params[1], res.params[4], rtol=1e-10)
    assert_allclose(res.params[1] + res.params[4], res_ns.params[1], rtol=1e-8)
    # the same convention as the pseudoinverse in irls
    assert_allclose(res.params, res_irls.params, rtol=1e-3)
    assert np.linalg.matrix_rank(res.cov_params()) == X.shape[1] - 1
    assert len(res.history["params"][-1]) == X.shape[1]


def test_fn_pandas_formula():
    data = sm.datasets.engel.load_pandas().data
    res_formula = sm.formula.quantreg("foodexp ~ income", data).fit(0.25, method="fn")
    res_pandas = QuantReg(data["foodexp"], sm.add_constant(data["income"])).fit(
        0.25, method="fn"
    )
    assert list(res_formula.params.index) == ["Intercept", "income"]
    assert list(res_pandas.params.index) == ["const", "income"]
    assert_allclose(res_formula.params.to_numpy(), res_pandas.params.to_numpy())
    params_r, _ = _r_fn_results("engel", "fn", 0.25)
    assert_allclose(res_formula.params.to_numpy(), params_r, rtol=1e-8)
    smry = res_formula.summary()
    assert "income" in str(smry)
    assert isinstance(res_formula.conf_int(), pd.DataFrame)


def test_fn_iteration_limit():
    endog, exog = _load_fn_data("sim")
    with pytest.warns(IterationLimitWarning):
        res = QuantReg(endog, exog).fit(0.5, method="fn", max_iter=2)
    assert res.iterations == 2
    assert len(res.history["gap"]) == 2


def test_fit_invalid_method_raises():
    endog, exog = _load_fn_data("engel")
    with pytest.raises(ValueError, match="method"):
        QuantReg(endog, exog).fit(0.5, method="br")


@pytest.mark.parametrize("weight_range", [0, 30])
def test_normal_solver(weight_range):
    # Weights spanning 30 orders of magnitude make exog' W exog too ill
    # conditioned for the Cholesky solve, the weighted least squares solve
    # gives the same solution.
    rng = np.random.default_rng(0)
    exog = rng.standard_normal((20, 3))
    weights = 10.0 ** rng.uniform(-weight_range / 2, weight_range / 2, 20)
    rhs = rng.standard_normal(20)
    sol = _normal_solver(exog * weights[:, None], exog, weights)(rhs)
    sqrt_w = np.sqrt(weights)
    expected = np.linalg.lstsq(exog * sqrt_w[:, None], sqrt_w * rhs, rcond=None)[0]
    assert_allclose(sol, expected, rtol=1e-8)


def test_normal_solver_not_finite():
    exog = np.ones((3, 1))
    weights = np.array([1.0, np.inf, 1.0])
    assert _normal_solver(exog * weights[:, None], exog, weights) is None


def test_max_step():
    v = np.array([1.0, 2.0, 3.0])
    assert _max_step(v, np.array([1.0, 0.0, 2.0])) == np.inf
    assert_allclose(_max_step(v, np.array([-2.0, -1.0, 1.0])), 0.5)
    # a variable at the boundary blocks a step in a decreasing direction
    assert _max_step(np.array([0.0, 1.0]), np.array([-1.0, -1.0])) == 0
    assert _max_step(np.array([0.0, 1.0]), np.array([0.0, -2.0])) == 0.5
