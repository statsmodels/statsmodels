"""
Test SVAR estimation
"""

from statsmodels.compat.platform import PLATFORM_WIN

from contextlib import nullcontext

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pandas as pd
import pytest
from scipy import stats

import statsmodels.datasets.macrodata
from statsmodels.tools.sm_exceptions import ConvergenceWarning, MissingDataError
from statsmodels.tools.testing import warns_or_none
from statsmodels.tsa.vector_ar.hypothesis_test_results import ErrorBand
from statsmodels.tsa.vector_ar.svar_model import SVAR, SVARProcess
from statsmodels.tsa.vector_ar.var_model import VAR

from .results import results_svar_vars

DECIMAL_6 = 6
DECIMAL_5 = 5
DECIMAL_4 = 4


class TestSVAR:
    @classmethod
    def setup_class(cls):
        mdata = statsmodels.datasets.macrodata.load_pandas().data
        mdata = mdata[["realgdp", "realcons", "realinv"]]
        data = mdata.values
        data = np.diff(np.log(data), axis=0)
        A = np.asarray([[1, 0, 0], ["E", 1, 0], ["E", "E", 1]], dtype="U")
        B = np.asarray([["E", 0, 0], [0, "E", 0], [0, 0, "E"]], dtype="U")
        results = SVAR(data, svar_type="AB", A=A, B=B).fit(maxlags=3)
        cls.res1 = results
        # cls.res2 = results_svar.SVARdataResults()
        from .results import results_svar_st

        cls.res2 = results_svar_st.results_svar1_small

    def _reformat(self, x):
        return x[[1, 4, 7, 2, 5, 8, 3, 6, 9, 0], :].ravel("F")

    def test_A(self):
        assert_almost_equal(self.res1.A, self.res2.A, DECIMAL_4)

    def test_B(self):
        # see issue #3148, adding np.abs to make solution positive
        # general case will need positive sqrt of covariance matrix
        assert_almost_equal(np.abs(self.res1.B), self.res2.B, DECIMAL_4)

    def test_basic(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(self._reformat(res1.params), res2.b_var, atol=1e-12)
        bse_st = np.sqrt(np.diag(res2.V_var))
        assert_allclose(self._reformat(res1.bse), bse_st, atol=1e-12)

    def test_llf_ic(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1.llf, res2.ll_var, atol=1e-12)
        # different definition, missing constant term ?
        corr_const = -8.51363119922803
        assert_allclose(res1.fpe, res2.fpe_var, atol=1e-12)
        assert_allclose(res1.aic - corr_const, res2.aic_var, atol=1e-12)
        assert_allclose(res1.bic - corr_const, res2.sbic_var, atol=1e-12)
        assert_allclose(res1.hqic - corr_const, res2.hqic_var, atol=1e-12)

    @pytest.mark.smoke
    def test_irf(self):
        # mostly SMOKE, API test
        # this only checks that the methods work and produce the same result
        res1 = self.res1
        # the ML fits of the replications may not converge, depending on the platform
        with (
            pytest.warns(FutureWarning),
            warns_or_none(ConvergenceWarning, match="Maximum Likelihood optim"),
        ):
            errband1 = res1.sirf_errband_mc(
                orth=False,
                repl=50,
                steps=10,
                signif=0.05,
                rng=987123,
                burn=100,
                cum=False,
            )
        assert isinstance(errband1, ErrorBand)

        irf = res1.irf()
        # the ML fits of the replications may not converge, depending on the platform
        with (
            pytest.warns(FutureWarning),
            warns_or_none(ConvergenceWarning, match="Maximum Likelihood optim"),
        ):
            errband2 = irf.errband_mc(
                orth=False, svar=True, repl=50, signif=0.05, rng=987123, burn=100
            )
        assert isinstance(errband2, ErrorBand)
        # Windows precision limits require non-zero atol
        atol = 1e-6 if PLATFORM_WIN else 1e-8
        assert_allclose(errband1.lower, errband2.lower, rtol=1e-8, atol=atol)
        assert_allclose(errband1.upper, errband2.upper, rtol=1e-8, atol=atol)

    def test_irf_var_order_not_implemented(self):
        # var_order is not supported and must not be silently ignored,
        # matching VARResults.irf
        with pytest.raises(NotImplementedError, match="variable order"):
            self.res1.irf(var_order=[2, 0, 1])

    @pytest.mark.parametrize(
        "rng",
        [0, np.random.RandomState(0), np.random.default_rng(0)],
        ids=["int", "randomstate", "generator"],
    )
    def test_sirf_errband_mc_replications_differ(self, rng):
        # GH: sirf_errband_mc's internal irf_resim-equivalent loop
        # re-derived a fresh generator from the raw seed on every Monte
        # Carlo iteration, so every replication was identical for an int
        # seed (or a freshly-instantiated RandomState/Generator). With
        # degenerate (identical) replications the resulting error band
        # collapses to lower == upper.
        warn_cm = pytest.warns(FutureWarning) if isinstance(rng, int) else nullcontext()
        # the ML fits of the replications may not converge, depending on the platform
        with (
            warn_cm,
            warns_or_none(ConvergenceWarning, match="Maximum Likelihood optim"),
        ):
            errband = self.res1.sirf_errband_mc(
                orth=False, repl=5, steps=3, signif=0.5, rng=rng, burn=10, cum=False
            )
        assert not np.allclose(errband.lower, errband.upper)

    def test_sirf_errband_mc_one_convergence_warning(self, monkeypatch):
        # The fit of each replication warned if it did not converge, which
        # gave one warning per replication.
        fit = SVAR.fit

        def fit_one_iteration(self, **kwds):
            return fit(self, maxiter=1, **kwds)

        monkeypatch.setattr(SVAR, "fit", fit_one_iteration)
        rng = np.random.default_rng(0)
        with pytest.warns(ConvergenceWarning, match="5 of 5 Monte Carlo") as record:
            self.res1.sirf_errband_mc(repl=5, steps=3, rng=rng, burn=10)
        assert len(record) == 1

    def test_sirf_errband_mc_reproducible_with_int_seed(self):
        # the ML fits of the replications may not converge, depending on the platform
        with (
            pytest.warns(FutureWarning),
            warns_or_none(ConvergenceWarning, match="Maximum Likelihood optim"),
        ):
            errband1 = self.res1.sirf_errband_mc(
                orth=False, repl=5, steps=3, signif=0.5, rng=0, burn=10, cum=False
            )
        # the ML fits of the replications may not converge, depending on the platform
        with (
            pytest.warns(FutureWarning),
            warns_or_none(ConvergenceWarning, match="Maximum Likelihood optim"),
        ):
            errband2 = self.res1.sirf_errband_mc(
                orth=False, repl=5, steps=3, signif=0.5, rng=0, burn=10, cum=False
            )
        assert_allclose(errband1.lower, errband2.lower)
        assert_allclose(errband1.upper, errband2.upper)


def test_oneparam():
    # regression test, one parameter in A, B, issue #9302
    rs = np.random.RandomState(873562)
    lags = 2
    nobs = 200
    y = rs.randn(nobs, 3)
    y[1:] += 0.5 * y[:-1]
    A = np.asarray([[1, "E"], [0, 1.0]])
    # A_guess = np.asarray([[1, 0.2], [0, 1.]])
    B = np.eye(2, dtype=object)

    k = 2
    model = SVAR(y[:, :k], svar_type="AB", A=A, B=B)
    model.k_exog_user = 0
    results = model.fit(maxlags=lags, solver="bfgs")  # "newton")
    results.k_exog_user = 0
    results.summary()

    # regression number
    assert_allclose(results.A[0, 1], -0.075818, atol=1e-5)

    results = model.fit(maxlags=lags, solver="newton")
    results.k_exog_user = 0
    results.summary()

    # regression number
    assert_allclose(results.A[0, 1], -0.075818, atol=1e-5)


def test_invalid_svar_type_raises():
    rs = np.random.RandomState(0)
    y = rs.randn(50, 2)
    with pytest.raises(ValueError, match="svar_type"):
        SVAR(y, svar_type="invalid")


def test_summary_no_user_exog():
    rs = np.random.RandomState(0)
    data = rs.randn(200, 3)
    B = np.asarray([[1, "E", "E"], [0, 1, "E"], [0, 0, 1]], dtype="U")
    results = SVAR(data, svar_type="B", B=B).fit(maxlags=2, solver="newton")
    assert results.k_exog_user == 0
    results.summary()


def _macro_data():
    mdata = statsmodels.datasets.macrodata.load_pandas().data
    mdata = mdata[["realgdp", "realcons", "realinv"]]
    return np.diff(np.log(mdata.values), axis=0)


_LOWER = np.array([["E", 0, 0], ["E", "E", 0], ["E", "E", "E"]], dtype="U")
_UNIT_LOWER = np.array([[1, 0, 0], ["E", 1, 0], ["E", "E", 1]], dtype="U")
_DIAG = np.array([["E", 0, 0], [0, "E", 0], [0, 0, "E"]], dtype="U")
_OVER = np.array([[1, 0, 0], ["E", 1, 0], [0, "E", 1]], dtype="U")

# The models of results/results_svar_vars.R: svar_type, A, B
VARS_MODELS = {
    "a": ("A", _LOWER, None),
    "b": ("B", None, _LOWER),
    "ab": ("AB", _UNIT_LOWER, _DIAG),
    "ab_over": ("AB", _OVER, _DIAG),
}


def _theta(model, A, B):
    # the free parameters in the order of SVAR.loglike: the "E" entries of A
    # and then those of B, each in row-major order
    return np.r_[A[model.A_mask], B[model.B_mask]]


def _vec_index(model):
    # positions of the free parameters in (vec(A), vec(B)), column-major
    k = model.neqs
    ia, ja = np.nonzero(model.A_mask)
    ib, jb = np.nonzero(model.B_mask)
    return np.r_[ja * k + ia, k**2 + jb * k + ib]


def _ma_coefs(coefs, maxn):
    # Phi_0, ..., Phi_maxn as the upper left blocks of the powers of the
    # companion matrix (Lutkepohl 2005, Section 2.1.2)
    p, k, _ = coefs.shape
    companion = np.zeros((k * p, k * p))
    companion[:k] = np.hstack(list(coefs))
    companion[k:, :-k] = np.eye(k * (p - 1))
    power = np.eye(k * p)
    phi = []
    for _ in range(maxn + 1):
        phi.append(power[:k, :k])
        power = power @ companion
    return np.array(phi)


def _fit_vars_model(name):
    svar_type, A, B = VARS_MODELS[name]
    ref = getattr(results_svar_vars, name)
    model = SVAR(_macro_data(), svar_type, A=A, B=B)
    # statsmodels does not normalize the signs as vars does, so start near the
    # estimate of vars; newton's numerical derivatives stop it within about
    # 2e-5 of the optimum
    res = model.fit(
        maxlags=3,
        solver="newton",
        A_guess=1.01 * ref["A"][model.A_mask],
        B_guess=1.01 * ref["B"][model.B_mask],
    )
    return model, res, ref


def _fit_b_model(endog, maxlags=None, trend="c", **kwargs):
    # B-model with a lower triangular B, whose estimate is the Cholesky factor
    # of sigma_u; newton started 1% away from it converges without warnings
    lags = 1 if maxlags is None else maxlags
    var_res = VAR(endog).fit(lags, trend=trend)
    chol = np.linalg.cholesky(np.asarray(var_res.sigma_u))
    k = chol.shape[0]
    B = np.where(np.tril(np.ones((k, k))) > 0, "E", "0")
    model = SVAR(endog, "B", B=B, **kwargs)
    res = model.fit(
        maxlags=maxlags,
        trend=trend,
        solver="newton",
        B_guess=1.01 * chol[model.B_mask],
    )
    return res, var_res


def test_vars_var_coefficients():
    _, res, _ = _fit_vars_model("ab")
    coef = results_svar_vars.var_coef
    # vars orders the coefficients of each equation as [lags, const]
    assert_allclose(res.params[1:], coef[:, :-1].T, rtol=1e-10)
    assert_allclose(res.params[0], coef[:, -1], rtol=1e-10)
    assert_allclose(res.sigma_u, results_svar_vars.sigma_u, rtol=1e-10)
    assert res.k_ar == 3
    assert res.nobs == len(_macro_data()) - 3


@pytest.mark.parametrize("name", list(VARS_MODELS))
def test_vars_estimates(name):
    model, res, ref = _fit_vars_model(name)
    _, A, B = VARS_MODELS[name]
    # newton stops within about 2e-5 of the optimum (numerical derivatives)
    # (where it stops depends on the platform)
    assert_allclose(res.A, ref["A"], rtol=2e-3)
    assert_allclose(res.B, ref["B"], rtol=2e-3)
    a_mask = np.zeros((3, 3), dtype=bool) if A is None else A == "E"
    b_mask = np.zeros((3, 3), dtype=bool) if B is None else B == "E"
    assert_equal(res.A_mask, a_mask)
    assert_equal(res.B_mask, b_mask)
    assert_equal(model.A_mask, a_mask)
    assert_equal(model.B_mask, b_mask)


@pytest.mark.parametrize("name", list(VARS_MODELS))
def test_vars_loglike(name):
    model, _, ref = _fit_vars_model(name)
    theta_hat = _theta(model, ref["A"], ref["B"])
    theta_fixed = _theta(model, ref["A_fixed"], ref["B_fixed"])
    assert_allclose(model.loglike(theta_hat), ref["loglike_hat"], rtol=1e-12)
    assert_allclose(model.loglike(theta_fixed), ref["loglike_fixed"], rtol=1e-12)


@pytest.mark.parametrize("name", list(VARS_MODELS))
def test_vars_score(name):
    model, _, ref = _fit_vars_model(name)
    theta_fixed = _theta(model, ref["A_fixed"], ref["B_fixed"])
    expected = _theta(model, ref["score_A_fixed"], ref["score_B_fixed"])
    # forward differences with a step of 1e-8 lose about
    # eps * |loglike| / 1e-8 to rounding; the margin allows for other
    # platforms, and a wrong score is off by much more
    atol = 40 * np.finfo(float).eps * abs(ref["loglike_fixed"]) / 1e-8
    score = model.score(theta_fixed)
    assert_allclose(score, expected, rtol=1e-3, atol=atol)
    # a column vector is accepted
    assert_allclose(model.score(theta_fixed[:, None]), score, rtol=1e-15)


@pytest.mark.parametrize("name", ["a", "b", "ab"])
def test_vars_hessian_inverse_information(name):
    # For a just-identified model the estimate reproduces sigma_u, so that the
    # observed information equals the information matrix of vars' scoring
    # method, whose inverse gives vars' Ase and Bse.
    model, _, ref = _fit_vars_model(name)
    theta_hat = _theta(model, ref["A"], ref["B"])
    idx = _vec_index(model)
    expected = ref["cov_hat"][np.ix_(idx, idx)]
    hess = model.hessian(theta_hat)
    cov = np.linalg.inv(-hess)
    # numerical second derivatives
    assert_allclose(np.sqrt(np.diag(cov)), np.sqrt(np.diag(expected)), rtol=5e-3)
    # entries that are zero in the information inverse come out at the size
    # of the error of the numerical Hessian
    atol = 1e-3 * np.max(np.diag(expected))
    assert_allclose(cov, expected, rtol=1e-2, atol=atol)
    # a column vector is accepted
    assert_allclose(model.hessian(theta_hat[:, None]), hess, rtol=1e-15)


@pytest.mark.parametrize("name", list(VARS_MODELS))
def test_vars_structural_irf(name):
    _, res, ref = _fit_vars_model(name)
    # entries that are zero in exact arithmetic (Theta_0 = inv(A) B is
    # triangular) are at rounding level
    atol = 1e-12 * np.max(np.abs(ref["irf"]))
    process = SVARProcess(res.coefs, res.intercept, res.sigma_u, ref["A"], ref["B"])
    assert_allclose(process.svar_ma_rep(10), ref["irf"], rtol=1e-10, atol=atol)

    # Theta_h = Phi_h inv(A) B at the estimates of statsmodels
    expected = _ma_coefs(res.coefs, 10) @ np.linalg.solve(res.A, res.B)
    irf = res.irf(10)
    assert_allclose(irf.svar_irfs, expected, rtol=1e-10, atol=atol)
    assert_allclose(res.svar_ma_rep(10), expected, rtol=1e-10, atol=atol)


def test_vars_lr_overidentification():
    # With a diagonal B, the trace term of the log-likelihood is neqs at the
    # estimate, so twice the difference of the log-likelihoods of the
    # just-identified and the over-identified model is vars' LR statistic
    # nobs * (log det(Sigma(theta)) - log det(sigma_u)).
    model_ab, _, ref_ab = _fit_vars_model("ab")
    model_over, _, ref_over = _fit_vars_model("ab_over")
    llf_ab = model_ab.loglike(_theta(model_ab, ref_ab["A"], ref_ab["B"]))
    llf_over = model_over.loglike(_theta(model_over, ref_over["A"], ref_over["B"]))
    assert_allclose(2 * (llf_ab - llf_over), ref_over["lr"], rtol=1e-10)
    # K (K + 1) / 2 identified parameters less the free ones
    n_free = model_over.A_mask.sum() + model_over.B_mask.sum()
    assert 3 * 4 // 2 - n_free == ref_over["lr_df"]


def test_svar_loglike_closed_form():
    # The log-likelihood is the Gaussian log-likelihood of the residuals with
    # covariance inv(A) B B' inv(A)', after scaling the residuals so that
    # their cross product divided by nobs is sigma_u (which divides by
    # df_resid). B has a fixed nonzero entry.
    rng = np.random.default_rng(37411)
    y = rng.standard_normal((150, 2))
    y[1:] += 0.4 * y[:-1]
    A = np.array([[1, 0], ["E", 1]], dtype="U")
    B = np.array([["E", 0], ["0.3", "E"]], dtype="U")
    model = SVAR(y, svar_type="AB", A=A, B=B)
    res = model.fit(maxlags=2, solver="newton", A_guess=[0.25], B_guess=[1.0, 1.0])
    assert_allclose(res.B[1, 0], 0.3, rtol=1e-15)
    a21, b11, b22 = 0.4, 1.3, 0.8
    a_mat = np.array([[1, 0], [a21, 1]])
    b_mat = np.array([[b11, 0], [0.3, b22]])
    a_inv = np.linalg.inv(a_mat)
    cov = a_inv @ b_mat @ b_mat.T @ a_inv.T
    scaled = res.resid * np.sqrt(res.nobs / res.df_resid)
    expected = stats.multivariate_normal(np.zeros(2), cov).logpdf(scaled).sum()
    assert_allclose(model.loglike(np.array([a21, b11, b22])), expected, rtol=1e-12)


def test_svar_lag_order_default_is_one():
    res, var_res = _fit_b_model(_macro_data(), maxlags=None)
    assert res.k_ar == 1
    assert_allclose(res.params, var_res.params, rtol=1e-10)
    assert_allclose(res.sigma_u, var_res.sigma_u, rtol=1e-10)
    # newton stops within about 2e-5 of the Cholesky factor
    # (where it stops depends on the platform)
    assert_allclose(res.B, np.linalg.cholesky(var_res.sigma_u), rtol=2e-3)


@pytest.mark.parametrize("trend", ["n", "c", "ct", "ctt"])
def test_svar_trend(trend):
    res, var_res = _fit_b_model(_macro_data(), maxlags=2, trend=trend)
    k_trend = {"n": 0, "c": 1, "ct": 2, "ctt": 3}[trend]
    # VAR counts the time trend from the first observation, SVAR from the
    # first observation of the estimation sample, so that only the lag
    # coefficients are comparable with a trend
    assert_allclose(res.params[k_trend:], var_res.params[k_trend:], rtol=1e-10)
    if trend in ("n", "c"):
        assert_allclose(res.params, var_res.params, rtol=1e-10)
    # residuals near zero differ at the rounding level of the fitted values,
    # which are computed from differently scaled trend columns
    resid_scale = np.max(np.abs(var_res.resid))
    assert_allclose(res.resid, var_res.resid, rtol=1e-10, atol=1e-10 * resid_scale)
    assert_allclose(res.sigma_u, var_res.sigma_u, rtol=1e-10)
    assert res.k_trend == k_trend
    assert res.trendorder == {"n": None, "c": 0, "ct": 1, "ctt": 2}[trend]
    # newton stops within about 2e-5 of the Cholesky factor
    # (where it stops depends on the platform)
    assert_allclose(res.B, np.linalg.cholesky(var_res.sigma_u), rtol=2e-3)
    expected = _ma_coefs(res.coefs, 4) @ res.B
    # zero entries of the triangular Theta_0 are at rounding level
    assert_allclose(res.svar_ma_rep(4), expected, rtol=1e-10, atol=1e-15)


@pytest.mark.parametrize(
    "svar_type, kwargs, match",
    [
        ("A", {}, "A array not given"),
        ("AB", {"B": _DIAG}, "A array not given"),
        ("B", {}, "B array not given"),
        ("AB", {"A": _UNIT_LOWER}, "B array not given"),
    ],
)
def test_svar_type_requires_matrices(svar_type, kwargs, match):
    with pytest.raises(ValueError, match=match):
        SVAR(_macro_data(), svar_type=svar_type, **kwargs)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"A_guess": [0.1, 0.1]}, r"len\(A_guess\) = 2, there are 3 parameters in A"),
        ({"B_guess": np.ones(4)}, r"len\(B_guess\) = 4, there are 3 parameters in B"),
    ],
)
def test_svar_start_values_length(kwargs, match):
    model = SVAR(_macro_data(), svar_type="AB", A=_UNIT_LOWER, B=_DIAG)
    with pytest.raises(ValueError, match=match):
        model.fit(maxlags=1, **kwargs)


def test_svar_order_condition():
    # 4 free parameters, more than K (K + 1) / 2 = 3
    rng = np.random.default_rng(5517)
    y = rng.standard_normal((100, 2))
    A = np.array([[1, "E"], ["E", 1]], dtype="U")
    B = np.array([["E", 0], [0, "E"]], dtype="U")
    with pytest.raises(ValueError, match="Order condition not met"):
        SVAR(y, svar_type="AB", A=A, B=B).fit(maxlags=1)


def test_svar_rank_condition():
    # 3 free parameters, but the implied sigma_u[1, 1] is 1 for all of them,
    # so that the row of the Jacobian for it is zero
    rng = np.random.default_rng(5518)
    y = rng.standard_normal((100, 2))
    A = np.array([[1, "E"], [0, 1]], dtype="U")
    B = np.array([["E", "E"], [0, 1]], dtype="U")
    with pytest.raises(ValueError, match="Rank condition not met"):
        SVAR(y, svar_type="AB", A=A, B=B).fit(maxlags=1)


def test_svar_override(capsys):
    data = _macro_data()
    res = SVAR(data, svar_type="AB", A=_UNIT_LOWER, B=_DIAG).fit(maxlags=3)
    assert capsys.readouterr().out == ""
    res_override = SVAR(data, svar_type="AB", A=_UNIT_LOWER, B=_DIAG).fit(
        maxlags=3, override=True
    )
    assert "Order/rank conditions have not been checked" in capsys.readouterr().out
    assert_allclose(res_override.A, res.A, rtol=1e-12)
    assert_allclose(res_override.B, res.B, rtol=1e-12)


def test_sirf_errband_mc_cumulative():
    _, res, _ = _fit_vars_model("ab")
    # with a single replication both bands are that replication
    bands = []
    for cum in (False, True):
        # the ML fit of the replication may not converge, depending on the
        # platform
        with warns_or_none(ConvergenceWarning, match="Maximum Likelihood optim"):
            bands.append(
                res.sirf_errband_mc(
                    repl=1, steps=4, rng=np.random.default_rng(1234), cum=cum
                )
            )
    band, band_cum = bands
    assert band.lower.shape == (5, 3, 3)
    assert_allclose(band.lower, band.upper, rtol=1e-15)
    assert_allclose(band_cum.lower, np.cumsum(band.lower, axis=0), rtol=1e-12)
    assert_allclose(band_cum.upper, np.cumsum(band.upper, axis=0), rtol=1e-12)


def test_svar_process_svar_ma_rep():
    rng = np.random.default_rng(90210)
    coefs = 0.2 * rng.standard_normal((2, 3, 3))
    A = np.array([[1.0, 0, 0], [0.5, 1, 0], [-0.3, 0.2, 1]])
    B = np.diag([0.5, 1.5, 2.0])
    process = SVARProcess(coefs, np.zeros(3), np.eye(3), A, B)
    phi = _ma_coefs(coefs, 6)
    expected = phi @ np.linalg.inv(A) @ B
    # zero entries of the triangular Theta_0 are at rounding level
    assert_allclose(process.svar_ma_rep(6), expected, rtol=1e-12, atol=1e-15)
    P = [[1.0, 0.0, 0.0], [0.4, 2.0, 0.0], [0.0, -1.0, 0.5]]
    expected = phi @ np.array(P)
    assert_allclose(process.svar_ma_rep(6, P=P), expected, rtol=1e-12, atol=1e-15)
    with pytest.raises(NotImplementedError):
        process.orth_ma_rep()


def _quarterly_frame(data, index_type):
    columns = ["gdp", "cons", "inv"]
    if index_type == "datetime":
        index = pd.date_range("1959-04-01", periods=len(data), freq="QS-OCT")
    elif index_type == "period":
        index = pd.period_range("1959Q2", periods=len(data), freq="Q")
    else:
        index = pd.RangeIndex(len(data))
    return pd.DataFrame(data, index=index, columns=columns)


@pytest.mark.parametrize("form", ["list", "tuple", "datetime", "period", "range"])
def test_svar_endog_forms(form):
    data = _macro_data()
    expected, _ = _fit_b_model(data, maxlags=2)
    if form == "list":
        endog = data.tolist()
    elif form == "tuple":
        endog = tuple(tuple(row) for row in data.tolist())
    else:
        endog = _quarterly_frame(data, form)
    res, _ = _fit_b_model(endog, maxlags=2)
    # the memory layout of endog can change the rounding
    assert_allclose(res.params, expected.params, rtol=1e-10)
    assert_allclose(res.sigma_u, expected.sigma_u, rtol=1e-10)
    # newton stops within about 2e-5 of the optimum
    # (where it stops depends on the platform)
    assert_allclose(res.B, expected.B, rtol=2e-3)
    if form in ("list", "tuple"):
        assert res.names == ["y1", "y2", "y3"]
        assert res.dates is None
    elif form == "range":
        assert res.names == ["gdp", "cons", "inv"]
        assert res.dates is None
    else:
        assert res.names == ["gdp", "cons", "inv"]
        assert res.dates.equals(endog.index)


def test_svar_dates_freq():
    data = _macro_data()
    dates = pd.date_range("1959-04-01", periods=len(data), freq="QS-OCT")
    expected, _ = _fit_b_model(data, maxlags=2)
    res, _ = _fit_b_model(data, maxlags=2, dates=dates, freq="QS-OCT")
    assert res.dates.equals(dates)
    assert res.model.data.freq == "QS-OCT"
    assert_allclose(res.params, expected.params, rtol=1e-10)


def test_svar_int_endog():
    rng = np.random.default_rng(80211)
    y = rng.integers(-20, 21, size=(150, 2))
    y[1:] += y[:-1] // 2
    res, _ = _fit_b_model(y, maxlags=1)
    expected, _ = _fit_b_model(y.astype(float), maxlags=1)
    assert res.params.dtype == np.float64
    assert_allclose(res.params, expected.params, rtol=1e-10)
    assert_allclose(res.sigma_u, expected.sigma_u, rtol=1e-10)
    # newton stops within about 2e-5 of the optimum
    # (where it stops depends on the platform)
    assert_allclose(res.B, expected.B, rtol=2e-3)


def test_svar_missing():
    data = _macro_data()
    with_nan = data.copy()
    with_nan[5, 1] = np.nan
    res, var_res = _fit_b_model(np.delete(data, 5, axis=0), maxlags=2)
    dropped = SVAR(with_nan, svar_type="B", B=_LOWER, missing="drop")
    assert dropped.endog.shape == (len(data) - 1, 3)
    # the start of _fit_b_model
    start = 1.01 * np.linalg.cholesky(var_res.sigma_u)[dropped.B_mask]
    res_drop = dropped.fit(maxlags=2, solver="newton", B_guess=start)
    assert_allclose(res_drop.params, res.params, rtol=1e-10)
    # newton stops within about 2e-5 of the optimum
    # (where it stops depends on the platform)
    assert_allclose(res_drop.B, res.B, rtol=2e-3)
    with pytest.raises(MissingDataError):
        SVAR(with_nan, svar_type="B", B=_LOWER, missing="raise")


@pytest.mark.parametrize("variant", ["lowercase", "object"])
def test_svar_mask_forms(variant):
    data = _macro_data()
    if variant == "lowercase":
        A, B = np.char.lower(_UNIT_LOWER), np.char.lower(_DIAG)
    else:
        A, B = _UNIT_LOWER.astype(object), _DIAG.astype(object)
    model = SVAR(data, svar_type="AB", A=A, B=B)
    assert_equal(model.A_mask, _UNIT_LOWER == "E")
    assert_equal(model.B_mask, _DIAG == "E")
    res = model.fit(maxlags=3)
    expected = SVAR(data, svar_type="AB", A=_UNIT_LOWER, B=_DIAG).fit(maxlags=3)
    assert_allclose(res.A, expected.A, rtol=1e-12)
    assert_allclose(res.B, expected.B, rtol=1e-12)


@pytest.mark.parametrize(
    "solver",
    [
        "newton",
        "nm",
        "bfgs",
        "lbfgs",
        "powell",
        "cg",
        "ncg",
        "minimize",
        # basinhopping draws from the global random state
        pytest.param("basinhopping", marks=pytest.mark.singleton_randomstate),
    ],
)
def test_svar_solver(solver):
    # every solver started 5% off the estimate of vars closes most of the gap
    # to the maximum of the log-likelihood
    ref = results_svar_vars.a
    model = SVAR(_macro_data(), svar_type="A", A=_LOWER)
    start = 1.05 * ref["A"][model.A_mask]
    # whether a solver reports convergence depends on the platform
    with warns_or_none(ConvergenceWarning, match="Maximum Likelihood optim"):
        res = model.fit(maxlags=3, solver=solver, A_guess=start)
    theta = res.A[model.A_mask].copy()
    gap_start = ref["loglike_hat"] - model.loglike(start)
    gap_end = ref["loglike_hat"] - model.loglike(theta)
    assert gap_start > 1.0
    assert gap_end < 0.05 * gap_start


def test_svar_solver_invalid():
    model = SVAR(_macro_data(), svar_type="A", A=_LOWER)
    with pytest.raises(ValueError, match="method must be one of"):
        model.fit(maxlags=3, solver="gradient-descent")


def _macro_dlog():
    mdata = statsmodels.datasets.macrodata.load_pandas().data
    return np.diff(np.log(mdata[["realgdp", "realcons", "realinv"]].values), axis=0)


def _ic_order(y, maxlags, trend, ic):
    # The lag order that minimizes the criterion of Lütkepohl (2005), Section
    # 4.3, computed on the sample of the largest model
    nobs, k = y.shape
    k_trend = int(trend == "c")
    t = nobs - maxlags
    values = {}
    for p in range(1 - k_trend, maxlags + 1):
        x = [np.ones((t, 1))] if k_trend else []
        x += [y[maxlags - i : nobs - i] for i in range(1, p + 1)]
        x = np.column_stack(x)
        resid = y[maxlags:] - x @ np.linalg.lstsq(x, y[maxlags:], rcond=None)[0]
        logdet = np.linalg.slogdet(resid.T @ resid / t)[1]
        n_params = k * (k * p + k_trend)
        penalty = 2 if ic == "aic" else np.log(t)
        values[p] = logdet + penalty * n_params / t
    return min(values, key=values.get)


@pytest.mark.parametrize(
    "trend, maxlags, ic",
    [("n", 6, "aic"), ("n", 6, "bic"), ("c", 6, "aic"), ("c", None, "aic")],
)
def test_fit_ic_selects_order(trend, maxlags, ic):
    # fit(ic=...) raised an AttributeError (SVAR has no select_order)
    data = _macro_dlog()
    A = np.asarray([[1, 0, 0], ["E", 1, 0], ["E", "E", 1]], dtype="U")
    B = np.asarray([["E", 0, 0], [0, "E", 0], [0, 0, "E"]], dtype="U")
    start = {"A_guess": [-0.5, -5.5, 3.0], "B_guess": [0.0075, 0.005, 0.02]}
    model = SVAR(data, "AB", A=A, B=B)
    res = model.fit(maxlags=maxlags, ic=ic, trend=trend, **start)
    # None is 12 * (nobs / 100) ** (1 / 4) lags, 14 for these data
    expected = _ic_order(data, 14 if maxlags is None else maxlags, trend, ic)
    assert res.k_ar == expected
    direct = SVAR(data, "AB", A=A, B=B).fit(maxlags=expected, trend=trend, **start)
    assert_allclose(res.params, direct.params, rtol=1e-12)
    assert_allclose(res.A, direct.A, rtol=1e-12)
    assert_allclose(res.B, direct.B, rtol=1e-12)


def test_fit_ic_verbose_and_unknown(capsys):
    data = _macro_dlog()[:, :2]
    B = np.asarray([["E", 0], ["E", "E"]], dtype="U")
    res = SVAR(data, "B", B=B).fit(maxlags=6, ic="bic", trend="n", verbose=True)
    assert res.k_ar == _ic_order(data, 6, "n", "bic")
    out = capsys.readouterr().out
    assert "Selected orders are" in out
    assert f"Using {res.k_ar} based on bic criterion" in out
    with pytest.raises(ValueError, match="sic not recognized"):
        SVAR(data, "B", B=B).fit(maxlags=6, ic="sic")


def _svar_ab_model():
    mdata = statsmodels.datasets.macrodata.load_pandas().data
    data = np.diff(np.log(mdata[["realgdp", "realcons", "realinv"]].values), axis=0)
    A = np.asarray([[1, 0, 0], ["E", 1, 0], ["E", "E", 1]], dtype="U")
    B = np.asarray([["E", 0, 0], [0, "E", 0], [0, 0, "E"]], dtype="U")
    return SVAR(data, svar_type="AB", A=A, B=B)


def test_results_keep_a_b_after_refit():
    # The results held the A and B arrays of the model, so a second fit
    # overwrote the estimates of the first results
    from .results import results_svar_st

    stata = results_svar_st.results_svar1_small
    model = _svar_ab_model()
    res = model.fit(maxlags=3)
    model.fit(maxlags=1)
    fresh = _svar_ab_model().fit(maxlags=3)
    assert_allclose(res.A, fresh.A, rtol=1e-12)
    assert_allclose(res.B, fresh.B, rtol=1e-12)
    assert_allclose(res.svar_ma_rep(4), fresh.svar_ma_rep(4), rtol=1e-12)
    # Stata's estimates, to a tolerance for where the optimizer stops
    assert_allclose(res.A, stata.A, rtol=1e-3)
    assert_allclose(np.abs(res.B), stata.B, rtol=1e-3)


def test_results_keep_a_b_after_loglike():
    # loglike, and so score and hessian, wrote the parameters into the A and
    # B of the model, which the results shared
    model = _svar_ab_model()
    res = model.fit(maxlags=3)
    fresh = _svar_ab_model().fit(maxlags=3)
    params = np.r_[res.A[res.A_mask], res.B[res.B_mask]]
    other = params + 0.5
    model.loglike(other)
    model.score(other)
    model.hessian(other)
    assert_allclose(res.A, fresh.A, rtol=1e-12)
    assert_allclose(res.B, fresh.B, rtol=1e-12)
    assert_allclose(res.svar_ma_rep(4), fresh.svar_ma_rep(4), rtol=1e-12)
    # The model keeps its restrictions, NaN for the estimated entries
    nan = np.nan
    assert_allclose(model.A, [[1, 0, 0], [nan, 1, 0], [nan, nan, 1]])
    assert_allclose(model.B, [[nan, 0, 0], [0, nan, 0], [0, 0, nan]])


def _svar_lower_b():
    mdata = statsmodels.datasets.macrodata.load_pandas().data
    data = np.diff(np.log(mdata[["realgdp", "realcons"]].values), axis=0)
    B = np.asarray([["E", 0], ["E", "E"]], dtype="U")
    return SVAR(data, "B", B=B)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"method": "gls"}, "method='gls' is ignored"),
        ({"s_method": "scoring"}, "s_method='scoring' is ignored"),
        ({"maxfun": 100}, "maxfun is ignored"),
    ],
)
def test_fit_ignored_arguments_deprecated(kwargs, match):
    # these arguments were ignored silently
    expected = _svar_lower_b().fit(maxlags=1)
    with pytest.warns(FutureWarning, match=match):
        res = _svar_lower_b().fit(maxlags=1, **kwargs)
    assert_allclose(res.params, expected.params, rtol=1e-12)
    assert_allclose(res.B, expected.B, rtol=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [{"method": "OLS"}, {"method": None}, {"s_method": "MLE"}, {"s_method": None}],
)
def test_fit_default_arguments_no_warning(kwargs):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        _svar_lower_b().fit(maxlags=1, **kwargs)


def test_sirf_errband_mc_orth_deprecated():
    # orth was ignored silently: the bands are those of the structural
    # responses
    import warnings

    res = _svar_lower_b().fit(maxlags=1)
    kwargs = {"repl": 5, "steps": 3, "burn": 10, "signif": 0.5}
    # the ML fits of the replications may not converge, depending on the platform
    with (
        warnings.catch_warnings(),
        warns_or_none(ConvergenceWarning, match="Maximum Likelihood optim"),
    ):
        warnings.simplefilter("error", FutureWarning)
        expected = res.sirf_errband_mc(rng=np.random.default_rng(0), **kwargs)
    with (
        pytest.warns(FutureWarning, match="orth=True is ignored"),
        warns_or_none(ConvergenceWarning, match="Maximum Likelihood optim"),
    ):
        band = res.sirf_errband_mc(orth=True, rng=np.random.default_rng(0), **kwargs)
    assert_allclose(band.lower, expected.lower, rtol=1e-12)
    assert_allclose(band.upper, expected.upper, rtol=1e-12)


@pytest.mark.parametrize(
    "svar_type, A, B",
    [
        ("B", [[1, 0], ["E", 1]], [["E", 0], [0, "E"]]),
        ("B", [[1, 0], ["e", 1]], [["E", 0], ["E", "E"]]),
        ("A", [[1, 0], ["E", 1]], [["E", 0], [0, "E"]]),
    ],
)
def test_e_entries_not_estimated_raise(svar_type, A, B):
    # svar_type="B" estimated the "E" entries of A too, with the free entries
    # of B tied to one value, or raised in fit; svar_type="A" raised in fit
    mdata = statsmodels.datasets.macrodata.load_pandas().data
    data = np.diff(np.log(mdata[["realgdp", "realcons"]].values), axis=0)
    other = "A" if svar_type == "B" else "B"
    match = f'svar_type="{svar_type}" does not estimate {other}'
    with pytest.raises(ValueError, match=match):
        SVAR(data, svar_type, A=np.array(A, dtype="U"), B=np.array(B, dtype="U"))
