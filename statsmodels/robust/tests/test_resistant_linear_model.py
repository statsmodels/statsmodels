"""
Tests for RLMDetS and RLMDetSMM (statsmodels.robust.resistant_linear_model).

Neither estimator has a closed-form solution or a reference implementation
in another package, so correctness is checked two independent ways:

* the M-scale defining equation ``mean(rho(resid / raw_scale)) ==
  scale_bias`` must hold at the returned ``(params, scale)`` -- this is
  the algebraic condition an S-estimator solution must satisfy by
  definition, derived directly from ``RLMDetS.__init__`` and
  ``RLM._estimate_scale``, independent of how ``fit`` gets there;
* under gross contamination, both estimators must recover the true
  regression coefficients far better than OLS, which is the entire
  reason to use them.
"""
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest

import statsmodels.api as sm
from statsmodels.robust.resistant_linear_model import (
    DetSStartResult,
    RLMDetS,
    RLMDetSMM,
)

cur_dir = Path(__file__).parent.resolve()


def _moment_condition(mod, res, rtol=1e-4):
    """(params, scale) satisfy the S-estimator's M-scale equation.

    Only valid where params and scale were determined jointly by a single
    S-type fixed point using ``mod.norm``/``mod.mscale`` -- i.e. for
    ``RLMDetS.fit``, not for ``RLMDetSMM.fit``'s second (M) stage, which
    reweights with a different norm and, in the fixed-scale branch, does
    not re-solve the scale equation at all.

    ``res.scale`` carries RLM's extra ``sqrt(nobs / df_resid)`` factor
    (see ``RLM._estimate_scale``), so back it out before checking against
    ``mscale.scale_bias``.
    """
    endog, exog = mod.endog, mod.exog
    resid = endog - exog @ res.params
    df_resid = exog.shape[0] - exog.shape[1]
    raw_scale = res.scale * np.sqrt(df_resid / exog.shape[0])
    lhs = np.mean(mod.norm.rho(resid / raw_scale))
    assert_allclose(lhs, mod.mscale.scale_bias, rtol=rtol)


def _normal_equations(mod, res, norm, atol=1e-4):
    """The weighted normal equations X' W(resid/scale) resid == 0 hold.

    This is the first-order condition any converged (weighted) IRLS fit
    satisfies for the norm that produced its final weights, regardless of
    how the scale itself was obtained -- so unlike ``_moment_condition``
    it applies uniformly to every RLMDetS/RLMDetSMM code path.
    """
    endog, exog = mod.endog, mod.exog
    resid = endog - exog @ res.params
    weights = norm.weights(resid / res.scale)
    grad = exog.T @ (weights * resid)
    assert_allclose(grad, 0, atol=atol)


def _stackloss():
    from statsmodels.datasets.stackloss import load
    data = load()
    endog = np.asarray(data.endog)
    exog = sm.add_constant(np.asarray(data.exog))
    return endog, exog


def _contaminated(n=200, k=3, frac=0.2, seed=12345):
    rng = np.random.default_rng(seed)
    exog = sm.add_constant(rng.standard_normal((n, k)))
    beta = np.array([1.0, 2.0, -1.5, 0.5])
    endog = exog @ beta + rng.standard_normal(n) * 0.5
    n_out = int(frac * n)
    idx = rng.choice(n, n_out, replace=False)
    endog[idx] += rng.choice([-1, 1], n_out) * rng.uniform(15, 25, n_out)
    return endog, exog, beta


class TestRLMDetS:

    def test_fit_moment_condition_and_start_results(self):
        endog, exog = _stackloss()
        mod = RLMDetS(endog, exog, breakdown_point=0.5)
        res = mod.fit(h=14)
        _moment_condition(mod, res)

        assert res.model_dets is mod
        results_iter = res._results.results_iter
        assert len(results_iter) > 0
        for entry in results_iter.values():
            assert isinstance(entry, DetSStartResult)
            assert entry.scale > 0
            assert entry.params.shape == (exog.shape[1],)

    def test_fit_beats_ols_under_contamination(self):
        endog, exog, beta = _contaminated()
        mod = RLMDetS(endog, exog)
        res = mod.fit(h=120)
        ols = sm.OLS(endog, exog).fit()

        err_dets = np.linalg.norm(res.params - beta)
        err_ols = np.linalg.norm(ols.params - beta)
        assert err_dets < 0.3
        assert err_dets < 0.3 * err_ols
        _moment_condition(mod, res)
        _normal_equations(mod, res, mod.norm)

    def test_start_params_extra(self):
        endog, exog = _stackloss()
        mod = RLMDetS(endog, exog)
        ols_params = sm.OLS(endog, exog).fit().params
        res = mod.fit(h=14, start_params_extra=[np.asarray(ols_params)])
        _moment_condition(mod, res)

    def test_univariate_quantile_start(self):
        # exog is constant-only, so data_start is empty and
        # _get_start_params falls back to endog quantiles.
        endog, _ = _stackloss()
        exog = np.ones((len(endog), 1))
        mod = RLMDetS(endog, exog)

        starts = mod._get_start_params(h=14)
        assert len(starts) == 3
        assert_allclose(sorted(s[0] for s in starts),
                        np.quantile(endog, [0.25, 0.5, 0.75]))

        res = mod.fit(h=14)
        _moment_condition(mod, res)

    def test_include_endog_and_col_indices(self):
        endog, exog = _stackloss()
        n_extra = exog.shape[1] - 1

        mod = RLMDetS(endog, exog, include_endog=True)
        assert mod.data_start.shape == (len(endog), n_extra + 1)
        res = mod.fit(h=14)
        _moment_condition(mod, res)

        mod2 = RLMDetS(endog, exog, col_indices=[1, 2])
        assert mod2.data_start.shape == (len(endog), 2)
        res2 = mod2.fit(h=14)
        _moment_condition(mod2, res2)


class TestRLMDetSMM:

    def test_fit_default_h(self):
        # Regression test: h=None previously reached np.argpartition(d,
        # None) inside _get_detcov_startidx and raised a bare
        # "TypeError: Partition index must be integer" -- fit()'s
        # documented default was unreachable.
        endog, exog = _stackloss()
        mod = RLMDetSMM(endog, exog)
        res = mod.fit()
        assert np.isfinite(res.scale)
        assert res.scale > 0
        assert res._results.results_dets is not None
        _normal_equations(mod, res, mod.norm_mean)

    def test_fit_beats_ols_under_contamination(self):
        endog, exog, beta = _contaminated()
        mod = RLMDetSMM(endog, exog, efficiency=0.95)
        res = mod.fit(h=120)
        ols = sm.OLS(endog, exog).fit()

        err_detsmm = np.linalg.norm(res.params - beta)
        err_ols = np.linalg.norm(ols.params - beta)
        assert err_detsmm < 0.3
        assert err_detsmm < 0.3 * err_ols
        _normal_equations(mod, res, mod.norm_mean)

    def test_efficiency_close_to_ols_on_clean_data(self):
        rng = np.random.default_rng(999)
        n, k = 300, 3
        exog = sm.add_constant(rng.standard_normal((n, k)))
        beta = np.array([1.0, 2.0, -1.5, 0.5])
        endog = exog @ beta + rng.standard_normal(n) * 0.5

        mod = RLMDetSMM(endog, exog, efficiency=0.95)
        res = mod.fit(h=200)
        ols = sm.OLS(endog, exog).fit()
        assert_allclose(res.params, ols.params, atol=0.25)
        _normal_equations(mod, res, mod.norm_mean)

    def test_scale_binding(self):
        endog, exog = _stackloss()
        mod = RLMDetSMM(endog, exog)
        res = mod.fit(h=14, scale_binding=True)
        assert np.isfinite(res.scale)
        assert res.scale > 0
        assert res._results.results_dets is not None
        _normal_equations(mod, res, mod.norm_mean)

    def test_start_tuple_skips_s_stage(self):
        endog, exog = _stackloss()
        res_s = RLMDetS(endog, exog).fit(h=14)

        mod = RLMDetSMM(endog, exog)
        res = mod.fit(start=(np.asarray(res_s.params), res_s.scale))
        assert res._results.results_dets is None
        assert np.isfinite(res.scale)
        assert res.scale > 0
        _normal_equations(mod, res, mod.norm_mean)

    @pytest.mark.parametrize("breakdown_point", [0.25, 0.5])
    def test_breakdown_point_passthrough(self, breakdown_point):
        endog, exog = _stackloss()
        mod = RLMDetSMM(endog, exog, breakdown_point=breakdown_point)
        assert mod.breakdown_point == breakdown_point
        res = mod.fit(h=14)
        assert np.isfinite(res.scale)
        assert res.scale > 0
        _normal_equations(mod, res, mod.norm_mean)
