"""
Tests for Fama-MacBeth regression

Author: Contributed to statsmodels
License: BSD-3
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_equal

import statsmodels.api as sm
from statsmodels.regression.fama_macbeth import FamaMacBeth


class TestFamaMacBeth:
    @classmethod
    def setup_class(cls):
        np.random.seed(42)
        cls.n_entities = 25
        cls.n_periods = 120
        cls.n_factors = 3
        cls.entities = np.repeat(np.arange(cls.n_entities), cls.n_periods)
        cls.time = np.tile(np.arange(cls.n_periods), cls.n_entities)
        cls.true_betas = np.random.uniform(0.5, 1.5, (cls.n_entities, cls.n_factors))
        cls.factor_returns = np.random.randn(cls.n_periods, cls.n_factors) * 0.05
        cls.returns = np.zeros(cls.n_entities * cls.n_periods)
        cls.factors = np.zeros((cls.n_entities * cls.n_periods, cls.n_factors))
        for i in range(cls.n_entities):
            idx = cls.entities == i
            cls.returns[idx] = (
                cls.factor_returns @ cls.true_betas[i]
                + np.random.randn(cls.n_periods) * 0.03
            )
            cls.factors[idx] = cls.factor_returns
        cls.exog = sm.add_constant(cls.factors)

    def test_basic_estimation(self):
        mod = FamaMacBeth(
            self.returns,
            self.exog,
            entity=self.entities,
            time=self.time,
        )
        res = mod.fit()
        assert hasattr(res, "params")
        assert hasattr(res, "bse")
        assert hasattr(res, "tvalues")
        assert hasattr(res, "pvalues")
        assert hasattr(res, "rsquared")
        assert_equal(len(res.params), self.n_factors + 1)
        assert_equal(len(res.bse), self.n_factors + 1)
        assert_equal(res.betas.shape, (self.n_factors + 1, self.n_entities))
        assert_equal(res.lambdas.shape, (self.n_periods, self.n_factors + 1))
        assert np.all(np.isfinite(res.params))
        assert np.all(np.isfinite(res.bse))
        assert np.all(res.bse > 0)

    def test_cov_types(self):
        mod = FamaMacBeth(
            self.returns,
            self.exog,
            entity=self.entities,
            time=self.time,
        )
        res_robust = mod.fit(cov_type="robust")
        assert res_robust.cov_type == "robust"
        assert res_robust.bandwidth is None
        res_hac = mod.fit(cov_type="HAC", bandwidth=4)
        assert res_hac.cov_type == "HAC"
        assert res_hac.bandwidth == 4
        assert not np.allclose(res_robust.bse, res_hac.bse)

    def test_hac_auto_bandwidth(self):
        mod = FamaMacBeth(
            self.returns,
            self.exog,
            entity=self.entities,
            time=self.time,
        )
        res = mod.fit(cov_type="kernel")
        assert res.bandwidth is not None
        assert res.bandwidth > 0
        expected_bw = int(np.floor(4 * (self.n_periods / 100) ** (2 / 9)))
        assert res.bandwidth == expected_bw

    def test_summary(self):
        mod = FamaMacBeth(
            self.returns,
            self.exog,
            entity=self.entities,
            time=self.time,
        )
        res = mod.fit()
        smry = res.summary()
        assert smry is not None
        smry_str = str(res)
        assert "Fama-MacBeth" in smry_str
        assert "coef" in smry_str
        assert "std err" in smry_str

    def test_conf_int(self):
        mod = FamaMacBeth(
            self.returns,
            self.exog,
            entity=self.entities,
            time=self.time,
        )
        res = mod.fit()
        ci = res.conf_int(alpha=0.05)
        assert ci.shape == (len(res.params), 2)
        assert np.all(ci[:, 0] < res.params)
        assert np.all(ci[:, 1] > res.params)
        ci_90 = res.conf_int(alpha=0.10)
        assert np.all(ci_90[:, 1] - ci_90[:, 0] < ci[:, 1] - ci[:, 0])

    def test_missing_data(self):
        returns = self.returns.copy()
        exog = self.exog.copy()
        returns[::10] = np.nan
        mod = FamaMacBeth(
            returns,
            exog,
            entity=self.entities,
            time=self.time,
            missing="drop",
        )
        res = mod.fit()
        assert np.all(np.isfinite(res.params))

    def test_insufficient_periods_warning(self):
        n_obs = self.n_entities * 5
        with pytest.warns(UserWarning, match="Only 5 time periods"):
            mod = FamaMacBeth(
                self.returns[:n_obs],
                self.exog[:n_obs],
                entity=self.entities[:n_obs],
                time=self.time[:n_obs],
            )

    def test_invalid_cov_type(self):
        mod = FamaMacBeth(
            self.returns,
            self.exog,
            entity=self.entities,
            time=self.time,
        )
        with pytest.raises(ValueError, match="Unknown cov_type"):
            mod.fit(cov_type="invalid")

    def test_invalid_kernel(self):
        mod = FamaMacBeth(
            self.returns,
            self.exog,
            entity=self.entities,
            time=self.time,
        )
        with pytest.raises(ValueError, match="not supported"):
            mod.fit(cov_type="kernel", kernel="invalid")

    def test_dimension_mismatch(self):
        with pytest.raises(ValueError, match="entity has length"):
            FamaMacBeth(
                self.returns,
                self.exog,
                entity=self.entities[:-10],
                time=self.time,
            )

    def test_no_constant(self):
        exog_no_const = self.factors
        mod = FamaMacBeth(
            self.returns,
            exog_no_const,
            entity=self.entities,
            time=self.time,
        )
        res = mod.fit()
        assert_equal(len(res.params), self.n_factors)


class TestFamaMacBethRealistic:
    def test_fama_french_factors(self):
        np.random.seed(123)
        n_stocks = 30
        n_months = 180
        mkt_rf = np.random.randn(n_months) * 0.04 + 0.005
        smb = np.random.randn(n_months) * 0.03
        hml = np.random.randn(n_months) * 0.03
        beta_mkt = np.random.uniform(0.7, 1.3, n_stocks)
        beta_smb = np.random.uniform(-0.5, 0.5, n_stocks)
        beta_hml = np.random.uniform(-0.5, 0.5, n_stocks)
        excess_returns = np.zeros((n_months, n_stocks))
        for i in range(n_stocks):
            excess_returns[:, i] = (
                beta_mkt[i] * mkt_rf
                + beta_smb[i] * smb
                + beta_hml[i] * hml
                + np.random.randn(n_months) * 0.02
            )
        entities = np.repeat(np.arange(n_stocks), n_months)
        time = np.tile(np.arange(n_months), n_stocks)
        returns = excess_returns.T.flatten()
        factors = np.zeros((n_stocks * n_months, 3))
        for t in range(n_months):
            idx = time == t
            factors[idx, 0] = mkt_rf[t]
            factors[idx, 1] = smb[t]
            factors[idx, 2] = hml[t]
        exog = sm.add_constant(factors)
        mod = FamaMacBeth(returns, exog, entity=entities, time=time)
        res = mod.fit(cov_type="robust")
        mkt_premium = res.params[1]
        assert mkt_premium > 0
        assert np.abs(res.params[0]) < 0.01


class TestFamaMacBethEquivalence:
    def test_single_period_cross_section(self):
        np.random.seed(456)
        n_stocks = 50
        n_factors = 2
        betas = np.random.randn(n_stocks, n_factors)
        true_lambda = np.array([0.05, 0.03])
        returns = betas @ true_lambda + np.random.randn(n_stocks) * 0.02
        entities = np.arange(n_stocks)
        time = np.zeros(n_stocks, dtype=int)
        factors = np.tile(true_lambda, (n_stocks, 1))
        exog = sm.add_constant(factors)
        mod_fm = FamaMacBeth(returns, exog, entity=entities, time=time)
        res_fm = mod_fm.fit()
        exog_beta = sm.add_constant(betas)
        mod_ols = sm.OLS(returns, exog_beta)
        res_ols = mod_ols.fit()
        assert_allclose(res_fm.params, res_ols.params, rtol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
