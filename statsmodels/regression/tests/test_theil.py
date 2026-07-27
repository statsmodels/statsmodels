"""
Created on Mon May 05 17:29:56 2014

Author: Josef Perktold
"""

from statsmodels.compat.scipy import SP_LT_116

from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from statsmodels.regression.linear_model import GLS, OLS
from statsmodels.sandbox.regression.penalized import TheilGLS


class TestTheilTextile:

    @classmethod
    def setup_class(cls):
        cur_dir = Path(__file__).resolve().parent
        filepath = Path(cur_dir).joinpath("results", "theil_textile_predict.csv")
        cls.res_predict = pd.read_csv(filepath, sep=",")
        data = np.array(
            "        1923\t1.99651\t1.98543\t2.00432\n        1924\t1.99564\t1.99167\t2.00043\n        1925\t2\t2\t2\n        1926\t2.04766\t2.02078\t1.95713\n        1927\t2.08707\t2.02078\t1.93702\n        1928\t2.07041\t2.03941\t1.95279\n        1929\t2.08314\t2.04454\t1.95713\n        1930\t2.13354\t2.05038\t1.91803\n        1931\t2.18808\t2.03862\t1.84572\n        1932\t2.18639\t2.02243\t1.81558\n        1933\t2.20003\t2.00732\t1.78746\n        1934\t2.14799\t1.97955\t1.79588\n        1935\t2.13418\t1.98408\t1.80346\n        1936\t2.22531\t1.98945\t1.72099\n        1937\t2.18837\t2.0103\t1.77597\n        1938\t2.17319\t2.00689\t1.77452\n        1939\t2.2188\t2.0162\t1.78746".split(),
            float,
        ).reshape(-1, 4)
        endog = data[:, 1]
        exog = np.column_stack((data[:, 2:], np.ones(endog.shape[0])))
        r_matrix = np.array([[1, 0, 0], [0, 1, 0]])
        r_mean = [1, -0.7]
        cov_r = np.array([[0.15**2, -0.01], [-0.01, 0.15**2]])
        mod = TheilGLS(endog, exog, r_matrix, q_matrix=r_mean, sigma_prior=cov_r)
        cls.res1 = mod.fit(cov_type="data-prior", use_t=True)
        cls.res1._cache["scale"] = 0.00018334123641580062
        from .results import results_theil_textile as resmodule

        cls.res2 = resmodule.results_theil_textile

    def test_basic(self):
        pt = self.res2.params_table[:, :6].T
        params2, bse2, tvalues2, pvalues2, ci_low, ci_upp = pt
        assert_allclose(self.res1.params, params2, rtol=2e-06)
        corr_fact = 0.9836026210570028
        corr_fact = 0.9737686504146373
        corr_fact = 1
        assert_allclose(self.res1.bse / corr_fact, bse2, rtol=2e-06)
        assert_allclose(self.res1.tvalues * corr_fact, tvalues2, rtol=2e-06)
        ci = self.res1.conf_int()
        assert_allclose(ci[:, 0], ci_low, rtol=0.01)
        assert_allclose(ci[:, 1], ci_upp, rtol=0.01)
        assert_allclose(self.res1.rsquared, self.res2.r2, rtol=2e-06)
        corr_fact = self.res1.df_resid / self.res2.df_r
        assert_allclose(
            np.sqrt(self.res1.mse_resid * corr_fact), self.res2.rmse, rtol=2e-06
        )
        assert_allclose(
            self.res1.fittedvalues, self.res_predict["fittedvalues"], atol=50000000.0
        )

    def test_other(self):
        tc = self.res1.test_compatibility()
        assert_allclose(np.squeeze(tc[0]), self.res2.compat, rtol=2e-06)
        assert_allclose(np.squeeze(tc[1]), self.res2.pvalue, rtol=2e-06)
        frac = self.res1.share_data()
        assert_allclose(frac, 0.6946116246864239, rtol=2e-06)

    def test_no_penalization(self):
        res_ols = OLS(self.res1.model.endog, self.res1.model.exog).fit()
        res_theil = self.res1.model.fit(pen_weight=0, cov_type="data-prior")
        assert_allclose(res_theil.params, res_ols.params, rtol=1e-10)
        assert_allclose(res_theil.bse, res_ols.bse, rtol=1e-10)

    @pytest.mark.smoke
    def test_summary(self):
        if SP_LT_116:
            with pytest.warns(UserWarning, match=".*kurtosistest.*"):
                self.res1.summary()
        else:
            self.res1.summary()


class CheckEquivalenceMixin:
    tol = {"default": (0.0001, 1e-20)}

    @classmethod
    def get_sample(cls):
        rs = np.random.RandomState(987456)
        nobs, k_vars = (200, 5)
        beta = 0.5 * np.array([0.1, 1, 1, 0, 0])
        x = rs.randn(nobs, k_vars)
        x[:, 0] = 1
        y = np.dot(x, beta) + 2 * rs.randn(nobs)
        return (y, x)

    def test_attributes(self):
        attributes_fit = [
            "params",
            "rsquared",
            "df_resid",
            "df_model",
            "llf",
            "aic",
            "bic",
        ]
        attributes_inference = ["bse", "tvalues", "pvalues"]
        import copy

        attributes = copy.copy(attributes_fit)
        if not getattr(self, "skip_inference", False):
            attributes.extend(attributes_inference)
        for att in attributes:
            r1 = getattr(self.res1, att)
            r2 = getattr(self.res2, att)
            if not np.size(r1) == 1:
                r1 = r1[: len(r2)]
            rtol, atol = self.tol.get(att, self.tol["default"])
            message = "attribute: " + att
            assert_allclose(r1, r2, rtol=rtol, atol=atol, err_msg=message)
        assert_allclose(
            self.res1.fittedvalues, self.res1.fittedvalues, rtol=0.001, atol=0.0001
        )
        assert_allclose(self.res1.resid, self.res1.resid, rtol=0.001, atol=0.0001)


class TestTheil1(CheckEquivalenceMixin):

    @classmethod
    def setup_class(cls):
        y, x = cls.get_sample()
        mod1 = TheilGLS(y, x, sigma_prior=[0, 0, 1.0, 1.0])
        cls.res1 = mod1.fit(200000)
        cls.res2 = OLS(y, x[:, :3]).fit()


class TestTheil2(CheckEquivalenceMixin):

    @classmethod
    def setup_class(cls):
        y, x = cls.get_sample()
        mod1 = TheilGLS(y, x, sigma_prior=[0, 0, 1.0, 1.0])
        cls.res1 = mod1.fit(0)
        cls.res2 = OLS(y, x).fit()


class TestTheil3(CheckEquivalenceMixin):

    @classmethod
    def setup_class(cls):
        cls.skip_inference = True
        y, x = cls.get_sample()
        xd = np.column_stack((x, x))
        r_matrix = np.eye(5, 10, 5)
        mod1 = TheilGLS(y, xd, r_matrix=r_matrix)
        cls.res1 = mod1.fit(0.001, cov_type="data-prior")
        cls.res2 = OLS(y, x).fit()


class TestTheilGLS(CheckEquivalenceMixin):

    @classmethod
    def setup_class(cls):
        y, x = cls.get_sample()
        nobs = len(y)
        weights = (np.arange(nobs) < nobs // 2) + 0.5
        mod1 = TheilGLS(y, x, sigma=weights, sigma_prior=[0, 0, 1.0, 1.0])
        cls.res1 = mod1.fit(200000)
        cls.res2 = GLS(y, x[:, :3], sigma=weights).fit()


class TestTheilLinRestriction(CheckEquivalenceMixin):

    @classmethod
    def setup_class(cls):
        y, x = cls.get_sample()
        x2 = x[:, :2].copy()
        x2[:, 1] += x[:, 2]
        mod1 = TheilGLS(y, x[:, :3], r_matrix=[[0, 1, -1]])
        cls.res1 = mod1.fit(200000)
        cls.res2 = OLS(y, x2).fit()
        tol = {"pvalues": (0.0001, 2e-07), "tvalues": (0.0005, 0)}
        tol.update(cls.tol)
        cls.tol = tol


class TestTheilLinRestrictionApprox(CheckEquivalenceMixin):

    @classmethod
    def setup_class(cls):
        y, x = cls.get_sample()
        x2 = x[:, :2].copy()
        x2[:, 1] += x[:, 2]
        mod1 = TheilGLS(y, x[:, :3], r_matrix=[[0, 1, -1]])
        cls.res1 = mod1.fit(100)
        cls.res2 = OLS(y, x2).fit()
        import copy

        tol = copy.copy(cls.tol)
        tol2 = {"default": (0.15, 0), "params": (0.05, 0), "pvalues": (0.02, 0.001)}
        tol.update(tol2)
        cls.tol = tol


class TestTheilPanel:

    @classmethod
    def setup_class(cls):
        nobs = 300
        nobs_i = 5
        n_groups = nobs // nobs_i
        k_vars = 3
        from statsmodels.sandbox.panel.random_panel import PanelSample

        dgp = PanelSample(nobs, k_vars, n_groups, rng=303305)
        if isinstance(dgp.random_state, np.random.RandomState):
            dgp.group_means = 2 + dgp.random_state.randn(n_groups)
        else:
            dgp.group_means = 2 + dgp.random_state.uniform(n_groups)
        print("rng", dgp.rng)
        y = dgp.generate_panel()
        x = np.column_stack(
            (dgp.exog[:, 1:], dgp.groups[:, None] == np.arange(n_groups))
        )
        cls.dgp = dgp
        cls.endog = y
        cls.exog = x
        cls.res_ols = OLS(y, x).fit()

    def test_regression(self):
        y = self.endog
        x = self.exog
        n_groups, k_vars = (self.dgp.n_groups, self.dgp.k_vars)
        Rg = np.eye(n_groups - 1) - 1.0 / n_groups * np.ones(
            (n_groups - 1, n_groups - 1)
        )
        R = np.c_[np.zeros((n_groups - 1, k_vars)), Rg]
        r = np.zeros(n_groups - 1)
        R[:, k_vars - 1] = -1
        lambd = 1
        mod = TheilGLS(y, x, r_matrix=R, q_matrix=r, sigma_prior=lambd)
        res = mod.fit()
        params1 = np.array(
            [
                0.9751655,
                1.05215277,
                0.37135028,
                2.0492626,
                2.82062503,
                2.82139775,
                1.92940468,
                2.96942081,
                2.86349583,
                3.20695368,
                4.04516422,
                3.04918839,
                4.54748808,
                3.49026961,
                3.15529618,
                4.25552932,
                2.65471759,
                3.62328747,
                3.07283053,
                3.49485898,
                3.42301424,
                2.94677593,
                2.81549427,
                2.24895113,
                2.29222784,
                2.89194946,
                3.17052308,
                2.37754241,
                3.54358533,
                3.79838425,
                1.91189071,
                1.15976407,
                4.05629691,
                1.58556827,
                4.49941666,
                4.08608599,
                3.1889269,
                2.86203652,
                3.06785013,
                1.9376162,
                2.90657681,
                3.71910592,
                3.15607617,
                3.58464547,
                2.15466323,
                4.87026717,
                2.92909833,
                2.64998337,
                2.891171,
                4.04422964,
                3.54616122,
                4.12135273,
                3.70232028,
                3.8314497,
                2.2591451,
                2.39321422,
                3.13064532,
                2.1569678,
                2.04667506,
                3.92064689,
                3.66243644,
                3.11742725,
            ]
        )
        assert_allclose(res.params, params1)
        pen_weight_aicc = mod.select_pen_weight(method="aicc")
        pen_weight_gcv = mod.select_pen_weight(method="gcv")
        pen_weight_cv = mod.select_pen_weight(method="cv")
        pen_weight_bic = mod.select_pen_weight(method="bic")
        assert_allclose(pen_weight_gcv, pen_weight_aicc, rtol=0.1)
        assert_allclose(pen_weight_aicc, 4.77333984, rtol=0.0001)
        assert_allclose(pen_weight_gcv, 4.45546875, rtol=0.0001)
        assert_allclose(pen_weight_bic, 9.35957031, rtol=0.0001)
        assert_allclose(pen_weight_cv, 1.99277344, rtol=0.0001)

    def test_combine_subset_regression(self):
        endog = self.endog
        exog = self.exog
        nobs = len(endog)
        n05 = nobs // 2
        rs = np.random.RandomState(987125)
        shuffle_idx = rs.permutation(np.arange(nobs))
        ys = endog[shuffle_idx]
        xs = exog[shuffle_idx]
        k = 10
        res_ols0 = OLS(ys[:n05], xs[:n05, :k]).fit()
        res_ols1 = OLS(ys[n05:], xs[n05:, :k]).fit()
        w = res_ols1.scale / res_ols0.scale
        mod_1 = TheilGLS(
            ys[n05:],
            xs[n05:, :k],
            r_matrix=np.eye(k),
            q_matrix=res_ols0.params,
            sigma_prior=w * res_ols0.cov_params(),
        )
        res_1p = mod_1.fit(cov_type="data-prior")
        res_1s = mod_1.fit(cov_type="sandwich")
        res_olsf = OLS(ys, xs[:, :k]).fit()
        assert_allclose(res_1p.params, res_olsf.params, rtol=1e-09)
        corr_fact = np.sqrt(res_1p.scale / res_olsf.scale)
        assert_allclose(res_1p.bse, res_olsf.bse * corr_fact, rtol=0.001)
        bse1 = np.array(
            [
                0.26589869,
                0.15224812,
                0.38407399,
                0.75679949,
                0.660842,
                0.5417408,
                0.53697607,
                0.66006377,
                0.38228551,
                0.53920485,
            ]
        )
        assert_allclose(res_1s.bse, bse1, rtol=1e-07)
