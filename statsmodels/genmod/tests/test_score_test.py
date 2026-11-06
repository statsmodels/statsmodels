"""
Created on Thu May 31 15:39:15 2018

Author: Josef Perktold
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy import stats

from statsmodels.base._parameter_inference import score_test
import statsmodels.discrete._diagnostics_count as diac
from statsmodels.discrete.discrete_model import Poisson
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM
import statsmodels.stats._diagnostic_other as diao


def test_score_test_invalid_hypothesis_raises():
    # hypothesis is validated before any use of `self`/`model`, so this
    # is cheap to check without a fitted model.
    with pytest.raises(ValueError, match="hypothesis"):
        score_test(None, hypothesis="not-a-hypothesis")


@pytest.mark.parametrize("model_class", ["poisson", "glm"])
def test_score_test_exog_extra_1d(model_class):
    # a 1-dim exog_extra is one additional column, as in 2-dim with one column
    rng = np.random.default_rng(982345)
    nobs = 300
    x = rng.standard_normal((nobs, 2))
    exog = np.column_stack((np.ones(nobs), x[:, 0]))
    extra = x[:, 1] ** 2
    y = rng.poisson(np.exp(0.3 + 0.2 * exog[:, 1] + 0.1 * extra))
    if model_class == "poisson":
        res = Poisson(y, exog).fit(disp=False)
    else:
        res = GLM(y, exog, family=families.Poisson()).fit()

    res_1d = score_test(res, exog_extra=extra)
    res_2d = score_test(res, exog_extra=extra[:, None])
    assert res_1d.k_constraint == 1
    assert res_2d.k_constraint == 1
    assert_allclose(res_1d.statistic, res_2d.statistic, rtol=1e-12)
    assert_allclose(res_1d.pvalue, res_2d.pvalue, rtol=1e-12)
    res_1d_hc = score_test(res, exog_extra=extra, cov_type="HC0")
    res_2d_hc = score_test(res, exog_extra=extra[:, None], cov_type="HC0")
    assert_allclose(res_1d_hc.statistic, res_2d_hc.statistic, rtol=1e-12)
    assert_allclose(res_1d_hc.pvalue, res_2d_hc.pvalue, rtol=1e-12)
    # the score test of the discrete models uses the same code
    if model_class == "poisson":
        res_method = res.score_test(exog_extra=extra)
        assert_allclose(res_method.statistic, res_1d.statistic, rtol=1e-12)


class CheckScoreTest:

    def test_wald_score(self):
        mod_full = self.model_full
        mod_drop = self.model_drop
        restriction = "x5=0, x6=0"
        res_full = mod_full.fit()
        res_constr = mod_full.fit_constrained("x5=0, x6=0")
        res_drop = mod_drop.fit()

        wald = res_full.wald_test(restriction, scalar=True)
        # note: need to use method for res_constr for correct df_resid
        res = res_constr.score_test()
        lm_constr = np.hstack([res.statistic, res.pvalue, res.k_constraint])
        res = score_test(res_drop, exog_extra=self.exog_extra)
        lm_extra = np.hstack([res.statistic, res.pvalue, res.k_constraint])
        res = res_full.score_test(
            params_constrained=res_constr.params, k_constraints=res_constr.k_constr
        )
        lm_full = np.hstack([res.statistic, res.pvalue, res.k_constraint])

        res_wald = np.hstack([wald.statistic, wald.pvalue, [wald.df_denom]])
        assert_allclose(lm_constr, res_wald, rtol=self.rtol_ws, atol=self.atol_ws)
        assert_allclose(lm_extra, res_wald, rtol=self.rtol_ws, atol=self.atol_ws)
        assert_allclose(lm_constr, lm_extra, rtol=1e-12, atol=1e-14)
        assert_allclose(lm_full, lm_extra, rtol=1e-12, atol=1e-14)
        # regression number
        assert_allclose(lm_constr[1], self.res_pvalue[0], rtol=1e-12, atol=1e-14)

        cov_type = "HC0"
        res_full_hc = mod_full.fit(cov_type=cov_type, start_params=res_full.params)
        wald = res_full_hc.wald_test(restriction, scalar=True)
        _res = score_test(res_constr, cov_type=cov_type)
        lm_constr = np.hstack([_res.statistic, _res.pvalue, _res.k_constraint])
        _res = score_test(res_drop, exog_extra=self.exog_extra, cov_type=cov_type)
        lm_extra = np.hstack([_res.statistic, _res.pvalue, _res.k_constraint])

        res_wald = np.hstack([wald.statistic, wald.pvalue, [wald.df_denom]])
        assert_allclose(lm_constr, res_wald, rtol=self.rtol_ws, atol=self.atol_ws)
        assert_allclose(lm_extra, res_wald, rtol=self.rtol_ws, atol=self.atol_ws)
        assert_allclose(lm_constr, lm_extra, rtol=1e-13)
        # regression number
        assert_allclose(lm_constr[1], self.res_pvalue[1], rtol=1e-12, atol=1e-14)

        if not self.skip_wooldridge:
            # compare with Wooldridge auxiliary regression
            # does not work for Poisson, even with family attribute
            # diao.lm_test_glm assumes fittedvalues is mean (not linear pred)
            lm_wooldridge = diao.lm_test_glm(res_drop, self.exog_extra)
            assert_allclose(
                lm_wooldridge.pval1, self.res_pvalue[0], rtol=1e-12, atol=1e-14
            )
            assert_allclose(
                lm_wooldridge.pval3, self.res_pvalue[1], rtol=self.rtol_wooldridge
            )
            # smoke test
            lm_wooldridge.summary()


class TestScoreTest(CheckScoreTest):
    # compares score to wald, and regression test for pvalues
    rtol_ws = 5e-3
    atol_ws = 0
    rtol_wooldridge = 0.004
    dispersed = False  # Poisson correctly specified
    # regression numbers
    res_pvalue = [0.31786373532550893, 0.32654081685271297]
    skip_wooldridge = False
    res_disptest = np.array(
        [
            [0.1392791916012637, 0.8892295323009857],
            [0.1392791916012645, 0.8892295323009850],
            [0.2129554490802097, 0.8313617120611572],
            [0.1493501809372359, 0.8812773205886350],
            [0.1493501809372359, 0.8812773205886350],
            [0.1454862255574059, 0.8843269904545624],
            [0.2281321688124869, 0.8195434922982738],
        ]
    )
    res_disptest_g = [0.052247629593715761, 0.81919738867722225]

    @classmethod
    def setup_class(cls):
        nobs, k_vars = 500, 5

        np.random.seed(786452)
        x = np.random.randn(nobs, k_vars)
        x[:, 0] = 1
        x2 = np.random.randn(nobs, 2)
        xx = np.column_stack((x, x2))

        if cls.dispersed:
            het = np.random.randn(nobs)
            y = np.random.poisson(np.exp(x.sum(1) * 0.5 + het))
            # y_mc = np.random.negative_binomial(np.exp(x.sum(1) * 0.5), 2)
        else:
            y = np.random.poisson(np.exp(x.sum(1) * 0.5))

        cls.exog_extra = x2
        cls.model_full = GLM(y, xx, family=families.Poisson())
        cls.model_drop = GLM(y, x, family=families.Poisson())

    def test_dispersion(self):
        res_drop = self.model_drop.fit()
        res_test = diac.test_poisson_dispersion(res_drop)
        res_test_ = np.column_stack((res_test.statistic, res_test.pvalue))
        assert_allclose(res_test_, self.res_disptest, rtol=1e-6, atol=1e-14)
        # constant only dispersion
        ex = np.ones((res_drop.model.endog.shape[0], 1))
        # ex = np.column_stack((np.ones(res_drop.model.endog.shape[0]),
        #                      res_drop.predict()))  # or **2
        # dispersion_poisson_generic might not be correct
        # or not clear what the alternative hypothesis is
        # choosing different `ex` implies different alternative hypotheses
        res_test = diac._test_poisson_dispersion_generic(res_drop, ex)
        assert_allclose(res_test, self.res_disptest_g, rtol=1e-6, atol=1e-14)


class TestScoreTestDispersed(TestScoreTest):
    rtol_ws = 0.11
    atol_ws = 0.015
    rtol_wooldridge = 0.03
    dispersed = True  # Poisson is mis-specified
    res_pvalue = [5.412978775609189e-14, 0.05027602575743518]
    res_disptest = np.array(
        [
            [1.2647363371056005e02, 0.0000000000000000e00],
            [1.2647363371056124e02, 0.0000000000000000e00],
            [1.1939362149777617e02, 0.0000000000000000e00],
            [4.5394051864300318e00, 5.6413139746586543e-06],
            [4.5394051864300318e00, 5.6413139746586543e-06],
            [2.9164548934767525e00, 3.5403391013549782e-03],
            [4.2714141112771529e00, 1.9423733575592056e-05],
        ]
    )
    res_disptest_g = [17.670784788586968, 2.6262956791721383e-05]


class TestScoreTestPoisson(TestScoreTest):
    # same as GLM above but for discrete Poisson
    # compares score to wald, and regression test for pvalues
    rtol_ws = 5e-3
    atol_ws = 0
    rtol_wooldridge = 0.004
    dispersed = False  # Poisson correctly specified
    # regression numbers
    res_pvalue = [0.31786373532550893, 0.32654081685271297]
    skip_wooldridge = False
    res_disptest = np.array(
        [
            [0.1392791916012637, 0.8892295323009857],
            [0.1392791916012645, 0.8892295323009850],
            [0.2129554490802097, 0.8313617120611572],
            [0.1493501809372359, 0.8812773205886350],
            [0.1493501809372359, 0.8812773205886350],
            [0.1454862255574059, 0.8843269904545624],
            [0.2281321688124869, 0.8195434922982738],
        ]
    )
    res_disptest_g = [0.052247629593715761, 0.81919738867722225]

    @classmethod
    def setup_class(cls):
        # copy-paste except for model
        nobs, k_vars = 500, 5

        np.random.seed(786452)
        x = np.random.randn(nobs, k_vars)
        x[:, 0] = 1
        x2 = np.random.randn(nobs, 2)
        xx = np.column_stack((x, x2))

        if cls.dispersed:
            het = np.random.randn(nobs)
            y = np.random.poisson(np.exp(x.sum(1) * 0.5 + het))
            # y_mc = np.random.negative_binomial(np.exp(x.sum(1) * 0.5), 2)
        else:
            y = np.random.poisson(np.exp(x.sum(1) * 0.5))

        cls.exog_extra = x2
        cls.model_full = Poisson(y, xx)
        cls.model_drop = Poisson(y, x)

    def test_wald_score(self):
        super().test_wald_score()


class TestScoreTestPoissonDispersed(TestScoreTestPoisson):
    # same as GLM above but for discrete Poisson
    rtol_ws = 0.11
    atol_ws = 0.015
    rtol_wooldridge = 0.03
    dispersed = True  # Poisson is mis-specified
    res_pvalue = [5.412978775609189e-14, 0.05027602575743518]
    res_disptest = np.array(
        [
            [1.2647363371056005e02, 0.0000000000000000e00],
            [1.2647363371056124e02, 0.0000000000000000e00],
            [1.1939362149777617e02, 0.0000000000000000e00],
            [4.5394051864300318e00, 5.6413139746586543e-06],
            [4.5394051864300318e00, 5.6413139746586543e-06],
            [2.9164548934767525e00, 3.5403391013549782e-03],
            [4.2714141112771529e00, 1.9423733575592056e-05],
        ]
    )
    res_disptest_g = [17.670784788586968, 2.6262956791721383e-05]


class TestScoreTestGaussian(CheckScoreTest):
    # compares score to wald, and regression test for pvalues
    rtol_ws = 0.01
    atol_ws = 0
    rtol_wooldridge = 0.06
    dispersed = False
    # regression numbers for nonrobust and HC0
    res_pvalue = [0.44423875090566467, 0.4370837418475849]
    # TODO: check why wooldrige is not very close, which pval1, pval2, pval3 ?
    skip_wooldridge = True

    @classmethod
    def setup_class(cls):
        nobs, k_vars = 500, 5

        np.random.seed(786452)
        x = np.random.randn(nobs, k_vars)
        x[:, 0] = 1
        x2 = np.random.randn(nobs, 2)
        xx = np.column_stack((x, x2))

        if cls.dispersed:
            het = np.random.randn(nobs)
            y = np.random.randn(nobs) + x.sum(1) * 0.5 + het
            # y_mc = np.random.negative_binomial(np.exp(x.sum(1) * 0.5), 2)
        else:
            y = np.random.randn(nobs) + x.sum(1) * 0.5

        cls.exog_extra = x2
        cls.model_full = GLM(y, xx, family=families.Gaussian())
        cls.model_drop = GLM(y, x, family=families.Gaussian())


class TestScoreTestRedundantExtra:
    # GH#5364: columns of exog_extra that are linear combinations of other
    # columns or of exog do not add constraints. The statistic is the same as
    # with the redundant columns dropped by hand, and df is the rank.

    @classmethod
    def setup_class(cls):
        rng = np.random.default_rng(5364)
        nobs = 300
        x = np.column_stack((np.ones(nobs), rng.standard_normal((nobs, 2))))
        y = rng.poisson(np.exp(0.2 + 0.3 * x[:, 1] - 0.2 * x[:, 2]))
        z = rng.standard_normal((nobs, 2))
        cls.exog = x
        cls.exog_extra = z
        # a copy of a column of exog and a combination of the new columns
        cls.exog_extra_redundant = np.column_stack((z, x[:, 1], z[:, 0] - 2 * z[:, 1]))
        cls.res_glm = GLM(y, x, family=families.Poisson()).fit()
        cls.res_poisson = Poisson(y, x).fit(disp=0)
        cls.noise = rng.standard_normal((nobs, 2))

    def test_rank(self):
        from statsmodels.stats.multivariate_tools import partial_project

        resid = partial_project(self.exog_extra_redundant, self.exog).resid
        assert np.linalg.matrix_rank(resid) == 2

    @pytest.mark.parametrize("cov_type", ["nonrobust", "HC0"])
    def test_exact_redundant(self, cov_type):
        for res in (self.res_glm, self.res_poisson):
            st = res.score_test(exog_extra=self.exog_extra, cov_type=cov_type)
            st_red = res.score_test(
                exog_extra=self.exog_extra_redundant, cov_type=cov_type
            )
            assert st_red.k_constraint == 2
            assert_allclose(st_red.statistic, st.statistic, rtol=1e-10)
            assert_allclose(st_red.pvalue, st.pvalue, rtol=1e-10)
            assert_allclose(
                st_red.pvalue, stats.chi2.sf(st_red.statistic, 2), rtol=1e-13
            )

    def test_exact_redundant_model_method(self):
        res = self.res_glm
        st, pv, df = res.model.score_test(res.params, exog_extra=self.exog_extra)
        st_red, pv_red, df_red = res.model.score_test(
            res.params, exog_extra=self.exog_extra_redundant
        )
        assert df == 2
        assert df_red == 2
        assert_allclose(st_red, st, rtol=1e-10)
        assert_allclose(pv_red, pv, rtol=1e-10)
        # k_constraints is added to the rank
        _, _, df_red = res.model.score_test(
            res.params, exog_extra=self.exog_extra_redundant, k_constraints=3
        )
        assert df_red == 5

    def test_nearly_redundant(self):
        # redundant up to noise below the tolerance of R's lm
        exog_extra = self.exog_extra_redundant.copy()
        exog_extra[:, 2:] += 1e-10 * self.noise
        for res in (self.res_glm, self.res_poisson):
            st = res.score_test(exog_extra=self.exog_extra)
            st_near = res.score_test(exog_extra=exog_extra)
            assert st_near.k_constraint == 2
            assert_allclose(st_near.statistic, st.statistic, rtol=1e-6)
            assert_allclose(st_near.pvalue, st.pvalue, rtol=1e-6)

    def test_independent_unchanged(self):
        # without redundancy the columns are used as given
        from statsmodels.base._parameter_inference import _full_rank_exog_extra

        ex, k = _full_rank_exog_extra(self.exog, self.exog_extra)
        assert k == 2
        assert ex is self.exog_extra
        ex, k = _full_rank_exog_extra(self.exog, self.exog_extra[:, 0])
        assert k == 1
        assert ex.shape == (self.exog.shape[0], 1)
        # a column with a small scale is not redundant
        ex, k = _full_rank_exog_extra(self.exog, 1e-12 * self.exog_extra)
        assert k == 2

    def test_all_redundant(self):
        # a column of exog as exog_extra adds no constraint
        for res in (self.res_glm, self.res_poisson):
            st = res.score_test(exog_extra=self.exog[:, 1])
            assert st.k_constraint == 0
            assert_allclose(st.statistic, 0, atol=1e-8)
        st, pv, df = self.res_glm.model.score_test(
            self.res_glm.params, exog_extra=self.exog[:, 1:]
        )
        assert df == 0
        assert_allclose(st, 0, atol=1e-8)

    def test_separate_unchanged(self):
        # the separate z-tests use the columns as given
        st = self.res_glm.score_test(
            exog_extra=self.exog_extra_redundant, hypothesis="separate"
        )
        # z-tests for the 3 parameters of the model and the 4 added columns
        assert st.statistic.shape == (7,)
        st_ind = self.res_glm.score_test(
            exog_extra=self.exog_extra, hypothesis="separate"
        )
        assert st_ind.statistic.shape == (5,)
        assert_allclose(st.statistic[:5], st_ind.statistic, rtol=1e-12)


def test_score_test_spline_basis_rank():
    # the example of GH#5364: the B-spline basis of two variables contains
    # their linear terms, which are already in the model
    from statsmodels.gam.smooth_basis import BSplines
    from statsmodels.stats.multivariate_tools import partial_project

    rng = np.random.default_rng(1234)
    nobs = 250
    x = rng.uniform(-1, 1, size=(nobs, 2))
    exog = np.column_stack((np.ones(nobs), x))
    y = rng.poisson(np.exp(0.5 + x[:, 0] - 0.5 * x[:, 1] ** 2))
    res = GLM(y, exog, family=families.Poisson()).fit()
    bs = BSplines(x, df=[6, 6], degree=[3, 3], include_intercept=False)
    basis = bs.basis
    k_extra = basis.shape[1]
    rank = np.linalg.matrix_rank(partial_project(basis, exog).resid)
    assert rank == k_extra - 2

    st_basis = res.score_test(exog_extra=basis)
    assert st_basis.k_constraint == rank
    # reduce the basis by hand to its principal components as in the issue
    u, s, _ = np.linalg.svd(partial_project(basis, exog).resid, full_matrices=False)
    reduced = u[:, :rank] * s[:rank]
    st_reduced = res.score_test(exog_extra=reduced)
    assert st_reduced.k_constraint == rank
    assert_allclose(st_basis.statistic, st_reduced.statistic, rtol=1e-8)
    assert_allclose(st_basis.pvalue, st_reduced.pvalue, rtol=1e-8)
    assert_allclose(st_basis.pvalue, stats.chi2.sf(st_basis.statistic, rank))
    st, pv, df = res.model.score_test(res.params, exog_extra=basis)
    assert df == rank
    assert_allclose(st, st_basis.statistic, rtol=1e-10)
