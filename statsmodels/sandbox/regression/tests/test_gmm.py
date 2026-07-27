"""

Created on Fri Oct 04 13:19:01 2013

Author: Josef Perktold
"""

from statsmodels.compat.python import lmap, lrange

import copy
import os
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest

from statsmodels.regression.linear_model import OLS
from statsmodels.sandbox.regression import gmm
from statsmodels.tools.tools import add_constant


def get_griliches76_data():
    curdir = os.path.split(__file__)[0]
    path = Path(curdir).joinpath("griliches76.dta")
    griliches76_data = pd.read_stata(path)
    years = griliches76_data["year"].unique()
    N = griliches76_data.shape[0]
    for yr in years:
        griliches76_data["D_%i" % yr] = np.zeros(N)
        for i in range(N):
            if griliches76_data.loc[griliches76_data.index[i], "year"] == yr:
                griliches76_data.loc[griliches76_data.index[i], "D_%i" % yr] = 1
            else:
                pass
    griliches76_data["const"] = 1
    X = add_constant(
        griliches76_data[
            [
                "s",
                "iq",
                "expr",
                "tenure",
                "rns",
                "smsa",
                "D_67",
                "D_68",
                "D_69",
                "D_70",
                "D_71",
                "D_73",
            ]
        ],
        prepend=True,
    )
    Z = add_constant(
        griliches76_data[
            [
                "expr",
                "tenure",
                "rns",
                "smsa",
                "D_67",
                "D_68",
                "D_69",
                "D_70",
                "D_71",
                "D_73",
                "med",
                "kww",
                "age",
                "mrt",
            ]
        ]
    )
    Y = griliches76_data["lw"]
    return (Y, X, Z)


yg_df, xg_df, zg_df = get_griliches76_data()
endog = np.asarray(yg_df, dtype=float)
exog, instrument = lmap(np.asarray, [xg_df, zg_df])
assert exog.dtype == np.float64
assert instrument.dtype == np.float64
varnames = np.array(
    [
        "(Intercept)",
        "s",
        "iq",
        "expr",
        "tenure",
        "rns",
        "smsa",
        "D_67",
        "D_68",
        "D_69",
        "D_70",
        "D_71",
        "D_73",
    ]
)
params = np.array(
    [
        4.03350989,
        0.17242531,
        -0.00909883,
        0.04928949,
        0.04221709,
        -0.10179345,
        0.12611095,
        -0.05961711,
        0.04867956,
        0.15281763,
        0.17443605,
        0.09166597,
        0.09323977,
    ]
)
bse = np.array(
    [
        0.31816162,
        0.02091823,
        0.00474527,
        0.00822543,
        0.00891969,
        0.03447337,
        0.03119615,
        0.05577582,
        0.05246796,
        0.05201092,
        0.06027671,
        0.05461436,
        0.05767865,
    ]
)
tvalues = np.array(
    [
        12.6775501,
        8.2428242,
        -1.9174531,
        5.9923305,
        4.7330205,
        -2.9528144,
        4.0425165,
        -1.0688701,
        0.9277959,
        2.9381834,
        2.8939212,
        1.6784225,
        1.6165385,
    ]
)
pvalues = np.array(
    [
        1.7236e-33,
        7.570254e-16,
        0.0555625,
        3.219967e-09,
        2.647391e-06,
        0.003247941,
        5.838099e-05,
        0.2854744,
        0.3538139,
        0.003403361,
        0.003915751,
        0.09368402,
        0.1064013,
    ]
)


def test_iv2sls_r():
    mod = gmm.IV2SLS(endog, exog, instrument)
    res = mod.fit()
    n, k = exog.shape
    assert_allclose(res.params, params, rtol=1e-07, atol=1e-09)
    assert_allclose(res.bse, bse, rtol=0, atol=3e-07)
    assert not hasattr(mod, "_results")


def test_ivgmm0_r():
    n, k = exog.shape
    nobs, k_instr = instrument.shape
    w0inv = np.dot(instrument.T, instrument) / nobs
    w0 = np.linalg.inv(w0inv)
    mod = gmm.IVGMM(endog, exog, instrument)
    res = mod.fit(
        np.ones(exog.shape[1], float),
        maxiter=0,
        inv_weights=w0inv,
        optim_method="bfgs",
        optim_args={"gtol": 1e-08, "disp": 0},
    )
    assert_allclose(res.params, params, rtol=0.0001, atol=0.0001)
    assert_allclose(res.bse, bse, rtol=0.09, atol=0)
    score = res.model.score(res.params, w0)
    assert_allclose(score, np.zeros(score.shape), rtol=0, atol=5e-06)


def test_ivgmm1_stata():
    np.array(
        [
            4.0335099,
            0.17242531,
            -0.00909883,
            0.04928949,
            0.04221709,
            -0.10179345,
            0.12611095,
            -0.05961711,
            0.04867956,
            0.15281763,
            0.17443605,
            0.09166597,
            0.09323976,
        ]
    )
    np.array(
        [
            0.33503289,
            0.02073947,
            0.00488624,
            0.0080498,
            0.00946363,
            0.03371053,
            0.03081138,
            0.05171372,
            0.04981322,
            0.0479285,
            0.06112515,
            0.0554618,
            0.06084901,
        ]
    )
    n, k = exog.shape
    nobs, k_instr = instrument.shape
    w0inv = np.dot(instrument.T, instrument) / nobs
    np.linalg.inv(w0inv)
    start = OLS(endog, exog).fit().params
    mod = gmm.IVGMM(endog, exog, instrument)
    mod.fit(
        start,
        maxiter=1,
        inv_weights=w0inv,
        optim_method="bfgs",
        optim_args={"gtol": 1e-06, "disp": 0},
    )


idx = lrange(len(params))
idx = idx[1:] + idx[:1]
exog_st = exog[:, idx]


class TestGMMOLS:

    @classmethod
    def setup_class(cls):
        exog = exog_st
        res_ols = OLS(endog, exog).fit()
        nobs, k_instr = exog.shape
        w0inv = np.dot(exog.T, exog) / nobs
        mod = gmm.IVGMM(endog, exog, exog)
        res = mod.fit(
            np.ones(exog.shape[1], float),
            maxiter=0,
            inv_weights=w0inv,
            optim_method="bfgs",
            optim_args={"gtol": 1e-06, "disp": 0},
        )
        cls.res1 = res
        cls.res2 = res_ols

    def test_basic(self):
        res1, res2 = (self.res1, self.res2)
        assert_allclose(res1.params, res2.params, rtol=0.0005, atol=0)
        assert_allclose(res1.params, res2.params, rtol=0, atol=1e-05)
        res1.model.exog.shape[0]
        dffac = 1
        assert_allclose(res1.bse * dffac, res2.HC0_se, rtol=5e-06, atol=0)
        assert_allclose(res1.bse * dffac, res2.HC0_se, rtol=0, atol=1e-07)

    @pytest.mark.xfail(
        reason="Not asserting anything meaningful",
        raises=NotImplementedError,
        strict=True,
    )
    def test_other(self):
        raise NotImplementedError


class CheckGMM:
    params_tol = [5e-06, 5e-06]
    bse_tol = [5e-07, 5e-07]

    def test_basic(self):
        res1, res2 = (self.res1, self.res2)
        rtol, atol = self.params_tol
        assert_allclose(res1.params, res2.params, rtol=rtol, atol=0)
        assert_allclose(res1.params, res2.params, rtol=0, atol=atol)
        res1.model.exog.shape[0]
        dffac = 1
        rtol, atol = self.bse_tol
        assert_allclose(res1.bse * dffac, res2.bse, rtol=rtol, atol=0)
        assert_allclose(res1.bse * dffac, res2.bse, rtol=0, atol=atol)

    def test_other(self):
        res1, res2 = (self.res1, self.res2)
        assert_allclose(res1.q, res2.Q, rtol=5e-06, atol=0)
        assert_allclose(res1.jval, res2.J, rtol=5e-05, atol=0)

    def test_hypothesis(self):
        res1, res2 = (self.res1, self.res2)
        restriction = np.eye(len(res1.params))
        res_t = res1.t_test(restriction)
        assert_allclose(res_t.tvalue, res1.tvalues, rtol=1e-12, atol=0)
        assert_allclose(res_t.pvalue, res1.pvalues, rtol=1e-12, atol=0)
        rtol, atol = self.bse_tol
        assert_allclose(res_t.tvalue, res2.tvalues, rtol=rtol * 10, atol=atol)
        assert_allclose(res_t.pvalue, res2.pvalues, rtol=rtol * 10, atol=atol)
        res1.f_test(restriction[:-1])
        res1.wald_test(restriction[:-1], scalar=True)

    @pytest.mark.smoke
    def test_summary(self):
        res1 = self.res1
        summ = res1.summary()
        assert_equal(len(summ.tables[1]), len(res1.params) + 1)

    def test_use_t(self):
        res1 = copy.deepcopy(self.res1)
        res1.use_t = True
        summ = res1.summary()
        assert "P>|t|" in str(summ)
        assert "P>|z|" not in str(summ)


class TestGMMSt1(CheckGMM):

    @classmethod
    def setup_class(cls):
        exog = exog_st
        start = OLS(endog, exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs
        mod = gmm.IVGMM(endog, exog, instrument)
        res10 = mod.fit(
            start,
            maxiter=10,
            inv_weights=w0inv,
            optim_method="bfgs",
            optim_args={"gtol": 1e-06, "disp": 0},
            wargs={"centered": False},
        )
        cls.res1 = res10
        from .results_gmm_griliches_iter import results

        cls.res2 = results


class TestGMMStTwostep(CheckGMM):

    @classmethod
    def setup_class(cls):
        cls.params_tol = [5e-05, 5e-06]
        cls.bse_tol = [5e-06, 5e-07]
        exog = exog_st
        start = OLS(endog, exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs
        mod = gmm.IVGMM(endog, exog, instrument)
        res10 = mod.fit(
            start,
            maxiter=2,
            inv_weights=w0inv,
            optim_method="bfgs",
            optim_args={"gtol": 1e-06, "disp": 0},
            wargs={"centered": False},
        )
        cls.res1 = res10
        from .results_gmm_griliches import results_twostep as results

        cls.res2 = results


class TestGMMStTwostepNO(CheckGMM):

    @classmethod
    def setup_class(cls):
        cls.params_tol = [5e-05, 5e-06]
        cls.bse_tol = [1e-06, 5e-05]
        exog = exog_st
        start = OLS(endog, exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs
        mod = gmm.IVGMM(endog, exog, instrument)
        res10 = mod.fit(
            start,
            maxiter=2,
            inv_weights=w0inv,
            optim_method="bfgs",
            optim_args={"gtol": 1e-06, "disp": 0},
            wargs={"centered": False},
            has_optimal_weights=False,
        )
        cls.res1 = res10
        from .results_gmm_griliches import results_twostep as results

        cls.res2 = results


class TestGMMStOnestep(CheckGMM):

    @classmethod
    def setup_class(cls):
        cls.params_tol = [0.0005, 5e-05]
        cls.bse_tol = [0.007, 0.0005]
        exog = exog_st
        start = OLS(endog, exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs
        mod = gmm.IVGMM(endog, exog, instrument)
        res = mod.fit(
            start,
            maxiter=0,
            inv_weights=w0inv,
            optim_method="bfgs",
            optim_args={"gtol": 1e-06, "disp": 0},
        )
        cls.res1 = res
        from .results_gmm_griliches import results_onestep as results

        cls.res2 = results

    def test_bse_other(self):
        res1 = self.res1
        np.sqrt(np.diag(res1._cov_params(has_optimal_weights=False)))
        self.res1.model.gmmobjective(self.res1.params, np.linalg.inv(self.res1.weights))

    @pytest.mark.xfail(
        reason="q vs Q comparison fails", raises=AssertionError, strict=True
    )
    def test_other(self):
        super().test_other()


class TestGMMStOnestepNO(CheckGMM):

    @classmethod
    def setup_class(cls):
        cls.params_tol = [1e-05, 1e-06]
        cls.bse_tol = [5e-06, 5e-07]
        exog = exog_st
        start = OLS(endog, exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs
        mod = gmm.IVGMM(endog, exog, instrument)
        res = mod.fit(
            start,
            maxiter=0,
            inv_weights=w0inv,
            optim_method="bfgs",
            optim_args={"gtol": 1e-06, "disp": 0},
            wargs={"centered": False},
            has_optimal_weights=False,
        )
        cls.res1 = res
        from .results_gmm_griliches import results_onestep as results

        cls.res2 = results

    @pytest.mark.xfail(
        reason="q vs Q comparison fails", raises=AssertionError, strict=True
    )
    def test_other(self):
        super().test_other()


class TestGMMStOneiter(CheckGMM):

    @classmethod
    def setup_class(cls):
        cls.params_tol = [0.0005, 5e-05]
        cls.bse_tol = [0.007, 0.0005]
        exog = exog_st
        start = OLS(endog, exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs
        mod = gmm.IVGMM(endog, exog, instrument)
        res = mod.fit(
            start,
            maxiter=1,
            inv_weights=w0inv,
            optim_method="bfgs",
            optim_args={"gtol": 1e-06, "disp": 0},
        )
        cls.res1 = res
        from .results_gmm_griliches import results_onestep as results

        cls.res2 = results

    @pytest.mark.xfail(
        reason="q vs Q comparison fails", raises=AssertionError, strict=True
    )
    def test_other(self):
        super().test_other()

    def test_bse_other(self):
        res1 = self.res1
        moms = res1.model.momcond(res1.params)
        res1.model.calc_weightmatrix(moms)
        np.sqrt(
            np.diag(res1._cov_params(has_optimal_weights=False, weights=res1.weights))
        )
        np.sqrt(np.diag(res1._cov_params(has_optimal_weights=False)))


class TestGMMStOneiterNO(CheckGMM):

    @classmethod
    def setup_class(cls):
        cls.params_tol = [1e-05, 1e-06]
        cls.bse_tol = [5e-06, 5e-07]
        exog = exog_st
        start = OLS(endog, exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs
        mod = gmm.IVGMM(endog, exog, instrument)
        res = mod.fit(
            start,
            maxiter=1,
            inv_weights=w0inv,
            optim_method="bfgs",
            optim_args={"gtol": 1e-06, "disp": 0},
            wargs={"centered": False},
            has_optimal_weights=False,
        )
        cls.res1 = res
        from .results_gmm_griliches import results_onestep as results

        cls.res2 = results

    @pytest.mark.xfail(
        reason="q vs Q comparison fails", raises=AssertionError, strict=True
    )
    def test_other(self):
        super().test_other()


class TestGMMStOneiterNO_Linear(CheckGMM):

    @classmethod
    def setup_class(cls):
        cls.params_tol = [5e-09, 1e-09]
        cls.bse_tol = [5e-10, 1e-10]
        exog = exog_st
        start = OLS(endog, exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs
        mod = gmm.LinearIVGMM(endog, exog, instrument)
        res = mod.fit(
            start,
            maxiter=1,
            inv_weights=w0inv,
            optim_method="bfgs",
            optim_args={"gtol": 1e-08, "disp": 0},
            wargs={"centered": False},
            has_optimal_weights=False,
        )
        cls.res1 = res
        mod = gmm.IVGMM(endog, exog, instrument)
        res = mod.fit(
            start,
            maxiter=1,
            inv_weights=w0inv,
            optim_method="bfgs",
            optim_args={"gtol": 1e-06, "disp": 0},
            wargs={"centered": False},
            has_optimal_weights=False,
        )
        cls.res3 = res
        from .results_gmm_griliches import results_onestep as results

        cls.res2 = results

    @pytest.mark.xfail(
        reason="q vs Q comparison fails", raises=AssertionError, strict=True
    )
    def test_other(self):
        super().test_other()


class TestGMMStOneiterNO_Nonlinear(CheckGMM):

    @classmethod
    def setup_class(cls):
        cls.params_tol = [5e-05, 5e-06]
        cls.bse_tol = [5e-06, 0.1]
        exog = exog_st
        start = OLS(endog, exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs

        def func(params, exog):
            return np.dot(exog, params)

        mod = gmm.NonlinearIVGMM(endog, exog, instrument, func)
        res = mod.fit(
            start,
            maxiter=1,
            inv_weights=w0inv,
            optim_method="bfgs",
            optim_args={"gtol": 1e-08, "disp": 0},
            wargs={"centered": False},
            has_optimal_weights=False,
        )
        cls.res1 = res
        mod = gmm.IVGMM(endog, exog, instrument)
        res = mod.fit(
            start,
            maxiter=1,
            inv_weights=w0inv,
            optim_method="bfgs",
            optim_args={"gtol": 1e-06, "disp": 0},
            wargs={"centered": False},
            has_optimal_weights=False,
        )
        cls.res3 = res
        from .results_gmm_griliches import results_onestep as results

        cls.res2 = results

    @pytest.mark.xfail(
        reason="q vs Q comparison fails", raises=AssertionError, strict=True
    )
    def test_other(self):
        super().test_other()

    def test_score(self):
        params = self.res1.params * 1.1
        weights = self.res1.weights
        sc1 = self.res1.model.score(params, weights)
        sc2 = super(self.res1.model.__class__, self.res1.model).score(params, weights)
        assert_allclose(sc1, sc2, rtol=1e-06, atol=0)
        assert_allclose(sc1, sc2, rtol=0, atol=1e-07)
        sc1 = self.res1.model.score(self.res1.params, weights)
        assert_allclose(sc1, np.zeros(len(params)), rtol=0, atol=1e-08)


class TestGMMStOneiterOLS_Linear(CheckGMM):

    @classmethod
    def setup_class(cls):
        cls.params_tol = [1e-11, 1e-12]
        cls.bse_tol = [1e-12, 1e-12]
        exog = exog_st
        res_ols = OLS(endog, exog).fit()
        start = np.ones(len(res_ols.params))
        nobs, k_instr = instrument.shape
        w0inv = np.dot(exog.T, exog) / nobs
        mod = gmm.LinearIVGMM(endog, exog, exog)
        res = mod.fit(
            start,
            maxiter=0,
            inv_weights=w0inv,
            optim_args={"disp": 0},
            weights_method="iid",
            wargs={"centered": False, "ddof": "k_params"},
            has_optimal_weights=True,
        )
        res.use_t = True
        res.df_resid = res.nobs - len(res.params)
        cls.res1 = res
        cls.res2 = res_ols

    @pytest.mark.xfail(
        reason="RegressionResults has no `Q` attribute",
        raises=AttributeError,
        strict=True,
    )
    def test_other(self):
        super().test_other()


class TestGMMSt2:

    @classmethod
    def setup_class(cls):
        exog = exog_st
        start = OLS(endog, exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs
        mod = gmm.IVGMM(endog, exog, instrument)
        res = mod.fit(
            start,
            maxiter=2,
            inv_weights=w0inv,
            wargs={"ddof": 0, "centered": False},
            optim_method="bfgs",
            optim_args={"gtol": 1e-06, "disp": 0},
        )
        cls.res1 = res
        from .results_ivreg2_griliches import results_gmm2s_robust as results

        cls.res2 = results
        mod = gmm.IVGMM(endog, exog, instrument)
        res = mod.fit(
            start,
            maxiter=1,
            inv_weights=w0inv,
            wargs={"ddof": 0, "centered": False},
            optim_method="bfgs",
            optim_args={"gtol": 1e-06, "disp": 0},
        )
        cls.res3 = res

    def test_basic(self):
        res1, res2 = (self.res1, self.res2)
        assert_allclose(res1.params, res2.params, rtol=5e-05, atol=0)
        assert_allclose(res1.params, res2.params, rtol=0, atol=5e-06)
        dffact = np.sqrt(745.0 / 758)
        assert_allclose(res1.bse * dffact, res2.bse, rtol=0.005, atol=0)
        assert_allclose(res1.bse * dffact, res2.bse, rtol=0, atol=0.005)
        np.sqrt(
            np.diag(res1._cov_params(has_optimal_weights=True, weights=res1.weights))
        )
        assert_allclose(res1.bse, res2.bse, rtol=0.5, atol=0)
        np.sqrt(
            np.diag(
                res1._cov_params(
                    has_optimal_weights=True, weights=res1.weights, use_weights=True
                )
            )
        )
        assert_allclose(res1.bse, res2.bse, rtol=0.05, atol=0)
        assert_allclose(self.res3.bse, res2.bse, rtol=5e-05, atol=0)
        assert_allclose(self.res3.bse, res2.bse, rtol=0, atol=5e-06)


class CheckIV2SLS:

    def test_basic(self):
        res1, res2 = (self.res1, self.res2)
        assert_allclose(res1.params, res2.params, rtol=1e-09, atol=0)
        assert_allclose(res1.params, res2.params, rtol=0, atol=1e-10)
        assert_allclose(res1.bse, res2.bse, rtol=1e-10, atol=0)
        assert_allclose(res1.bse, res2.bse, rtol=0, atol=1e-11)
        assert_allclose(res1.tvalues, res2.tvalues, rtol=5e-10, atol=0)

    def test_other(self):
        res1, res2 = (self.res1, self.res2)
        assert_allclose(res1.rsquared, res2.r2, rtol=1e-07, atol=0)
        assert_allclose(res1.rsquared_adj, res2.r2_a, rtol=1e-07, atol=0)
        assert_allclose(res1.fvalue, res2.F, rtol=1e-10, atol=0)
        assert_allclose(res1.f_pvalue, res2.Fp, rtol=1e-08, atol=0)
        assert_allclose(np.sqrt(res1.mse_resid), res2.rmse, rtol=1e-10, atol=0)
        assert_allclose(res1.ssr, res2.rss, rtol=1e-10, atol=0)
        assert_allclose(res1.uncentered_tss, res2.yy, rtol=1e-10, atol=0)
        assert_allclose(res1.centered_tss, res2.yyc, rtol=1e-10, atol=0)
        assert_allclose(res1.ess, res2.mss, rtol=1e-09, atol=0)
        assert_equal(res1.df_model, res2.df_m)
        assert_equal(res1.df_resid, res2.df_r)

    def test_hypothesis(self):
        res1, res2 = (self.res1, self.res2)
        restriction = np.eye(len(res1.params))
        res_t = res1.t_test(restriction)
        assert_allclose(res_t.tvalue, res1.tvalues, rtol=1e-12, atol=0)
        assert_allclose(res_t.pvalue, res1.pvalues, rtol=1e-12, atol=0)
        res_f = res1.f_test(restriction[:-1])
        assert_allclose(res_f.fvalue, res1.fvalue, rtol=1e-12, atol=0)
        assert_allclose(res_f.pvalue, res1.f_pvalue, rtol=1e-10, atol=0)
        assert_allclose(res_f.fvalue, res2.F, rtol=1e-10, atol=0)
        assert_allclose(res_f.pvalue, res2.Fp, rtol=1e-08, atol=0)

    def test_hausman(self):
        res1, res2 = (self.res1, self.res2)
        hausm = res1.spec_hausman()
        assert_allclose(hausm[0], res2.hausman["DWH"], rtol=1e-11, atol=0)
        assert_allclose(hausm[1], res2.hausman["DWHp"], rtol=1e-10, atol=1e-25)

    @pytest.mark.smoke
    def test_summary(self):
        res1 = self.res1
        summ = res1.summary()
        assert_equal(len(summ.tables[1]), len(res1.params) + 1)


class TestIV2SLSSt1(CheckIV2SLS):

    @classmethod
    def setup_class(cls):
        exog = exog_st
        mod = gmm.IV2SLS(endog, exog, instrument)
        res = mod.fit()
        cls.res1 = res
        from .results_ivreg2_griliches import results_small as results

        cls.res2 = results

    def test_input_dimensions(self):
        rs = np.random.RandomState(1234)
        x = rs.randn(200, 2)
        z = rs.randn(200)
        x[:, 0] = np.sqrt(0.5) * x[:, 0] + np.sqrt(0.5) * z
        z = np.column_stack((x[:, [1]], z[:, None]))
        e = np.sqrt(0.5) * rs.randn(200) + np.sqrt(0.5) * x[:, 0]
        y_1d = y = x[:, 0] + x[:, 1] + e
        y_2d = y[:, None]
        y_series = pd.Series(y)
        y_df = pd.DataFrame(y_series)
        x_1d = x[:, 0]
        x_2d = x
        x_df = pd.DataFrame(x)
        x_df_single = x_df.iloc[:, [0]]
        x_series = x_df.iloc[:, 0]
        z_2d = z
        z_series = pd.Series(z[:, 1])
        z_1d = z_series.values
        z_df = pd.DataFrame(z)
        ys = (y_df, y_series, y_2d, y_1d)
        xs = (x_2d, x_1d, x_df_single, x_df, x_series)
        zs = (z_1d, z_2d, z_series, z_df)
        res2 = gmm.IV2SLS(y_1d, x_2d, z_2d).fit()
        res1 = gmm.IV2SLS(y_1d, x_1d, z_1d).fit()
        res1_2sintr = gmm.IV2SLS(y_1d, x_1d, z_2d).fit()
        for _y in ys:
            for _x in xs:
                for _z in zs:
                    x_1d = np.size(_x) == _x.shape[0]
                    z_1d = np.size(_z) == _z.shape[0]
                    if z_1d and (not x_1d):
                        continue
                    res = gmm.IV2SLS(_y, _x, _z).fit()
                    if z_1d:
                        assert_allclose(res.params, res1.params)
                    elif x_1d and (not z_1d):
                        assert_allclose(res.params, res1_2sintr.params)
                    else:
                        assert_allclose(res.params, res2.params)


def test_noconstant():
    exog = exog_st[:, :-1]
    mod = gmm.IV2SLS(endog, exog, instrument)
    res = mod.fit()
    assert_equal(res.fvalue, np.nan)
    summ = res.summary()
    assert_equal(len(summ.tables[1]), len(res.params) + 1)


def test_gmm_basic():
    cd = np.array(
        [
            1.5,
            1.5,
            1.7,
            2.2,
            2.0,
            1.8,
            1.8,
            2.2,
            1.9,
            1.6,
            1.8,
            2.2,
            2.0,
            1.5,
            1.1,
            1.5,
            1.4,
            1.7,
            1.42,
            1.9,
        ]
    )
    dcd = np.array(
        [
            0,
            0.2,
            0.5,
            -0.2,
            -0.2,
            0,
            0.4,
            -0.3,
            -0.3,
            0.2,
            0.4,
            -0.2,
            -0.5,
            -0.4,
            0.4,
            -0.1,
            0.3,
            -0.28,
            0.48,
            0.2,
        ]
    )
    inst = np.column_stack((np.ones(len(cd)), cd))

    class GMMbase(gmm.GMM):

        def momcond(self, params):
            p0, p1, p2, p3 = params
            endog = self.endog[:, None]
            exog = self.exog
            inst = self.instrument
            mom0 = (endog - p0 - p1 * exog) * inst
            mom1 = ((endog - p0 - p1 * exog) ** 2 - p2 * exog ** (2 * p3) / 12) * inst
            g = np.column_stack((mom0, mom1))
            return g

    beta0 = np.array([0.1, 0.1, 0.01, 1])
    res = GMMbase(endog=dcd, exog=cd, instrument=inst, k_moms=4, k_params=4).fit(
        beta0, optim_args={"disp": 0}
    )
    summ = res.summary()
    assert_equal(len(summ.tables[1]), len(res.params) + 1)
    pnames = ["p%2d" % i for i in range(len(res.params))]
    assert_equal(res.model.exog_names, pnames)
    mod = GMMbase(endog=dcd, exog=cd, instrument=inst, k_moms=4, k_params=4)
    pnames = ["beta", "gamma", "psi", "phi"]
    mod.set_param_names(pnames)
    res1 = mod.fit(beta0, optim_args={"disp": 0})
    assert_equal(res1.model.exog_names, pnames)
