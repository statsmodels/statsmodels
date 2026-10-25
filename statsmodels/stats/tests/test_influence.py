"""
Created on Tue Jun 12 13:18:12 2018

Author: Josef Perktold
"""
from statsmodels.compat.pandas import testing as pdt

from pathlib import Path
import warnings

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.regression.linear_model import GLS, OLS, WLS
from statsmodels.stats.outliers_influence import (
    GLMInfluence,
    MLEInfluence,
    OLSInfluence,
    variance_inflation_factor,
)
from statsmodels.tools.sm_exceptions import SingularMatrixWarning

cur_dir = Path(__file__).parent.resolve()

file_name = "binary_constrict.csv"
file_path = Path(cur_dir).joinpath("results", file_name)
data_bin = pd.read_csv(file_path, index_col=0)

file_name = "results_influence_logit.csv"
file_path = Path(cur_dir).joinpath("results", file_name)
results_sas_df = pd.read_csv(file_path, index_col=0)


def test_influence_glm_bernoulli():
    # example uses Finney's data and is used in Pregibon 1981

    df = data_bin
    results_sas = np.asarray(results_sas_df)

    res = GLM(
        df["constrict"],
        df[["const", "log_rate", "log_volumne"]],
        family=families.Binomial(),
    ).fit(attach_wls=True, atol=1e-10)

    infl = res.get_influence(observed=False)

    k_vars = 3
    assert_allclose(infl.dfbetas, results_sas[:, 5:8], atol=1e-4)
    assert_allclose(infl.d_params, results_sas[:, 5:8] * res.bse.values, atol=1e-4)
    assert_allclose(infl.cooks_distance[0] * k_vars, results_sas[:, 8], atol=6e-5)
    assert_allclose(infl.hat_matrix_diag, results_sas[:, 4], atol=6e-5)

    c_bar = infl.cooks_distance[0] * 3 * (1 - infl.hat_matrix_diag)
    assert_allclose(c_bar, results_sas[:, 9], atol=6e-5)


def test_glminfluence_direct_constructor():
    # GH#9415: GLMInfluence constructed directly raised AttributeError
    # on cooks_distance
    df = data_bin
    res = GLM(
        df["constrict"],
        df[["const", "log_rate", "log_volumne"]],
        family=families.Binomial(),
    ).fit(attach_wls=True, atol=1e-10)

    infl_direct = GLMInfluence(res)
    infl_method = res.get_influence()

    assert_allclose(infl_direct.cooks_distance[0],
                    infl_method.cooks_distance[0], rtol=1e-12)


class InfluenceCompareExact:
    # Mixin to compare and test two Influence instances

    def test_basics(self):
        infl1 = self.infl1
        infl0 = self.infl0

        assert_allclose(infl0.hat_matrix_diag, infl1.hat_matrix_diag, rtol=1e-12)

        assert_allclose(
            infl0.resid_studentized, infl1.resid_studentized, rtol=1e-12, atol=1e-7
        )

        cd_rtol = getattr(self, "cd_rtol", 1e-7)
        assert_allclose(
            infl0.cooks_distance[0], infl1.cooks_distance[0], rtol=cd_rtol, atol=1e-14
        )  # very small values possible
        assert_allclose(infl0.dfbetas, infl1.dfbetas, rtol=1e-9, atol=5e-9)
        assert_allclose(infl0.d_params, infl1.d_params, rtol=1e-9, atol=5e-9)
        assert_allclose(
            infl0.d_fittedvalues, infl1.d_fittedvalues, rtol=5e-9, atol=1e-14
        )
        assert_allclose(
            infl0.d_fittedvalues_scaled,
            infl1.d_fittedvalues_scaled,
            rtol=5e-9,
            atol=1e-14,
        )

    @pytest.mark.smoke
    @pytest.mark.thread_unsafe(reason="Uses matplotlib")
    @pytest.mark.matplotlib
    def test_plots(self, close_figures):
        import matplotlib.pyplot as plt

        infl1 = self.infl1
        infl0 = self.infl0

        infl0.plot_influence(external=False)
        infl1.plot_influence(external=False)

        infl0.plot_index("resid", threshold=0.2, title="")
        infl1.plot_index("resid", threshold=0.2, title="")
        plt.close("all")

        infl0.plot_index("dfbeta", idx=1, threshold=0.2, title="")
        infl1.plot_index("dfbeta", idx=1, threshold=0.2, title="")

        infl0.plot_index("cook", idx=1, threshold=0.2, title="")
        infl1.plot_index("cook", idx=1, threshold=0.2, title="")
        plt.close("all")

        infl0.plot_index("hat", idx=1, threshold=0.2, title="")
        infl1.plot_index("hat", idx=1, threshold=0.2, title="")

    def test_summary(self):
        infl1 = self.infl1
        infl0 = self.infl0

        df0 = infl0.summary_frame()
        df1 = infl1.summary_frame()
        assert_allclose(df0.values, df1.values, rtol=5e-5, atol=1e-14)
        pdt.assert_index_equal(df0.index, df1.index)


def _check_looo(self):
    infl = self.infl1
    # unwrap if needed
    results = getattr(infl.results, "_results", infl.results)

    res_looo = infl._res_looo
    mask_infl = infl.cooks_distance[0] > 2 * infl.cooks_distance[0].std()
    mask_low = ~mask_infl
    diff_params = results.params - res_looo["params"]
    assert_allclose(infl.d_params[mask_low], diff_params[mask_low], atol=0.05)
    assert_allclose(infl.params_one[mask_low], res_looo["params"][mask_low], rtol=0.01)


class TestInfluenceLogitGLMMLE(InfluenceCompareExact):

    @classmethod
    def setup_class(cls):
        df = data_bin
        res = GLM(
            df["constrict"],
            df[["const", "log_rate", "log_volumne"]],
            family=families.Binomial(),
        ).fit(attach_wls=True, atol=1e-10)

        cls.infl1 = res.get_influence()
        cls.infl0 = MLEInfluence(res)

    def test_looo(self):
        _check_looo(self)


class TestInfluenceBinomialGLMMLE(InfluenceCompareExact):
    # example based on Williams and R docs

    @classmethod
    def setup_class(cls):
        yi = np.array([0, 2, 14, 19, 30])
        ni = 40 * np.ones(len(yi))
        xi = np.arange(1, len(yi) + 1)
        exog = np.column_stack((np.ones(len(yi)), xi))
        endog = np.column_stack((yi, ni - yi))

        res = GLM(endog, exog, family=families.Binomial()).fit()

        cls.infl1 = res.get_influence()
        cls.infl0 = MLEInfluence(res)
        cls.cd_rtol = 5e-5

    def test_looo(self):
        _check_looo(self)

    def test_r(self):
        # values from R,
        # > xi <- 1:5
        # > yi <- c(0,2,14,19,30)    # number of mice responding to dose xi
        # > mi <- rep(40, 5)         # number of mice exposed
        # > glmI <- glm(cbind(yi, mi -yi) ~ xi, family = binomial)
        # > imI <- influence.measures(glmI)
        # > t(imI$infmat)

        # dfbeta/dfbetas and dffits do not make sense to me and are furthe away from
        # looo than mine
        # resid seem to be resid_deviance based and not resid_pearson
        # I did not compare cov.r
        infl1 = self.infl1
        cooks_d = [
            0.25220202795934726,
            0.26107981497746285,
            1.28985614424132389,
            0.08449722285516942,
            0.36362110845918005,
        ]
        hat = [
            0.2594393406119333,
            0.3696442663244837,
            0.3535768402250521,
            0.389209198535791057,
            0.6281303543027403,
        ]

        assert_allclose(infl1.hat_matrix_diag, hat, rtol=5e-6)
        assert_allclose(infl1.cooks_distance[0], cooks_d, rtol=1e-5)


class TestInfluenceGaussianGLMMLE(InfluenceCompareExact):

    @classmethod
    def setup_class(cls):
        from .test_diagnostic import get_duncan_data

        endog, exog, labels = get_duncan_data()
        data = pd.DataFrame(
            np.column_stack((endog, exog)),
            columns="y const var1 var2".split(),
            index=labels,
        )

        res = GLM.from_formula("y ~ const + var1 + var2 - 1", data).fit()
        # res = GLM(endog, exog).fit()

        cls.infl1 = res.get_influence()
        cls.infl0 = MLEInfluence(res)

    def test_looo(self):
        _check_looo(self)


class TestInfluenceGaussianGLMOLS(InfluenceCompareExact):

    @classmethod
    def setup_class(cls):
        from .test_diagnostic import get_duncan_data

        endog, exog, labels = get_duncan_data()
        data = pd.DataFrame(
            np.column_stack((endog, exog)),
            columns="y const var1 var2".split(),
            index=labels,
        )

        res0 = GLM.from_formula("y ~ const + var1 + var2 - 1", data).fit()
        res1 = OLS.from_formula("y ~ const + var1 + var2 - 1", data).fit()
        cls.infl1 = res1.get_influence()
        cls.infl0 = res0.get_influence()

    def test_basics(self):
        # needs to override attributes that are not equivalent,
        # i.e., not available or different definition like external vs internal
        infl1 = self.infl1
        infl0 = self.infl0

        assert_allclose(infl0.hat_matrix_diag, infl1.hat_matrix_diag, rtol=1e-12)
        assert_allclose(
            infl0.resid_studentized, infl1.resid_studentized, rtol=1e-12, atol=1e-7
        )
        assert_allclose(
            infl0.cooks_distance, infl1.cooks_distance, rtol=1e-7, atol=1e-14
        )  # very small values possible
        assert_allclose(infl0.dfbetas, infl1.dfbetas, rtol=0.1)  # changed
        # OLSInfluence only has looo dfbeta/d_params
        assert_allclose(infl0.d_params, infl1.dfbeta, rtol=1e-9, atol=1e-14)
        # d_fittedvalues is not available in OLSInfluence, i.e., only scaled dffits
        # assert_allclose(infl0.d_fittedvalues, infl1.d_fittedvalues, rtol=1e-9)
        assert_allclose(
            infl0.d_fittedvalues_scaled, infl1.dffits_internal[0], rtol=1e-9
        )

        # specific to linear link
        assert_allclose(infl0.d_linpred, infl0.d_fittedvalues, rtol=1e-12)
        assert_allclose(infl0.d_linpred_scaled, infl0.d_fittedvalues_scaled, rtol=1e-12)

    def test_summary(self):
        infl1 = self.infl1
        infl0 = self.infl0

        df0 = infl0.summary_frame()
        df1 = infl1.summary_frame()
        # just some basic check on overlap except for dfbetas
        cols = ["cooks_d", "standard_resid", "hat_diag", "dffits_internal"]
        assert_allclose(df0[cols].values, df1[cols].values, rtol=1e-5)
        pdt.assert_index_equal(df0.index, df1.index)


class TestInfluenceLogitCompare(InfluenceCompareExact):

    @classmethod
    def setup_class(cls):
        df = data_bin
        mod = GLM(
            df["constrict"],
            df[["const", "log_rate", "log_volumne"]],
            family=families.Binomial(),
        )
        res = mod.fit(method="newton", tol=1e-10)
        from statsmodels.discrete.discrete_model import Logit

        mod2 = Logit(df["constrict"], df[["const", "log_rate", "log_volumne"]])
        res2 = mod2.fit(method="newton", tol=1e-10)

        cls.infl1 = res.get_influence()
        cls.infl0 = res2.get_influence()


class TestInfluenceProbitCompare(InfluenceCompareExact):

    @classmethod
    def setup_class(cls):
        df = data_bin
        mod = GLM(
            df["constrict"],
            df[["const", "log_rate", "log_volumne"]],
            family=families.Binomial(link=families.links.Probit()),
        )
        res = mod.fit(method="newton", tol=1e-10)
        from statsmodels.discrete.discrete_model import Probit

        mod2 = Probit(df["constrict"], df[["const", "log_rate", "log_volumne"]])
        res2 = mod2.fit(method="newton", tol=1e-10)

        cls.infl1 = MLEInfluence(res)  # res.get_influence()
        cls.infl0 = res2.get_influence()

    def test_basics_specific(self):
        infl1 = self.infl1
        infl0 = self.infl0
        res1 = self.infl1.results
        res0 = self.infl0.results

        assert_allclose(res1.params, res1.params, rtol=1e-10)

        d1 = res1.model._deriv_mean_dparams(res1.params)
        d0 = res1.model._deriv_mean_dparams(res0.params)
        assert_allclose(d0, d1, rtol=1e-10)

        d1 = res1.model._deriv_score_obs_dendog(res1.params)
        d0 = res1.model._deriv_score_obs_dendog(res0.params)
        assert_allclose(d0, d1, rtol=1e-10)

        s1 = res1.model.score_obs(res1.params)
        s0 = res1.model.score_obs(res0.params)
        assert_allclose(s0, s1, rtol=1e-10)

        assert_allclose(infl0.hessian, infl1.hessian, rtol=1e-10)


class TestInfluencePoissonCompare(InfluenceCompareExact):

    @classmethod
    def setup_class(cls):
        df = data_bin
        mod = GLM(
            df["constrict"],
            df[["const", "log_rate", "log_volumne"]],
            family=families.Poisson(),
        )
        res = mod.fit(attach_wls=True, atol=1e-10)
        from statsmodels.discrete.discrete_model import Poisson

        mod2 = Poisson(df["constrict"], df[["const", "log_rate", "log_volumne"]])
        res2 = mod2.fit(tol=1e-10)

        cls.infl0 = res.get_influence()
        cls.infl1 = res2.get_influence()


def test_vif_instability():
    rs = np.random.RandomState(42)
    n = 500
    x1 = rs.uniform(1000, 100000, n)
    x2 = rs.uniform(5, 50, n)

    exog = np.column_stack([np.ones(n), np.log(x1), 1.0 / x2, x1, x2])

    # 1. Test happy path (standardize=True)
    vif_const = variance_inflation_factor(exog, 0, standardize=True)
    assert_allclose(vif_const, 1.0, atol=1e-2)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        variance_inflation_factor(exog, 1, standardize=True)

    # 2. Test legacy path warning and explosion (standardize=False)
    with pytest.warns(UserWarning, match="The design matrix is poorly conditioned"):
        vif_legacy = variance_inflation_factor(exog, 0, standardize=False)
    assert vif_legacy > 100

    # 3. Test that standardize=False computes perfectly on clean, well-conditioned data
    clean_exog = np.column_stack([np.ones(100), rs.normal(0, 1, (100, 2))])
    vif_true = variance_inflation_factor(clean_exog, 1, standardize=True)
    vif_false = variance_inflation_factor(clean_exog, 1, standardize=False)
    assert_allclose(vif_true, vif_false, rtol=1e-7)


def test_olsinfluence_closed_form_measures():
    rs = np.random.RandomState(20260821)
    n = 60
    exog = np.column_stack([np.ones(n), rs.standard_normal((n, 2))])
    endog = exog @ [1.0, 0.5, -0.5] + rs.standard_normal(n)
    res = OLS(endog, exog).fit()
    infl = OLSInfluence(res)

    hii = infl.hat_matrix_diag
    assert_allclose(infl.resid_press, infl.resid / (1 - hii))
    assert_allclose(infl.hat_diag_factor, hii / (1 - hii))
    assert_allclose(infl.ess_press, np.dot(infl.resid_press, infl.resid_press))
    assert_allclose(infl.resid_var, infl.scale * (1 - hii))
    assert_allclose(infl.resid_std, np.sqrt(infl.resid_var))


def test_olsinfluence_ols_xnoti_and_get_drop_vari():
    rs = np.random.RandomState(20260822)
    n = 50
    exog = np.column_stack([np.ones(n), rs.standard_normal((n, 3))])
    endog = exog @ [1.0, 0.5, -0.5, 0.2] + rs.standard_normal(n)
    res = OLS(endog, exog).fit()
    infl = OLSInfluence(res)

    # _ols_xnoti: exog[:, endog_idx] regressed on the other exog columns
    drop_idx, endog_idx = 1, 2
    aux = infl._ols_xnoti(drop_idx, endog_idx=endog_idx)
    mask = np.arange(exog.shape[1]) != drop_idx
    expected = OLS(exog[:, endog_idx], exog[:, mask]).fit()
    assert_allclose(aux.params, expected.params)

    # cached on repeated calls with store=True, and the cache actually
    # persists on the instance (this used to silently discard it: see the
    # BUG commit removing a stray `stored = {}` and a broken `hasattr`
    # check that made this method crash on every call)
    assert infl.aux_regression_exog[endog_idx][drop_idx] is aux
    aux2 = infl._ols_xnoti(drop_idx, endog_idx=endog_idx)
    assert aux2 is aux

    # endog_idx="endog": original endog regressed on exog minus drop_idx
    aux_endog = infl._ols_xnoti(drop_idx)
    expected_endog = OLS(endog, exog[:, mask]).fit()
    assert_allclose(aux_endog.params, expected_endog.params)
    assert infl.aux_regression_endog[drop_idx] is aux_endog
    assert infl._ols_xnoti(drop_idx) is aux_endog

    # _get_drop_vari: leave-one-*variable*-out loop, collecting the
    # requested attribute from each k_vars-1 auxiliary fit
    res_loo = infl._get_drop_vari(["params"])
    assert len(res_loo["params"]) == infl.k_vars
    for j, params_j in enumerate(res_loo["params"]):
        mask_j = np.arange(infl.k_vars) != j
        expected_j = OLS(endog, exog[:, mask_j]).fit().params
        assert_allclose(params_j, expected_j)


# OLSInfluence should not warn about the rank deficient auxiliary regression
@pytest.mark.filterwarnings(
    "error::statsmodels.tools.sm_exceptions.SingularMatrixWarning"
)
@pytest.mark.parametrize("leverage_one", [False, True])
def test_olsinfluence_looo_closed_form(leverage_one):
    # GH#9009: for OLS, the LOOO results use closed form updates.
    # Compare with explicit leave-one-observation-out regressions.
    rng = np.random.default_rng(9009)
    n = 40
    exog = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
    exog[:2, 2] *= 20  # a few high leverage observations
    if leverage_one:
        # dummy for a single observation, hat diagonal is one, dropping it
        # makes exog rank deficient
        exog = np.column_stack([exog, np.eye(n)[:, 5]])
    endog = exog @ np.linspace(1, -1, exog.shape[1]) + rng.standard_normal(n)
    res = OLS(endog, exog).fit()
    infl = OLSInfluence(res)

    params = np.empty(exog.shape)
    mse_resid = np.empty(n)
    det_cov_params = np.empty(n)
    for i in range(n):
        mask = np.arange(n) != i
        with warnings.catch_warnings():
            # dropping the observation with hat diagonal one gives a rank
            # deficient exog
            warnings.simplefilter("ignore", SingularMatrixWarning)
            res_i = OLS(endog[mask], exog[mask]).fit()
            params[i] = res_i.params
            mse_resid[i] = res_i.mse_resid
            det_cov_params[i] = np.linalg.det(res_i.cov_params())

    assert_allclose(infl.params_not_obsi, params, rtol=1e-10, atol=1e-12)
    assert_allclose(infl.sigma2_not_obsi, mse_resid, rtol=1e-10)
    assert_allclose(infl.det_cov_params_not_obsi, det_cov_params, rtol=1e-10)
    assert_allclose(infl.dfbeta, res.params - params, rtol=1e-10, atol=1e-12)
    dfbetas = (res.params - params) / np.sqrt(
        mse_resid[:, None] * np.diag(res.normalized_cov_params)
    )
    assert_allclose(infl.dfbetas, dfbetas, rtol=1e-10, atol=1e-12)
    resid_ext = res.resid / np.sqrt(mse_resid * (1 - infl.hat_matrix_diag))
    if leverage_one:
        # resid and 1 - hii are both zero up to rounding for observation 5
        resid_ext[5] = infl.resid_studentized_external[5]
    assert_allclose(infl.resid_studentized_external, resid_ext, rtol=1e-10)
    cov_ratio = det_cov_params / np.linalg.det(res.cov_params())
    assert_allclose(infl.cov_ratio, cov_ratio, rtol=1e-10)


def test_olsinfluence_looo_loop_matches_closed_form():
    # A model class other than OLS takes the leave-one-observation-out
    # regression loop; it must agree with the closed form used for OLS.
    class LoopOLS(OLS):
        pass

    rng = np.random.default_rng(10367)
    n = 30
    exog = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
    endog = exog @ [1.0, 0.5, -0.5] + rng.standard_normal(n)
    infl_closed = OLSInfluence(OLS(endog, exog).fit())
    infl_loop = OLSInfluence(LoopOLS(endog, exog).fit())
    assert infl_loop.model_class is LoopOLS

    assert_allclose(
        infl_loop.params_not_obsi, infl_closed.params_not_obsi, rtol=1e-10
    )
    assert_allclose(
        infl_loop.sigma2_not_obsi, infl_closed.sigma2_not_obsi, rtol=1e-10
    )
    assert_allclose(
        infl_loop.det_cov_params_not_obsi,
        infl_closed.det_cov_params_not_obsi,
        rtol=1e-10,
    )
    assert_allclose(infl_loop.dfbetas, infl_closed.dfbetas, rtol=1e-10)
    assert_allclose(infl_loop.cov_ratio, infl_closed.cov_ratio, rtol=1e-10)


@pytest.mark.parametrize("model_type", ["WLS", "WLS subclass"])
def test_olsinfluence_wls_matches_weighted_looo(model_type):
    # GH#9067: OLSInfluence on WLS results refit the leave-one-observation-out
    # regressions without weights and used an unweighted hat matrix, so
    # dfbeta, cooks_distance and the studentized residuals were wrong.
    # Compare with explicit weighted leave-one-observation-out regressions.
    # The WLS subclass takes the regression loop instead of the closed form.
    class LoopWLS(WLS):
        pass

    model_class = WLS if model_type == "WLS" else LoopWLS
    rng = np.random.default_rng(9067)
    n, k_vars = 50, 3
    exog = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
    exog[:2, 2] *= 10  # a few high leverage observations
    weights = rng.uniform(0.2, 5, n)
    endog = exog @ [1.0, 2.0, -1.0] + rng.standard_normal(n) / np.sqrt(weights)
    res = model_class(endog, exog, weights=weights).fit()
    infl = OLSInfluence(res)

    params = np.empty(exog.shape)
    mse_resid = np.empty(n)
    det_cov_params = np.empty(n)
    for i in range(n):
        mask = np.arange(n) != i
        res_i = WLS(endog[mask], exog[mask], weights=weights[mask]).fit()
        params[i] = res_i.params
        mse_resid[i] = res_i.mse_resid
        det_cov_params[i] = np.linalg.det(res_i.cov_params())

    sqrt_w = np.sqrt(weights)
    wexog = exog * sqrt_w[:, None]
    xtwx_inv = np.linalg.inv(wexog.T @ wexog)
    hat = np.einsum("ij,jk,ik->i", wexog, xtwx_inv, wexog)
    assert_allclose(infl.hat_matrix_diag, hat, rtol=1e-10)

    assert_allclose(infl.params_not_obsi, params, rtol=1e-10, atol=1e-12)
    assert_allclose(infl.sigma2_not_obsi, mse_resid, rtol=1e-10)
    assert_allclose(infl.det_cov_params_not_obsi, det_cov_params, rtol=1e-10)
    dfbeta = res.params - params
    assert_allclose(infl.dfbeta, dfbeta, rtol=1e-10, atol=1e-12)
    cov_ratio = det_cov_params / np.linalg.det(res.cov_params())
    assert_allclose(infl.cov_ratio, cov_ratio, rtol=1e-10)

    # Cook's distance from its definition with the weighted cross product
    cooks = np.einsum("ij,jk,ik->i", dfbeta, np.linalg.inv(xtwx_inv), dfbeta)
    cooks /= k_vars * res.mse_resid
    assert_allclose(infl.cooks_distance[0], cooks, rtol=1e-10)

    wresid = res.resid * sqrt_w
    resid_int = wresid / np.sqrt(res.mse_resid * (1 - hat))
    assert_allclose(infl.resid_studentized_internal, resid_int, rtol=1e-10)
    resid_ext = wresid / np.sqrt(mse_resid * (1 - hat))
    assert_allclose(infl.resid_studentized_external, resid_ext, rtol=1e-10)

    # PRESS residuals stay on the scale of endog: the prediction error of
    # the leave-one-observation-out fit.
    resid_press = endog - np.einsum("ij,ij->i", exog, params)
    assert_allclose(infl.resid_press, resid_press, rtol=1e-10)


def test_olsinfluence_gls_without_sigma_matches_ols():
    # Models other than OLS and WLS keep endog and exog unwhitened and take the
    # regression loop. GLS without sigma is OLS, so all measures must agree.
    rng = np.random.default_rng(90670)
    n = 30
    exog = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
    endog = exog @ [1.0, 0.5, -0.5] + rng.standard_normal(n)
    infl_gls = OLSInfluence(GLS(endog, exog).fit())
    infl_ols = OLSInfluence(OLS(endog, exog).fit())

    assert_allclose(infl_gls.hat_matrix_diag, infl_ols.hat_matrix_diag, rtol=1e-10)
    assert_allclose(infl_gls.dfbeta, infl_ols.dfbeta, rtol=1e-10, atol=1e-12)
    assert_allclose(
        infl_gls.resid_studentized_external,
        infl_ols.resid_studentized_external,
        rtol=1e-10,
    )
    assert_allclose(infl_gls.cooks_distance[0], infl_ols.cooks_distance[0], rtol=1e-10)


def test_olsinfluence_wls_r_influence_measures():
    # GH#9067: Reference values from R 4.6.1 (stats, base R).
    #   y <- c(2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.5)
    #   x1 <- 1:8
    #   x2 <- c(1, 0, 1, 0, 1, 0, 0, 1)
    #   w <- c(1, 2, 0.5, 1, 3, 1, 0.25, 2)
    #   fit <- lm(y ~ x1 + x2, weights = w)
    #   hatvalues(fit); rstandard(fit); rstudent(fit); cooks.distance(fit)
    #   dffits(fit); covratio(fit); dfbeta(fit); dfbetas(fit)
    # cooks.distance, dffits and covratio are identical to the columns of
    # influence.measures(fit)$infmat. The values are printed with 15 digits.
    y = np.array([2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.5])
    x1 = np.arange(1.0, 9.0)
    x2 = np.array([1.0, 0, 1, 0, 1, 0, 0, 1])
    weights = np.array([1.0, 2, 0.5, 1, 3, 1, 0.25, 2])
    exog = np.column_stack([np.ones(8), x1, x2])

    # fmt: off
    hat = [
        0.500818926296633, 0.587625113739763, 0.123566878980892,
        0.237033666969973, 0.462966333030027, 0.341128298453139,
        0.113375796178344, 0.633484986351228,
    ]
    rstandard = [
        0.786053746421487, 0.657706054624384, 0.375104442479753,
        -1.01528011893883, -1.62812210912733, 0.9172300591081,
        -0.966429023119503, 1.47503314381012,
    ]
    rstudent = [
        0.751000331903412, 0.615497410810194, 0.34032628788785,
        -1.01921086041045, -2.12449284531458, 0.89955982581948,
        -0.958551551257641, 1.75540815948704,
    ]
    cooks = [
        0.206635935679727, 0.205471014136662, 0.00661251174830449,
        0.106746874989939, 0.761729176106538, 0.145195178797201,
        0.039810665732168, 1.25350727920438,
    ]
    dffits = [
        0.752231368698551, 0.734734275237678, 0.127787138346727,
        -0.568088990886466, -1.97255545541674, 0.647274500367194,
        -0.34277240991915, 2.30781180875216,
    ]
    covratio = [
        2.63399270087544, 3.61028467846914, 2.04560362400882,
        1.28063586685915, 0.377214773394677, 1.70564445917374,
        1.18464262309972, 0.960395746774834,
    ]
    # columns are const, x1, x2
    dfbeta = [
        [0.0611321464480507, -0.0164959760256645, 0.0542678078404213],
        [0.0931801035523767, -0.00881970852102587, -0.0477246827132434],
        [0.00807213746111685, -0.00217819582284106, 0.0108909791142053],
        [-0.0440238673093067, -0.00122027777189976, 0.0503129912106363],
        [-0.00783122206226107, 0.0021131869056895, -0.108145447526462],
        [0.0129035560946731, 0.00925332806223374, -0.0605937853384931],
        [0.000928325646094153, -0.00603411669961197, 0.0301705834980598],
        [-0.129725488459762, 0.0350052905367612, 0.0434087863571919],
    ]
    dfbetas = [
        [0.460086768752802, -0.626121722962813, 0.444764380975708],
        [0.686909924840982, -0.327900056075996, -0.383122161429812],
        [0.0576917947881697, -0.0785114643733512, 0.0847637277882975],
        [-0.348135702275125, -0.0486664568917755, 0.433269514459581],
        [-0.0804970657845683, 0.10954664412348, -1.21053228073044],
        [0.0996881184615388, 0.360530989434074, -0.509776358670679],
        [0.00725317358218219, -0.237767276679144, 0.256701882690098],
        [-1.21614080482432, 1.65501863517206, 0.443153120178539],
    ]
    # fmt: on

    infl = OLSInfluence(WLS(y, exog, weights=weights).fit())
    rtol = 1e-8
    assert_allclose(infl.hat_matrix_diag, hat, rtol=rtol)
    assert_allclose(infl.resid_studentized_internal, rstandard, rtol=rtol)
    assert_allclose(infl.resid_studentized_external, rstudent, rtol=rtol)
    assert_allclose(infl.cooks_distance[0], cooks, rtol=rtol)
    assert_allclose(infl.dffits[0], dffits, rtol=rtol)
    assert_allclose(infl.cov_ratio, covratio, rtol=rtol)
    assert_allclose(infl.dfbeta, dfbeta, rtol=rtol)
    assert_allclose(infl.dfbetas, dfbetas, rtol=rtol)


def _wls_leave_one_out(endog, exog, weights):
    # explicit leave-one-observation-out WLS regressions
    n = len(endog)
    params = np.empty(exog.shape)
    mse_resid = np.empty(n)
    det_cov_params = np.empty(n)
    for i in range(n):
        mask = np.arange(n) != i
        with warnings.catch_warnings():
            # dropping an observation can make exog rank deficient
            warnings.simplefilter("ignore", SingularMatrixWarning)
            res_i = WLS(endog[mask], exog[mask], weights=weights[mask]).fit()
            params[i] = res_i.params
            mse_resid[i] = res_i.mse_resid
            det_cov_params[i] = np.linalg.det(res_i.cov_params())
    return params, mse_resid, det_cov_params


@pytest.mark.parametrize(
    "case",
    [
        # exog is rank deficient, so the model fit and each auxiliary
        # regression warn, as for OLS
        pytest.param(
            "rank_deficient",
            marks=pytest.mark.filterwarnings(
                "ignore::statsmodels.tools.sm_exceptions.SingularMatrixWarning"
            ),
        ),
        # OLSInfluence should not warn about the rank deficient auxiliary
        # regression
        pytest.param(
            "leverage_one",
            marks=pytest.mark.filterwarnings(
                "error::statsmodels.tools.sm_exceptions.SingularMatrixWarning"
            ),
        ),
    ],
)
def test_olsinfluence_wls_looo_rank_deficient(case):
    # GH#9067: Weighted regressions without the closed form. If exog is rank
    # deficient, then all observations take the regression loop. If a
    # hat diagonal is one, then only that observation takes the loop.
    rng = np.random.default_rng(90671)
    n = 40
    exog = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
    exog[:2, 2] *= 10  # a few high leverage observations
    weights = rng.uniform(0.2, 5, n)
    if case == "rank_deficient":
        exog = np.column_stack([exog, exog[:, 1] + exog[:, 2]])
    else:
        # dummy for a single observation, hat diagonal is one, dropping it
        # makes exog rank deficient
        exog = np.column_stack([exog, np.eye(n)[:, 5]])
    k_vars = exog.shape[1]
    endog = exog @ np.linspace(1, -1, k_vars) + rng.standard_normal(n) / np.sqrt(
        weights
    )
    res = WLS(endog, exog, weights=weights).fit()
    infl = OLSInfluence(res)

    sqrt_w = np.sqrt(weights)
    wexog = exog * sqrt_w[:, None]
    hat = np.einsum("ij,ji->i", wexog, np.linalg.pinv(wexog))
    assert_allclose(infl.hat_matrix_diag, hat, rtol=1e-10, atol=1e-12)
    if case == "rank_deficient":
        assert res.model.rank == k_vars - 1
    else:
        assert_allclose(infl.hat_matrix_diag[5], 1, atol=1e-10)

    params, mse_resid, det_cov_params = _wls_leave_one_out(endog, exog, weights)
    assert_allclose(infl.params_not_obsi, params, rtol=1e-10, atol=1e-12)
    assert_allclose(infl.sigma2_not_obsi, mse_resid, rtol=1e-10)
    assert_allclose(infl.dfbeta, res.params - params, rtol=1e-10, atol=1e-12)

    # observation 5 has a hat diagonal of one in the leverage_one case and its
    # residual and 1 - hii are both zero up to rounding
    keep = np.ones(n, dtype=bool)
    if case == "leverage_one":
        keep[5] = False
    wresid = (res.resid * sqrt_w)[keep]
    resid_int = wresid / np.sqrt(res.mse_resid * (1 - hat[keep]))
    assert_allclose(infl.resid_studentized_internal[keep], resid_int, rtol=1e-10)
    resid_ext = wresid / np.sqrt(mse_resid[keep] * (1 - hat[keep]))
    assert_allclose(infl.resid_studentized_external[keep], resid_ext, rtol=1e-10)
    # PRESS residuals stay on the scale of endog
    resid_press = endog - np.einsum("ij,ij->i", exog, params)
    assert_allclose(infl.resid_press[keep], resid_press[keep], rtol=1e-8)

    if case == "leverage_one":
        # full rank exog, the covariance ratio and Cook's distance are defined
        assert res.model.rank == k_vars
        assert_allclose(
            infl.det_cov_params_not_obsi,
            det_cov_params,
            rtol=1e-10,
            atol=1e-12 * np.abs(det_cov_params).max(),
        )
        cov_ratio = det_cov_params / np.linalg.det(res.cov_params())
        assert_allclose(
            infl.cov_ratio, cov_ratio, rtol=1e-10, atol=1e-12 * cov_ratio.max()
        )
        xtwx = wexog.T @ wexog
        dfbeta = res.params - params
        cooks = np.einsum("ij,jk,ik->i", dfbeta, xtwx, dfbeta)
        cooks /= k_vars * res.mse_resid
        assert_allclose(infl.cooks_distance[0][keep], cooks[keep], rtol=1e-8)


def test_olsinfluence_wls_zero_weights():
    # GH#9067: WLS.df_resid counts observations with weight zero, R drops them
    rng = np.random.default_rng(90672)
    n, k_vars = 30, 3
    exog = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
    weights = rng.uniform(0.3, 3, n)
    zero = [3, 16]
    weights[zero] = 0
    endog = exog @ [1.0, 0.5, -0.5] + rng.standard_normal(n)
    pos = weights > 0

    res = WLS(endog, exog, weights=weights).fit()
    infl = OLSInfluence(res)
    res_pos = WLS(endog[pos], exog[pos], weights=weights[pos]).fit()
    infl_pos = OLSInfluence(res_pos)

    # df_resid counts the observations with weight zero, not as in R
    assert res.df_resid == n - k_vars
    assert res_pos.df_resid == n - len(zero) - k_vars
    factor = res.df_resid / res_pos.df_resid
    assert factor > 1

    # observations with weight zero have no influence on the fit
    for name in ["hat_matrix_diag", "resid_studentized_internal", "dfbeta"]:
        assert_allclose(getattr(infl, name)[zero], 0, atol=1e-12)
    assert_allclose(infl.resid_studentized_external[zero], 0, atol=1e-12)
    assert_allclose(infl.cooks_distance[0][zero], 0, atol=1e-12)
    assert_allclose(infl.dffits[0][zero], 0, atol=1e-12)
    assert_allclose(infl.dfbetas[zero], 0, atol=1e-12)

    # hat and dfbeta are the same as WLS without the observations
    assert_allclose(infl.hat_matrix_diag[pos], infl_pos.hat_matrix_diag, rtol=1e-10)
    assert_allclose(infl.dfbeta[pos], infl_pos.dfbeta, rtol=1e-10)
    # measures that depend on the residual degrees of freedom differ as in the
    # class docstring, R corresponds to the fit without the observations
    assert_allclose(
        infl.cooks_distance[0][pos], infl_pos.cooks_distance[0] * factor, rtol=1e-10
    )
    assert_allclose(
        infl.resid_studentized_internal[pos],
        infl_pos.resid_studentized_internal * np.sqrt(factor),
        rtol=1e-10,
    )
