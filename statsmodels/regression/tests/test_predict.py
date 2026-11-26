"""
Created on Sun Apr 20 17:12:53 2014

author: Josef Perktold

"""

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest
from scipy import special, stats as sp_stats

from statsmodels.discrete.discrete_model import Logit
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.regression._prediction import get_prediction
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.sandbox.regression.predstd import wls_prediction_std


def test_predict_se():
    # this test does not use reference values
    # checks conistency across options, and compares to direct calculation

    # generate dataset
    nsample = 50
    x1 = np.linspace(0, 20, nsample)
    x = np.c_[x1, (x1 - 5) ** 2, np.ones(nsample)]
    rs = np.random.RandomState(0)  # 9876789) #9876543)
    beta = [0.5, -0.01, 5.0]
    y_true2 = np.dot(x, beta)
    w = np.ones(nsample)
    w[int(nsample * 6.0 / 10) :] = 3
    sig = 0.5
    y2 = y_true2 + sig * w * rs.normal(size=nsample)
    x2 = x[:, [0, 2]]

    # estimate OLS
    res2 = OLS(y2, x2).fit()

    # direct calculation
    covb = res2.cov_params()
    predvar = res2.mse_resid + (x2 * np.dot(covb, x2.T).T).sum(1)
    predstd = np.sqrt(predvar)

    prstd, iv_l, iv_u = wls_prediction_std(res2)
    np.testing.assert_almost_equal(prstd, predstd, 15)

    # stats.t.isf(0.05/2., 50 - 2)
    q = 2.0106347546964458
    ci_half = q * predstd
    np.testing.assert_allclose(iv_u, res2.fittedvalues + ci_half, rtol=1e-9)
    np.testing.assert_allclose(iv_l, res2.fittedvalues - ci_half, rtol=1e-9)

    prstd, iv_l, iv_u = wls_prediction_std(res2, x2[:3, :])
    np.testing.assert_equal(prstd, prstd[:3])
    np.testing.assert_allclose(iv_u, res2.fittedvalues[:3] + ci_half[:3], rtol=1e-9)
    np.testing.assert_allclose(iv_l, res2.fittedvalues[:3] - ci_half[:3], rtol=1e-9)

    # check WLS
    res3 = WLS(y2, x2, 1.0 / w).fit()

    # direct calculation
    covb = res3.cov_params()
    predvar = res3.mse_resid * w + (x2 * np.dot(covb, x2.T).T).sum(1)
    predstd = np.sqrt(predvar)

    prstd, iv_l, iv_u = wls_prediction_std(res3)
    np.testing.assert_almost_equal(prstd, predstd, 15)

    # stats.t.isf(0.05/2., 50 - 2)
    q = 2.0106347546964458
    ci_half = q * predstd
    np.testing.assert_allclose(iv_u, res3.fittedvalues + ci_half, rtol=1e-9)
    np.testing.assert_allclose(iv_l, res3.fittedvalues - ci_half, rtol=1e-9)

    # testing shapes of exog
    prstd, iv_l, iv_u = wls_prediction_std(res3, x2[-1:, :], weights=3.0)
    np.testing.assert_equal(prstd, prstd[-1])
    prstd, iv_l, iv_u = wls_prediction_std(res3, x2[-1, :], weights=3.0)
    np.testing.assert_equal(prstd, prstd[-1])

    prstd, iv_l, iv_u = wls_prediction_std(res3, x2[-2:, :], weights=3.0)
    np.testing.assert_equal(prstd, prstd[-2:])

    prstd, iv_l, iv_u = wls_prediction_std(res3, x2[-2:, :], weights=[3, 3])
    np.testing.assert_equal(prstd, prstd[-2:])

    prstd, iv_l, iv_u = wls_prediction_std(res3, x2[:3, :])
    np.testing.assert_equal(prstd, prstd[:3])
    np.testing.assert_allclose(iv_u, res3.fittedvalues[:3] + ci_half[:3], rtol=1e-9)
    np.testing.assert_allclose(iv_l, res3.fittedvalues[:3] - ci_half[:3], rtol=1e-9)

    # use wrong size for exog
    # prstd, iv_l, iv_u = wls_prediction_std(res3, x2[-1,0], weights=3.)
    with pytest.raises(ValueError):
        wls_prediction_std(res3, x2[-1, 0], weights=3.0)

    # check some weight values
    sew1 = wls_prediction_std(res3, x2[-3:, :])[0] ** 2
    for wv in np.linspace(0.5, 3, 5):

        sew = wls_prediction_std(res3, x2[-3:, :], weights=1.0 / wv)[0] ** 2
        np.testing.assert_allclose(sew, sew1 + res3.scale * (wv - 1))


class TestWLSPrediction:

    @classmethod
    def setup_class(cls):

        # from example wls.py
        rs = np.random.RandomState(3237219)
        nsample = 50
        x = np.linspace(0, 20, nsample)
        X = np.column_stack((x, (x - 5) ** 2))
        from statsmodels.tools.tools import add_constant

        X = add_constant(X)
        beta = [5.0, 0.5, -0.01]
        sig = 0.5
        w = np.ones(nsample)
        w[int(nsample * 6.0 / 10) :] = 3
        y_true = np.dot(X, beta)
        e = rs.normal(size=nsample)
        y = y_true + sig * w * e
        X = X[:, [0, 1]]

        # # WLS knowing the true variance ratio of heteroscedasticity

        mod_wls = WLS(y, X, weights=1.0 / w)
        cls.res_wls = mod_wls.fit()

    def test_ci(self):
        res_wls = self.res_wls
        prstd, iv_l, iv_u = wls_prediction_std(res_wls)
        pred_res = get_prediction(res_wls)
        ci = pred_res.conf_int(obs=True)

        assert_allclose(pred_res.se_obs, prstd, rtol=1e-13)
        assert_allclose(ci, np.column_stack((iv_l, iv_u)), rtol=1e-13)

        sf = pred_res.summary_frame()

        col_names = [
            "mean",
            "mean_se",
            "mean_ci_lower",
            "mean_ci_upper",
            "obs_ci_lower",
            "obs_ci_upper",
        ]
        assert_equal(sf.columns.tolist(), col_names)

        pred_res2 = res_wls.get_prediction()
        ci2 = pred_res2.conf_int(obs=True)

        assert_allclose(pred_res2.se_obs, prstd, rtol=1e-13)
        assert_allclose(ci2, np.column_stack((iv_l, iv_u)), rtol=1e-13)

        sf2 = pred_res2.summary_frame()
        assert_equal(sf2.columns.tolist(), col_names)

        # check that list works, issue 4437
        x = res_wls.model.exog.mean(0)
        pred_res3 = res_wls.get_prediction([x])
        ci3 = pred_res3.conf_int(obs=True)
        pred_res3b = res_wls.get_prediction(x.tolist())
        ci3b = pred_res3b.conf_int(obs=True)
        assert_allclose(pred_res3b.se_obs, pred_res3.se_obs, rtol=1e-13)
        assert_allclose(ci3b, ci3, rtol=1e-13)
        res_df = pred_res3b.summary_frame()
        assert_equal(res_df.index.values, [0])

        x = res_wls.model.exog[-2:]
        pred_res3 = res_wls.get_prediction(x)
        ci3 = pred_res3.conf_int(obs=True)
        pred_res3b = res_wls.get_prediction(x.tolist())
        ci3b = pred_res3b.conf_int(obs=True)
        assert_allclose(pred_res3b.se_obs, pred_res3.se_obs, rtol=1e-13)
        assert_allclose(ci3b, ci3, rtol=1e-13)
        res_df = pred_res3b.summary_frame()
        assert_equal(res_df.index.values, [0, 1])

    def test_weights_shape(self):
        # GH-10283: number of rows equal to number of params raised
        res_wls = self.res_wls
        k_params = len(res_wls.params)
        exog = res_wls.model.exog[:k_params]
        weights = res_wls.model.weights[:k_params]
        pred = res_wls.get_prediction(exog, weights=weights)
        pred_full = res_wls.get_prediction()
        assert_allclose(pred.se_obs, pred_full.se_obs[:k_params], rtol=1e-13)

        with pytest.raises(ValueError, match="weights has wrong shape"):
            res_wls.get_prediction(exog[:1], weights=weights)

    def test_glm(self):
        # prelimnimary, getting started with basic test for GLM.get_prediction
        from statsmodels.genmod.generalized_linear_model import GLM

        res_wls = self.res_wls
        mod_wls = res_wls.model
        y, X, wi = mod_wls.endog, mod_wls.exog, mod_wls.weights

        w_sqrt = np.sqrt(wi)  # notation wi is weights, `w` is var
        mod_glm = GLM(y * w_sqrt, X * w_sqrt[:, None])

        # compare using t distribution
        res_glm = mod_glm.fit(use_t=True)
        pred_glm = res_glm.get_prediction()
        sf_glm = pred_glm.summary_frame()

        pred_res_wls = res_wls.get_prediction()
        sf_wls = pred_res_wls.summary_frame()
        n_compare = 30  # in glm with predict wendog
        assert_allclose(sf_glm.values[:n_compare], sf_wls.values[:n_compare, :4])

        # compare using normal distribution

        res_glm = mod_glm.fit()  # default use_t=False
        pred_glm = res_glm.get_prediction()
        sf_glm = pred_glm.summary_frame()

        res_wls = mod_wls.fit(use_t=False)
        pred_res_wls = res_wls.get_prediction()
        sf_wls = pred_res_wls.summary_frame()
        assert_allclose(sf_glm.values[:n_compare], sf_wls.values[:n_compare, :4])

        # function for parameter transformation
        # should be separate test method
        from statsmodels.base._prediction_inference import params_transform_univariate

        rates = params_transform_univariate(res_glm.params, res_glm.cov_params())

        rates2 = np.column_stack(
            (
                np.exp(res_glm.params),
                res_glm.bse * np.exp(res_glm.params),
                np.exp(res_glm.conf_int()),
            )
        )
        assert_allclose(rates.summary_frame().values, rates2, rtol=1e-13)

        from statsmodels.genmod.families import links

        # with identity transform
        pt = params_transform_univariate(
            res_glm.params, res_glm.cov_params(), link=links.Identity()
        )

        assert_allclose(pt.tvalues, res_glm.tvalues, rtol=1e-13)
        assert_allclose(pt.se_mean, res_glm.bse, rtol=1e-13)
        ptt = pt.t_test()
        assert_allclose(ptt[0], res_glm.tvalues, rtol=1e-13)
        assert_allclose(ptt[1], res_glm.pvalues, rtol=1e-13)

        # prediction with exog and no weights does not error
        res_glm = mod_glm.fit()
        pred_glm = res_glm.get_prediction(X)

        # check that list works, issue 4437
        x = res_glm.model.exog.mean(0)
        pred_res3 = res_glm.get_prediction(x)
        ci3 = pred_res3.conf_int()
        pred_res3b = res_glm.get_prediction(x.tolist())
        ci3b = pred_res3b.conf_int()
        assert_allclose(pred_res3b.se_mean, pred_res3.se_mean, rtol=1e-13)
        assert_allclose(ci3b, ci3, rtol=1e-13)
        res_df = pred_res3b.summary_frame()
        assert_equal(res_df.index.values, [0])

        x = res_glm.model.exog[-2:]
        pred_res3 = res_glm.get_prediction(x)
        ci3 = pred_res3.conf_int()
        pred_res3b = res_glm.get_prediction(x.tolist())
        ci3b = pred_res3b.conf_int()
        assert_allclose(pred_res3b.se_mean, pred_res3.se_mean, rtol=1e-13)
        assert_allclose(ci3b, ci3, rtol=1e-13)
        res_df = pred_res3b.summary_frame()
        assert_equal(res_df.index.values, [0, 1])


def test_predict_remove_data():
    # GH6887
    rs = np.random.RandomState(3821010)
    endog = [i + rs.normal(scale=0.1) for i in range(100)]
    exog = list(range(100))
    model = WLS(endog, exog, weights=[1 for _ in range(100)]).fit()
    # we need to compute scale before we remove wendog, wexog
    assert isinstance(model.scale, float)
    model.remove_data()
    scalar = model.get_prediction(1).predicted_mean
    pred = model.get_prediction([1])
    one_d = pred.predicted_mean
    assert_allclose(scalar, one_d)
    # smoke test for inferenctial part
    pred.summary_frame()

    series = model.get_prediction(pd.Series([1])).predicted_mean
    assert_allclose(scalar, series)


def test_prediction_results_t_test():
    from scipy import stats as sp_stats

    from statsmodels.genmod.generalized_linear_model import GLM

    rs = np.random.RandomState(918273)
    n = 40
    exog = np.column_stack([np.ones(n), rs.standard_normal((n, 2))])
    endog = exog @ [1.0, 0.5, -0.5] + rs.standard_normal(n)
    # GLM.get_prediction returns PredictionResultsMean, the class that
    # actually implements t_test; OLS's get_prediction is a different,
    # unrelated PredictionResults class of the same name.
    res = GLM(endog, exog).fit()
    pred = res.get_prediction()

    stat, pvalue = pred.t_test(value=0, alternative="two-sided")
    expected_stat = pred.predicted / pred.se
    assert_allclose(stat, expected_stat)
    assert_allclose(pvalue, pred.dist.sf(np.abs(expected_stat), *pred.dist_args) * 2)

    stat_l, pvalue_l = pred.t_test(value=1.0, alternative="larger")
    expected_stat_l = (pred.predicted - 1.0) / pred.se
    assert_allclose(stat_l, expected_stat_l)
    assert_allclose(pvalue_l, pred.dist.sf(expected_stat_l, *pred.dist_args))

    stat_s, pvalue_s = pred.t_test(value=1.0, alternative="smaller")
    assert_allclose(stat_s, expected_stat_l)
    assert_allclose(pvalue_s, pred.dist.cdf(expected_stat_l, *pred.dist_args))

    # smaller-side p-value is 1 minus the larger-side p-value for the same
    # statistic, by definition
    assert_allclose(pvalue_s, 1 - pvalue_l, atol=1e-10)

    with pytest.raises(ValueError, match="alternative must be one of"):
        pred.t_test(alternative="not-a-real-option")

    # undocumented short forms still work but warn, and are equivalent to
    # spelling out the documented alternative
    for alias, canonical in [("2s", "two-sided"), ("l", "larger"), ("s", "smaller")]:
        with pytest.warns(FutureWarning, match="is a deprecated alias"):
            alias_result = pred.t_test(value=1.0, alternative=alias)
        canonical_result = pred.t_test(value=1.0, alternative=canonical)
        assert_allclose(alias_result, canonical_result)

    # GLM defaults to use_t=False -> normal reference distribution
    assert pred.dist is sp_stats.norm
    assert pred.dist_args == ()

    res_t = GLM(endog, exog).fit(use_t=True)
    pred_t = res_t.get_prediction()
    assert pred_t.dist is sp_stats.t
    assert pred_t.dist_args == (pred_t.df,)


def test_prediction_results_mean_conf_int_invalid_method():
    from statsmodels.genmod.generalized_linear_model import GLM

    rs = np.random.RandomState(918273)
    n = 40
    exog = np.column_stack([np.ones(n), rs.standard_normal((n, 2))])
    endog = exog @ [1.0, 0.5, -0.5] + rs.standard_normal(n)
    res = GLM(endog, exog).fit()
    pred = res.get_prediction()

    assert pred.conf_int(method="endpoint") is not None
    assert pred.conf_int(method="delta") is not None
    with pytest.raises(ValueError, match="method"):
        pred.conf_int(method="not-a-method")


def test_prediction_results_mean_var_pred_mean():
    # PredictionResultsMean.var_pred_mean had no test coverage -- existing
    # tests only ever access the related se_mean (== sqrt(self.var_pred)),
    # which reads self.var_pred directly rather than going through the
    # var_pred_mean property, so var_pred_mean's own body never ran.
    #
    # Use a Gaussian-family, identity-link GLM (numerically equivalent to
    # OLS) so the get_prediction_glm formula
    # var_pred_mean = link_deriv**2 * (exog*cov_params@exog.T).sum(1)
    # reduces to the textbook mean-prediction variance x0 @ cov_params @ x0
    # for each row x0, which can be computed by hand independently.
    from statsmodels.genmod.generalized_linear_model import GLM

    rng = np.random.default_rng(918273)
    n = 40
    exog = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
    endog = exog @ [1.0, 0.5, -0.5] + rng.standard_normal(n)
    res = GLM(endog, exog).fit()

    exog0 = np.column_stack([np.ones(5), rng.standard_normal((5, 2))])
    pred = res.get_prediction(exog0)

    covb = res.cov_params()
    expected = np.array([x0 @ covb @ x0 for x0 in exog0])
    assert_allclose(pred.var_pred_mean, expected, rtol=1e-10)

    # it is documented as a backwards-compatibility alias for var_pred
    assert pred.var_pred_mean is pred.var_pred
    assert_allclose(pred.var_pred_mean, pred.se_mean**2, rtol=1e-10)


def _make_cluster_model(model, n_groups=8, nobs_group=15, seed=91827):
    # model with a cluster structure for the tests of the degrees of freedom
    # of the prediction intervals: returns the model, the groups and the extra
    # arguments for fit
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), nobs_group)
    n = groups.shape[0]
    exog = np.column_stack([np.ones(n), rng.standard_normal(n)])
    linpred = exog @ [0.5, 0.5] + 0.5 * rng.standard_normal(n_groups)[groups]
    if model == "glm-poisson":
        endog = rng.poisson(np.exp(linpred))
        mod = GLM(endog, exog, family=families.Poisson())
    elif model == "logit":
        endog = (rng.uniform(size=n) < 1 / (1 + np.exp(-linpred))).astype(float)
        mod = Logit(endog, exog)
    else:
        endog = linpred + rng.standard_normal(n)
        if model == "ols":
            mod = OLS(endog, exog)
        elif model == "wls":
            mod = WLS(endog, exog, weights=rng.uniform(0.5, 2, size=n))
        else:
            mod = GLM(endog, exog)

    kwds = {"disp": 0} if model == "logit" else {}
    return mod, groups, kwds


def _linear_ci_from_cov(res, exog0, dist):
    # interval of the linear predictor from the covariance of the parameters,
    # the same formula as for ``t_test``, independent of ``get_prediction``
    lin0 = exog0 @ res.params
    se0 = np.sqrt(np.einsum("ij,jk,ik->i", exog0, res.cov_params(), exog0))
    q = dist.ppf(0.975)
    return lin0, se0, np.column_stack([lin0 - q * se0, lin0 + q * se0])


def _linear_ci_prediction(res, model, exog0):
    # the results of the linear models do not have the ``which`` option
    if model in ["ols", "wls"]:
        return res.get_prediction(exog0).conf_int()
    return res.get_prediction(exog0, which="linear").conf_int()


@pytest.mark.parametrize("model", ["ols", "wls", "glm", "glm-poisson", "logit"])
def test_prediction_cluster_use_t_df(model):
    # with cov_type="cluster" and use_t=True, prediction intervals use the
    # same t(n_groups - 1) distribution as conf_int and t_test
    n_groups = 8
    mod, groups, kwds = _make_cluster_model(model, n_groups=n_groups)
    res = mod.fit(cov_type="cluster", cov_kwds={"groups": groups}, use_t=True, **kwds)
    assert res.df_resid_inference == n_groups - 1

    exog0 = np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 2.0]])
    lin0, se0, ci_lin = _linear_ci_from_cov(res, exog0, sp_stats.t(n_groups - 1))
    # t_test of the same linear combinations uses df_resid_inference
    assert_allclose(res.t_test(exog0).conf_int(), ci_lin, rtol=1e-10)

    sf = res.get_prediction(exog0).summary_frame()
    if model in ["ols", "wls", "glm"]:
        assert_allclose(sf[["mean_ci_lower", "mean_ci_upper"]], ci_lin, rtol=1e-10)
    if model in ["ols", "wls"]:
        q = sp_stats.t.ppf(0.975, n_groups - 1)
        se_obs = np.sqrt(se0**2 + res.scale)
        ci_obs = np.column_stack([lin0 - q * se_obs, lin0 + q * se_obs])
        assert_allclose(sf[["obs_ci_lower", "obs_ci_upper"]], ci_obs, rtol=1e-10)
    if model in ["glm", "glm-poisson", "logit"]:
        pred = res.get_prediction(exog0, which="linear")
        assert_allclose(pred.conf_int(), ci_lin, rtol=1e-10)
        pred = res.get_prediction(exog0, which="mean")
        if model == "logit":
            ci_mean = special.expit(ci_lin)
        else:
            ci_mean = res.model.family.link.inverse(ci_lin)
        assert_allclose(pred.conf_int(), ci_mean, rtol=1e-10)


# Five clusters of four observations as literal numbers: draws of
# ``np.random.default_rng(20261116)`` (x = N(0, 1) + cluster effect,
# y = 1 + 0.5 x + cluster effect + N(0, 1)) rounded to three digits.
_X_R = [
    -0.498, 0.631, -0.725, -0.158, 0.188, -0.045, 0.953, -0.82, -1.694, 0.149,
    -1.593, 1.084, -0.03, 0.253, 1.177, -1.743, 1.035, 0.441, 0.723, 2.202,
]  # fmt: skip
_Y_R = [
    0.658, 0.262, 0.809, 2.362, 2.629, 1.19, 2.126, 1.318, 0.676, 0.004,
    -1.372, 0.825, 2.905, 2.356, 3.589, 2.267, 0.85, 0.71, -1.088, 1.71,
]  # fmt: skip

# Reference values from R 4.6.1 with sandwich 3.1.3 and lmtest 0.9.40. The
# confidence interval of the mean prediction at x0 is the interval of the
# intercept of the regression on x - x0:
#   d <- data.frame(g = rep(0:4, each = 4), x = x, y = y)
#   fit <- lm(y ~ I(x - x0), data = d)
#   v <- sandwich::vcovCL(fit, cluster = d$g, type = "HC1", cadjust = TRUE)
#   lmtest::coefci(fit, parm = "(Intercept)", vcov. = v, df = 4)  # use_t=True
#   lmtest::coefci(fit, parm = "(Intercept)", vcov. = v, df = Inf)  # False
_X0_R = [-0.5, 1.5]
_CI_R = {
    True: [
        [-0.408853965825686, 2.59075655066675],
        [0.0214611166522727, 3.18974733170635],
    ],
    False: [
        [0.032200199347159, 2.14970238549391],
        [0.487316874900288, 2.72389157345833],
    ],
}


@pytest.mark.parametrize("use_t", [True, False], ids=["t", "normal"])
def test_prediction_cluster_use_t_df_r_reference(use_t):
    # rtol 1e-8: statsmodels and R agree to about 1e-14 (the same linear
    # function of the parameters and the same HC1 cluster covariance), the
    # values of R are printed with 15 digits
    groups = np.repeat(np.arange(5), 4)
    exog = np.column_stack([np.ones(20), _X_R])
    res = OLS(_Y_R, exog).fit(
        cov_type="cluster", cov_kwds={"groups": groups}, use_t=use_t
    )
    exog0 = np.column_stack([np.ones(2), _X0_R])
    ci = res.get_prediction(exog0).conf_int()
    assert_allclose(ci, _CI_R[use_t], rtol=1e-8)


def _cov_type_case(case):
    # returns cov_type, cov_kwds and the expected df_resid_inference for OLS
    # with 8 groups of 15 observations (time index 0, ..., 14 in each group)
    n_groups, nobs_group = 8, 15
    n = n_groups * nobs_group
    groups = np.repeat(np.arange(n_groups), nobs_group)
    time = np.tile(np.arange(nobs_group), n_groups)
    if case in ["cluster", "cluster-crv3", "cluster-jk"]:
        return case, {"groups": groups}, n_groups - 1
    if case == "hac-panel":
        return case, {"groups": groups, "maxlags": 2}, n_groups - 1
    if case == "hac-groupsum":
        return case, {"time": time, "maxlags": 2}, n_groups - 1
    # two-way clustering, the smaller number of groups defines the df
    two_way = np.column_stack([groups, np.arange(n) % 5])
    return "cluster", {"groups": two_way}, 5 - 1


@pytest.mark.parametrize(
    "case",
    ["cluster", "cluster-crv3", "cluster-jk", "hac-panel", "hac-groupsum", "two-way"],
)
def test_prediction_cov_types_df_resid_inference(case):
    # every covariance type that sets df_resid_inference changes the degrees
    # of freedom of the prediction intervals, OLS can use all of them
    cov_type, cov_kwds, df_expected = _cov_type_case(case)
    mod, _, _ = _make_cluster_model("ols", n_groups=8, nobs_group=15)
    res = mod.fit(cov_type=cov_type, cov_kwds=cov_kwds, use_t=True)
    assert res.df_resid_inference == df_expected
    assert df_expected != res.df_resid

    exog0 = np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 2.0]])
    lin0, se0, ci_mean = _linear_ci_from_cov(res, exog0, sp_stats.t(df_expected))
    sf = res.get_prediction(exog0).summary_frame()
    assert_allclose(sf[["mean_ci_lower", "mean_ci_upper"]], ci_mean, rtol=1e-10)

    q = sp_stats.t.ppf(0.975, df_expected)
    se_obs = np.sqrt(se0**2 + res.scale)
    ci_obs = np.column_stack([lin0 - q * se_obs, lin0 + q * se_obs])
    assert_allclose(sf[["obs_ci_lower", "obs_ci_upper"]], ci_obs, rtol=1e-10)


@pytest.mark.parametrize("model", ["ols", "wls", "glm", "glm-poisson", "logit"])
@pytest.mark.parametrize("option", ["df_correction", "use_t"])
def test_prediction_cluster_df_resid_unchanged(model, option):
    # negative control: without df_resid_inference (df_correction=False) the
    # intervals use t(df_resid) and with use_t=False they are normal
    mod, groups, kwds = _make_cluster_model(model)
    cov_kwds = {"groups": groups}
    if option == "df_correction":
        cov_kwds["df_correction"] = False
    res = mod.fit(
        cov_type="cluster", cov_kwds=cov_kwds, use_t=option != "use_t", **kwds
    )
    if option == "df_correction":
        assert not hasattr(res, "df_resid_inference")
        dist = sp_stats.t(res.df_resid)
    else:
        dist = sp_stats.norm

    exog0 = np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 2.0]])
    _, _, ci_lin = _linear_ci_from_cov(res, exog0, dist)
    assert_allclose(res.t_test(exog0).conf_int(), ci_lin, rtol=1e-10)
    assert_allclose(_linear_ci_prediction(res, model, exog0), ci_lin, rtol=1e-10)
