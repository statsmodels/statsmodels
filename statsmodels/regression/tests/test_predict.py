"""
Created on Sun Apr 20 17:12:53 2014

author: Josef Perktold

"""

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest

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
