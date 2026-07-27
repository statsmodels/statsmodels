"""
Tests for recursive least squares models

Author: Chad Fulton
License: Simplified-BSD
"""

from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest
from scipy.stats import norm

from statsmodels.datasets import macrodata
from statsmodels.genmod.api import GLM
from statsmodels.regression.linear_model import OLS
from statsmodels.regression.recursive_ls import RecursiveLS
from statsmodels.stats.diagnostic import recursive_olsresiduals
from statsmodels.tools import add_constant
from statsmodels.tools.eval_measures import aic, bic

current_path = Path(__file__).resolve().parent
results_R_path = Path("results") / "results_rls_R.csv"
results_R = pd.read_csv(current_path / results_R_path)
results_stata_path = Path("results") / "results_rls_stata.csv"
results_stata = pd.read_csv(current_path / results_stata_path)
dta = macrodata.load_pandas().data
dta.index = pd.date_range(start="1959-01-01", end="2009-07-01", freq="QS")
endog = dta["cpi"]
exog = add_constant(dta["m1"])


def test_endog():
    mod = RecursiveLS(endog.values, exog.values)
    res = mod.fit()
    mod_ols = OLS(endog, exog)
    res_ols = mod_ols.fit()
    assert_allclose(res.params, res_ols.params)
    mod = RecursiveLS(endog, dta["m1"].values)
    res = mod.fit()
    mod_ols = OLS(endog, dta["m1"])
    res_ols = mod_ols.fit()
    assert_allclose(res.params, res_ols.params)


def test_ols():
    mod = RecursiveLS(endog, dta["m1"])
    res = mod.fit()
    mod_ols = OLS(endog, dta["m1"])
    res_ols = mod_ols.fit()
    assert_allclose(res.params, res_ols.params)
    assert_allclose(res.bse, res_ols.bse)
    assert_allclose(res.filter_results.obs_cov[0, 0], res_ols.scale)
    actual = mod.endog[:, 0] - np.sum(
        mod["design", 0, :, :] * res.smoothed_state, axis=0
    )
    assert_allclose(actual, res_ols.resid)
    desired = mod_ols.loglike(res_ols.params, scale=res_ols.scale)
    assert_allclose(res.llf_recursive, desired)
    scale_alternative = (
        np.sum(
            (
                res.standardized_forecasts_error[0, 1:]
                * res.filter_results.obs_cov[0, 0] ** 0.5
            )
            ** 2
        )
        / mod.nobs
    )
    llf_alternative = np.log(
        norm.pdf(res.resid_recursive, loc=0, scale=scale_alternative**0.5)
    ).sum()
    assert_allclose(llf_alternative, res_ols.llf)
    actual = res.forecast(10, design=np.ones((1, 1, 10)))
    assert_allclose(actual, res_ols.predict(np.ones((10, 1))))
    assert_allclose(res.ess, res_ols.ess)
    assert_allclose(res.ssr, res_ols.ssr)
    assert_allclose(res.centered_tss, res_ols.centered_tss)
    assert_allclose(res.uncentered_tss, res_ols.uncentered_tss)
    assert_allclose(res.rsquared, res_ols.rsquared)
    assert_allclose(res.mse_model, res_ols.mse_model)
    assert_allclose(res.mse_resid, res_ols.mse_resid)
    assert_allclose(res.mse_total, res_ols.mse_total)
    actual = res.t_test("m1 = 0")
    desired = res_ols.t_test("m1 = 0")
    assert_allclose(actual.statistic, desired.statistic)
    assert_allclose(actual.pvalue, desired.pvalue, atol=1e-15)
    actual = res.f_test("m1 = 0")
    desired = res_ols.f_test("m1 = 0")
    assert_allclose(actual.statistic, desired.statistic)
    assert_allclose(actual.pvalue, desired.pvalue, atol=1e-15)
    actual_aic = aic(llf_alternative, res.nobs_effective, res.df_model)
    assert_allclose(actual_aic, res_ols.aic)
    actual_bic = bic(llf_alternative, res.nobs_effective, res.df_model)
    assert_allclose(actual_bic, res_ols.bic)


@pytest.mark.parametrize(
    "constraints", [None, "m1 + unemp = 1"], ids=["No constraint", "Constrained"]
)
def test_glm(constraints):
    endog = dta.infl
    exog = add_constant(dta[["unemp", "m1"]])
    constraints = None
    mod = RecursiveLS(endog, exog, constraints=constraints)
    res = mod.fit()
    mod_glm = GLM(endog, exog)
    if constraints is None:
        res_glm = mod_glm.fit()
    else:
        res_glm = mod_glm.fit_constrained(constraints=constraints)
    assert_allclose(res.params, res_glm.params)
    assert_allclose(res.bse, res_glm.bse, atol=1e-06)
    assert_allclose(res.filter_results.obs_cov[0, 0], res_glm.scale)
    assert_equal(res.df_model - 1, res_glm.df_model)
    actual = mod.endog[:, 0] - np.sum(
        mod["design", 0, :, :] * res.smoothed_state, axis=0
    )
    assert_allclose(actual, res_glm.resid_response, atol=1e-07)
    desired = mod_glm.loglike(res_glm.params, scale=res_glm.scale)
    assert_allclose(res.llf_recursive, desired)
    scale_alternative = (
        np.sum(
            (
                res.standardized_forecasts_error[0, 1:]
                * res.filter_results.obs_cov[0, 0] ** 0.5
            )
            ** 2
        )
        / mod.nobs
    )
    llf_alternative = np.log(
        norm.pdf(res.resid_recursive, loc=0, scale=scale_alternative**0.5)
    ).sum()
    assert_allclose(llf_alternative, res_glm.llf)
    if constraints is None:
        design = np.ones((1, 3, 10))
        actual = res.forecast(10, design=design)
        assert_allclose(actual, res_glm.predict(np.ones((10, 3))))
    else:
        design = np.ones((2, 3, 10))
        with pytest.raises(NotImplementedError):
            res.forecast(10, design=design)
    actual = res.t_test("m1 = 0")
    desired = res_glm.t_test("m1 = 0")
    assert_allclose(actual.statistic, desired.statistic)
    assert_allclose(actual.pvalue, desired.pvalue, atol=1e-15)
    actual = res.f_test("m1 = 0")
    desired = res_glm.f_test("m1 = 0")
    assert_allclose(actual.statistic, desired.statistic)
    assert_allclose(actual.pvalue, desired.pvalue)
    actual_aic = aic(llf_alternative, res.nobs_effective, res.df_model)
    assert_allclose(actual_aic, res_glm.aic)


def test_filter():
    mod = RecursiveLS(endog, exog)
    res = mod.filter()
    mod_ols = OLS(endog, exog)
    res_ols = mod_ols.fit()
    assert_allclose(res.params, res_ols.params)


def test_estimates():
    mod = RecursiveLS(endog, exog)
    res = mod.fit()
    assert_equal(mod.start_params, 0)
    assert_allclose(
        res.recursive_coefficients.filtered[:, 2:10].T,
        results_R.iloc[:8][["beta1", "beta2"]],
        rtol=1e-05,
    )
    assert_allclose(
        res.recursive_coefficients.filtered[:, 9:20].T,
        results_R.iloc[7:18][["beta1", "beta2"]],
    )
    assert_allclose(
        res.recursive_coefficients.filtered[:, 19:].T,
        results_R.iloc[17:][["beta1", "beta2"]],
    )
    mod_ols = OLS(endog, exog)
    res_ols = mod_ols.fit()
    assert_allclose(res.params, res_ols.params)


@pytest.mark.thread_unsafe(reason="uses matplotlib")
@pytest.mark.matplotlib
def test_plots(close_figures):
    import matplotlib.pyplot as plt
    from pandas.plotting import register_matplotlib_converters

    register_matplotlib_converters()
    exog = add_constant(dta[["m1", "pop"]])
    mod = RecursiveLS(endog, exog)
    res = mod.fit()
    res.plot_recursive_coefficient()
    res.plot_recursive_coefficient(variables=["m1"])
    res.plot_recursive_coefficient(variables=[0, "m1", "pop"])
    plt.close("all")
    res.plot_cusum()
    for alpha in [0.01, 0.1]:
        res.plot_cusum(alpha=alpha)
    with pytest.raises(ValueError):
        res.plot_cusum(alpha=0.123)
    res.plot_cusum_squares()
    plt.close("all")
    mod = RecursiveLS(endog.values, exog.values)
    res = mod.fit()
    res.plot_recursive_coefficient()
    res.plot_cusum()
    res.plot_cusum_squares()


def test_from_formula():
    mod = RecursiveLS.from_formula("cpi ~ m1", data=dta)
    res = mod.fit()
    mod_ols = OLS.from_formula("cpi ~ m1", data=dta)
    res_ols = mod_ols.fit()
    assert_allclose(res.params, res_ols.params)


def test_resid_recursive():
    mod = RecursiveLS(endog, exog)
    res = mod.fit()
    assert_allclose(res.resid_recursive[2:10].T, results_R.iloc[:8]["rec_resid"])
    assert_allclose(res.resid_recursive[9:20].T, results_R.iloc[7:18]["rec_resid"])
    assert_allclose(res.resid_recursive[19:].T, results_R.iloc[17:]["rec_resid"])
    assert_allclose(
        res.resid_recursive[3:], results_stata.iloc[3:]["rr"], atol=1e-05, rtol=1e-05
    )
    mod_ols = OLS(endog, exog)
    res_ols = mod_ols.fit()
    desired_resid_recursive = recursive_olsresiduals(res_ols)[4][2:]
    assert_allclose(res.resid_recursive[2:], desired_resid_recursive)


def test_recursive_olsresiduals_bad_input():
    from statsmodels.tsa.arima.model import ARIMA

    rs = np.random.RandomState(9889821)
    e = rs.standard_normal(250)
    y = e.copy()
    for i in range(1, y.shape[0]):
        y[i] += 0.1 + 0.8 * y[i - 1] + e[i]
    res = ARIMA(y[20:], order=(1, 0, 0), trend="c").fit()
    with pytest.raises(TypeError, match="res a regression results instance"):
        recursive_olsresiduals(res)


def test_cusum():
    mod = RecursiveLS(endog, exog)
    res = mod.fit()
    d = res.nobs_diffuse
    cusum = res.cusum * np.std(res.resid_recursive[d:], ddof=1)
    cusum -= res.resid_recursive[d]
    cusum /= np.std(res.resid_recursive[d + 1 :], ddof=1)
    cusum = cusum[1:]
    assert_allclose(cusum, results_stata.iloc[3:]["cusum"], atol=1e-06, rtol=1e-05)
    mod_ols = OLS(endog, exog)
    res_ols = mod_ols.fit()
    desired_cusum = recursive_olsresiduals(res_ols)[-2][1:]
    assert_allclose(res.cusum, desired_cusum, rtol=1e-06)
    actual_bounds = res._cusum_significance_bounds(
        alpha=0.05, ddof=1, points=np.arange(d + 1, res.nobs)
    )
    desired_bounds = results_stata.iloc[3:][["lw", "uw"]].T
    assert_allclose(actual_bounds, desired_bounds, rtol=1e-06)
    actual_bounds = res._cusum_significance_bounds(
        alpha=0.05, ddof=0, points=np.arange(d, res.nobs)
    )
    desired_bounds = recursive_olsresiduals(res_ols)[-1]
    assert_allclose(actual_bounds, desired_bounds)
    with pytest.raises(ValueError):
        res._cusum_squares_significance_bounds(alpha=0.123)


def test_stata():
    mod = RecursiveLS(endog, exog, loglikelihood_burn=3)
    with pytest.warns(UserWarning, match="Care should be used when applying"):
        res = mod.fit()
    d = max(res.nobs_diffuse, res.loglikelihood_burn)
    assert_allclose(
        res.resid_recursive[3:], results_stata.iloc[3:]["rr"], atol=1e-05, rtol=1e-05
    )
    assert_allclose(res.cusum, results_stata.iloc[3:]["cusum"], atol=1e-05)
    assert_allclose(res.cusum_squares, results_stata.iloc[3:]["cusum2"], atol=1e-05)
    actual_bounds = res._cusum_significance_bounds(
        alpha=0.05, ddof=0, points=np.arange(d + 1, res.nobs + 1)
    )
    desired_bounds = results_stata.iloc[3:][["lw", "uw"]].T
    assert_allclose(actual_bounds, desired_bounds, atol=1e-05)
    actual_bounds = res._cusum_squares_significance_bounds(
        alpha=0.05, points=np.arange(d + 1, res.nobs + 1)
    )
    desired_bounds = results_stata.iloc[3:][["lww", "uww"]].T
    assert_allclose(actual_bounds, desired_bounds, atol=0.01)


def test_constraints_stata():
    endog = dta["infl"]
    exog = add_constant(dta[["m1", "unemp"]])
    mod = RecursiveLS(endog, exog, constraints="m1 + unemp = 1")
    res = mod.fit()
    desired = [-0.7001083844336, -0.001847751406, 1.001847751406]
    assert_allclose(res.params, desired)
    desired = [0.4699552366, 0.0005369357, 0.0005369357]
    bse = np.asarray(res.bse)
    assert_allclose(bse[0], desired[0], atol=0.1)
    assert_allclose(bse[1:], desired[1:], atol=0.0001)
    desired = -534.4292052931121
    scale_alternative = (
        np.sum(
            (
                res.standardized_forecasts_error[0, 1:]
                * res.filter_results.obs_cov[0, 0] ** 0.5
            )
            ** 2
        )
        / mod.nobs
    )
    llf_alternative = np.log(
        norm.pdf(res.resid_recursive, loc=0, scale=scale_alternative**0.5)
    ).sum()
    assert_allclose(llf_alternative, desired)


def test_multiple_constraints():
    endog = dta["infl"]
    exog = add_constant(dta[["m1", "unemp", "cpi"]])
    constraints = ["m1 + unemp = 1", "cpi = 0"]
    mod = RecursiveLS(endog, exog, constraints=constraints)
    res = mod.fit()
    desired = [-0.7001083844336, -0.001847751406, 1.001847751406, 0]
    assert_allclose(res.params, desired, atol=1e-10)
    desired = [0.4699552366, 0.0005369357, 0.0005369357, 0]
    bse = np.asarray(res.bse)
    assert_allclose(bse[0], desired[0], atol=0.1)
    assert_allclose(bse[1:-1], desired[1:-1], atol=0.0001)
    desired = -534.4292052931121
    scale_alternative = (
        np.sum(
            (
                res.standardized_forecasts_error[0, 1:]
                * res.filter_results.obs_cov[0, 0] ** 0.5
            )
            ** 2
        )
        / mod.nobs
    )
    llf_alternative = np.log(
        norm.pdf(res.resid_recursive, loc=0, scale=scale_alternative**0.5)
    ).sum()
    assert_allclose(llf_alternative, desired)


def test_fix_params():
    mod = RecursiveLS([0, 1, 0, 1], [1, 1, 1, 1])
    with pytest.raises(
        ValueError, match="Linear constraints on coefficients should be given"
    ):
        with mod.fix_params({"const": 0.1}):
            mod.fit()
    with pytest.raises(
        ValueError, match="Linear constraints on coefficients should be given"
    ):
        mod.fit_constrained({"const": 0.1})
