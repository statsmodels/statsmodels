"""
Author: Samuel Scherrer
"""

from statsmodels.compat.pandas import QUARTER_END
from statsmodels.compat.platform import PLATFORM_32, PLATFORM_LINUX32

from itertools import product
import json
import pathlib
import pickle
import re

import numpy as np
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
)
import pandas as pd
import pytest
import scipy.stats

from statsmodels.iolib.summary import Summary
from statsmodels.tools._decorators import cache_readonly
from statsmodels.tools.numdiff import approx_hess_cs
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    MissingDataError,
    PrecisionWarning,
)
from statsmodels.tools.tools import Bunch
from statsmodels.tsa import holtwinters
from statsmodels.tsa.exponential_smoothing.base import (
    StateSpaceMLEModel,
    StateSpaceMLEResults,
)
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.exponential_smoothing.initialization import (
    _initialization_heuristic,
    _initialization_simple,
)
import statsmodels.tsa.statespace.exponential_smoothing as statespace
from statsmodels.tsa.tests._contract import series_forms

# This contains tests for the exponential smoothing implementation in
# tsa/exponential_smoothing/ets.py.
#
# Tests are mostly done by comparing results with the R implementation in the
# package forecast for the datasets `oildata` (non-seasonal) and `austourists`
# (seasonal).
#
# Therefore, a parametrized pytest fixture ``setup_model`` is provided, which
# returns a constructed model, model parameters from R in the format expected
# by ETSModel, and a dictionary of reference results. Use like this:
#
#     def test_<testname>(setup_model):
#         model, params, results_R = setup_model
#         # perform some tests
#         ...

#
# UTILS
#

# Below I define parameter lists for all possible model and data combinations
# (for data, see below). These are used for parametrizing the pytest fixture
# ``setup_model``, which should be used for all tests comparing to R output.


def remove_invalid_models_from_list(modellist):
    # remove invalid models (no trend but damped)
    for i, model in enumerate(modellist):
        if model[1] is None and model[3]:
            del modellist[i]


ERRORS = ("add", "mul")
TRENDS = ("add", "mul", None)
SEASONALS = ("add", "mul", None)
DAMPED = (True, False)

MODELS_DATA_SEASONAL = list(
    product(
        ERRORS,
        TRENDS,
        ("add", "mul"),
        DAMPED,
        ("austourists",),
    )
)
MODELS_DATA_NONSEASONAL = list(
    product(
        ERRORS,
        TRENDS,
        (None,),
        DAMPED,
        ("oildata",),
    )
)
remove_invalid_models_from_list(MODELS_DATA_SEASONAL)
remove_invalid_models_from_list(MODELS_DATA_NONSEASONAL)


def short_model_name(error, trend, seasonal, damped=False):
    short_name = {"add": "A", "mul": "M", None: "N", True: "d", False: ""}
    return (
        short_name[error]
        + short_name[trend]
        + short_name[damped]
        + short_name[seasonal]
    )


ALL_MODELS_AND_DATA = MODELS_DATA_NONSEASONAL + MODELS_DATA_SEASONAL
ALL_MODEL_IDS = [short_model_name(*mod[:3], mod[3]) for mod in ALL_MODELS_AND_DATA]


@pytest.fixture(params=ALL_MODELS_AND_DATA, ids=ALL_MODEL_IDS)
def setup_model(
    request,
    austourists,
    oildata,
    ets_austourists_fit_results_R,
    ets_oildata_fit_results_R,
):
    params = request.param
    error, trend, seasonal, damped = params[0:4]
    data = params[4]
    if data == "austourists":
        data = austourists
        seasonal_periods = 4
        results = ets_austourists_fit_results_R[damped]
    else:
        data = oildata
        seasonal_periods = None
        results = ets_oildata_fit_results_R[damped]

    name = short_model_name(error, trend, seasonal)
    if name not in results:
        pytest.skip(f"model {name} not implemented or not converging in R")

    results_R = results[name]
    params = get_params_from_R(results_R)

    model = ETSModel
    model_args = (data,)
    model_kwargs = dict(
        seasonal_periods=seasonal_periods,
        error=error,
        trend=trend,
        seasonal=seasonal,
        damped_trend=damped,
    )

    return model, model_args, model_kwargs, params, results_R


@pytest.fixture
def austourists_model(austourists):
    return ETSModel(
        austourists,
        seasonal_periods=4,
        error="add",
        trend="add",
        seasonal="add",
        damped_trend=True,
    )


@pytest.fixture
def austourists_model_fit(austourists):
    mod = ETSModel
    mod_args = (austourists,)
    mod_kwargs = dict(
        seasonal_periods=4,
        error="add",
        trend="add",
        seasonal="add",
        damped_trend=True,
    )
    return mod, mod_args, mod_kwargs


@pytest.fixture
def oildata_model(oildata):
    return ETSModel(
        oildata,
        error="add",
        trend="add",
        damped_trend=True,
    )


#
# DATA
#


@pytest.fixture
def austourists():
    # austourists dataset from fpp2 package
    # https://cran.r-project.org/web/packages/fpp2/index.html
    data = [
        30.05251300,
        19.14849600,
        25.31769200,
        27.59143700,
        32.07645600,
        23.48796100,
        28.47594000,
        35.12375300,
        36.83848500,
        25.00701700,
        30.72223000,
        28.69375900,
        36.64098600,
        23.82460900,
        29.31168300,
        31.77030900,
        35.17787700,
        19.77524400,
        29.60175000,
        34.53884200,
        41.27359900,
        26.65586200,
        28.27985900,
        35.19115300,
        42.20566386,
        24.64917133,
        32.66733514,
        37.25735401,
        45.24246027,
        29.35048127,
        36.34420728,
        41.78208136,
        49.27659843,
        31.27540139,
        37.85062549,
        38.83704413,
        51.23690034,
        31.83855162,
        41.32342126,
        42.79900337,
        55.70835836,
        33.40714492,
        42.31663797,
        45.15712257,
        59.57607996,
        34.83733016,
        44.84168072,
        46.97124960,
        60.01903094,
        38.37117851,
        46.97586413,
        50.73379646,
        61.64687319,
        39.29956937,
        52.67120908,
        54.33231689,
        66.83435838,
        40.87118847,
        51.82853579,
        57.49190993,
        65.25146985,
        43.06120822,
        54.76075713,
        59.83447494,
        73.25702747,
        47.69662373,
        61.09776802,
        66.05576122,
    ]
    index = pd.date_range("1999-01-01", "2015-12-31", freq=QUARTER_END)
    return pd.Series(data, index)


@pytest.fixture
def oildata():
    # oildata dataset from fpp2 package
    # https://cran.r-project.org/web/packages/fpp2/index.html
    data = [
        111.0091346,
        130.8284341,
        141.2870879,
        154.2277747,
        162.7408654,
        192.1664835,
        240.7997253,
        304.2173901,
        384.0045673,
        429.6621566,
        359.3169299,
        437.2518544,
        468.4007898,
        424.4353365,
        487.9794299,
        509.8284478,
        506.3472527,
        340.1842374,
        240.2589210,
        219.0327876,
        172.0746632,
        252.5900922,
        221.0710774,
        276.5187735,
        271.1479517,
        342.6186005,
        428.3558357,
        442.3945534,
        432.7851482,
        437.2497186,
        437.2091599,
        445.3640981,
        453.1950104,
        454.4096410,
        422.3789058,
        456.0371217,
        440.3866047,
        425.1943725,
        486.2051735,
        500.4290861,
        521.2759092,
        508.9476170,
        488.8888577,
        509.8705750,
        456.7229123,
        473.8166029,
        525.9508706,
        549.8338076,
        542.3404698,
    ]
    return pd.Series(data, index=pd.date_range("1965", "2013", freq="YS"))


#
# REFERENCE RESULTS
#


def obtain_R_results(path):
    with path.open("r", encoding="utf-8") as f:
        R_results = json.load(f)

    # remove invalid models
    results = {}
    for damped in R_results:
        new_key = damped == "TRUE"
        results[new_key] = {}
        for model in R_results[damped]:
            if len(R_results[damped][model]):
                results[new_key][model] = R_results[damped][model]

    # get correct types
    for damped, result_value in results.items():
        for model in result_value:
            for key in ["alpha", "beta", "gamma", "phi", "sigma2"]:
                results[damped][model][key] = float(result_value[model][key][0])
            for key in [
                "states",
                "initstate",
                "residuals",
                "fitted",
                "forecast",
                "simulation",
            ]:
                results[damped][model][key] = np.asarray(result_value[model][key])
    return results


@pytest.fixture
def ets_austourists_fit_results_R():
    """
    Dictionary of ets fit results obtained with script ``results/fit_ets.R``.
    """
    path = pathlib.Path(__file__).parent / "results" / "fit_ets_results_seasonal.json"
    return obtain_R_results(path)


@pytest.fixture
def ets_oildata_fit_results_R():
    """
    Dictionary of ets fit results obtained with script ``results/fit_ets.R``.
    """
    path = (
        pathlib.Path(__file__).parent / "results" / "fit_ets_results_nonseasonal.json"
    )
    return obtain_R_results(path)


def fit_austourists_with_R_params(model, results_R, set_state=False):
    """
    Fit the model with params as found by R's forecast package
    """
    params = get_params_from_R(results_R)
    with model.fix_params(dict(zip(model.param_names, params, strict=True))):
        fit = model.fit(disp=False, param_index="integer")

    if set_state:
        states_R = get_states_from_R(results_R, model._k_states)
        fit.states = states_R
    return fit


def get_params_from_R(results_R):
    # get params from R
    params = [results_R[name] for name in ["alpha", "beta", "gamma", "phi"]]
    # in R, initial states are order l[-1], b[-1], s[-1], s[-2], ..., s[-m]
    params += list(results_R["initstate"])
    params = list(filter(np.isfinite, params))
    return params


def get_states_from_R(results_R, k_states):
    if k_states > 1:
        xhat_R = results_R["states"][1:, 0:k_states]
    else:
        xhat_R = results_R["states"][1:]
        xhat_R = np.reshape(xhat_R, (len(xhat_R), 1))
    return xhat_R


#
# BASIC TEST CASES
#


def test_fit_model_austouritsts(setup_model):
    model_class, model_args, model_kwargs, params, results_R = setup_model
    model = model_class(*model_args, **model_kwargs)
    model.fit(disp=False, param_index="integer")


#
# TEST OF MODEL EQUATIONS VS R
#


def test_smooth_vs_R(setup_model):
    model_class, model_args, model_kwargs, params, results_R = setup_model
    model = model_class(*model_args, **model_kwargs)

    yhat, xhat = model.smooth(params, return_raw=True)

    yhat_R = results_R["fitted"]
    xhat_R = get_states_from_R(results_R, model._k_states)

    assert_allclose(xhat, xhat_R, rtol=1e-5, atol=1e-5)
    assert_allclose(yhat, yhat_R, rtol=1e-5, atol=1e-5)


def test_residuals_vs_R(setup_model):
    model_class, model_args, model_kwargs, params, results_R = setup_model
    model = model_class(*model_args, **model_kwargs)

    yhat = model.smooth(params, return_raw=True)[0]

    residuals = model._residuals(yhat)
    assert_allclose(residuals, results_R["residuals"], rtol=1e-5, atol=1e-5)


def test_loglike_vs_R(setup_model):
    model_class, model_args, model_kwargs, params, results_R = setup_model
    model = model_class(*model_args, **model_kwargs)

    loglike = model.loglike(params)
    # the calculation of log likelihood in R is only up to a constant:
    const = -model.nobs / 2 * (np.log(2 * np.pi / model.nobs) + 1)
    loglike_R = results_R["loglik"][0] + const

    assert_allclose(loglike, loglike_R, rtol=1e-5, atol=1e-5)


def test_forecast_vs_R(setup_model):
    model_class, model_args, model_kwargs, params, results_R = setup_model
    model = model_class(*model_args, **model_kwargs)

    fit = fit_austourists_with_R_params(model, results_R, set_state=True)

    fcast = fit.forecast(4)
    expected = np.asarray(results_R["forecast"])

    assert_allclose(expected, fcast.values, rtol=1e-3, atol=1e-4)


def test_simulate_vs_R(setup_model):
    model_class, model_args, model_kwargs, params, results_R = setup_model
    model = model_class(*model_args, **model_kwargs)

    fit = fit_austourists_with_R_params(model, results_R, set_state=True)

    innov = np.asarray([[1.76405235, 0.40015721, 0.97873798, 2.2408932]]).T
    sim = fit.simulate(4, anchor="end", repetitions=1, random_errors=innov)
    expected = np.asarray(results_R["simulation"])

    assert_allclose(expected, sim.values, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(PLATFORM_32, reason="Fails on 32 bit systems")
def test_fit_vs_R(setup_model):
    model_class, model_args, model_kwargs, params, results_R = setup_model
    model = model_class(*model_args, **model_kwargs)

    # The fit of the damped additive trend and additive seasonal model has a
    # local optimum that the default starting values lead to on some platforms
    # and BLAS kernels, so start from the parameters estimated by R.
    if model.short_name == "AAdA":
        start = params
    else:
        start = None
    fit = model.fit(disp=True, pgtol=1e-8, start_params=start, param_index="integer")

    # check log likelihood: we want to have a fit that is better, i.e., a fit
    # that has a **higher** log-likelihood
    const = -model.nobs / 2 * (np.log(2 * np.pi / model.nobs) + 1)
    loglike_R = results_R["loglik"][0] + const
    loglike = fit.llf
    assert loglike >= loglike_R - 1e-4


def test_predict_vs_R(setup_model):
    model_class, model_args, model_kwargs, params, results_R = setup_model
    model = model_class(*model_args, **model_kwargs)

    fit = fit_austourists_with_R_params(model, results_R, set_state=True)

    n = fit.nobs
    prediction = fit.predict(end=n + 3, dynamic=n)

    yhat_R = results_R["fitted"]
    assert_allclose(prediction[:n], yhat_R, rtol=1e-5, atol=1e-5)

    forecast_R = results_R["forecast"]
    assert_allclose(prediction[n:], forecast_R, rtol=1e-3, atol=1e-4)


#
# OTHER TESTS
#


def test_initialization_known(austourists):
    initial_level, initial_trend = [36.46466837, 34.72584983]
    model = ETSModel(
        austourists,
        error="add",
        trend="add",
        damped_trend=True,
        initialization_method="known",
        initial_level=initial_level,
        initial_trend=initial_trend,
    )
    internal_params = model._internal_params(model._start_params)
    assert initial_level == internal_params[4]
    assert initial_trend == internal_params[5]
    assert internal_params[6] == 0


@pytest.mark.parametrize("trend", [False, None])
def test_initialization_simple_seasonal_no_trend(trend):
    # `False` is the no-trend sentinel used by ExponentialSmoothing, so a
    # seasonal model without a trend must not get an initial trend value
    y = np.arange(1.0, 25.0) + np.tile([1.0, 2.0, 3.0, 0.0], 6)
    level, initial_trend, seasonal = _initialization_simple(
        y, trend=trend, seasonal="add", seasonal_periods=4
    )
    assert initial_trend is None
    assert level is not None
    assert seasonal is not None


@pytest.mark.parametrize("trend", ["add", "mul"])
def test_initialization_simple_seasonal_with_trend(trend):
    y = np.arange(1.0, 25.0) + np.tile([1.0, 2.0, 3.0, 0.0], 6)
    _, initial_trend, _ = _initialization_simple(
        y, trend=trend, seasonal="add", seasonal_periods=4
    )
    assert_allclose(initial_trend, 1.0)


def test_initialization_heuristic(oildata):
    model_estimated = ETSModel(
        oildata,
        error="add",
        trend="add",
        damped_trend=True,
        initialization_method="estimated",
    )
    model_heuristic = ETSModel(
        oildata,
        error="add",
        trend="add",
        damped_trend=True,
        initialization_method="heuristic",
    )
    fit_estimated = model_estimated.fit(disp=False, param_index="integer")
    fit_heuristic = model_heuristic.fit(disp=False, param_index="integer")
    yhat_estimated = fit_estimated.fittedvalues.values
    yhat_heuristic = fit_heuristic.fittedvalues.values

    # this test is mostly just to see if it works, so we only test whether the
    # result is not totally off
    assert_allclose(yhat_estimated[10:], yhat_heuristic[10:], rtol=0.5)


def test_bounded_fit(oildata):
    beta = [0.99, 0.99]
    model1 = ETSModel(
        oildata,
        error="add",
        trend="add",
        damped_trend=True,
        bounds={"smoothing_trend": beta},
    )
    fit1 = model1.fit(disp=False, param_index="integer")
    assert fit1.smoothing_trend == 0.99

    # same using with fix_params semantic
    model2 = ETSModel(
        oildata,
        error="add",
        trend="add",
        damped_trend=True,
    )
    with model2.fix_params({"smoothing_trend": 0.99}):
        fit2 = model2.fit(disp=False, param_index="integer")
    assert fit2.smoothing_trend == 0.99
    assert_allclose(fit1.params, fit2.params)
    fit2.summary()  # check if summary runs without failing

    # using fit_constrained
    fit3 = model2.fit_constrained({"smoothing_trend": 0.99}, param_index="integer")
    assert fit3.smoothing_trend == 0.99
    assert_allclose(fit1.params, fit3.params)
    fit3.summary()


def test_seasonal_periods(austourists):
    # test auto-deduction of period
    model = ETSModel(austourists, error="add", trend="add", seasonal="add")
    assert model.seasonal_periods == 4

    # test if seasonal period raises error
    try:
        model = ETSModel(austourists, seasonal="add", seasonal_periods=0)
    except ValueError:
        pass


def test_simulate_keywords(austourists_model_fit):
    """
    check whether all keywords are accepted and work without throwing errors.
    """
    model_class, model_args, model_kwargs = austourists_model_fit
    fit = model_class(*model_args, **model_kwargs).fit(
        disp=False, param_index="integer"
    )

    rs = np.random.RandomState(3232912)
    # smoke test different random error options
    fit.simulate(4, repetitions=10, rng=rs)
    fit.simulate(4, repetitions=10, random_errors=scipy.stats.norm, rng=rs)
    fit.simulate(4, repetitions=10, random_errors=scipy.stats.norm(), rng=rs)
    fit.simulate(4, repetitions=10, random_errors=rs.randn(4, 10), rng=rs)
    fit.simulate(4, repetitions=10, random_errors="bootstrap", rng=rs)


@pytest.mark.thread_unsafe(reason="Issues and checks warnings")
def test_simulate_keywords_warnings(austourists_model_fit):
    """
    check whether all keywords are accepted and work without throwing errors.
    """
    model_class, model_args, model_kwargs = austourists_model_fit
    fit = model_class(*model_args, **model_kwargs).fit(
        disp=False, param_index="integer"
    )

    # test anchor
    with pytest.warns(FutureWarning, match="After statsmodels 0.15 is released"):
        sim_0 = fit.simulate(4, anchor=-1, rng=0).values
    with pytest.warns(FutureWarning, match="After statsmodels 0.15 is released"):
        sim_1 = fit.simulate(4, anchor="2015-12-31", rng=0).values
    assert_almost_equal(sim_0, sim_1)
    with pytest.warns(FutureWarning, match="After statsmodels 0.15 is released"):
        sim_2 = fit.simulate(4, anchor="end", rng=0).values
    with pytest.warns(FutureWarning, match="After statsmodels 0.15 is released"):
        sim_3 = fit.simulate(4, anchor="2015-12-31", rng=0).values
    assert_almost_equal(sim_2, sim_3)

    # test seeding
    with pytest.warns(FutureWarning, match="After statsmodels 0.15 is released"):
        res = fit.simulate(4, repetitions=10, rng=10).values
    res2 = fit.simulate(4, repetitions=10, rng=np.random.RandomState(10)).values
    assert np.all(res == res2)


def test_predict_ranges(austourists_model_fit):
    # in total 68 observations
    model_class, model_args, model_kwargs = austourists_model_fit
    fit = model_class(*model_args, **model_kwargs).fit(
        disp=False, param_index="integer"
    )

    # first prediction is 0, last is 10 -> 11 predictions
    pred = fit.predict(start=0, end=10)
    assert len(pred) == 11

    pred = fit.predict(start=10, end=20)
    assert len(pred) == 11

    pred = fit.predict(start=10, dynamic=10, end=30)
    assert len(pred) == 21

    # try boolean dynamic
    pred = fit.predict(start=0, dynamic=True, end=70)
    assert len(pred) == 71
    pred = fit.predict(start=0, dynamic=True, end=70)
    assert len(pred) == 71

    # try only out oof sample prediction
    pred = fit.predict(start=80, end=84)
    assert len(pred) == 5


def test_summary(austourists_model_fit):
    # just try to run summary to see if it works
    model_class, model_args, model_kwargs = austourists_model_fit
    fit = model_class(*model_args, **model_kwargs).fit(
        disp=False, param_index="integer"
    )
    fit.summary()

    # now without estimated initial states
    ets_model = model_class(*model_args, **model_kwargs)
    ets_model.set_initialization_method("heuristic")
    fit = ets_model.fit(disp=False, param_index="integer")
    fit.summary()

    # and with fixed params
    ets_model = model_class(*model_args, **model_kwargs)
    fit = ets_model.fit_constrained({"smoothing_trend": 0.9}, param_index="integer")
    fit.summary()


def test_score(austourists_model_fit):
    model_class, model_args, model_kwargs = austourists_model_fit
    fit = model_class(*model_args, **model_kwargs).fit(
        disp=False, param_index="integer"
    )

    score_cs = fit.model.score(fit.params)
    score_fd = fit.model.score(
        fit.params,
        approx_complex_step=False,
        approx_centered=True,
    )
    assert_almost_equal(score_cs, score_fd, 4)


def test_hessian(austourists_model_fit):
    # The hessian approximations are not very consistent, but the test makes
    # sure they run
    model_class, model_args, model_kwargs = austourists_model_fit
    fit = model_class(*model_args, **model_kwargs).fit(
        disp=False, param_index="integer"
    )

    fit.model.hessian(fit.params)
    # The finite difference hessian always warns about its approximation error
    with pytest.warns(PrecisionWarning, match="Calculation of the Hessian"):
        fit.model.hessian(
            fit.params,
            approx_complex_step=False,
            approx_centered=True,
        )


def test_diagnostics_invalid_method_raises(austourists_model_fit):
    model_class, model_args, model_kwargs = austourists_model_fit
    fit = model_class(*model_args, **model_kwargs).fit(
        disp=False, param_index="integer"
    )

    with pytest.raises(ValueError, match="method"):
        fit.test_serial_correlation(method="not-a-method")
    with pytest.raises(ValueError, match="method"):
        fit.test_heteroskedasticity(method="not-a-method")
    with pytest.raises(ValueError, match="method"):
        fit.test_normality(method="not-a-method")


def test_prediction_results(austourists_model_fit):
    # simple test case starting at 0
    model_class, model_args, model_kwargs = austourists_model_fit
    fit = model_class(*model_args, **model_kwargs).fit(
        disp=False, param_index="integer"
    )

    pred = fit.get_prediction(
        start=0,
        dynamic=30,
        end=40,
    )
    summary = pred.summary_frame()
    assert len(summary["mean"].values) == 41
    assert np.all(~np.isnan(summary["mean"]))

    # simple test case starting at not 0
    pred = fit.get_prediction(start=10, dynamic=30, end=40)
    summary = pred.summary_frame()
    assert len(summary["mean"].values) == 31
    assert np.all(~np.isnan(summary["mean"]))

    # long out of sample prediction
    pred = fit.get_prediction(start=0, dynamic=30, end=80)
    summary = pred.summary_frame()
    assert len(summary["mean"].values) == 81
    assert np.all(~np.isnan(summary["mean"]))

    # long out of sample, starting in-sample
    pred = fit.get_prediction(start=67, end=80)
    summary = pred.summary_frame()
    assert len(summary["mean"].values) == 14
    assert np.all(~np.isnan(summary["mean"]))

    # long out of sample, starting at end of sample
    pred = fit.get_prediction(start=68, end=80)
    summary = pred.summary_frame()
    assert len(summary["mean"].values) == 13
    assert np.all(~np.isnan(summary["mean"]))

    # long out of sample, starting just out of sample
    pred = fit.get_prediction(start=69, end=80)
    summary = pred.summary_frame()
    assert len(summary["mean"].values) == 12
    assert np.all(~np.isnan(summary["mean"]))

    # long out of sample, starting long out of sample
    pred = fit.get_prediction(start=79, end=80)
    summary = pred.summary_frame()
    assert len(summary["mean"].values) == 2
    assert np.all(~np.isnan(summary["mean"]))

    # long out of sample, `start`== `end`
    pred = fit.get_prediction(start=80, end=80)
    summary = pred.summary_frame()
    assert len(summary["mean"].values) == 1
    assert np.all(~np.isnan(summary["mean"]))


@pytest.fixture
def statespace_comparison(austourists):
    ets_model = ETSModel(
        austourists,
        seasonal_periods=4,
        error="add",
        trend="add",
        seasonal="add",
        damped_trend=True,
    )
    ets_results = ets_model.fit(disp=False, param_index="integer")

    statespace_model = statespace.ExponentialSmoothing(
        austourists,
        trend=True,
        damped_trend=True,
        seasonal=4,
        initialization_method="known",
        initial_level=ets_results.initial_level,
        initial_trend=ets_results.initial_trend,
        # See GH 7893
        initial_seasonal=ets_results.initial_seasonal[::-1],
    )
    with statespace_model.fix_params(
        {
            "smoothing_level": ets_results.smoothing_level,
            "smoothing_trend": ets_results.smoothing_trend,
            "smoothing_seasonal": ets_results.smoothing_seasonal,
            "damping_trend": ets_results.damping_trend,
        }
    ):
        statespace_results = statespace_model.fit()
    ets_results.test_serial_correlation("ljungbox")
    statespace_results.test_serial_correlation("ljungbox")
    return ets_results, statespace_results


def test_results_vs_statespace(statespace_comparison):
    ets_results, statespace_results = statespace_comparison

    assert_almost_equal(ets_results.llf, statespace_results.llf)
    assert_almost_equal(ets_results.scale, statespace_results.scale)
    assert_almost_equal(
        ets_results.fittedvalues.values, statespace_results.fittedvalues.values
    )

    # compare diagnostics
    assert_almost_equal(
        ets_results.test_serial_correlation(method="ljungbox"),
        statespace_results.test_serial_correlation(method="ljungbox"),
    )
    assert_almost_equal(
        ets_results.test_normality(method="jarquebera"),
        statespace_results.test_normality(method="jarquebera"),
    )

    # heteroskedasticity is somewhat different, because of burn in period?
    ets_het = ets_results.test_heteroskedasticity(method="breakvar")[0]
    statespace_het = statespace_results.test_heteroskedasticity(method="breakvar")[0]
    # het[0] is test statistic, het[1] p-value
    if not PLATFORM_LINUX32:
        # Skip on Linux-32 bit due to random failures. These values are not
        # close at all in any way so it isn't clear what this is testing
        assert_allclose(ets_het[0], statespace_het[0], rtol=0.2)
        assert_allclose(ets_het[1], statespace_het[1], rtol=0.7)

    # the undocumented "d" alias for "decreasing" still works but warns,
    # and is equivalent to spelling out "decreasing"
    with pytest.warns(FutureWarning, match="is a deprecated alias"):
        ets_het_alias = ets_results.test_heteroskedasticity(
            method="breakvar", alternative="d"
        )
    ets_het_canonical = ets_results.test_heteroskedasticity(
        method="breakvar", alternative="decreasing"
    )
    assert_allclose(ets_het_alias, ets_het_canonical)

    with pytest.raises(ValueError, match="alternative must be one of"):
        ets_results.test_heteroskedasticity(method="breakvar", alternative="bogus")


def test_prediction_results_vs_statespace(statespace_comparison):
    ets_results, statespace_results = statespace_comparison

    # comparison of two predictions
    ets_pred = ets_results.get_prediction(start=10, dynamic=10, end=40)
    statespace_pred = statespace_results.get_prediction(start=10, dynamic=10, end=40)

    statespace_summary = statespace_pred.summary_frame()
    ets_summary = ets_pred.summary_frame()

    # import matplotlib.pyplot as plt
    # plt.switch_backend('TkAgg')
    # plt.plot(ets_summary["mean"] - statespace_summary["mean"])
    # plt.grid()
    # plt.show()

    assert_almost_equal(
        ets_summary["mean"].values[:-10],
        statespace_summary["mean"].values[:-10],
    )

    assert_almost_equal(
        ets_summary["mean"].values[-10:],
        statespace_summary["mean"].values[-10:],
        4,
    )

    # comparison of dynamic prediction at end of sample -> this works
    ets_pred = ets_results.get_prediction(
        start=60,
        end=80,
    )
    statespace_pred = statespace_results.get_prediction(start=60, end=80)
    statespace_summary = statespace_pred.summary_frame()
    ets_summary = ets_pred.summary_frame()

    assert_almost_equal(
        ets_summary["mean"].values, statespace_summary["mean"].values, 4
    )


@pytest.mark.matplotlib
@pytest.mark.slow
def test_prediction_results_slow_AAN(oildata, close_figures):
    # slow test with high number of simulation repetitions for comparison
    # Note: runs succesfull with specified tolerance
    fit = ETSModel(oildata, error="add", trend="add").fit(
        disp=False, param_index="integer"
    )

    pred_exact = fit.get_prediction(start=40, end=55)
    summary_exact = pred_exact.summary_frame()

    pred_sim = fit.get_prediction(
        start=40,
        end=55,
        simulate_repetitions=int(1e6),
        rng=np.random.RandomState(11),
        method="simulated",
    )
    summary_sim = pred_sim.summary_frame()
    # check if mean converges to expected mean
    assert_allclose(
        summary_sim["mean"].values,
        summary_sim["mean_numerical"].values,
        rtol=1e-3,
        atol=1e-3,
    )

    import matplotlib.pyplot as plt

    for i in range(1000):
        plt.plot(
            pred_sim._results.simulation_results.iloc[:, i],
            color="grey",
            alpha=0.1,
        )
    plt.plot(oildata[40:], "-", label="data")
    plt.plot(summary_exact["mean"], "--", label="mean")
    plt.plot(summary_sim["pi_lower"], ":", label="sim lower")
    plt.plot(summary_exact["pi_lower"], ".-", label="exact lower")
    plt.plot(summary_sim["pi_upper"], ":", label="sim upper")
    plt.plot(summary_exact["pi_upper"], ".-", label="exact upper")
    # plt.legend()

    # check if prediction intervals are equal
    assert_allclose(
        summary_sim["pi_lower"].values,
        summary_exact["pi_lower"].values,
        rtol=1e-2,
        atol=1e-4,
    )

    assert_allclose(
        summary_sim["pi_upper"].values,
        summary_exact["pi_upper"].values,
        rtol=1e-2,
        atol=1e-4,
    )


@pytest.mark.matplotlib
@pytest.mark.slow
def test_prediction_results_slow_AAdA(austourists, close_figures):
    # slow test with high number of simulation repetitions for comparison
    # Note: succesfull with specified tolerance
    fit = ETSModel(
        austourists,
        error="add",
        trend="add",
        seasonal="add",
        damped_trend=True,
        seasonal_periods=4,
    ).fit(disp=False, param_index="integer")
    pred_exact = fit.get_prediction(start=60, end=75)
    summary_exact = pred_exact.summary_frame()

    pred_sim = fit.get_prediction(
        start=60,
        end=75,
        simulate_repetitions=int(1e6),
        rng=np.random.default_rng(2381032),
        method="simulated",
    )
    summary_sim = pred_sim.summary_frame()
    # check if mean converges to expected mean
    assert_allclose(
        summary_sim["mean"].values,
        summary_sim["mean_numerical"].values,
        rtol=1e-3,
        atol=1e-3,
    )

    import matplotlib.pyplot as plt

    for i in range(1000):
        plt.plot(
            pred_sim._results.simulation_results.iloc[:, i],
            color="grey",
            alpha=0.1,
        )
    plt.plot(fit.endog[60:], "-", label="data")
    plt.plot(summary_exact["mean"], "--", label="mean")
    plt.plot(summary_sim["pi_lower"], ":", label="sim lower")
    plt.plot(summary_exact["pi_lower"], ".-", label="exact lower")
    plt.plot(summary_sim["pi_upper"], ":", label="sim upper")
    plt.plot(summary_exact["pi_upper"], ".-", label="exact upper")

    # check if prediction intervals are equal
    assert_allclose(
        summary_sim["pi_lower"].values,
        summary_exact["pi_lower"].values,
        rtol=2e-2,
        atol=1e-4,
    )

    assert_allclose(
        summary_sim["pi_upper"].values,
        summary_exact["pi_upper"].values,
        rtol=2e-2,
        atol=1e-4,
    )


def test_convergence_simple():
    # issue 6883
    rs = np.random.RandomState(0)
    e = rs.standard_normal(12000)
    y = e.copy()
    for i in range(1, e.shape[0]):
        y[i] = y[i - 1] - 0.2 * e[i - 1] + e[i]
    y = y[200:]
    mod = holtwinters.ExponentialSmoothing(y, initialization_method="estimated")
    res = mod.fit()
    ets_res = ETSModel(y).fit()

    # the smoothing level should be very similar, the initial state might be
    # different as it doesn't influence the final result too much
    assert_allclose(
        res.params["smoothing_level"],
        ets_res.smoothing_level,
        rtol=1e-4,
        atol=1e-4,
    )

    # the first few values are influenced by differences in initial state, so
    # we don't test them here
    assert_allclose(
        res.fittedvalues[10:], ets_res.fittedvalues[10:], rtol=1e-4, atol=1e-4
    )


def test_exact_prediction_intervals(austourists_model_fit):
    model_class, model_args, model_kwargs = austourists_model_fit
    fit = model_class(*model_args, **model_kwargs).fit(
        disp=False, param_index="integer"
    )
    fit = fit._results

    class DummyModel:
        def __init__(self, short_name):
            self.short_name = short_name

    # compare AAdN with AAN
    fit.damping_trend = 1 - 1e-3
    fit.model = DummyModel("AAdN")
    steps = 5
    s_AAdN = fit._relative_forecast_variance(steps)
    fit.model = DummyModel("AAN")
    s_AAN = fit._relative_forecast_variance(steps)
    assert_almost_equal(s_AAdN, s_AAN, 2)

    # compare AAdA with AAA
    fit.damping_trend = 1 - 1e-3
    fit.model = DummyModel("AAdA")
    steps = 5
    s_AAdA = fit._relative_forecast_variance(steps)
    fit.model = DummyModel("AAA")
    s_AAA = fit._relative_forecast_variance(steps)
    assert_almost_equal(s_AAdA, s_AAA, 2)


def test_one_step_ahead(setup_model):
    rs = np.random.RandomState(78437941)
    model_class, model_args, model_kwargs, params, results_R = setup_model
    model = model_class(*model_args, **model_kwargs)

    model2 = ETSModel(
        pd.Series(model.endog),
        seasonal_periods=model.seasonal_periods,
        error=model.error,
        trend=model.trend,
        seasonal=model.seasonal,
        damped_trend=model.damped_trend,
    )
    res = model2.smooth(params, param_index="integer")
    fcast1 = res.forecast(steps=1)
    fcast2 = res.forecast(steps=2)
    assert_allclose(fcast1.iloc[0], fcast2.iloc[0])

    pred1 = res.get_prediction(
        start=model2.nobs, end=model2.nobs, simulate_repetitions=2, rng=rs
    )
    res.get_prediction(
        start=model2.nobs, end=model2.nobs + 1, simulate_repetitions=2, rng=rs
    )
    df1 = pred1.summary_frame(alpha=0.05)
    df2 = pred1.summary_frame(alpha=0.05)
    assert_allclose(df1.iloc[0, 0], df2.iloc[0, 0])


@pytest.mark.parametrize("trend", [None, "add"])
@pytest.mark.parametrize("seasonal", [None, "add"])
@pytest.mark.parametrize("nobs", [9, 10])
def test_estimated_initialization_short_data(oildata, trend, seasonal, nobs):
    # GH 7319
    res = ETSModel(
        oildata[:nobs],
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=4,
        initialization_method="estimated",
    ).fit(param_index="integer")
    assert ~np.any(np.isnan(res.params))


@pytest.mark.parametrize("method", ["estimated", "heuristic"])
def test_seasonal_order(method):
    seasonal = np.arange(12.0)
    time_series = np.array(list(seasonal) * 100)
    mod = ETSModel(
        time_series,
        seasonal="add",
        seasonal_periods=12,
        initialization_method=method,
    )
    if method == "estimated":
        # The series has no noise, so the likelihood has no maximum: it grows
        # without bound as the residual variance goes to zero. The start
        # values already fit the series up to rounding, and the optimizer
        # stops there without converging.
        with pytest.warns(ConvergenceWarning, match="Maximum Likelihood"):
            res = mod.fit()
    else:
        res = mod.fit()
    assert_allclose(
        res.initial_seasonal + res.initial_level,
        seasonal,
        atol=1e-4,
        rtol=1e-4,
    )
    assert res.mae < 1e-6


def test_aicc_0_dof():
    # GH8172
    endog = [109.0, 101.0, 104.0, 90.0, 105.0]

    model = ETSModel(
        endog=endog,
        initialization_method="known",
        initial_level=100.0,
        initial_trend=0.0,
        error="add",
        trend="add",
        damped_trend=True,
    )
    aicc = model.fit().aicc
    assert not np.isfinite(aicc)
    assert aicc > 0


def test_summary_after_remove_data(oildata):
    # summary() must still work after remove_data() has been called
    model = ETSModel(oildata, error="add", trend="add", damped_trend=True)
    res = model.fit(disp=False, param_index="integer")

    assert isinstance(res.summary(), Summary)
    res.remove_data()
    assert isinstance(res.summary(), Summary)


# Monthly demand series from GH 8551 (24 observations, seasonal_periods=12)
GH8551_DATA = np.array(
    [
        52.0, 250.43, 261.01, 265.0, 56.77, 16.0, 50.09, 65.0, 21.0, 89.5,
        20.0, 1.0, 123.0, 84.0, 91.0, 163.0, 16.0, 27.0, 78.5, 38.0, 61.0,
        5.0, 82.0, 15.0,
    ]
)  # fmt: skip

GH8551_MODELS = [
    ("gh8551", "mul", "add", True, "mul"),
    ("gh8551", "mul", None, False, "mul"),
    ("gh8551", "mul", None, False, "add"),
    ("gh8551", "add", "add", True, "add"),
] + [
    ("austourists", error, trend, damped, seasonal)
    for error, (trend, damped), seasonal in product(
        ["add", "mul"],
        [(None, False), ("add", False), ("add", True), ("mul", False)],
        ["add", "mul"],
    )
]


@pytest.mark.parametrize("data, error, trend, damped, seasonal", GH8551_MODELS)
def test_estimated_start_params_are_heuristic_states(
    austourists, data, error, trend, damped, seasonal
):
    # GH 8551: the start parameters of initialization_method="estimated" are
    # the heuristic initial states. The parameter vector holds the initial
    # seasonals as s[-1], ..., s[-m], while model.initial_seasonal starts with
    # the seasonal of the first observation, s[-m].
    if data == "gh8551":
        endog, m = GH8551_DATA, 12
    else:
        endog, m = austourists, 4
    kwargs = dict(
        error=error,
        trend=trend,
        damped_trend=damped,
        seasonal=seasonal,
        seasonal_periods=m,
    )
    model = ETSModel(endog, initialization_method="estimated", **kwargs)
    model_heuristic = ETSModel(endog, initialization_method="heuristic", **kwargs)
    start_params = model.start_params
    smoothing_params = start_params[: len(model_heuristic.param_names)]

    assert_allclose(
        model.loglike(start_params),
        model_heuristic.loglike(smoothing_params),
        rtol=1e-12,
    )
    res = model.smooth(start_params, param_index="integer")
    res_heuristic = model_heuristic.smooth(smoothing_params, param_index="integer")
    assert_allclose(res.initial_seasonal, model_heuristic.initial_seasonal)
    assert_allclose(res.initial_state, res_heuristic.initial_state)
    assert_allclose(res.fittedvalues, res_heuristic.fittedvalues)
    # start_params does not change the heuristic initial states of the model
    assert_allclose(model.initial_seasonal, model_heuristic.initial_seasonal)
    assert_allclose(model.start_params, start_params)


@pytest.mark.parametrize("seasonal", ["add", "mul"])
@pytest.mark.parametrize("trend", [None, "add"])
@pytest.mark.parametrize(
    "m, nobs", [(3, 11), (3, 40), (4, 12), (4, 40), (7, 15), (7, 60)]
)
def test_estimated_start_params_seasonals_normalized(seasonal, trend, m, nobs):
    # GH 8551: the start values of the initial seasonals are used as they
    # are, so the heuristic initialization (long series) and the simple one
    # (short series, nobs < 10 + 2 * (m // 2)) must already sum to 0
    # (additive) or to m (multiplicative). The parameter vector starts with
    # initial_seasonal.0 = s[-1], while model.initial_seasonal starts with
    # s[-m].
    rng = np.random.default_rng(8551 + m + nobs)
    t = np.arange(nobs)
    endog = 50 + 0.3 * t + 8 * np.sin(2 * np.pi * t / m) + rng.normal(0, 3, nobs)
    kwargs = dict(trend=trend, seasonal=seasonal, seasonal_periods=m)
    model = ETSModel(endog, initialization_method="estimated", **kwargs)

    # check that both initialization paths are tested
    if nobs < 10 + 2 * (m // 2):
        init = _initialization_simple
    else:
        init = _initialization_heuristic
    _, _, expected = init(endog, trend=trend, seasonal=seasonal, seasonal_periods=m)
    assert_allclose(model.initial_seasonal, expected)

    total = m if seasonal == "mul" else 0.0
    start_seasonal = model.start_params[-m:]
    assert_allclose(np.sum(start_seasonal), total, rtol=1e-12, atol=1e-12)
    assert_allclose(np.sum(model.initial_seasonal), total, rtol=1e-12, atol=1e-12)
    assert_allclose(start_seasonal, model.initial_seasonal[::-1])


@pytest.mark.parametrize("trend, damped_trend", [("add", True), (None, False)])
def test_estimated_initial_seasonal_mul(trend, damped_trend):
    # GH 8551: with multiplicative seasonality, the estimated initialization
    # divided the heuristic initial seasonals by the last one (0.0175 here)
    # and fixed it to 1. The fit then ended in a much worse optimum than the
    # heuristic initial states, which are a feasible point of the same
    # problem, and the forecasts of this positive series were all negative.
    kwargs = dict(
        error="mul",
        trend=trend,
        damped_trend=damped_trend,
        seasonal="mul",
        seasonal_periods=12,
    )
    model = ETSModel(GH8551_DATA, initialization_method="estimated", **kwargs)
    res_heuristic = ETSModel(
        GH8551_DATA, initialization_method="heuristic", **kwargs
    ).fit(disp=False)
    res_estimated = model.fit(disp=False)

    # the optimizer does not end below its start, the heuristic states
    assert res_estimated.llf >= model.loglike(model.start_params)
    # this holds with a margin of more than 10 here
    assert res_estimated.llf >= res_heuristic.llf

    fcast = res_estimated.forecast(12)
    assert np.all(np.isfinite(fcast))
    assert np.all(fcast > 0)
    assert np.all(fcast < 2 * GH8551_DATA.max())

    # the initial seasonals are normalized to have mean 1
    assert_allclose(res_estimated.initial_seasonal.mean(), 1.0, rtol=1e-10)
    assert_allclose(res_estimated.params[-12:].mean(), 1.0, rtol=1e-10)


@pytest.mark.parametrize("error", ["add", "mul"])
@pytest.mark.parametrize("trend", [None, "add"])
@pytest.mark.parametrize("seasonal", ["add", "mul"])
def test_estimated_initial_seasonal_normalized(austourists, error, trend, seasonal):
    # With initialization_method="estimated", m - 1 initial seasonals are
    # estimated and the last one is determined so that the initial seasonals
    # sum to 0 (additive) or to m (multiplicative), as in R's forecast::ets.
    model = ETSModel(
        austourists,
        error=error,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=4,
        initialization_method="estimated",
    )
    start_params = model.start_params
    res = model.fit(disp=False, param_index="integer")
    assert_allclose(res.llf, model.loglike(res.params))

    params = np.asarray(res.params)
    for seas in (start_params[-4:], params[-4:], res.initial_seasonal):
        if seasonal == "mul":
            assert_allclose(np.mean(seas), 1.0, rtol=1e-10)
        else:
            assert_allclose(np.sum(seas), 0.0, atol=1e-8)


@pytest.mark.parametrize("seasonal", ["add", "mul"])
def test_estimated_initial_seasonal_fixed_or_bounded_other(austourists, seasonal):
    # A fixed or bounded initial seasonal other than the last one does not
    # change that the last one is computed from the others.
    kwargs = dict(error="add", trend="add", seasonal=seasonal, seasonal_periods=4)
    total = 4.0 if seasonal == "mul" else 0.0
    model = ETSModel(austourists, **kwargs)
    value = model.start_params[-3] + (0.05 if seasonal == "mul" else 1.0)
    with model.fix_params({"initial_seasonal.1": value}):
        res = model.fit(disp=False, param_index="integer")
    params = np.asarray(res.params)
    assert params[-3] == value
    assert_allclose(params[-1], total - np.sum(params[-4:-1]), rtol=1e-12)
    assert_allclose(res.llf, model.loglike(params), rtol=1e-12)

    model = ETSModel(austourists, **kwargs)
    bounds = (value - 0.01, value) if seasonal == "mul" else (value - 0.1, value)
    model.set_bounds({"initial_seasonal.2": bounds})
    res = model.fit(disp=False, param_index="integer")
    params = np.asarray(res.params)
    assert bounds[0] <= params[-2] <= bounds[1]
    assert_allclose(params[-1], total - np.sum(params[-4:-1]), rtol=1e-12)
    assert_allclose(res.llf, model.loglike(params), rtol=1e-12)


def test_estimated_initial_seasonal_fixed_last(austourists):
    # If the user fixes the last initial seasonal, it is not determined by the
    # others any more and the fixed value is used.
    kwargs = dict(error="add", trend="add", seasonal="add", seasonal_periods=4)
    model = ETSModel(austourists, **kwargs)
    with model.fix_params({"initial_seasonal.3": 0.0}):
        res = model.fit(disp=False, param_index="integer")
    assert np.asarray(res.params)[-1] == 0.0
    assert np.isfinite(res.llf)
    assert "initial_seasonal.3 (fixed)" in res.param_names
    # the initial level and seasonals are identified by the fixed value
    assert np.all(np.isfinite(res.bse[4:-1]))

    # results from smooth (no fit, so the fixed value is not in the bounds
    # of the model) use the same parametrization for the covariance
    model = ETSModel(austourists, **kwargs)
    with model.fix_params({"initial_seasonal.3": 0.0}):
        res_smooth = model.smooth(res.params, param_index="integer")
    assert_allclose(res_smooth.cov_params(), res.cov_params(), rtol=1e-10)


def test_estimated_initial_seasonal_bounded_last(austourists):
    # A bound cannot be imposed on a computed value, so if the user gives
    # bounds for the last initial seasonal, it is a free parameter within
    # them, as before GH 8551. The fitted sum-zero value would be outside
    # these bounds.
    kwargs = dict(error="add", trend="add", seasonal="add", seasonal_periods=4)
    model = ETSModel(austourists, **kwargs)
    model.set_bounds({"initial_seasonal.3": (20.0, 21.0)})
    res = model.fit(disp=False, param_index="integer")
    params = np.asarray(res.params)
    assert 20.0 <= params[-1] <= 21.0
    assert_allclose(res.llf, model.loglike(params), rtol=1e-12)


@pytest.mark.parametrize("seasonal", ["add", "mul"])
@pytest.mark.parametrize("fixed", [None, 0, 1, 2])
def test_loglike_internal_derived_last_seasonal(austourists, seasonal, fixed):
    # GH 8551: the objective of the fit is `_loglike_internal` evaluated at
    # the free parameters, with the last initial seasonal computed from the
    # others, including a fixed initial seasonal other than the last one. It
    # must equal the public log-likelihood at the parameters with
    # s[-m] = total - sum(s[-1], ..., s[-m+1]). The tests of the fits only
    # look at parameters that are recomputed after the fit, so they cannot
    # see an objective that ignores a fixed seasonal. The point is a random
    # one, not the optimum.
    m = 4
    total = m if seasonal == "mul" else 0.0
    scale = 0.05 if seasonal == "mul" else 1.0
    rng = np.random.default_rng(8551)
    model = ETSModel(
        austourists, error="add", trend="add", seasonal=seasonal, seasonal_periods=m
    )
    start = model.start_params.copy()
    start[3:] += scale * rng.standard_normal(len(start) - 3)
    full = start.copy()
    fixed_params = {}
    if fixed is not None:
        # a fixed value that is far from the other initial seasonals
        idx = model.param_names.index(f"initial_seasonal.{fixed}")
        full[idx] += 10 * scale
        fixed_params[f"initial_seasonal.{fixed}"] = full[idx]
    full[-1] = total - np.sum(full[-m:-1])

    # build the arguments as `fit` does
    with model.fix_params(fixed_params):
        assert model._derive_last_initial_seasonal
        internal_start = model._convert_and_bound_start_params(start)
        bounds = model._setup_bounds()
        use_beta_star = "smoothing_trend" not in model.bounds
        use_gamma_star = "smoothing_seasonal" not in model.bounds
    internal_start[1] /= internal_start[0]
    internal_start[2] /= 1 - internal_start[0]
    is_fixed = np.zeros(model._k_params_internal, dtype=np.int64)
    fixed_values = np.empty_like(internal_start)
    free_params = []
    for i, (lower, upper) in enumerate(bounds):
        if lower == upper:
            is_fixed[i] = True
            fixed_values[i] = lower
        elif i == model._k_params_internal - 1:
            is_fixed[i] = True
            fixed_values[i] = model._last_initial_seasonal(internal_start[6:-1])
        else:
            free_params.append(internal_start[i])
    expected_fixed = np.zeros(m, dtype=np.int64)
    expected_fixed[-1] = 1
    if fixed is not None:
        expected_fixed[fixed] = 1
    assert_array_equal(is_fixed[6:], expected_fixed)
    # In a fit, the optimizer evaluates the objective at other points than
    # the start, where the last value is stale. It must be recomputed in
    # every evaluation.
    fixed_values[-1] = np.nan

    yhat = np.zeros(model.nobs)
    xhat = np.zeros((model.nobs, model._k_states_internal))
    llf = model._loglike_internal(
        np.array(free_params),
        yhat,
        xhat,
        is_fixed,
        fixed_values,
        use_beta_star,
        use_gamma_star,
        True,
    )
    assert_allclose(llf, model.loglike(full), rtol=1e-10)
    # the last initial seasonal of the objective is the derived one
    assert_allclose(fixed_values[-1], full[-1], rtol=1e-12)


@pytest.mark.parametrize("name", ["AAdA", "ANA", "MNA", "MNM"])
def test_estimated_initial_seasonal_cov(
    austourists, ets_austourists_fit_results_R, name
):
    # The Hessian of all m initial seasonals is singular (for additive
    # seasonality, a constant can be moved between the level and the
    # seasonals). The covariance is computed for the m - 1 free initial
    # seasonals and mapped to the derived one, s[-m] = total - sum(others).
    error, trend, damped, seasonal = {
        "AAdA": ("add", "add", True, "add"),
        "ANA": ("add", None, False, "add"),
        "MNA": ("mul", None, False, "add"),
        "MNM": ("mul", None, False, "mul"),
    }[name]
    results_R = ets_austourists_fit_results_R[damped][
        short_model_name(error, trend, seasonal)
    ]
    params = np.array(get_params_from_R(results_R))
    model = ETSModel(
        austourists,
        error=error,
        trend=trend,
        damped_trend=damped,
        seasonal=seasonal,
        seasonal_periods=4,
    )
    res = model.smooth(params, param_index="integer")
    cov = np.asarray(res.cov_params())
    k = len(params)
    jac = np.eye(k)[:, :-1]
    jac[-1, -3:] = -1
    total = params[-1] + params[-4:-1].sum()

    def loglike_free(free_params, *args, **kwargs):
        full = jac @ free_params
        full[-1] += total
        return model.loglike(full)

    hessian = approx_hess_cs(params[:-1], loglike_free)
    expected = jac @ np.linalg.inv(-hessian) @ jac.T
    assert_allclose(cov, expected, rtol=1e-5, atol=1e-10)
    # the derived seasonal moves with the others, so the sum of the
    # initial seasonals has no variance
    assert_allclose(cov[:, -4:].sum(1), 0, atol=1e-10 * np.abs(cov).max())
    # the initial level and seasonals have finite standard errors
    idx = [model.param_names.index("initial_level")] + list(range(k - 4, k))
    bse = np.asarray(res.bse)[idx]
    assert np.all(np.isfinite(bse))
    assert np.all(bse > 0)


@pytest.mark.parametrize("dtype", [np.float32, ">f8", "<f8"])
def test_params_dtype(austourists_model, dtype):
    # GH 10770: params that are not native float64 failed in the smoother
    model = austourists_model
    params = model._start_params.astype(dtype)
    native = params.astype(np.float64)
    # the reference has to be a valid fit, or NaN would compare equal to NaN
    llf = model.loglike(native)
    fitted = model.smooth(native, param_index="integer").fittedvalues
    assert np.isfinite(llf)
    assert np.isfinite(fitted).all()
    assert_allclose(model.loglike(params), llf, rtol=1e-12)
    assert_allclose(
        model.smooth(params, param_index="integer").fittedvalues, fitted, rtol=1e-12
    )


def test_params_dtype_int(austourists_model):
    # GH 10770: integer params were rejected. Integer values cannot be valid
    # smoothing parameters, which have to be strictly between 0 and 1, so this
    # only checks that they are accepted and give what float64 gives. The
    # smoother uses beta / alpha and gamma / (1 - alpha), which are not finite
    # for alpha = 0 or 1 (rounding the start params gives alpha = 0, so every
    # value would be NaN and the complex-step Hessian of smooth would warn),
    # so alpha = 2: outside the usual bounds, but the output is finite.
    model = austourists_model
    params = np.array([2, 0, 0, 1, 26, 1, 2, -1, -7, 6], dtype=np.int64)
    native = params.astype(np.float64)
    llf = model.loglike(native)
    fitted = model.smooth(native, param_index="integer").fittedvalues
    assert np.isfinite(llf)
    assert np.isfinite(fitted).all()
    assert_allclose(model.loglike(params), llf, rtol=1e-12)
    assert_allclose(
        model.smooth(params, param_index="integer").fittedvalues, fitted, rtol=1e-12
    )


@pytest.mark.parametrize("nsimulations", [0, np.int64(0), -1])
@pytest.mark.parametrize("anchor", [None, "end"])
def test_simulate_nsimulations_at_least_one(oildata, nsimulations, anchor):
    # simulate(0) wrote the initial states outside of a zero-row array in
    # the compiled smoother; a negative number raised from np.zeros
    res = ETSModel(oildata).smooth([0.3, 110.0], param_index="integer")
    with pytest.raises(ValueError, match="nsimulations must be at least 1"):
        res.simulate(nsimulations, anchor=anchor, rng=np.random.default_rng(0))


@pytest.mark.parametrize("steps", [0, np.int64(0), -2])
def test_forecast_steps_at_least_one(oildata, steps):
    # forecast(0) simulated zero steps, which wrote outside of an array
    res = ETSModel(oildata).smooth([0.3, 110.0], param_index="integer")
    with pytest.raises(ValueError, match="steps must be at least 1"):
        res.forecast(steps)
    # one step is the smallest valid horizon: the level after the last
    # observation, from the simple exponential smoothing recursion
    level = 110.0
    for value in oildata:
        level = 0.3 * value + 0.7 * level
    assert_allclose(res.forecast(1).iloc[0], level, rtol=1e-12)


@pytest.mark.parametrize("initialization_method", ["known", "estimated", "heuristic"])
def test_no_observations(initialization_method):
    # A model without observations could be created with known initial
    # values, and smoothing it wrote the states of the last observation out
    # of bounds. Only the constructor is called here.
    kwargs = {"initial_level": 1.0} if initialization_method == "known" else {}
    with pytest.raises(ValueError, match="at least one observation"):
        ETSModel(np.array([]), initialization_method=initialization_method, **kwargs)
#
# CONTRACTS OF THE PUBLIC API
#

# The tests below compare with closed forms, identities and R at these
# smoothing parameters, not at estimated ones
FIXED_SMOOTHING = {
    "smoothing_level": 0.3,
    "smoothing_trend": 0.1,
    "smoothing_seasonal": 0.2,
    "damping_trend": 0.9,
}


def fixed_params(model):
    # the start parameters of the model with the smoothing parameters above
    params = np.array(model.start_params, dtype=float)
    for i, name in enumerate(model.param_names):
        if name in FIXED_SMOOTHING:
            params[i] = FIXED_SMOOTHING[name]
    return params


def ljung_box(x, lags, boxpierce=False):
    # Ljung-Box (or Box-Pierce) statistics of lags 1, ..., lags and their
    # chi2(lag) p-values
    x = x - x.mean()
    n = x.shape[0]
    k = np.arange(1, lags + 1)
    acf = np.array([x[j:] @ x[:-j] for j in k]) / (x @ x)
    if boxpierce:
        stat = n * np.cumsum(acf**2)
    else:
        stat = n * (n + 2) * np.cumsum(acf**2 / (n - k))
    return np.vstack([stat, scipy.stats.chi2.sf(stat, k)])


def breakvar(x, h):
    # the sum of squares of the last h values over that of the first h, and
    # its two-sided F(h, h) p-value
    stat = np.sum(x[-h:] ** 2) / np.sum(x[:h] ** 2)
    dist = scipy.stats.f(h, h)
    return np.array([stat, 2 * min(dist.cdf(stat), dist.sf(stat))])


def jarque_bera(x):
    # the Jarque-Bera statistic, its chi2(2) p-value, skewness and kurtosis
    dev = x - x.mean()
    m2, m3, m4 = (np.mean(dev**p) for p in (2, 3, 4))
    skew = m3 / m2**1.5
    kurtosis = m4 / m2**2
    stat = x.shape[0] / 6 * (skew**2 + (kurtosis - 3) ** 2 / 4)
    return np.array([stat, scipy.stats.chi2.sf(stat, 2), skew, kurtosis])


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"error": "mul"}, "strictly positive"),
        ({"trend": "mul"}, "strictly positive"),
        ({"seasonal": "mul", "seasonal_periods": 4}, "strictly positive"),
        ({"damped_trend": True}, "Can only dampen"),
        ({"seasonal": "add", "seasonal_periods": 1}, "larger than 1"),
        ({"error": "none"}, "error must be one of"),
        ({"trend": "damped"}, "trend must be one of"),
        ({"seasonal": "seasonal"}, "seasonal must be one of"),
        ({"initialization_method": "simple"}, "initialization_method must be"),
        ({"initialization_method": "known"}, "`initial_level` argument"),
        (
            {"initialization_method": "known", "initial_level": 30.0, "trend": "add"},
            "`initial_trend` argument",
        ),
        (
            {
                "initialization_method": "known",
                "initial_level": 30.0,
                "seasonal": "add",
                "seasonal_periods": 4,
            },
            "`initial_seasonal` argument",
        ),
        ({"bounds": [(0.1, 0.5)]}, "bounds must be a dictionary"),
        ({"bounds": {"smoothing_trend": (0.1, 0.5)}}, "Invalid key: smoothing_trend"),
        ({"bounds": {"smoothing_level": (0.1, 0.2, 0.5)}}, r"bounds\[smoothing_level\]"),
    ],
)
def test_invalid_arguments_raise(austourists, kwargs, match):
    # a zero observation is only invalid with a multiplicative component
    endog = austourists.copy()
    endog.iloc[10] = 0.0
    with pytest.raises(ValueError, match=match):
        ETSModel(endog, **kwargs)


def test_endog_two_columns_raises(austourists):
    with pytest.raises(ValueError, match="endog must be 1-dimensional"):
        ETSModel(np.column_stack([austourists, austourists]))


def endog_form(values, form):
    if form in ("list", "tuple", "series"):
        return series_forms(values)[form]
    elif form == "int":
        return values.astype(np.int64)
    elif form == "period":
        index = pd.period_range("1999Q1", periods=values.shape[0], freq="Q")
        return pd.Series(values, index=index)
    return pd.Series(values)


def extended_index(index, steps):
    # the index of the `steps` periods after the end of `index`
    n = len(index)
    if isinstance(index, pd.PeriodIndex):
        return pd.period_range(index[0], periods=n + steps, freq=index.freq)[n:]
    elif isinstance(index, pd.DatetimeIndex):
        return pd.date_range(index[0], periods=n + steps, freq=index.freq)[n:]
    return pd.RangeIndex(n, n + steps)


@pytest.mark.parametrize("form", ["list", "tuple", "series", "int", "period", "range"])
def test_endog_forms(austourists, form):
    # endog is documented as array_like: each form gives the results of the
    # float ndarray, and pandas input gives pandas output indexed by the
    # predicted periods. An (n, 1) column and a one-column DataFrame are
    # rejected as not 1-dimensional, so they are not tested here.
    values = np.round(austourists.to_numpy())  # integer values, for "int"
    n = values.shape[0]
    kwargs = dict(error="add", trend="add", seasonal="add", seasonal_periods=4)
    ref_model = ETSModel(values, **kwargs)
    params = fixed_params(ref_model)
    ref = ref_model.smooth(params)
    endog = endog_form(values, form)
    res = ETSModel(endog, **kwargs).smooth(params, param_index="integer")
    errors = np.array([[0.5], [-1.0], [2.0]])
    outputs = {
        "fittedvalues": (res.fittedvalues, ref.fittedvalues),
        "forecast": (res.forecast(6), ref.forecast(6)),
        "predict": (
            res.predict(start=n - 2, end=n + 3),
            ref.predict(start=n - 2, end=n + 3),
        ),
        "simulate": (
            res.simulate(3, anchor="end", random_errors=errors),
            ref.simulate(3, anchor="end", random_errors=errors),
        ),
    }

    assert_allclose(res.llf, ref.llf, rtol=1e-12)
    for actual, expected in outputs.values():
        assert_allclose(np.asarray(actual), expected, rtol=1e-12)
    if not isinstance(endog, pd.Series):
        for actual, _ in outputs.values():
            assert isinstance(actual, np.ndarray)
    else:
        index = endog.index
        future = extended_index(index, 6)
        expected_index = {
            "fittedvalues": index,
            "forecast": future,
            "predict": index[n - 2 :].append(future[:4]),
            "simulate": future[:3],
        }
        for key, (actual, _) in outputs.items():
            assert isinstance(actual, pd.Series)
            assert actual.index.equals(expected_index[key])
        frame = res.get_prediction(start=n - 2, end=n + 3).summary_frame()
        expected = ETSModel(pd.Series(values), **kwargs).smooth(
            params, param_index="integer"
        )
        expected = expected.get_prediction(start=n - 2, end=n + 3).summary_frame()
        assert_allclose(frame, expected, rtol=1e-12)
        assert frame.index.equals(expected_index["predict"])


@pytest.mark.parametrize(
    "error, trend, damped, seasonal, form",
    [
        ("add", None, False, None, None),
        ("add", "add", False, None, None),
        ("add", None, False, "add", list),
        ("add", "add", True, "add", tuple),
        ("add", "add", False, "add", np.asarray),
        ("mul", "mul", False, "mul", pd.Series),
    ],
)
def test_known_initialization(austourists, error, trend, damped, seasonal, form):
    # With initialization_method="known", the results are those of the model
    # with estimated initial states at these states. The parameter vector
    # holds the initial seasonals as s[-1], ..., s[-m], while
    # initial_seasonal starts with s[-m] (GH 7893).
    kwargs = dict(
        error=error,
        trend=trend,
        damped_trend=damped,
        seasonal=seasonal,
        seasonal_periods=4 if seasonal else None,
    )
    estimated = ETSModel(austourists, **kwargs)
    params = fixed_params(estimated)
    names = estimated.param_names
    initial_seasonal = params[-4:][::-1]
    known_kwargs = dict(
        initial_level=params[names.index("initial_level")],
        initial_trend=params[names.index("initial_trend")] if trend else None,
        initial_seasonal=form(initial_seasonal.tolist()) if seasonal else None,
    )
    known = ETSModel(
        austourists, initialization_method="known", **known_kwargs, **kwargs
    )
    k = known.k_params
    assert known.param_names == names[:k]
    assert known.initial_state_names == names[k:]

    res = known.smooth(params[:k], param_index="integer")
    expected = estimated.smooth(params, param_index="integer")
    assert_allclose(res.llf, expected.llf, rtol=1e-12)
    assert_allclose(res.fittedvalues, expected.fittedvalues, rtol=1e-12)
    assert_allclose(res.states, expected.states, rtol=1e-12)
    assert_allclose(res.initial_state, expected.initial_state, rtol=1e-12)
    if seasonal:
        assert_allclose(res.initial_seasonal, initial_seasonal, rtol=1e-12)

    # set_initialization_method changes an existing model into the same one
    switched = ETSModel(austourists, **kwargs)
    switched.set_initialization_method("known", **known_kwargs)
    assert switched.param_names == known.param_names
    assert_allclose(
        switched.smooth(params[:k], param_index="integer").fittedvalues,
        expected.fittedvalues,
        rtol=1e-12,
    )

    # the summary has a table of the initial states
    text = res.summary().as_text()
    assert "initialization method: known" in text
    for name in known.initial_state_names:
        assert name in text


def test_fit_return_params(oildata):
    model = ETSModel(oildata, error="add", trend="add")
    params = model.fit(disp=False, return_params=True)
    assert isinstance(params, np.ndarray)
    # the parameters of the same optimization as a fit that returns results
    assert_allclose(
        params, model.fit(disp=False, param_index="integer").params, rtol=1e-12
    )

    # with all parameters fixed, the fixed values in the order of param_names
    # whatever the order of the dictionary
    values = [0.3, 0.1, 110.0, 5.0]
    fixed = dict(zip(model.param_names, values, strict=True))
    with model.fix_params(dict(reversed(fixed.items()))):
        params = model.fit(disp=False, return_params=True)
        res = model.fit(disp=False, param_index="integer")
    assert_allclose(params, values, rtol=1e-12)
    assert_allclose(res.params, values, rtol=1e-12)
    assert res.fixed_params == model.param_names
    assert res.mle_retvals is None


@pytest.mark.parametrize("trend", ["add", "mul"])
def test_estimated_initialization_short_nonseasonal(austourists, trend):
    # With fewer than 10 observations, the start values of the initial level
    # and trend are the first observation and the first difference or ratio
    # (Hyndman and Athanasopoulos, Section 7.6)
    y = austourists.to_numpy()[:8]
    model = ETSModel(y, trend=trend)
    slope = y[1] - y[0] if trend == "add" else y[1] / y[0]
    assert_allclose(model.start_params[-2:], [y[0], slope], rtol=1e-12)


@pytest.mark.parametrize(
    "nobs, seasonal_periods, method, match",
    [
        (7, 4, "estimated", "two full seasonal cycles"),
        (9, None, "heuristic", "less than 10 observations"),
        (15, 8, "heuristic", "two full seasonal cycles"),
        (12, 4, "heuristic", r"10 \+ 2 \* \(seasonal_periods // 2\)"),
    ],
)
def test_initialization_too_few_observations(
    austourists, nobs, seasonal_periods, method, match
):
    with pytest.raises(ValueError, match=match):
        ETSModel(
            austourists.iloc[:nobs],
            seasonal="add" if seasonal_periods else None,
            seasonal_periods=seasonal_periods,
            initialization_method=method,
        )


@pytest.mark.parametrize(
    "random_errors, match",
    [
        (np.zeros((4, 2)), "must have shape"),
        (np.zeros(4), "must have shape"),
        ("normal", "unexpected value"),
        ([0.0, 0.0, 0.0, 0.0], "unexpected value"),
    ],
)
def test_simulate_invalid_random_errors(austourists_model, random_errors, match):
    res = austourists_model.smooth(
        fixed_params(austourists_model), param_index="integer"
    )
    with pytest.raises(ValueError, match=match):
        res.simulate(4, random_errors=random_errors)


def test_simulate_anchor(austourists_model):
    res = austourists_model.smooth(
        fixed_params(austourists_model), param_index="integer"
    )
    fitted = res.fittedvalues
    zero = np.zeros((1, 1))
    # the simulation starts after the anchor: without errors, one step from
    # anchor t is the one-step prediction of observation t + 1
    for t in [0, 30, res.nobs - 2]:
        sim = res.simulate(1, anchor=t, random_errors=zero)
        assert_allclose(sim, fitted.iloc[t + 1 : t + 2], rtol=1e-12)
        assert sim.index.equals(fitted.index[t + 1 : t + 2])
    sim = res.simulate(1, anchor="start", random_errors=zero)
    assert_allclose(sim, fitted.iloc[:1], rtol=1e-12)

    # a partial date anchors at its first period, here 2005-03-31
    zeros = np.zeros((4, 1))
    sim = res.simulate(4, anchor="2005", random_errors=zeros)
    expected = res.simulate(4, anchor="2005-03-31", random_errors=zeros)
    assert_allclose(sim, expected, rtol=1e-12)
    assert sim.index[0] == pd.Timestamp("2005-06-30")
    assert_allclose(sim.iloc[0], fitted["2005-06-30"], rtol=1e-12)

    # the last anchor in the sample, nobs - 1, is "end"
    with pytest.raises(ValueError, match="outside of the sample"):
        res.simulate(1, anchor=res.nobs)


def test_predict_dynamic_date(austourists, austourists_model):
    # dynamic can be a date, which is not an offset from start: before it the
    # predictions are one-step predictions, from it on they are forecasts
    # from the period before it
    res = austourists_model.smooth(
        fixed_params(austourists_model), param_index="integer"
    )
    start, dynamic, end = 40, 48, 70
    expected = res.predict(start=start, end=end, dynamic=dynamic - start)
    assert_allclose(
        expected.iloc[: dynamic - start],
        res.fittedvalues.iloc[start:dynamic],
        rtol=1e-12,
    )
    steps = end - dynamic + 1
    forecast = res.simulate(
        steps, anchor=dynamic - 1, random_errors=np.zeros((steps, 1))
    )
    assert_allclose(expected.iloc[dynamic - start :], forecast, rtol=1e-12)
    date = austourists.index[dynamic]
    for key in [str(date.date()), date, date.to_pydatetime()]:
        pred = res.predict(start=start, end=end, dynamic=key)
        assert_allclose(pred, expected, rtol=1e-12)


@pytest.mark.parametrize("start, dynamic", [(10, False), (10, 5), (0, 5)])
def test_get_prediction_simulated_ann(austourists, start, dynamic):
    # For the ANN model with additive errors e, y[t] = l[t-1] + e[t] and
    # l[t] = l[t-1] + alpha e[t]. The non-dynamic part is simulated one step
    # at a time from the smoothed level (the initial level for t = 0), the
    # dynamic part from the level of the period before it. The errors are
    # sigma times the standard normal draws from rng, one row per call of
    # simulate.
    smoothing_level, reps, seed = 0.3, 6, 1234
    res = ETSModel(austourists).smooth([smoothing_level, 30.0], param_index="integer")
    end = 20
    npred = end - start + 1
    nsmooth = npred if dynamic is False else dynamic
    pred = res.get_prediction(
        start=start,
        end=end,
        dynamic=dynamic,
        method="simulated",
        simulate_repetitions=reps,
        rng=np.random.default_rng(seed),
    )

    errors = np.random.default_rng(seed).standard_normal((npred, reps))
    errors *= np.sqrt(res.scale)
    fitted = res.fittedvalues.to_numpy()[start:]
    sims = fitted[:nsmooth, None] + errors[:nsmooth]
    mean = fitted[:npred].copy()
    if nsmooth < npred:
        level = res.level.iloc[start + nsmooth - 1]
        dyn_errors = errors[nsmooth:]
        past = np.cumsum(dyn_errors, axis=0) - dyn_errors
        sims = np.vstack([sims, level + smoothing_level * past + dyn_errors])
        mean[nsmooth:] = level

    frame = pred.summary_frame(alpha=0.1)
    assert frame.index.equals(austourists.index[start : end + 1])
    assert_allclose(frame["mean"], mean, rtol=1e-12)
    assert_allclose(frame["mean_numerical"], sims.mean(1), rtol=1e-10)
    lower_upper = np.quantile(sims, [0.05, 0.95], axis=1).T
    assert_allclose(frame[["pi_lower", "pi_upper"]], lower_upper, rtol=1e-10)
    assert_allclose(pred.var_pred_mean, sims.var(1, ddof=1), rtol=1e-10)


EXACT_MODELS = {
    "ANN": (None, False, None),
    "AAN": ("add", False, None),
    "AAdN": ("add", True, None),
    "ANA": (None, False, "add"),
    "AAA": ("add", False, "add"),
    "AAdA": ("add", True, "add"),
}

# Relative variance of the h-step forecast error, h = 1, ..., 10, of the
# models with additive errors, at FIXED_SMOOTHING and m = 4. Computed by
# forecast:::class1 of R's forecast package 9.0.2 (R 4.6.2) with sigma2 = 1,
# as 1 + sum_j c_j^2 with c_j = H F^(j-1) G from the state space matrices,
# not from the closed forms. class1 puts gamma into the third state whatever
# the trend, which is the newest seasonal only in the models with a trend, so
# the values of ANA are those of the equivalent AAA model with beta = 0.
R_RELATIVE_FORECAST_VARIANCE = {
    "ANN": [1, 1.09, 1.18, 1.27, 1.36, 1.45, 1.54, 1.63, 1.72, 1.81],
    "AAN": [1, 1.16, 1.41, 1.77, 2.26, 2.9, 3.71, 4.71, 5.92, 7.36],
    "AAdN": [
        1, 1.1521, 1.373941, 1.66976821, 2.0412706501, 2.4882417865810003,
        3.0090971511306104, 3.6012778660157942, 4.2615633277127936,
        4.9863117260633629,
    ],
    "ANA": [1, 1.09, 1.18, 1.27, 1.52, 1.61, 1.7, 1.79, 2.04, 2.13],
    "AAA": [1, 1.16, 1.41, 1.77, 2.58, 3.22, 4.03, 5.03, 6.72, 8.16],
    "AAdA": [
        1, 1.1521, 1.373941, 1.66976821, 2.3250746500999999, 2.7720457865810002,
        3.2929011511306103, 3.8850818660157942, 4.9103991321127936,
        5.6351475304633629,
    ],
}  # fmt: skip


@pytest.mark.parametrize("name", list(EXACT_MODELS))
def test_get_prediction_exact(austourists, name):
    # The prediction error variance of the exact method is mse times the
    # relative variance: 1 for the one-step (in-sample) predictions, and that
    # of R's forecast package for the h-step forecasts
    trend, damped, seasonal = EXACT_MODELS[name]
    model = ETSModel(
        austourists,
        trend=trend,
        damped_trend=damped,
        seasonal=seasonal,
        seasonal_periods=4 if seasonal else None,
    )
    res = model.smooth(fixed_params(model), param_index="integer")
    n = res.nobs
    q = scipy.stats.norm.ppf(0.95)

    pred = res.get_prediction(start=10, end=20)
    assert pred.method == "exact"
    assert_allclose(pred.var_pred_mean, np.full(11, res.mse), rtol=1e-12)
    fitted = res.fittedvalues.iloc[10:21]
    pred_int = pred.pred_int(alpha=0.1)
    assert pred_int.index.equals(fitted.index)
    half_width = q * np.sqrt(res.mse)
    assert_allclose(pred_int.iloc[:, 0], fitted - half_width, rtol=1e-12)
    assert_allclose(pred_int.iloc[:, 1], fitted + half_width, rtol=1e-12)

    pred = res.get_prediction(start=n, end=n + 9)
    relative = pred.var_pred_mean / res.mse
    assert_allclose(relative, R_RELATIVE_FORECAST_VARIANCE[name], rtol=1e-12)
    frame = pred.summary_frame(alpha=0.1)
    half_width = q * np.sqrt(pred.var_pred_mean)
    assert_allclose(frame["pi_upper"] - frame["mean"], half_width, rtol=1e-10)
    assert_allclose(frame["mean"] - frame["pi_lower"], half_width, rtol=1e-10)


def test_forecast_closed_form(setup_model):
    # The point forecasts of Hyndman et al. (2008), Table 2.1, from the last
    # states: l + phi_h b (additive trend) or l b^phi_h (multiplicative), with
    # phi_h = phi + ... + phi^h (h without damping), and the seasonal of the
    # same season in the last cycle, s[n + h - m (k + 1)] with
    # k = (h - 1) // m, added or multiplied.
    model_class, model_args, model_kwargs, params, _ = setup_model
    res = model_class(*model_args, **model_kwargs).smooth(params, param_index="integer")
    h = np.arange(1, 11)
    level = res.level.iloc[-1]
    expected = np.full(h.shape, level)
    if res.has_trend:
        slope = res.slope.iloc[-1]
        phi = res.damping_trend if res.damped_trend else 1.0
        phi_h = np.cumsum(phi**h)
        if res.trend == "add":
            expected = level + phi_h * slope
        else:
            expected = level * slope**phi_h
    if res.has_seasonal:
        m = res.seasonal_periods
        season = res.season.to_numpy()[-m:][(h - 1) % m]
        if res.seasonal == "add":
            expected = expected + season
        else:
            expected = expected * season
    assert_allclose(res.forecast(10), expected, rtol=1e-12)


@pytest.mark.parametrize("error", ["add", "mul"])
def test_loglike_formula(austourists, error):
    # The log-likelihood of the docstring, with k = 1 (additive errors) or
    # k = yhat (multiplicative errors), at the one-step predictions yhat that
    # the tests against R check
    model = ETSModel(
        austourists, error=error, trend="add", seasonal="mul", seasonal_periods=4
    )
    params = fixed_params(model)
    res = model.smooth(params, param_index="integer")
    y = austourists.to_numpy()
    yhat = res.fittedvalues.to_numpy()
    k = yhat if error == "mul" else np.ones_like(y)
    n = y.shape[0]
    s2 = np.mean(((y - yhat) / k) ** 2)
    expected = -n / 2 * (np.log(2 * np.pi * s2) + 1) - np.sum(np.log(np.abs(k)))
    assert_allclose(model.loglike(params), expected, rtol=1e-12)
    assert_allclose(res.llf, expected, rtol=1e-12)
    assert_allclose(res.resid, (y - yhat) / k, rtol=1e-12)
    assert_allclose(res.scale, s2, rtol=1e-12)


def test_information_criteria(austourists):
    model = ETSModel(austourists, trend="add", seasonal="add", seasonal_periods=4)
    params = fixed_params(model)
    with model.fix_params({"smoothing_trend": 0.1}):
        res = model.smooth(params, param_index="integer")
    n = res.nobs
    llf = res.llf
    # the free parameters and the variance of the errors
    df = (len(params) - 1) + 1
    assert res.df_model == df
    assert res.df_resid == n - df
    assert_allclose(res.aic, -2 * llf + 2 * df, rtol=1e-12)
    assert_allclose(res.aicc, -2 * llf + 2 * df * n / (n - df - 1), rtol=1e-12)
    assert_allclose(res.bic, -2 * llf + np.log(n) * df, rtol=1e-12)
    assert_allclose(res.hqic, -2 * llf + 2 * np.log(np.log(n)) * df, rtol=1e-12)
    resid = austourists.to_numpy() - res.fittedvalues.to_numpy()
    assert_allclose(res.sse, np.sum(resid**2), rtol=1e-12)
    assert_allclose(res.mse, np.mean(resid**2), rtol=1e-12)
    assert_allclose(res.mae, np.mean(np.abs(resid)), rtol=1e-12)

    # the fixed parameter has no standard error, z-statistic or p-value
    bse = np.asarray(res.bse)
    zvalues = np.asarray(res.zvalues)
    pvalues = np.asarray(res.pvalues)
    assert np.isnan(bse[1])
    assert np.isnan(zvalues[1])
    assert np.isnan(pvalues[1])
    free = np.arange(len(params)) != 1
    assert_allclose(zvalues[free], params[free] / bse[free], rtol=1e-12)
    expected = 2 * scipy.stats.norm.sf(np.abs(zvalues[free]))
    assert_allclose(pvalues[free], expected, rtol=1e-12)


def test_diagnostics_vs_formulas(austourists_model):
    res = austourists_model.smooth(
        fixed_params(austourists_model), param_index="integer"
    )
    resid = np.asarray(res.resid)
    errors = np.asarray(res.standardized_forecasts_error)
    expected = (resid - resid.mean()) / resid.std(ddof=1)
    assert_allclose(errors, expected, rtol=1e-12)

    # method=None selects the only test; the default number of lags of a
    # seasonal model is min(2 m, nobs // 5) = 8
    lb = res.test_serial_correlation(None)
    assert lb.shape == (1, 2, 8)
    assert_allclose(lb[0], ljung_box(errors, 8), rtol=1e-10)
    bp = res.test_serial_correlation("boxpierce", lags=5)
    assert_allclose(bp[0], ljung_box(errors, 5, boxpierce=True), rtol=1e-10)
    # a list of lags reports only these lags
    lb = res.test_serial_correlation("ljungbox", lags=[2, 5])
    assert_allclose(lb[0], ljung_box(errors, 5)[:, [1, 4]], rtol=1e-10)

    # h = round(68 / 3) = 23
    het = res.test_heteroskedasticity(None)
    assert_allclose(het[0], breakvar(errors, 23), rtol=1e-10)
    # use_f=False refers h H(h) to chi2(h)
    het = res.test_heteroskedasticity("breakvar", use_f=False)
    stat = het[0, 0] * 23
    expected = 2 * min(scipy.stats.chi2.cdf(stat, 23), scipy.stats.chi2.sf(stat, 23))
    assert_allclose(het[0, 1], expected, rtol=1e-10)
    assert_allclose(res.test_normality(None)[0], jarque_bera(errors), rtol=1e-10)


def test_summary_sample(austourists):
    # The sample is shown as dates for a date index, else as observation
    # numbers, and start changes its first period
    res = ETSModel(austourists).smooth([0.3, 30.0], param_index="integer")
    text = res.summary(start=4).as_text()
    assert re.search(r"Sample:\s+03-31-2000\s", text)
    assert "- 12-31-2015" in text
    res = ETSModel(austourists.to_numpy()).smooth([0.3, 30.0])
    text = res.summary().as_text()
    assert re.search(r"Sample:\s+0\s", text)
    assert "- 68" in text


def test_dates_freq_missing(austourists):
    values = austourists.to_numpy()
    params = [0.3, 30.0]
    ref = ETSModel(austourists).smooth(params, param_index="integer")
    # with dates, predictions can use date strings, but they are ndarrays
    res = ETSModel(values, dates=austourists.index).smooth(params)
    pred = res.predict(start="2014-03-31", end="2016-12-31")
    expected = ref.predict(start="2014-03-31", end="2016-12-31")
    assert isinstance(pred, np.ndarray)
    assert_allclose(pred, expected, rtol=1e-12)

    # freq is the frequency of an index without one
    index = pd.DatetimeIndex(austourists.index.tolist())
    assert index.freq is None
    res = ETSModel(pd.Series(values, index=index), freq=QUARTER_END)
    assert (
        res.smooth(params, param_index="integer")
        .forecast(4)
        .index.equals(ref.forecast(4).index)
    )

    # "drop" drops the observations with NaN, "raise" raises
    with_nan = values.copy()
    with_nan[5] = np.nan
    res = ETSModel(with_nan, missing="drop").smooth(params)
    expected = ETSModel(np.delete(values, 5)).smooth(params)
    assert_allclose(res.fittedvalues, expected.fittedvalues, rtol=1e-12)
    assert_allclose(res.llf, expected.llf, rtol=1e-12)
    with pytest.raises(MissingDataError):
        ETSModel(with_nan, missing="raise")


def test_hessian_scaled_by_nobs(austourists):
    # ETSModel.hessian is the Hessian of loglike divided by nobs, here checked
    # against central second differences of loglike for the ANN model
    model = ETSModel(austourists)
    params = np.array([0.3, 30.0])
    steps = np.diag([1e-4, 1e-2])
    expected = np.empty((2, 2))
    for i, j in product(range(2), range(2)):
        d_i, d_j = steps[i], steps[j]
        expected[i, j] = (
            model.loglike(params + d_i + d_j)
            - model.loglike(params + d_i - d_j)
            - model.loglike(params - d_i + d_j)
            + model.loglike(params - d_i - d_j)
        ) / (4 * steps[i, i] * steps[j, j] * model.nobs)
    assert_allclose(model.hessian(params), expected, rtol=1e-5)
    # The finite difference approximations are numerical differences of
    # numerical scores, first order accurate (forward) or second order
    # (centered). Their error is relative to the scale of the whole matrix,
    # not to each element, hence atol. The forward error depends on the
    # platform (2% of the scale on Linux with Python 3.13), so it is only
    # checked to the scale of the matrix, which catches a missing division
    # by nobs (a factor of 68 here).
    scale = np.abs(expected).max()
    with pytest.warns(PrecisionWarning, match="Calculation of the Hessian"):
        centered = model.hessian(
            params, approx_complex_step=False, approx_centered=True
        )
    assert_allclose(centered, expected, rtol=0, atol=0.01 * scale)
    with pytest.warns(PrecisionWarning, match="Calculation of the Hessian"):
        forward = model.hessian(params, approx_complex_step=False)
    assert_allclose(forward, expected, rtol=0, atol=0.1 * scale)


#
# THE PUBLIC BASE CLASSES
#


class MeanModel(StateSpaceMLEModel):
    """
    Normal observations with an unknown mean, a minimal subclass of the
    public base class StateSpaceMLEModel

    The variance is concentrated out of the log-likelihood. The second
    parameter does not enter it, so that the Hessian is exactly singular. The
    parameter names are the defaults of the base class.
    """

    def __init__(self, endog, **kwargs):
        super().__init__(endog, **kwargs)
        # StateSpaceMLEResults labels the parameters with these
        self.data.param_names = self.param_names

    @staticmethod
    def prepare_data(data):
        return np.asarray(data.orig_endog, dtype=float), None

    @property
    def _start_params(self):
        return np.zeros(2)

    @property
    def nobs_effective(self):
        return self.nobs

    @property
    def k_endog(self):
        return 1

    def loglike(self, params, **kwargs):
        resid = self.endog - params[0]
        return -self.nobs / 2 * (np.log(2 * np.pi * np.mean(resid**2)) + 1)

    def hessian(self, params, **kwargs):
        return self._hessian_complex_step(params, **kwargs)

    def update(self, params, **kwargs):
        pass

    def fit(self, start_params=None, **kwargs):
        # the maximum likelihood estimate of the mean; the second parameter
        # does not enter the likelihood and is set to 1
        params = np.array([np.mean(self.endog), 1.0])
        if self._has_fixed_params:
            params[self._fixed_params_index] = list(self._fixed_params.values())
        return MeanResults(self, params)


class MeanResults(StateSpaceMLEResults):
    """Results of MeanModel, with the covariance computed as by ETSResults"""

    def __init__(self, model, params):
        resid = model.endog - params[0]
        super().__init__(model, params, scale=np.mean(resid**2))
        self._resid = resid
        self.df_model = self.k_params - len(self.fixed_params) + 1
        self.standardized_forecasts_error = (resid - resid.mean()) / resid.std(
            ddof=1
        )
        self._cov_approx_complex_step = True
        self._cov_approx_centered = False
        self.cov_params_default = self.cov_params_approx

    @cache_readonly
    def nobs_effective(self):
        return self.nobs

    @cache_readonly
    def llf(self):
        return self.model.loglike(self.params)

    @cache_readonly
    def resid(self):
        return self._resid


class EmptyModel(StateSpaceMLEModel):
    """A subclass of StateSpaceMLEModel without parameters"""

    @staticmethod
    def prepare_data(data):
        return np.asarray(data.orig_endog, dtype=float), None


@pytest.fixture
def mean_data():
    return 10 + np.random.default_rng(10583).standard_normal(60)


def test_base_model_param_names_and_k_params(mean_data):
    # Without _param_names, the parameters are named after start_params, and
    # k_params counts them
    model = MeanModel(mean_data)
    assert model.param_names == ["param.0", "param.1"]
    assert model.k_params == 2
    model = EmptyModel(mean_data)
    assert model.param_names == []
    assert model.k_params == 0
    with pytest.raises(NotImplementedError):
        _ = model.start_params


def test_base_results_covariance_and_summary(mean_data):
    # The variance of the estimated mean is sigma2 / nobs. The second
    # parameter does not enter the likelihood, so its row of the Hessian is
    # exactly zero, the covariance matrix is singular and the summary says so.
    # Its standard error is zero, so its z-statistic divides by zero.
    model = MeanModel(mean_data)
    res = model.fit()
    n = mean_data.shape[0]
    sigma2 = np.mean((mean_data - mean_data.mean()) ** 2)
    assert_allclose(res.cov_params(), [[sigma2 / n, 0.0], [0.0, 0.0]], rtol=1e-8)
    assert res.df_resid == n - 3
    with np.errstate(divide="ignore", invalid="ignore"):
        text = res.summary().as_text()
    assert "Statespace Model Results" in text
    assert re.search(r"Model:\s+MeanModel\s", text)
    assert "Covariance Type" not in text
    assert "param.1" in text
    assert "Covariance matrix is singular or near-singular" in text

    # with the mean fixed, the free parameter has a zero Hessian
    res = model.fit_constrained({"param.0": 10.0})
    assert res.param_names == ["param.0 (fixed)", "param.1"]
    cov = res.cov_params()
    assert np.isnan(cov[0]).all()
    assert np.isnan(cov[:, 0]).all()
    with np.errstate(divide="ignore", invalid="ignore"):
        summary = res.summary(
            title="Mean model", model_name=["Mean", "Unused"], display_params=False
        )
    text = summary.as_text()
    assert "Mean model" in text
    assert re.search(r"Model:\s+Mean\s", text)
    assert "+ Unused" in text
    assert "param.1" not in text
    assert "Covariance matrix is singular or near-singular" in text

    # with the second parameter fixed, the covariance matrix of the free
    # parameter is not singular and the summary has no warnings
    res = model.fit_constrained({"param.1": 1.0})
    assert_allclose(res.cov_params()[0, 0], sigma2 / n, rtol=1e-8)
    text = res.summary().as_text()
    assert "param.1 (fixed)" in text
    assert "Warnings" not in text


def test_base_results_without_standardized_errors(mean_data):
    res = MeanModel(mean_data).fit()
    res.standardized_forecasts_error = None
    tests = [
        (res.test_serial_correlation, "ljungbox"),
        (res.test_heteroskedasticity, "breakvar"),
        (res.test_normality, "jarquebera"),
    ]
    for test, method in tests:
        with pytest.raises(ValueError, match="standardized forecast errors"):
            test(method)
    # the summary shows the diagnostics as nan
    with np.errstate(divide="ignore", invalid="ignore"):
        text = res.summary().as_text()
    for label in ["Ljung-Box (Q):", "Prob(H) (two-sided):", "Jarque-Bera (JB):"]:
        assert re.search(re.escape(label) + r"\s+nan", text)


def test_base_results_heteroskedasticity_2d(mean_data):
    # standardized errors of shape (k_endog, nobs); h = 60 / 3 = 20
    res = MeanModel(mean_data).fit()
    errors = np.asarray(res.standardized_forecasts_error)
    expected = breakvar(errors, 20)
    assert_allclose(res.test_heteroskedasticity("breakvar")[0], expected, rtol=1e-10)
    res.standardized_forecasts_error = errors[None, :]
    assert_allclose(res.test_heteroskedasticity("breakvar")[0], expected, rtol=1e-10)


@pytest.mark.parametrize("concentrated", [True, False])
def test_base_results_statespace_attributes(mean_data, concentrated):
    # Results with the attributes of the statespace MLEResults: the
    # diagnostics leave out the first d = max(loglikelihood_burn,
    # nobs_diffuse) = 4 standardized errors of filter_results, and the
    # summary shows the scale only if the filter concentrated it out
    res = MeanModel(mean_data).fit()
    errors = np.random.default_rng(2045).standard_normal(mean_data.shape[0])
    res.loglikelihood_burn = 4
    res.nobs_diffuse = 2
    res.filter_results = Bunch(
        standardized_forecasts_error=errors[None, :],
        filter_concentrated=concentrated,
    )
    res.rsquared = 0.25
    used = errors[4:]
    # the default lags, min(10, (nobs - d) // 5), as the model has no
    # seasonal_periods
    lb = res.test_serial_correlation("ljungbox")
    assert lb.shape == (1, 2, 10)
    assert_allclose(lb[0], ljung_box(used, 10), rtol=1e-10)
    # h = round((60 - 4) / 3) = 19
    het = res.test_heteroskedasticity("breakvar")
    assert_allclose(het[0], breakvar(used, 19), rtol=1e-10)
    assert_allclose(res.test_normality("jarquebera")[0], jarque_bera(used), rtol=1e-10)

    with np.errstate(divide="ignore", invalid="ignore"):
        text = res.summary().as_text()
    assert re.search(r"R-squared:\s+0\.250", text)
    assert ("Scale" in text) == concentrated


@pytest.mark.parametrize(
    "error, seasonal, start, end",
    [
        ("add", None, 10, 10),
        ("add", None, 0, 0),
        ("add", None, 67, 71),
        ("mul", "mul", 67, 71),
    ],
)
def test_get_prediction_simulated_one_step(austourists, error, seasonal, start, end):
    # With exactly one in-sample prediction, its one-step simulation was
    # skipped (`if nsmooth > 1`), so get_prediction raised "need at least
    # one array to concatenate", or the simulations missed the first period
    # and the prediction intervals raised a shape error
    model = ETSModel(
        austourists,
        error=error,
        seasonal=seasonal,
        seasonal_periods=4 if seasonal else None,
    )
    res = model.smooth(model.start_params, param_index="integer")
    reps = 50
    pred = res.get_prediction(
        start=start,
        end=end,
        method="simulated",
        simulate_repetitions=reps,
        rng=np.random.default_rng(1),
    )
    frame = pred.summary_frame()
    assert frame.shape[0] == end - start + 1

    # The simulation of the first period is the one-step prediction with a
    # simulated error, drawn first from the generator; the out-of-sample
    # periods are simulated from the end of the sample after it
    rng = np.random.default_rng(1)
    eps = rng.standard_normal(reps) * np.sqrt(res.scale)
    fitted = np.asarray(res.fittedvalues)[start]
    first = fitted + eps if error == "add" else fitted * (1 + eps)
    expected = [first]
    if end > start:
        sims = res.simulate(end - start, anchor="end", repetitions=reps, rng=rng)
        expected += list(np.asarray(sims))
    expected = np.asarray(expected)
    assert_allclose(frame["mean"].iloc[0], fitted, rtol=1e-12)
    assert_allclose(frame["mean_numerical"], expected.mean(1), rtol=1e-12)
    assert_allclose(frame["pi_lower"], np.quantile(expected, 0.025, axis=1), rtol=1e-12)
    assert_allclose(frame["pi_upper"], np.quantile(expected, 0.975, axis=1), rtol=1e-12)


#
# THE INDEX OF THE RESULTS INDEXED BY PARAMETER (param_index)
#

PARAM_INDEX_WARNING = 'The default param_index will change from "integer"'


def param_results(res):
    # The results that are indexed by parameter
    return {
        "params": res.params,
        "bse": res.bse,
        "tvalues": res.tvalues,
        "pvalues": res.pvalues,
        "conf_int": res.conf_int(alpha=0.1),
        "cov_params": res.cov_params(),
    }


def param_index_warnings(record):
    return [w for w in record if PARAM_INDEX_WARNING in str(w.message)]


def param_index_call(model, method):
    # The methods that return results, with fixed parameters for smooth
    params = fixed_params(model)
    return {
        "fit": lambda **kwargs: model.fit(disp=False, **kwargs),
        "fit_constrained": lambda **kwargs: model.fit_constrained(
            {"smoothing_trend": 0.1}, **kwargs
        ),
        "smooth": lambda **kwargs: model.smooth(params, **kwargs),
    }[method]


@pytest.mark.parametrize("method", ["fit", "fit_constrained", "smooth"])
def test_param_index_default_warns(austourists_model, method):
    # Unset, param_index is "integer" with one FutureWarning that points at
    # the caller and says how to opt in or out
    call = param_index_call(austourists_model, method)
    with pytest.warns(FutureWarning, match=PARAM_INDEX_WARNING) as rec:
        res = call()
    found = param_index_warnings(rec)
    assert len(found) == 1
    assert found[0].filename == __file__
    message = str(found[0].message)
    assert "after statsmodels 0.17 is released" in message
    assert 'param_index="names"' in message
    assert 'param_index="integer"' in message

    k = len(austourists_model.param_names)
    for key, value in param_results(res).items():
        assert value.index.equals(pd.RangeIndex(k)), key
    expected = param_results(call(param_index="integer"))
    for key, value in param_results(res).items():
        assert type(value) is type(expected[key])
        assert_allclose(value, expected[key], rtol=1e-12)


@pytest.mark.parametrize("param_index", ["names", "integer"])
def test_param_index_options(austourists_model, param_index, recwarn):
    # "names" labels every result indexed by parameter with the parameter
    # names of the model, "integer" with 0, 1, ...; the values are those of
    # the other option, and neither warns
    params = fixed_params(austourists_model)
    res = austourists_model.smooth(params, param_index=param_index)
    other = "integer" if param_index == "names" else "names"
    expected = param_results(austourists_model.smooth(params, param_index=other))
    names = austourists_model.param_names
    labels = pd.Index(names) if param_index == "names" else pd.RangeIndex(len(names))
    for key, value in param_results(res).items():
        assert value.index.equals(labels), key
        assert_allclose(value.to_numpy(), expected[key].to_numpy(), rtol=1e-12)
    assert res.cov_params().columns.equals(labels)
    assert list(res.conf_int().columns) == [0, 1]
    # the lower and upper bounds of a 90% interval
    q = scipy.stats.norm.ppf(0.95)
    assert_allclose(
        res.conf_int(alpha=0.1)[0], res.params - q * res.bse, rtol=1e-12
    )
    assert_allclose(
        res.conf_int(alpha=0.1)[1], res.params + q * res.bse, rtol=1e-12
    )
    assert not param_index_warnings(recwarn)


def test_param_index_names_fit(austourists_model, recwarn):
    # The names of a fit: the values, the summary and the predictions are
    # those of the integer index, they survive pickling, and the results of
    # another fit of the same model keep their own index
    res = austourists_model.fit(disp=False, param_index="names")
    expected = austourists_model.fit(disp=False, param_index="integer")
    names = pd.Index(austourists_model.param_names)
    assert res.params.index.equals(names)
    assert expected.params.index.equals(pd.RangeIndex(len(names)))
    assert_allclose(res.params.to_numpy(), expected.params.to_numpy(), rtol=1e-12)
    # the parameter table, from the same numbers
    table = res.summary().tables[1].as_text()
    assert table == expected.summary().tables[1].as_text()
    assert all(name in table for name in names)
    frame = res.get_prediction(start=60, end=72).summary_frame()
    assert_allclose(
        frame,
        expected.get_prediction(start=60, end=72).summary_frame(),
        rtol=1e-12,
    )
    restored = pickle.loads(pickle.dumps(res))
    assert restored.params.index.equals(names)
    assert restored.cov_params().columns.equals(names)
    assert_allclose(restored.bse, res.bse, rtol=1e-12)
    assert not param_index_warnings(recwarn)


def test_param_index_fixed_params(austourists_model):
    # Fixed parameters are labelled with their names in the model, without
    # the " (fixed)" of res.param_names, and keep their value
    params = fixed_params(austourists_model)
    with austourists_model.fix_params({"smoothing_trend": 0.1}):
        res = austourists_model.smooth(params, param_index="names")
        expected = austourists_model.smooth(params, param_index="integer")
        fitted = austourists_model.fit(disp=False, param_index="names")
    names = austourists_model.param_names
    assert "smoothing_trend (fixed)" in res.param_names
    for key, value in param_results(res).items():
        assert list(value.index) == names, key
        assert_allclose(value.to_numpy(), param_results(expected)[key], rtol=1e-12)
    assert list(fitted.params.index) == names
    assert_allclose(fitted.params["smoothing_trend"], 0.1, rtol=1e-12)
    assert np.isnan(fitted.bse["smoothing_trend"])


@pytest.mark.parametrize("param_index", [None, "names", "integer"])
def test_param_index_ndarray(austourists, param_index, recwarn):
    # With ndarray data, the results are ndarrays whatever param_index, and
    # the default does not warn
    kwargs = dict(trend="add", seasonal="add", seasonal_periods=4)
    model = ETSModel(austourists.to_numpy(), **kwargs)
    params = fixed_params(model)
    res = model.smooth(params, param_index=param_index)
    expected = param_results(
        ETSModel(austourists, **kwargs).smooth(params, param_index="integer")
    )
    for key, value in param_results(res).items():
        assert isinstance(value, np.ndarray), key
        assert_allclose(value, expected[key].to_numpy(), rtol=1e-12)
    assert not param_index_warnings(recwarn)


def test_param_index_no_results_no_warning(austourists_model, recwarn):
    # Without a results object, the default does not warn
    params = fixed_params(austourists_model)
    fitted = austourists_model.fit(disp=False, return_params=True)
    assert isinstance(fitted, np.ndarray)
    yhat, _ = austourists_model.smooth(params, return_raw=True)
    assert isinstance(yhat, pd.Series)
    assert not param_index_warnings(recwarn)


@pytest.mark.parametrize("method", ["fit", "smooth"])
@pytest.mark.parametrize("param_index", ["Names", "labels"])
def test_param_index_invalid(austourists_model, method, param_index):
    call = param_index_call(austourists_model, method)
    with pytest.raises(ValueError, match="param_index"):
        call(param_index=param_index)


@pytest.mark.parametrize(
    "steps",
    [
        "2016-01-01",
        pd.Timestamp("2016-01-01"),
        pd.Timestamp("2016-01-01").to_pydatetime(),
        3,
        np.int64(3),
        np.array(3),
    ],
    ids=["str", "Timestamp", "datetime", "int", "int64", "0-d array"],
)
@pytest.mark.parametrize("use_pandas", [True, False])
def test_forecast_date(oildata, steps, use_pandas):
    # forecast with a date raised "TypeError: 'str' object cannot be
    # interpreted as an integer"; a date forecasts until and including it
    endog = oildata if use_pandas else oildata.to_numpy()
    kwargs = {} if use_pandas else {"dates": oildata.index}
    res = ETSModel(endog, **kwargs).smooth([0.3, 110.0], param_index="integer")
    # the forecasts of simple exponential smoothing are the last level
    level = 110.0
    for value in oildata:
        level = 0.3 * value + 0.7 * level
    fcast = res.forecast(steps)
    assert_allclose(np.asarray(fcast), np.full(3, level), rtol=1e-12)
    if use_pandas:
        expected_index = pd.date_range("2014-01-01", periods=3, freq="YS")
        assert_array_equal(fcast.index, expected_index)
    else:
        assert isinstance(fcast, np.ndarray)


@pytest.mark.parametrize(
    "end, dynamic",
    [
        (20, 11),
        (20, 12),
        (20, 15),
        (20, "2005-06-30"),
        (75, 58),
        (75, 59),
        (75, 65),
        (75, 70),
        (75, "2016-06-30"),
    ],
)
def test_predict_dynamic_after_end_or_sample(austourists, end, dynamic):
    # Dynamic prediction begins at start + dynamic, and the forecasts after
    # the sample are dynamic anyway, so a dynamic after the end of the
    # prediction or after the sample has no effect. Both raised a ValueError
    # (a broadcast error, or "Cannot anchor simulation outside of the
    # sample"); start + dynamic = end + 1 (here 21) or nobs (68) worked. The
    # date keys are the periods 25 and 69. The predictions are the one-step
    # predictions in the sample followed by the forecasts from its end, with
    # the variances of the exact method: mse in the sample, and mse times
    # the relative variances of R's forecast package after it.
    model = ETSModel(austourists)
    res = model.smooth(fixed_params(model), param_index="integer")
    n, start = res.nobs, 10
    nforecast = max(end - n + 1, 0)
    expected = res.fittedvalues.to_numpy()[start : end + 1]
    variance = np.full(expected.shape[0], res.mse)
    if nforecast:
        expected = np.r_[expected, res.forecast(nforecast).to_numpy()]
        relative = R_RELATIVE_FORECAST_VARIANCE["ANN"][:nforecast]
        variance = np.r_[variance, res.mse * np.asarray(relative)]
    index = austourists.index
    index = pd.date_range(index[0], periods=end + 1, freq=index.freq)[start:]

    pred = res.predict(start=start, end=end, dynamic=dynamic)
    assert pred.index.equals(index)
    assert_allclose(pred, expected, rtol=1e-12)

    pred = res.get_prediction(start=start, end=end, dynamic=dynamic)
    frame = pred.summary_frame()
    assert frame.index.equals(index)
    assert_allclose(frame["mean"], expected, rtol=1e-12)
    assert_allclose(pred.var_pred_mean, variance, rtol=1e-12)

    # the simulated intervals are those of the prediction without dynamic
    kwargs = dict(start=start, end=end, method="simulated", simulate_repetitions=5)
    pred = res.get_prediction(dynamic=dynamic, rng=np.random.default_rng(0), **kwargs)
    expected = res.get_prediction(rng=np.random.default_rng(0), **kwargs)
    assert_allclose(
        pred.summary_frame(), expected.summary_frame(), rtol=1e-12
    )


@pytest.mark.parametrize("dynamic", [True, 0, 1, 3, "date", "later date"])
@pytest.mark.parametrize("start", [68, 70])
def test_predict_dynamic_start_after_sample(austourists, start, dynamic):
    # With a start after the end of the sample (nobs = 68), the predictions
    # are the forecasts from the end of the sample whatever dynamic is. A
    # dynamic at start (True, 0, or the date of start) began the forecasts
    # at start, and a later one after it, so these raised "Cannot anchor
    # simulation outside of the sample", except for start = nobs with
    # dynamic at start. The exact variances are mse times the relative
    # variances of R's forecast package.
    model = ETSModel(austourists)
    res = model.smooth(fixed_params(model), param_index="integer")
    n, end = res.nobs, 75
    index = austourists.index
    index = pd.date_range(index[0], periods=end + 1, freq=index.freq)
    if dynamic in ("date", "later date"):
        dynamic = str(index[start + 2 * (dynamic == "later date")].date())
    index = index[start:]
    expected = res.forecast(end - n + 1).to_numpy()[start - n :]
    relative = R_RELATIVE_FORECAST_VARIANCE["ANN"][start - n : end - n + 1]

    pred = res.predict(start=start, end=end, dynamic=dynamic)
    assert pred.index.equals(index)
    assert_allclose(pred, expected, rtol=1e-12)

    pred = res.get_prediction(start=start, end=end, dynamic=dynamic)
    frame = pred.summary_frame()
    assert frame.index.equals(index)
    assert_allclose(frame["mean"], expected, rtol=1e-12)
    assert_allclose(pred.var_pred_mean, res.mse * np.asarray(relative), rtol=1e-12)


@pytest.mark.parametrize(
    "data, seasonal, lags",
    [("oildata", None, 9), ("austourists", None, 10), ("austourists", "add", 8)],
)
def test_serial_correlation_default_lags(oildata, austourists, data, seasonal, lags):
    # Non-seasonal models store seasonal_periods = 1, so the default number
    # of lags was min(2, nobs // 5) instead of min(10, nobs // 5). The
    # cases are min(10, 49 // 5), min(10, 68 // 5) and min(2 * 4, 68 // 5).
    from statsmodels.stats.diagnostic import acorr_ljungbox

    endog = oildata if data == "oildata" else austourists
    seasonal_periods = 4 if seasonal else None
    model = ETSModel(
        endog, trend="add", seasonal=seasonal, seasonal_periods=seasonal_periods
    )
    res = model.smooth(model.start_params, param_index="integer")
    resid = np.asarray(res.resid)
    std_resid = (resid - resid.mean()) / resid.std(ddof=1)
    expected = acorr_ljungbox(std_resid, lags=lags)

    actual = res.test_serial_correlation("ljungbox")
    assert actual.shape == (1, 2, lags)
    assert_allclose(actual[0, 0], expected["lb_stat"], rtol=1e-10)
    assert_allclose(actual[0, 1], expected["lb_pvalue"], rtol=1e-10)
    # the summary shows the statistic at the largest lag, with two decimals
    row = res.summary().tables[2][0]
    assert str(row[0]) == "Ljung-Box (Q):"
    assert_allclose(float(str(row[1])), expected["lb_stat"].iloc[-1], atol=0.005)


def test_predict_dynamic_key_covers_several_dates():
    # A dynamic date key that covers several dates (all of April 2001 in daily
    # data) gives a slice of locations, which raised a TypeError. Dynamic
    # prediction begins at the first date of the key.
    rng = np.random.default_rng(20261121)
    index = pd.date_range("2000-01-01", periods=500, freq="D")
    y = pd.Series(10 + rng.standard_normal(500).cumsum(), index=index)
    res = ETSModel(y).smooth([0.5, 10.0], param_index="integer")
    loc = 366 + 31 + 28 + 31
    assert index[loc] == pd.Timestamp("2001-04-01")
    pred = res.predict(start=10, dynamic="2001-04")
    expected = res.predict(start=10, dynamic=loc - 10)
    assert_allclose(pred, expected, rtol=1e-12)
    pd.testing.assert_index_equal(pred.index, index[10:])


@pytest.mark.parametrize("method", ["exact", "simulated"])
@pytest.mark.parametrize("dates", [False, True])
def test_get_prediction_numpy(oildata, method, dates):
    # get_prediction raised AttributeError for numpy data (it read the index
    # of the predicted mean). The numpy path must give the pandas values;
    # the variance of the simulations uses ddof=1, the default of pandas.
    res = ETSModel(oildata).smooth([0.3, 110.0], param_index="integer")
    kwargs = {"dates": oildata.index} if dates else {}
    res_np = ETSModel(oildata.to_numpy(), **kwargs).smooth(
        [0.3, 110.0], param_index="integer"
    )
    pred_kwargs = dict(start=46, end=51, method=method, simulate_repetitions=20)
    pred = res.get_prediction(rng=np.random.default_rng(0), **pred_kwargs)
    pred_np = res_np.get_prediction(rng=np.random.default_rng(0), **pred_kwargs)

    assert isinstance(pred_np.predicted_mean, np.ndarray)
    assert_allclose(
        pred_np.predicted_mean, pred.predicted_mean.to_numpy(), rtol=1e-12
    )
    assert_allclose(pred_np.var_pred_mean, np.asarray(pred.var_pred_mean), rtol=1e-12)
    pred_int = pred_np.pred_int()
    assert isinstance(pred_int, np.ndarray)
    assert_allclose(pred_int, pred.pred_int().to_numpy(), rtol=1e-12)
    frame_np = pred_np.summary_frame()
    frame = pred.summary_frame()
    assert list(frame_np.columns) == list(frame.columns)
    assert_allclose(frame_np.to_numpy(), frame.to_numpy(), rtol=1e-12)


@pytest.mark.parametrize("data", ["pandas", "numpy"])
@pytest.mark.parametrize("error", ["add", "mul"])
def test_get_prediction_simulated_start_after_sample(austourists, error, data):
    # With a start after the end of the sample, the predictions are the
    # forecasts from the end of the sample without those before start. The
    # simulations kept those, so pred_int and summary_frame raised "Shape of
    # passed values is (6, 2), indices imply (4, 2)" (summary_frame "All
    # arrays must be of the same length" for numpy data) and var_pred_mean
    # had a value for each of them. The simulations are those of simulate from
    # the end of the sample, drawn from the same generator, from start on.
    n = austourists.shape[0]
    endog = austourists if data == "pandas" else austourists.to_numpy()
    model = ETSModel(endog, error=error)
    res = model.smooth(fixed_params(model), param_index="integer")
    reps, start, end = 20, n + 2, n + 5
    pred = res.get_prediction(
        start=start,
        end=end,
        method="simulated",
        simulate_repetitions=reps,
        rng=np.random.default_rng(0),
    )
    steps = end - n + 1
    sims = res.simulate(
        steps, anchor="end", repetitions=reps, rng=np.random.default_rng(0)
    )
    sims = np.asarray(sims)[start - n :]
    forecast = np.asarray(res.forecast(steps))[start - n :]

    frame = pred.summary_frame(alpha=0.1)
    assert_allclose(frame["mean"], forecast, rtol=1e-12)
    # rtol: pandas and numpy may sum in another order
    assert_allclose(frame["mean_numerical"], sims.mean(1), rtol=1e-10)
    assert_allclose(frame["pi_lower"], np.quantile(sims, 0.05, axis=1), rtol=1e-12)
    assert_allclose(frame["pi_upper"], np.quantile(sims, 0.95, axis=1), rtol=1e-12)
    pred_int = pred.pred_int(alpha=0.1)
    assert_allclose(pred_int, frame[["pi_lower", "pi_upper"]], rtol=1e-12)
    assert_allclose(pred.var_pred_mean, sims.var(1, ddof=1), rtol=1e-10)
    if data == "pandas":
        index = extended_index(austourists.index, steps)[start - n :]
        assert frame.index.equals(index)
        assert pred_int.index.equals(index)
        assert pred.var_pred_mean.index.equals(index)


@pytest.mark.parametrize("start", [66, 70])
@pytest.mark.parametrize("method", ["exact", "simulated"])
def test_get_prediction_endog_start_after_sample(austourists, method, start):
    # endog of the prediction results has one value per prediction, the
    # observation or NaN after the sample (nobs = 68). With a start after
    # the sample it also had one for each forecast before start, so reading
    # it raised "Length of values (8) does not match length of index (6)".
    model = ETSModel(austourists)
    res = model.smooth(fixed_params(model), param_index="integer")
    n, end = res.nobs, 75
    pred = res.get_prediction(
        start=start,
        end=end,
        method=method,
        simulate_repetitions=5,
        rng=np.random.default_rng(0),
    )
    expected = np.r_[austourists.to_numpy(), np.full(end + 1 - n, np.nan)]
    index = austourists.index
    index = pd.date_range(index[0], periods=end + 1, freq=index.freq)[start:]
    assert pred.endog.index.equals(index)
    assert_allclose(pred.endog, expected[start:], rtol=1e-12)


@pytest.mark.parametrize(
    "start, end, dynamic",
    [
        (10, 14, True),
        (10, 14, 0),
        (10, 14, 2),
        (38, 42, True),
        (38, 42, 0),
        (38, 42, 1),
        (39, 42, True),
        (45, 50, False),
    ],
)
@pytest.mark.parametrize("use_pandas", [False, True])
def test_get_prediction_endog_dynamic(use_pandas, start, end, dynamic):
    # With all predictions dynamic (True or 0), endog was NaN also in the
    # sample. It holds the observed values of the in-sample periods
    rng = np.random.default_rng(20261124)
    y = 10 + 0.3 * rng.standard_normal(40).cumsum()
    index = pd.date_range("2000-01-01", periods=40, freq="MS")
    data = pd.Series(y, index=index) if use_pandas else y
    res = ETSModel(data).smooth([0.4, 10.0], param_index="integer")
    pred = res.get_prediction(start=start, end=end, dynamic=dynamic)
    expected = np.full(end - start + 1, np.nan)
    nobs_in = max(min(end, 39) - start + 1, 0)
    expected[:nobs_in] = y[start : start + nobs_in]
    assert_allclose(np.asarray(pred.endog), expected, rtol=1e-15)
    if use_pandas:
        pd.testing.assert_index_equal(
            pred.endog.index, pred.predicted_mean.index, exact=False
        )
