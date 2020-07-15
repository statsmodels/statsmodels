"""
Author: Samuel Scherrer
"""
from itertools import product
import json
import pathlib

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest
import scipy.stats

from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing

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

###############################################################################
# UTILS
###############################################################################

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
    product(ERRORS, TRENDS, ("add", "mul"), DAMPED, ("austourists",),)
)
MODELS_DATA_NONSEASONAL = list(
    product(ERRORS, TRENDS, (None,), DAMPED, ("oildata",),)
)
remove_invalid_models_from_list(MODELS_DATA_SEASONAL)
remove_invalid_models_from_list(MODELS_DATA_NONSEASONAL)

ALL_MODELS_AND_DATA = MODELS_DATA_NONSEASONAL + MODELS_DATA_SEASONAL


def short_model_name(error, trend, seasonal):
    short_name = {"add": "A", "mul": "M", None: "N"}
    return short_name[error] + short_name[trend] + short_name[seasonal]


@pytest.fixture(params=ALL_MODELS_AND_DATA)
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

    model = ETSModel(
        data,
        seasonal_periods=seasonal_periods,
        error=error,
        trend=trend,
        seasonal=seasonal,
        damped_trend=damped,
    )

    return model, params, results_R


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
def oildata_model(oildata):
    return ETSModel(oildata, error="add", trend="add", damped_trend=True,)


#############################################################################
# DATA
#############################################################################


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
    index = pd.date_range("1999-01-01", "2015-12-31", freq="Q")
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
    return pd.Series(data, index=pd.date_range("1965", "2013", freq="AS"))


#############################################################################
# REFERENCE RESULTS
#############################################################################


def obtain_R_results(path):
    with path.open("r") as f:
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
    for damped in results:
        for model in results[damped]:
            for key in ["alpha", "beta", "gamma", "phi", "sigma2"]:
                results[damped][model][key] = float(
                    results[damped][model][key][0]
                )
            for key in [
                "states",
                "initstate",
                "residuals",
                "fitted",
                "forecast",
                "simulation",
            ]:
                results[damped][model][key] = np.asarray(
                    results[damped][model][key]
                )
    return results


@pytest.fixture
def ets_austourists_fit_results_R():
    """
    Dictionary of ets fit results obtained with script ``results/fit_ets.R``.
    """
    path = (
        pathlib.Path(__file__).parent
        / "results"
        / "fit_ets_results_seasonal.json"
    )
    return obtain_R_results(path)


@pytest.fixture
def ets_oildata_fit_results_R():
    """
    Dictionary of ets fit results obtained with script ``results/fit_ets.R``.
    """
    path = (
        pathlib.Path(__file__).parent
        / "results"
        / "fit_ets_results_nonseasonal.json"
    )
    return obtain_R_results(path)


def fit_austourists_with_R_params(model, results_R, set_state=False):
    """
    Fit the model with params as found by R's forecast package
    """
    params = get_params_from_R(results_R)
    with model.fix_params(dict(zip(model.param_names, params))):
        fit = model.fit(disp=False)

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


#############################################################################
# BASIC TEST CASES
#############################################################################


def test_fit_model_austouritsts(setup_model):
    model, params, results_R = setup_model
    model.fit(disp=False)


#############################################################################
# TEST OF MODEL EQUATIONS VS R
#############################################################################


def test_smooth_vs_R(setup_model):
    model, params, results_R = setup_model

    yhat, xhat = model.smooth(params, return_raw=True)

    yhat_R = results_R["fitted"]
    xhat_R = get_states_from_R(results_R, model._k_states)

    assert_allclose(xhat, xhat_R, rtol=1e-5, atol=1e-5)
    assert_allclose(yhat, yhat_R, rtol=1e-5, atol=1e-5)


def test_residuals_vs_R(setup_model):
    model, params, results_R = setup_model

    yhat = model.smooth(params, return_raw=True)[0]

    residuals = model._residuals(yhat)
    assert_allclose(residuals, results_R["residuals"], rtol=1e-5, atol=1e-5)


def test_loglike_vs_R(setup_model):
    model, params, results_R = setup_model

    loglike = model.loglike(params)
    # the calculation of log likelihood in R is only up to a constant:
    const = -model.nobs / 2 * (np.log(2 * np.pi / model.nobs) + 1)
    loglike_R = results_R["loglik"][0] + const

    assert_allclose(loglike, loglike_R, rtol=1e-5, atol=1e-5)


def test_forecast_vs_R(setup_model):
    model, params, results_R = setup_model

    fit = fit_austourists_with_R_params(model, results_R, set_state=True)

    fcast = fit.forecast(4)
    expected = np.asarray(results_R["forecast"])

    assert_allclose(expected, fcast.values, rtol=1e-3, atol=1e-4)


def test_simulate_vs_R(setup_model):
    model, params, results_R = setup_model

    fit = fit_austourists_with_R_params(model, results_R, set_state=True)

    innov = np.asarray([[1.76405235, 0.40015721, 0.97873798, 2.2408932]]).T
    sim = fit.simulate(4, anchor="end", repetitions=1, random_errors=innov)
    expected = np.asarray(results_R["simulation"])

    assert_allclose(expected, sim.values, rtol=1e-5, atol=1e-5)


def test_fit_vs_R(setup_model, reset_randomstate):
    model, params, results_R = setup_model
    fit = model.fit(disp=True, tol=1e-8)

    # check log likelihood: we want to have a fit that is better, i.e. a fit
    # that has a **higher** log-likelihood
    const = -model.nobs / 2 * (np.log(2 * np.pi / model.nobs) + 1)
    loglike_R = results_R["loglik"][0] + const
    loglike = fit.llf
    assert loglike >= loglike_R - 1e-4


def test_predict_vs_R(setup_model):
    model, params, results_R = setup_model
    fit = fit_austourists_with_R_params(model, results_R, set_state=True)

    n = fit.nobs
    prediction = fit.predict(end=n + 3, dynamic=n)

    yhat_R = results_R["fitted"]
    assert_allclose(prediction[:n], yhat_R, rtol=1e-5, atol=1e-5)

    forecast_R = results_R["forecast"]
    assert_allclose(prediction[n:], forecast_R, rtol=1e-3, atol=1e-4)


#############################################################################
# OTHER TESTS
#############################################################################


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
    fit_estimated = model_estimated.fit(disp=False)
    fit_heuristic = model_heuristic.fit(disp=False)
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
    fit1 = model1.fit(disp=False)
    assert fit1.smoothing_trend == 0.99

    # same using with fix_params semantic
    model2 = ETSModel(oildata, error="add", trend="add", damped_trend=True,)
    with model2.fix_params({"smoothing_trend": 0.99}):
        fit2 = model2.fit(disp=False)
    assert fit2.smoothing_trend == 0.99
    assert_allclose(fit1.params, fit2.params)
    fit2.summary()  # check if summary runs without failing

    # using fit_constrained
    fit3 = model2.fit_constrained({"smoothing_trend": 0.99})
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


def test_simulate_keywords(austourists_model):
    """
    check whether all keywords are accepted and work without throwing errors.
    """
    fit = austourists_model.fit(disp=False)

    # test anchor
    assert_almost_equal(
        fit.simulate(4, anchor=0, random_state=0).values,
        fit.simulate(4, anchor="start", random_state=0).values,
    )
    assert_almost_equal(
        fit.simulate(4, anchor=-1, random_state=0).values,
        fit.simulate(4, anchor="2015-12-31", random_state=0).values,
    )
    assert_almost_equal(
        fit.simulate(4, anchor="end", random_state=0).values,
        fit.simulate(4, anchor="2016-03-31", random_state=0).values,
    )

    # test different random error options
    fit.simulate(4, repetitions=10)
    fit.simulate(4, repetitions=10, random_errors=scipy.stats.norm)
    fit.simulate(4, repetitions=10, random_errors=scipy.stats.norm())
    fit.simulate(4, repetitions=10, random_errors=np.random.randn(4, 10))
    fit.simulate(4, repetitions=10, random_errors="bootstrap")

    # test seeding
    res = fit.simulate(4, repetitions=10, random_state=10).values
    res2 = fit.simulate(
        4, repetitions=10, random_state=np.random.RandomState(10)
    ).values
    assert np.all(res == res2)


def test_summary(austourists_model):
    # just try to run summary to see if it works
    fit = austourists_model.fit(disp=False)
    fit.summary()

    # now without estimated initial states
    austourists_model.set_initialization_method("heuristic")
    fit = austourists_model.fit(disp=False)
    fit.summary()

    # and with fixed params
    fit = austourists_model.fit_constrained({"smoothing_trend": 0.9})
    fit.summary()


def test_score(austourists_model):
    fit = austourists_model.fit(disp=False)
    score_cs = austourists_model.score(fit.params)
    score_fd = austourists_model.score(
        fit.params, approx_complex_step=False, approx_centered=True,
    )
    assert_almost_equal(score_cs, score_fd, 4)


def test_hessian(austourists_model):
    # The hessian approximations are not very consistent, but the test makes
    # sure they run
    fit = austourists_model.fit(disp=False)
    austourists_model.hessian(fit.params)
    austourists_model.hessian(
        fit.params, approx_complex_step=False, approx_centered=True,
    )


def test_convergence_simple():
    # issue 6883
    gen = np.random.RandomState(0)
    e = gen.standard_normal(12000)
    y = e.copy()
    for i in range(1, e.shape[0]):
        y[i] = y[i - 1] - 0.2 * e[i - 1] + e[i]
    y = y[200:]
    res = ExponentialSmoothing(y).fit()
    ets_res = ETSModel(y).fit()

    # the smoothing level should be very similar, the initial state might be
    # different as it doesn't influence the final result too much
    assert_almost_equal(
        res.params['smoothing_level'],
        ets_res.smoothing_level,
        3
    )

    # the first few values are influenced by differences in initial state, so
    # we don't test them here
    assert_almost_equal(
        res.fittedvalues[10:],
        ets_res.fittedvalues[10:],
        3
    )
