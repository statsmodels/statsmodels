"""
Author: Samuel Scherrer
"""
import json
import pathlib
from itertools import product

import numpy as np
import pandas as pd
import scipy.stats
import pytest
from numpy.testing import assert_almost_equal, assert_allclose

from statsmodels.tsa.exponential_smoothing.ets import (
    ETSModel,
)


###############################################################################
# UTILS
###############################################################################

ERRORS = ("add", "mul")
TRENDS = ("add", "mul", None)
SEASONALS = ("add", "mul", None)
DAMPED = (True, False)
MODELLIST = list(product(ERRORS, TRENDS, SEASONALS, DAMPED))
# remove invalid models (no trend but damped)
for i, model in enumerate(MODELLIST):
    if model[1] is None and model[3]:
        del MODELLIST[i]
ALL_MODELS = (["error", "trend", "seasonal", "damped"], MODELLIST)


def short_model_name(error, trend, seasonal):
    short_name = {"add": "A", "mul": "M", None: "N"}
    return short_name[error] +  short_name[trend] + short_name[seasonal]


###############################################################################
# DATA
###############################################################################

@pytest.fixture
def austourists():
    # austourists dataset from fpp2 package
    # https://cran.r-project.org/web/packages/fpp2/index.html
    data = [30.05251, 19.14850, 25.31769, 27.59144, 32.07646,
            23.48796, 28.47594, 35.12375, 36.83848, 25.00702,
            30.72223, 28.69376, 36.64099, 23.82461, 29.31168,
            31.77031, 35.17788, 19.77524, 29.60175, 34.53884,
            41.27360, 26.65586, 28.27986, 35.19115, 42.20566,
            24.64917, 32.66734, 37.25735, 45.24246, 29.35048,
            36.34421, 41.78208, 49.27660, 31.27540, 37.85063,
            38.83704, 51.23690, 31.83855, 41.32342, 42.79900,
            55.70836, 33.40714, 42.31664, 45.15712, 59.57608,
            34.83733, 44.84168, 46.97125, 60.01903, 38.37118,
            46.97586, 50.73380, 61.64687, 39.29957, 52.67121,
            54.33232, 66.83436, 40.87119, 51.82854, 57.49191,
            65.25147, 43.06121, 54.76076, 59.83447, 73.25703,
            47.69662, 61.09777, 66.05576,]
    index = pd.date_range("1999-03-01", "2015-12-01", freq="3MS")
    return pd.Series(data, index)

@pytest.fixture
def oildata():
    # oildata dataset from fpp2 package
    # https://cran.r-project.org/web/packages/fpp2/index.html
    data = [
        111.0091, 130.8284, 141.2871, 154.2278,
        162.7409, 192.1665, 240.7997, 304.2174,
        384.0046, 429.6622, 359.3169, 437.2519,
        468.4008, 424.4353, 487.9794, 509.8284,
        506.3473, 340.1842, 240.2589, 219.0328,
        172.0747, 252.5901, 221.0711, 276.5188,
        271.1480, 342.6186, 428.3558, 442.3946,
        432.7851, 437.2497, 437.2092, 445.3641,
        453.1950, 454.4096, 422.3789, 456.0371,
        440.3866, 425.1944, 486.2052, 500.4291,
        521.2759, 508.9476, 488.8889, 509.8706,
        456.7229, 473.8166, 525.9509, 549.8338,
        542.3405
    ]
    return pd.Series(data, index=pd.date_range('1965', '2013', freq='AS'))

###############################################################################
# REFERENCE RESULTS
###############################################################################

@pytest.fixture
def ets_austourists_fit_results_R():
    """
    Dictionary of ets fit results obtained with script ``results/fit_ets.R``.
    """
    path = pathlib.Path(__file__).parent / "results" / "fit_ets_results.json"
    with path.open('r') as f:
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
                results[damped][model][key] = float(results[damped][model][key][0])
            for key in ["states", "initstate", "residuals", "fitted",
                        "forecast", "simulation"]:
                results[damped][model][key] = np.asarray(results[damped][model][key])
    return results


def fit_austourists_with_R_params(model, results_R, set_state=False):
    """Fit the model with params as found by R's forecast package"""
    fit = model.fit(
        smoothing_level=results_R["alpha"], smoothing_slope=results_R["beta"],
        smoothing_seasonal=results_R["gamma"], damping_slope=results_R["phi"],
        optimized=False
    )

    if set_state:
        states = results_R["states"]
        if states.ndim == 1:
            states = np.reshape(states, (len(states), 1))
        fit.level[:] = states[1:, 0]
        idx = 1
        if not np.isnan(results_R["beta"]):
            fit.slope[:] = states[1:, idx]
            idx += 1
        if not np.isnan(results_R["gamma"]):
            fit.season[:] = states[1:, idx]
    return fit

###############################################################################
# BASIC TEST CASES
###############################################################################

@pytest.mark.parametrize(*ALL_MODELS)
def test_fit_model(error, trend, seasonal, damped, austourists, oildata):
    if seasonal is None:
        data = oildata
    else:
        data = austourists
    model = ETSModel(
        data, seasonal_periods=4,
        error=error, trend=trend, seasonal=seasonal, damped_trend=damped
    )

    result = model.fit()


###############################################################################
# SIMULATE TESTS
###############################################################################

@pytest.mark.skip
@pytest.mark.parametrize(*ALL_MODELS)
def test_simulate_vs_R(austourists, ets_austourists_fit_results_R,
                       error, trend, seasonal, damped):
    """
    Test for :meth:``statsmodels.tsa.holtwinters.HoltWintersResults.simulate``.

    The tests are using the implementation in the R package ``forecast`` as
    reference, and example data is taken from ``fpp2`` (package and book).
    """

    model_name = short_model_name(error, trend, seasonal)
    results = ets_austourists_fit_results_R[damped]
    if model_name not in results:
        pytest.skip(f"model {model_name} not implemented or not converging in R")

    results_R = results[model_name]

    model = ETSModel(
        austourists, seasonal_periods=4,
        error=error, trend=trend, seasonal=seasonal, damped=damped
    )
    fit = fit_austourists_with_R_params(model, results_R, set_state=True)

    innov = np.asarray([[1.76405235, 0.40015721, 0.97873798, 2.2408932]]).T
    sim = fit.simulate(4, repetitions=1, error=error, random_errors=innov)
    expected = np.asarray(results_R["simulation"])

    # should be the same up to 4 decimals
    assert_almost_equal(expected, sim.values, 4)


@pytest.mark.skip
def test_simulate_keywords(austourists):
    """
    check whether all keywords are accepted and work without throwing errors.
    """
    fit = ETSModel(
        austourists, seasonal_periods=4,
        error="add", trend="add", seasonal="add", damped=True
    ).fit()

    # test anchor
    assert_almost_equal(
        fit.simulate(4, anchor=0, random_state=0).values,
        fit.simulate(4, anchor="start", random_state=0).values
    )
    assert_almost_equal(
        fit.simulate(4, anchor=-1, random_state=0).values,
        fit.simulate(4, anchor="2015-12-01", random_state=0).values
    )
    assert_almost_equal(
        fit.simulate(4, anchor="end", random_state=0).values,
        fit.simulate(4, anchor="2016-03-01", random_state=0).values
    )

    # test different random error options
    fit.simulate(4, repetitions=10, random_errors=scipy.stats.norm)
    fit.simulate(4, repetitions=10, random_errors=scipy.stats.norm())

    fit.simulate(4, repetitions=10, random_errors=np.random.randn(4,10))
    fit.simulate(4, repetitions=10, random_errors="bootstrap")

    # test seeding
    res = fit.simulate(4, repetitions=10, random_state=10).values
    res2 = fit.simulate(
        4, repetitions=10, random_state=np.random.RandomState(10)
    ).values
    assert np.all(res == res2)


@pytest.mark.skip
def test_simulate_boxcox(austourists):
    """
    Check if simulation results with boxcox fits are reasonable.

    This test is related to issue #6629
    """
    fit = ETSModel(
        austourists, seasonal_periods=4,
        trend="add", seasonal="mul", damped=False
    ).fit(use_boxcox=True)
    expected = fit.forecast(4).values

    res = fit.simulate(4, repetitions=10, random_state=0).values
    mean = np.mean(res, axis=1)

    assert np.all(np.abs(mean - expected) < 5)
