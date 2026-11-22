"""
General tests for Markov switching models

Author: Chad Fulton
License: BSD-3
"""

from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest
from scipy import stats

from statsmodels.tools.numdiff import approx_fprime
from statsmodels.tools.sm_exceptions import ConvergenceWarning, MissingDataError
from statsmodels.tsa.regime_switching import (
    markov_autoregression,
    markov_regression,
    markov_switching,
)
from statsmodels.tsa.regime_switching.tests.test_markov_regression import fedfunds

RESULTS_PATH = Path(__file__).resolve().parent / "results"

# Stata 14 estimates of the two-regime switching mean model of fedfunds
# (https://www.stata.com/manuals14/tsmswitch.pdf, as in TestFedFundsConst):
# p[0->0], p[1->0], the two regime means and sigma2
FEDFUNDS_PARAMS = np.r_[0.9820939, 0.0503587, 3.70877, 9.556793, 2.107562**2]
# fedfunds is quarterly from 1954q3 to 2010q4
FEDFUNDS_INDEX = pd.date_range("1954-07-01", periods=len(fedfunds), freq="QS")


def test_params():
    def check_transtion_2(params):
        assert_equal(params["regime_transition"], np.s_[0:2])
        assert_equal(params[0, "regime_transition"], [0])
        assert_equal(params[1, "regime_transition"], [1])
        assert_equal(params["regime_transition", 0], [0])
        assert_equal(params["regime_transition", 1], [1])

    def check_transition_3(params):
        assert_equal(params["regime_transition"], np.s_[0:6])
        assert_equal(params[0, "regime_transition"], [0, 3])
        assert_equal(params[1, "regime_transition"], [1, 4])
        assert_equal(params[2, "regime_transition"], [2, 5])
        assert_equal(params["regime_transition", 0], [0, 3])
        assert_equal(params["regime_transition", 1], [1, 4])
        assert_equal(params["regime_transition", 2], [2, 5])

    params = markov_switching.MarkovSwitchingParams(k_regimes=2)
    params["regime_transition"] = [1]
    assert_equal(params.k_params, 1 * 2)
    assert_equal(params[0], [0])
    assert_equal(params[1], [1])
    check_transtion_2(params)

    params["exog"] = [0, 1]
    assert_equal(params.k_params, 1 * 2 + 1 + 1 * 2)
    assert_equal(params[0], [0, 2, 3])
    assert_equal(params[1], [1, 2, 4])
    check_transtion_2(params)
    assert_equal(params["exog"], np.s_[2:5])
    assert_equal(params[0, "exog"], [2, 3])
    assert_equal(params[1, "exog"], [2, 4])
    assert_equal(params["exog", 0], [2, 3])
    assert_equal(params["exog", 1], [2, 4])

    params = markov_switching.MarkovSwitchingParams(k_regimes=3)
    params["regime_transition"] = [1, 1]
    assert_equal(params.k_params, 2 * 3)
    assert_equal(params[0], [0, 3])
    assert_equal(params[1], [1, 4])
    assert_equal(params[2], [2, 5])
    check_transition_3(params)

    # Test for invalid parameter setting
    with pytest.raises(IndexError):
        params.__setitem__(None, [1, 1])

    # Test for invalid parameter selection
    with pytest.raises(IndexError):
        params.__getitem__(None)
    with pytest.raises(IndexError):
        params.__getitem__((0, 0))
    with pytest.raises(IndexError):
        params.__getitem__(("exog", "exog"))
    with pytest.raises(IndexError):
        params.__getitem__(("exog", 0, 1))


def test_init_endog():
    index = pd.date_range(start="1950-01-01", periods=10, freq="D")
    endog = [
        np.ones(10),
        pd.Series(np.ones(10), index=index),
        np.ones((10, 1)),
        pd.DataFrame(np.ones((10, 1)), index=index),
    ]
    for _endog in endog:
        mod = markov_switching.MarkovSwitching(_endog, k_regimes=2)
        assert_equal(mod.nobs, 10)
        assert_equal(mod.endog, _endog.squeeze())
        assert_equal(mod.k_regimes, 2)
        assert_equal(mod.tvtp, False)
        assert_equal(mod.k_tvtp, 0)
        assert_equal(mod.k_params, 2)

    # Invalid: k_regimes < 2
    endog = np.ones(10)
    with pytest.raises(ValueError):
        markov_switching.MarkovSwitching(endog, k_regimes=1)

    # Invalid: multiple endog columns
    endog = np.ones((10, 2))
    with pytest.raises(ValueError):
        markov_switching.MarkovSwitching(endog, k_regimes=2)


def test_init_exog_tvtp():
    endog = np.ones(10)
    exog_tvtp = np.c_[np.ones((10, 1)), (np.arange(10) + 1)[:, np.newaxis]]
    mod = markov_switching.MarkovSwitching(endog, k_regimes=2, exog_tvtp=exog_tvtp)
    assert_equal(mod.tvtp, True)
    assert_equal(mod.k_tvtp, 2)

    # Invalid exog_tvtp (too many obs)
    exog_tvtp = np.c_[np.ones((11, 1)), (np.arange(11) + 1)[:, np.newaxis]]
    with pytest.raises(ValueError):
        markov_switching.MarkovSwitching(endog, k_regimes=2, exog_tvtp=exog_tvtp)


@pytest.mark.parametrize(
    "model, kwargs",
    [
        (markov_switching.MarkovSwitching, {}),
        (markov_regression.MarkovRegression, {}),
        (markov_autoregression.MarkovAutoregression, {"order": 1}),
    ],
    ids=["MarkovSwitching", "MarkovRegression", "MarkovAutoregression"],
)
@pytest.mark.parametrize("missing", ["none", "drop", "raise"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("use_pandas", [False, True])
def test_init_exog_tvtp_not_finite(model, kwargs, missing, value, use_pandas):
    endog = np.arange(10.0)
    exog_tvtp = np.c_[np.ones((10, 1)), (np.arange(10) + 1)[:, np.newaxis]]
    exog_tvtp[3, 1] = value
    if use_pandas:
        exog_tvtp = pd.DataFrame(exog_tvtp, columns=["const", "x"])
    with pytest.raises(MissingDataError, match="exog_tvtp contains inf or nans"):
        model(endog, k_regimes=2, exog_tvtp=exog_tvtp, missing=missing, **kwargs)


def test_transition_matrix():
    # k_regimes = 2
    endog = np.ones(10)
    mod = markov_switching.MarkovSwitching(endog, k_regimes=2)
    params = np.r_[0.0, 0.0, 1.0]
    transition_matrix = np.zeros((2, 2, 1))
    transition_matrix[1, :] = 1.0
    assert_allclose(mod.regime_transition_matrix(params), transition_matrix)

    # k_regimes = 3
    endog = np.ones(10)
    mod = markov_switching.MarkovSwitching(endog, k_regimes=3)
    params = np.r_[[0] * 3, [0.2] * 3, 1.0]
    transition_matrix = np.zeros((3, 3, 1))
    transition_matrix[1, :, 0] = 0.2
    transition_matrix[2, :, 0] = 0.8
    assert_allclose(mod.regime_transition_matrix(params), transition_matrix)

    # k_regimes = 2, tvtp
    endog = np.ones(10)
    exog_tvtp = np.c_[np.ones((10, 1)), (np.arange(10) + 1)[:, np.newaxis]]
    mod = markov_switching.MarkovSwitching(endog, k_regimes=2, exog_tvtp=exog_tvtp)

    # If all TVTP regression coefficients are zero, then the logit transform
    # results in exp(0) / (1 + exp(0)) = 0.5 for all parameters; since it's
    # k_regimes=2 the remainder calculation is also 0.5.
    params = np.r_[0, 0, 0, 0]
    assert_allclose(mod.regime_transition_matrix(params), 0.5)

    # Manually compute the TVTP coefficients
    params = np.r_[1, 2, 1, 2]
    transition_matrix = np.zeros((2, 2, 10))

    coeffs0 = np.sum(exog_tvtp, axis=1)
    p11 = np.exp(coeffs0) / (1 + np.exp(coeffs0))
    transition_matrix[0, 0, :] = p11
    transition_matrix[1, 0, :] = 1 - p11

    coeffs1 = np.sum(2 * exog_tvtp, axis=1)
    p21 = np.exp(coeffs1) / (1 + np.exp(coeffs1))
    transition_matrix[0, 1, :] = p21
    transition_matrix[1, 1, :] = 1 - p21
    assert_allclose(mod.regime_transition_matrix(params), transition_matrix, atol=1e-10)

    # k_regimes = 3, tvtp
    endog = np.ones(10)
    exog_tvtp = np.c_[np.ones((10, 1)), (np.arange(10) + 1)[:, np.newaxis]]
    mod = markov_switching.MarkovSwitching(endog, k_regimes=3, exog_tvtp=exog_tvtp)

    # If all TVTP regression coefficients are zero, then the logit transform
    # results in exp(0) / (1 + exp(0) + exp(0)) = 1/3 for all parameters;
    # since it's k_regimes=3 the remainder calculation is also 1/3.
    params = np.r_[[0] * 12]
    assert_allclose(mod.regime_transition_matrix(params), 1 / 3)

    # Manually compute the TVTP coefficients for the first column
    params = np.r_[[0] * 6, [2] * 6]
    transition_matrix = np.zeros((3, 3, 10))

    p11 = np.zeros(10)
    p12 = 2 * np.sum(exog_tvtp, axis=1)
    tmp = np.exp(np.c_[p11, p12]).T
    transition_matrix[:2, 0, :] = tmp / (1 + np.sum(tmp, axis=0))
    transition_matrix[2, 0, :] = 1 - np.sum(transition_matrix[:2, 0, :], axis=0)
    assert_allclose(
        mod.regime_transition_matrix(params)[:, 0, :],
        transition_matrix[:, 0, :],
        atol=1e-10,
    )


def test_initial_probabilities():
    endog = np.ones(10)
    mod = markov_switching.MarkovSwitching(endog, k_regimes=2)
    params = np.r_[0.5, 0.5, 1.0]

    # Valid known initial probabilities
    mod.initialize_known([0.2, 0.8])
    assert_allclose(mod.initial_probabilities(params), [0.2, 0.8])

    # Invalid known initial probabilities (too many elements)
    with pytest.raises(ValueError):
        mod.initialize_known([0.2, 0.2, 0.6])

    # Invalid known initial probabilities (does not sum to 1)
    with pytest.raises(ValueError):
        mod.initialize_known([0.2, 0.2])

    # Valid steady-state probabilities
    mod.initialize_steady_state()
    assert_allclose(mod.initial_probabilities(params), [0.5, 0.5])

    # Invalid steady-state probabilities (when mod has tvtp)
    endog = np.ones(10)
    mod = markov_switching.MarkovSwitching(endog, k_regimes=2, exog_tvtp=endog)
    with pytest.raises(ValueError):
        mod.initialize_steady_state()


@pytest.mark.parametrize("convert", [list, tuple, pd.Series, np.asarray])
def test_initialize_known_array_like(convert):
    mod = markov_switching.MarkovSwitching(np.ones(10), k_regimes=3)
    mod.initialize_known(convert([0.2, 0.3, 0.5]))
    # Known initial probabilities do not depend on the parameters
    params = np.full(6, 0.3)
    assert_allclose(mod.initial_probabilities(params), [0.2, 0.3, 0.5], rtol=1e-15)


def test_initialize_known_copies_probabilities():
    # Changing the array afterwards does not change the model
    mod = markov_switching.MarkovSwitching(np.ones(10), k_regimes=2)
    probabilities = np.array([0.2, 0.8])
    mod.initialize_known(probabilities)
    probabilities[:] = [0.9, 0.1]
    params = np.r_[0.5, 0.5]
    assert_allclose(mod.initial_probabilities(params), [0.2, 0.8], rtol=1e-15)


@pytest.mark.parametrize(
    "probabilities, ndim",
    [([[0.2], [0.8]], 2), ([[0.2, 0.8]], 2), ([[[0.2, 0.8]]], 3)],
    ids=["column", "row", "3d"],
)
def test_initialize_known_not_1d_raises(probabilities, ndim):
    mod = markov_switching.MarkovSwitching(np.ones(10), k_regimes=2)
    match = f"probabilities must have 1 dimension, but has {ndim}"
    with pytest.raises(ValueError, match=match):
        mod.initialize_known(probabilities)


@pytest.mark.parametrize(
    "probabilities", [1.0, [0.2, 0.2, 0.6]], ids=["scalar", "too-long"]
)
def test_initialize_known_wrong_length_raises(probabilities):
    # A scalar is a vector with one element, which is too short
    mod = markov_switching.MarkovSwitching(np.ones(10), k_regimes=2)
    with pytest.raises(ValueError, match=r"a vector of shape \(k_regimes,\)"):
        mod.initialize_known(probabilities)


@pytest.mark.parametrize(
    "probabilities",
    [[0.2, 0.2], [0.2, 0.2, 0.6], [[0.2, 0.8]]],
    ids=["not-summing-to-one", "too-long", "2d"],
)
@pytest.mark.parametrize("previous", ["steady-state", "known"])
def test_initialize_known_failure_keeps_initialization(probabilities, previous):
    # The initialization was set to known before the checks, so that after
    # the ValueError initial_probabilities raised a TypeError (None >= 0)
    mod = markov_switching.MarkovSwitching(np.ones(10), k_regimes=2)
    params = np.r_[0.9, 0.2]
    if previous == "known":
        mod.initialize_known([0.3, 0.7])
        expected = [0.3, 0.7]
    else:
        # The steady state of p[0->0] = 0.9 and p[1->0] = 0.2 solves
        # 0.1 * pi_0 = 0.2 * pi_1
        expected = [2 / 3, 1 / 3]
    with pytest.raises(ValueError):
        mod.initialize_known(probabilities)
    assert_allclose(mod.initial_probabilities(params), expected, rtol=1e-12)


@pytest.mark.parametrize(
    "trend, switching_trend, switching_exog, coefficients",
    [
        # Only the coefficients that switch have a regime suffix
        ("n", True, [True, False], ["xa[0]", "xa[1]", "xb"]),
        ("n", True, [False, True], ["xa", "xb[0]", "xb[1]"]),
        ("c", False, [True], ["const", "x1[0]", "x1[1]"]),
        ("c", True, [False], ["const[0]", "const[1]", "x1"]),
        ("n", True, [True, True], ["xa[0]", "xa[1]", "xb[0]", "xb[1]"]),
        ("c", False, [False], ["const", "x1"]),
    ],
)
def test_param_names_mixed_switching_exog(
    trend, switching_trend, switching_exog, coefficients
):
    rng = np.random.default_rng(20261122)
    endog = rng.standard_normal(40)
    if trend == "n":
        exog = pd.DataFrame(rng.standard_normal((40, 2)), columns=["xa", "xb"])
    else:
        exog = rng.standard_normal(40)
    mod = markov_regression.MarkovRegression(
        endog,
        k_regimes=2,
        trend=trend,
        exog=exog,
        switching_trend=switching_trend,
        switching_exog=switching_exog,
    )
    assert mod.param_names == ["p[0->0]", "p[1->0]", *coefficients, "sigma2"]


@pytest.mark.parametrize(
    "switching_ar, ar_names",
    [
        ([True, False], ["ar.L1[0]", "ar.L1[1]", "ar.L2"]),
        ([False, True], ["ar.L1", "ar.L2[0]", "ar.L2[1]"]),
        ([True, False, True], ["ar.L1[0]", "ar.L1[1]", "ar.L2", "ar.L3[0]", "ar.L3[1]"]),
        ([True, True], ["ar.L1[0]", "ar.L1[1]", "ar.L2[0]", "ar.L2[1]"]),
        ([False, False], ["ar.L1", "ar.L2"]),
    ],
)
def test_param_names_mixed_switching_ar(switching_ar, ar_names):
    endog = np.random.default_rng(20261122).standard_normal(40)
    mod = markov_autoregression.MarkovAutoregression(
        endog, k_regimes=2, order=len(switching_ar), switching_ar=switching_ar
    )
    expected = ["p[0->0]", "p[1->0]", "const[0]", "const[1]", "sigma2", *ar_names]
    assert mod.param_names == expected


def hamilton_filter_log_reference(log_densities, transition):
    """
    Hamilton (1989) filter written out for a likelihood that depends only on
    the current regime: log_densities[i, t] = log f(y_t | S_t = i, Y_{t-1})
    and transition[i, j] = Pr[S_t = i | S_{t-1} = j]. The regime before the
    first observation has the stationary distribution. Each log f(y_t |
    Y_{t-1}) is a log-sum-exp shifted by its largest term.
    """
    k_regimes, nobs = log_densities.shape
    lhs = np.vstack([np.eye(k_regimes) - transition, np.ones((1, k_regimes))])
    filtered = np.linalg.lstsq(lhs, np.r_[np.zeros(k_regimes), 1.0], rcond=None)[0]
    llf_obs = np.empty(nobs)
    filtered_probabilities = np.empty((k_regimes, nobs))
    for t in range(nobs):
        log_joint = np.log(transition @ filtered) + log_densities[:, t]
        largest = log_joint.max()
        llf_obs[t] = largest + np.log(np.exp(log_joint - largest).sum())
        filtered = np.exp(log_joint - llf_obs[t])
        filtered_probabilities[:, t] = filtered
    return llf_obs, filtered_probabilities


class GivenLoglikelihoods(markov_switching.MarkovSwitching):
    # A model whose log-likelihood depends only on the current regime
    # (order 0), given as an array shaped (k_regimes, nobs)
    def __init__(self, conditional_loglikelihoods):
        self._given = conditional_loglikelihoods
        super().__init__(np.zeros(conditional_loglikelihoods.shape[1]), k_regimes=2)

    def _conditional_loglikelihoods(self, params):
        return self._given


@pytest.mark.parametrize("model", ["regression", "order 0"])
def test_hamilton_filter_extreme_observation(model):
    # Observation 10 is about 200 standard deviations from both regime means,
    # so its log density is about -2e4 in both regimes: the filter must not
    # underflow to a log-likelihood of -inf
    rng = np.random.default_rng(20261122)
    endog = rng.standard_normal(20) + np.repeat([0.0, 3.0], 10)
    endog[10] = 200.0
    transition = np.array([[0.9, 0.2], [0.1, 0.8]])
    mu = np.array([0.2, 2.8])
    log_densities = -0.5 * (endog - mu[:, None]) ** 2 - 0.5 * np.log(2 * np.pi)
    if model == "regression":
        mod = markov_regression.MarkovRegression(endog, k_regimes=2)
        params = np.r_[transition[0], mu, 1.0]
    else:
        mod = GivenLoglikelihoods(log_densities)
        params = transition[0]
    llf_obs, filtered = hamilton_filter_log_reference(log_densities, transition)
    assert llf_obs[10] < -1.9e4

    res = mod.smooth(params, cov_type="none")
    assert_allclose(res.llf_obs, llf_obs, rtol=1e-12)
    assert_allclose(res.llf, llf_obs.sum(), rtol=1e-12)
    assert_allclose(res.filtered_marginal_probabilities, filtered.T, rtol=1e-10)
    # The smoothed probabilities are probabilities
    smoothed = res.smoothed_marginal_probabilities
    assert np.all((smoothed >= 0) & (smoothed <= 1))
    assert_allclose(smoothed.sum(axis=1), 1.0, rtol=1e-12)


@pytest.mark.parametrize("display_params", [True, False])
def test_summary_display_params(display_params):
    # Switching constant, common variance: two regime tables, one table of
    # non-switching parameters and one of transition parameters
    rng = np.random.default_rng(20261122)
    endog = rng.standard_normal(40) + np.repeat([0.0, 3.0, 0.0, 3.0], 10)
    mod = markov_regression.MarkovRegression(endog, k_regimes=2)
    res = mod.smooth(np.r_[0.9, 0.2, 0.2, 2.8, 1.0])
    summary = res.summary(display_params=display_params)
    titles = [table.title for table in summary.tables]
    expected = ["Markov Switching Model Results"]
    if display_params:
        expected += [
            "Regime 0 parameters",
            "Regime 1 parameters",
            "Non-switching parameters",
            "Regime transition parameters",
        ]
    assert titles == expected
    text = summary.as_text()
    for name in ["const", "sigma2", "p[0->0]", "p[1->0]"]:
        assert (name in text) is display_params
    # The covariance note is kept
    assert "Covariance matrix calculated using the outer product" in text


@pytest.mark.parametrize("kind", ["smoothed", "filtered", "predicted"])
@pytest.mark.parametrize("as_list", [False, True])
def test_predict_with_probabilities_array(kind, as_list):
    rng = np.random.default_rng(20261122)
    endog = rng.standard_normal(40) + np.repeat([0.0, 3.0, 0.0, 3.0], 10)
    mu = np.array([0.2, 2.8])
    phi = np.array([0.5, -0.2])

    # MarkovRegression: sum_i mu_i Pr[S_t = i]
    mod = markov_regression.MarkovRegression(endog, k_regimes=2)
    params = np.r_[0.9, 0.2, mu, 1.0]
    res = mod.smooth(params, cov_type="none")
    joint = getattr(res, f"{kind}_joint_probabilities")
    marginal = getattr(res, f"{kind}_marginal_probabilities")
    probabilities = joint.tolist() if as_list else joint
    predict = mod.predict(params, probabilities=probabilities)
    assert_allclose(predict, marginal @ mu, rtol=1e-12)
    assert_allclose(predict, mod.predict(params, probabilities=kind), rtol=1e-13)

    # AR(1) MarkovAutoregression: the weighted average over (S_t, S_{t-1})
    # = (i, j) of mu_i + phi_i (y_{t-1} - mu_j); prediction n is for
    # endog[n + 1], so its lag is endog[n]
    mod = markov_autoregression.MarkovAutoregression(endog, k_regimes=2, order=1)
    params = np.r_[0.9, 0.2, mu, 1.0, phi]
    res = mod.smooth(params, cov_type="none")
    joint = getattr(res, f"{kind}_joint_probabilities")
    means = mu[:, None, None] + phi[:, None, None] * (
        endog[:-1] - mu[None, :, None]
    )
    probabilities = joint.tolist() if as_list else joint
    predict = mod.predict(params, probabilities=probabilities)
    assert_allclose(predict, (means * joint).sum(axis=(0, 1)), rtol=1e-12)
    assert_allclose(predict, mod.predict(params, probabilities=kind), rtol=1e-13)


def test_parameters_getitem_numpy_integer():
    # NumPy integers are accepted as regime indices in the parameters object
    endog = np.random.default_rng(0).standard_normal(100)
    mod = markov_regression.MarkovRegression(
        endog, k_regimes=2, trend="c", switching_variance=True
    )
    params = mod.parameters
    for key in [np.int64(0), np.int32(1)]:
        assert_equal(params[key], params[int(key)])
        assert_equal(
            params[key, "regime_transition"], params[int(key), "regime_transition"]
        )
        assert_equal(
            params["regime_transition", key], params["regime_transition", int(key)]
        )


@pytest.mark.parametrize(
    "flag, expected", [(np.True_, True), (np.False_, False), (1, True), (0, False)]
)
@pytest.mark.parametrize("name", ["switching_trend", "switching_exog", "switching_ar"])
def test_scalar_switching_flags(flag, expected, name):
    # A numpy bool or a 0/1 integer is the same as the Python bool
    rng = np.random.default_rng(20261122)
    endog = rng.standard_normal(40)
    exog = rng.standard_normal((40, 2))
    if name == "switching_ar":
        model = markov_autoregression.MarkovAutoregression
        kwargs = {"order": 2}
    else:
        model = markov_regression.MarkovRegression
        kwargs = {"exog": exog}
    mod = model(endog, k_regimes=2, **{name: flag}, **kwargs)
    mod_bool = model(endog, k_regimes=2, **{name: expected}, **kwargs)
    assert getattr(mod, name) == getattr(mod_bool, name)
    # The flags are stored as Python bools, as for the bool argument
    assert all(value is expected for value in getattr(mod, name))
    assert mod.param_names == mod_bool.param_names
    params = mod_bool.start_params
    assert_allclose(mod.loglikeobs(params), mod_bool.loglikeobs(params), rtol=1e-13)


@pytest.mark.parametrize("order", [0, 1])
def test_results_predict_conditional_with_pandas_data(order):
    rng = np.random.default_rng(20261122)
    endog = rng.standard_normal(40) + np.repeat([0.0, 3.0, 0.0, 3.0], 10)
    index = pd.date_range("2000-01-01", periods=40, freq="MS")
    mu = np.array([0.2, 2.8])
    phi = np.array([0.5, -0.2])
    if order == 0:
        model, kwargs = markov_regression.MarkovRegression, {}
        params = np.r_[0.9, 0.2, mu, 1.0]
        # Conditional on S_t = i, the prediction is mu_i
        conditional = True
        expected = np.repeat(mu[:, None], 40, axis=1)
    else:
        model, kwargs = markov_autoregression.MarkovAutoregression, {"order": 1}
        params = np.r_[0.9, 0.2, mu, 1.0, phi]
        # Conditional on (S_t, S_{t-1}) = (i, j): mu_i + phi_i (y_{t-1} - mu_j)
        conditional = 2
        expected = mu[:, None, None] + phi[:, None, None] * (
            endog[:-1] - mu[None, :, None]
        )
    res = model(pd.Series(endog, index=index), k_regimes=2, **kwargs).smooth(
        params, cov_type="none"
    )

    # Conditional predictions are an ndarray, as for an ndarray endog
    predict = res.predict(conditional=conditional)
    assert type(predict) is np.ndarray
    assert_allclose(predict, expected, rtol=1e-13)

    # Unconditional predictions are a Series indexed by the dates, with the
    # values for an ndarray endog
    res_array = model(endog, k_regimes=2, **kwargs).smooth(params, cov_type="none")
    predict = res.predict()
    assert isinstance(predict, pd.Series)
    pd.testing.assert_index_equal(predict.index, index[order:])
    assert_allclose(predict, res_array.predict(), rtol=1e-13)


def transform_params_models():
    rng = np.random.default_rng(20261122)
    endog = rng.standard_normal(40)
    exog_tvtp = np.column_stack([np.ones(40), rng.standard_normal(40)])
    return {
        "base": markov_switching.MarkovSwitching(endog, k_regimes=2),
        "base-3": markov_switching.MarkovSwitching(endog, k_regimes=3),
        "base-tvtp": markov_switching.MarkovSwitching(
            endog, k_regimes=2, exog_tvtp=exog_tvtp
        ),
        "regression": markov_regression.MarkovRegression(
            endog, k_regimes=2, switching_variance=True
        ),
        "autoregression": markov_autoregression.MarkovAutoregression(
            endog, k_regimes=2, order=1
        ),
    }


def transform_params_reference(name, unconstrained):
    # Closed forms of the transformations: the multinomial logistic for the
    # transition probabilities (unless they are time-varying), the identity
    # for regression coefficients, squares for variances and -u / sqrt(1 + u^2)
    # for an AR(1) coefficient (constrain_stationary_univariate)
    u = np.asarray(unconstrained, dtype=float)
    if name == "base-tvtp":
        return u.copy()
    if name == "base-3":
        # p[j->0], p[j->1] of regime j are u[j] and u[3 + j]
        out = u.copy()
        for j in range(3):
            idx = [j, 3 + j]
            out[idx] = np.exp(u[idx]) / (1 + np.exp(u[idx]).sum())
        return out
    out = u.copy()
    out[:2] = np.exp(u[:2]) / (1 + np.exp(u[:2]))
    if name == "regression":
        # p[0->0], p[1->0], const[0], const[1], sigma2[0], sigma2[1]
        out[4:6] = u[4:6] ** 2
    elif name == "autoregression":
        # p[0->0], p[1->0], const[0], const[1], sigma2, ar.L1[0], ar.L1[1]
        out[4] = u[4] ** 2
        out[5:7] = -u[5:7] / np.sqrt(1 + u[5:7] ** 2)
    return out


@pytest.mark.parametrize(
    "name", ["base", "base-3", "base-tvtp", "regression", "autoregression"]
)
@pytest.mark.parametrize("form", [list, tuple, pd.Series, np.asarray])
def test_transform_params_array_like(name, form):
    mod = transform_params_models()[name]
    unconstrained = np.linspace(-1.2, 0.9, mod.k_params)
    expected = transform_params_reference(name, unconstrained)
    constrained = mod.transform_params(form(unconstrained.tolist()))
    assert type(constrained) is np.ndarray
    assert_allclose(constrained, expected, rtol=1e-13)
    # untransform_params is the inverse; the root finder used with three
    # regimes is accurate to its own tolerance
    rtol = 1e-6 if name == "base-3" else 1e-12
    restored = mod.untransform_params(form(constrained.tolist()))
    assert type(restored) is np.ndarray
    assert_allclose(restored, unconstrained, rtol=rtol)
class ParamsFreeLoglikelihoods(markov_switching.MarkovSwitching):
    # Conditional log-likelihoods that do not depend on the parameters, so
    # they stay real when complex-step differentiation makes the parameters,
    # and so the transition probabilities, complex
    def __init__(self, conditional_loglikelihoods):
        self._given = conditional_loglikelihoods
        nobs = conditional_loglikelihoods.shape[-1]
        super().__init__(np.zeros(nobs), k_regimes=2)

    def _conditional_loglikelihoods(self, params):
        return self._given


@pytest.mark.parametrize("lagged", [False, True])
def test_complex_transition_with_real_loglikelihoods(lagged):
    from statsmodels.tools.numdiff import approx_fprime, approx_hess3

    # Log-likelihoods that depend on the current regime only, shaped
    # (k_regimes, nobs), or also on the previous one, (k_regimes,) * 2 + (nobs,)
    rng = np.random.default_rng(20261122)
    shape = (2, 2, 30) if lagged else (2, 30)
    mod = ParamsFreeLoglikelihoods(np.log(rng.uniform(0.1, 1.0, shape)))
    params = np.r_[0.9, 0.2]

    # Complex-step derivatives against central differences of the real
    # log-likelihood, to the accuracy of the central differences
    score_obs = approx_fprime(params, mod.loglikeobs, centered=True)
    assert_allclose(mod.score_obs(params), score_obs, rtol=1e-6)
    assert_allclose(mod.score(params), score_obs.sum(0), rtol=1e-6)
    hessian = approx_hess3(params, mod.loglike)
    assert_allclose(mod.hessian(params), hessian, rtol=1e-4)

    # The default covariance matrix of smooth, the outer product of gradients
    res = mod.smooth(params)
    assert_allclose(
        res.cov_params(), np.linalg.inv(score_obs.T @ score_obs), rtol=1e-5
    )
    assert np.isrealobj(res.filtered_marginal_probabilities)

    # Complex log-likelihoods with real transition probabilities give the
    # same values, with a zero imaginary part
    complex_mod = ParamsFreeLoglikelihoods(mod._given.astype(complex))
    llf_obs = complex_mod.loglikeobs(params)
    assert np.iscomplexobj(llf_obs)
    assert_allclose(llf_obs, mod.loglikeobs(params), rtol=1e-13)


def _int_like_params():
    # Two switching transition parameters, then a non-switching and a
    # switching exog parameter
    params = markov_switching.MarkovSwitchingParams(k_regimes=2)
    params["regime_transition"] = [1, 1]
    params["exog"] = [0, 1]
    return params


@pytest.mark.parametrize(
    "to_key", [int, np.int64, np.int32, np.array], ids=["int", "int64", "int32", "0d"]
)
def test_params_int_like_regime(to_key):
    # A NumPy integer or a 0-d integer array is a regime number, as an int
    # is. A 0-d array raised an IndexError.
    params = _int_like_params()
    assert_equal(params[to_key(0)], [0, 2, 4, 5])
    assert_equal(params[to_key(1)], [1, 3, 4, 6])
    assert_equal(params[to_key(0), "exog"], [4, 5])
    assert_equal(params["exog", to_key(1)], [4, 6])
    assert_equal(params[to_key(1), "regime_transition"], [1, 3])


@pytest.mark.parametrize(
    "key", [True, np.True_, 1.0], ids=["True", "np.True_", "float"]
)
def test_params_bool_float_regime(key):
    # A bool or a float is not a regime number: True was regime 1 in a tuple
    params = _int_like_params()
    with pytest.raises(IndexError, match="Invalid index"):
        params[key]
    with pytest.raises(IndexError, match="Invalid index"):
        params[key, "exog"]
    with pytest.raises(IndexError, match="Invalid index"):
        params["exog", key]


@pytest.mark.parametrize("k_regimes", [2, np.int64(2)], ids=["int", "int64"])
def test_params_counts_are_python_ints(k_regimes):
    # np.sum made k_parameters, k_params and the slice bounds NumPy integers,
    # which print as np.int64(4) with NumPy 2
    params = markov_switching.MarkovSwitchingParams(k_regimes=k_regimes)
    params["regime_transition"] = [1, 1]
    params["exog"] = [0, 1]
    assert params.k_parameters == {"regime_transition": 4, "exog": 3}
    assert params.k_params == 7
    assert params["exog"] == slice(4, 7)
    values = [*params.k_parameters.values(), params.k_params]
    for block in (params["regime_transition"], params["exog"]):
        values += [block.start, block.stop]
    assert all(type(value) is int for value in values)
    assert repr(params["exog"]) == "slice(4, 7, None)"
    assert repr(params.k_parameters["exog"]) == "3"


def test_model_k_params_is_python_int():
    endog = np.random.default_rng(0).standard_normal(50)
    mod = markov_regression.MarkovRegression(endog, k_regimes=2)
    # 2 transition probabilities, 2 switching constants and 1 variance
    assert mod.k_params == 5
    assert type(mod.k_params) is int


def test_cov_kwds_not_changed():
    rng = np.random.default_rng(20261122)
    endog = rng.standard_normal(40) + np.repeat([0.0, 3.0, 0.0, 3.0], 10)
    mod = markov_regression.MarkovRegression(endog, k_regimes=2)
    params = np.r_[0.9, 0.2, 0.2, 2.8, 1.0]
    cov_kwds = {"approx_complex_step": True, "approx_centered": False}
    res = mod.smooth(params, cov_type="approx", cov_kwds=cov_kwds)
    assert cov_kwds == {"approx_complex_step": True, "approx_centered": False}
    # The defaults: complex-step differentiation
    assert_allclose(res.cov_params(), mod.smooth(params).cov_params_approx)
    assert "complex-step" in res.cov_kwds["description"]


@pytest.mark.parametrize("cov_type", ["approx", "opg", "robust"])
@pytest.mark.parametrize("centered", [False, True])
def test_cov_kwds_finite_differences(cov_type, centered):
    from statsmodels.tools.numdiff import approx_fprime, approx_hess1, approx_hess3

    rng = np.random.default_rng(20261122)
    endog = rng.standard_normal(40) + np.repeat([0.0, 3.0, 0.0, 3.0], 10)
    mod = markov_regression.MarkovRegression(endog, k_regimes=2)
    params = np.r_[0.9, 0.2, 0.2, 2.8, 1.0]
    cov_kwds = {"approx_complex_step": False, "approx_centered": centered}
    res = mod.smooth(params, cov_type=cov_type, cov_kwds=cov_kwds)
    assert cov_kwds == {"approx_complex_step": False, "approx_centered": centered}

    # The covariance matrix from finite-difference derivatives: forward
    # differences, or centered ones
    approx_hess = approx_hess3 if centered else approx_hess1
    hessian = approx_hess(params, mod.loglike)
    scores = approx_fprime(params, mod.loglikeobs, centered=centered)
    opg = np.linalg.pinv(scores.T @ scores)
    expected = {
        "approx": -np.linalg.pinv(hessian),
        "opg": opg,
        "robust": np.linalg.pinv(hessian @ opg @ hessian),
    }[cov_type]
    assert_allclose(res.cov_params(), expected, rtol=1e-8)
    method = "centered finite differences" if centered else "finite differences"
    assert f"({method})" in res.cov_kwds["description"]

    # The finite differences approximate the complex-step derivatives, up to
    # their truncation error
    cov_cs = mod.smooth(params, cov_type=cov_type).cov_params()
    scale = np.sqrt(np.abs(np.outer(np.diag(cov_cs), np.diag(cov_cs))))
    assert_allclose(res.cov_params() / scale, cov_cs / scale, atol=1e-2)


def _ergodic_probabilities(transition):
    """Ergodic probabilities from the eigenvector of the unit eigenvalue"""
    eigval, eigvec = np.linalg.eig(transition)
    vec = np.real(eigvec[:, np.argmin(np.abs(eigval - 1))])
    return vec / vec.sum()


def _hamilton_kim(transition, initial, densities):
    """
    Hamilton filter and Kim smoother in probability space

    transition[i, j, t] is Pr[S_t = i | S_{t-1} = j], initial is the
    distribution of S_{-1} and densities[i, t] is f(y_t | S_t = i).
    """
    k_regimes, nobs = densities.shape
    predicted = np.zeros((k_regimes, nobs))
    filtered = np.zeros((k_regimes, nobs))
    llf_obs = np.zeros(nobs)
    previous = np.asarray(initial)
    for t in range(nobs):
        predicted[:, t] = transition[:, :, t] @ previous
        joint = densities[:, t] * predicted[:, t]
        llf_obs[t] = np.log(joint.sum())
        filtered[:, t] = joint / joint.sum()
        previous = filtered[:, t]
    # Kim (1994): Pr[S_t = j | Y_T] = Pr[S_t = j | Y_t] *
    #   sum_i Pr[S_{t+1} = i | S_t = j] Pr[S_{t+1} = i | Y_T] / Pr[S_{t+1} = i | Y_t]
    smoothed = np.zeros((k_regimes, nobs))
    smoothed[:, -1] = filtered[:, -1]
    for t in range(nobs - 2, -1, -1):
        ratio = smoothed[:, t + 1] / predicted[:, t + 1]
        smoothed[:, t] = filtered[:, t] * (transition[:, :, t + 1].T @ ratio)
    return predicted, filtered, llf_obs, smoothed


def _fedfunds_densities(means, sigma2):
    return stats.norm.pdf(
        np.asarray(fedfunds), loc=np.asarray(means)[:, None], scale=np.sqrt(sigma2)
    )


@pytest.fixture(scope="module")
def stata_fedfunds():
    return pd.read_csv(RESULTS_PATH / "results_predict_fedfunds.csv")


@pytest.mark.parametrize("use_pandas", [False, True])
def test_filter_smoother_fixed_params(use_pandas, stata_fedfunds):
    # Hamilton filter and Kim smoother at Stata's estimates, against Stata's
    # predicted, filtered and smoothed probabilities and against the
    # recursions written out in probability space
    endog = np.asarray(fedfunds)
    if use_pandas:
        endog = pd.Series(endog, index=FEDFUNDS_INDEX)
    mod = markov_regression.MarkovRegression(endog, k_regimes=2)
    res = mod.smooth(FEDFUNDS_PARAMS)
    nobs = len(fedfunds)

    p00, p10 = FEDFUNDS_PARAMS[:2]
    p11 = 1 - p10
    transition = np.array([[p00, p10], [1 - p00, p11]])
    # Ergodic probabilities of a two-regime chain
    ergodic = np.r_[1 - p11, 1 - p00] / (2 - p00 - p11)
    assert_allclose(mod.initial_probabilities(FEDFUNDS_PARAMS), ergodic, rtol=1e-12)
    assert_allclose(res.initial_probabilities, ergodic, rtol=1e-12)

    densities = _fedfunds_densities(FEDFUNDS_PARAMS[2:4], FEDFUNDS_PARAMS[4])
    tvtransition = np.repeat(transition[:, :, None], nobs, axis=2)
    predicted, filtered, llf_obs, smoothed = _hamilton_kim(
        tvtransition, ergodic, densities
    )
    assert_allclose(np.asarray(res.predicted_marginal_probabilities), predicted.T)
    assert_allclose(np.asarray(res.filtered_marginal_probabilities), filtered.T)
    assert_allclose(np.asarray(res.smoothed_marginal_probabilities), smoothed.T)
    assert_allclose(res.llf_obs, llf_obs, rtol=1e-10)
    assert_allclose(res.joint_loglikelihoods, llf_obs, rtol=1e-10)
    assert_allclose(res.joint_likelihoods, np.exp(llf_obs), rtol=1e-10)
    assert_allclose(mod.loglikeobs(FEDFUNDS_PARAMS), llf_obs, rtol=1e-10)
    assert_allclose(res.llf, llf_obs.sum(), rtol=1e-12)
    assert_allclose(mod.loglike(FEDFUNDS_PARAMS), llf_obs.sum(), rtol=1e-12)

    # The joint probabilities of (S_t, S_{t-1})
    previous = np.column_stack([ergodic, filtered[:, :-1]])
    joint = transition[:, :, None] * previous[None, :, :]
    assert_allclose(res.predicted_joint_probabilities, joint)
    assert_allclose(
        res.filtered_joint_probabilities,
        densities[:, None, :] * joint / np.exp(llf_obs),
    )
    assert_allclose(
        res.smoothed_joint_probabilities,
        (smoothed / predicted)[:, None, :] * joint,
    )

    # Stata's values are rounded to 7 decimals
    for i in range(2):
        assert_allclose(
            np.asarray(res.predicted_marginal_probabilities)[:, i],
            stata_fedfunds[f"const_p{i + 1}"],
            atol=1e-6,
        )
        assert_allclose(
            np.asarray(res.filtered_marginal_probabilities)[:, i],
            stata_fedfunds[f"const_f{i + 1}"],
            atol=1e-6,
        )

    # Information criteria and expected durations
    k = len(FEDFUNDS_PARAMS)
    assert_allclose(res.aic, -2 * res.llf + 2 * k, rtol=1e-12)
    assert_allclose(res.bic, -2 * res.llf + k * np.log(nobs), rtol=1e-12)
    assert_allclose(res.hqic, -2 * res.llf + 2 * k * np.log(np.log(nobs)), rtol=1e-12)
    assert isinstance(res.expected_durations, np.ndarray)
    assert_allclose(res.expected_durations, 1 / (1 - np.r_[p00, p11]), rtol=1e-12)

    if use_pandas:
        for name in ["predicted", "filtered", "smoothed"]:
            probs = getattr(res, f"{name}_marginal_probabilities")
            assert isinstance(probs, pd.DataFrame)
            assert probs.index.equals(FEDFUNDS_INDEX)


@pytest.mark.parametrize("use_pandas", [False, True])
def test_filter_smoother_tvtp_fixed_params(use_pandas):
    # Time-varying transition probabilities, against the recursions written
    # out with the logistic transition probabilities
    nobs = len(fedfunds)
    exog_tvtp = np.column_stack([np.ones(nobs), np.sin(np.arange(nobs) / 8)])
    endog = np.asarray(fedfunds)
    if use_pandas:
        endog = pd.Series(endog, index=FEDFUNDS_INDEX)
        exog_tvtp = pd.DataFrame(exog_tvtp, index=FEDFUNDS_INDEX)
    mod = markov_regression.MarkovRegression(endog, k_regimes=2, exog_tvtp=exog_tvtp)
    assert mod.param_names[:4] == [
        "p[0->0].tvtp0",
        "p[1->0].tvtp0",
        "p[0->0].tvtp1",
        "p[1->0].tvtp1",
    ]
    params = np.r_[4.0, -3.0, 1.0, 0.5, FEDFUNDS_PARAMS[2:]]
    res = mod.smooth(params)

    x = np.asarray(exog_tvtp)
    stay0 = 1 / (1 + np.exp(-(x @ params[[0, 2]])))
    move1 = 1 / (1 + np.exp(-(x @ params[[1, 3]])))
    transition = np.array([[stay0, move1], [1 - stay0, 1 - move1]])
    assert_allclose(mod.regime_transition_matrix(params), transition, rtol=1e-10)

    # Without a known initialization, the initial probabilities are the
    # ergodic probabilities of the first transition matrix
    initial = _ergodic_probabilities(transition[..., 0])
    assert_allclose(res.initial_probabilities, initial, rtol=1e-10)
    densities = _fedfunds_densities(params[4:6], params[6])
    predicted, filtered, llf_obs, smoothed = _hamilton_kim(
        transition, initial, densities
    )
    assert_allclose(np.asarray(res.predicted_marginal_probabilities), predicted.T)
    assert_allclose(np.asarray(res.filtered_marginal_probabilities), filtered.T)
    assert_allclose(np.asarray(res.smoothed_marginal_probabilities), smoothed.T)
    assert_allclose(res.llf_obs, llf_obs, rtol=1e-10)

    expected_durations = 1 / (1 - np.column_stack([stay0, 1 - move1]))
    assert_allclose(np.asarray(res.expected_durations), expected_durations)
    if use_pandas:
        assert isinstance(res.expected_durations, pd.DataFrame)
        assert res.expected_durations.index.equals(FEDFUNDS_INDEX)
    else:
        assert isinstance(res.expected_durations, np.ndarray)


@pytest.mark.parametrize("tvtp", [False, True])
def test_cy_filter_smoother_order_zero(tvtp):
    # The module-level filter and smoother with an order-0 likelihood: the
    # initial probabilities are those of the period before the first
    # observation, and the smoother takes the log probabilities
    rng = np.random.default_rng(8126)
    nobs = 25
    densities = rng.uniform(0.05, 1.0, size=(2, nobs))
    initial = np.r_[0.3, 0.7]
    if tvtp:
        stay = rng.uniform(0.6, 0.95, size=(2, nobs))
    else:
        stay = np.repeat([[0.9], [0.8]], nobs, axis=1)
    transition = np.array([[stay[0], 1 - stay[1]], [1 - stay[0], stay[1]]])
    regime_transition = transition if tvtp else transition[..., :1]

    out = markov_switching.cy_hamilton_filter_log(
        initial, regime_transition, np.log(densities), model_order=0
    )
    predicted, filtered, llf_obs, smoothed = _hamilton_kim(
        transition, initial, densities
    )
    assert_allclose(out[0], filtered)
    assert_allclose(out[1], predicted)
    assert_allclose(out[2], llf_obs, rtol=1e-10)
    assert_allclose(out[3], filtered)
    assert_allclose(out[4], np.log(predicted), rtol=1e-10)
    assert_allclose(out[5], np.log(filtered), rtol=1e-10)

    smoothed_joint, smoothed_marginal = markov_switching.cy_kim_smoother_log(
        regime_transition, out[4], out[5]
    )
    assert_allclose(smoothed_joint, smoothed)
    assert_allclose(smoothed_marginal, smoothed)


class FixedLikelihoods(markov_switching.MarkovSwitching):
    """Markov switching model with given conditional loglikelihoods"""

    def __init__(self, conditional_loglikelihoods, k_regimes):
        self._fixed_loglikelihoods = np.asarray(conditional_loglikelihoods)
        nobs = self._fixed_loglikelihoods.shape[-1]
        super().__init__(np.zeros(nobs), k_regimes)

    def _conditional_loglikelihoods(self, params):
        return self._fixed_loglikelihoods


def test_base_class_results_known_initialization():
    # A subclass that only provides the conditional loglikelihoods gets the
    # base results class, and its parameters are the transition
    # probabilities only
    rng = np.random.default_rng(31415)
    nobs = 30
    densities = rng.uniform(0.05, 1.0, size=(2, nobs))
    mod = FixedLikelihoods(np.log(densities), k_regimes=2)
    mod.initialize_known([0.3, 0.7])
    params = np.r_[0.9, 0.2]
    res = mod.smooth(params, cov_type="none")
    assert type(res) is markov_switching.MarkovSwitchingResultsWrapper
    assert type(res._results) is markov_switching.MarkovSwitchingResults

    transition = np.repeat([[[0.9], [0.2]], [[0.1], [0.8]]], nobs, axis=2)
    predicted, filtered, llf_obs, smoothed = _hamilton_kim(
        transition, np.r_[0.3, 0.7], densities
    )
    assert_allclose(res.initial_probabilities, [0.3, 0.7])
    assert_allclose(res.predicted_marginal_probabilities, predicted.T)
    assert_allclose(res.filtered_marginal_probabilities, filtered.T)
    assert_allclose(res.smoothed_marginal_probabilities, smoothed.T)
    assert_allclose(res.llf, llf_obs.sum(), rtol=1e-12)

    titles = [table.title for table in res.summary().tables]
    assert titles == ["Markov Switching Model Results", "Regime transition parameters"]


@pytest.mark.parametrize("k_regimes", [2, 3])
def test_initial_probabilities_ergodic(k_regimes):
    if k_regimes == 2:
        transition = np.array([[0.9, 0.3], [0.1, 0.7]])
        # (1 - p11, 1 - p00) / (2 - p00 - p11)
        ergodic = np.r_[0.3, 0.1] / 0.4
    else:
        transition = np.array([[0.8, 0.1, 0.3], [0.15, 0.7, 0.2], [0.05, 0.2, 0.5]])
        ergodic = _ergodic_probabilities(transition)
    assert_allclose(transition @ ergodic, ergodic)

    mod = markov_switching.MarkovSwitching(np.zeros(10), k_regimes=k_regimes)
    # The parameters are Pr[S_t = i | S_{t-1} = j] for i < k_regimes - 1
    params = transition[:-1].ravel()
    assert_allclose(mod.regime_transition_matrix(params)[..., 0], transition)
    assert_allclose(mod.initial_probabilities(params), ergodic, rtol=1e-12)
    # A given transition matrix, two-dimensional or three-dimensional (where
    # the first period is used), replaces the one implied by params
    other = np.full(len(params), 1 / k_regimes)
    for regime_transition in [
        transition,
        transition[..., None],
        np.stack([transition, np.full_like(transition, 1 / k_regimes)], axis=-1),
    ]:
        actual = mod.initial_probabilities(other, regime_transition=regime_transition)
        assert_allclose(actual, ergodic, rtol=1e-12)

    # Known initial probabilities ignore the transition matrix
    mod.initialize_known(list(ergodic[::-1]))
    assert_allclose(mod.initial_probabilities(params), ergodic[::-1])


@pytest.mark.parametrize("form", ["ndarray", "list", "DataFrame"])
def test_regime_transition_matrix_new_exog_tvtp(form):
    nobs = 20
    exog_tvtp = np.column_stack([np.ones(nobs), np.linspace(-1, 1, nobs)])
    mod = markov_switching.MarkovSwitching(
        np.zeros(nobs), k_regimes=3, exog_tvtp=exog_tvtp
    )
    params = np.linspace(-1.5, 2.0, 12)
    # Coefficients of Pr[S_t = i | S_{t-1} = j] for i < 2 on the columns of
    # exog_tvtp, in the order of param_names
    coeffs = params.reshape(2, 2, 3)  # (i, column, j)
    assert mod.param_names[3:6] == ["p[0->0].tvtp1", "p[1->0].tvtp1", "p[2->0].tvtp1"]

    new = np.column_stack([np.ones(5), np.r_[-2.0, -0.5, 0.0, 1.5, 3.0]])
    index = np.einsum("tc,icj->ijt", new, coeffs)
    denominator = 1 + np.exp(index).sum(axis=0)
    expected = np.concatenate([np.exp(index), np.ones((1, 3, 5))]) / denominator

    if form == "list":
        new = new.tolist()
    elif form == "DataFrame":
        new = pd.DataFrame(new, columns=["const", "x"])
    actual = mod.regime_transition_matrix(params, exog_tvtp=new)
    assert actual.shape == (3, 3, 5)
    assert_allclose(actual, expected, rtol=1e-12)


def test_predict_start_end():
    endog = pd.Series(fedfunds, index=FEDFUNDS_INDEX)
    mod = markov_regression.MarkovRegression(endog, k_regimes=2)
    res = mod.smooth(FEDFUNDS_PARAMS)
    full = mod.predict(FEDFUNDS_PARAMS)
    # The prediction weights the regime means by the smoothed probabilities
    smoothed = np.asarray(res.smoothed_marginal_probabilities)
    assert_allclose(full, smoothed @ FEDFUNDS_PARAMS[2:4], rtol=1e-12)

    assert_allclose(mod.predict(FEDFUNDS_PARAMS, start=10, end=20), full[10:21])
    actual = res.predict(start="1960-01-01", end="1962-10-01")
    assert isinstance(actual, pd.Series)
    assert actual.index.equals(FEDFUNDS_INDEX[22:34])
    assert_allclose(actual, full[22:34])
    with pytest.raises(NotImplementedError):
        mod.predict(FEDFUNDS_PARAMS, end=len(fedfunds))


def test_filter_untransformed_params():
    mod = markov_regression.MarkovRegression(fedfunds, k_regimes=2)
    unconstrained = np.r_[4.0, -3.0, 3.7, 9.5, 2.1]
    res_u = mod.filter(unconstrained, transformed=False)
    res = mod.filter(mod.transform_params(unconstrained))
    # The transition probabilities are logistic in the unconstrained values
    assert_allclose(res_u.params[:2], 1 / (1 + np.exp(-unconstrained[:2])))
    assert_allclose(res_u.params, res.params, rtol=1e-12)
    assert_allclose(res_u.llf, res.llf, rtol=1e-12)
    assert_allclose(
        res_u.filtered_marginal_probabilities, res.filtered_marginal_probabilities
    )


def test_results_class_arguments():
    mod = markov_regression.MarkovRegression(fedfunds, k_regimes=2)
    default = mod.smooth(FEDFUNDS_PARAMS)
    assert type(default._results) is markov_regression.MarkovRegressionResults
    for method in [mod.filter, mod.smooth]:
        res = method(
            FEDFUNDS_PARAMS,
            results_class=markov_switching.MarkovSwitchingResults,
            results_wrapper_class=markov_switching.MarkovSwitchingResultsWrapper,
        )
        assert type(res) is markov_switching.MarkovSwitchingResultsWrapper
        assert type(res._results) is markov_switching.MarkovSwitchingResults
        assert_allclose(res.llf, default.llf, rtol=1e-12)
        assert_allclose(
            res.filtered_marginal_probabilities,
            default.filtered_marginal_probabilities,
        )


def test_filter_results_pandas(stata_fedfunds):
    endog = pd.Series(fedfunds, index=FEDFUNDS_INDEX)
    mod = markov_regression.MarkovRegression(endog, k_regimes=2)
    res = mod.filter(FEDFUNDS_PARAMS)
    assert res.smoother_results is None
    assert res.smoothed_marginal_probabilities is None
    assert res.smoothed_joint_probabilities is None
    for name in ["predicted", "filtered"]:
        probs = getattr(res, f"{name}_marginal_probabilities")
        assert isinstance(probs, pd.DataFrame)
        assert probs.index.equals(FEDFUNDS_INDEX)
        # Stata's values are rounded to 7 decimals
        assert_allclose(probs[0], stata_fedfunds[f"const_{name[0]}1"], atol=1e-6)


def test_raw_filter_and_smoother_output(stata_fedfunds):
    mod = markov_regression.MarkovRegression(fedfunds, k_regimes=2)
    raw = mod.filter(FEDFUNDS_PARAMS, return_raw=True)
    assert type(raw) is markov_switching.HamiltonFilterResults
    predicted = raw.predicted_marginal_probabilities
    # Computed once and then reused
    assert raw.predicted_marginal_probabilities is predicted
    assert_allclose(predicted, raw.predicted_joint_probabilities.sum(axis=1))
    # Stata's values are rounded to 7 decimals
    assert_allclose(predicted[0], stata_fedfunds["const_p1"], atol=1e-6)
    assert_allclose(
        raw.filtered_marginal_probabilities[0], stata_fedfunds["const_f1"], atol=1e-6
    )
    assert_allclose(raw.llf, raw.llf_obs.sum(), rtol=1e-12)
    assert_allclose(raw.llf, -508.63592, rtol=1e-7)

    raw = mod.smooth(FEDFUNDS_PARAMS, return_raw=True)
    assert type(raw) is markov_switching.KimSmootherResults
    assert_allclose(
        raw.smoothed_marginal_probabilities[0], stata_fedfunds["const_sm1"], atol=1e-6
    )
    assert_allclose(
        raw.smoothed_marginal_probabilities, raw.smoothed_joint_probabilities.sum(1)
    )


@pytest.mark.parametrize(
    "cov_type, description",
    [
        ("approx", "complex-step"),
        ("opg", "outer product of gradients"),
        ("robust", "Quasi-maximum likelihood"),
        ("none", "not calculated"),
    ],
)
def test_cov_type(cov_type, description):
    mod = markov_regression.MarkovRegression(fedfunds, k_regimes=2)
    res = mod.smooth(FEDFUNDS_PARAMS, cov_type=cov_type, cov_kwds={})
    assert res.cov_type == cov_type
    if cov_type == "none":
        assert np.all(np.isnan(res.cov_params()))
        assert np.all(np.isnan(res.bse))
    else:
        expected = getattr(res, f"cov_params_{cov_type}")
        assert_allclose(res.cov_params(), expected, rtol=1e-12)
        assert_allclose(res.bse, np.sqrt(np.diag(expected)), rtol=1e-12)
    assert description in res.summary().extra_txt


def test_cov_type_defaults():
    mod = markov_regression.MarkovRegression(fedfunds, k_regimes=2)
    assert mod.filter(FEDFUNDS_PARAMS).cov_type == "opg"
    assert mod.smooth(FEDFUNDS_PARAMS).cov_type == "opg"
    with pytest.warns(ConvergenceWarning):
        res = mod.fit(start_params=FEDFUNDS_PARAMS, em_iter=0, maxiter=0)
    assert res.cov_type == "approx"


def test_cov_params_opg_outer_product():
    # The OPG covariance against the outer product of central-difference
    # scores. Finite differences are accurate to about 1e-7 here, so the
    # comparison only catches a wrong element, sign or scale.
    mod = markov_regression.MarkovRegression(fedfunds, k_regimes=2)
    res = mod.smooth(FEDFUNDS_PARAMS, cov_type="opg")
    score_obs = approx_fprime(FEDFUNDS_PARAMS, mod.loglikeobs, centered=True)
    assert_allclose(
        mod.score_obs(FEDFUNDS_PARAMS),
        score_obs,
        rtol=1e-3,
        atol=1e-6 * np.abs(score_obs).max(),
    )
    assert_allclose(res.cov_params(), np.linalg.inv(score_obs.T @ score_obs), rtol=1e-3)
    # The score is the sum of the scores of the observations (away from the
    # maximum, where it is not a sum of terms that nearly cancel)
    other = np.r_[0.95, 0.1, 3.5, 9.0, 4.0]
    assert_allclose(mod.score(other), mod.score_obs(other).sum(0), rtol=1e-10)


def test_cov_type_custom():
    # cov_type="custom" (not documented) takes the covariance matrix and
    # its description from cov_kwds
    mod = markov_regression.MarkovRegression(fedfunds, k_regimes=2)
    cov = np.diag(np.arange(1.0, 6.0))
    res = mod.smooth(
        FEDFUNDS_PARAMS,
        cov_type="custom",
        cov_kwds={
            "custom_cov_type": "given",
            "custom_cov_params": cov,
            "custom_description": "Supplied covariance matrix.",
        },
    )
    assert res.cov_type == "given"
    assert_allclose(res.cov_params(), cov)
    assert_allclose(res.bse, np.sqrt(np.arange(1.0, 6.0)))
    assert "Supplied covariance matrix." in res.summary().extra_txt


@pytest.mark.parametrize("transformed", [True, False])
def test_fit_start_params_maxiter_zero(transformed):
    # With maxiter=0 and no EM steps, fit returns the start parameters
    mod = markov_regression.MarkovRegression(fedfunds, k_regimes=2)
    unconstrained = np.r_[4.0, -3.0, 3.7, 9.5, 2.1]
    constrained = mod.transform_params(unconstrained)
    start = constrained if transformed else unconstrained
    with pytest.warns(ConvergenceWarning):
        params = mod.fit(
            start_params=start,
            transformed=transformed,
            em_iter=0,
            maxiter=0,
            return_params=True,
        )
    assert isinstance(params, np.ndarray)
    assert_allclose(params, constrained, rtol=1e-10)
    assert_allclose(params[:2], 1 / (1 + np.exp(-unconstrained[:2])), rtol=1e-10)

    with pytest.warns(ConvergenceWarning):
        res = mod.fit(
            start_params=list(start),
            transformed=transformed,
            cov_type="opg",
            em_iter=0,
            maxiter=0,
        )
    assert res.cov_type == "opg"
    assert_allclose(res.params, constrained, rtol=1e-10)
    assert_allclose(res.llf, mod.loglike(constrained), rtol=1e-12)


def test_fit_search_per_parameter_scale():
    mod = markov_regression.MarkovRegression(fedfunds, k_regimes=2)
    # Unconstrained start parameters away from the maximum
    start = np.r_[1.0, -1.0, 2.0, 8.0, 2.5]
    scale = np.r_[0.0, 0.0, 4.0, 4.0, 0.0]
    reps = 10
    with pytest.warns(ConvergenceWarning):
        params = mod.fit(
            start_params=start,
            transformed=False,
            search_reps=reps,
            search_iter=0,
            search_scale=scale,
            em_iter=0,
            maxiter=0,
            return_params=True,
            rng=1234,
        )
    # Parameters with a zero scale are not searched, the others by at most
    # half the scale
    constrained_start = mod.transform_params(start)
    searched = scale > 0
    assert_allclose(params[~searched], constrained_start[~searched], rtol=1e-10)
    assert np.all(np.abs(params[searched] - start[searched]) <= scale[searched] / 2)

    # The candidates are the start plus uniform(-scale / 2, scale / 2)
    # variates drawn parameter by parameter; the best one is kept
    rng = np.random.default_rng(1234)
    variates = np.column_stack([s * rng.uniform(-0.5, 0.5, size=reps) for s in scale])
    candidates = np.vstack([start, start + variates])
    llf = [mod.loglike(c, transformed=False) for c in candidates]
    assert np.argmax(llf) > 0
    assert_allclose(params, mod.transform_params(candidates[np.argmax(llf)]))


def test_fit_search_scale_size():
    mod = markov_regression.MarkovRegression(fedfunds, k_regimes=2)
    with pytest.raises(ValueError, match="must be given for each parameter"):
        mod.fit(search_reps=2, search_scale=[1.0, 2.0], rng=0)


def test_fit_search_tvtp_keeps_transition_coefficients():
    # The EM step of the random search leaves time-varying transition
    # coefficients unchanged and updates the regime means and the variance
    nobs = len(fedfunds)
    exog_tvtp = np.column_stack([np.ones(nobs), np.sin(np.arange(nobs) / 8)])
    mod = markov_regression.MarkovRegression(fedfunds, k_regimes=2, exog_tvtp=exog_tvtp)
    start = np.r_[4.0, -3.0, 1.0, 0.5, 3.0, 9.0, 4.0]
    with pytest.warns(ConvergenceWarning):
        params = mod.fit(
            start_params=start,
            search_reps=2,
            search_iter=1,
            search_scale=0.0,
            maxiter=0,
            return_params=True,
            rng=0,
        )
    # With a zero scale each candidate is one EM step from the start
    weights = np.asarray(mod.smooth(start).smoothed_marginal_probabilities).T
    y = np.asarray(fedfunds)
    means = weights @ y / weights.sum(1)
    sigma2 = np.sum(weights * (y - means[:, None]) ** 2) / nobs
    assert_allclose(params, np.r_[start[:4], means, sigma2], rtol=1e-8)
    assert mod.loglike(params) > mod.loglike(start)


def test_transform_untransform_params():
    # Two regimes: logistic
    mod = markov_switching.MarkovSwitching(np.zeros(10), k_regimes=2)
    unconstrained = np.r_[0.5, -1.5]
    constrained = 1 / (1 + np.exp(-unconstrained))
    assert_allclose(mod.transform_params(unconstrained), constrained)
    assert_allclose(mod.untransform_params(constrained), unconstrained)

    # Three regimes: multinomial logistic in each column of the transition
    # matrix, inverted numerically
    mod = markov_switching.MarkovSwitching(np.zeros(10), k_regimes=3)
    constrained = np.r_[0.3, 0.2, 0.1, 0.5, 0.6, 0.2]
    unconstrained = mod.untransform_params(constrained)
    tmp = np.exp(unconstrained.reshape(2, 3))
    assert_allclose((tmp / (1 + tmp.sum(0))).ravel(), constrained, rtol=1e-6)
    assert_allclose(mod.transform_params(unconstrained), constrained, rtol=1e-6)
    # No solution if the probabilities of leaving regime 0 exceed one
    invalid = np.r_[0.6, 0.2, 0.1, 0.6, 0.6, 0.2]
    with pytest.raises(ValueError, match="Could not untransform"):
        mod.untransform_params(invalid)

    # Time-varying transition coefficients are not transformed
    mod = markov_switching.MarkovSwitching(
        np.zeros(10), k_regimes=2, exog_tvtp=np.ones(10)
    )
    params = np.r_[0.5, -1.5]
    assert_allclose(mod.transform_params(params), params)
    assert_allclose(mod.untransform_params(params), params)


def _summary_rows(summary):
    """The first two words of each row of the top table of a summary"""
    return [line.split()[:2] for line in summary.tables[0].as_text().splitlines()]


def test_summary_sample_and_model_name():
    mod = markov_regression.MarkovRegression(
        pd.Series(fedfunds, index=FEDFUNDS_INDEX), k_regimes=2
    )
    res = mod.smooth(FEDFUNDS_PARAMS)
    summary = res.summary(start=3, title="Fed funds", model_name=["Means", "Extra"])
    assert "Fed funds" in summary.tables[0].as_text().splitlines()[0]
    rows = _summary_rows(summary)
    # The sample runs from the date of observation start to the last date
    assert ["Sample:", FEDFUNDS_INDEX[3].strftime("%m-%d-%Y")] in rows
    assert ["-", FEDFUNDS_INDEX[-1].strftime("%m-%d-%Y")] in rows
    assert ["Model:", "Means"] in rows
    assert ["+", "Extra"] in rows

    rows = _summary_rows(res.summary(model_name="Single"))
    assert ["Model:", "Single"] in rows
    assert ["Sample:", FEDFUNDS_INDEX[0].strftime("%m-%d-%Y")] in rows

    # Without dates, the sample is given by observation numbers
    mod = markov_regression.MarkovRegression(fedfunds, k_regimes=2)
    rows = _summary_rows(mod.smooth(FEDFUNDS_PARAMS).summary(start=3))
    assert ["Sample:", "3"] in rows
    assert ["-", str(len(fedfunds))] in rows
    assert ["Model:", "MarkovRegression"] in rows


@pytest.mark.parametrize("cov_type", ["approx", "opg", "robust", "none"])
def test_summary_singular_covariance(cov_type):
    # A regressor of zeros does not enter the likelihood, so its rows and
    # columns of the Hessian and the outer product of scores are zero
    nobs = len(fedfunds)
    mod = markov_regression.MarkovRegression(fedfunds, k_regimes=2, exog=np.zeros(nobs))
    params = np.r_[FEDFUNDS_PARAMS[:4], 0.5, 0.5, FEDFUNDS_PARAMS[4]]
    res = mod.smooth(params, cov_type=cov_type)
    extra = res.summary().extra_txt
    singular = "Covariance matrix is singular or near-singular"
    if cov_type == "none":
        assert singular not in extra
        return
    assert singular in extra
    # The pseudo-inverse gives the other parameters the standard errors of
    # the model without the regressor
    reduced = markov_regression.MarkovRegression(fedfunds, k_regimes=2)
    expected = reduced.smooth(FEDFUNDS_PARAMS, cov_type=cov_type).bse
    assert_allclose(res.bse[[0, 1, 2, 3, 6]], expected, rtol=1e-6)
    # The standard errors of the zero regressor are zero up to rounding
    assert_allclose(res.bse[[4, 5]], 0, atol=1e-6 * expected.max())


def test_summary_without_switching_parameters():
    mod = markov_regression.MarkovRegression(fedfunds, k_regimes=2, switching_trend=False)
    res = mod.smooth(np.r_[FEDFUNDS_PARAMS[:2], 5.0, 4.0])
    titles = [table.title for table in res.summary().tables]
    assert titles == [
        "Markov Switching Model Results",
        "Non-switching parameters",
        "Regime transition parameters",
    ]


def test_markov_switching_params_docstring_example():
    parameters = markov_switching.MarkovSwitchingParams(k_regimes=2)
    parameters["regime_transition"] = [1, 1]
    parameters["exog"] = [0, 1]
    assert_equal(parameters.k_params, 7)
    assert_equal(parameters[0], [0, 2, 4, 5])
    assert_equal(parameters[1], [1, 3, 4, 6])
    assert_equal(parameters["exog"], np.s_[4:7])
    assert_equal(parameters[0, "exog"], [4, 5])
    assert_equal(parameters[1, "exog"], [4, 6])
    assert_equal(parameters.k_parameters["regime_transition"], 4)
    assert_equal(parameters.k_parameters["exog"], 3)


def _endog_form(form, values):
    """endog in the given form and its index (None for array forms)"""
    period_index = pd.period_range("1954Q3", periods=len(values), freq="Q")
    range_index = pd.RangeIndex(len(values))
    forms = {
        "list": (values.tolist(), None),
        "tuple": (tuple(values), None),
        "column": (values[:, None], None),
        "Series-datetime": (pd.Series(values, index=FEDFUNDS_INDEX), FEDFUNDS_INDEX),
        "Series-period": (pd.Series(values, index=period_index), period_index),
        "Series-range": (pd.Series(values, index=range_index), range_index),
        "DataFrame": (
            pd.DataFrame({"fedfunds": values}, index=FEDFUNDS_INDEX),
            FEDFUNDS_INDEX,
        ),
        "int": (values.astype(np.int64), None),
    }
    return forms[form]


@pytest.mark.parametrize(
    "form",
    [
        "list",
        "tuple",
        "column",
        "Series-datetime",
        "Series-period",
        "Series-range",
        "DataFrame",
        "int",
    ],
)
def test_endog_forms(form):
    values = np.asarray(fedfunds)
    if form == "int":
        values = np.round(values)
    endog, index = _endog_form(form, values)

    base = markov_switching.MarkovSwitching(endog, k_regimes=2)
    assert_equal(base.nobs, len(values))
    assert_allclose(base.endog, values)

    mod = markov_regression.MarkovRegression(endog, k_regimes=2)
    res = mod.smooth(FEDFUNDS_PARAMS)
    ref = markov_regression.MarkovRegression(values, k_regimes=2).smooth(
        FEDFUNDS_PARAMS
    )
    assert_allclose(res.llf, ref.llf, rtol=1e-12)
    assert_allclose(
        np.asarray(res.smoothed_marginal_probabilities),
        ref.smoothed_marginal_probabilities,
        rtol=1e-10,
    )
    if index is not None:
        assert res.smoothed_marginal_probabilities.index.equals(index)
        assert res.fittedvalues.index.equals(index)


def test_dates_freq_and_missing():
    values = np.asarray(fedfunds)
    mod = markov_regression.MarkovRegression(
        values, k_regimes=2, dates=FEDFUNDS_INDEX, freq="QS"
    )
    assert mod.data.dates.equals(FEDFUNDS_INDEX)
    full = mod.predict(FEDFUNDS_PARAMS)
    actual = mod.predict(FEDFUNDS_PARAMS, start="1960-01-01", end="1960-10-01")
    assert_allclose(actual, full[22:26])

    endog = values.copy()
    endog[5] = np.nan
    with pytest.raises(MissingDataError):
        markov_switching.MarkovSwitching(endog, k_regimes=2, missing="raise")
    mod = markov_regression.MarkovRegression(endog, k_regimes=2, missing="drop")
    assert_equal(mod.nobs, len(values) - 1)
    ref = markov_regression.MarkovRegression(np.delete(values, 5), k_regimes=2)
    assert_allclose(
        mod.loglike(FEDFUNDS_PARAMS), ref.loglike(FEDFUNDS_PARAMS), rtol=1e-12
    )


@pytest.mark.parametrize("form", ["list", "DataFrame", "1-dim", "Series"])
def test_exog_tvtp_forms(form):
    nobs = len(fedfunds)
    if form in ("1-dim", "Series"):
        exog_tvtp = np.ones((nobs, 1))
        params = np.r_[3.0, -2.0, FEDFUNDS_PARAMS[2:]]
    else:
        exog_tvtp = np.column_stack([np.ones(nobs), np.sin(np.arange(nobs) / 8)])
        params = np.r_[4.0, -3.0, 1.0, 0.5, FEDFUNDS_PARAMS[2:]]
    ref = markov_regression.MarkovRegression(fedfunds, k_regimes=2, exog_tvtp=exog_tvtp)
    if form == "list":
        exog_tvtp = exog_tvtp.tolist()
    elif form == "DataFrame":
        exog_tvtp = pd.DataFrame(exog_tvtp, columns=["const", "x"])
    elif form == "1-dim":
        exog_tvtp = exog_tvtp[:, 0]
    else:
        exog_tvtp = pd.Series(exog_tvtp[:, 0], name="const")
    mod = markov_regression.MarkovRegression(fedfunds, k_regimes=2, exog_tvtp=exog_tvtp)
    assert_equal(mod.k_tvtp, ref.k_tvtp)
    assert mod.param_names == ref.param_names
    assert_allclose(
        mod.regime_transition_matrix(params), ref.regime_transition_matrix(params)
    )
    assert_allclose(mod.loglike(params), ref.loglike(params), rtol=1e-12)
