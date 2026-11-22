"""
General tests for Markov switching models

Author: Chad Fulton
License: BSD-3
"""

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest

from statsmodels.tools.numdiff import approx_fprime_cs
from statsmodels.tools.sm_exceptions import MissingDataError
from statsmodels.tsa.regime_switching import (
    markov_autoregression,
    markov_regression,
    markov_switching,
)


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


def test_logistic():
    logistic = markov_switching._logistic

    # For a number, logistic(x) = np.exp(x) / (1 + np.exp(x))
    cases = [0, 10.0, -4]
    for x in cases:
        # Have to use allclose b/c logistic() actually uses logsumexp, so
        # they're not equal
        assert_allclose(logistic(x), np.exp(x) / (1 + np.exp(x)))

    # For a vector, logistic(x) returns
    # np.exp(x[i]) / (1 + np.sum(np.exp(x[:]))) for each i
    # but squeezed
    cases = [[1.0], [0, 1.0], [-2, 3.0, 1.2, -30.0]]
    for x in cases:
        actual = logistic(x)
        desired = [np.exp(i) / (1 + np.sum(np.exp(x))) for i in x]
        assert_allclose(actual, desired)

    # For a 2-dim, logistic(x) returns
    # np.exp(x[i, t]) / (1 + np.sum(np.exp(x[:, t]))) for each i, each t
    # but squeezed
    case = [[1.0]]
    actual = logistic(case)
    assert_equal(actual.shape, (1, 1))
    assert_allclose(actual, np.exp(1) / (1 + np.exp(1)))

    # Here, np.array(case) is 2x1, so it is interpreted as i=0, 1 and t=0
    case = [[0], [1.0]]
    actual = logistic(case)
    desired = [np.exp(i) / (1 + np.sum(np.exp(case))) for i in case]
    assert_allclose(actual, desired)

    # Here, np.array(case) is 1x2, so it is interpreted as i=0 and t=0, 1
    case = [[0, 1.0]]
    actual = logistic(case)
    desired = np.exp(case) / (1 + np.exp(case))
    assert_allclose(actual, desired)

    # For a 3-dim, logistic(x) returns
    # np.exp(x[i, j, t]) / (1 + np.sum(np.exp(x[:, j, t])))
    # for each i, each j, each t
    case = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    actual = logistic(case)
    for j in range(3):
        assert_allclose(actual[:, j, :], logistic(case[:, j, :]))


def test_partials_logistic():
    # Here we compare to analytic derivatives and to finite-difference
    # approximations
    logistic = markov_switching._logistic
    partials_logistic = markov_switching._partials_logistic

    # For a number, logistic(x) = np.exp(x) / (1 + np.exp(x))
    # Then d/dx = logistix(x) - logistic(x)**2
    cases = [0, 10.0, -4]
    for x in cases:
        assert_allclose(partials_logistic(x), logistic(x) - logistic(x) ** 2)
        assert_allclose(partials_logistic(x), approx_fprime_cs([x], logistic))

    # For a vector, logistic(x) returns
    # np.exp(x[i]) / (1 + np.sum(np.exp(x[:]))) for each i
    # Then d logistic(x[i]) / dx[i] = (logistix(x) - logistic(x)**2)[i]
    # And d logistic(x[i]) / dx[j] = -(logistic(x[i]) * logistic[x[j]])
    cases = [[1.0], [0, 1.0], [-2, 3.0, 1.2, -30.0]]
    for x in cases:
        evaluated = np.atleast_1d(logistic(x))
        partials = np.diag(evaluated - evaluated**2)
        for i in range(len(x)):
            for j in range(i):
                partials[i, j] = partials[j, i] = -evaluated[i] * evaluated[j]
        assert_allclose(partials_logistic(x), partials)
        assert_allclose(partials_logistic(x), approx_fprime_cs(x, logistic))

    # For a 2-dim, logistic(x) returns
    # np.exp(x[i, t]) / (1 + np.sum(np.exp(x[:, t]))) for each i, each t
    # but squeezed
    case = [[1.0]]
    evaluated = logistic(case)
    partial = [evaluated - evaluated**2]
    assert_allclose(partials_logistic(case), partial)
    assert_allclose(partials_logistic(case), approx_fprime_cs(case, logistic))

    # # Here, np.array(case) is 2x1, so it is interpreted as i=0, 1 and t=0
    case = [[0], [1.0]]
    evaluated = logistic(case)[:, 0]
    partials = np.diag(evaluated - evaluated**2)
    partials[0, 1] = partials[1, 0] = -np.multiply(*evaluated)
    assert_allclose(partials_logistic(case)[:, :, 0], partials)
    assert_allclose(
        partials_logistic(case), approx_fprime_cs(np.squeeze(case), logistic)[..., None]
    )

    # Here, np.array(case) is 1x2, so it is interpreted as i=0 and t=0, 1
    case = [[0, 1.0]]
    evaluated = logistic(case)
    partials = (evaluated - evaluated**2)[None, ...]
    assert_allclose(partials_logistic(case), partials)
    assert_allclose(partials_logistic(case), approx_fprime_cs(case, logistic).T)

    # For a 3-dim, logistic(x) returns
    # np.exp(x[i, j, t]) / (1 + np.sum(np.exp(x[:, j, t])))
    # for each i, each j, each t
    case = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    evaluated = logistic(case)
    partials = partials_logistic(case)
    for t in range(4):
        for j in range(3):
            desired = np.diag(evaluated[:, j, t] - evaluated[:, j, t] ** 2)
            desired[0, 1] = desired[1, 0] = -np.multiply(*evaluated[:, j, t])
            assert_allclose(partials[..., j, t], desired)


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
