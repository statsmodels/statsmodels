"""
Tests for setting options in KalmanFilter, KalmanSmoother, SimulationSmoother

(does not test the filtering, smoothing, or simulation smoothing for each
option)

Author: Chad Fulton
License: Simplified-BSD
"""

import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest

from statsmodels.tsa.statespace.kalman_filter import (
    FILTER_AUGMENTED,
    FILTER_CHANDRASEKHAR,
    FILTER_COLLAPSED,
    FILTER_CONCENTRATED,
    FILTER_CONVENTIONAL,
    FILTER_EXACT_INITIAL,
    FILTER_EXTENDED,
    FILTER_SQUARE_ROOT,
    FILTER_UNIVARIATE,
    FILTER_UNSCENTED,
    INVERT_CHOLESKY,
    INVERT_LU,
    INVERT_UNIVARIATE,
    MEMORY_CONSERVE,
    MEMORY_NO_FILTERED,
    MEMORY_NO_FORECAST,
    MEMORY_NO_GAIN,
    MEMORY_NO_LIKELIHOOD,
    MEMORY_NO_PREDICTED,
    MEMORY_NO_SMOOTHING,
    MEMORY_NO_STD_FORECAST,
    MEMORY_STORE_ALL,
    SOLVE_CHOLESKY,
    SOLVE_LU,
    STABILITY_FORCE_SYMMETRY,
    TIMING_INIT_FILTERED,
    TIMING_INIT_PREDICTED,
)
from statsmodels.tsa.statespace.kalman_smoother import (
    SMOOTH_ALTERNATIVE,
    SMOOTH_UNIVARIATE,
    SMOOTHER_ALL,
    SMOOTHER_DISTURBANCE,
    SMOOTHER_DISTURBANCE_COV,
    SMOOTHER_STATE,
    SMOOTHER_STATE_AUTOCOV,
    SMOOTHER_STATE_COV,
    KalmanSmoother,
)
from statsmodels.tsa.statespace.simulation_smoother import (
    SIMULATION_ALL,
    SIMULATION_STATE,
    SimulationSmoother,
)


def setup_model(*args, **kwargs):

    # Dummy data
    endog = np.arange(10)
    k_states = 1

    model = SimulationSmoother(*args, k_endog=1, k_states=k_states, **kwargs)
    model.bind(endog)
    return model


def test_filter_methods():
    model = setup_model()

    # TODO test FilterResults for accurante boolean versions of options
    # Clear the filter method
    model.filter_method = 0

    # Try setting via boolean
    model.filter_conventional = True
    assert_equal(model.filter_method, FILTER_CONVENTIONAL)

    model.filter_collapsed = True
    assert_equal(model.filter_method, FILTER_CONVENTIONAL | FILTER_COLLAPSED)
    model.filter_conventional = False
    assert_equal(model.filter_method, FILTER_COLLAPSED)

    # Try setting directly via method
    model.set_filter_method(FILTER_AUGMENTED)
    assert_equal(model.filter_method, FILTER_AUGMENTED)

    # Try setting via boolean via method
    model.set_filter_method(filter_conventional=True, filter_augmented=False)
    assert_equal(model.filter_method, FILTER_CONVENTIONAL)

    # Try setting and unsetting all
    model.filter_method = 0
    for name in model.filter_methods:
        setattr(model, name, True)
    assert_equal(
        model.filter_method,
        FILTER_CONVENTIONAL
        | FILTER_EXACT_INITIAL
        | FILTER_AUGMENTED
        | FILTER_SQUARE_ROOT
        | FILTER_UNIVARIATE
        | FILTER_COLLAPSED
        | FILTER_EXTENDED
        | FILTER_UNSCENTED
        | FILTER_CONCENTRATED
        | FILTER_CHANDRASEKHAR,
    )
    for name in model.filter_methods:
        setattr(model, name, False)
    assert_equal(model.filter_method, 0)


def test_inversion_methods():
    model = setup_model()

    # Clear the inversion method
    model.inversion_method = 0

    # Try setting via boolean
    model.invert_univariate = True
    assert_equal(model.inversion_method, INVERT_UNIVARIATE)
    model.invert_cholesky = True
    assert_equal(model.inversion_method, INVERT_UNIVARIATE | INVERT_CHOLESKY)
    model.invert_univariate = False
    assert_equal(model.inversion_method, INVERT_CHOLESKY)

    # Try setting directly via method
    model.set_inversion_method(INVERT_LU)
    assert_equal(model.inversion_method, INVERT_LU)

    # Try setting via boolean via method
    model.set_inversion_method(
        invert_cholesky=True, invert_univariate=True, invert_lu=False
    )
    assert_equal(model.inversion_method, INVERT_UNIVARIATE | INVERT_CHOLESKY)

    # Try setting and unsetting all
    model.inversion_method = 0
    for name in model.inversion_methods:
        setattr(model, name, True)
    assert_equal(
        model.inversion_method,
        INVERT_UNIVARIATE | SOLVE_LU | INVERT_LU | SOLVE_CHOLESKY | INVERT_CHOLESKY,
    )
    for name in model.inversion_methods:
        setattr(model, name, False)
    assert_equal(model.inversion_method, 0)


def test_stability_methods():
    model = setup_model()

    # Clear the stability method
    model.stability_method = 0

    # Try setting via boolean
    model.stability_force_symmetry = True
    assert_equal(model.stability_method, STABILITY_FORCE_SYMMETRY)
    model.stability_force_symmetry = False
    assert_equal(model.stability_method, 0)

    # Try setting directly via method
    model.stability_method = 0
    model.set_stability_method(STABILITY_FORCE_SYMMETRY)
    assert_equal(model.stability_method, STABILITY_FORCE_SYMMETRY)

    # Try setting via boolean via method
    model.stability_method = 0
    model.set_stability_method(stability_method=True)
    assert_equal(model.stability_method, STABILITY_FORCE_SYMMETRY)

    # Try setting via keyword via method
    model.stability_method = 0
    model.set_stability_method(stability_force_symmetry=True)
    assert_equal(model.stability_method, STABILITY_FORCE_SYMMETRY)


def test_conserve_memory():
    model = setup_model()

    # Clear the filter method
    model.conserve_memory = MEMORY_STORE_ALL

    # Try setting via boolean
    model.memory_no_forecast = True
    assert_equal(model.conserve_memory, MEMORY_NO_FORECAST)
    model.memory_no_filtered = True
    assert_equal(model.conserve_memory, MEMORY_NO_FORECAST | MEMORY_NO_FILTERED)
    model.memory_no_forecast = False
    assert_equal(model.conserve_memory, MEMORY_NO_FILTERED)

    # Try setting directly via method
    model.set_conserve_memory(MEMORY_NO_PREDICTED)
    assert_equal(model.conserve_memory, MEMORY_NO_PREDICTED)

    # Try setting via boolean via method
    model.set_conserve_memory(memory_no_filtered=True, memory_no_predicted=False)
    assert_equal(model.conserve_memory, MEMORY_NO_FILTERED)

    # Try setting and unsetting all
    model.conserve_memory = 0
    for name in model.memory_options:
        if name == "memory_conserve":
            continue
        setattr(model, name, True)
    assert_equal(
        model.conserve_memory,
        MEMORY_NO_FORECAST
        | MEMORY_NO_PREDICTED
        | MEMORY_NO_FILTERED
        | MEMORY_NO_LIKELIHOOD
        | MEMORY_NO_GAIN
        | MEMORY_NO_SMOOTHING
        | MEMORY_NO_STD_FORECAST,
    )
    assert_equal(model.conserve_memory & MEMORY_CONSERVE, MEMORY_CONSERVE)
    for name in model.memory_options:
        if name == "memory_conserve":
            continue
        setattr(model, name, False)
    assert_equal(model.conserve_memory, 0)


def test_smoother_outputs():
    model = setup_model()

    # TODO test SmootherResults for accurante boolean versions of options

    # Clear the smoother output
    model.smoother_output = 0

    # Try setting via boolean
    model.smoother_state = True
    assert_equal(model.smoother_output, SMOOTHER_STATE)
    model.smoother_disturbance = True
    assert_equal(model.smoother_output, SMOOTHER_STATE | SMOOTHER_DISTURBANCE)
    model.smoother_state = False
    assert_equal(model.smoother_output, SMOOTHER_DISTURBANCE)

    # Try setting directly via method
    model.set_smoother_output(SMOOTHER_DISTURBANCE_COV)
    assert_equal(model.smoother_output, SMOOTHER_DISTURBANCE_COV)

    # Try setting via boolean via method
    model.set_smoother_output(smoother_disturbance=True, smoother_disturbance_cov=False)
    assert_equal(model.smoother_output, SMOOTHER_DISTURBANCE)

    # Try setting and unsetting all
    model.smoother_output = 0
    for name in model.smoother_outputs:
        if name == "smoother_all":
            continue
        setattr(model, name, True)
    assert_equal(
        model.smoother_output,
        SMOOTHER_STATE
        | SMOOTHER_STATE_COV
        | SMOOTHER_STATE_AUTOCOV
        | SMOOTHER_DISTURBANCE
        | SMOOTHER_DISTURBANCE_COV,
    )
    assert_equal(model.smoother_output, SMOOTHER_ALL)
    for name in model.smoother_outputs:
        if name == "smoother_all":
            continue
        setattr(model, name, False)
    assert_equal(model.smoother_output, 0)


def test_simulation_outputs():
    # TODO test changing simulation options in SimulationSmoothResults
    # instance
    model = setup_model()
    assert_equal(model.get_simulation_output(SIMULATION_STATE), SIMULATION_STATE)
    assert_equal(
        model.get_simulation_output(simulate_state=True, simulate_disturbance=True),
        SIMULATION_ALL,
    )


@pytest.mark.parametrize(
    "kwargs, desired",
    [
        ({"timing_init_filtered": True}, TIMING_INIT_FILTERED),
        ({"timing_init_filtered": False}, TIMING_INIT_PREDICTED),
        ({"timing_init_predicted": True}, TIMING_INIT_PREDICTED),
        ({"timing_init_predicted": False}, TIMING_INIT_FILTERED),
    ],
)
def test_filter_timing_keywords(kwargs, desired):
    model = setup_model(**kwargs)
    assert_equal(model.filter_timing, desired)
    model = setup_model()
    model.set_filter_timing(**kwargs)
    assert_equal(model.filter_timing, desired)


def test_fixed_scale_requires_concentrated_filter():
    model = setup_model()
    with pytest.raises(ValueError, match="Cannot provide scale if filter method"):
        with model.fixed_scale(2.0):
            pass


def filter_model(k_endog=2, nobs=60):
    # Up to two observed series and two states, with correlated observation
    # and state disturbances
    rng = np.random.default_rng(24680)
    endog = rng.standard_normal((nobs, 2)).cumsum(0) * 0.1
    endog += rng.standard_normal((nobs, 2))
    model = KalmanSmoother(np.ascontiguousarray(endog[:, :k_endog]), k_states=2)
    model["design"] = np.array([[1.0, 0.2], [0.3, 1.0]])[:k_endog]
    model["transition"] = np.array([[0.5, 0.1], [0.0, 0.3]])
    model["selection"] = np.eye(2)
    model["state_cov"] = np.array([[1.0, 0.2], [0.2, 0.5]])
    model["obs_cov"] = np.array([[1.0, 0.3], [0.3, 2.0]])[:k_endog, :k_endog]
    model.initialize_known(np.zeros(2), np.eye(2))
    return model


INVERSIONS = [(INVERT_CHOLESKY, SOLVE_CHOLESKY), (INVERT_LU, SOLVE_LU)]


@pytest.mark.parametrize("k_endog", [1, 2])
@pytest.mark.parametrize("method, reference", INVERSIONS)
def test_inversion_methods_agree(k_endog, method, reference):
    # Inverting F_t, or solving with the same Cholesky or LU factorization,
    # must give the same filter and smoother output. Set alone, the inversion
    # is used also for a single series.
    results = {}
    for inversion_method in (method, reference):
        model = filter_model(k_endog)
        model.inversion_method = inversion_method
        results[inversion_method] = model.smooth()
    actual, desired = results[method], results[reference]
    assert_allclose(actual.llf, desired.llf, rtol=1e-12)
    for name in [
        "filtered_state",
        "filtered_state_cov",
        "standardized_forecasts_error",
        "kalman_gain",
        "smoothed_state",
        "smoothed_state_cov",
    ]:
        # atol: entries near zero differ by rounding (about 1e-15) between the
        # two computations
        assert_allclose(
            getattr(actual, name), getattr(desired, name), rtol=1e-10, atol=1e-13
        )


@pytest.mark.parametrize(
    "conserve_memory", [MEMORY_CONSERVE, MEMORY_CONSERVE | MEMORY_NO_STD_FORECAST]
)
@pytest.mark.parametrize("k_endog", [1, 2])
@pytest.mark.parametrize("method, reference", INVERSIONS)
def test_inversion_methods_conserve_memory(conserve_memory, k_endog, method, reference):
    desired = filter_model(k_endog).filter()
    for inversion_method in (method, reference):
        model = filter_model(k_endog)
        model.inversion_method = inversion_method
        model.set_conserve_memory(conserve_memory)
        res = model.filter()
        assert_allclose(res.llf, desired.llf, rtol=1e-12)
        assert_allclose(res.filtered_state[:, -1], desired.filtered_state[:, -1])
        assert_allclose(model.loglike(), desired.llf, rtol=1e-12)


def test_invalid_inversion_method():
    model = filter_model()
    model.inversion_method = 0
    with pytest.raises(NotImplementedError, match="Invalid inversion method"):
        model.filter()


def test_invalid_filter_method():
    # The concentrated likelihood is an addition to a filtering method
    model = filter_model()
    model.filter_method = FILTER_CONCENTRATED
    with pytest.raises(NotImplementedError, match="Invalid filtering method"):
        model.filter()


@pytest.mark.parametrize(
    "name, value",
    [
        ("filter_method", FILTER_CONVENTIONAL | FILTER_UNIVARIATE),
        ("stability_method", 0),
        ("loglikelihood_burn", 5),
    ],
)
def test_filter_arguments(name, value):
    # An option passed to `filter` is used for that call only, and gives the
    # output of the filter with that option set on the model
    model = filter_model()
    default = model.filter()
    actual = model.filter(**{name: value})
    assert_equal(getattr(actual, name), value)
    assert_equal(getattr(model, name), getattr(default, name))

    setattr(model, name, value)
    desired = model.filter()
    assert_allclose(actual.llf, desired.llf, rtol=1e-12)
    assert_allclose(actual.filtered_state, desired.filtered_state, rtol=1e-12)
    assert_equal(actual.period_converged, desired.period_converged)
    if name == "loglikelihood_burn":
        # The first `value` periods are left out of the loglikelihood
        assert_allclose(actual.llf, default.llf_obs[value:].sum(), rtol=1e-12)
    else:
        # Univariate filtering and the symmetry step change the output only by
        # rounding
        assert_allclose(actual.llf, default.llf, rtol=1e-10)


def test_smooth_univariate_option():
    # set_smooth_method(smooth_univariate=True) and the constructor argument
    # set the univariate smoothing flag if univariate filtering is used, as
    # setting the attribute does (the former had no effect and the latter
    # raised a TypeError)
    attribute = setup_model(filter_univariate=True)
    attribute.smooth_univariate = True
    assert_equal(attribute.smooth_method, SMOOTH_UNIVARIATE)

    model = setup_model(filter_univariate=True)
    model.set_smooth_method(smooth_univariate=True)
    assert_equal(model.smooth_method, SMOOTH_UNIVARIATE)
    assert model.smooth_univariate
    model.set_smooth_method(smooth_univariate=False)
    assert_equal(model.smooth_method, 0)

    model = setup_model(filter_univariate=True, smooth_univariate=True)
    assert_equal(model.smooth_method, SMOOTH_UNIVARIATE)

    # Without univariate filtering, the flag is still ignored (smoothing runs
    # with the default method), with a FutureWarning
    with pytest.warns(FutureWarning, match="smooth_univariate=True has no"):
        model = setup_model(smooth_univariate=True)
    assert_equal(model.smooth_method, 0)
    model = setup_model()
    with pytest.warns(FutureWarning, match="smooth_univariate=True has no"):
        model.set_smooth_method(smooth_univariate=True)
    assert_equal(model.smooth_method, 0)
    assert not model.smooth_univariate
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.set_smooth_method(smooth_univariate=False)
        model.set_smooth_method(smooth_alternative=True)
    assert_equal(model.smooth_method, SMOOTH_ALTERNATIVE)

    # The univariate smoother is used: it requires the univariate filter, and
    # with it gives the output of the default (univariate) smoothing method
    def get_model(**kwargs):
        model = setup_model(**kwargs)
        model["design"] = [[1.0]]
        model["obs_cov"] = [[0.5]]
        model["transition"] = [[0.8]]
        model["selection"] = [[1.0]]
        model["state_cov"] = [[1.2]]
        model.initialize_known(np.array([0.3]), np.array([[2.0]]))
        return model

    model = get_model()
    with pytest.warns(FutureWarning, match="smooth_univariate=True has no"):
        model.set_smooth_method(smooth_univariate=True)
    res = model.smooth()
    desired = get_model().smooth()
    assert_allclose(res.smoothed_state, desired.smoothed_state, rtol=1e-14)
    model = get_model()
    model.smooth_univariate = True
    with pytest.raises(ValueError, match="Invalid smoothing method"):
        model.smooth()
    model = get_model(filter_univariate=True)
    model.set_smooth_method(smooth_univariate=True)
    assert model.smooth_univariate
    res = model.smooth()
    desired = get_model(filter_univariate=True).smooth()
    assert_allclose(res.smoothed_state, desired.smoothed_state, rtol=1e-14)
    assert_allclose(res.smoothed_state_cov, desired.smoothed_state_cov, rtol=1e-14)


def _option_flags_model(k_endog=2, diffuse=False, concentrated=False):
    # Two states and up to three observed series (three for the collapsed
    # filter, which needs k_endog > k_states)
    rng = np.random.default_rng(97531)
    endog = rng.standard_normal((50, 3)).cumsum(0) * 0.1
    endog += rng.standard_normal((50, 3))
    model = KalmanSmoother(np.ascontiguousarray(endog[:, :k_endog]), k_states=2)
    model["design"] = np.array([[1.0, 0.2], [0.3, 1.0], [0.5, 0.5]])[:k_endog]
    model["transition"] = np.array([[0.5, 0.1], [0.0, 0.3]])
    model["selection"] = np.eye(2)
    model["state_cov"] = np.array([[1.0, 0.2], [0.2, 0.5]])
    model["obs_cov"] = np.diag([1.0, 2.0, 1.5])[:k_endog, :k_endog]
    if diffuse:
        # A random walk first state, initialized as diffuse
        model["transition", 0, 0] = 1.0
        model.initialize_diffuse()
    else:
        model.initialize_known(np.zeros(2), np.eye(2))
    model.filter_concentrated = concentrated
    return model


# The scale is concentrated out by the call, also with a diffuse
# initialization, or no longer concentrated out
CONCENTRATED_CASES = [
    ("filter_method", FILTER_CONVENTIONAL | FILTER_CONCENTRATED, {}),
    ("filter_method", FILTER_CONVENTIONAL | FILTER_CONCENTRATED, {"diffuse": True}),
    ("filter_method", FILTER_CONVENTIONAL, {"concentrated": True}),
]
OPTION_FLAG_CASES = [
    ("inversion_method", INVERT_LU, {}),
    ("inversion_method", SOLVE_LU, {}),
    *CONCENTRATED_CASES,
    ("filter_method", FILTER_CONVENTIONAL | FILTER_COLLAPSED, {"k_endog": 3}),
]


def _set_option(model, name, value):
    setattr(model, name, value)
    return model


# `smooth(filter_method=...)` raises TypeError (filter_method is also passed
# positionally to `_initialize_filter`), so `smooth` is checked only for the
# inversion method
@pytest.mark.parametrize(
    "method, name, value, model_kwargs",
    [("filter", *case) for case in OPTION_FLAG_CASES]
    + [("smooth", *case) for case in OPTION_FLAG_CASES if case[0] != "filter_method"],
)
def test_filter_option_arguments_results(method, name, value, model_kwargs):
    # The results of an option given for one call match those of the option
    # set on the model, also for output that depends on the boolean option
    # flags: the concentrated scale (FILTER_CONCENTRATED), the standardized
    # forecast errors (computed in Python for LU) and the forecasts of the
    # observed series (rebuilt from the collapsed ones)
    actual = getattr(_option_flags_model(**model_kwargs), method)(**{name: value})
    model = _set_option(_option_flags_model(**model_kwargs), name, value)
    desired = getattr(model, method)()

    for flag in desired._filter_options:
        assert getattr(actual, flag) == getattr(desired, flag), flag
    assert_allclose(actual.llf, desired.llf, rtol=1e-12)
    assert_allclose(actual.scale, desired.scale, rtol=1e-12)
    for attr in [
        "filtered_state",
        "filtered_state_cov",
        "forecasts",
        "forecasts_error",
        "forecasts_error_cov",
        "standardized_forecasts_error",
    ]:
        assert_allclose(getattr(actual, attr), getattr(desired, attr), rtol=1e-12)
    if method == "smooth":
        assert_allclose(actual.smoothed_state_cov, desired.smoothed_state_cov, rtol=1e-12)


@pytest.mark.parametrize("method", ["loglike", "loglikeobs"])
@pytest.mark.parametrize("name, value, model_kwargs", CONCENTRATED_CASES)
def test_filter_option_arguments_loglike(method, name, value, model_kwargs):
    actual = getattr(_option_flags_model(**model_kwargs), method)(**{name: value})
    model = _set_option(_option_flags_model(**model_kwargs), name, value)
    desired = getattr(model, method)()
    assert_allclose(actual, desired, rtol=1e-12)
    # The loglikelihood of the filter output, as an independent check of the
    # concentrated scale adjustment
    llf_obs = model.filter().llf_obs
    assert_allclose(desired, llf_obs.sum() if method == "loglike" else llf_obs, rtol=1e-12)
