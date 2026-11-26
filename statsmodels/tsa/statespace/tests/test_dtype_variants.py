"""
Tests of the float32, complex64 and complex128 versions of the Cython code

The Cython templates of the state space, regime switching and innovations
code are compiled for four dtypes, with the BLAS prefixes "s" (float32), "d"
(float64), "c" (complex64) and "z" (complex128). Models use float64, and
complex-step derivatives use complex128 for filtering only. The other
variants are used by the low-level classes (``KalmanFilter``,
``KalmanSmoother``, ``SimulationSmoother``) created with ``dtype`` and with
the data and every system matrix in that dtype, by ``MLEModel`` created with
``dtype``, by the module-level Hamilton filter and Kim smoother functions
given arrays of that dtype, and by ``arma_innovations`` with ``prefix``.

Each test runs a small model in float32, complex64 and complex128 and
compares the output with the float64 run of the same model. Every output is
checked to have the dtype of the variant, so that a silent switch to float64
(for example a system matrix set from a float64 array) fails.
"""

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest

from statsmodels.tools.sm_exceptions import OutputWarning
from statsmodels.tsa.innovations import arma_innovations
from statsmodels.tsa.regime_switching.markov_switching import (
    cy_hamilton_filter_log,
    cy_kim_smoother_log,
)
from statsmodels.tsa.statespace import tools
from statsmodels.tsa.statespace.initialization import Initialization
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
from statsmodels.tsa.statespace.kalman_smoother import (
    SMOOTH_ALTERNATIVE,
    SMOOTH_CLASSICAL,
    SMOOTH_CONVENTIONAL,
    SMOOTH_UNIVARIATE,
    KalmanSmoother,
)
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.statespace.simulation_smoother import SimulationSmoother

DTYPES = [np.float32, np.complex64, np.complex128]
DTYPE_IDS = ["float32", "complex64", "complex128"]
PREFIX = {np.float32: "s", np.complex64: "c", np.complex128: "z"}
# In these small models the single precision variants (machine epsilon
# 1.2e-7) differ from float64 by at most 5e-6 relative to the largest element
# of an output (in the exact diffuse periods; 5e-7 elsewhere), and complex128
# by at most 1e-15, on Windows and Linux. The imaginary parts are zero on
# Windows, and at most 1.2e-7 (complex64) and 2e-16 (complex128) relative to
# the largest element on Linux.
RTOL = {np.float32: 1e-4, np.complex64: 1e-4, np.complex128: 1e-12}

DIAGONALIZED = "The univariate filtering approach diagonalized"

FILTER_NAMES = [
    "llf_obs",
    "forecasts",
    "forecasts_error",
    "forecasts_error_cov",
    "filtered_state",
    "filtered_state_cov",
    "predicted_state",
    "predicted_state_cov",
]
SMOOTHER_NAMES = [
    "smoothed_state",
    "smoothed_state_cov",
    "smoothed_state_autocov",
    "smoothed_measurement_disturbance",
    "smoothed_state_disturbance",
    "smoothed_measurement_disturbance_cov",
    "smoothed_state_disturbance_cov",
    "scaled_smoothed_estimator",
    "scaled_smoothed_estimator_cov",
    "smoothing_error",
]
DIFFUSE_NAMES = [
    "initial_diffuse_state_cov",
    "forecasts_error_diffuse_cov",
    "predicted_diffuse_state_cov",
]
DIFFUSE_SMOOTHER_NAMES = [
    "scaled_smoothed_diffuse_estimator",
    "scaled_smoothed_diffuse1_estimator_cov",
    "scaled_smoothed_diffuse2_estimator_cov",
]

dtypes = pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)


def assert_matches(actual, desired, dtype, name=""):
    """
    Check the output of a dtype variant against the float64 output
    """
    actual = np.asarray(actual)
    desired = np.asarray(desired)
    assert actual.dtype == dtype, name
    rtol = RTOL[dtype]
    finite = np.isfinite(desired)
    scale = np.abs(desired[finite]).max() if finite.any() else 0.0
    # Rounding errors are relative to the largest element, so the elements
    # near zero (covariances, states crossing zero) get an absolute tolerance
    # from it
    atol = rtol * scale
    assert_allclose(actual.real, desired, rtol=rtol, atol=atol, err_msg=name)
    if np.iscomplexobj(actual):
        # The imaginary part is zero in exact arithmetic (except where the
        # output is NaN, as the real part checks)
        assert_allclose(actual.imag[finite], 0, atol=atol, err_msg=name)


def assert_results_match(res, desired, dtype, names):
    for name in names:
        expected = getattr(desired, name)
        if expected is None:
            assert getattr(res, name) is None, name
        else:
            assert_matches(getattr(res, name), expected, dtype, name)


def build(
    cls,
    dtype,
    k_endog=2,
    k_states=2,
    nobs=15,
    missing=None,
    init="known",
    singular=False,
    obs_intercept=True,
):
    """
    A small time-invariant state space model with every matrix in `dtype`
    """
    endog = np.random.default_rng(20261123).standard_normal((nobs, k_endog))
    if missing in ("partial", "both"):
        endog[3, 0] = np.nan
        endog[8, -1] = np.nan
    if missing in ("all", "both"):
        endog[5:7] = np.nan

    design = np.zeros((k_endog, k_states))
    for i in range(k_endog):
        design[i, i % k_states] = 1.0
        design[i, (i + 1) % k_states] += 0.5 * (i + 1) / k_endog
    obs_cov = 0.4 * np.eye(k_endog) + 0.1
    if singular:
        # The last observation does not depend on the state and has variance
        # zero, so the forecast error covariance matrix is singular
        design[-1] = 0
        obs_cov[-1] = 0
        obs_cov[:, -1] = 0
    transition = np.diag(np.linspace(0.6, 0.3, k_states))
    if k_states > 1:
        transition[0, 1] = 0.1

    mod = cls(k_endog, k_states, k_posdef=k_states, dtype=dtype)
    mod.bind(endog.astype(dtype))
    mod["design"] = design.astype(dtype)
    if obs_intercept:
        mod["obs_intercept"] = np.linspace(0.1, 0.3, k_endog).astype(dtype)
    mod["obs_cov"] = obs_cov.astype(dtype)
    mod["transition"] = transition.astype(dtype)
    mod["state_intercept"] = np.linspace(0.2, -0.1, k_states).astype(dtype)
    mod["selection"] = np.eye(k_states, dtype=dtype)
    mod["state_cov"] = (0.8 * np.eye(k_states) + 0.1).astype(dtype)

    if init == "known":
        mod.initialize_known(
            np.linspace(-0.5, 0.5, k_states), np.diag(np.linspace(1, 2, k_states))
        )
    elif init == "stationary":
        mod.initialize_stationary()
    elif init == "diffuse":
        mod.initialize_diffuse()
    elif init == "mixed":
        # A diffuse, a known and a stationary block
        initialization = Initialization(k_states)
        initialization.set(0, "diffuse")
        initialization.set(1, "known", constant=[0.3], stationary_cov=[[1.5]])
        initialization.set((2, k_states), "stationary")
        mod.initialize(initialization)
    assert mod.prefix == PREFIX.get(dtype, "d")
    return mod


FILTER_CASES = {
    "conventional": {},
    "univariate": {"options": {"filter_univariate": True}},
    "univariate_fallback": {"model": {"singular": True}},
    "collapsed": {
        "model": {"k_endog": 3, "obs_intercept": False},
        "options": {"filter_collapsed": True},
    },
    "chandrasekhar": {
        "model": {"init": "stationary"},
        "options": {"filter_chandrasekhar": True},
    },
    "concentrated": {"options": {"filter_concentrated": True}},
    "conserve_memory": {"options": {"memory_conserve": True}},
    # The default tolerance (1e-19) is met in float64 and complex128 but not
    # in single precision. With 1e-12 every run reaches the steady state.
    "converged": {"options": {"tolerance": 1e-12}},
    # A converged filter with the filtered timing leaves the last predicted
    # state covariance at zero (a separate issue), so no run may converge
    "timing_init_filtered": {"options": {"timing_init_filtered": True, "tolerance": 0}},
}


@pytest.mark.parametrize("case", list(FILTER_CASES))
@dtypes
def test_filter(case, dtype):
    spec = FILTER_CASES[case]

    def run(dtype):
        mod = build(KalmanFilter, dtype, **spec.get("model", {}))
        for key, value in spec.get("options", {}).items():
            setattr(mod, key, value)
        return mod.filter()

    res = run(dtype)
    desired = run(np.float64)
    assert res.prefix == PREFIX[dtype]

    names = FILTER_NAMES
    if case == "univariate_fallback":
        # Every period falls back to the univariate filter
        assert_equal(res.univariate_filter, np.ones(res.nobs))
        assert_equal(desired.univariate_filter, np.ones(res.nobs))
    elif case == "concentrated":
        # The covariances are multiplied by the scale in Python, whose dtype
        # is float64 for the single precision filters
        names = ["llf_obs", "forecasts", "forecasts_error", "filtered_state"]
        assert_allclose(res.scale.real, desired.scale, rtol=RTOL[dtype])
        assert np.asarray(res.scale).dtype == np.result_type(dtype, np.float64)
    elif case == "conserve_memory":
        # Only the loglikelihood and the last periods are stored
        assert res.llf_obs is None
        assert_matches(res.llf, desired.llf, dtype, "llf")
        names = names[1:]
    elif case == "converged":
        assert res.converged
        assert desired.converged
    elif case == "timing_init_filtered":
        assert not res.converged
        assert not desired.converged
    assert_results_match(res, desired, dtype, names)


@pytest.mark.parametrize("init", ["known", "stationary", "diffuse", "mixed"])
@dtypes
def test_initialization(init, dtype):
    def run(dtype):
        return build(KalmanFilter, dtype, k_states=3, init=init).filter()

    res = run(dtype)
    desired = run(np.float64)
    assert res.prefix == PREFIX[dtype]
    assert res.nobs_diffuse == desired.nobs_diffuse
    assert (res.nobs_diffuse > 0) == (init in ("diffuse", "mixed"))
    names = FILTER_NAMES + ["initial_state", "initial_state_cov"]
    if res.nobs_diffuse > 0:
        names += DIFFUSE_NAMES
    assert_results_match(res, desired, dtype, names)


def smooth(mod):
    # The univariate approach (used by the univariate filter, and in diffuse
    # periods) replaces the non-diagonal observation covariance matrix by a
    # diagonal one, which the smoother warns about
    diffuse = mod.initialization.initialization_type == "diffuse"
    if mod.filter_univariate or diffuse:
        with pytest.warns(OutputWarning, match=DIAGONALIZED):
            return mod.smooth()
    return mod.smooth()


@pytest.mark.parametrize("missing", ["partial", "all"])
@pytest.mark.parametrize(
    "univariate", [False, True], ids=["conventional", "univariate"]
)
@dtypes
def test_missing(missing, univariate, dtype):
    def run(dtype):
        mod = build(KalmanSmoother, dtype, missing=missing)
        mod.filter_univariate = univariate
        return smooth(mod)

    res = run(dtype)
    desired = run(np.float64)
    assert res.prefix == PREFIX[dtype]
    assert_equal(res.nmissing, desired.nmissing)
    assert res.nmissing.max() == (1 if missing == "partial" else 2)
    assert_results_match(res, desired, dtype, FILTER_NAMES + SMOOTHER_NAMES)


SMOOTH_METHODS = {
    "conventional": SMOOTH_CONVENTIONAL,
    "classical": SMOOTH_CLASSICAL,
    "alternative": SMOOTH_ALTERNATIVE,
    "univariate": SMOOTH_UNIVARIATE,
}


@pytest.mark.parametrize("method", list(SMOOTH_METHODS))
@pytest.mark.parametrize("init", ["known", "diffuse"])
@dtypes
def test_smoother(method, init, dtype):
    def run(dtype):
        mod = build(KalmanSmoother, dtype, init=init, missing="both")
        if method == "univariate":
            mod.filter_univariate = True
        setattr(mod, "smooth_" + method, True)
        res = smooth(mod)
        # The Cython smoother of the dtype used the method
        smoother = mod._kalman_smoother
        assert type(smoother).__name__ == mod.prefix + "KalmanSmoother"
        assert smoother._smooth_method == SMOOTH_METHODS[method]
        return res

    res = run(dtype)
    desired = run(np.float64)
    assert res.prefix == PREFIX[dtype]
    names = FILTER_NAMES + SMOOTHER_NAMES
    if init == "diffuse":
        assert res.nobs_diffuse > 0
        names += DIFFUSE_SMOOTHER_NAMES
    assert_results_match(res, desired, dtype, names)


SIMULATION_CASES = {
    "kfs": {},
    "kfs_diffuse_missing": {"init": "diffuse", "missing": "both"},
    "cfa": {},
    "cfa_missing": {"missing": "both"},
}


@pytest.mark.parametrize("case", list(SIMULATION_CASES))
@dtypes
def test_simulation_smoother(case, dtype):
    kwargs = SIMULATION_CASES[case]
    nobs, k_states = 15, 2

    def run(dtype):
        mod = build(SimulationSmoother, dtype, nobs=nobs, k_states=k_states, **kwargs)
        if case.startswith("kfs"):
            sim = mod.simulation_smoother()
            # The default variates (drawn as float64 and converted to dtype)
            sim.simulate(rng=np.random.default_rng(1234))
        else:
            sim = mod.simulation_smoother(method="cfa")
            variates = np.random.default_rng(1234).standard_normal((k_states, nobs))
            sim.simulate(variates=variates.astype(dtype))
        return mod, sim

    mod, sim = run(dtype)
    _, desired = run(np.float64)
    assert mod.prefix == PREFIX[dtype]
    if case.startswith("kfs"):
        names = [
            "simulated_state",
            "simulated_measurement_disturbance",
            "simulated_state_disturbance",
        ]
    else:
        # (posterior_cov is computed from the Cholesky factor in Python)
        names = ["posterior_mean", "posterior_cov_inv_chol_sparse", "simulated_state"]
    for name in names:
        assert_matches(getattr(sim, name), getattr(desired, name), dtype, name)


@pytest.mark.parametrize(
    "func",
    [
        "reorder_missing_matrix",
        "reorder_missing_vector",
        "copy_missing_matrix",
        "copy_missing_vector",
        "copy_index_matrix",
        "copy_index_vector",
    ],
)
@dtypes
def test_tools_copy_reorder(func, dtype):
    # The helpers only move values, so the real and the imaginary parts are
    # moved as the float64 helper moves real values
    rng = np.random.default_rng(20261123)
    n, nobs = 3, 4
    missing = np.zeros((n, nobs), dtype=bool)
    missing[1, 1] = True
    missing[0, 2] = True
    missing[:, 3] = True
    index = ~missing
    matrix = rng.standard_normal((n, n, nobs))
    vector = rng.standard_normal((n, nobs))
    if np.issubdtype(dtype, np.complexfloating):
        matrix = matrix + 1j * rng.standard_normal((n, n, nobs))
        vector = vector + 1j * rng.standard_normal((n, nobs))
    matrix = matrix.astype(dtype)
    vector = vector.astype(dtype)

    def call(func, matrix, vector, rows, cols, diagonal, time_varying):
        f = getattr(tools, func)
        if func.startswith("reorder"):
            if func.endswith("vector"):
                return f(vector, missing)
            return f(matrix, missing, rows, cols, diagonal)
        source = matrix if func.endswith("matrix") else vector
        if not time_varying:
            source = source[..., :1]
        out = np.zeros_like(matrix if func.endswith("matrix") else vector)
        mask = missing if "missing" in func else index
        if func.endswith("vector"):
            return f(source, out, mask)
        return f(source, out, mask, rows, cols, diagonal)

    # rows, cols, diagonal, time_varying (the copy helpers also accept a
    # time-invariant source)
    if func.endswith("matrix"):
        options = [
            (True, False, False, True),
            (False, True, False, True),
            (True, True, False, True),
            (True, True, True, True),
        ]
    else:
        options = [(False, False, False, True)]
    if func.startswith("copy"):
        options += [option[:3] + (False,) for option in options]
    for option in options:
        out = call(func, matrix, vector, *option)
        assert out.dtype == dtype
        real = call(func, matrix.real.astype(float), vector.real.astype(float), *option)
        assert_allclose(out.real, real, rtol=1e-15, err_msg=str(option))
        if np.issubdtype(dtype, np.complexfloating):
            imag = call(
                func, matrix.imag.astype(float), vector.imag.astype(float), *option
            )
            assert_allclose(out.imag, imag, rtol=1e-15, err_msg=str(option))


@pytest.mark.parametrize("missing", [None, "partial"])
@dtypes
def test_smoothed_state_weights(missing, dtype):
    def run(dtype):
        mod = build(KalmanSmoother, dtype, missing=missing)
        # MLEModel with the same representation, which keeps the dtype
        model = MLEModel(mod.endog.T, k_states=2, k_posdef=2, dtype=dtype)
        for name in mod.shapes:
            if name != "obs":
                model[name] = getattr(mod, name)
        model.ssm.initialize(mod.initialization)
        res = model.smooth([])
        assert model.ssm.prefix == PREFIX.get(dtype, "d")
        return tools.compute_smoothed_state_weights(res)

    weights = run(dtype)
    desired = run(np.float64)
    names = ["weights", "state_intercept_weights", "prior_weights"]
    for name, actual, expected in zip(names, weights, desired, strict=True):
        assert_matches(actual, expected, dtype, name)


@pytest.mark.parametrize(
    "order, time_varying", [(0, False), (1, True)], ids=["order0", "order1-tvtp"]
)
@dtypes
def test_hamilton_filter_kim_smoother(order, time_varying, dtype):
    rng = np.random.default_rng(20261123)
    k_regimes, nobs = 2, 12
    initial_probabilities = np.array([0.4, 0.6])
    n_transition = nobs + order if time_varying else 1
    stay = rng.uniform(0.6, 0.95, size=(k_regimes, n_transition))
    regime_transition = np.empty((k_regimes, k_regimes, n_transition))
    regime_transition[0, 0] = stay[0]
    regime_transition[1, 0] = 1 - stay[0]
    regime_transition[1, 1] = stay[1]
    regime_transition[0, 1] = 1 - stay[1]
    shape = (k_regimes,) * (order + 1) + (nobs,)
    conditional_loglikelihoods = rng.standard_normal(shape) - 1.0

    def run(dtype):
        transition = regime_transition.astype(dtype)
        filtered = cy_hamilton_filter_log(
            initial_probabilities.astype(dtype),
            transition,
            conditional_loglikelihoods.astype(dtype),
            order,
        )
        smoothed = cy_kim_smoother_log(transition, filtered[4], filtered[5])
        return filtered + smoothed

    out = run(dtype)
    desired = run(np.float64)
    for actual, expected in zip(out, desired, strict=True):
        assert_matches(actual, expected, dtype)


@pytest.mark.parametrize(
    "func", ["arma_innovations", "arma_loglike", "arma_loglikeobs"]
)
@dtypes
def test_arma_innovations(func, dtype):
    endog = np.random.default_rng(20261123).standard_normal(20)
    f = getattr(arma_innovations, func)
    out = f(endog, ar_params=[0.5], ma_params=[0.2], sigma2=1.3, prefix=PREFIX[dtype])
    desired = f(endog, ar_params=[0.5], ma_params=[0.2], sigma2=1.3)
    if func == "arma_innovations":
        for actual, expected in zip(out, desired, strict=True):
            assert_matches(actual, expected, dtype)
    else:
        assert_matches(out, desired, dtype)
