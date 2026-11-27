"""
Tests for initialization

Author: Chad Fulton
License: Simplified-BSD
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.linalg import solve_discrete_lyapunov

from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tsa.statespace import sarimax, varmax
from statsmodels.tsa.statespace.initialization import Initialization
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
from statsmodels.tsa.statespace.representation import Representation


def check_initialization(mod, init, a_true, Pinf_true, Pstar_true):
    # Check the Python version
    a, Pinf, Pstar = init(model=mod)
    assert_allclose(a, a_true)
    assert_allclose(Pinf, Pinf_true)
    assert_allclose(Pstar, Pstar_true)

    # Check the Cython version
    mod.ssm._initialize_representation()
    init._initialize_initialization(prefix=mod.ssm.prefix)
    _statespace = mod.ssm._statespace
    _statespace.initialize(init)
    assert_allclose(np.array(_statespace.initial_state), a_true)
    assert_allclose(np.array(_statespace.initial_diffuse_state_cov), Pinf_true)
    assert_allclose(np.array(_statespace.initial_state_cov), Pstar_true)


def test_global_known():
    # Test for global known initialization

    # - 1-dimensional -
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0))

    # Known, mean
    init = Initialization(mod.k_states, "known", constant=[1.5])
    check_initialization(mod, init, [1.5], np.diag([0]), np.diag([0]))

    # Known, covariance
    init = Initialization(mod.k_states, "known", stationary_cov=np.diag([1]))
    check_initialization(mod, init, [0], np.diag([0]), np.diag([1]))

    # Known, both
    init = Initialization(
        mod.k_states, "known", constant=[1.5], stationary_cov=np.diag([1])
    )
    check_initialization(mod, init, [1.5], np.diag([0]), np.diag([1]))

    # - n-dimensional -
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(2, 0, 0))

    # Known, mean
    init = Initialization(mod.k_states, "known", constant=[1.5, -0.2])
    check_initialization(mod, init, [1.5, -0.2], np.diag([0, 0]), np.diag([0, 0]))

    # Known, covariance
    init = Initialization(mod.k_states, "known", stationary_cov=np.diag([1, 4.2]))
    check_initialization(mod, init, [0, 0], np.diag([0, 0]), np.diag([1, 4.2]))

    # Known, both
    init = Initialization(
        mod.k_states, "known", constant=[1.5, -0.2], stationary_cov=np.diag([1, 4.2])
    )
    check_initialization(mod, init, [1.5, -0.2], np.diag([0, 0]), np.diag([1, 4.2]))


def test_global_diffuse():
    # Test for global diffuse initialization

    # - 1-dimensional -
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0))

    init = Initialization(mod.k_states, "diffuse")
    check_initialization(mod, init, [0], np.eye(1), np.diag([0]))

    # - n-dimensional -
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(2, 0, 0))

    init = Initialization(mod.k_states, "diffuse")
    check_initialization(mod, init, [0, 0], np.eye(2), np.diag([0, 0]))


def test_global_approximate_diffuse():
    # Test for global approximate diffuse initialization

    # - 1-dimensional -
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0))

    init = Initialization(mod.k_states, "approximate_diffuse")
    check_initialization(mod, init, [0], np.diag([0]), np.eye(1) * 1e6)

    init = Initialization(mod.k_states, "approximate_diffuse", constant=[1.2])
    check_initialization(mod, init, [1.2], np.diag([0]), np.eye(1) * 1e6)

    init = Initialization(
        mod.k_states, "approximate_diffuse", approximate_diffuse_variance=1e10
    )
    check_initialization(mod, init, [0], np.diag([0]), np.eye(1) * 1e10)

    # - n-dimensional -
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(2, 0, 0))

    init = Initialization(mod.k_states, "approximate_diffuse")
    check_initialization(mod, init, [0, 0], np.diag([0, 0]), np.eye(2) * 1e6)

    init = Initialization(mod.k_states, "approximate_diffuse", constant=[1.2, -0.2])
    check_initialization(mod, init, [1.2, -0.2], np.diag([0, 0]), np.eye(2) * 1e6)

    init = Initialization(
        mod.k_states, "approximate_diffuse", approximate_diffuse_variance=1e10
    )
    check_initialization(mod, init, [0, 0], np.diag([0, 0]), np.eye(2) * 1e10)


def test_global_stationary():
    # Test for global approximate diffuse initialization

    # - 1-dimensional -
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0), trend="c")

    # no intercept
    intercept = 0
    phi = 0.5
    sigma2 = 2.0
    mod.update(np.r_[intercept, phi, sigma2])
    init = Initialization(mod.k_states, "stationary")
    check_initialization(
        mod, init, [0], np.diag([0]), np.eye(1) * sigma2 / (1 - phi**2)
    )

    # intercept
    intercept = 1.2
    phi = 0.5
    sigma2 = 2.0
    mod.update(np.r_[intercept, phi, sigma2])
    init = Initialization(mod.k_states, "stationary")
    check_initialization(
        mod,
        init,
        [intercept / (1 - phi)],
        np.diag([0]),
        np.eye(1) * sigma2 / (1 - phi**2),
    )

    # - n-dimensional -
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(2, 0, 0), trend="c")

    # no intercept
    intercept = 0
    phi = [0.5, -0.2]
    sigma2 = 2.0
    mod.update(np.r_[intercept, phi, sigma2])
    init = Initialization(mod.k_states, "stationary")
    T = np.array([[0.5, 1], [-0.2, 0]])
    Q = np.diag([sigma2, 0])
    desired_cov = solve_discrete_lyapunov(T, Q)
    check_initialization(mod, init, [0, 0], np.diag([0, 0]), desired_cov)

    # intercept
    intercept = 1.2
    phi = [0.5, -0.2]
    sigma2 = 2.0
    mod.update(np.r_[intercept, phi, sigma2])
    init = Initialization(mod.k_states, "stationary")
    desired_intercept = np.linalg.inv(np.eye(2) - T).dot([intercept, 0])
    check_initialization(mod, init, desired_intercept, np.diag([0, 0]), desired_cov)


def test_mixed_basic():
    # Performs a number of tests for setting different initialization for
    # different blocks

    # - 2-dimensional -
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(2, 0, 0))
    phi = [0.5, -0.2]
    sigma2 = 2.0
    mod.update(np.r_[phi, sigma2])

    # known has constant
    init = Initialization(mod.k_states)
    init.set(0, "known", constant=[1.2])

    # > known has constant
    init.set(1, "known", constant=[-0.2])
    check_initialization(mod, init, [1.2, -0.2], np.diag([0, 0]), np.diag([0, 0]))

    # > diffuse
    init.unset(1)
    init.set(1, "diffuse")
    check_initialization(mod, init, [1.2, 0], np.diag([0, 1]), np.diag([0, 0]))

    # > approximate diffuse
    init.unset(1)
    init.set(1, "approximate_diffuse")
    check_initialization(mod, init, [1.2, 0], np.diag([0, 0]), np.diag([0, 1e6]))

    # > stationary
    init.unset(1)
    init.set(1, "stationary")
    check_initialization(mod, init, [1.2, 0], np.diag([0, 0]), np.diag([0, 0]))

    # known has cov
    init = Initialization(mod.k_states)
    init.set(0, "known", stationary_cov=np.diag([1]))
    init.set(1, "diffuse")
    check_initialization(mod, init, [0, 0], np.diag([0, 1]), np.diag([1, 0]))

    # known has both
    init = Initialization(mod.k_states)
    init.set(0, "known", constant=[1.2], stationary_cov=np.diag([1]))
    init.set(1, "diffuse")
    check_initialization(mod, init, [1.2, 0], np.diag([0, 1]), np.diag([1, 0]))

    # - 3-dimensional -
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(3, 0, 0))

    # known has constant
    init = Initialization(mod.k_states)
    init.set((0, 2), "known", constant=[1.2, -0.2])
    init.set(2, "diffuse")
    check_initialization(
        mod, init, [1.2, -0.2, 0], np.diag([0, 0, 1]), np.diag([0, 0, 0])
    )

    # known has cov
    init = Initialization(mod.k_states)
    init.set((0, 2), "known", stationary_cov=np.diag([1, 4.2]))
    init.set(2, "diffuse")
    check_initialization(mod, init, [0, 0, 0], np.diag([0, 0, 1]), np.diag([1, 4.2, 0]))

    # known has both
    init = Initialization(mod.k_states)
    init.set((0, 2), "known", constant=[1.2, -0.2], stationary_cov=np.diag([1, 4.2]))
    init.set(2, "diffuse")
    check_initialization(
        mod, init, [1.2, -0.2, 0], np.diag([0, 0, 1]), np.diag([1, 4.2, 0])
    )


def test_mixed_stationary():
    # More specific tests when one or more blocks are initialized as stationary
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(2, 1, 0))
    phi = [0.5, -0.2]
    sigma2 = 2.0
    mod.update(np.r_[phi, sigma2])

    init = Initialization(mod.k_states)
    init.set(0, "diffuse")
    init.set((1, 3), "stationary")
    desired_cov = np.zeros((3, 3))
    T = np.array([[0.5, 1], [-0.2, 0]])
    Q = np.diag([sigma2, 0])
    desired_cov[1:, 1:] = solve_discrete_lyapunov(T, Q)
    check_initialization(mod, init, [0, 0, 0], np.diag([1, 0, 0]), desired_cov)

    init.clear()
    init.set(0, "diffuse")
    init.set(1, "stationary")
    init.set(2, "approximate_diffuse")
    T = np.array([[0.5]])
    Q = np.diag([sigma2])
    desired_cov = np.diag([0, np.squeeze(solve_discrete_lyapunov(T, Q)), 1e6])
    check_initialization(mod, init, [0, 0, 0], np.diag([1, 0, 0]), desired_cov)

    init.clear()
    init.set(0, "diffuse")
    init.set(1, "stationary")
    init.set(2, "stationary")
    desired_cov[2, 2] = 0
    check_initialization(mod, init, [0, 0, 0], np.diag([1, 0, 0]), desired_cov)

    # Test with a VAR model
    endog = np.zeros((10, 2))
    mod = varmax.VARMAX(
        endog,
        order=(1, 0),
    )
    intercept = [1.5, -0.1]
    transition = np.array([[0.5, -0.2], [0.1, 0.8]])
    cov = np.array([[1.2, -0.4], [-0.4, 0.4]])
    tril = np.tril_indices(2)
    params = np.r_[intercept, transition.ravel(), np.linalg.cholesky(cov)[tril]]
    mod.update(params)

    # > stationary, global
    init = Initialization(mod.k_states, "stationary")
    desired_intercept = np.linalg.solve(np.eye(2) - transition, intercept)
    desired_cov = solve_discrete_lyapunov(transition, cov)
    check_initialization(mod, init, desired_intercept, np.diag([0, 0]), desired_cov)

    # > diffuse, global
    init.set(None, "diffuse")
    check_initialization(mod, init, [0, 0], np.eye(2), np.diag([0, 0]))

    # > stationary, individually
    init.unset(None)
    init.set(0, "stationary")
    init.set(1, "stationary")
    a, Pinf, Pstar = init(model=mod)
    desired_intercept = [
        intercept[0] / (1 - transition[0, 0]),
        intercept[1] / (1 - transition[1, 1]),
    ]
    desired_cov = np.diag(
        [
            cov[0, 0] / (1 - transition[0, 0] ** 2),
            cov[1, 1] / (1 - transition[1, 1] ** 2),
        ]
    )
    check_initialization(mod, init, desired_intercept, np.diag([0, 0]), desired_cov)


def test_nested():
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(6, 0, 0))
    phi = [0.5, -0.2, 0.1, 0.0, 0.1, 0.0]
    sigma2 = 2.0
    mod.update(np.r_[phi, sigma2])

    # Create the initialization object as a series of nested objects
    init1_1 = Initialization(3)
    init1_1_1 = Initialization(2, "stationary")
    init1_1_2 = Initialization(
        1, "approximate_diffuse", approximate_diffuse_variance=1e9
    )
    init1_1.set((0, 2), init1_1_1)
    init1_1.set(2, init1_1_2)

    init1_2 = Initialization(3)
    init1_2_1 = Initialization(1, "known", constant=[1], stationary_cov=[[2.0]])
    init1_2.set(0, init1_2_1)
    init1_2_2 = Initialization(1, "diffuse")
    init1_2.set(1, init1_2_2)
    init1_2_3 = Initialization(1, "approximate_diffuse")
    init1_2.set(2, init1_2_3)

    init = Initialization(6)
    init.set((0, 3), init1_1)
    init.set((3, 6), init1_2)

    # Check the output
    desired_cov = np.zeros((6, 6))
    T = np.array([[0.5, 1], [-0.2, 0]])
    Q = np.array([[sigma2, 0], [0, 0]])
    desired_cov[:2, :2] = solve_discrete_lyapunov(T, Q)
    desired_cov[2, 2] = 1e9
    desired_cov[3, 3] = 2.0
    desired_cov[5, 5] = 1e6
    check_initialization(
        mod, init, [0, 0, 0, 1, 0, 0], np.diag([0, 0, 0, 0, 1, 0]), desired_cov
    )


def test_invalid():
    # Invalid initializations (also tests for some invalid calls to set)
    with pytest.raises(ValueError):
        Initialization(5, "")
    with pytest.raises(ValueError):
        Initialization(5, "stationary", constant=[1, 2])
    with pytest.raises(ValueError):
        Initialization(5, "stationary", stationary_cov=[1, 2])
    with pytest.raises(ValueError):
        Initialization(5, "known")
    with pytest.raises(ValueError):
        Initialization(5, "known", constant=[1])
    with pytest.raises(ValueError):
        Initialization(5, "known", stationary_cov=[0])

    # Invalid set() / unset() calls
    init = Initialization(5)
    with pytest.raises(ValueError):
        init.set(-1, "diffuse")
    with pytest.raises(ValueError):
        init.unset(-1)
    with pytest.raises(ValueError):
        init.set(5, "diffuse")
    with pytest.raises(ValueError):
        init.set("x", "diffuse")
    with pytest.raises(ValueError):
        init.unset("x")
    with pytest.raises(ValueError):
        init.set((1, 2, 3), "diffuse")
    with pytest.raises(ValueError):
        init.unset((1, 2, 3))
    init.set(None, "diffuse")
    with pytest.raises(ValueError):
        init.set(1, "diffuse")
    init.clear()
    init.set(1, "diffuse")
    with pytest.raises(ValueError):
        init.set(None, "stationary")

    init.clear()
    with pytest.raises(ValueError):
        init.unset(1)

    # Invalid __call__
    init = Initialization(2)
    with pytest.raises(ValueError):
        init()
    init = Initialization(2, "stationary")
    with pytest.raises(ValueError):
        init()


def test_uses_stationary():
    assert Initialization(2, "stationary")._uses_stationary()
    assert not Initialization(2, "diffuse")._uses_stationary()
    assert not Initialization(2, "known", constant=[0, 0])._uses_stationary()
    assert not Initialization(2)._uses_stationary()

    init = Initialization(3)
    init.set(0, "diffuse")
    assert not init._uses_stationary()
    init.set((1, 2), "stationary")
    assert init._uses_stationary()

    nested = Initialization(3)
    nested.set((0, 2), init_inner := Initialization(2))
    init_inner.set(0, "known", constant=[0.0])
    init_inner.set(1, "stationary")
    nested.set(2, "diffuse")
    assert nested._uses_stationary()


def test_stationary_blocks():
    def blocks(init):
        return [list(index) for index in init._stationary_blocks()]

    assert blocks(Initialization(3, "stationary")) == [[0, 1, 2]]
    assert blocks(Initialization(3, "diffuse")) == []
    assert blocks(Initialization(3)) == []

    nested = Initialization(5)
    nested.set(0, "diffuse")
    init_inner = Initialization(2)
    init_inner.set(0, "known", constant=[0.0])
    init_inner.set(1, "stationary")
    nested.set((1, 3), init_inner)
    nested.set((3, 5), "stationary")
    assert blocks(nested) == [[2], [3, 4]]


@pytest.mark.parametrize("dtype", [np.float32, np.complex64])
def test_stationary_single_precision(dtype):
    # The stationary initialization of a single precision model solves the
    # discrete Lyapunov equation with the "s" or "c" Cython routines. The
    # complex64 version raised "Buffer dtype mismatch, expected 'float32_t'
    # but got 'double'".
    def build(dtype):
        endog = np.random.default_rng(1234).standard_normal((15, 2))
        kf = KalmanFilter(k_endog=2, k_states=2, dtype=dtype)
        kf.bind(endog.astype(dtype))
        kf["design"] = np.array([[1.0, 0.0], [0.5, 1.0]], dtype=dtype)
        kf["obs_intercept"] = np.array([[0.1], [0.3]], dtype=dtype)
        kf["obs_cov"] = np.array([[0.8, 0.2], [0.2, 0.5]], dtype=dtype)
        kf["transition"] = transition.astype(dtype)
        kf["state_intercept"] = state_intercept.astype(dtype)
        kf["selection"] = np.eye(2, dtype=dtype)
        kf["state_cov"] = state_cov.astype(dtype)
        kf.initialize_stationary()
        return kf

    transition = np.array([[0.5, 0.1], [-0.2, 0.3]])
    state_intercept = np.array([[0.2], [-0.1]])
    state_cov = np.array([[1.0, 0.3], [0.3, 0.6]])
    kf = build(dtype)
    res = kf.filter()
    desired = build(np.float64).filter()
    assert kf.prefix == {np.float32: "s", np.complex64: "c"}[dtype]
    assert res.prefix == kf.prefix

    # Single precision: the float64 results agree to about 4e-6 (relative)
    rtol = 1e-4
    initial_state = np.linalg.solve(np.eye(2) - transition, state_intercept[:, 0])
    initial_state_cov = solve_discrete_lyapunov(transition, state_cov)
    names = [
        "initial_state",
        "initial_state_cov",
        "llf_obs",
        "filtered_state",
        "filtered_state_cov",
        "forecasts",
        "forecasts_error_cov",
    ]
    for name in names:
        actual = getattr(res, name)
        assert actual.dtype == dtype
        assert_allclose(actual.real, getattr(desired, name), rtol=rtol)
        # The imaginary part is zero in exact arithmetic. The complex Schur
        # decomposition in the Lyapunov solver leaves rounding errors of the
        # order of the single precision epsilon times the size of the values
        # (6e-8 here), hence the atol relative to the real part.
        atol = 1e-5 * np.abs(actual.real).max()
        assert_allclose(actual.imag, 0, atol=atol)
    assert_allclose(res.initial_state.real, initial_state, rtol=rtol)
    assert_allclose(res.initial_state_cov.real, initial_state_cov, rtol=rtol)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("scale", [1.0, 1e295, 1e300])
def test_stationary_cov_large_state_cov(scale, dtype):
    # For a state covariance of about 1e295 or more, LAPACK's Sylvester
    # solver scales its solution down to avoid overflow. The compiled
    # stationary initialization multiplied by that factor instead of
    # dividing, which gave a stationary covariance of about zero.
    transition = np.array([[0.5, 0.1], [0.0, 0.3]])
    kf = KalmanFilter(k_endog=1, k_states=2, dtype=dtype)
    kf.bind(np.zeros(5, dtype=dtype))
    kf["design"] = np.array([[1.0, 0.0]], dtype=dtype)
    kf["transition"] = transition.astype(dtype)
    kf["selection"] = np.eye(2, dtype=dtype)
    kf["state_cov"] = np.eye(2, dtype=dtype) * scale
    kf["obs_cov"] = np.eye(1, dtype=dtype)
    kf.initialize_stationary()
    res = kf.filter()

    # vec(P) = (I - T kron T)^{-1} vec(Q), here with Q = scale * I
    vec_cov = np.linalg.solve(
        np.eye(4) - np.kron(transition, transition), np.eye(2).ravel(order="F")
    )
    desired = vec_cov.reshape(2, 2, order="F")
    assert_allclose(res.initial_state_cov / scale, desired, rtol=1e-10)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"Pinf": np.eye(2), "A": np.eye(2)}, "Cannot specify both the diffuse"),
        ({"a": [1.0], "Pstar": np.eye(2)}, "Must provide constant initialization"),
        ({"R0": np.eye(2)}, "If specifying either of R0 or Q0"),
        ({"Q0": np.eye(2)}, "If specifying either of R0 or Q0"),
        (
            {"Pstar": np.eye(2), "Pinf": np.diag([1.0, 0.0])},
            "The state at position 0 was specified as diffuse in Pinf",
        ),
        ({"Pinf": np.diag([1.0, 0.0])}, "Must provide initial covariance matrix"),
    ],
)
def test_from_components_invalid(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Initialization.from_components(2, **kwargs)


@pytest.mark.parametrize("kwargs", [{"Pinf": np.eye(2)}, {"A": np.eye(2)}])
def test_from_components_all_diffuse(kwargs):
    # With every state diffuse no stationary covariance is needed, and the
    # result is the global diffuse initialization
    init = Initialization.from_components(2, **kwargs)
    desired = Initialization(2, "diffuse")()
    for actual_part, desired_part in zip(init(), desired, strict=True):
        assert_allclose(actual_part, desired_part)
    assert_allclose(desired[1], np.eye(2))


def test_set_global_approximate_diffuse_variance():
    init = Initialization(2)
    init.set(None, "approximate_diffuse", approximate_diffuse_variance=50.0)
    a, Pinf, Pstar = init()
    assert_allclose(Pstar, 50.0 * np.eye(2))
    assert_allclose(init.approximate_diffuse_variance, 50.0)

    init = Initialization(2)
    with pytest.raises(ValueError, match="`approximate_diffuse_variance` can only"):
        init.set(None, "known", constant=[0, 0], approximate_diffuse_variance=50.0)


def test_set_block_approximate_diffuse_variance():
    init = Initialization(3)
    init.set((0, 2), "approximate_diffuse", approximate_diffuse_variance=7.0)
    init.set(2, "known", constant=[3.0], stationary_cov=[[2.0]])
    a, Pinf, Pstar = init()
    assert_allclose(a, [0.0, 0.0, 3.0])
    assert_allclose(Pstar, np.diag([7.0, 7.0, 2.0]))
    # The variance of the whole object is unchanged
    assert_allclose(init.approximate_diffuse_variance, 1e6)


def test_set_diffuse_with_constant():
    # A constant has no effect under exact diffuse initialization: the
    # loglikelihood equals the one with a zero constant
    endog = np.random.default_rng(112233).standard_normal((20, 2))
    llf = {}
    for constant in (None, [1.0, -2.0]):
        init = Initialization(2)
        if constant is None:
            init.set(None, "diffuse")
        else:
            with pytest.warns(SpecificationWarning, match="Constant values provided"):
                init.set(None, "diffuse", constant=constant)
        mod = KalmanFilter(np.ascontiguousarray(endog), k_states=2)
        mod["design"] = np.array([[1.0, 0.2], [0.3, 1.0]])
        mod["transition"] = np.array([[0.5, 0.1], [0.0, 0.3]])
        mod["selection"] = np.eye(2)
        mod["state_cov"] = np.eye(2)
        mod["obs_cov"] = np.eye(2)
        mod.initialize(init)
        llf[constant is None] = mod.loglike()
        assert_allclose(init()[1], np.eye(2))
    assert_allclose(llf[False], llf[True], rtol=1e-12)


def test_unset_empty_index():
    # An empty index changes nothing
    init = Initialization(2, "diffuse")
    init.unset((1, 1))
    assert init.initialization_type == "diffuse"
    assert_allclose(init()[1], np.eye(2))


def test_call_stationary_unit_root():
    mod = Representation(1, k_states=2)
    mod["transition"] = np.diag([1.0, 0.5])
    init = Initialization(2, "stationary")
    with pytest.raises(ValueError, match="Transition equation is not stationary"):
        init(model=mod)


def test_stationary_initialization_without_state_disturbance():
    # With k_posdef=0 the stationary covariance solves P = T P T', so P = 0,
    # and the mean is (I - T)^{-1} c: the same as a known initialization
    endog = np.random.default_rng(445566).standard_normal(30)
    transition = np.array([[0.5, 0.1], [0.0, 0.3]])
    state_intercept = np.array([0.4, -0.2])
    mod = KalmanFilter(k_endog=1, k_states=2, k_posdef=0)
    mod.bind(endog)
    mod["design"] = np.array([[1.0, 0.5]])
    mod["transition"] = transition
    mod["state_intercept"] = state_intercept
    mod["obs_cov"] = np.array([[1.0]])
    mod.initialize_stationary()
    res = mod.filter()
    mean = np.linalg.solve(np.eye(2) - transition, state_intercept)
    assert_allclose(res.initial_state, mean)
    # atol: zero by construction; a nonzero R Q R' gives entries of order one
    assert_allclose(res.initial_state_cov, np.zeros((2, 2)), atol=1e-14)

    mod.initialize_known(mean, np.zeros((2, 2)))
    assert_allclose(res.llf, mod.filter().llf, rtol=1e-12)


@pytest.mark.parametrize(
    "to_key", [int, np.int64, np.int32, np.array], ids=["int", "int64", "int32", "0d"]
)
def test_set_unset_int_like_index(to_key):
    # A NumPy integer or a 0-d integer array selects a state, as an int does.
    # A 0-d array raised a ValueError.
    init = Initialization(2)
    init.set(to_key(0), "known", constant=[1.2])
    init.set(to_key(1), "diffuse")
    constant, diffuse_cov, stationary_cov = init()
    assert_allclose(constant, [1.2, 0.0], rtol=1e-15)
    assert_allclose(diffuse_cov, np.diag([0.0, 1.0]), rtol=1e-15)
    assert_allclose(stationary_cov, np.zeros((2, 2)), rtol=1e-15)

    init.unset(to_key(1))
    init.set(1, "known", constant=[-0.3])
    constant, diffuse_cov, _ = init()
    assert_allclose(constant, [1.2, -0.3], rtol=1e-15)
    assert_allclose(diffuse_cov, np.zeros((2, 2)), rtol=1e-15)


@pytest.mark.parametrize(
    "index", [True, np.True_, 1.0], ids=["True", "np.True_", "float"]
)
@pytest.mark.parametrize("method", ["set", "unset"])
def test_set_unset_bool_float_index(index, method):
    # A bool or a float does not select a state: True selected state 1
    init = Initialization(2)
    init.set(1, "diffuse")
    with pytest.raises(ValueError, match="Invalid index"):
        if method == "set":
            init.set(index, "diffuse")
        else:
            init.unset(index)


def _set_object_reference(kind):
    if kind == "known":
        return Initialization(
            2, "known", constant=[1.0, 2.0], stationary_cov=[[2.0, 0.5], [0.5, 1.0]]
        )
    elif kind == "approximate_diffuse":
        return Initialization(
            2, "approximate_diffuse", approximate_diffuse_variance=50.0
        )
    elif kind == "mixed":
        init = Initialization(2)
        init.set(0, "diffuse")
        init.set(1, "stationary")
        return init
    return Initialization(2, kind)


@pytest.mark.parametrize(
    "kind", ["known", "diffuse", "approximate_diffuse", "stationary", "mixed"]
)
@pytest.mark.parametrize("use_constructor", [False, True])
def test_set_initialization_object_all_states(kind, use_constructor):
    # An Initialization object given for all the states is used as their
    # initialization, as it is for a block of states
    mod = sarimax.SARIMAX(np.zeros(10), order=(2, 0, 0))
    mod.update([0.5, -0.2, 1.0])
    reference = _set_object_reference(kind)
    desired = reference(model=mod.ssm)
    if use_constructor:
        init = Initialization(2, reference)
    else:
        init = Initialization(2)
        init.set(None, reference)
    assert init.initialized
    check_initialization(mod, init, *desired)


def test_set_initialization_object_replaces_global():
    known = {"constant": [1.0, 2.0], "stationary_cov": [[2.0, 0.5], [0.5, 1.0]]}
    init = Initialization(2, "known", **known)
    init.set(None, Initialization(2, "diffuse"))
    assert init.initialization_type is None
    a, Pinf, Pstar = init()
    assert_allclose(a, np.zeros(2))
    assert_allclose(Pinf, np.eye(2))
    assert_allclose(Pstar, np.zeros((2, 2)))

    # A global initialization replaces the block of all the states
    init.set(None, "known", **known)
    assert init.blocks == {}
    a, Pinf, Pstar = init()
    assert_allclose(a, known["constant"])
    assert_allclose(Pinf, np.zeros((2, 2)))
    assert_allclose(Pstar, known["stationary_cov"])
    init.unset(None)
    assert not init.initialized

    init.set(None, Initialization(2, "diffuse"))
    init.unset(None)
    assert not init.initialized
    assert init.blocks == {}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initialization_type": "foo"},
        {"initialization_type": "known"},
        {"initialization_type": "known", "stationary_cov": np.eye(3)},
        {
            "initialization_type": "known",
            "constant": [1.0, 2.0, 3.0],
            "stationary_cov": np.eye(2),
        },
        {
            "initialization_type": "approximate_diffuse",
            "approximate_diffuse_variance": 50.0,
            "constant": [1.0],
        },
        {"initialization_type": "stationary", "constant": [1.0, 2.0]},
        {"initialization_type": "diffuse", "stationary_cov": np.eye(2)},
    ],
)
@pytest.mark.parametrize("previous", [None, "approximate_diffuse"])
def test_failed_set_leaves_initialization(kwargs, previous):
    # A call that raises does not change the initialization
    init = Initialization(2)
    if previous is not None:
        init.set(None, previous, approximate_diffuse_variance=7.0)
    initialization_type = init.initialization_type
    initialized = init.initialized
    variance = init.approximate_diffuse_variance
    constant = init.constant.copy()
    stationary_cov = init.stationary_cov.copy()

    with pytest.raises(ValueError):
        init.set(None, **kwargs)
    assert init.initialization_type == initialization_type
    assert init.initialized == initialized
    assert_allclose(init.approximate_diffuse_variance, variance, rtol=1e-12)
    assert_allclose(init.constant, constant, rtol=1e-12)
    assert_allclose(init.stationary_cov, stationary_cov, rtol=1e-12)
    if previous is not None:
        # The previous initialization, with its variance, is still used
        assert_allclose(init()[2], 7.0 * np.eye(2), rtol=1e-12)


@pytest.mark.parametrize(
    "index, k_block, k_object",
    [((0, 2), 2, 1), ((0, 1), 1, 2), (1, 1, 2), (None, 3, 1), (None, 3, 2)],
    ids=["block-of-2-one", "block-of-1-two", "int-two", "all-one", "all-two"],
)
def test_set_object_size_mismatch(index, k_block, k_object):
    # An Initialization object must have as many states as the block. An
    # object with one state was used for every state of the block when the
    # initialization was computed in Python (and for only the first state by
    # the Kalman filter), and a larger object raised only when it was used.
    init = Initialization(3)
    block = Initialization(
        k_object,
        "known",
        constant=np.arange(1.0, k_object + 1),
        stationary_cov=np.eye(k_object),
    )
    match = f"has {k_object} states, but the block of states .* has {k_block}"
    with pytest.raises(ValueError, match=match):
        init.set(index, block)
    # Nothing was set
    assert init.blocks == {}
    assert init.initialization_type is None
    assert not init.initialized


def test_set_object_size_mismatch_keeps_global():
    # The check comes before a global initialization is replaced
    init = Initialization(2)
    init.set(None, "approximate_diffuse", approximate_diffuse_variance=7.0)
    with pytest.raises(ValueError, match="has 1 states"):
        init.set(None, Initialization(1, "diffuse"))
    assert init.initialization_type == "approximate_diffuse"
    assert init.blocks == {}
    assert_allclose(init()[2], 7.0 * np.eye(2), rtol=1e-12)


def test_set_object_size_match():
    # An object with as many states as the block initializes the block, in
    # Python and in the Kalman filter, as the block set with the same values
    mod = sarimax.SARIMAX(np.zeros(10), order=(3, 0, 0))
    cov = np.array([[4.0, 1.0], [1.0, 3.0]])
    init = Initialization(3)
    init.set((0, 2), Initialization(2, "known", constant=[1.0, 2.0], stationary_cov=cov))
    init.set(2, "diffuse")

    direct = Initialization(3)
    direct.set((0, 2), "known", constant=[1.0, 2.0], stationary_cov=cov)
    direct.set(2, "diffuse")
    a, Pinf, Pstar = direct()
    assert_allclose(a, [1.0, 2.0, 0.0], rtol=1e-12)
    check_initialization(mod, init, a, Pinf, Pstar)


@pytest.mark.parametrize("method", ["set", "unset"])
@pytest.mark.parametrize(
    "index", [2, np.int64(2), -1], ids=["k_states", "int64-k_states", "negative"]
)
def test_set_unset_index_out_of_range(index, method):
    # unset accepted an index equal to k_states and did nothing, since the
    # slice of states it selected was empty. set raised.
    init = Initialization(2)
    init.set(1, "diffuse")
    with pytest.raises(ValueError, match="Invalid index"):
        if method == "set":
            init.set(index, "diffuse")
        else:
            init.unset(index)
    assert list(init.blocks) == [(1,)]
    init.unset(1)
    assert list(init.blocks) == []
