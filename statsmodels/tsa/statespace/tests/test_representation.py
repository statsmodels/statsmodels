"""
Tests for python wrapper of state space representation and filtering

Author: Chad Fulton
License: Simplified-BSD

References
----------

Kim, Chang-Jin, and Charles R. Nelson. 1999.
"State-Space Models with Regime Switching:
Classical and Gibbs-Sampling Approaches with Applications".
MIT Press Books. The MIT Press.
"""
from pathlib import Path
import warnings

import numpy as np
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_equal,
)
import pandas as pd
import pytest

from statsmodels.tsa.statespace import sarimax, tools
from statsmodels.tsa.statespace.initialization import Initialization
from statsmodels.tsa.statespace.kalman_filter import (
    FilterResults,
    KalmanFilter,
    PredictionResults,
)
from statsmodels.tsa.statespace.kalman_smoother import KalmanSmoother
from statsmodels.tsa.statespace.representation import (
    FrozenRepresentation,
    Representation,
)
from statsmodels.tsa.statespace.simulation_smoother import SimulationSmoother

from .results import results_kalman_filter

current_path = Path(__file__).resolve().parent

clark1989_path = Path("results").joinpath("results_clark1989_R.csv")
clark1989_results = pd.read_csv(Path(current_path).joinpath(clark1989_path))


class Clark1987:
    """
    Clark's (1987) univariate unobserved components model of real GDP (as
    presented in Kim and Nelson, 1999)

    Test data produced using GAUSS code described in Kim and Nelson (1999) and
    found at http://econ.korea.ac.kr/~cjkim/SSMARKOV.htm

    See `results.results_kalman_filter` for more information.
    """

    @classmethod
    def setup_class(cls, dtype=float, **kwargs):
        cls.true = results_kalman_filter.uc_uni
        cls.true_states = pd.DataFrame(cls.true["states"])

        # GDP, Quarterly, 1947.1 - 1995.3
        data = pd.DataFrame(
            cls.true["data"],
            index=pd.date_range("1947-01-01", "1995-07-01", freq="QS"),
            columns=["GDP"],
        )
        data["lgdp"] = np.log(data["GDP"])

        # Construct the statespace representation, with the data and every
        # matrix in dtype (the matrices below are set in place)
        k_states = 4
        cls.dtype = np.dtype(dtype)
        cls.model = KalmanFilter(k_endog=1, k_states=k_states, dtype=dtype, **kwargs)
        cls.model.bind(data["lgdp"].values.astype(dtype))

        cls.model.design[:, :, 0] = [1, 1, 0, 0]
        cls.model.transition[
            ([0, 0, 1, 1, 2, 3], [0, 3, 1, 2, 1, 3], [0, 0, 0, 0, 0, 0])
        ] = [1, 1, 0, 0, 1, 1]
        cls.model.selection = np.eye(cls.model.k_states, dtype=dtype)

        # Update matrices with given parameters
        sigma_v, sigma_e, sigma_w, phi_1, phi_2 = np.array(cls.true["parameters"])
        cls.model.transition[([1, 1], [1, 2], [0, 0])] = [phi_1, phi_2]
        cls.model.state_cov[
            np.diag_indices(k_states) + (np.zeros(k_states, dtype=int),)
        ] = [sigma_v**2, sigma_e**2, 0, sigma_w**2]

        # Initialization
        initial_state = np.zeros((k_states,))
        initial_state_cov = np.eye(k_states) * 100

        # Initialization: modification
        initial_state_cov = np.dot(
            np.dot(cls.model.transition[:, :, 0], initial_state_cov),
            cls.model.transition[:, :, 0].T,
        )
        cls.model.initialize_known(initial_state, initial_state_cov)

    @classmethod
    def run_filter(cls):
        # Filter the data
        return cls.model.filter()

    def test_dtype(self):
        # The model and the filter output use the dtype of the class
        assert self.model.dtype == self.dtype
        assert self.results.prefix == self.model.prefix
        assert self.results.llf_obs.dtype == self.dtype
        state = self.results.filtered_state
        assert state.dtype == self.dtype
        if np.iscomplexobj(state):
            # The imaginary part is zero in exact arithmetic; allow rounding
            # errors relative to the size of the states
            atol = 100 * np.finfo(self.dtype).eps * np.abs(state.real).max()
            assert_allclose(state.imag, 0, atol=atol)

    def test_loglike(self):
        assert_almost_equal(
            self.results.llf_obs[self.true["start"] :].sum(), self.true["loglike"], 5
        )

    def test_filtered_state(self):
        assert_almost_equal(
            self.results.filtered_state[0][self.true["start"] :],
            self.true_states.iloc[:, 0],
            4,
        )
        assert_almost_equal(
            self.results.filtered_state[1][self.true["start"] :],
            self.true_states.iloc[:, 1],
            4,
        )
        assert_almost_equal(
            self.results.filtered_state[3][self.true["start"] :],
            self.true_states.iloc[:, 2],
            4,
        )


class Clark1987SinglePrecision(Clark1987):
    """
    Tolerances for single precision

    In single precision the loglikelihood differs from the published value by
    1.8e-4 (relative) and the filtered states by at most 9.1e-4 (absolute),
    as in test_kalman.py, whose single precision tolerances these are.
    """

    def test_loglike(self):
        assert_allclose(
            self.results.llf_obs[self.true["start"] :].sum(),
            self.true["loglike"],
            rtol=1e-3,
        )

    def test_filtered_state(self):
        start = self.true["start"]
        for i, j in [(0, 0), (1, 1), (3, 2)]:
            assert_allclose(
                self.results.filtered_state[i][start:],
                self.true_states.iloc[:, j],
                atol=1e-2,
            )


class TestClark1987Single(Clark1987SinglePrecision):
    """
    Basic single precision test for the loglikelihood and filtered states.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class(dtype=np.float32, conserve_memory=0)
        cls.results = cls.run_filter()


class TestClark1987Double(Clark1987):
    """
    Basic double precision test for the loglikelihood and filtered states.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class(dtype=float, conserve_memory=0)
        cls.results = cls.run_filter()


class TestClark1987SingleComplex(Clark1987SinglePrecision):
    """
    Basic single precision complex test for the loglikelihood and filtered
    states.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class(dtype=np.complex64, conserve_memory=0)
        cls.results = cls.run_filter()


class TestClark1987DoubleComplex(Clark1987):
    """
    Basic double precision complex test for the loglikelihood and filtered
    states.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class(dtype=complex, conserve_memory=0)
        cls.results = cls.run_filter()


class TestClark1987Conserve(Clark1987):
    """
    Memory conservation test for the loglikelihood and filtered states.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class(dtype=float, conserve_memory=0x01 | 0x02)
        cls.results = cls.run_filter()


class Clark1987Forecast(Clark1987):
    """
    Forecasting test for the loglikelihood and filtered states.
    """

    @classmethod
    def setup_class(cls, dtype=float, nforecast=100, conserve_memory=0):
        super().setup_class(dtype=dtype, conserve_memory=conserve_memory)
        cls.nforecast = nforecast

        # Add missing observations to the end (to forecast)
        cls.model.endog = np.array(
            np.r_[cls.model.endog[0, :], [np.nan] * nforecast],
            ndmin=2,
            dtype=dtype,
            order="F",
        )
        cls.model.nobs = cls.model.endog.shape[1]

    def test_filtered_state(self):
        assert_almost_equal(
            self.results.filtered_state[0][self.true["start"] : -self.nforecast],
            self.true_states.iloc[:, 0],
            4,
        )
        assert_almost_equal(
            self.results.filtered_state[1][self.true["start"] : -self.nforecast],
            self.true_states.iloc[:, 1],
            4,
        )
        assert_almost_equal(
            self.results.filtered_state[3][self.true["start"] : -self.nforecast],
            self.true_states.iloc[:, 2],
            4,
        )


class TestClark1987ForecastDouble(Clark1987Forecast):
    """
    Basic double forecasting test for the loglikelihood and filtered states.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.results = cls.run_filter()


class TestClark1987ForecastDoubleComplex(Clark1987Forecast):
    """
    Basic double complex forecasting test for the loglikelihood and filtered
    states.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class(dtype=complex)
        cls.results = cls.run_filter()


class TestClark1987ForecastConserve(Clark1987Forecast):
    """
    Memory conservation forecasting test for the loglikelihood and filtered
    states.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class(dtype=float, conserve_memory=0x01 | 0x02)
        cls.results = cls.run_filter()


class TestClark1987ConserveAll(Clark1987):
    """
    Memory conservation forecasting test for the loglikelihood and filtered
    states.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class(dtype=float, conserve_memory=0x01 | 0x02 | 0x04 | 0x08)
        cls.model.loglikelihood_burn = cls.true["start"]
        cls.results = cls.run_filter()

    def test_loglike(self):
        assert_almost_equal(self.results.llf, self.true["loglike"], 5)

    def test_filtered_state(self):
        end = self.true_states.shape[0]
        assert_almost_equal(
            self.results.filtered_state[0][-1], self.true_states.iloc[end - 1, 0], 4
        )
        assert_almost_equal(
            self.results.filtered_state[1][-1], self.true_states.iloc[end - 1, 1], 4
        )


class Clark1989:
    """
    Clark's (1989) bivariate unobserved components model of real GDP (as
    presented in Kim and Nelson, 1999)

    Tests two-dimensional observation data.

    Test data produced using GAUSS code described in Kim and Nelson (1999) and
    found at http://econ.korea.ac.kr/~cjkim/SSMARKOV.htm

    See `results.results_kalman_filter` for more information.
    """

    @classmethod
    def setup_class(cls, dtype=float, **kwargs):
        cls.true = results_kalman_filter.uc_bi
        cls.true_states = pd.DataFrame(cls.true["states"])

        # GDP and Unemployment, Quarterly, 1948.1 - 1995.3
        data = pd.DataFrame(
            cls.true["data"],
            index=pd.date_range("1947-01-01", "1995-07-01", freq="QS"),
            columns=["GDP", "UNEMP"],
        )[4:]
        data["GDP"] = np.log(data["GDP"])
        data["UNEMP"] = data["UNEMP"] / 100

        k_states = 6
        cls.model = KalmanFilter(k_endog=2, k_states=k_states, **kwargs)
        cls.model.bind(np.ascontiguousarray(data.values))

        # Statespace representation
        cls.model.design[:, :, 0] = [[1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]]
        cls.model.transition[
            (
                [0, 0, 1, 1, 2, 3, 4, 5],
                [0, 4, 1, 2, 1, 2, 4, 5],
                [0, 0, 0, 0, 0, 0, 0, 0],
            )
        ] = [1, 1, 0, 0, 1, 1, 1, 1]
        cls.model.selection = np.eye(cls.model.k_states)

        # Update matrices with given parameters
        (
            sigma_v,
            sigma_e,
            sigma_w,
            sigma_vl,
            sigma_ec,
            phi_1,
            phi_2,
            alpha_1,
            alpha_2,
            alpha_3,
        ) = np.array(
            cls.true["parameters"],
        )
        cls.model.design[([1, 1, 1], [1, 2, 3], [0, 0, 0])] = [
            alpha_1,
            alpha_2,
            alpha_3,
        ]
        cls.model.transition[([1, 1], [1, 2], [0, 0])] = [phi_1, phi_2]
        cls.model.obs_cov[1, 1, 0] = sigma_ec**2
        cls.model.state_cov[
            np.diag_indices(k_states) + (np.zeros(k_states, dtype=int),)
        ] = [sigma_v**2, sigma_e**2, 0, 0, sigma_w**2, sigma_vl**2]

        # Initialization
        initial_state = np.zeros((k_states,))
        initial_state_cov = np.eye(k_states) * 100

        # Initialization: cls.modelification
        initial_state_cov = np.dot(
            np.dot(cls.model.transition[:, :, 0], initial_state_cov),
            cls.model.transition[:, :, 0].T,
        )
        cls.model.initialize_known(initial_state, initial_state_cov)

    @classmethod
    def run_filter(cls):
        # Filter the data
        return cls.model.filter()

    def test_loglike(self):
        assert_almost_equal(
            # self.results.llf_obs[self.true['start']:].sum(),
            self.results.llf_obs[0:].sum(),
            self.true["loglike"],
            2,
        )

    def test_filtered_state(self):
        assert_almost_equal(
            self.results.filtered_state[0][self.true["start"] :],
            self.true_states.iloc[:, 0],
            4,
        )
        assert_almost_equal(
            self.results.filtered_state[1][self.true["start"] :],
            self.true_states.iloc[:, 1],
            4,
        )
        assert_almost_equal(
            self.results.filtered_state[4][self.true["start"] :],
            self.true_states.iloc[:, 2],
            4,
        )
        assert_almost_equal(
            self.results.filtered_state[5][self.true["start"] :],
            self.true_states.iloc[:, 3],
            4,
        )


class TestClark1989(Clark1989):
    """
    Basic double precision test for the loglikelihood and filtered
    states with two-dimensional observation vector.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class(dtype=float, conserve_memory=0)
        cls.results = cls.run_filter()

    def test_kalman_gain(self):
        assert_allclose(
            self.results.kalman_gain.sum(axis=1).sum(axis=0),
            clark1989_results["V1"],
            atol=1e-4,
        )


class TestClark1989Conserve(Clark1989):
    """
    Memory conservation test for the loglikelihood and filtered states with
    two-dimensional observation vector.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class(dtype=float, conserve_memory=0x01 | 0x02)
        cls.results = cls.run_filter()


class Clark1989Forecast(Clark1989):
    """
    Memory conservation test for the loglikelihood and filtered states with
    two-dimensional observation vector.
    """

    @classmethod
    def setup_class(cls, dtype=float, nforecast=100, conserve_memory=0):
        super().setup_class(dtype=dtype, conserve_memory=conserve_memory)
        cls.nforecast = nforecast

        # Add missing observations to the end (to forecast)
        cls.model.endog = np.array(
            np.c_[
                cls.model.endog,
                np.r_[[np.nan, np.nan] * nforecast].reshape(2, nforecast),
            ],
            ndmin=2,
            dtype=dtype,
            order="F",
        )
        cls.model.nobs = cls.model.endog.shape[1]

        cls.results = cls.run_filter()

    def test_filtered_state(self):
        assert_almost_equal(
            self.results.filtered_state[0][self.true["start"] : -self.nforecast],
            self.true_states.iloc[:, 0],
            4,
        )
        assert_almost_equal(
            self.results.filtered_state[1][self.true["start"] : -self.nforecast],
            self.true_states.iloc[:, 1],
            4,
        )
        assert_almost_equal(
            self.results.filtered_state[4][self.true["start"] : -self.nforecast],
            self.true_states.iloc[:, 2],
            4,
        )
        assert_almost_equal(
            self.results.filtered_state[5][self.true["start"] : -self.nforecast],
            self.true_states.iloc[:, 3],
            4,
        )


class TestClark1989ForecastDouble(Clark1989Forecast):
    """
    Basic double forecasting test for the loglikelihood and filtered states.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.results = cls.run_filter()


class TestClark1989ForecastDoubleComplex(Clark1989Forecast):
    """
    Basic double complex forecasting test for the loglikelihood and filtered
    states.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class(dtype=complex)
        cls.results = cls.run_filter()


class TestClark1989ForecastConserve(Clark1989Forecast):
    """
    Memory conservation forecasting test for the loglikelihood and filtered
    states.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class(dtype=float, conserve_memory=0x01 | 0x02)
        cls.results = cls.run_filter()


class TestClark1989ConserveAll(Clark1989):
    """
    Memory conservation forecasting test for the loglikelihood and filtered
    states.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class(dtype=float, conserve_memory=0x01 | 0x02 | 0x04 | 0x08)
        # cls.model.loglikelihood_burn = cls.true['start']
        cls.model.loglikelihood_burn = 0
        cls.results = cls.run_filter()

    def test_loglike(self):
        assert_almost_equal(self.results.llf, self.true["loglike"], 2)

    def test_filtered_state(self):
        end = self.true_states.shape[0]
        assert_almost_equal(
            self.results.filtered_state[0][-1], self.true_states.iloc[end - 1, 0], 4
        )
        assert_almost_equal(
            self.results.filtered_state[1][-1], self.true_states.iloc[end - 1, 1], 4
        )
        assert_almost_equal(
            self.results.filtered_state[4][-1], self.true_states.iloc[end - 1, 2], 4
        )
        assert_almost_equal(
            self.results.filtered_state[5][-1], self.true_states.iloc[end - 1, 3], 4
        )


class TestClark1989PartialMissing(Clark1989):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        endog = cls.model.endog
        endog[1, -51:] = np.nan
        cls.model.bind(endog)

        cls.results = cls.run_filter()

    def test_loglike(self):
        assert_allclose(self.results.llf_obs[0:].sum(), 1232.113456)

    def test_filtered_state(self):
        # Could do this, but no need really.
        pass

    def test_predicted_state(self):
        assert_allclose(
            self.results.predicted_state.T[1:], clark1989_results.iloc[:, 1:], atol=1e-8
        )


# Miscellaneous coverage-related tests
def test_slice_notation():
    # Test setting and getting state space representation matrices using the
    # slice notation.

    endog = np.arange(10) * 1.0
    mod = KalmanFilter(k_endog=1, k_states=2)
    mod.bind(endog)

    # Test invalid __setitem__
    def set_designs():
        mod["designs"] = 1

    def set_designs2():
        mod["designs", 0, 0] = 1

    def set_designs3():
        mod[0] = 1

    with pytest.raises(IndexError):
        set_designs()
    with pytest.raises(IndexError):
        set_designs2()
    with pytest.raises(IndexError):
        set_designs3()

    # Test invalid __getitem__
    with pytest.raises(IndexError):
        mod["designs"]
    with pytest.raises(IndexError):
        mod["designs", 0, 0, 0]
    with pytest.raises(IndexError):
        mod[0]

    # Test valid __setitem__, __getitem__
    assert_equal(mod.design[0, 0, 0], 0)
    mod["design", 0, 0, 0] = 1
    assert_equal(mod["design"].sum(), 1)
    assert_equal(mod.design[0, 0, 0], 1)
    assert_equal(mod["design", 0, 0, 0], 1)

    # Test valid __setitem__, __getitem__ with unspecified time index
    mod["design"] = np.zeros(mod["design"].shape)
    assert_equal(mod.design[0, 0], 0)
    mod["design", 0, 0] = 1
    assert_equal(mod.design[0, 0], 1)
    assert_equal(mod["design", 0, 0], 1)


def test_representation():
    # Test Representation construction

    # Test an invalid number of states
    def zero_kstates():
        Representation(1, 0)

    with pytest.raises(ValueError):
        zero_kstates()

    # Test an invalid endogenous array
    def empty_endog():
        endog = np.zeros((0, 0))
        Representation(endog, k_states=2)

    with pytest.raises(ValueError):
        empty_endog()

    # Test a Fortran-ordered endogenous array (which will be assumed to be in
    # wide format: k_endog x nobs)
    nobs = 10
    k_endog = 2
    arr = np.arange(nobs * k_endog).reshape(k_endog, nobs) * 1.0
    endog = np.asfortranarray(arr)
    mod = Representation(endog, k_states=2)
    assert_equal(mod.nobs, nobs)
    assert_equal(mod.k_endog, k_endog)

    # Test a C-ordered endogenous array (which will be assumed to be in
    # tall format: nobs x k_endog)
    nobs = 10
    k_endog = 2
    endog = np.arange(nobs * k_endog).reshape(nobs, k_endog) * 1.0
    mod = Representation(endog, k_states=2)
    assert_equal(mod.nobs, nobs)
    assert_equal(mod.k_endog, k_endog)

    # Test getting the statespace representation
    assert_equal(mod._statespace, None)
    mod._initialize_representation()
    assert_equal(mod._statespace is not None, True)


def test_bind():
    # Test binding endogenous data to Kalman filter

    mod = Representation(2, k_states=2)

    # Test invalid endogenous array (it must be ndarray)
    with pytest.raises(ValueError):
        mod.bind([1, 2, 3, 4])

    # Test valid (nobs x 1) endogenous array
    mod.bind(np.arange(10).reshape((5, 2)) * 1.0)
    assert_equal(mod.nobs, 5)

    # Test valid (k_endog x 0) endogenous array
    mod.bind(np.zeros((0, 2), dtype=np.float64))

    # Test invalid (3-dim) endogenous array
    with pytest.raises(ValueError):
        mod.bind(np.arange(12).reshape(2, 2, 3) * 1.0)

    # Test valid F-contiguous
    mod.bind(np.asfortranarray(np.arange(10).reshape(2, 5)))
    assert_equal(mod.nobs, 5)

    # Test valid C-contiguous
    mod.bind(np.arange(10).reshape(5, 2))
    assert_equal(mod.nobs, 5)

    # Test invalid F-contiguous
    with pytest.raises(ValueError):
        mod.bind(np.asfortranarray(np.arange(10).reshape(5, 2)))

    # Test invalid C-contiguous
    with pytest.raises(ValueError):
        mod.bind(np.arange(10).reshape(2, 5))


def test_initialization():
    # Test Kalman filter initialization

    mod = Representation(1, k_states=2)

    # Test invalid state initialization
    with pytest.raises(RuntimeError):
        mod._initialize_state()

    # Test valid initialization
    initial_state = (
        np.zeros(
            2,
        )
        + 1.5
    )
    initial_state_cov = np.eye(2) * 3.0
    mod.initialize_known(initial_state, initial_state_cov)
    assert_equal(mod.initialization.constant.sum(), 3)
    assert_equal(mod.initialization.stationary_cov.diagonal().sum(), 6)

    # Test invalid initial_state
    initial_state = np.zeros(
        10,
    )
    with pytest.raises(ValueError):
        mod.initialize_known(initial_state, initial_state_cov)
    initial_state = np.zeros((10, 10))
    with pytest.raises(ValueError):
        mod.initialize_known(initial_state, initial_state_cov)

    # Test invalid initial_state_cov
    initial_state = (
        np.zeros(
            2,
        )
        + 1.5
    )
    initial_state_cov = np.eye(3)
    with pytest.raises(ValueError):
        mod.initialize_known(initial_state, initial_state_cov)


@pytest.mark.parametrize("kind", ["ndarray", "list", "pandas"])
def test_initialize_known_array_like(kind):
    # The first predicted state and its covariance are the known moments
    constant = np.array([1.5, -0.5])
    cov = np.array([[2.0, 0.3], [0.3, 1.0]])
    mod = sarimax.SARIMAX(np.zeros(5), order=(2, 0, 0))
    if kind == "ndarray":
        mod.initialize_known(constant, cov)
    elif kind == "list":
        mod.initialize_known(constant.tolist(), cov.tolist())
    else:
        mod.initialize_known(pd.Series(constant), pd.DataFrame(cov))
    res = mod.filter([0.5, -0.2, 1.0])
    assert_allclose(res.predicted_state[:, 0], constant, rtol=1e-15)
    assert_allclose(res.predicted_state_cov[:, :, 0], cov, rtol=1e-15)


def test_initialize_known_keeps_complex():
    mod = Representation(1, k_states=2)
    mod.initialize_known([1.0 + 2.0j, 3.0], np.eye(2))
    assert_allclose(mod.initialization.constant, [1.0 + 2.0j, 3.0], rtol=1e-15)


@pytest.mark.parametrize(
    "k_states, constant, cov, match",
    [
        (1, 1.0, [[1.0]], "constant must have 1 dimension, but has 0"),
        (1, [[1.0]], [[1.0]], "constant must have 1 dimension, but has 2"),
        (2, np.ones((2, 1)), np.eye(2), "constant must have 1 dimension, but has 2"),
        (1, [1.0], 1.0, "stationary_cov must have 2 dimensions, but has 0"),
        (2, np.ones(2), np.ones(2), "stationary_cov must have 2 dimensions, but has 1"),
        (
            2,
            np.ones(2),
            np.eye(2)[:, :, None],
            "stationary_cov must have 2 dimensions, but has 3",
        ),
        (2, np.ones(3), np.eye(2), r"Requires shape \(2,\), got \(3,\)"),
        (2, np.ones(2), np.eye(3), r"Requires shape \(2,2\), got \(3, 3\)"),
    ],
    ids=[
        "scalar-constant",
        "1x1-constant",
        "column-constant",
        "scalar-cov",
        "1d-cov",
        "3d-cov",
        "constant-length",
        "cov-shape",
    ],
)
def test_initialize_known_invalid_shape_raises(k_states, constant, cov, match):
    mod = Representation(1, k_states=k_states)
    with pytest.raises(ValueError, match=match):
        mod.initialize_known(constant, cov)


def test_init_matrices_time_invariant():
    # Test setting state space system matrices in __init__, with time-invariant
    # matrices
    k_endog = 2
    k_states = 3
    k_posdef = 1

    endog = np.zeros((10, 2))
    obs_intercept = np.arange(k_endog) * 1.0
    design = np.reshape(np.arange(k_endog * k_states) * 1.0, (k_endog, k_states))
    obs_cov = np.reshape(np.arange(k_endog**2) * 1.0, (k_endog, k_endog))
    state_intercept = np.arange(k_states) * 1.0
    transition = np.reshape(np.arange(k_states**2) * 1.0, (k_states, k_states))
    selection = np.reshape(np.arange(k_states * k_posdef) * 1.0, (k_states, k_posdef))
    state_cov = np.reshape(np.arange(k_posdef**2) * 1.0, (k_posdef, k_posdef))

    mod = Representation(
        endog,
        k_states=k_states,
        k_posdef=k_posdef,
        obs_intercept=obs_intercept,
        design=design,
        obs_cov=obs_cov,
        state_intercept=state_intercept,
        transition=transition,
        selection=selection,
        state_cov=state_cov,
    )

    assert_allclose(mod["obs_intercept"], obs_intercept)
    assert_allclose(mod["design"], design)
    assert_allclose(mod["obs_cov"], obs_cov)
    assert_allclose(mod["state_intercept"], state_intercept)
    assert_allclose(mod["transition"], transition)
    assert_allclose(mod["selection"], selection)
    assert_allclose(mod["state_cov"], state_cov)


def test_init_matrices_time_varying():
    # Test setting state space system matrices in __init__, with time-varying
    # matrices
    nobs = 10
    k_endog = 2
    k_states = 3
    k_posdef = 1

    endog = np.zeros((10, 2))
    obs_intercept = np.reshape(np.arange(k_endog * nobs) * 1.0, (k_endog, nobs))
    design = np.reshape(
        np.arange(k_endog * k_states * nobs) * 1.0, (k_endog, k_states, nobs)
    )
    obs_cov = np.reshape(np.arange(k_endog**2 * nobs) * 1.0, (k_endog, k_endog, nobs))
    state_intercept = np.reshape(np.arange(k_states * nobs) * 1.0, (k_states, nobs))
    transition = np.reshape(
        np.arange(k_states**2 * nobs) * 1.0, (k_states, k_states, nobs)
    )
    selection = np.reshape(
        np.arange(k_states * k_posdef * nobs) * 1.0, (k_states, k_posdef, nobs)
    )
    state_cov = np.reshape(
        np.arange(k_posdef**2 * nobs) * 1.0, (k_posdef, k_posdef, nobs)
    )

    mod = Representation(
        endog,
        k_states=k_states,
        k_posdef=k_posdef,
        obs_intercept=obs_intercept,
        design=design,
        obs_cov=obs_cov,
        state_intercept=state_intercept,
        transition=transition,
        selection=selection,
        state_cov=state_cov,
    )

    assert_allclose(mod["obs_intercept"], obs_intercept)
    assert_allclose(mod["design"], design)
    assert_allclose(mod["obs_cov"], obs_cov)
    assert_allclose(mod["state_intercept"], state_intercept)
    assert_allclose(mod["transition"], transition)
    assert_allclose(mod["selection"], selection)
    assert_allclose(mod["state_cov"], state_cov)


def test_no_endog():
    # Test for RuntimeError when no endog is provided by the time filtering
    # is initialized.

    mod = KalmanFilter(k_endog=1, k_states=1)

    # directly call the _initialize_filter function
    with pytest.raises(RuntimeError):
        mod._initialize_filter()
    # indirectly call it through filtering
    mod.initialize_approximate_diffuse()
    with pytest.raises(RuntimeError):
        mod.filter()


def test_cython():
    # Test the cython _kalman_filter creation, re-creation, calling, etc.

    # Check that datatypes are correct:
    for prefix, dtype in tools.prefix_dtype_map.items():
        endog = np.array(1.0, ndmin=2, dtype=dtype)
        mod = KalmanFilter(k_endog=1, k_states=1, dtype=dtype)

        # Bind data and initialize the ?KalmanFilter object
        mod.bind(endog)
        mod._initialize_filter()

        # Check that the dtype and prefix are correct
        assert_equal(mod.prefix, prefix)
        assert_equal(mod.dtype, dtype)

        # Test that a dKalmanFilter instance was created
        assert_equal(prefix in mod._kalman_filters, True)
        kf = mod._kalman_filters[prefix]
        assert isinstance(kf, tools.prefix_kalman_filter_map[prefix])

        # Test that the default returned _kalman_filter is the above instance
        assert_equal(mod._kalman_filter, kf)

    # Check that upcasting datatypes / ?KalmanFilter works (e.g., d -> z)
    mod = KalmanFilter(k_endog=1, k_states=1)

    # Default dtype is float
    assert_equal(mod.prefix, "d")
    assert_equal(mod.dtype, np.float64)

    # Prior to initialization, no ?KalmanFilter exists
    assert_equal(mod._kalman_filter, None)

    # Bind data and initialize the ?KalmanFilter object
    endog = np.ascontiguousarray(np.array([1.0, 2.0], dtype=np.float64))
    mod.bind(endog)
    mod._initialize_filter()
    kf = mod._kalman_filters["d"]

    # Rebind data, still float, check that we have not changed
    mod.bind(endog)
    mod._initialize_filter()
    assert_equal(mod._kalman_filter, kf)

    # Force creating new ?Statespace and ?KalmanFilter, by changing the
    # time-varying character of an array
    mod.design = np.zeros((1, 1, 2))
    mod._initialize_filter()
    assert_equal(mod._kalman_filter == kf, False)
    kf = mod._kalman_filters["d"]

    # Rebind data, now complex, check that the ?KalmanFilter instance has
    # changed
    endog = np.ascontiguousarray(np.array([1.0, 2.0], dtype=np.complex128))
    mod.bind(endog)
    assert_equal(mod._kalman_filter == kf, False)


def test_filter():
    # Tests of invalid calls to the filter function

    endog = np.ones((10, 1))
    mod = KalmanFilter(endog, k_states=1, initialization="approximate_diffuse")
    mod["design", :] = 1
    mod["selection", :] = 1
    mod["state_cov", :] = 1

    # Test default filter results
    res = mod.filter()
    assert_equal(isinstance(res, FilterResults), True)


def test_loglike():
    # Tests of invalid calls to the loglike function

    endog = np.ones((10, 1))
    mod = KalmanFilter(endog, k_states=1, initialization="approximate_diffuse")
    mod["design", :] = 1
    mod["selection", :] = 1
    mod["state_cov", :] = 1

    # Test that self.memory_no_likelihood = True raises an error
    mod.memory_no_likelihood = True
    with pytest.raises(RuntimeError):
        mod.loglikeobs()


def test_predict():
    # Tests of invalid calls to the predict function

    warnings.simplefilter("always")

    endog = np.ones((10, 1))
    mod = KalmanFilter(endog, k_states=1, initialization="approximate_diffuse")
    mod["design", :] = 1
    mod["obs_intercept"] = np.zeros((1, 10))
    mod["selection", :] = 1
    mod["state_cov", :] = 1

    # Check that we need both forecasts and predicted output for dynamic
    # prediction
    mod.memory_no_forecast = True
    res = mod.filter()
    with pytest.raises(ValueError):
        res.predict()
    mod.memory_no_forecast = False

    mod.memory_no_predicted = True
    res = mod.filter()
    with pytest.raises(ValueError):
        res.predict(dynamic=True)
    mod.memory_no_predicted = False

    # Now get a clean filter object
    res = mod.filter()

    # Check that start < 0 is an error
    with pytest.raises(ValueError):
        res.predict(start=-1)

    # Check that end < start is an error
    with pytest.raises(ValueError):
        res.predict(start=2, end=1)

    # Check that dynamic < 0 is an error
    with pytest.raises(ValueError):
        res.predict(dynamic=-1)

    # Check that dynamic > end is an warning
    with warnings.catch_warnings(record=True) as w:
        res.predict(end=1, dynamic=2)
        message = (
            "Dynamic prediction specified to begin after the end of"
            " prediction, and so has no effect."
        )
        assert_equal(str(w[0].message), message)

    # Check that dynamic > nobs is an warning
    with warnings.catch_warnings(record=True) as w:
        res.predict(end=11, dynamic=11, obs_intercept=np.zeros((1, 1)))
        message = (
            "Dynamic prediction specified to begin during"
            " out-of-sample forecasting period, and so has no"
            " effect."
        )
        assert_equal(str(w[0].message), message)

    # Check for a warning when providing a non-used statespace matrix
    with pytest.raises(ValueError):
        res.predict(end=res.nobs + 1, design=True, obs_intercept=np.zeros((1, 1)))

    # Check that an error is raised when a new time-varying matrix is not
    # provided
    with pytest.raises(ValueError):
        res.predict(end=res.nobs + 1)

    # Check that an error is raised when an obs_intercept with incorrect length
    # is given
    with pytest.raises(ValueError):
        res.predict(end=res.nobs + 1, obs_intercept=np.zeros(2))

    # Check that start=None gives start=0 and end=None gives end=nobs
    assert_equal(res.predict().forecasts.shape, (1, res.nobs))

    # Check that dynamic=True begins dynamic prediction immediately
    # TODO just a smoke test
    res.predict(dynamic=True)

    # Check that on success, PredictionResults object is returned
    prediction_results = res.predict(start=3, end=5)
    assert_equal(isinstance(prediction_results, PredictionResults), True)

    # Check for correctly subset representation arrays
    # (k_endog, npredictions) = (1, 2)
    assert_equal(prediction_results.endog.shape, (1, 2))
    # (k_endog, npredictions) = (1, 2)
    assert_equal(prediction_results.obs_intercept.shape, (1, 2))
    # (k_endog, k_states) = (1, 1)
    assert_equal(prediction_results.design.shape, (1, 1))
    # (k_endog, k_endog) = (1, 1)
    assert_equal(prediction_results.obs_cov.shape, (1, 1))
    # (k_state,) = (1,)
    assert_equal(prediction_results.state_intercept.shape, (1,))
    # (k_state, npredictions) = (1, 2)
    assert_equal(prediction_results.obs_intercept.shape, (1, 2))
    # (k_state, k_state) = (1, 1)
    assert_equal(prediction_results.transition.shape, (1, 1))
    # (k_state, k_posdef) = (1, 1)
    assert_equal(prediction_results.selection.shape, (1, 1))
    # (k_posdef, k_posdef) = (1, 1)
    assert_equal(prediction_results.state_cov.shape, (1, 1))

    # Check for correctly subset filter output arrays
    # (k_endog, npredictions) = (1, 2)
    assert_equal(prediction_results.forecasts.shape, (1, 2))
    assert_equal(prediction_results.forecasts_error.shape, (1, 2))
    # (k_states, npredictions) = (1, 2)
    assert_equal(prediction_results.filtered_state.shape, (1, 2))
    assert_equal(prediction_results.predicted_state.shape, (1, 2))
    # (k_endog, k_endog, npredictions) = (1, 1, 2)
    assert_equal(prediction_results.forecasts_error_cov.shape, (1, 1, 2))
    # (k_states, k_states, npredictions) = (1, 1, 2)
    assert_equal(prediction_results.filtered_state_cov.shape, (1, 1, 2))
    assert_equal(prediction_results.predicted_state_cov.shape, (1, 1, 2))

    # Check for invalid attribute
    with pytest.raises(AttributeError):
        _ = prediction_results.test

    # Check that an error is raised when a non-two-dimensional obs_cov
    # is given
    # ...and...
    # Check that an error is raised when an obs_cov that is too short is given
    mod = KalmanFilter(endog, k_states=1, initialization="approximate_diffuse")
    mod["design", :] = 1
    mod["obs_cov"] = np.zeros((1, 1, 10))
    mod["selection", :] = 1
    mod["state_cov", :] = 1
    res = mod.filter()

    with pytest.raises(ValueError):
        res.predict(end=res.nobs + 2, obs_cov=np.zeros((1, 1)))
    with pytest.raises(ValueError):
        res.predict(end=res.nobs + 2, obs_cov=np.zeros((1, 1, 1)))


def test_standardized_forecasts_error():
    # Simple test that standardized forecasts errors are calculated correctly.

    # Just uses a different calculation method on a univariate series.

    # Get the dataset
    true = results_kalman_filter.uc_uni
    data = pd.DataFrame(
        true["data"],
        index=pd.date_range("1947-01-01", "1995-07-01", freq="QS"),
        columns=["GDP"],
    )
    data["lgdp"] = np.log(data["GDP"])

    # Fit an ARIMA(1, 1, 0) to log GDP
    mod = sarimax.SARIMAX(data["lgdp"], order=(1, 1, 0), use_exact_diffuse=True)
    res = mod.fit(disp=-1)
    d = np.maximum(res.loglikelihood_burn, res.nobs_diffuse)

    standardized_forecasts_error = res.filter_results.forecasts_error[0] / np.sqrt(
        res.filter_results.forecasts_error_cov[0, 0]
    )

    assert_allclose(
        res.filter_results.standardized_forecasts_error[0, d:],
        standardized_forecasts_error[..., d:],
    )


def test_simulate():
    # Test for simulation of new time-series
    from scipy.signal import lfilter

    # Common parameters
    nsimulations = 10
    sigma2 = 2
    measurement_shocks = np.zeros(nsimulations)
    rs = np.random.RandomState(9991617)
    state_shocks = rs.normal(scale=sigma2**0.5, size=nsimulations)

    # Random walk model, so simulated series is just the cumulative sum of
    # the shocks
    mod = SimulationSmoother(np.r_[0], k_states=1, initialization="diffuse")
    mod["design", 0, 0] = 1.0
    mod["transition", 0, 0] = 1.0
    mod["selection", 0, 0] = 1.0

    actual = mod.simulate(
        nsimulations, measurement_shocks=measurement_shocks, state_shocks=state_shocks
    )[0].squeeze()
    desired = np.r_[0, np.cumsum(state_shocks)[:-1]]

    assert_allclose(actual, desired)

    # Local level model, so simulated series is just the cumulative sum of
    # the shocks plus the measurement shock
    mod = SimulationSmoother(np.r_[0], k_states=1, initialization="diffuse")
    mod["design", 0, 0] = 1.0
    mod["transition", 0, 0] = 1.0
    mod["selection", 0, 0] = 1.0

    actual = mod.simulate(
        nsimulations,
        measurement_shocks=np.ones(nsimulations),
        state_shocks=state_shocks,
    )[0].squeeze()
    desired = np.r_[1, np.cumsum(state_shocks)[:-1] + 1]

    assert_allclose(actual, desired)

    # Local level-like model with observation and state intercepts, so
    # simulated series is just the cumulative sum of the shocks minus the state
    # intercept, plus the observation intercept and the measurement shock
    mod = SimulationSmoother(np.zeros((1, 10)), k_states=1, initialization="diffuse")
    mod["obs_intercept", 0, 0] = 5.0
    mod["design", 0, 0] = 1.0
    mod["state_intercept", 0, 0] = -2.0
    mod["transition", 0, 0] = 1.0
    mod["selection", 0, 0] = 1.0

    actual = mod.simulate(
        nsimulations,
        measurement_shocks=np.ones(nsimulations),
        state_shocks=state_shocks,
    )[0].squeeze()
    desired = np.r_[1 + 5, np.cumsum(state_shocks - 2)[:-1] + 1 + 5]

    assert_allclose(actual, desired)

    # Model with time-varying observation intercept
    mod = SimulationSmoother(
        np.zeros((1, 10)), k_states=1, nobs=10, initialization="diffuse"
    )
    mod["obs_intercept"] = (np.arange(10) * 1.0).reshape(1, 10)
    mod["design", 0, 0] = 1.0
    mod["transition", 0, 0] = 1.0
    mod["selection", 0, 0] = 1.0

    actual = mod.simulate(
        nsimulations, measurement_shocks=measurement_shocks, state_shocks=state_shocks
    )[0].squeeze()
    desired = np.r_[0, np.cumsum(state_shocks)[:-1] + np.arange(1, 10)]

    assert_allclose(actual, desired)

    # Model with time-varying observation intercept, check that error is raised
    # if more simulations are requested than are nobs.
    mod = SimulationSmoother(
        np.zeros((1, 10)), k_states=1, nobs=10, initialization="diffuse"
    )
    mod["obs_intercept"] = (np.arange(10) * 1.0).reshape(1, 10)
    mod["design", 0, 0] = 1.0
    mod["transition", 0, 0] = 1.0
    mod["selection", 0, 0] = 1.0
    with pytest.raises(ValueError):
        mod.simulate(nsimulations + 1, measurement_shocks, state_shocks)

    # ARMA(1, 1): phi = [0.1], theta = [0.5], sigma^2 = 2
    phi = 0.1
    theta = 0.5
    mod = sarimax.SARIMAX([0], order=(1, 0, 1))
    mod.update(np.r_[phi, theta, sigma2])

    actual = mod.ssm.simulate(
        nsimulations,
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=np.zeros(mod.k_states),
    )[0].squeeze()
    desired = lfilter([1, theta], [1, -phi], np.r_[0, state_shocks[:-1]])

    assert_allclose(actual, desired)

    # SARIMAX(1, 0, 1)x(1, 0, 1, 4), this time using the results object call
    mod = sarimax.SARIMAX(
        [0.1, 0.5, -0.2], order=(1, 0, 1), seasonal_order=(1, 0, 1, 4)
    )
    res = mod.filter([0.1, 0.5, 0.2, -0.3, 1])

    actual = res.simulate(
        nsimulations,
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=np.zeros(mod.k_states),
    )
    desired = lfilter(
        res.polynomial_reduced_ma,
        res.polynomial_reduced_ar,
        np.r_[0, state_shocks[:-1]],
    )

    assert_allclose(actual, desired)


def test_impulse_responses():
    # Test for impulse response functions

    # Random walk: 1-unit impulse response (i.e., non-orthogonalized irf) is 1
    # for all periods
    mod = SimulationSmoother(k_endog=1, k_states=1, initialization="diffuse")
    mod["design", 0, 0] = 1.0
    mod["transition", 0, 0] = 1.0
    mod["selection", 0, 0] = 1.0
    mod["state_cov", 0, 0] = 2.0

    actual = mod.impulse_responses(steps=10)
    desired = np.ones((11, 1))

    assert_allclose(actual, desired)

    # Random walk: 2-unit impulse response (i.e., non-orthogonalized irf) is 2
    # for all periods
    mod = SimulationSmoother(k_endog=1, k_states=1, initialization="diffuse")
    mod["design", 0, 0] = 1.0
    mod["transition", 0, 0] = 1.0
    mod["selection", 0, 0] = 1.0
    mod["state_cov", 0, 0] = 2.0

    actual = mod.impulse_responses(steps=10, impulse=[2])
    desired = np.ones((11, 1)) * 2

    assert_allclose(actual, desired)

    # Random walk: 1-standard-deviation response (i.e., orthogonalized irf) is
    # sigma for all periods (here sigma^2 = 2)
    mod = SimulationSmoother(k_endog=1, k_states=1, initialization="diffuse")
    mod["design", 0, 0] = 1.0
    mod["transition", 0, 0] = 1.0
    mod["selection", 0, 0] = 1.0
    mod["state_cov", 0, 0] = 2.0

    actual = mod.impulse_responses(steps=10, orthogonalized=True)
    desired = np.ones((11, 1)) * 2**0.5

    assert_allclose(actual, desired)

    # Random walk: 1-standard-deviation cumulative response (i.e., cumulative
    # orthogonalized irf)
    mod = SimulationSmoother(k_endog=1, k_states=1, initialization="diffuse")
    mod["design", 0, 0] = 1.0
    mod["transition", 0, 0] = 1.0
    mod["selection", 0, 0] = 1.0
    mod["state_cov", 0, 0] = 2.0

    actual = mod.impulse_responses(steps=10, orthogonalized=True, cumulative=True)
    desired = np.cumsum(np.ones((11, 1)) * 2**0.5)[:, np.newaxis]

    actual = mod.impulse_responses(
        steps=10, impulse=[1], orthogonalized=True, cumulative=True
    )
    desired = np.cumsum(np.ones((11, 1)) * 2**0.5)[:, np.newaxis]

    assert_allclose(actual, desired)

    # Random walk: 1-unit impulse response (i.e., non-orthogonalized irf) is 1
    # for all periods, even when intercepts are present
    mod = SimulationSmoother(k_endog=1, k_states=1, initialization="diffuse")
    mod["state_intercept", 0] = 100.0
    mod["design", 0, 0] = 1.0
    mod["obs_intercept", 0] = -1000.0
    mod["transition", 0, 0] = 1.0
    mod["selection", 0, 0] = 1.0
    mod["state_cov", 0, 0] = 2.0

    actual = mod.impulse_responses(steps=10)
    desired = np.ones((11, 1))

    assert_allclose(actual, desired)

    # Univariate model (random walk): test that an error is thrown when
    # a multivariate or empty "impulse" is sent
    mod = SimulationSmoother(k_endog=1, k_states=1, initialization="diffuse")
    with pytest.raises(ValueError):
        mod.impulse_responses(impulse=1)
    with pytest.raises(ValueError):
        mod.impulse_responses(impulse=[1, 1])
    with pytest.raises(ValueError):
        mod.impulse_responses(impulse=[])

    # Univariate model with two uncorrelated shocks
    mod = SimulationSmoother(k_endog=1, k_states=2, initialization="diffuse")
    mod["design", 0, 0:2] = 1.0
    mod["transition", :, :] = np.eye(2)
    mod["selection", :, :] = np.eye(2)
    mod["state_cov", :, :] = np.eye(2)

    desired = np.ones((11, 1))

    actual = mod.impulse_responses(steps=10, impulse=0)
    assert_allclose(actual, desired)

    actual = mod.impulse_responses(steps=10, impulse=[1, 0])
    assert_allclose(actual, desired)

    actual = mod.impulse_responses(steps=10, impulse=1)
    assert_allclose(actual, desired)

    actual = mod.impulse_responses(steps=10, impulse=[0, 1])
    assert_allclose(actual, desired)

    # In this case (with sigma=sigma^2=1), orthogonalized is the same as not
    actual = mod.impulse_responses(steps=10, impulse=0, orthogonalized=True)
    assert_allclose(actual, desired)

    actual = mod.impulse_responses(steps=10, impulse=[1, 0], orthogonalized=True)
    assert_allclose(actual, desired)

    actual = mod.impulse_responses(steps=10, impulse=[0, 1], orthogonalized=True)
    assert_allclose(actual, desired)

    # Univariate model with two correlated shocks
    mod = SimulationSmoother(k_endog=1, k_states=2, initialization="diffuse")
    mod["design", 0, 0:2] = 1.0
    mod["transition", :, :] = np.eye(2)
    mod["selection", :, :] = np.eye(2)
    mod["state_cov", :, :] = np.array([[1, 0.5], [0.5, 1.25]])

    desired = np.ones((11, 1))

    # Non-orthogonalized (i.e., 1-unit) impulses still just generate 1's
    actual = mod.impulse_responses(steps=10, impulse=0)
    assert_allclose(actual, desired)

    actual = mod.impulse_responses(steps=10, impulse=1)
    assert_allclose(actual, desired)

    # Orthogonalized (i.e., 1-std-dev) impulses now generate different responses
    actual = mod.impulse_responses(steps=10, impulse=0, orthogonalized=True)
    assert_allclose(actual, desired + desired * 0.5)

    actual = mod.impulse_responses(steps=10, impulse=1, orthogonalized=True)
    assert_allclose(actual, desired)

    # Multivariate model with two correlated shocks
    mod = SimulationSmoother(k_endog=2, k_states=2, initialization="diffuse")
    mod["design", :, :] = np.eye(2)
    mod["transition", :, :] = np.eye(2)
    mod["selection", :, :] = np.eye(2)
    mod["state_cov", :, :] = np.array([[1, 0.5], [0.5, 1.25]])

    ones = np.ones((11, 1))
    zeros = np.zeros((11, 1))

    # Non-orthogonalized (i.e., 1-unit) impulses still just generate 1's, but
    # only for the appropriate series
    actual = mod.impulse_responses(steps=10, impulse=0)
    assert_allclose(actual, np.c_[ones, zeros])

    actual = mod.impulse_responses(steps=10, impulse=1)
    assert_allclose(actual, np.c_[zeros, ones])

    # Orthogonalized (i.e., 1-std-dev) impulses now generate different
    # responses, and only for the appropriate series
    actual = mod.impulse_responses(steps=10, impulse=0, orthogonalized=True)
    assert_allclose(actual, np.c_[ones, ones * 0.5])

    actual = mod.impulse_responses(steps=10, impulse=1, orthogonalized=True)
    assert_allclose(actual, np.c_[zeros, ones])

    # AR(1) model generates a geometrically declining series
    mod = sarimax.SARIMAX([0.1, 0.5, -0.2], order=(1, 0, 0))
    phi = 0.5
    mod.update([phi, 1])

    desired = np.cumprod(np.r_[1, [phi] * 10])

    # Test going through the model directly
    actual = mod.ssm.impulse_responses(steps=10)
    assert_allclose(actual[:, 0], desired)

    # Test going through the results object
    res = mod.filter([phi, 1.0])
    actual = res.impulse_responses(steps=10)
    assert_allclose(actual, desired)


def test_missing():
    # Datasets
    endog = np.arange(10).reshape(10, 1)
    endog_pre_na = np.ascontiguousarray(
        np.c_[endog.copy() * np.nan, endog.copy() * np.nan, endog, endog]
    )
    endog_post_na = np.ascontiguousarray(
        np.c_[endog, endog, endog.copy() * np.nan, endog.copy() * np.nan]
    )
    endog_inject_na = np.ascontiguousarray(
        np.c_[endog, endog.copy() * np.nan, endog, endog.copy() * np.nan]
    )

    # Base model
    mod = KalmanFilter(
        np.ascontiguousarray(np.c_[endog, endog]),
        k_states=1,
        initialization="approximate_diffuse",
    )
    mod["design", :, :] = 1
    mod["obs_cov", :, :] = np.eye(mod.k_endog) * 0.5
    mod["transition", :, :] = 0.5
    mod["selection", :, :] = 1
    mod["state_cov", :, :] = 0.5
    llf = mod.loglikeobs()

    # Model with prepended nans
    mod = KalmanFilter(endog_pre_na, k_states=1, initialization="approximate_diffuse")
    mod["design", :, :] = 1
    mod["obs_cov", :, :] = np.eye(mod.k_endog) * 0.5
    mod["transition", :, :] = 0.5
    mod["selection", :, :] = 1
    mod["state_cov", :, :] = 0.5
    llf_pre_na = mod.loglikeobs()

    assert_allclose(llf_pre_na, llf)

    # Model with appended nans
    mod = KalmanFilter(endog_post_na, k_states=1, initialization="approximate_diffuse")
    mod["design", :, :] = 1
    mod["obs_cov", :, :] = np.eye(mod.k_endog) * 0.5
    mod["transition", :, :] = 0.5
    mod["selection", :, :] = 1
    mod["state_cov", :, :] = 0.5
    llf_post_na = mod.loglikeobs()

    assert_allclose(llf_post_na, llf)

    # Model with injected nans
    mod = KalmanFilter(
        endog_inject_na, k_states=1, initialization="approximate_diffuse"
    )
    mod["design", :, :] = 1
    mod["obs_cov", :, :] = np.eye(mod.k_endog) * 0.5
    mod["transition", :, :] = 0.5
    mod["selection", :, :] = 1
    mod["state_cov", :, :] = 0.5
    llf_inject_na = mod.loglikeobs()

    assert_allclose(llf_inject_na, llf)


def test_initialize_components():
    # `initialize_components` builds an Initialization from the component
    # matrices (a, Pstar, Pinf/A) as documented in `Initialization.
    # from_components` (Durbin and Koopman (2012), Sec. 5.2): the resulting
    # initialization, evaluated via `__call__`, must reproduce `a` as the
    # initial state mean, `Pinf = A @ A.T` as the diffuse covariance, and
    # `Pstar` as the stationary covariance.
    mod = Representation(1, k_states=3)

    a = np.array([1.0, 2.0, 0.0])
    Pstar = np.diag([4.0, 9.0, 0.0])
    A = np.zeros((3, 1))
    A[2, 0] = 1.0

    mod.initialize_components(a=a, Pstar=Pstar, A=A)

    assert mod.initialization.initialized
    mean, diffuse_cov, stationary_cov = mod.initialization()
    assert_allclose(mean, a)
    assert_allclose(diffuse_cov, A.dot(A.T))
    assert_allclose(stationary_cov, Pstar)


def test_initialize_components_r0_q0_equivalent_to_pstar():
    # Per the documented contract, `Pstar = R0 @ Q0 @ R0.T`, so specifying
    # R0/Q0 must give identical results to specifying the equivalent Pstar
    # directly.
    Pstar = np.array([[2.0, 0.5], [0.5, 3.0]])

    mod1 = Representation(1, k_states=2)
    mod1.initialize_components(a=[0.0, 0.0], Pstar=Pstar)

    mod2 = Representation(1, k_states=2)
    R0 = np.eye(2)
    Q0 = Pstar
    mod2.initialize_components(a=[0.0, 0.0], R0=R0, Q0=Q0)

    mean1, diffuse_cov1, stationary_cov1 = mod1.initialization()
    mean2, diffuse_cov2, stationary_cov2 = mod2.initialization()

    assert_allclose(stationary_cov1, Pstar)
    assert_allclose(stationary_cov1, stationary_cov2)
    assert_allclose(mean1, mean2)
    assert_allclose(diffuse_cov1, diffuse_cov2)


def test_initialize_components_default_a_is_zero():
    # `a` defaults to a zero vector when not specified
    mod = Representation(1, k_states=2)
    mod.initialize_components(Pstar=np.eye(2))

    mean, _, stationary_cov = mod.initialization()
    assert_allclose(mean, np.zeros(2))
    assert_allclose(stationary_cov, np.eye(2))


def test_initialize_components_pstar_and_r0q0_raises():
    # Cannot specify both Pstar and R0/Q0 (they are documented as mutually
    # exclusive parameterizations of the same stationary covariance)
    mod = Representation(1, k_states=2)
    with pytest.raises(ValueError):
        mod.initialize_components(
            a=[0.0, 0.0], Pstar=np.eye(2), R0=np.eye(2), Q0=np.eye(2)
        )


def _bivariate_endog(nobs=40):
    rng = np.random.default_rng(97531)
    level = rng.standard_normal((nobs, 2)).cumsum(0) * 0.1
    return level + rng.standard_normal((nobs, 2))


def _bivariate_model(cls=KalmanFilter, endog=None, nobs=40):
    # Two observed series and two states, with correlated observation and
    # state disturbances, initialized with a known distribution
    if endog is None:
        endog = _bivariate_endog(nobs)
    mod = cls(np.ascontiguousarray(endog), k_states=2)
    mod["design"] = np.array([[1.0, 0.2], [0.3, 1.0]])
    mod["transition"] = np.array([[0.5, 0.1], [0.0, 0.3]])
    mod["selection"] = np.eye(2)
    mod["state_cov"] = np.array([[1.0, 0.2], [0.2, 0.5]])
    mod["obs_cov"] = np.array([[1.0, 0.3], [0.3, 2.0]])
    mod.initialize_known(np.zeros(2), np.eye(2))
    return mod


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({}, "Initial state must be provided"),
        ({"constant": [0.0]}, "Initial state covariance matrix must be provided"),
        ({"initial_state": [0.0]}, "Initial state covariance matrix must be"),
    ],
)
def test_known_initialization_requires_mean_and_cov(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Representation(1, k_states=1, initialization="known", **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"constant": [1.5], "stationary_cov": [[2.0]]},
        {"initial_state": [1.5], "initial_state_cov": [[2.0]]},
    ],
)
def test_known_initialization_keywords(kwargs):
    # Both spellings give the initial state distribution N(1.5, 2)
    mod = Representation(1, k_states=1, initialization="known", **kwargs)
    mean, _, stationary_cov = mod.initialization()
    assert_allclose(mean, [1.5])
    assert_allclose(stationary_cov, [[2.0]])


def test_invalid_initialization_method():
    match = "Invalid state space initialization method"
    with pytest.raises(ValueError, match=match):
        Representation(1, k_states=1, initialization="unknown")
    mod = Representation(1, k_states=1)
    with pytest.raises(ValueError, match=match):
        mod.initialize("unknown")
    assert mod.initialization is None


def test_clone_dimensions_and_time_varying_matrices():
    mod = _bivariate_model()
    endog = np.arange(10.0).reshape(5, 2)
    # Passing the current dimensions is allowed
    clone = mod.clone(endog, k_states=2, k_posdef=2)
    assert_allclose(clone.endog, endog.T)
    for name in ["design", "obs_cov", "transition", "selection", "state_cov"]:
        assert_allclose(clone[name], mod[name])

    with pytest.raises(ValueError, match="Cannot change the dimension of k_states"):
        mod.clone(endog, k_states=3)
    with pytest.raises(ValueError, match="Cannot change the dimension of k_posdef"):
        mod.clone(endog, k_posdef=1)

    mod["obs_intercept"] = np.zeros((2, mod.nobs))
    with pytest.raises(ValueError, match="The `obs_intercept` matrix is time-varying"):
        mod.clone(endog)


def test_extend_ranges():
    mod = _bivariate_model(nobs=10)
    new = np.arange(6.0).reshape(3, 2) + 100
    # By default the whole sample is kept and the new data are appended
    ext = mod.extend(new)
    assert_allclose(ext.endog, np.r_[mod.endog.T, new].T)
    assert_allclose(ext["transition"], mod["transition"])
    # Negative start and end count from the end of the sample
    ext = mod.extend(new, start=-3, end=-1)
    assert_allclose(ext.endog, np.r_[mod.endog.T[7:9], new].T)

    # A 1-dim array is a univariate series
    mod = KalmanFilter(np.arange(10.0).reshape(10, 1), k_states=1)
    ext = mod.extend(np.array([20.0, 21.0]))
    assert_allclose(ext.endog, [np.r_[np.arange(10.0), 20.0, 21.0]])


@pytest.mark.parametrize(
    "nobs_new, kwargs, match",
    [
        (3, {"start": 11}, "The `start` argument .* after the end of the base"),
        (3, {"end": 11}, "The `end` argument .* after the end of the base"),
        (3, {"start": 5, "end": 3}, "cannot be after the `end` argument"),
        (
            0,
            {"obs_intercept": np.zeros((2, 3))},
            "Extension is being performed within-sample",
        ),
    ],
)
def test_extend_invalid(nobs_new, kwargs, match):
    mod = _bivariate_model(nobs=10)
    with pytest.raises(ValueError, match=match):
        mod.extend(np.ones((nobs_new, 2)), **kwargs)


def test_diff_endog_too_short():
    mod = _bivariate_model(nobs=10)
    with pytest.raises(ValueError, match=r"Given data \(length 5\) is too short"):
        mod.diff_endog(np.zeros((5, 2)))


def test_bind_one_dimensional_observation():
    # A 1-dim array with k_endog elements is a single observation, as is the
    # same values as one row-ordered row
    mod = Representation(2, k_states=1)
    mod.bind(np.array([1.0, 2.0]))
    desired = Representation(2, k_states=1)
    desired.bind(np.array([[1.0, 2.0]]))
    assert mod.nobs == desired.nobs == 1
    assert_allclose(mod.endog, desired.endog)
    assert_allclose(mod.endog, [[1.0], [2.0]])
    with pytest.raises(ValueError, match="must have first axis shape of `k_endog`"):
        mod.bind(np.arange(6.0))


def test_bind_non_contiguous():
    mod = Representation(2, k_states=1)
    with pytest.raises(ValueError, match="must be ordered in contiguous memory"):
        mod.bind(np.zeros((10, 4))[:, ::2])


def test_filter_incomplete_initialization():
    mod = _bivariate_model()
    init = Initialization(2)
    init.set(0, "diffuse")
    mod.initialize(init)
    with pytest.raises(RuntimeError, match="Initialization is incomplete"):
        mod.filter()


def test_frozen_representation_without_initialization():
    mod = _bivariate_model()
    # Filtering creates the compiled representation that is frozen below
    mod.filter()
    mod.initialization = None
    frozen = FrozenRepresentation(mod)
    assert frozen.initialization is None
    assert frozen.initial_state is None
    assert frozen.initial_state_cov is None
    assert_allclose(frozen.endog, mod.endog)
    for name in ["design", "obs_cov", "transition", "selection", "state_cov"]:
        assert_allclose(getattr(frozen, name), getattr(mod, name))


def test_update_representation_only_options():
    mod = _bivariate_model()
    res = mod.filter()
    obs_cov = mod["obs_cov"].copy()
    mod.filter_univariate = True
    mod["obs_cov"] = 5 * np.eye(2)
    # Only the options are taken from the model
    res.update_representation(mod, only_options=True)
    assert res.filter_univariate
    assert_allclose(res.obs_cov[..., 0], obs_cov)
    # Otherwise the system matrices are updated too
    res.update_representation(mod)
    assert_allclose(res.obs_cov[..., 0], 5 * np.eye(2))


@pytest.mark.parametrize("missing", ["none", "partial", "entire"])
def test_kalman_gain_not_stored(missing):
    # Without a stored gain, `kalman_gain` computes K_t = T P_t Z' F_t^{-1}
    # from the stored covariances; the filter's own gain is the reference
    endog = _bivariate_endog()
    if missing == "partial":
        endog[3, 0] = np.nan
    elif missing == "entire":
        endog[5] = np.nan
    desired = _bivariate_model(endog=endog).filter().kalman_gain
    mod = _bivariate_model(endog=endog)
    mod.memory_no_gain = True
    actual = mod.filter().kalman_gain
    # atol: the gain columns of missing observations are zero in both
    assert_allclose(actual, desired, rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize(
    "option", ["memory_no_forecast_mean", "memory_no_forecast_cov"]
)
def test_partially_missing_forecasts_memory_options(option):
    # For a partially missing observation, the forecasts and their covariance
    # are rebuilt for all series from the predicted state, Z a_t + d and
    # Z P_t Z' + H, unless they are not stored
    endog = _bivariate_endog()
    endog[3, 0] = np.nan
    mod = _bivariate_model(endog=endog)
    setattr(mod, option, True)
    res = mod.filter()
    design = mod["design"]
    if option == "memory_no_forecast_cov":
        forecast = design @ res.predicted_state[:, 3]
        assert_allclose(res.forecasts[:, 3], forecast)
        # The filter stores the forecast of the observed series first
        assert_allclose(res.forecasts[1, 3], res.missing_forecasts[0, 3])
        assert np.isnan(res.forecasts_error[0, 3])
        assert_allclose(res.forecasts_error[1, 3], endog[3, 1] - forecast[1])
    else:
        P = res.predicted_state_cov[:, :, 3]
        desired = design @ P @ design.T + mod["obs_cov"]
        assert_allclose(res.forecasts_error_cov[:, :, 3], desired)
        assert_allclose(
            res.forecasts_error_cov[1, 1, 3], res.missing_forecasts_error_cov[0, 0, 3]
        )


def test_impulse_responses_column_impulse():
    mod = _bivariate_model(cls=SimulationSmoother)
    design, transition = mod["design"], mod["transition"]
    impulse = np.array([0.5, -1.0])
    # Response h periods after the impulse: Z T^h R impulse
    desired = np.array(
        [
            design @ np.linalg.matrix_power(transition, h) @ mod["selection"] @ impulse
            for h in range(4)
        ]
    )
    actual = mod.impulse_responses(steps=3, impulse=impulse[:, None])
    assert_allclose(actual, desired)

    mod["design"] = np.repeat(design[:, :, None], mod.nobs, axis=2)
    match = "In a time-varying model, cannot create more impulse responses"
    with pytest.raises(ValueError, match=match):
        mod.impulse_responses(steps=mod.nobs)


def _quadratic_form(design, cov):
    # Z P_t Z' for each period t of cov
    return np.einsum("ij,jkt,lk->ilt", design, cov, design)


def test_prediction_results_signal_cov_before_signal():
    mod = _bivariate_model(cls=KalmanSmoother)
    obs_intercept = np.array([0.5, -0.5])
    mod["obs_intercept"] = obs_intercept
    res = mod.smooth()
    design = mod["design"]
    for name in ["predicted", "filtered", "smoothed"]:
        pred = res.predict(start=0, end=mod.nobs + 5)
        states = getattr(pred, f"{name}_state")
        states_cov = getattr(pred, f"{name}_state_cov")
        # The covariance first, then the mean
        assert_allclose(
            getattr(pred, f"{name}_signal_cov"), _quadratic_form(design, states_cov)
        )
        assert_allclose(getattr(pred, f"{name}_signal"), design @ states)
        if name == "predicted":
            continue
        forecasts = obs_intercept[:, None] + design @ states
        forecasts_cov = _quadratic_form(design, states_cov) + mod["obs_cov"][..., None]
        assert_allclose(getattr(pred, f"{name}_forecasts"), forecasts)
        assert_allclose(getattr(pred, f"{name}_forecasts_error_cov"), forecasts_cov)
        # A second access returns the stored values
        assert_allclose(getattr(pred, f"{name}_forecasts"), forecasts)


def test_prediction_results_time_varying_dynamic():
    mod = _bivariate_model()
    nobs = mod.nobs
    obs_intercept = np.linspace(-1, 1, 2 * nobs).reshape(2, nobs)
    mod["obs_intercept"] = obs_intercept
    res = mod.filter()
    # Static, dynamic and out-of-sample periods, with the intercept of the
    # out-of-sample periods given
    new_intercept = np.linspace(2, 3, 2 * 5).reshape(2, 5)
    pred = res.predict(
        start=0, end=nobs + 5, dynamic=nobs - 10, obs_intercept=new_intercept
    )
    assert (pred.ndynamic, pred.nforecast) == (10, 5)
    desired = np.c_[obs_intercept, new_intercept]
    assert_allclose(pred.obs_intercept, desired)
    # Each forecast uses the intercept of its period
    forecasts = mod["design"] @ pred.predicted_state + desired
    assert_allclose(pred.forecasts, forecasts)


def _initial_values_model(**kwargs):
    mod = KalmanFilter(np.arange(10.0)[:, None].copy(), k_states=2, **kwargs)
    mod["design"] = [[1.0, 0.5]]
    mod["transition"] = [[0.5, 0.1], [0.0, 0.3]]
    mod["selection"] = np.eye(2)
    mod["state_cov"] = np.eye(2)
    mod["obs_cov"] = [[1.0]]
    return mod


@pytest.mark.parametrize("initial_variance", [None, 123.0])
def test_filter_results_initial_variance(initial_variance):
    # The results hold the `initial_variance` of the model (1e6 unless
    # given), which is the variance of the approximate diffuse initialization
    kwargs = {}
    desired = 1e6
    if initial_variance is not None:
        kwargs["initial_variance"] = initial_variance
        desired = initial_variance
    mod = _initial_values_model(**kwargs)
    mod.initialize_approximate_diffuse()
    res = mod.filter()
    assert_allclose(res.initial_variance, desired, rtol=1e-12)
    assert_allclose(res.initial_state_cov, desired * np.eye(2), rtol=1e-12)


def test_frozen_representation_initial_diffuse_state_cov():
    # Without an initialization the initial values are None, as documented
    mod = _initial_values_model()
    mod.initialize_diffuse()
    # Filtering creates the compiled representation that is read
    mod.filter()
    frozen = FrozenRepresentation(mod)
    assert_allclose(frozen.initial_diffuse_state_cov, np.eye(2), rtol=1e-12)

    mod.initialization = None
    frozen = FrozenRepresentation(mod)
    assert frozen.initialization is None
    assert frozen.initial_state is None
    assert frozen.initial_state_cov is None
    assert frozen.initial_diffuse_state_cov is None
    assert_allclose(frozen.initial_variance, 1e6, rtol=1e-12)
