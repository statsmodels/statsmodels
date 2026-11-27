"""
Tests for _representation and _kalman_filter modules

Author: Chad Fulton
License: Simplified-BSD

References
----------

Kim, Chang-Jin, and Charles R. Nelson. 1999.
"State-Space Models with Regime Switching:
Classical and Gibbs-Sampling Approaches with Applications".
MIT Press Books. The MIT Press.

Hamilton, James D. 1994.
Time Series Analysis.
Princeton, N.J.: Princeton University Press.
"""
import copy
from pathlib import Path
import pickle

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov
from scipy.linalg.blas import find_best_blas_type

from statsmodels.tsa.statespace import _kalman_filter, _representation
from statsmodels.tsa.statespace.kalman_filter import (
    MEMORY_CONSERVE,
    MEMORY_NO_FORECAST,
    MEMORY_NO_PREDICTED,
    KalmanFilter,
)
from statsmodels.tsa.statespace.mlemodel import MLEModel

from .results import results_kalman_filter

prefix_statespace_map = {
    "s": _representation.sStatespace,
    "d": _representation.dStatespace,
    "c": _representation.cStatespace,
    "z": _representation.zStatespace,
}
prefix_kalman_filter_map = {
    "s": _kalman_filter.sKalmanFilter,
    "d": _kalman_filter.dKalmanFilter,
    "c": _kalman_filter.cKalmanFilter,
    "z": _kalman_filter.zKalmanFilter,
}

current_path = Path(__file__).resolve().parent


class Clark1987:
    """
    Clark's (1987) univariate unobserved components model of real GDP (as
    presented in Kim and Nelson, 1999)

    Test data produced using GAUSS code described in Kim and Nelson (1999) and
    found at http://econ.korea.ac.kr/~cjkim/SSMARKOV.htm

    See `results.results_kalman_filter` for more information.
    """

    # Tolerance for the pickled and the copied filter, which repeat the same
    # computations
    copy_rtol = 1e-7

    @classmethod
    def setup_class(cls, dtype=float, conserve_memory=0, loglikelihood_burn=0):
        cls.true = results_kalman_filter.uc_uni
        cls.true_states = pd.DataFrame(cls.true["states"])

        # GDP, Quarterly, 1947.1 - 1995.3
        data = pd.DataFrame(
            cls.true["data"],
            index=pd.date_range("1947-01-01", "1995-07-01", freq="QS"),
            columns=["GDP"],
        )
        data["lgdp"] = np.log(data["GDP"])

        # Parameters
        cls.conserve_memory = conserve_memory
        cls.loglikelihood_burn = loglikelihood_burn

        # Observed data
        cls.obs = np.require(
            np.array(data["lgdp"], ndmin=2, dtype=dtype, order="F"), requirements="W"
        )

        # Measurement equation
        cls.k_endog = k_endog = 1  # dimension of observed data
        # design matrix
        cls.design = np.zeros((k_endog, 4, 1), dtype=dtype, order="F")
        cls.design[:, :, 0] = [1, 1, 0, 0]
        # observation intercept
        cls.obs_intercept = np.zeros((k_endog, 1), dtype=dtype, order="F")
        # observation covariance matrix
        cls.obs_cov = np.zeros((k_endog, k_endog, 1), dtype=dtype, order="F")

        # Transition equation
        cls.k_states = k_states = 4  # dimension of state space
        # transition matrix
        cls.transition = np.zeros((k_states, k_states, 1), dtype=dtype, order="F")
        cls.transition[([0, 0, 1, 1, 2, 3], [0, 3, 1, 2, 1, 3], [0, 0, 0, 0, 0, 0])] = [
            1,
            1,
            0,
            0,
            1,
            1,
        ]
        # state intercept
        cls.state_intercept = np.zeros((k_states, 1), dtype=dtype, order="F")
        # selection matrix
        cls.selection = np.asfortranarray(np.eye(k_states)[:, :, None], dtype=dtype)
        # state covariance matrix
        cls.state_cov = np.zeros((k_states, k_states, 1), dtype=dtype, order="F")

        # Initialization: Diffuse priors
        cls.initial_state = np.zeros((k_states,), dtype=dtype, order="F")
        cls.initial_state_cov = np.asfortranarray(np.eye(k_states) * 100, dtype=dtype)

        # Update matrices with given parameters
        sigma_v, sigma_e, sigma_w, phi_1, phi_2 = np.array(
            cls.true["parameters"], dtype=dtype
        )
        cls.transition[([1, 1], [1, 2], [0, 0])] = [phi_1, phi_2]
        cls.state_cov[np.diag_indices(k_states) + (np.zeros(k_states, dtype=int),)] = [
            sigma_v**2,
            sigma_e**2,
            0,
            sigma_w**2,
        ]

        # Initialization: modification
        # Due to the difference in the way Kim and Nelson (1999) and Durbin
        # and Koopman (2012) define the order of the Kalman filter routines,
        # we need to modify the initial state covariance matrix to match
        # Kim and Nelson's results, since the *Statespace models follow Durbin
        # and Koopman.
        cls.initial_state_cov = np.asfortranarray(
            np.dot(
                np.dot(cls.transition[:, :, 0], cls.initial_state_cov),
                cls.transition[:, :, 0].T,
            )
        )

    @classmethod
    def init_filter(cls):
        # Use the appropriate Statespace model
        prefix = find_best_blas_type((cls.obs,))
        klass = prefix_statespace_map[prefix[0]]

        # Instantiate the statespace model
        model = klass(
            cls.obs,
            cls.design,
            cls.obs_intercept,
            cls.obs_cov,
            cls.transition,
            cls.state_intercept,
            cls.selection,
            cls.state_cov,
        )
        model.initialize_known(cls.initial_state, cls.initial_state_cov)

        # Initialize the appropriate Kalman filter
        klass = prefix_kalman_filter_map[prefix[0]]
        kfilter = klass(
            model,
            conserve_memory=cls.conserve_memory,
            loglikelihood_burn=cls.loglikelihood_burn,
        )

        return model, kfilter

    @classmethod
    def run_filter(cls):
        # Filter the data
        cls.filter()

        # Get results
        return {
            "loglike": lambda burn: np.sum(cls.filter.loglikelihood[burn:]),
            "state": np.array(cls.filter.filtered_state),
        }

    def test_loglike(self):
        assert_almost_equal(
            self.result["loglike"](self.true["start"]), self.true["loglike"], 5
        )

    def test_filtered_state(self):
        assert_almost_equal(
            self.result["state"][0][self.true["start"] :],
            self.true_states.iloc[:, 0],
            4,
        )
        assert_almost_equal(
            self.result["state"][1][self.true["start"] :],
            self.true_states.iloc[:, 1],
            4,
        )
        assert_almost_equal(
            self.result["state"][3][self.true["start"] :],
            self.true_states.iloc[:, 2],
            4,
        )

    @pytest.mark.thread_unsafe("Pickle can't be tested in parallel")
    def test_pickled_filter(self):
        pickled = pickle.loads(pickle.dumps(self.filter))
        #  Run the filters
        self.filter()
        pickled()

        assert id(filter) != id(pickled)
        assert_allclose(
            np.array(self.filter.filtered_state),
            np.array(pickled.filtered_state),
            rtol=self.copy_rtol,
        )
        assert_allclose(
            np.array(self.filter.loglikelihood),
            np.array(pickled.loglikelihood),
            rtol=self.copy_rtol,
        )

    def test_copied_filter(self):
        # To make thread-safe, we need a fitler that is local to this test
        # rather than self.filter
        _, filter = self.init_filter()
        filter()
        copied = copy.deepcopy(filter)
        #  Run the filters
        copied()

        assert id(filter) != id(copied)
        assert_allclose(
            np.array(self.filter.filtered_state),
            np.array(copied.filtered_state),
            rtol=self.copy_rtol,
        )

        assert_allclose(
            np.array(self.filter.loglikelihood),
            np.array(copied.loglikelihood),
            rtol=self.copy_rtol,
        )


class TestClark1987Single(Clark1987):
    """
    Basic single precision test for the loglikelihood and filtered states.

    In single precision the loglikelihood differs from the published value by
    about 2e-4 (relative) and the filtered states by about 1e-3 (absolute),
    hence the tolerances of test_loglike and test_filtered_state.
    """

    # Allow a few units in the last place for the pickled and copied filters,
    # whose arrays are new allocations (the summation order of some BLAS
    # libraries depends on the memory alignment)
    copy_rtol = 1e-5

    @classmethod
    def setup_class(cls):
        super().setup_class(dtype=np.float32, conserve_memory=0)
        cls.model, cls.filter = cls.init_filter()
        cls.result = cls.run_filter()

    def test_dtype(self):
        # The single precision routines are used
        assert isinstance(self.model, _representation.sStatespace)
        assert isinstance(self.filter, _kalman_filter.sKalmanFilter)
        assert self.result["state"].dtype == np.float32
        assert np.asarray(self.filter.loglikelihood).dtype == np.float32

    def test_loglike(self):
        assert_allclose(
            self.result["loglike"](self.true["start"]), self.true["loglike"], rtol=1e-3
        )

    def test_filtered_state(self):
        assert_allclose(
            self.result["state"][0][self.true["start"] :],
            self.true_states.iloc[:, 0],
            atol=1e-2,
        )
        assert_allclose(
            self.result["state"][1][self.true["start"] :],
            self.true_states.iloc[:, 1],
            atol=1e-2,
        )
        assert_allclose(
            self.result["state"][3][self.true["start"] :],
            self.true_states.iloc[:, 2],
            atol=1e-2,
        )


class TestClark1987Double(Clark1987):
    """
    Basic double precision test for the loglikelihood and filtered states.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class(dtype=float, conserve_memory=0)
        cls.model, cls.filter = cls.init_filter()
        cls.result = cls.run_filter()


class TestClark1987SingleComplex(Clark1987):
    """
    Basic single precision complex test for the loglikelihood and filtered
    states.

    The tolerances are those of TestClark1987Single.
    """

    copy_rtol = 1e-5

    @classmethod
    def setup_class(cls):
        super().setup_class(dtype=np.complex64, conserve_memory=0)
        cls.model, cls.filter = cls.init_filter()
        cls.result = cls.run_filter()

    def test_dtype(self):
        # The single precision complex routines are used
        assert isinstance(self.model, _representation.cStatespace)
        assert isinstance(self.filter, _kalman_filter.cKalmanFilter)
        assert self.result["state"].dtype == np.complex64
        assert np.asarray(self.filter.loglikelihood).dtype == np.complex64
        # The imaginary part is zero in exact arithmetic; allow single
        # precision rounding errors relative to the size of the states
        state = self.result["state"]
        assert_allclose(state.imag, 0, atol=1e-6 * np.abs(state.real).max())

    def test_loglike(self):
        assert_allclose(
            self.result["loglike"](self.true["start"]), self.true["loglike"], rtol=1e-3
        )

    def test_filtered_state(self):
        assert_allclose(
            self.result["state"][0][self.true["start"] :],
            self.true_states.iloc[:, 0],
            atol=1e-2,
        )
        assert_allclose(
            self.result["state"][1][self.true["start"] :],
            self.true_states.iloc[:, 1],
            atol=1e-2,
        )
        assert_allclose(
            self.result["state"][3][self.true["start"] :],
            self.true_states.iloc[:, 2],
            atol=1e-2,
        )


class TestClark1987DoubleComplex(Clark1987):
    """
    Basic double precision complex test for the loglikelihood and filtered
    states.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class(dtype=complex, conserve_memory=0)
        cls.model, cls.filter = cls.init_filter()
        cls.result = cls.run_filter()


class TestClark1987Conserve(Clark1987):
    """
    Memory conservation test for the loglikelihood and filtered states.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class(
            dtype=float, conserve_memory=MEMORY_NO_FORECAST | MEMORY_NO_PREDICTED
        )
        cls.model, cls.filter = cls.init_filter()
        cls.result = cls.run_filter()


class Clark1987Forecast(Clark1987):
    """
    Forecasting test for the loglikelihood and filtered states.
    """

    @classmethod
    def setup_class(cls, dtype=float, nforecast=100, conserve_memory=0):
        super().setup_class(dtype, conserve_memory)
        cls.nforecast = nforecast

        # Add missing observations to the end (to forecast)
        cls._obs = cls.obs
        cls.obs = np.array(
            np.r_[cls.obs[0, :], [np.nan] * nforecast], ndmin=2, dtype=dtype, order="F"
        )

    def test_filtered_state(self):
        assert_almost_equal(
            self.result["state"][0][self.true["start"] : -self.nforecast],
            self.true_states.iloc[:, 0],
            4,
        )
        assert_almost_equal(
            self.result["state"][1][self.true["start"] : -self.nforecast],
            self.true_states.iloc[:, 1],
            4,
        )
        assert_almost_equal(
            self.result["state"][3][self.true["start"] : -self.nforecast],
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
        cls.model, cls.filter = cls.init_filter()
        cls.result = cls.run_filter()


class TestClark1987ForecastDoubleComplex(Clark1987Forecast):
    """
    Basic double complex forecasting test for the loglikelihood and filtered
    states.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class(dtype=complex)
        cls.model, cls.filter = cls.init_filter()
        cls.result = cls.run_filter()


class TestClark1987ForecastConserve(Clark1987Forecast):
    """
    Memory conservation forecasting test for the loglikelihood and filtered
    states.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class(
            dtype=float, conserve_memory=MEMORY_NO_FORECAST | MEMORY_NO_PREDICTED
        )
        cls.model, cls.filter = cls.init_filter()
        cls.result = cls.run_filter()


class TestClark1987ConserveAll(Clark1987):
    """
    Memory conservation forecasting test for the loglikelihood and filtered
    states.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class(dtype=float, conserve_memory=MEMORY_CONSERVE)
        cls.loglikelihood_burn = cls.true["start"]
        cls.model, cls.filter = cls.init_filter()
        cls.result = cls.run_filter()

    def test_loglike(self):
        assert_almost_equal(self.result["loglike"](0), self.true["loglike"], 5)

    def test_filtered_state(self):
        end = self.true_states.shape[0]
        assert_almost_equal(
            self.result["state"][0][-1], self.true_states.iloc[end - 1, 0], 4
        )
        assert_almost_equal(
            self.result["state"][1][-1], self.true_states.iloc[end - 1, 1], 4
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
    def setup_class(cls, dtype=float, conserve_memory=0, loglikelihood_burn=0):
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

        # Observed data
        cls.obs = np.array(data, ndmin=2, dtype=dtype, order="C").T

        # Parameters
        cls.k_endog = k_endog = 2  # dimension of observed data
        cls.k_states = k_states = 6  # dimension of state space
        cls.conserve_memory = conserve_memory
        cls.loglikelihood_burn = loglikelihood_burn

        # Measurement equation

        # design matrix
        cls.design = np.zeros((k_endog, k_states, 1), dtype=dtype, order="F")
        cls.design[:, :, 0] = [[1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]]
        # observation intercept
        cls.obs_intercept = np.zeros((k_endog, 1), dtype=dtype, order="F")
        # observation covariance matrix
        cls.obs_cov = np.zeros((k_endog, k_endog, 1), dtype=dtype, order="F")

        # Transition equation

        # transition matrix
        cls.transition = np.zeros((k_states, k_states, 1), dtype=dtype, order="F")
        cls.transition[
            (
                [0, 0, 1, 1, 2, 3, 4, 5],
                [0, 4, 1, 2, 1, 2, 4, 5],
                [0, 0, 0, 0, 0, 0, 0, 0],
            )
        ] = [1, 1, 0, 0, 1, 1, 1, 1]
        # state intercept
        cls.state_intercept = np.zeros((k_states, 1), dtype=dtype, order="F")
        # selection matrix
        cls.selection = np.asfortranarray(np.eye(k_states)[:, :, None], dtype=dtype)
        # state covariance matrix
        cls.state_cov = np.zeros((k_states, k_states, 1), dtype=dtype, order="F")

        # Initialization: Diffuse priors
        cls.initial_state = np.zeros((k_states,), dtype=dtype)
        cls.initial_state_cov = np.asfortranarray(np.eye(k_states) * 100, dtype=dtype)

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
        ) = np.array(cls.true["parameters"], dtype=dtype)
        cls.design[([1, 1, 1], [1, 2, 3], [0, 0, 0])] = [alpha_1, alpha_2, alpha_3]
        cls.transition[([1, 1], [1, 2], [0, 0])] = [phi_1, phi_2]
        cls.obs_cov[1, 1, 0] = sigma_ec**2
        cls.state_cov[np.diag_indices(k_states) + (np.zeros(k_states, dtype=int),)] = [
            sigma_v**2,
            sigma_e**2,
            0,
            0,
            sigma_w**2,
            sigma_vl**2,
        ]

        # Initialization: modification
        # Due to the difference in the way Kim and Nelson (1999) and Drubin
        # and Koopman (2012) define the order of the Kalman filter routines,
        # we need to modify the initial state covariance matrix to match
        # Kim and Nelson's results, since the *Statespace models follow Durbin
        # and Koopman.
        cls.initial_state_cov = np.asfortranarray(
            np.dot(
                np.dot(cls.transition[:, :, 0], cls.initial_state_cov),
                cls.transition[:, :, 0].T,
            )
        )

    @classmethod
    def init_filter(cls):
        # Use the appropriate Statespace model
        prefix = find_best_blas_type((cls.obs,))
        klass = prefix_statespace_map[prefix[0]]

        # Instantiate the statespace model
        model = klass(
            cls.obs,
            cls.design,
            cls.obs_intercept,
            cls.obs_cov,
            cls.transition,
            cls.state_intercept,
            cls.selection,
            cls.state_cov,
        )
        model.initialize_known(cls.initial_state, cls.initial_state_cov)

        # Initialize the appropriate Kalman filter
        klass = prefix_kalman_filter_map[prefix[0]]
        kfilter = klass(
            model,
            conserve_memory=cls.conserve_memory,
            loglikelihood_burn=cls.loglikelihood_burn,
        )

        return model, kfilter

    @classmethod
    def run_filter(cls):
        # Filter the data
        cls.filter()

        # Get results
        return {
            "loglike": lambda burn: np.sum(cls.filter.loglikelihood[burn:]),
            "state": np.array(cls.filter.filtered_state),
        }

    def test_loglike(self):
        assert_almost_equal(
            # self.result['loglike'](self.true['start']),
            self.result["loglike"](0),
            self.true["loglike"],
            2,
        )

    def test_filtered_state(self):
        assert_almost_equal(
            self.result["state"][0][self.true["start"] :],
            self.true_states.iloc[:, 0],
            4,
        )
        assert_almost_equal(
            self.result["state"][1][self.true["start"] :],
            self.true_states.iloc[:, 1],
            4,
        )
        assert_almost_equal(
            self.result["state"][4][self.true["start"] :],
            self.true_states.iloc[:, 2],
            4,
        )
        assert_almost_equal(
            self.result["state"][5][self.true["start"] :],
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
        cls.model, cls.filter = cls.init_filter()
        cls.result = cls.run_filter()


class TestClark1989Conserve(Clark1989):
    """
    Memory conservation test for the loglikelihood and filtered states with
    two-dimensional observation vector.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class(
            dtype=float, conserve_memory=MEMORY_NO_FORECAST | MEMORY_NO_PREDICTED
        )
        cls.model, cls.filter = cls.init_filter()
        cls.result = cls.run_filter()


class Clark1989Forecast(Clark1989):
    """
    Memory conservation test for the loglikelihood and filtered states with
    two-dimensional observation vector.
    """

    @classmethod
    def setup_class(cls, dtype=float, nforecast=100, conserve_memory=0):
        super().setup_class(dtype, conserve_memory)
        cls.nforecast = nforecast

        # Add missing observations to the end (to forecast)
        cls._obs = cls.obs
        cls.obs = np.array(
            np.c_[cls._obs, np.r_[[np.nan, np.nan] * nforecast].reshape(2, nforecast)],
            ndmin=2,
            dtype=dtype,
            order="F",
        )

        cls.model, cls.filter = cls.init_filter()
        cls.result = cls.run_filter()

    def test_filtered_state(self):
        assert_almost_equal(
            self.result["state"][0][self.true["start"] : -self.nforecast],
            self.true_states.iloc[:, 0],
            4,
        )
        assert_almost_equal(
            self.result["state"][1][self.true["start"] : -self.nforecast],
            self.true_states.iloc[:, 1],
            4,
        )
        assert_almost_equal(
            self.result["state"][4][self.true["start"] : -self.nforecast],
            self.true_states.iloc[:, 2],
            4,
        )
        assert_almost_equal(
            self.result["state"][5][self.true["start"] : -self.nforecast],
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
        cls.model, cls.filter = cls.init_filter()
        cls.result = cls.run_filter()


class TestClark1989ForecastDoubleComplex(Clark1989Forecast):
    """
    Basic double complex forecasting test for the loglikelihood and filtered
    states.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class(dtype=complex)
        cls.model, cls.filter = cls.init_filter()
        cls.result = cls.run_filter()


class TestClark1989ForecastConserve(Clark1989Forecast):
    """
    Memory conservation forecasting test for the loglikelihood and filtered
    states.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class(
            dtype=float, conserve_memory=MEMORY_NO_FORECAST | MEMORY_NO_PREDICTED
        )
        cls.model, cls.filter = cls.init_filter()
        cls.result = cls.run_filter()


class TestClark1989ConserveAll(Clark1989):
    """
    Memory conservation forecasting test for the loglikelihood and filtered
    states.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class(
            dtype=float,
            conserve_memory=MEMORY_CONSERVE,
        )
        # cls.loglikelihood_burn = cls.true['start']
        cls.loglikelihood_burn = 0
        cls.model, cls.filter = cls.init_filter()
        cls.result = cls.run_filter()

    def test_loglike(self):
        assert_almost_equal(self.result["loglike"](0), self.true["loglike"], 2)

    def test_filtered_state(self):
        end = self.true_states.shape[0]
        assert_almost_equal(
            self.result["state"][0][-1], self.true_states.iloc[end - 1, 0], 4
        )
        assert_almost_equal(
            self.result["state"][1][-1], self.true_states.iloc[end - 1, 1], 4
        )
        assert_almost_equal(
            self.result["state"][4][-1], self.true_states.iloc[end - 1, 2], 4
        )
        assert_almost_equal(
            self.result["state"][5][-1], self.true_states.iloc[end - 1, 3], 4
        )


def check_stationary_initialization_1dim(dtype=float):
    endog = np.zeros(10, dtype=dtype)

    # 1-dimensional example
    mod = MLEModel(endog, k_states=1, k_posdef=1)
    mod.ssm.initialize_stationary()
    intercept = np.array([2.3], dtype=dtype)
    phi = np.diag([0.9]).astype(dtype)
    sigma2 = np.diag([1.3]).astype(dtype)

    mod["state_intercept"] = intercept
    mod["transition"] = phi
    mod["selection"] = np.eye(1).astype(dtype)
    mod["state_cov"] = sigma2

    mod.ssm._initialize_filter()
    mod.ssm._initialize_state()

    _statespace = mod.ssm._statespace
    initial_state = np.array(_statespace.initial_state)
    initial_state_cov = np.array(_statespace.initial_state_cov)
    # precision reductions only required for float complex case

    # mean = intercept + phi * mean
    # intercept = (1 - phi) * mean
    # mean = intercept / (1 - phi)
    assert_allclose(initial_state, intercept / (1 - phi[0, 0]))
    desired = np.linalg.inv(np.eye(1) - phi).dot(intercept)
    assert_allclose(initial_state, desired)
    # var = phi**2 var + sigma2
    # var = sigma2 / (1 - phi**2)
    assert_allclose(initial_state_cov, sigma2 / (1 - phi**2))
    assert_allclose(initial_state_cov, solve_discrete_lyapunov(phi, sigma2))


def check_stationary_initialization_2dim(dtype=float):
    endog = np.zeros(10, dtype=dtype)
    # 2-dimensional example
    mod = MLEModel(endog, k_states=2, k_posdef=2)
    mod.ssm.initialize_stationary()
    intercept = np.array([2.3, -10.2], dtype=dtype)
    phi = np.array([[0.8, 0.1], [-0.2, 0.7]], dtype=dtype)
    sigma2 = np.array([[1.4, -0.2], [-0.2, 4.5]], dtype=dtype)

    mod["state_intercept"] = intercept
    mod["transition"] = phi
    mod["selection"] = np.eye(2).astype(dtype)
    mod["state_cov"] = sigma2

    mod.ssm._initialize_filter()
    mod.ssm._initialize_state()

    _statespace = mod.ssm._statespace
    initial_state = np.array(_statespace.initial_state)
    initial_state_cov = np.array(_statespace.initial_state_cov)

    desired = np.linalg.solve(np.eye(2).astype(dtype) - phi, intercept)
    assert_allclose(initial_state, desired)
    desired = solve_discrete_lyapunov(phi, sigma2)
    # precision reductions only required for single precision float / complex
    assert_allclose(initial_state_cov, desired, atol=1e-5)


def test_stationary_initialization():
    check_stationary_initialization_1dim(np.float32)
    check_stationary_initialization_1dim(np.float64)
    check_stationary_initialization_1dim(np.complex64)
    check_stationary_initialization_1dim(np.complex128)

    check_stationary_initialization_2dim(np.float32)
    check_stationary_initialization_2dim(np.float64)
    check_stationary_initialization_2dim(np.complex64)
    check_stationary_initialization_2dim(np.complex128)


def test_prediction_results_clear():
    # `PredictionResults.clear()` deletes the cached `_<attr>` values for
    # `endog` plus the `representation_attributes` and `filter_attributes`
    # lists (see the method body), forcing them to be recomputed via
    # `__getattr__` on next access. It deliberately leaves other caches
    # (e.g. smoother_attributes) untouched.
    endog = np.arange(10, dtype=float)
    mod = MLEModel(endog, k_states=1, k_posdef=1)
    mod.ssm.initialize_known(np.array([0.0]), np.array([[1.0]]))
    mod["design", 0, 0] = 1.0
    mod["obs_cov", 0, 0] = 1.0
    mod["transition", 0, 0] = 0.5
    mod["selection", 0, 0] = 1.0
    mod["state_cov", 0, 0] = 1.0

    res = mod.ssm.filter()
    pred = res.predict()

    cached_attrs = ["endog"] + pred.representation_attributes + pred.filter_attributes

    # Access each attribute once so that it gets cached as `_<attr>`
    values_before = {}
    for attr in cached_attrs:
        values_before[attr] = np.array(getattr(pred, attr), copy=True)
        assert hasattr(pred, "_" + attr)

    pred.clear()

    # Every cache that `clear()` is documented to reset is actually gone
    for attr in cached_attrs:
        assert not hasattr(pred, "_" + attr)

    # Re-accessing reproduces identical values and re-populates the cache
    # (clear() only drops the cache; it cannot change the underlying,
    # already-computed filter output)
    for attr in cached_attrs:
        assert_allclose(getattr(pred, attr), values_before[attr])
        assert hasattr(pred, "_" + attr)


@pytest.mark.parametrize("filter_univariate", [False, True])
@pytest.mark.parametrize("k_states", [1, 2])
def test_filtered_timing_converged_last_predicted_state_cov(k_states, filter_univariate):
    # With filtered timing, the predicted state covariance matrix for the
    # period after the sample (the last column of predicted_state_cov) was
    # left at zero once the filter had converged, since a converged filter
    # does not compute it. It is the converged matrix, as without convergence
    # (tolerance=0) and as the solution of the Riccati equation.
    endog = np.random.default_rng(5820).standard_normal((50, k_states))
    design = np.eye(k_states)
    obs_cov = np.diag([1.0, 0.6][:k_states])
    transition = np.array([[0.5, 0.1], [0.0, 0.3]])[:k_states, :k_states]
    state_cov = np.array([[1.0, 0.2], [0.2, 0.5]])[:k_states, :k_states]
    kf = KalmanFilter(k_states, k_states, k_posdef=k_states)
    kf.bind(endog)
    kf["design"] = design
    kf["obs_cov"] = obs_cov
    kf["transition"] = transition
    kf["selection"] = np.eye(k_states)
    kf["state_cov"] = state_cov
    kf.initialize_known(np.zeros(k_states), np.eye(k_states))
    kf.timing_init_filtered = True
    kf.filter_univariate = filter_univariate

    res = kf.filter()
    assert res.converged
    assert res.period_converged < kf.nobs - 1
    kf.tolerance = 0
    desired = kf.filter()
    assert not desired.converged
    riccati = solve_discrete_are(transition.T, design.T, state_cov, obs_cov)

    actual = res.predicted_state_cov[..., -1]
    assert_allclose(actual, desired.predicted_state_cov[..., -1], rtol=1e-8)
    assert_allclose(actual, riccati, rtol=1e-8)
    assert_allclose(
        res.predicted_state[..., -1], desired.predicted_state[..., -1], rtol=1e-8
    )


@pytest.mark.parametrize("dynamic", [None, 2])
def test_filtered_timing_prediction(dynamic):
    # FilterResults.predict (forecasts, dynamic prediction, and so
    # get_forecast and get_prediction) runs a filter on missing data that
    # starts from the predicted state of its first period. With filtered
    # timing that filter treated it as the filtered state of the period before,
    # so the predictions were one period late. They equal those with the
    # predicted timing and the closed form of an AR(1) observed with noise.
    phi, sigma2, obs_var = 0.5, 1.0, 0.8
    endog = np.random.default_rng(3029).standard_normal((20, 1))

    def predict(filtered_timing):
        kf = KalmanFilter(1, 1, k_posdef=1)
        kf.bind(endog)
        kf["design"] = [[1.0]]
        kf["obs_cov"] = [[obs_var]]
        kf["transition"] = [[phi]]
        kf["selection"] = [[1.0]]
        kf["state_cov"] = [[sigma2]]
        # The stationary distribution, the same for alpha_0 and alpha_1
        kf.initialize_known(np.zeros(1), [[sigma2 / (1 - phi**2)]])
        kf.timing_init_filtered = filtered_timing
        res = kf.filter()
        if dynamic is None:
            start, end = kf.nobs, kf.nobs + 3
        else:
            start, end = 10, 15
        return res, res.predict(start=start, end=end, dynamic=dynamic)

    res, pred = predict(True)
    _, desired = predict(False)
    assert_allclose(pred.forecasts, desired.forecasts, rtol=1e-10)
    assert_allclose(pred.forecasts_error_cov, desired.forecasts_error_cov, rtol=1e-10)

    # Closed form: from the filtered state a_{t|t} and its variance P_{t|t}
    # of the last observed period t, the h-step prediction has mean
    # phi^h a_{t|t} and variance phi^(2h) P_{t|t} + sigma2 (1 + phi^2 + ...
    # + phi^(2(h-1))) + obs_var
    last = res.nobs - 1 if dynamic is None else 11
    steps = np.arange(1, 4)
    mean = phi**steps * res.filtered_state[0, last]
    var = (
        phi ** (2 * steps) * res.filtered_state_cov[0, 0, last]
        + sigma2 * (1 - phi ** (2 * steps)) / (1 - phi**2)
        + obs_var
    )
    assert_allclose(pred.forecasts[0, -3:], mean, rtol=1e-10)
    assert_allclose(pred.forecasts_error_cov[0, 0, -3:], var, rtol=1e-10)
    if dynamic is not None:
        # The first two periods are one-step-ahead predictions
        assert_allclose(pred.forecasts[0, :2], res.forecasts[0, 10:12], rtol=1e-10)


def _filtered_timing_model(endog, matrices, initial_state, initial_state_cov,
                           filtered_timing, **options):
    k_states, k_posdef = matrices["selection"].shape[:2]
    mod = MLEModel(endog, k_states=k_states, k_posdef=k_posdef, **options)
    for name, value in matrices.items():
        mod.ssm[name] = value
    mod.ssm.initialize_known(initial_state, initial_state_cov)
    mod.ssm.timing_init_filtered = filtered_timing
    return mod.ssm


def _period(matrix, t):
    # Period t of a system matrix that may be time-invariant
    return matrix[..., t if matrix.shape[-1] > 1 else 0]


@pytest.mark.parametrize(
    "time_varying, case",
    [
        ("all", "conventional"),
        ("all", "univariate"),
        ("all", "collapsed"),
        ("all", "missing"),
        ("all", "conserve_memory"),
        ("transition", "conventional"),
        ("state_intercept", "conventional"),
        ("state_cov", "conventional"),
        ("selection", "conventional"),
        ("none", "conventional"),
    ],
)
def test_filtered_timing_time_varying(time_varying, case):
    # With the filtered timing (timing_init_filtered), period t >= 1 is
    # predicted from the filtered state of period t - 1 with the transition
    # matrix, state intercept and selected state covariance matrix of period
    # t - 1 (column t - 1 is the transition from t - 1 to t, as with the
    # predicted timing, in the smoother, in prediction and in the models), and
    # period 0 from the initialization with those of period 0. The filter used
    # those of period t, so the filter (and the smoother, which uses period t's
    # for the transition from t to t + 1) differed from the model with the
    # predicted timing whose initial state is the initialization pushed
    # through period 0's transition. Time-invariant models are unchanged.
    rng = np.random.default_rng(4471)
    nobs, k_endog, k_states, k_posdef = 12, 3, 2, 2

    def cov(k):
        root = rng.standard_normal((k, k, nobs))
        return np.einsum("ijt,kjt->ikt", root, root) + 0.5 * np.eye(k)[..., None]

    matrices = {
        "design": rng.standard_normal((k_endog, k_states, nobs)),
        "obs_intercept": rng.standard_normal((k_endog, nobs)),
        "obs_cov": cov(k_endog),
        "transition": 0.6 * rng.standard_normal((k_states, k_states, nobs)),
        "state_intercept": rng.standard_normal((k_states, nobs)),
        "selection": rng.standard_normal((k_states, k_posdef, nobs)),
        "state_cov": cov(k_posdef),
    }
    if time_varying != "all":
        for name in matrices:
            if name != time_varying:
                matrices[name] = matrices[name][..., :1]
    if case == "univariate":
        # Without the diagonalization (and its OutputWarning) in smoothing
        matrices["obs_cov"] = matrices["obs_cov"] * np.eye(k_endog)[..., None]
    elif case == "collapsed":
        # Not supported by the collapsed filter
        matrices["obs_intercept"] = np.zeros((k_endog, 1))
    endog = rng.standard_normal((nobs, k_endog))
    if case == "missing":
        endog[3, 1] = np.nan
        endog[6] = np.nan
    options = {
        "univariate": {"filter_univariate": True},
        "collapsed": {"filter_collapsed": True},
        "conserve_memory": {"memory_conserve": True},
    }.get(case, {})
    initial_state = rng.standard_normal(k_states)
    root = rng.standard_normal((k_states, k_states))
    initial_state_cov = root @ root.T + np.eye(k_states)

    # The equivalent model with the predicted timing
    transition = _period(matrices["transition"], 0)
    selection = _period(matrices["selection"], 0)
    selected_state_cov = selection @ _period(matrices["state_cov"], 0) @ selection.T
    desired = _filtered_timing_model(
        endog,
        matrices,
        transition @ initial_state + _period(matrices["state_intercept"], 0),
        transition @ initial_state_cov @ transition.T + selected_state_cov,
        False,
        **options,
    )
    actual = _filtered_timing_model(
        endog, matrices, initial_state, initial_state_cov, True, **options
    )
    if case == "conserve_memory":
        assert_allclose(actual.filter().llf, desired.filter().llf, rtol=1e-10)
        return
    res = actual.smooth()
    desired = desired.smooth()
    names = [
        "llf_obs",
        "forecasts",
        "forecasts_error_cov",
        "predicted_state",
        "predicted_state_cov",
        "filtered_state",
        "filtered_state_cov",
        "kalman_gain",
        "smoothed_state",
        "smoothed_state_cov",
        "smoothed_state_disturbance",
        "smoothed_state_disturbance_cov",
        "smoothed_measurement_disturbance",
    ]
    for name in names:
        assert_allclose(
            getattr(res, name), getattr(desired, name), rtol=1e-10, err_msg=name
        )

    # The prediction step, from the initialization and the filtered states
    for t in range(nobs + 1):
        prev = max(t - 1, 0)
        transition = _period(matrices["transition"], prev)
        selection = _period(matrices["selection"], prev)
        if t == 0:
            state, state_cov = initial_state, initial_state_cov
        else:
            state = res.filtered_state[:, t - 1]
            state_cov = res.filtered_state_cov[:, :, t - 1]
        mean = transition @ state + _period(matrices["state_intercept"], prev)
        cov = (
            transition @ state_cov @ transition.T
            + selection @ _period(matrices["state_cov"], prev) @ selection.T
        )
        assert_allclose(res.predicted_state[:, t], mean, rtol=1e-10)
        assert_allclose(res.predicted_state_cov[:, :, t], cov, rtol=1e-10)


def test_filtered_timing_time_varying_smoother():
    # A time-varying AR(1) observed with noise with the filtered timing: the
    # filtered states equalled those of the model with the predicted timing
    # and the matrices shifted by one period, but the smoothed states did not
    # (up to 0.146 apart here). Filter and smoother agree with the model with
    # the predicted timing, the same matrices and the initialization pushed
    # through period 0's transition.
    rng = np.random.default_rng(0)
    nobs = 8
    endog = rng.standard_normal((nobs, 1))
    intercept = rng.standard_normal(nobs + 1)[:nobs]
    transition = (0.2 + 0.7 * rng.uniform(size=nobs + 1))[:nobs]
    matrices = {
        "design": [[1.0]],
        "obs_cov": [[0.5]],
        "transition": transition[None, None, :],
        "state_intercept": intercept[None, :],
        "selection": np.ones((1, 1, 1)),
        "state_cov": [[1.0]],
    }
    res = _filtered_timing_model(endog, matrices, [0.3], [[2.0]], True).smooth()
    desired = _filtered_timing_model(
        endog,
        matrices,
        [transition[0] * 0.3 + intercept[0]],
        [[transition[0] ** 2 * 2.0 + 1.0]],
        False,
    ).smooth()
    assert_allclose(res.llf, desired.llf, rtol=1e-10)
    for name in ["filtered_state", "smoothed_state", "smoothed_state_cov"]:
        assert_allclose(
            getattr(res, name), getattr(desired, name), rtol=1e-10, err_msg=name
        )
    # a_t = T_{t-1} a_{t-1|t-1} + c_{t-1} for t >= 1
    assert_allclose(
        res.predicted_state[0, 1:],
        transition * res.filtered_state[0] + intercept,
        rtol=1e-10,
    )
