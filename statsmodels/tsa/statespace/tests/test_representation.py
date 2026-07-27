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
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pandas as pd
import pytest

from statsmodels.tsa.statespace import sarimax, tools
from statsmodels.tsa.statespace.kalman_filter import (
    FilterResults,
    KalmanFilter,
    PredictionResults,
)
from statsmodels.tsa.statespace.representation import Representation
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
        data = pd.DataFrame(
            cls.true["data"],
            index=pd.date_range("1947-01-01", "1995-07-01", freq="QS"),
            columns=["GDP"],
        )
        data["lgdp"] = np.log(data["GDP"])
        k_states = 4
        cls.model = KalmanFilter(k_endog=1, k_states=k_states, **kwargs)
        cls.model.bind(data["lgdp"].values)
        cls.model.design[:, :, 0] = [1, 1, 0, 0]
        cls.model.transition[
            [0, 0, 1, 1, 2, 3], [0, 3, 1, 2, 1, 3], [0, 0, 0, 0, 0, 0]
        ] = [1, 1, 0, 0, 1, 1]
        cls.model.selection = np.eye(cls.model.k_states)
        sigma_v, sigma_e, sigma_w, phi_1, phi_2 = np.array(cls.true["parameters"])
        cls.model.transition[[1, 1], [1, 2], [0, 0]] = [phi_1, phi_2]
        cls.model.state_cov[
            np.diag_indices(k_states) + (np.zeros(k_states, dtype=int),)
        ] = [sigma_v**2, sigma_e**2, 0, sigma_w**2]
        initial_state = np.zeros((k_states,))
        initial_state_cov = np.eye(k_states) * 100
        initial_state_cov = np.dot(
            np.dot(cls.model.transition[:, :, 0], initial_state_cov),
            cls.model.transition[:, :, 0].T,
        )
        cls.model.initialize_known(initial_state, initial_state_cov)

    @classmethod
    def run_filter(cls):
        return cls.model.filter()

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


class TestClark1987Single(Clark1987):
    """
    Basic single precision test for the loglikelihood and filtered states.
    """

    @classmethod
    def setup_class(cls):
        pytest.skip("Not implemented")
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


@pytest.mark.skip("Not implemented")
class TestClark1987SingleComplex(Clark1987):
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
        super().setup_class(dtype=float, conserve_memory=1 | 2)
        cls.results = cls.run_filter()


class Clark1987Forecast(Clark1987):
    """
    Forecasting test for the loglikelihood and filtered states.
    """

    @classmethod
    def setup_class(cls, dtype=float, nforecast=100, conserve_memory=0):
        super().setup_class(dtype=dtype, conserve_memory=conserve_memory)
        cls.nforecast = nforecast
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
        super().setup_class(dtype=float, conserve_memory=1 | 2)
        cls.results = cls.run_filter()


class TestClark1987ConserveAll(Clark1987):
    """
    Memory conservation forecasting test for the loglikelihood and filtered
    states.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class(dtype=float, conserve_memory=1 | 2 | 4 | 8)
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
        cls.model.design[:, :, 0] = [[1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]]
        cls.model.transition[
            [0, 0, 1, 1, 2, 3, 4, 5], [0, 4, 1, 2, 1, 2, 4, 5], [0, 0, 0, 0, 0, 0, 0, 0]
        ] = [1, 1, 0, 0, 1, 1, 1, 1]
        cls.model.selection = np.eye(cls.model.k_states)
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
        ) = np.array(cls.true["parameters"])
        cls.model.design[[1, 1, 1], [1, 2, 3], [0, 0, 0]] = [alpha_1, alpha_2, alpha_3]
        cls.model.transition[[1, 1], [1, 2], [0, 0]] = [phi_1, phi_2]
        cls.model.obs_cov[1, 1, 0] = sigma_ec**2
        cls.model.state_cov[
            np.diag_indices(k_states) + (np.zeros(k_states, dtype=int),)
        ] = [sigma_v**2, sigma_e**2, 0, 0, sigma_w**2, sigma_vl**2]
        initial_state = np.zeros((k_states,))
        initial_state_cov = np.eye(k_states) * 100
        initial_state_cov = np.dot(
            np.dot(cls.model.transition[:, :, 0], initial_state_cov),
            cls.model.transition[:, :, 0].T,
        )
        cls.model.initialize_known(initial_state, initial_state_cov)

    @classmethod
    def run_filter(cls):
        return cls.model.filter()

    def test_loglike(self):
        assert_almost_equal(self.results.llf_obs[0:].sum(), self.true["loglike"], 2)

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
            atol=0.0001,
        )


class TestClark1989Conserve(Clark1989):
    """
    Memory conservation test for the loglikelihood and filtered states with
    two-dimensional observation vector.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class(dtype=float, conserve_memory=1 | 2)
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
        super().setup_class(dtype=float, conserve_memory=1 | 2)
        cls.results = cls.run_filter()


class TestClark1989ConserveAll(Clark1989):
    """
    Memory conservation forecasting test for the loglikelihood and filtered
    states.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class(dtype=float, conserve_memory=1 | 2 | 4 | 8)
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
        pass

    def test_predicted_state(self):
        assert_allclose(
            self.results.predicted_state.T[1:],
            clark1989_results.iloc[:, 1:],
            atol=1e-08,
        )


def test_slice_notation():
    endog = np.arange(10) * 1.0
    mod = KalmanFilter(k_endog=1, k_states=2)
    mod.bind(endog)

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
    with pytest.raises(IndexError):
        mod["designs"]
    with pytest.raises(IndexError):
        mod["designs", 0, 0, 0]
    with pytest.raises(IndexError):
        mod[0]
    assert_equal(mod.design[0, 0, 0], 0)
    mod["design", 0, 0, 0] = 1
    assert_equal(mod["design"].sum(), 1)
    assert_equal(mod.design[0, 0, 0], 1)
    assert_equal(mod["design", 0, 0, 0], 1)
    mod["design"] = np.zeros(mod["design"].shape)
    assert_equal(mod.design[0, 0], 0)
    mod["design", 0, 0] = 1
    assert_equal(mod.design[0, 0], 1)
    assert_equal(mod["design", 0, 0], 1)


def test_representation():

    def zero_kstates():
        Representation(1, 0)

    with pytest.raises(ValueError):
        zero_kstates()

    def empty_endog():
        endog = np.zeros((0, 0))
        Representation(endog, k_states=2)

    with pytest.raises(ValueError):
        empty_endog()
    nobs = 10
    k_endog = 2
    arr = np.arange(nobs * k_endog).reshape(k_endog, nobs) * 1.0
    endog = np.asfortranarray(arr)
    mod = Representation(endog, k_states=2)
    assert_equal(mod.nobs, nobs)
    assert_equal(mod.k_endog, k_endog)
    nobs = 10
    k_endog = 2
    endog = np.arange(nobs * k_endog).reshape(nobs, k_endog) * 1.0
    mod = Representation(endog, k_states=2)
    assert_equal(mod.nobs, nobs)
    assert_equal(mod.k_endog, k_endog)
    assert_equal(mod._statespace, None)
    mod._initialize_representation()
    assert_equal(mod._statespace is not None, True)


def test_bind():
    mod = Representation(2, k_states=2)
    with pytest.raises(ValueError):
        mod.bind([1, 2, 3, 4])
    mod.bind(np.arange(10).reshape((5, 2)) * 1.0)
    assert_equal(mod.nobs, 5)
    mod.bind(np.zeros((0, 2), dtype=np.float64))
    with pytest.raises(ValueError):
        mod.bind(np.arange(12).reshape(2, 2, 3) * 1.0)
    mod.bind(np.asfortranarray(np.arange(10).reshape(2, 5)))
    assert_equal(mod.nobs, 5)
    mod.bind(np.arange(10).reshape(5, 2))
    assert_equal(mod.nobs, 5)
    with pytest.raises(ValueError):
        mod.bind(np.asfortranarray(np.arange(10).reshape(5, 2)))
    with pytest.raises(ValueError):
        mod.bind(np.arange(10).reshape(2, 5))


def test_initialization():
    mod = Representation(1, k_states=2)
    with pytest.raises(RuntimeError):
        mod._initialize_state()
    initial_state = np.zeros(2) + 1.5
    initial_state_cov = np.eye(2) * 3.0
    mod.initialize_known(initial_state, initial_state_cov)
    assert_equal(mod.initialization.constant.sum(), 3)
    assert_equal(mod.initialization.stationary_cov.diagonal().sum(), 6)
    initial_state = np.zeros(10)
    with pytest.raises(ValueError):
        mod.initialize_known(initial_state, initial_state_cov)
    initial_state = np.zeros((10, 10))
    with pytest.raises(ValueError):
        mod.initialize_known(initial_state, initial_state_cov)
    initial_state = np.zeros(2) + 1.5
    initial_state_cov = np.eye(3)
    with pytest.raises(ValueError):
        mod.initialize_known(initial_state, initial_state_cov)


def test_init_matrices_time_invariant():
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
    mod = KalmanFilter(k_endog=1, k_states=1)
    with pytest.raises(RuntimeError):
        mod._initialize_filter()
    mod.initialize_approximate_diffuse()
    with pytest.raises(RuntimeError):
        mod.filter()


def test_cython():
    for prefix, dtype in tools.prefix_dtype_map.items():
        endog = np.array(1.0, ndmin=2, dtype=dtype)
        mod = KalmanFilter(k_endog=1, k_states=1, dtype=dtype)
        mod.bind(endog)
        mod._initialize_filter()
        assert_equal(mod.prefix, prefix)
        assert_equal(mod.dtype, dtype)
        assert_equal(prefix in mod._kalman_filters, True)
        kf = mod._kalman_filters[prefix]
        assert isinstance(kf, tools.prefix_kalman_filter_map[prefix])
        assert_equal(mod._kalman_filter, kf)
    mod = KalmanFilter(k_endog=1, k_states=1)
    assert_equal(mod.prefix, "d")
    assert_equal(mod.dtype, np.float64)
    assert_equal(mod._kalman_filter, None)
    endog = np.ascontiguousarray(np.array([1.0, 2.0], dtype=np.float64))
    mod.bind(endog)
    mod._initialize_filter()
    kf = mod._kalman_filters["d"]
    mod.bind(endog)
    mod._initialize_filter()
    assert_equal(mod._kalman_filter, kf)
    mod.design = np.zeros((1, 1, 2))
    mod._initialize_filter()
    assert_equal(mod._kalman_filter == kf, False)
    kf = mod._kalman_filters["d"]
    endog = np.ascontiguousarray(np.array([1.0, 2.0], dtype=np.complex128))
    mod.bind(endog)
    assert_equal(mod._kalman_filter == kf, False)


def test_filter():
    endog = np.ones((10, 1))
    mod = KalmanFilter(endog, k_states=1, initialization="approximate_diffuse")
    mod["design", :] = 1
    mod["selection", :] = 1
    mod["state_cov", :] = 1
    res = mod.filter()
    assert_equal(isinstance(res, FilterResults), True)


def test_loglike():
    endog = np.ones((10, 1))
    mod = KalmanFilter(endog, k_states=1, initialization="approximate_diffuse")
    mod["design", :] = 1
    mod["selection", :] = 1
    mod["state_cov", :] = 1
    mod.memory_no_likelihood = True
    with pytest.raises(RuntimeError):
        mod.loglikeobs()


def test_predict():
    warnings.simplefilter("always")
    endog = np.ones((10, 1))
    mod = KalmanFilter(endog, k_states=1, initialization="approximate_diffuse")
    mod["design", :] = 1
    mod["obs_intercept"] = np.zeros((1, 10))
    mod["selection", :] = 1
    mod["state_cov", :] = 1
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
    res = mod.filter()
    with pytest.raises(ValueError):
        res.predict(start=-1)
    with pytest.raises(ValueError):
        res.predict(start=2, end=1)
    with pytest.raises(ValueError):
        res.predict(dynamic=-1)
    with warnings.catch_warnings(record=True) as w:
        res.predict(end=1, dynamic=2)
        message = "Dynamic prediction specified to begin after the end of prediction, and so has no effect."
        assert_equal(str(w[0].message), message)
    with warnings.catch_warnings(record=True) as w:
        res.predict(end=11, dynamic=11, obs_intercept=np.zeros((1, 1)))
        message = "Dynamic prediction specified to begin during out-of-sample forecasting period, and so has no effect."
        assert_equal(str(w[0].message), message)
    with pytest.raises(ValueError):
        res.predict(end=res.nobs + 1, design=True, obs_intercept=np.zeros((1, 1)))
    with pytest.raises(ValueError):
        res.predict(end=res.nobs + 1)
    with pytest.raises(ValueError):
        res.predict(end=res.nobs + 1, obs_intercept=np.zeros(2))
    assert_equal(res.predict().forecasts.shape, (1, res.nobs))
    res.predict(dynamic=True)
    prediction_results = res.predict(start=3, end=5)
    assert_equal(isinstance(prediction_results, PredictionResults), True)
    assert_equal(prediction_results.endog.shape, (1, 2))
    assert_equal(prediction_results.obs_intercept.shape, (1, 2))
    assert_equal(prediction_results.design.shape, (1, 1))
    assert_equal(prediction_results.obs_cov.shape, (1, 1))
    assert_equal(prediction_results.state_intercept.shape, (1,))
    assert_equal(prediction_results.obs_intercept.shape, (1, 2))
    assert_equal(prediction_results.transition.shape, (1, 1))
    assert_equal(prediction_results.selection.shape, (1, 1))
    assert_equal(prediction_results.state_cov.shape, (1, 1))
    assert_equal(prediction_results.forecasts.shape, (1, 2))
    assert_equal(prediction_results.forecasts_error.shape, (1, 2))
    assert_equal(prediction_results.filtered_state.shape, (1, 2))
    assert_equal(prediction_results.predicted_state.shape, (1, 2))
    assert_equal(prediction_results.forecasts_error_cov.shape, (1, 1, 2))
    assert_equal(prediction_results.filtered_state_cov.shape, (1, 1, 2))
    assert_equal(prediction_results.predicted_state_cov.shape, (1, 1, 2))
    with pytest.raises(AttributeError):
        _ = prediction_results.test
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
    true = results_kalman_filter.uc_uni
    data = pd.DataFrame(
        true["data"],
        index=pd.date_range("1947-01-01", "1995-07-01", freq="QS"),
        columns=["GDP"],
    )
    data["lgdp"] = np.log(data["GDP"])
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
    from scipy.signal import lfilter

    nsimulations = 10
    sigma2 = 2
    measurement_shocks = np.zeros(nsimulations)
    rs = np.random.RandomState(9991617)
    state_shocks = rs.normal(scale=sigma2**0.5, size=nsimulations)
    mod = SimulationSmoother(np.r_[0], k_states=1, initialization="diffuse")
    mod["design", 0, 0] = 1.0
    mod["transition", 0, 0] = 1.0
    mod["selection", 0, 0] = 1.0
    actual = mod.simulate(
        nsimulations, measurement_shocks=measurement_shocks, state_shocks=state_shocks
    )[0].squeeze()
    desired = np.r_[0, np.cumsum(state_shocks)[:-1]]
    assert_allclose(actual, desired)
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
    mod = SimulationSmoother(
        np.zeros((1, 10)), k_states=1, nobs=10, initialization="diffuse"
    )
    mod["obs_intercept"] = (np.arange(10) * 1.0).reshape(1, 10)
    mod["design", 0, 0] = 1.0
    mod["transition", 0, 0] = 1.0
    mod["selection", 0, 0] = 1.0
    with pytest.raises(ValueError):
        mod.simulate(nsimulations + 1, measurement_shocks, state_shocks)
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
    mod = SimulationSmoother(k_endog=1, k_states=1, initialization="diffuse")
    mod["design", 0, 0] = 1.0
    mod["transition", 0, 0] = 1.0
    mod["selection", 0, 0] = 1.0
    mod["state_cov", 0, 0] = 2.0
    actual = mod.impulse_responses(steps=10)
    desired = np.ones((11, 1))
    assert_allclose(actual, desired)
    mod = SimulationSmoother(k_endog=1, k_states=1, initialization="diffuse")
    mod["design", 0, 0] = 1.0
    mod["transition", 0, 0] = 1.0
    mod["selection", 0, 0] = 1.0
    mod["state_cov", 0, 0] = 2.0
    actual = mod.impulse_responses(steps=10, impulse=[2])
    desired = np.ones((11, 1)) * 2
    assert_allclose(actual, desired)
    mod = SimulationSmoother(k_endog=1, k_states=1, initialization="diffuse")
    mod["design", 0, 0] = 1.0
    mod["transition", 0, 0] = 1.0
    mod["selection", 0, 0] = 1.0
    mod["state_cov", 0, 0] = 2.0
    actual = mod.impulse_responses(steps=10, orthogonalized=True)
    desired = np.ones((11, 1)) * 2**0.5
    assert_allclose(actual, desired)
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
    mod = SimulationSmoother(k_endog=1, k_states=1, initialization="diffuse")
    with pytest.raises(ValueError):
        mod.impulse_responses(impulse=1)
    with pytest.raises(ValueError):
        mod.impulse_responses(impulse=[1, 1])
    with pytest.raises(ValueError):
        mod.impulse_responses(impulse=[])
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
    actual = mod.impulse_responses(steps=10, impulse=0, orthogonalized=True)
    assert_allclose(actual, desired)
    actual = mod.impulse_responses(steps=10, impulse=[1, 0], orthogonalized=True)
    assert_allclose(actual, desired)
    actual = mod.impulse_responses(steps=10, impulse=[0, 1], orthogonalized=True)
    assert_allclose(actual, desired)
    mod = SimulationSmoother(k_endog=1, k_states=2, initialization="diffuse")
    mod["design", 0, 0:2] = 1.0
    mod["transition", :, :] = np.eye(2)
    mod["selection", :, :] = np.eye(2)
    mod["state_cov", :, :] = np.array([[1, 0.5], [0.5, 1.25]])
    desired = np.ones((11, 1))
    actual = mod.impulse_responses(steps=10, impulse=0)
    assert_allclose(actual, desired)
    actual = mod.impulse_responses(steps=10, impulse=1)
    assert_allclose(actual, desired)
    actual = mod.impulse_responses(steps=10, impulse=0, orthogonalized=True)
    assert_allclose(actual, desired + desired * 0.5)
    actual = mod.impulse_responses(steps=10, impulse=1, orthogonalized=True)
    assert_allclose(actual, desired)
    mod = SimulationSmoother(k_endog=2, k_states=2, initialization="diffuse")
    mod["design", :, :] = np.eye(2)
    mod["transition", :, :] = np.eye(2)
    mod["selection", :, :] = np.eye(2)
    mod["state_cov", :, :] = np.array([[1, 0.5], [0.5, 1.25]])
    ones = np.ones((11, 1))
    zeros = np.zeros((11, 1))
    actual = mod.impulse_responses(steps=10, impulse=0)
    assert_allclose(actual, np.c_[ones, zeros])
    actual = mod.impulse_responses(steps=10, impulse=1)
    assert_allclose(actual, np.c_[zeros, ones])
    actual = mod.impulse_responses(steps=10, impulse=0, orthogonalized=True)
    assert_allclose(actual, np.c_[ones, ones * 0.5])
    actual = mod.impulse_responses(steps=10, impulse=1, orthogonalized=True)
    assert_allclose(actual, np.c_[zeros, ones])
    mod = sarimax.SARIMAX([0.1, 0.5, -0.2], order=(1, 0, 0))
    phi = 0.5
    mod.update([phi, 1])
    desired = np.cumprod(np.r_[1, [phi] * 10])
    actual = mod.ssm.impulse_responses(steps=10)
    assert_allclose(actual[:, 0], desired)
    res = mod.filter([phi, 1.0])
    actual = res.impulse_responses(steps=10)
    assert_allclose(actual, desired)


def test_missing():
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
    mod = KalmanFilter(endog_pre_na, k_states=1, initialization="approximate_diffuse")
    mod["design", :, :] = 1
    mod["obs_cov", :, :] = np.eye(mod.k_endog) * 0.5
    mod["transition", :, :] = 0.5
    mod["selection", :, :] = 1
    mod["state_cov", :, :] = 0.5
    llf_pre_na = mod.loglikeobs()
    assert_allclose(llf_pre_na, llf)
    mod = KalmanFilter(endog_post_na, k_states=1, initialization="approximate_diffuse")
    mod["design", :, :] = 1
    mod["obs_cov", :, :] = np.eye(mod.k_endog) * 0.5
    mod["transition", :, :] = 0.5
    mod["selection", :, :] = 1
    mod["state_cov", :, :] = 0.5
    llf_post_na = mod.loglikeobs()
    assert_allclose(llf_post_na, llf)
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
