"""
Tests for univariate treatment of multivariate models

A non-diagonal observation covariance is first diagonalized as
H_t = C_t D_t C_t', with the univariate recursions then run on C_t^{-1} y_t
(Durbin and Koopman, 2012, section 6.4.3), so the smoothed measurement
disturbance and its covariance are computed for C_t^{-1} \\varepsilon_t. The
mean is premultiplied by C_t again on output, which makes it comparable to the
conventional smoother. The covariance matrix is not, because undoing the
transformation needs the off-diagonal elements of Var(C_t^{-1} \\varepsilon_t)
and the univariate recursions only produce its diagonal; those comparisons are
skipped for the non-diagonal cases.

Author: Chad Fulton
License: Simplified-BSD
"""
import contextlib
from pathlib import Path
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest

from statsmodels import datasets
from statsmodels.tools.sm_exceptions import OutputWarning
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.tests.results import results_kalman_filter

current_path = Path(__file__).resolve().parent

NONDIAGONAL_OBS_COV_SKIP = (
    "the univariate recursions only produce the diagonal of"
    " Var(C_t^{-1} eps_t), which is not enough to undo the diagonalization"
)
PSD_OBS_COV_MSG = "Positive semi-definite observation covariance matrix"


def simulation_smoother(model, seed=1234):
    """Simulation smoother driven by fixed variates, shared across methods."""
    rs = np.random.RandomState(seed)
    sim = model.simulation_smoother()
    sim.simulate(
        measurement_disturbance_variates=rs.normal(
            size=(model.nobs, model.k_endog)),
        state_disturbance_variates=rs.normal(
            size=(model.nobs, model.k_posdef)),
        initial_state_variates=np.zeros(model.k_states),
    )
    return sim


class TestClark1989:
    """
    Clark's (1989) bivariate unobserved components model of real GDP (as
    presented in Kim and Nelson, 1999)

    Tests two-dimensional observation data.

    Test data produced using GAUSS code described in Kim and Nelson (1999) and
    found at http://econ.korea.ac.kr/~cjkim/SSMARKOV.htm

    See `results.results_kalman_filter` for more information.
    """
    @classmethod
    def setup_class(cls, dtype=float, alternate_timing=False, **kwargs):

        cls.true = results_kalman_filter.uc_bi
        cls.true_states = pd.DataFrame(cls.true["states"])

        # GDP and Unemployment, Quarterly, 1948.1 - 1995.3
        data = pd.DataFrame(
            cls.true["data"],
            index=pd.date_range("1947-01-01", "1995-07-01", freq="QS"),
            columns=["GDP", "UNEMP"]
        )[4:]
        data["GDP"] = np.log(data["GDP"])
        data["UNEMP"] = (data["UNEMP"]/100)

        k_states = 6
        cls.mlemodel = MLEModel(data, k_states=k_states, **kwargs)
        cls.model = cls.mlemodel.ssm

        # Statespace representation
        cls.model.design[:, :, 0] = [[1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]]
        cls.model.transition[
            ([0, 0, 1, 1, 2, 3, 4, 5],
             [0, 4, 1, 2, 1, 2, 4, 5],
             [0, 0, 0, 0, 0, 0, 0, 0])
        ] = [1, 1, 0, 0, 1, 1, 1, 1]
        cls.model.selection = np.eye(cls.model.k_states)

        # Update matrices with given parameters
        (sigma_v, sigma_e, sigma_w, sigma_vl, sigma_ec,
         phi_1, phi_2, alpha_1, alpha_2, alpha_3) = np.array(
            cls.true["parameters"],
        )
        cls.model.design[([1, 1, 1], [1, 2, 3], [0, 0, 0])] = [
            alpha_1, alpha_2, alpha_3
        ]
        cls.model.transition[([1, 1], [1, 2], [0, 0])] = [phi_1, phi_2]
        cls.model.obs_cov[1, 1, 0] = sigma_ec**2
        cls.model.state_cov[
            np.diag_indices(k_states)+(np.zeros(k_states, dtype=int),)] = [
            sigma_v**2, sigma_e**2, 0, 0, sigma_w**2, sigma_vl**2
        ]

        # Initialization
        initial_state = np.zeros((k_states,))
        initial_state_cov = np.eye(k_states)*100

        # Initialization: cls.modification
        if not alternate_timing:
            initial_state_cov = np.dot(
                np.dot(cls.model.transition[:, :, 0], initial_state_cov),
                cls.model.transition[:, :, 0].T
            )
        else:
            cls.model.timing_init_filtered = True
        cls.model.initialize_known(initial_state, initial_state_cov)

        # Conventional filtering, smoothing, and simulation smoothing
        cls.model.filter_conventional = True
        cls.conventional_results = cls.model.smooth()
        cls.conventional_sim = simulation_smoother(cls.model)

        # Univariate filtering, smoothing, and simulation smoothing
        cls.model.filter_univariate = True
        cls.univariate_results = cls.model.smooth()
        cls.univariate_sim = simulation_smoother(cls.model)

    def test_using_univariate(self):
        # Regression test to make sure the univariate_results actually
        # used the univariate Kalman filtering approach (i.e., that the flag
        # being set actually caused the filter to not use the conventional
        # filter)
        assert not self.conventional_results.filter_univariate
        assert self.univariate_results.filter_univariate

        assert_allclose(
            self.conventional_results.forecasts_error_cov[1, 1, 0],
            143.03724478030821
        )
        assert_allclose(
            self.univariate_results.forecasts_error_cov[1, 1, 0],
            120.66208525029386
        )

    def test_forecasts(self):
        assert_almost_equal(
            self.conventional_results.forecasts[0, :],
            self.univariate_results.forecasts[0, :], 9
        )

    def test_forecasts_error(self):
        assert_almost_equal(
            self.conventional_results.forecasts_error[0, :],
            self.univariate_results.forecasts_error[0, :], 9
        )

    def test_forecasts_error_cov(self):
        assert_almost_equal(
            self.conventional_results.forecasts_error_cov[0, 0, :],
            self.univariate_results.forecasts_error_cov[0, 0, :], 9
        )

    def test_filtered_state(self):
        assert_almost_equal(
            self.conventional_results.filtered_state,
            self.univariate_results.filtered_state, 8
        )

    def test_filtered_state_cov(self):
        assert_almost_equal(
            self.conventional_results.filtered_state_cov,
            self.univariate_results.filtered_state_cov, 9
        )

    def test_predicted_state(self):
        assert_almost_equal(
            self.conventional_results.predicted_state,
            self.univariate_results.predicted_state, 8
        )

    def test_predicted_state_cov(self):
        assert_almost_equal(
            self.conventional_results.predicted_state_cov,
            self.univariate_results.predicted_state_cov, 9
        )

    def test_loglike(self):
        assert_allclose(
            self.conventional_results.llf_obs,
            self.univariate_results.llf_obs
        )

    def test_smoothed_states(self):
        assert_almost_equal(
            self.conventional_results.smoothed_state,
            self.univariate_results.smoothed_state, 7
        )

    def test_smoothed_states_cov(self):
        assert_almost_equal(
            self.conventional_results.smoothed_state_cov,
            self.univariate_results.smoothed_state_cov, 6
        )

    def test_smoothed_measurement_disturbance(self):
        assert_almost_equal(
            self.conventional_results.smoothed_measurement_disturbance,
            self.univariate_results.smoothed_measurement_disturbance, 9
        )

    def test_smoothed_measurement_disturbance_cov(self):
        conv = self.conventional_results
        univ = self.univariate_results
        assert_almost_equal(
            conv.smoothed_measurement_disturbance_cov.diagonal(),
            univ.smoothed_measurement_disturbance_cov.diagonal(), 9
        )

    def test_smoothed_state_disturbance(self):
        assert_allclose(
            self.conventional_results.smoothed_state_disturbance,
            self.univariate_results.smoothed_state_disturbance,
            atol=1e-7
        )

    def test_smoothed_state_disturbance_cov(self):
        assert_almost_equal(
            self.conventional_results.smoothed_state_disturbance_cov,
            self.univariate_results.smoothed_state_disturbance_cov, 9
        )

    def test_simulation_smoothed_state(self):
        assert_allclose(
            self.conventional_sim.simulated_state,
            self.univariate_sim.simulated_state, atol=1e-7
        )

    def test_simulation_smoothed_measurement_disturbance(self):
        assert_allclose(
            self.conventional_sim.simulated_measurement_disturbance,
            self.univariate_sim.simulated_measurement_disturbance, atol=1e-7
        )

    def test_simulation_smoothed_state_disturbance(self):
        assert_allclose(
            self.conventional_sim.simulated_state_disturbance,
            self.univariate_sim.simulated_state_disturbance, atol=1e-7
        )


class TestClark1989Alternate(TestClark1989):
    @classmethod
    def setup_class(cls, *args, **kwargs):
        super().setup_class(*args, alternate_timing=True, **kwargs)

    def test_using_alterate(self):
        assert self.model._kalman_filter.filter_timing == 1


class MultivariateMissingGeneralObsCov:
    @classmethod
    def setup_class(cls, which, dtype=float, alternate_timing=False, **kwargs):
        # Results
        path = Path(current_path).joinpath("results", "results_smoothing_generalobscov_R.csv")
        cls.desired = pd.read_csv(path)

        # Data
        dta = datasets.macrodata.load_pandas().data
        dta.index = pd.date_range(start="1959-01-01",
                                  end="2009-7-01", freq="QS")
        obs = dta[["realgdp", "realcons", "realinv"]].diff().iloc[1:]

        if which == "all":
            obs.iloc[:50, :] = np.nan
            obs.iloc[119:130, :] = np.nan
        elif which == "partial":
            obs.iloc[0:50, 0] = np.nan
            obs.iloc[119:130, 0] = np.nan
        elif which == "mixed":
            obs.iloc[0:50, 0] = np.nan
            obs.iloc[19:70, 1] = np.nan
            obs.iloc[39:90, 2] = np.nan
            obs.iloc[119:130, 0] = np.nan
            obs.iloc[119:130, 2] = np.nan

        # Create the model
        mod = MLEModel(obs, k_states=3, k_posdef=3, **kwargs)
        mod["design"] = np.eye(3)
        X = (np.arange(9) + 1).reshape((3, 3)) / 10.
        mod["obs_cov"] = np.dot(X, X.T)
        mod["transition"] = np.eye(3)
        mod["selection"] = np.eye(3)
        mod["state_cov"] = np.eye(3)
        mod.initialize_approximate_diffuse(1e6)
        cls.model = mod.ssm

        # Conventional filtering, smoothing, and simulation smoothing
        cls.model.filter_conventional = True
        cls.conventional_results = cls.model.smooth()
        cls.conventional_sim = simulation_smoother(cls.model)

        # Univariate filtering, smoothing, and simulation smoothing
        cls.model.filter_univariate = True
        # The non-diagonal obs_cov is diagonalized by the univariate filter.
        # It has rank 2, so its LDL decomposition warns.
        msg = "The univariate filtering approach diagonalized"
        with pytest.warns(UserWarning, match=PSD_OBS_COV_MSG):
            with pytest.warns(OutputWarning, match=msg):
                cls.univariate_results = cls.model.smooth()
        with pytest.warns(UserWarning, match=PSD_OBS_COV_MSG):
            cls.univariate_sim = simulation_smoother(cls.model)

    def test_using_univariate(self):
        # Regression test to make sure the univariate_results actually
        # used the univariate Kalman filtering approach (i.e., that the flag
        # being set actually caused the filter to not use the conventional
        # filter)
        assert not self.conventional_results.filter_univariate
        assert self.univariate_results.filter_univariate

        assert_allclose(
            self.conventional_results.forecasts_error_cov[1, 1, 0],
            1000000.77
        )
        assert_allclose(
            self.univariate_results.forecasts_error_cov[1, 1, 0],
            1000000.77
        )

    def test_forecasts(self):
        assert_almost_equal(
            self.conventional_results.forecasts[0, :],
            self.univariate_results.forecasts[0, :], 9
        )

    def test_forecasts_error(self):
        assert_almost_equal(
            self.conventional_results.forecasts_error[0, :],
            self.univariate_results.forecasts_error[0, :], 9
        )

    def test_forecasts_error_cov(self):
        assert_almost_equal(
            self.conventional_results.forecasts_error_cov[0, 0, :],
            self.univariate_results.forecasts_error_cov[0, 0, :], 9
        )

    def test_filtered_state(self):
        assert_almost_equal(
            self.conventional_results.filtered_state,
            self.univariate_results.filtered_state, 8
        )

    def test_filtered_state_cov(self):
        assert_almost_equal(
            self.conventional_results.filtered_state_cov,
            self.univariate_results.filtered_state_cov, 9
        )

    def test_predicted_state(self):
        assert_almost_equal(
            self.conventional_results.predicted_state,
            self.univariate_results.predicted_state, 8
        )

    def test_predicted_state_cov(self):
        assert_almost_equal(
            self.conventional_results.predicted_state_cov,
            self.univariate_results.predicted_state_cov, 9
        )

    def test_loglike(self):
        assert_allclose(
            self.conventional_results.llf_obs,
            self.univariate_results.llf_obs
        )

    def test_smoothed_states(self):
        assert_almost_equal(
            self.conventional_results.smoothed_state,
            self.univariate_results.smoothed_state, 7
        )

    def test_smoothed_states_cov(self):
        assert_almost_equal(
            self.conventional_results.smoothed_state_cov,
            self.univariate_results.smoothed_state_cov, 6
        )

    def test_smoothed_measurement_disturbance(self):
        assert_allclose(
            self.conventional_results.smoothed_measurement_disturbance,
            self.univariate_results.smoothed_measurement_disturbance,
            atol=1e-7
        )

    @pytest.mark.skip(reason=NONDIAGONAL_OBS_COV_SKIP)
    def test_smoothed_measurement_disturbance_cov(self):
        conv = self.conventional_results
        univ = self.univariate_results
        assert_almost_equal(
            conv.smoothed_measurement_disturbance_cov.diagonal(),
            univ.smoothed_measurement_disturbance_cov.diagonal(), 9
        )

    def test_smoothed_state_disturbance(self):
        assert_allclose(
            self.conventional_results.smoothed_state_disturbance,
            self.univariate_results.smoothed_state_disturbance,
            atol=1e-7
        )

    def test_smoothed_state_disturbance_cov(self):
        assert_almost_equal(
            self.conventional_results.smoothed_state_disturbance_cov,
            self.univariate_results.smoothed_state_disturbance_cov, 9
        )

    def test_simulation_smoothed_state(self):
        assert_allclose(
            self.conventional_sim.simulated_state,
            self.univariate_sim.simulated_state, atol=1e-7
        )

    def test_simulation_smoothed_measurement_disturbance(self):
        assert_allclose(
            self.conventional_sim.simulated_measurement_disturbance,
            self.univariate_sim.simulated_measurement_disturbance, atol=1e-7
        )

    def test_simulation_smoothed_state_disturbance(self):
        assert_allclose(
            self.conventional_sim.simulated_state_disturbance,
            self.univariate_sim.simulated_state_disturbance, atol=1e-7
        )


class TestMultivariateGeneralObsCov(MultivariateMissingGeneralObsCov):
    """
    This class tests the univariate method when the observation covariance
    matrix is not diagonal and all data is available.

    Tests are against the conventional smoother.
    """
    @classmethod
    def setup_class(cls, *args, **kwargs):
        super().setup_class("none")


class TestMultivariateAllMissingGeneralObsCov(
        MultivariateMissingGeneralObsCov):
    """
    This class tests the univariate method when the observation covariance
    matrix is not diagonal and there are cases of fully missing data only.

    Tests are against the conventional smoother.
    """
    @classmethod
    def setup_class(cls, *args, **kwargs):
        super().setup_class("all")


class TestMultivariatePartialMissingGeneralObsCov(
        MultivariateMissingGeneralObsCov):
    """
    This class tests the univariate method when the observation covariance
    matrix is not diagonal and there are cases of partially missing data only.

    Tests are against the conventional smoother.
    """
    @classmethod
    def setup_class(cls, *args, **kwargs):
        super().setup_class("partial")

    def test_forecasts(self):
        assert_almost_equal(
            self.conventional_results.forecasts[0, :],
            self.univariate_results.forecasts[0, :], 8
        )

    def test_forecasts_error(self):
        assert_almost_equal(
            self.conventional_results.forecasts_error[0, :],
            self.univariate_results.forecasts_error[0, :], 8
        )


class TestMultivariateMixedMissingGeneralObsCov(
        MultivariateMissingGeneralObsCov):
    """
    This class tests the univariate method when the observation covariance
    matrix is not diagonal and there are cases of both partially missing and
    fully missing data.

    Tests are against the conventional smoother.
    """
    @classmethod
    def setup_class(cls, *args, **kwargs):
        super().setup_class("mixed")

    def test_forecasts(self):
        assert_almost_equal(
            self.conventional_results.forecasts[0, :],
            self.univariate_results.forecasts[0, :], 8
        )

    def test_forecasts_error(self):
        assert_almost_equal(
            self.conventional_results.forecasts_error[0, :],
            self.univariate_results.forecasts_error[0, :], 8
        )


class TestMultivariateVAR:
    @classmethod
    def setup_class(cls, which="none", **kwargs):
        # Results
        path = Path(current_path).joinpath("results", "results_smoothing_generalobscov_R.csv")
        cls.desired = pd.read_csv(path)

        # Data
        dta = datasets.macrodata.load_pandas().data
        dta.index = pd.date_range(start="1959-01-01",
                                  end="2009-7-01", freq="QS")
        obs = dta[["realgdp", "realcons", "realinv"]].diff().iloc[1:]

        if which == "all":
            obs.iloc[:50, :] = np.nan
            obs.iloc[119:130, :] = np.nan
        elif which == "partial":
            obs.iloc[0:50, 0] = np.nan
            obs.iloc[119:130, 0] = np.nan
        elif which == "mixed":
            obs.iloc[0:50, 0] = np.nan
            obs.iloc[19:70, 1] = np.nan
            obs.iloc[39:90, 2] = np.nan
            obs.iloc[119:130, 0] = np.nan
            obs.iloc[119:130, 2] = np.nan

        # Create the model
        mod = MLEModel(obs, k_states=3, k_posdef=3, **kwargs)
        mod["design"] = np.eye(3)
        mod["obs_cov"] = np.array([
            [609.0746647855,    0.,              0.],
            [0.,                1.8774916622,    0.],
            [0.,                0.,            124.6768281675]])
        mod["transition"] = np.array([
            [-0.8110473405,  1.8005304445,  1.0215975772],
            [-1.9846632699,  2.4091302213,  1.9264449765],
            [0.9181658823,  -0.2442384581, -0.6393462272]])
        mod["selection"] = np.eye(3)
        mod["state_cov"] = np.array([
            [1552.9758843938,   612.7185121905,   877.6157204992],
            [612.7185121905,    467.8739411204,    70.608037339],
            [877.6157204992,     70.608037339,    900.5440385836]])
        mod.initialize_approximate_diffuse(1e6)
        cls.model = mod.ssm

        # Conventional filtering, smoothing, and simulation smoothing
        cls.model.filter_conventional = True
        cls.conventional_results = cls.model.smooth()
        cls.conventional_sim = simulation_smoother(cls.model)

        # Univariate filtering, smoothing, and simulation smoothing
        cls.model.filter_univariate = True
        cls.univariate_results = cls.model.smooth()
        cls.univariate_sim = simulation_smoother(cls.model)

    def test_forecasts(self):
        assert_almost_equal(
            self.conventional_results.forecasts[0, :],
            self.univariate_results.forecasts[0, :], 9
        )

    def test_forecasts_error(self):
        assert_almost_equal(
            self.conventional_results.forecasts_error[0, :],
            self.univariate_results.forecasts_error[0, :], 9
        )

    def test_forecasts_error_cov(self):
        assert_almost_equal(
            self.conventional_results.forecasts_error_cov[0, 0, :],
            self.univariate_results.forecasts_error_cov[0, 0, :], 9
        )

    def test_filtered_state(self):
        assert_almost_equal(
            self.conventional_results.filtered_state,
            self.univariate_results.filtered_state, 8
        )

    def test_filtered_state_cov(self):
        assert_almost_equal(
            self.conventional_results.filtered_state_cov,
            self.univariate_results.filtered_state_cov, 9
        )

    def test_predicted_state(self):
        assert_almost_equal(
            self.conventional_results.predicted_state,
            self.univariate_results.predicted_state, 8
        )

    def test_predicted_state_cov(self):
        assert_almost_equal(
            self.conventional_results.predicted_state_cov,
            self.univariate_results.predicted_state_cov, 9
        )

    def test_loglike(self):
        assert_allclose(
            self.conventional_results.llf_obs,
            self.univariate_results.llf_obs
        )

    def test_smoothed_states(self):
        assert_allclose(
            self.conventional_results.smoothed_state,
            self.univariate_results.smoothed_state
        )

    def test_smoothed_states_cov(self):
        assert_allclose(
            self.conventional_results.smoothed_state_cov,
            self.univariate_results.smoothed_state_cov, atol=1e-9
        )

    def test_smoothed_measurement_disturbance(self):
        assert_almost_equal(
            self.conventional_results.smoothed_measurement_disturbance,
            self.univariate_results.smoothed_measurement_disturbance, 9
        )

    def test_smoothed_measurement_disturbance_cov(self):
        conv = self.conventional_results
        univ = self.univariate_results
        assert_almost_equal(
            conv.smoothed_measurement_disturbance_cov.diagonal(),
            univ.smoothed_measurement_disturbance_cov.diagonal(),
            9
        )

    def test_smoothed_state_disturbance(self):
        assert_allclose(
            self.conventional_results.smoothed_state_disturbance,
            self.univariate_results.smoothed_state_disturbance,
            atol=1e-7
        )

    def test_smoothed_state_disturbance_cov(self):
        assert_almost_equal(
            self.conventional_results.smoothed_state_disturbance_cov,
            self.univariate_results.smoothed_state_disturbance_cov, 9
        )

    def test_simulation_smoothed_state(self):
        assert_allclose(
            self.conventional_sim.simulated_state,
            self.univariate_sim.simulated_state, atol=1e-7
        )

    def test_simulation_smoothed_measurement_disturbance(self):
        assert_allclose(
            self.conventional_sim.simulated_measurement_disturbance,
            self.univariate_sim.simulated_measurement_disturbance, atol=1e-7
        )

    def test_simulation_smoothed_state_disturbance(self):
        assert_allclose(
            self.conventional_sim.simulated_state_disturbance,
            self.univariate_sim.simulated_state_disturbance, atol=1e-7
        )


def test_time_varying_transition():
    # Test for correct univariate filtering/smoothing when we have a
    # time-varying transition matrix
    endog = np.array([10, 5, 2.5, 1.25, 2.5, 5, 10])
    transition = np.ones((1, 1, 7))
    transition[..., :5] = 0.5
    transition[..., 5:] = 2

    # Conventional filter / smoother
    mod1 = SARIMAX(endog, order=(1, 0, 0), measurement_error=True)
    mod1.update([2., 1., 1.])
    mod1.ssm["transition"] = transition
    res1 = mod1.ssm.smooth()

    # Univariate filter / smoother
    mod2 = SARIMAX(endog, order=(1, 0, 0), measurement_error=True)
    mod2.ssm.filter_univariate = True
    mod2.update([2., 1., 1.])
    mod2.ssm["transition"] = transition
    res2 = mod2.ssm.smooth()

    # Simulation smoothers
    sim1 = simulation_smoother(mod1.ssm)
    sim2 = simulation_smoother(mod2.ssm)

    # Test for correctness
    assert_allclose(res1.forecasts[0, :], res2.forecasts[0, :])
    assert_allclose(res1.forecasts_error[0, :], res2.forecasts_error[0, :])
    assert_allclose(res1.forecasts_error_cov[0, 0, :],
                    res2.forecasts_error_cov[0, 0, :])
    assert_allclose(res1.filtered_state, res2.filtered_state)
    assert_allclose(res1.filtered_state_cov, res2.filtered_state_cov)
    assert_allclose(res1.predicted_state, res2.predicted_state)
    assert_allclose(res1.predicted_state_cov, res2.predicted_state_cov)
    assert_allclose(res1.llf_obs, res2.llf_obs)
    assert_allclose(res1.smoothed_state, res2.smoothed_state)
    assert_allclose(res1.smoothed_state_cov, res2.smoothed_state_cov)
    assert_allclose(res1.smoothed_measurement_disturbance,
                    res2.smoothed_measurement_disturbance)
    assert_allclose(res1.smoothed_measurement_disturbance_cov.diagonal(),
                    res2.smoothed_measurement_disturbance_cov.diagonal())
    assert_allclose(res1.smoothed_state_disturbance,
                    res2.smoothed_state_disturbance)
    assert_allclose(res1.smoothed_state_disturbance_cov,
                    res2.smoothed_state_disturbance_cov)

    assert_allclose(sim1.simulated_state, sim2.simulated_state)
    assert_allclose(sim1.simulated_measurement_disturbance,
                    sim2.simulated_measurement_disturbance)
    assert_allclose(sim1.simulated_state_disturbance,
                    sim2.simulated_state_disturbance)


def nondiagonal_obs_cov_model(initialization="approximate_diffuse",
                              missing=False):
    dta = datasets.macrodata.load_pandas().data
    dta.index = pd.date_range(start="1959-01-01", end="2009-7-01", freq="QS")
    obs = dta[["realgdp", "realcons", "realinv"]].diff().iloc[1:]
    if missing:
        obs.iloc[0:50, 0] = np.nan
        obs.iloc[119:130, 2] = np.nan

    mod = MLEModel(obs, k_states=3, k_posdef=3)
    mod["design"] = np.eye(3)
    X = (np.arange(9) + 1).reshape((3, 3)) / 10.
    mod["obs_cov"] = np.dot(X, X.T)
    mod["transition"] = np.eye(3)
    mod["selection"] = np.eye(3)
    mod["state_cov"] = np.eye(3)
    if initialization == "diffuse":
        mod.ssm.initialize_diffuse()
    else:
        mod.ssm.initialize_approximate_diffuse(1e6)
    return obs, mod


@pytest.mark.filterwarnings("ignore:The univariate filtering approach")
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("filter_univariate", [False, True])
@pytest.mark.parametrize("initialization", ["approximate_diffuse", "diffuse"])
def test_nondiagonal_obs_cov_smoothed_decomposition(initialization,
                                                    filter_univariate,
                                                    missing):
    # y_t = Z_t alpha-hat_t + eps-hat_t holds by construction, so it pins down
    # the smoothed measurement disturbance without a reference implementation.
    # The exact diffuse periods use the univariate approach either way.
    obs, mod = nondiagonal_obs_cov_model(initialization, missing)
    mod.ssm.filter_univariate = filter_univariate
    # The univariate approach warns that obs_cov has rank 2
    if filter_univariate or initialization == "diffuse":
        ctx = pytest.warns(UserWarning, match=PSD_OBS_COV_MSG)
    else:
        ctx = contextlib.nullcontext()
    with ctx:
        res = mod.ssm.smooth()

    fitted = (mod["design"] @ res.smoothed_state
              + res.smoothed_measurement_disturbance)
    mask = ~np.isnan(obs.values.T)
    assert_allclose(fitted[mask], obs.values.T[mask], atol=1e-7)


def test_nondiagonal_obs_cov_disturbance_cov_warns():
    _, mod = nondiagonal_obs_cov_model()
    mod.ssm.filter_univariate = True
    with pytest.warns(UserWarning, match=PSD_OBS_COV_MSG):
        with pytest.warns(OutputWarning, match="diagonalized"):
            mod.ssm.smooth()

    # Nothing to warn about when the observation equation is not diagonalized
    mod.ssm.filter_univariate = False
    with warnings.catch_warnings():
        warnings.simplefilter("error", OutputWarning)
        mod.ssm.smooth()


def ldl_obs_cov_model(obs_cov):
    # Local level model for each series, with a given (non-diagonal) obs_cov
    obs_cov = np.asarray(obs_cov, dtype=float)
    k_endog = obs_cov.shape[0]
    endog = np.random.default_rng(1).standard_normal((60, k_endog))
    mod = MLEModel(endog, k_states=k_endog, k_posdef=k_endog)
    mod["design"] = np.eye(k_endog)
    mod["obs_cov"] = obs_cov
    mod["transition"] = np.eye(k_endog)
    mod["selection"] = np.eye(k_endog)
    mod["state_cov"] = 0.5 * np.eye(k_endog)
    mod.ssm.initialize_known(np.zeros(k_endog), np.eye(k_endog))
    return mod


# The univariate approach diagonalizes a non-diagonal obs_cov with an LDL
# decomposition, so it must give the loglikelihood of the conventional
# filter, which does not decompose obs_cov
PSD_OBS_COV = {
    "rank 1 of 2": [[1.0, 1.0], [1.0, 1.0]],
    # Zero pivots in the second and third column
    "rank 1 of 3": [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]],
    # u u' with u of shape (3, 2): rounding leaves the last pivot slightly
    # negative (about -1e-15 times the diagonal element)
    "rank 2 of 3, a": [
        [67.18770623032997, 7.216822768597606, -6.143090820525295],
        [7.216822768597606, 4.3482658442951525, -9.268905216211522],
        [-6.143090820525295, -9.268905216211522, 21.304493156804693]],
    "rank 2 of 3, b": [
        [2.0209946816759143, 2.0720688554242166, 8.98201555216445],
        [2.0720688554242166, 49.067190529262085, -22.007502043723502],
        [8.98201555216445, -22.007502043723502, 60.67795241745098]],
    "rank 2 of 3, c": [
        [37.381500458335324, 35.643608705601444, -39.70094764775798],
        [35.643608705601444, 93.7861839217789, -9.866472318169604],
        [-39.70094764775798, -9.866472318169604, 55.264219631272326]],
    "zero row": [[0.0, 0.0, 0.0], [0.0, 1.0, 0.5], [0.0, 0.5, 1.0]],
    # Positive definite, but the last pivot is below 1e-10 times the diagonal
    # element, so it warns; the pivot itself is kept
    "numerically singular": [[1.0, 1.0], [1.0, 1.0 + 1e-12]],
}
PD_OBS_COV = {
    "2x2": [[1.0, 0.5], [0.5, 1.0]],
    "3x3": [[2.0, 0.6, -0.4], [0.6, 1.5, 0.3], [-0.4, 0.3, 1.0]],
    # The pivot 1e-8 is above the tolerance
    "nearly singular": [[1.0, 1.0], [1.0, 1.0 + 1e-8]],
    # The tolerance is relative to the diagonal element
    "small variance": [[1e-11, 1e-9], [1e-9, 1.0]],
    "small uncorrelated variance": [[1e-11, 0.0, 0.0], [0.0, 1.0, 0.5],
                                    [0.0, 0.5, 1.0]],
}
INDEFINITE_OBS_COV = {
    "negative pivot": [[1.0, 2.0], [2.0, 1.0]],
    "negative first variance": [[-1.0, 0.5], [0.5, 1.0]],
    "negative second variance": [[1.0, 0.5], [0.5, -1.0]],
    "zero variance": [[1.0, 0.5], [0.5, 0.0]],
    "zero pivot": [[0.0, 1.0], [1.0, 1.0]],
    "zero pivot, 3x3": [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]],
    # D C D with C = [[1, 1, 0], [1, 1, 1e-6], [0, 1e-6, 1]] (smallest
    # eigenvalue about -5e-13) and D = diag(1, 2**-20, 2**-20): the check of
    # the column below a zero pivot does not depend on the scale of the series
    "zero pivot, scaled": [[1.0, 2.0**-20, 0.0],
                           [2.0**-20, 2.0**-40, 1e-6 * 2.0**-40],
                           [0.0, 1e-6 * 2.0**-40, 2.0**-40]],
}


@pytest.mark.parametrize("complex_step", [False, True])
@pytest.mark.parametrize("obs_cov", list(PSD_OBS_COV.values()),
                         ids=list(PSD_OBS_COV))
def test_univariate_psd_obs_cov(obs_cov, complex_step):
    mod = ldl_obs_cov_model(obs_cov)
    desired = mod.ssm.loglike()

    mod.ssm.filter_univariate = True
    msg = PSD_OBS_COV_MSG + " encountered at period 0"
    with pytest.warns(UserWarning, match=msg):
        llf = mod.ssm.loglike(complex_step=complex_step)
    # The conventional filter does not decompose obs_cov; the two agree up to
    # rounding
    assert_allclose(llf, desired, rtol=1e-12)


@pytest.mark.parametrize("complex_step", [False, True])
@pytest.mark.parametrize("obs_cov", list(PD_OBS_COV.values()),
                         ids=list(PD_OBS_COV))
def test_univariate_pd_obs_cov(obs_cov, complex_step):
    mod = ldl_obs_cov_model(obs_cov)
    desired = mod.ssm.loglike()

    mod.ssm.filter_univariate = True
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=PSD_OBS_COV_MSG)
        llf = mod.ssm.loglike(complex_step=complex_step)
    assert_allclose(llf, desired, rtol=1e-12)


@pytest.mark.parametrize("complex_step", [False, True])
@pytest.mark.parametrize("obs_cov", list(INDEFINITE_OBS_COV.values()),
                         ids=list(INDEFINITE_OBS_COV))
def test_univariate_indefinite_obs_cov_raises(obs_cov, complex_step):
    mod = ldl_obs_cov_model(obs_cov)
    mod.ssm.filter_univariate = True
    msg = "Invalid value in LDL factorization of observation covariance"
    with pytest.raises(np.linalg.LinAlgError, match=msg):
        mod.ssm.loglike(complex_step=complex_step)


def test_univariate_ill_conditioned_pd_obs_cov():
    # L D L' with L = [[1, 0, 0], [0.3, 1, 0], [0.2, 5e5, 1]] and
    # D = diag(1, 1e-12, 0.75): positive definite, condition number about
    # 2e12. The second pivot is below the tolerance, but the rest of its
    # column is not small, so it is used as a positive pivot, as before
    # (statsmodels 0.15.0 gave the loglikelihood -297.21603787944184)
    obs_cov = [[1.0, 0.3, 0.2], [0.3, 0.090000000001, 0.0600005],
               [0.2, 0.0600005, 1.04]]
    mod = ldl_obs_cov_model(obs_cov)
    desired = mod.ssm.loglike()

    mod.ssm.filter_univariate = True
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=PSD_OBS_COV_MSG)
        llf = mod.ssm.loglike()
    # The decomposition has an entry of 5e5 in L, so changing obs_cov by one
    # ulp moves this loglikelihood by up to 7e-7 (relative), and the
    # conventional filter differs from it by 4e-7
    assert_allclose(llf, -297.21603787944184, rtol=5e-6)
    assert_allclose(llf, desired, rtol=5e-6)


def test_univariate_small_last_pivot_kept():
    # The last pivot of obs_cov (5e-11) is below the tolerance, so it warns,
    # but the pivot is kept. It matters here: the states add only about 3e-9
    # to the variance of the second transformed series, and setting the
    # pivot to zero would change the loglikelihood by 4%. The forecast error
    # covariance of the conventional filter has a condition number of about
    # 1e9, so the two agree to about 1e-7
    endog = np.random.default_rng(1).standard_normal((60, 2))
    mod = MLEModel(endog, k_states=2, k_posdef=2)
    mod["design"] = np.eye(2)
    mod["obs_cov"] = [[1.0, 1.0], [1.0, 1.0 + 5e-11]]
    mod["transition"] = 0.5 * np.eye(2)
    mod["selection"] = np.eye(2)
    mod["state_cov"] = 1e-9 * np.eye(2)
    mod.ssm.initialize_known(np.zeros(2), 1e-9 * np.eye(2))
    desired = mod.ssm.loglike()

    mod.ssm.filter_univariate = True
    msg = PSD_OBS_COV_MSG + " encountered at period 0"
    with pytest.warns(UserWarning, match=msg):
        llf = mod.ssm.loglike()
    assert_allclose(llf, desired, rtol=1e-6)
