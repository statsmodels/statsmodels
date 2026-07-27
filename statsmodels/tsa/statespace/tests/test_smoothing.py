"""
Tests for smoothing and estimation of unobserved states and disturbances

- Predicted states: :math:`E(\\alpha_t | Y_{t-1})`
- Filtered states: :math:`E(\\alpha_t | Y_t)`
- Smoothed states: :math:`E(\\alpha_t | Y_n)`
- Smoothed disturbances :math:`E(\\varepsilon_t | Y_n), E(\\eta_t | Y_n)`

Tested against R (FKF, KalmanRun / KalmanSmooth), Stata (sspace), and
MATLAB (ssm toolbox)

Author: Chad Fulton
License: Simplified-BSD
"""

from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pandas as pd
import pytest

from statsmodels import datasets
from statsmodels.tsa.statespace import mlemodel, sarimax, varmax
from statsmodels.tsa.statespace.kalman_filter import FILTER_UNIVARIATE
from statsmodels.tsa.statespace.kalman_smoother import (
    SMOOTH_ALTERNATIVE,
    SMOOTH_CLASSICAL,
    SMOOTH_UNIVARIATE,
)
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS

current_path = Path(__file__).resolve().parent


class TestStatesAR3:

    @classmethod
    def setup_class(cls, *args, alternate_timing=False, **kwargs):
        path = Path(current_path).joinpath("results", "results_wpi1_ar3_stata.csv")
        cls.stata = pd.read_csv(path)
        cls.stata.index = pd.date_range(start="1960-01-01", periods=124, freq="QS")
        path = Path(current_path).joinpath("results", "results_wpi1_ar3_matlab_ssm.csv")
        matlab_names = [
            "a1",
            "a2",
            "a3",
            "detP",
            "alphahat1",
            "alphahat2",
            "alphahat3",
            "detV",
            "eps",
            "epsvar",
            "eta",
            "etavar",
        ]
        cls.matlab_ssm = pd.read_csv(path, header=None, names=matlab_names)
        cls.model = sarimax.SARIMAX(
            cls.stata["wpi"],
            *args,
            order=(3, 1, 0),
            simple_differencing=True,
            hamilton_representation=True,
            **kwargs,
        )
        if alternate_timing:
            cls.model.ssm.timing_init_filtered = True
        params = np.r_[0.5270715, 0.0952613, 0.2580355, 0.5307459]
        cls.results = cls.model.smooth(params, cov_type="none")
        cls.results.det_predicted_state_cov = np.zeros((1, cls.model.nobs))
        cls.results.det_smoothed_state_cov = np.zeros((1, cls.model.nobs))
        for i in range(cls.model.nobs):
            cls.results.det_predicted_state_cov[0, i] = np.linalg.det(
                cls.results.filter_results.predicted_state_cov[:, :, i]
            )
            cls.results.det_smoothed_state_cov[0, i] = np.linalg.det(
                cls.results.smoother_results.smoothed_state_cov[:, :, i]
            )
        nobs = cls.model.nobs
        k_endog = cls.model.k_endog
        k_posdef = cls.model.ssm.k_posdef
        cls.sim = cls.model.simulation_smoother(filter_timing=0)
        cls.sim.simulate(
            measurement_disturbance_variates=np.zeros(nobs * k_endog),
            state_disturbance_variates=np.zeros(nobs * k_posdef),
            initial_state_variates=np.zeros(cls.model.k_states),
        )

    def test_predict_obs(self):
        assert_almost_equal(
            self.results.filter_results.predict().forecasts[0],
            self.stata.iloc[1:]["dep1"],
            4,
        )

    def test_standardized_residuals(self):
        assert_almost_equal(
            self.results.filter_results.standardized_forecasts_error[0],
            self.stata.iloc[1:]["sr1"],
            4,
        )

    def test_predicted_states(self):
        assert_almost_equal(
            self.results.filter_results.predicted_state[:, :-1].T,
            self.stata.iloc[1:][["sp1", "sp2", "sp3"]],
            4,
        )
        assert_almost_equal(
            self.results.filter_results.predicted_state[:, :-1].T,
            self.matlab_ssm[["a1", "a2", "a3"]],
            4,
        )

    def test_predicted_states_cov(self):
        assert_almost_equal(
            self.results.det_predicted_state_cov.T, self.matlab_ssm[["detP"]], 4
        )

    def test_filtered_states(self):
        assert_almost_equal(
            self.results.filter_results.filtered_state.T,
            self.stata.iloc[1:][["sf1", "sf2", "sf3"]],
            4,
        )

    def test_smoothed_states(self):
        assert_almost_equal(
            self.results.smoother_results.smoothed_state.T,
            self.stata.iloc[1:][["sm1", "sm2", "sm3"]],
            4,
        )
        assert_almost_equal(
            self.results.smoother_results.smoothed_state.T,
            self.matlab_ssm[["alphahat1", "alphahat2", "alphahat3"]],
            4,
        )

    def test_smoothed_states_cov(self):
        assert_almost_equal(
            self.results.det_smoothed_state_cov.T, self.matlab_ssm[["detV"]], 4
        )

    def test_smoothed_measurement_disturbance(self):
        assert_almost_equal(
            self.results.smoother_results.smoothed_measurement_disturbance.T,
            self.matlab_ssm[["eps"]],
            4,
        )

    def test_smoothed_measurement_disturbance_cov(self):
        res = self.results.smoother_results
        assert_almost_equal(
            res.smoothed_measurement_disturbance_cov[0].T,
            self.matlab_ssm[["epsvar"]],
            4,
        )

    def test_smoothed_state_disturbance(self):
        assert_almost_equal(
            self.results.smoother_results.smoothed_state_disturbance.T,
            self.matlab_ssm[["eta"]],
            4,
        )

    def test_smoothed_state_disturbance_cov(self):
        assert_almost_equal(
            self.results.smoother_results.smoothed_state_disturbance_cov[0].T,
            self.matlab_ssm[["etavar"]],
            4,
        )


class TestStatesAR3AlternateTiming(TestStatesAR3):

    @classmethod
    def setup_class(cls, *args, **kwargs):
        super().setup_class(*args, alternate_timing=True, **kwargs)


class TestStatesAR3AlternativeSmoothing(TestStatesAR3):

    @classmethod
    def setup_class(cls, *args, **kwargs):
        super().setup_class(*args, smooth_method=SMOOTH_ALTERNATIVE, **kwargs)

    def test_smoothed_states(self):
        assert_almost_equal(
            self.results.smoother_results.smoothed_state.T[2:],
            self.stata.iloc[3:][["sm1", "sm2", "sm3"]],
            4,
        )
        assert_almost_equal(
            self.results.smoother_results.smoothed_state.T[2:],
            self.matlab_ssm.iloc[2:][["alphahat1", "alphahat2", "alphahat3"]],
            4,
        )

    def test_smoothed_states_cov(self):
        assert_almost_equal(
            self.results.det_smoothed_state_cov.T[1:],
            self.matlab_ssm.iloc[1:][["detV"]],
            4,
        )

    def test_smooth_method(self):
        assert_equal(self.model.ssm.smooth_method, SMOOTH_ALTERNATIVE)
        assert_equal(self.model.ssm._kalman_smoother.smooth_method, SMOOTH_ALTERNATIVE)
        assert_equal(self.model.ssm._kalman_smoother._smooth_method, SMOOTH_ALTERNATIVE)


class TestStatesAR3UnivariateSmoothing(TestStatesAR3):

    @classmethod
    def setup_class(cls, *args, **kwargs):
        super().setup_class(*args, filter_method=FILTER_UNIVARIATE, **kwargs)

    def test_smooth_method(self):
        assert_equal(self.model.ssm.smooth_method, 0)
        assert_equal(self.model.ssm._kalman_smoother.smooth_method, 0)
        assert_equal(self.model.ssm._kalman_smoother._smooth_method, SMOOTH_UNIVARIATE)


class TestStatesMissingAR3:

    @classmethod
    def setup_class(cls, alternate_timing=False, *args, **kwargs):
        path = Path(current_path).joinpath("results", "results_wpi1_ar3_stata.csv")
        cls.stata = pd.read_csv(path)
        cls.stata.index = pd.date_range(start="1960-01-01", periods=124, freq="QS")
        path = Path(current_path).joinpath(
            "results", "results_wpi1_missing_ar3_matlab_ssm.csv"
        )
        matlab_names = [
            "a1",
            "a2",
            "a3",
            "detP",
            "alphahat1",
            "alphahat2",
            "alphahat3",
            "detV",
            "eps",
            "epsvar",
            "eta",
            "etavar",
        ]
        cls.matlab_ssm = pd.read_csv(path, header=None, names=matlab_names)
        path = Path(current_path).joinpath("results", "results_smoothing3_R.csv")
        cls.R_ssm = pd.read_csv(path)
        cls.stata["dwpi"] = cls.stata["wpi"].diff()
        cls.stata.loc[cls.stata.index[10:21], "dwpi"] = np.nan
        cls.model = sarimax.SARIMAX(
            cls.stata.loc[cls.stata.index[1:], "dwpi"],
            *args,
            order=(3, 0, 0),
            hamilton_representation=True,
            **kwargs,
        )
        if alternate_timing:
            cls.model.ssm.timing_init_filtered = True
        params = np.r_[0.5270715, 0.0952613, 0.2580355, 0.5307459]
        cls.results = cls.model.smooth(params, return_ssm=True)
        cls.results.det_predicted_state_cov = np.zeros((1, cls.model.nobs))
        cls.results.det_smoothed_state_cov = np.zeros((1, cls.model.nobs))
        for i in range(cls.model.nobs):
            cls.results.det_predicted_state_cov[0, i] = np.linalg.det(
                cls.results.predicted_state_cov[:, :, i]
            )
            cls.results.det_smoothed_state_cov[0, i] = np.linalg.det(
                cls.results.smoothed_state_cov[:, :, i]
            )
        cls.smoothed_forecasts = cls.results.smoothed_forecasts
        nobs = cls.model.nobs
        k_endog = cls.model.k_endog
        k_posdef = cls.model.ssm.k_posdef
        cls.sim = cls.model.simulation_smoother()
        cls.sim.simulate(
            measurement_disturbance_variates=np.zeros(nobs * k_endog),
            state_disturbance_variates=np.zeros(nobs * k_posdef),
            initial_state_variates=np.zeros(cls.model.k_states),
        )

    def test_predicted_states(self):
        assert_almost_equal(
            self.results.predicted_state[:, :-1].T,
            self.matlab_ssm[["a1", "a2", "a3"]],
            4,
        )

    def test_predicted_states_cov(self):
        assert_almost_equal(
            self.results.det_predicted_state_cov.T, self.matlab_ssm[["detP"]], 4
        )

    def test_smoothed_states(self):
        assert_almost_equal(
            self.results.smoothed_state.T,
            self.matlab_ssm[["alphahat1", "alphahat2", "alphahat3"]],
            4,
        )

    def test_smoothed_states_cov(self):
        assert_almost_equal(
            self.results.det_smoothed_state_cov.T, self.matlab_ssm[["detV"]], 4
        )

    def test_smoothed_measurement_disturbance(self):
        assert_almost_equal(
            self.results.smoothed_measurement_disturbance.T, self.matlab_ssm[["eps"]], 4
        )

    def test_smoothed_measurement_disturbance_cov(self):
        assert_almost_equal(
            self.results.smoothed_measurement_disturbance_cov[0].T,
            self.matlab_ssm[["epsvar"]],
            4,
        )

    def test_smoothed_state_disturbance(self):
        assert_almost_equal(
            self.results.smoothed_state_disturbance.T, self.R_ssm[["etahat"]], 9
        )

    def test_smoothed_state_disturbance_cov(self):
        assert_almost_equal(
            self.results.smoothed_state_disturbance_cov[0, 0, :],
            self.R_ssm["detVeta"],
            9,
        )


class TestStatesMissingAR3AlternateTiming(TestStatesMissingAR3):

    @classmethod
    def setup_class(cls, *args, **kwargs):
        super().setup_class(*args, alternate_timing=True, **kwargs)


class TestStatesMissingAR3AlternativeSmoothing(TestStatesMissingAR3):

    @classmethod
    def setup_class(cls, *args, **kwargs):
        super().setup_class(*args, smooth_method=SMOOTH_ALTERNATIVE, **kwargs)

    def test_smooth_method(self):
        assert_equal(self.model.ssm.smooth_method, SMOOTH_ALTERNATIVE)
        assert_equal(self.model.ssm._kalman_smoother.smooth_method, SMOOTH_ALTERNATIVE)
        assert_equal(self.model.ssm._kalman_smoother._smooth_method, SMOOTH_ALTERNATIVE)


class TestStatesMissingAR3UnivariateSmoothing(TestStatesMissingAR3):

    @classmethod
    def setup_class(cls, *args, **kwargs):
        super().setup_class(*args, filter_method=FILTER_UNIVARIATE, **kwargs)

    def test_smooth_method(self):
        assert_equal(self.model.ssm.smooth_method, 0)
        assert_equal(self.model.ssm._kalman_smoother.smooth_method, 0)
        assert_equal(self.model.ssm._kalman_smoother._smooth_method, SMOOTH_UNIVARIATE)


class TestMultivariateMissing:
    """
    Tests for most filtering and smoothing variables against output from the
    R library KFAS.

    Note that KFAS uses the univariate approach which generally will result in
    different predicted values and covariance matrices associated with the
    measurement equation (e.g. forecasts, etc.). In this case, although the
    model is multivariate, each of the series is truly independent so the
    values will be the same regardless of whether the univariate approach
    is used or not.
    """

    @classmethod
    def setup_class(cls, **kwargs):
        path = Path(current_path).joinpath("results", "results_smoothing_R.csv")
        cls.desired = pd.read_csv(path)
        dta = datasets.macrodata.load_pandas().data
        dta.index = pd.date_range(start="1959-01-01", end="2009-7-01", freq="QS")
        obs = dta[["realgdp", "realcons", "realinv"]].diff().iloc[1:]
        obs.iloc[0:50, 0] = np.nan
        obs.iloc[19:70, 1] = np.nan
        obs.iloc[39:90, 2] = np.nan
        obs.iloc[119:130, 0] = np.nan
        obs.iloc[119:130, 2] = np.nan
        mod = mlemodel.MLEModel(obs, k_states=3, k_posdef=3, **kwargs)
        mod["design"] = np.eye(3)
        mod["obs_cov"] = np.eye(3)
        mod["transition"] = np.eye(3)
        mod["selection"] = np.eye(3)
        mod["state_cov"] = np.eye(3)
        mod.initialize_approximate_diffuse(1000000.0)
        cls.model = mod
        cls.results = mod.smooth([], return_ssm=True)
        cls.results.det_scaled_smoothed_estimator_cov = np.zeros((1, cls.model.nobs))
        cls.results.det_predicted_state_cov = np.zeros((1, cls.model.nobs))
        cls.results.det_smoothed_state_cov = np.zeros((1, cls.model.nobs))
        cls.results.det_smoothed_state_disturbance_cov = np.zeros((1, cls.model.nobs))
        cls.smoothed_forecasts = cls.results.smoothed_forecasts
        for i in range(cls.model.nobs):
            cls.results.det_scaled_smoothed_estimator_cov[0, i] = np.linalg.det(
                cls.results.scaled_smoothed_estimator_cov[:, :, i]
            )
            cls.results.det_predicted_state_cov[0, i] = np.linalg.det(
                cls.results.predicted_state_cov[:, :, i + 1]
            )
            cls.results.det_smoothed_state_cov[0, i] = np.linalg.det(
                cls.results.smoothed_state_cov[:, :, i]
            )
            cls.results.det_smoothed_state_disturbance_cov[0, i] = np.linalg.det(
                cls.results.smoothed_state_disturbance_cov[:, :, i]
            )

    def test_loglike(self):
        assert_allclose(np.sum(self.results.llf_obs), -205310.9767)

    def test_scaled_smoothed_estimator(self):
        assert_allclose(
            self.results.scaled_smoothed_estimator.T, self.desired[["r1", "r2", "r3"]]
        )

    def test_scaled_smoothed_estimator_cov(self):
        assert_allclose(
            self.results.det_scaled_smoothed_estimator_cov.T, self.desired[["detN"]]
        )

    def test_forecasts(self):
        assert_allclose(self.results.forecasts.T, self.desired[["m1", "m2", "m3"]])

    def test_forecasts_error(self):
        assert_allclose(
            self.results.forecasts_error.T, self.desired[["v1", "v2", "v3"]]
        )

    def test_forecasts_error_cov(self):
        assert_allclose(
            self.results.forecasts_error_cov.diagonal(),
            self.desired[["F1", "F2", "F3"]],
        )

    def test_predicted_states(self):
        assert_allclose(
            self.results.predicted_state[:, 1:].T, self.desired[["a1", "a2", "a3"]]
        )

    def test_predicted_states_cov(self):
        assert_allclose(self.results.det_predicted_state_cov.T, self.desired[["detP"]])

    def test_smoothed_states(self):
        assert_allclose(
            self.results.smoothed_state.T,
            self.desired[["alphahat1", "alphahat2", "alphahat3"]],
        )

    def test_smoothed_states_cov(self):
        assert_allclose(self.results.det_smoothed_state_cov.T, self.desired[["detV"]])

    def test_smoothed_forecasts(self):
        assert_allclose(
            self.smoothed_forecasts.T, self.desired[["muhat1", "muhat2", "muhat3"]]
        )

    def test_smoothed_state_disturbance(self):
        assert_allclose(
            self.results.smoothed_state_disturbance.T,
            self.desired[["etahat1", "etahat2", "etahat3"]],
        )

    def test_smoothed_state_disturbance_cov(self):
        assert_allclose(
            self.results.det_smoothed_state_disturbance_cov.T, self.desired[["detVeta"]]
        )

    def test_smoothed_measurement_disturbance(self):
        assert_allclose(
            self.results.smoothed_measurement_disturbance.T,
            self.desired[["epshat1", "epshat2", "epshat3"]],
        )

    def test_smoothed_measurement_disturbance_cov(self):
        assert_allclose(
            self.results.smoothed_measurement_disturbance_cov.diagonal(),
            self.desired[["Veps1", "Veps2", "Veps3"]],
        )


class TestMultivariateMissingClassicalSmoothing(TestMultivariateMissing):

    @classmethod
    def setup_class(cls, *args, **kwargs):
        super().setup_class(*args, smooth_method=SMOOTH_CLASSICAL, **kwargs)

    def test_smooth_method(self):
        assert_equal(self.model.ssm.smooth_method, SMOOTH_CLASSICAL)
        assert_equal(self.model.ssm._kalman_smoother.smooth_method, SMOOTH_CLASSICAL)
        assert_equal(self.model.ssm._kalman_smoother._smooth_method, SMOOTH_CLASSICAL)


class TestMultivariateMissingAlternativeSmoothing(TestMultivariateMissing):

    @classmethod
    def setup_class(cls, *args, **kwargs):
        super().setup_class(*args, smooth_method=SMOOTH_ALTERNATIVE, **kwargs)

    def test_smooth_method(self):
        assert_equal(self.model.ssm.smooth_method, SMOOTH_ALTERNATIVE)
        assert_equal(self.model.ssm._kalman_smoother.smooth_method, SMOOTH_ALTERNATIVE)
        assert_equal(self.model.ssm._kalman_smoother._smooth_method, SMOOTH_ALTERNATIVE)


class TestMultivariateMissingUnivariateSmoothing(TestMultivariateMissing):

    @classmethod
    def setup_class(cls, **kwargs):
        super().setup_class(filter_method=FILTER_UNIVARIATE, **kwargs)

    def test_smooth_method(self):
        assert_equal(self.model.ssm.smooth_method, 0)
        assert_equal(self.model.ssm._kalman_smoother.smooth_method, 0)
        assert_equal(self.model.ssm._kalman_smoother._smooth_method, SMOOTH_UNIVARIATE)


class TestMultivariateVAR:
    """
    Tests for most filtering and smoothing variables against output from the
    R library KFAS.

    Note that KFAS uses the univariate approach which generally will result in
    different predicted values and covariance matrices associated with the
    measurement equation (e.g. forecasts, etc.). In this case, although the
    model is multivariate, each of the series is truly independent so the
    values will be the same regardless of whether the univariate approach is
    used or not.
    """

    @classmethod
    def setup_class(cls, *args, **kwargs):
        path = Path(current_path).joinpath("results", "results_smoothing2_R.csv")
        cls.desired = pd.read_csv(path)
        dta = datasets.macrodata.load_pandas().data
        dta.index = pd.date_range(start="1959-01-01", end="2009-7-01", freq="QS")
        obs = np.log(dta[["realgdp", "realcons", "realinv"]]).diff().iloc[1:]
        mod = mlemodel.MLEModel(obs, k_states=3, k_posdef=3, **kwargs)
        mod["design"] = np.eye(3)
        mod["obs_cov"] = np.array(
            [[6.40649e-05, 0.0, 0.0], [0.0, 5.72802e-05, 0.0], [0.0, 0.0, 0.0017088585]]
        )
        mod["transition"] = np.array(
            [
                [-0.1119908792, 0.8441841604, 0.0238725303],
                [0.2629347724, 0.4996718412, -0.0173023305],
                [-3.2192369082, 4.1536028244, 0.4514379215],
            ]
        )
        mod["selection"] = np.eye(3)
        mod["state_cov"] = np.array(
            [
                [6.40649e-05, 3.88496e-05, 0.0002148769],
                [3.88496e-05, 5.72802e-05, 1.555e-06],
                [0.0002148769, 1.555e-06, 0.0017088585],
            ]
        )
        mod.initialize_approximate_diffuse(1000000.0)
        cls.model = mod
        cls.results = mod.smooth([], return_ssm=True)
        cls.results.det_scaled_smoothed_estimator_cov = np.zeros((1, cls.model.nobs))
        cls.results.det_predicted_state_cov = np.zeros((1, cls.model.nobs))
        cls.results.det_smoothed_state_cov = np.zeros((1, cls.model.nobs))
        cls.results.det_smoothed_state_disturbance_cov = np.zeros((1, cls.model.nobs))
        cls.smoothed_forecasts = cls.results.smoothed_forecasts
        for i in range(cls.model.nobs):
            cls.results.det_scaled_smoothed_estimator_cov[0, i] = np.linalg.det(
                cls.results.scaled_smoothed_estimator_cov[:, :, i]
            )
            cls.results.det_predicted_state_cov[0, i] = np.linalg.det(
                cls.results.predicted_state_cov[:, :, i + 1]
            )
            cls.results.det_smoothed_state_cov[0, i] = np.linalg.det(
                cls.results.smoothed_state_cov[:, :, i]
            )
            cls.results.det_smoothed_state_disturbance_cov[0, i] = np.linalg.det(
                cls.results.smoothed_state_disturbance_cov[:, :, i]
            )

    def test_loglike(self):
        assert_allclose(np.sum(self.results.llf_obs), 1695.34872)

    def test_scaled_smoothed_estimator(self):
        assert_allclose(
            self.results.scaled_smoothed_estimator.T,
            self.desired[["r1", "r2", "r3"]],
            atol=0.0001,
        )

    def test_scaled_smoothed_estimator_cov(self):
        assert_allclose(
            np.log(self.results.det_scaled_smoothed_estimator_cov.T[:-1]),
            np.log(self.desired[["detN"]][:-1]),
            atol=1e-06,
        )

    def test_forecasts(self):
        assert_allclose(
            self.results.forecasts.T, self.desired[["m1", "m2", "m3"]], atol=1e-06
        )

    def test_forecasts_error(self):
        assert_allclose(
            self.results.forecasts_error.T[:, 0], self.desired["v1"], atol=1e-06
        )

    def test_forecasts_error_cov(self):
        assert_allclose(
            self.results.forecasts_error_cov.diagonal()[:, 0],
            self.desired["F1"],
            atol=1e-06,
        )

    def test_predicted_states(self):
        assert_allclose(
            self.results.predicted_state[:, 1:].T,
            self.desired[["a1", "a2", "a3"]],
            atol=1e-06,
        )

    def test_predicted_states_cov(self):
        assert_allclose(
            self.results.det_predicted_state_cov.T, self.desired[["detP"]], atol=1e-16
        )

    def test_smoothed_states(self):
        assert_allclose(
            self.results.smoothed_state.T,
            self.desired[["alphahat1", "alphahat2", "alphahat3"]],
            atol=1e-06,
        )

    def test_smoothed_states_cov(self):
        assert_allclose(
            self.results.det_smoothed_state_cov.T, self.desired[["detV"]], atol=1e-16
        )

    def test_smoothed_forecasts(self):
        assert_allclose(
            self.smoothed_forecasts.T,
            self.desired[["muhat1", "muhat2", "muhat3"]],
            atol=1e-06,
        )

    def test_smoothed_state_disturbance(self):
        assert_allclose(
            self.results.smoothed_state_disturbance.T,
            self.desired[["etahat1", "etahat2", "etahat3"]],
            atol=1e-06,
        )

    def test_smoothed_state_disturbance_cov(self):
        assert_allclose(
            self.results.det_smoothed_state_disturbance_cov.T,
            self.desired[["detVeta"]],
            atol=1e-18,
        )

    def test_smoothed_measurement_disturbance(self):
        assert_allclose(
            self.results.smoothed_measurement_disturbance.T,
            self.desired[["epshat1", "epshat2", "epshat3"]],
            atol=1e-06,
        )

    def test_smoothed_measurement_disturbance_cov(self):
        assert_allclose(
            self.results.smoothed_measurement_disturbance_cov.diagonal(),
            self.desired[["Veps1", "Veps2", "Veps3"]],
            atol=1e-06,
        )


class TestMultivariateVARAlternativeSmoothing(TestMultivariateVAR):

    @classmethod
    def setup_class(cls, *args, **kwargs):
        super().setup_class(*args, smooth_method=SMOOTH_ALTERNATIVE, **kwargs)

    def test_smooth_method(self):
        assert_equal(self.model.ssm.smooth_method, SMOOTH_ALTERNATIVE)
        assert_equal(self.model.ssm._kalman_smoother.smooth_method, SMOOTH_ALTERNATIVE)
        assert_equal(self.model.ssm._kalman_smoother._smooth_method, SMOOTH_ALTERNATIVE)


class TestMultivariateVARClassicalSmoothing(TestMultivariateVAR):

    @classmethod
    def setup_class(cls, *args, **kwargs):
        super().setup_class(*args, smooth_method=SMOOTH_CLASSICAL, **kwargs)

    def test_smooth_method(self):
        assert_equal(self.model.ssm.smooth_method, SMOOTH_CLASSICAL)
        assert_equal(self.model.ssm._kalman_smoother.smooth_method, SMOOTH_CLASSICAL)
        assert_equal(self.model.ssm._kalman_smoother._smooth_method, SMOOTH_CLASSICAL)


class TestMultivariateVARUnivariate:
    """
    Tests for most filtering and smoothing variables against output from the
    R library KFAS.

    Note that KFAS uses the univariate approach which generally will result in
    different predicted values and covariance matrices associated with the
    measurement equation (e.g. forecasts, etc.). In this case, although the
    model is multivariate, each of the series is truly independent so the
    values will be the same regardless of whether the univariate approach is
    used or not.
    """

    @classmethod
    def setup_class(cls, *args, **kwargs):
        path = Path(current_path).joinpath("results", "results_smoothing2_R.csv")
        cls.desired = pd.read_csv(path)
        dta = datasets.macrodata.load_pandas().data
        dta.index = pd.date_range(start="1959-01-01", end="2009-7-01", freq="QS")
        obs = np.log(dta[["realgdp", "realcons", "realinv"]]).diff().iloc[1:]
        mod = mlemodel.MLEModel(obs, k_states=3, k_posdef=3, **kwargs)
        mod.ssm.filter_univariate = True
        mod["design"] = np.eye(3)
        mod["obs_cov"] = np.array(
            [[6.40649e-05, 0.0, 0.0], [0.0, 5.72802e-05, 0.0], [0.0, 0.0, 0.0017088585]]
        )
        mod["transition"] = np.array(
            [
                [-0.1119908792, 0.8441841604, 0.0238725303],
                [0.2629347724, 0.4996718412, -0.0173023305],
                [-3.2192369082, 4.1536028244, 0.4514379215],
            ]
        )
        mod["selection"] = np.eye(3)
        mod["state_cov"] = np.array(
            [
                [6.40649e-05, 3.88496e-05, 0.0002148769],
                [3.88496e-05, 5.72802e-05, 1.555e-06],
                [0.0002148769, 1.555e-06, 0.0017088585],
            ]
        )
        mod.initialize_approximate_diffuse(1000000.0)
        cls.model = mod
        cls.results = mod.smooth([], return_ssm=True)
        cls.results.det_scaled_smoothed_estimator_cov = np.zeros((1, cls.model.nobs))
        cls.results.det_predicted_state_cov = np.zeros((1, cls.model.nobs))
        cls.results.det_smoothed_state_cov = np.zeros((1, cls.model.nobs))
        cls.results.det_smoothed_state_disturbance_cov = np.zeros((1, cls.model.nobs))
        cls.smoothed_forecasts = cls.results.smoothed_forecasts
        for i in range(cls.model.nobs):
            cls.results.det_scaled_smoothed_estimator_cov[0, i] = np.linalg.det(
                cls.results.scaled_smoothed_estimator_cov[:, :, i]
            )
            cls.results.det_predicted_state_cov[0, i] = np.linalg.det(
                cls.results.predicted_state_cov[:, :, i + 1]
            )
            cls.results.det_smoothed_state_cov[0, i] = np.linalg.det(
                cls.results.smoothed_state_cov[:, :, i]
            )
            cls.results.det_smoothed_state_disturbance_cov[0, i] = np.linalg.det(
                cls.results.smoothed_state_disturbance_cov[:, :, i]
            )

    def test_loglike(self):
        assert_allclose(np.sum(self.results.llf_obs), 1695.34872)

    def test_scaled_smoothed_estimator(self):
        assert_allclose(
            self.results.scaled_smoothed_estimator.T,
            self.desired[["r1", "r2", "r3"]],
            atol=0.0001,
        )

    def test_scaled_smoothed_estimator_cov(self):
        assert_allclose(
            np.log(self.results.det_scaled_smoothed_estimator_cov.T[:-1]),
            np.log(self.desired[["detN"]][:-1]),
        )

    def test_forecasts(self):
        assert_allclose(self.results.forecasts.T[:, 0], self.desired["m1"], atol=1e-06)

    def test_forecasts_error(self):
        assert_allclose(
            self.results.forecasts_error.T, self.desired[["v1", "v2", "v3"]], atol=1e-06
        )

    def test_forecasts_error_cov(self):
        assert_allclose(
            self.results.forecasts_error_cov.diagonal(),
            self.desired[["F1", "F2", "F3"]],
        )

    def test_predicted_states(self):
        assert_allclose(
            self.results.predicted_state[:, 1:].T,
            self.desired[["a1", "a2", "a3"]],
            atol=1e-08,
        )

    def test_predicted_states_cov(self):
        assert_allclose(
            self.results.det_predicted_state_cov.T, self.desired[["detP"]], atol=1e-18
        )

    def test_smoothed_states(self):
        assert_allclose(
            self.results.smoothed_state.T,
            self.desired[["alphahat1", "alphahat2", "alphahat3"]],
            atol=1e-06,
        )

    def test_smoothed_states_cov(self):
        assert_allclose(
            self.results.det_smoothed_state_cov.T, self.desired[["detV"]], atol=1e-18
        )

    def test_smoothed_forecasts(self):
        assert_allclose(
            self.smoothed_forecasts.T,
            self.desired[["muhat1", "muhat2", "muhat3"]],
            atol=1e-06,
        )

    def test_smoothed_state_disturbance(self):
        assert_allclose(
            self.results.smoothed_state_disturbance.T,
            self.desired[["etahat1", "etahat2", "etahat3"]],
            atol=1e-06,
        )

    def test_smoothed_state_disturbance_cov(self):
        assert_allclose(
            self.results.det_smoothed_state_disturbance_cov.T,
            self.desired[["detVeta"]],
            atol=1e-18,
        )

    def test_smoothed_measurement_disturbance(self):
        assert_allclose(
            self.results.smoothed_measurement_disturbance.T,
            self.desired[["epshat1", "epshat2", "epshat3"]],
            atol=1e-06,
        )

    def test_smoothed_measurement_disturbance_cov(self):
        assert_allclose(
            self.results.smoothed_measurement_disturbance_cov.diagonal(),
            self.desired[["Veps1", "Veps2", "Veps3"]],
        )


class TestMultivariateVARUnivariateSmoothing(TestMultivariateVARUnivariate):

    @classmethod
    def setup_class(cls, *args, **kwargs):
        super().setup_class(*args, filter_method=FILTER_UNIVARIATE, **kwargs)

    def test_filter_method(self):
        assert_equal(self.model.ssm.filter_method, FILTER_UNIVARIATE)
        assert_equal(self.model.ssm._kalman_smoother.filter_method, FILTER_UNIVARIATE)

    def test_smooth_method(self):
        assert_equal(self.model.ssm.smooth_method, 0)
        assert_equal(self.model.ssm._kalman_smoother.smooth_method, 0)
        assert_equal(self.model.ssm._kalman_smoother._smooth_method, SMOOTH_UNIVARIATE)


class TestVARAutocovariances:

    @classmethod
    def setup_class(cls, which="mixed", *args, **kwargs):
        dta = datasets.macrodata.load_pandas().data
        dta.index = pd.date_range(start="1959-01-01", end="2009-7-01", freq="QS")
        obs = np.log(dta[["realgdp", "realcons", "realinv"]]).diff().iloc[1:]
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
        mod = mlemodel.MLEModel(obs, k_states=3, k_posdef=3, **kwargs)
        mod["design"] = np.eye(3)
        mod["obs_cov"] = np.array(
            [
                [609.0746647855, 0.0, 0.0],
                [0.0, 1.8774916622, 0.0],
                [0.0, 0.0, 124.6768281675],
            ]
        )
        mod["transition"] = np.array(
            [
                [-0.8110473405, 1.8005304445, 1.0215975772],
                [-1.9846632699, 2.4091302213, 1.9264449765],
                [0.9181658823, -0.2442384581, -0.6393462272],
            ]
        )
        mod["selection"] = np.eye(3)
        mod["state_cov"] = np.array(
            [
                [1552.9758843938, 612.7185121905, 877.6157204992],
                [612.7185121905, 467.8739411204, 70.608037339],
                [877.6157204992, 70.608037339, 900.5440385836],
            ]
        )
        mod.initialize_approximate_diffuse(1000000.0)
        cls.model = mod
        cls.results = mod.smooth([], return_ssm=True)
        cls.smoothed_forecasts = cls.results.smoothed_forecasts
        kwargs.pop("filter_collapsed", None)
        mod = mlemodel.MLEModel(obs, k_states=6, k_posdef=3, **kwargs)
        mod["design", :3, :3] = np.eye(3)
        mod["obs_cov"] = np.array(
            [
                [609.0746647855, 0.0, 0.0],
                [0.0, 1.8774916622, 0.0],
                [0.0, 0.0, 124.6768281675],
            ]
        )
        mod["transition", :3, :3] = np.array(
            [
                [-0.8110473405, 1.8005304445, 1.0215975772],
                [-1.9846632699, 2.4091302213, 1.9264449765],
                [0.9181658823, -0.2442384581, -0.6393462272],
            ]
        )
        mod["transition", 3:, :3] = np.eye(3)
        mod["selection", :3, :3] = np.eye(3)
        mod["state_cov"] = np.array(
            [
                [1552.9758843938, 612.7185121905, 877.6157204992],
                [612.7185121905, 467.8739411204, 70.608037339],
                [877.6157204992, 70.608037339, 900.5440385836],
            ]
        )
        mod.initialize_approximate_diffuse(1000000.0)
        cls.augmented_model = mod
        cls.augmented_results = mod.smooth([], return_ssm=True)

    def test_smoothed_state_autocov(self):
        assert_allclose(
            self.results.smoothed_state_autocov[:, :, 0:5],
            self.augmented_results.smoothed_state_cov[:3, 3:, 1:6],
            atol=0.0001,
        )
        assert_allclose(
            self.results.smoothed_state_autocov[:, :, 5:-1],
            self.augmented_results.smoothed_state_cov[:3, 3:, 6:],
            atol=1e-07,
        )


class TestVARAutocovariancesAlternativeSmoothing(TestVARAutocovariances):

    @classmethod
    def setup_class(cls, *args, **kwargs):
        super().setup_class(*args, smooth_method=SMOOTH_ALTERNATIVE, **kwargs)

    def test_smooth_method(self):
        assert_equal(self.model.ssm.smooth_method, SMOOTH_ALTERNATIVE)
        assert_equal(self.model.ssm._kalman_smoother.smooth_method, SMOOTH_ALTERNATIVE)
        assert_equal(self.model.ssm._kalman_smoother._smooth_method, SMOOTH_ALTERNATIVE)


class TestVARAutocovariancesClassicalSmoothing(TestVARAutocovariances):

    @classmethod
    def setup_class(cls, *args, **kwargs):
        super().setup_class(*args, smooth_method=SMOOTH_CLASSICAL, **kwargs)

    def test_smooth_method(self):
        assert_equal(self.model.ssm.smooth_method, SMOOTH_CLASSICAL)
        assert_equal(self.model.ssm._kalman_smoother.smooth_method, SMOOTH_CLASSICAL)
        assert_equal(self.model.ssm._kalman_smoother._smooth_method, SMOOTH_CLASSICAL)


class TestVARAutocovariancesUnivariateSmoothing(TestVARAutocovariances):

    @classmethod
    def setup_class(cls, *args, **kwargs):
        super().setup_class(*args, filter_method=FILTER_UNIVARIATE, **kwargs)

    def test_filter_method(self):
        assert_equal(self.model.ssm.filter_method, FILTER_UNIVARIATE)
        assert_equal(self.model.ssm._kalman_smoother.filter_method, FILTER_UNIVARIATE)

    def test_smooth_method(self):
        assert_equal(self.model.ssm.smooth_method, 0)
        assert_equal(self.model.ssm._kalman_smoother.smooth_method, 0)
        assert_equal(self.model.ssm._kalman_smoother._smooth_method, SMOOTH_UNIVARIATE)


class TVSSWithLags(TVSS):

    def __init__(self, endog):
        super().__init__(endog, _k_states=8)
        self["transition", 2:, :6] = np.eye(6)[..., None]
        self.ssm.initialize_approximate_diffuse(0.0001)


def get_acov_model(
    missing, filter_univariate, tvp, oos=None, params=None, return_ssm=True
):
    dta = datasets.macrodata.load_pandas().data
    dta.index = pd.date_range(start="1959-01-01", end="2009-7-01", freq="QS")
    endog = np.log(dta[["realgdp", "realcons"]]).diff().iloc[1:]
    if missing == "all":
        endog.iloc[:5, :] = np.nan
        endog.iloc[11:13, :] = np.nan
    elif missing == "partial":
        endog.iloc[0:5, 0] = np.nan
        endog.iloc[11:13, 0] = np.nan
    elif missing == "mixed":
        endog.iloc[0:5, 0] = np.nan
        endog.iloc[1:7, 1] = np.nan
        endog.iloc[11:13, 0] = np.nan
    if oos is not None:
        new_ix = pd.date_range(
            start=endog.index[0], periods=len(endog) + oos, freq="QS"
        )
        endog = endog.reindex(new_ix)
    if not tvp:
        mod = varmax.VARMAX(endog, order=(4, 0, 0), measurement_error=True, tolerance=0)
        mod.ssm.filter_univariate = filter_univariate
        if params is None:
            params = mod.start_params
        res = mod.smooth(params, return_ssm=return_ssm)
    else:
        mod = TVSSWithLags(endog)
        mod.ssm.filter_univariate = filter_univariate
        res = mod.smooth([], return_ssm=return_ssm)
    return (mod, res)


@pytest.mark.parametrize("missing", ["all", "partial", "mixed", None])
@pytest.mark.parametrize("filter_univariate", [True, False])
@pytest.mark.parametrize("tvp", [True, False])
def test_smoothed_state_autocovariances_backwards(missing, filter_univariate, tvp):
    """
    Test for Cov(t, t - lag)
    """
    _, res = get_acov_model(missing, filter_univariate, tvp)
    cov = res.smoothed_state_cov.transpose(2, 0, 1)
    desired_acov1 = cov[:, :2, 2:4]
    desired_acov2 = cov[:, :2, 4:6]
    desired_acov3 = cov[:, :2, 6:8]
    acov1 = res.smoothed_state_autocovariance(1).transpose(2, 0, 1)
    assert_allclose(acov1[1:, :2, :2], desired_acov1[1:], rtol=1e-06, atol=1e-06)
    assert_equal(acov1[:1], np.nan)
    acov2 = res.smoothed_state_autocovariance(2).transpose(2, 0, 1)
    assert_allclose(acov2[2:, :2, :2], desired_acov2[2:], rtol=1e-06, atol=1e-06)
    assert_equal(acov2[:2], np.nan)
    acov3 = res.smoothed_state_autocovariance(3).transpose(2, 0, 1)
    assert_allclose(acov3[3:, :2, :2], desired_acov3[3:], rtol=1e-06, atol=1e-06)
    assert_equal(acov3[:3], np.nan)
    acov1 = res.smoothed_state_autocovariance(1, t=0)
    assert_allclose(acov1, np.nan)
    acov1 = res.smoothed_state_autocovariance(1, t=1)
    assert_allclose(acov1[:2, :2], desired_acov1[1], rtol=1e-06, atol=1e-06)
    acov1 = res.smoothed_state_autocovariance(1, start=8, end=9).transpose(2, 0, 1)
    assert_allclose(acov1[:, :2, :2], desired_acov1[8:9], rtol=1e-06, atol=1e-06)
    acov2 = res.smoothed_state_autocovariance(2, t=0)
    assert_allclose(acov2, np.nan)
    acov2 = res.smoothed_state_autocovariance(2, t=1)
    assert_allclose(acov2, np.nan)
    acov2 = res.smoothed_state_autocovariance(2, t=2)
    assert_allclose(acov2[:2, :2], desired_acov2[2], rtol=1e-06, atol=1e-06)
    acov2 = res.smoothed_state_autocovariance(2, start=8, end=9).transpose(2, 0, 1)
    assert_allclose(acov2[:, :2, :2], desired_acov2[8:9], rtol=1e-06, atol=1e-06)


@pytest.mark.parametrize("missing", ["all", "partial", "mixed", None])
@pytest.mark.parametrize("filter_univariate", [True, False])
@pytest.mark.parametrize("tvp", [True, False])
def test_smoothed_state_autocovariances_forwards(missing, filter_univariate, tvp):
    """
    Test for Cov(t, t + lag)
    """
    mod_oos, res_oos = get_acov_model(missing, filter_univariate, tvp, oos=3)
    names = [
        "obs_intercept",
        "design",
        "obs_cov",
        "transition",
        "selection",
        "state_cov",
    ]
    if not tvp:
        mod, res = get_acov_model(
            missing, filter_univariate, tvp, params=mod_oos.start_params
        )
    else:
        mod, _ = get_acov_model(missing, filter_univariate, tvp)
        for name in names:
            mod[name] = mod_oos[name, ..., :-3]
        res = mod.ssm.smooth()
    extend_kwargs1 = {}
    extend_kwargs2 = {}
    if tvp:
        keys = [
            "obs_intercept",
            "design",
            "obs_cov",
            "transition",
            "selection",
            "state_cov",
        ]
        for key in keys:
            extend_kwargs1[key] = mod_oos[key, ..., -3:-2]
            extend_kwargs2[key] = mod_oos[key, ..., -3:-1]
    assert_allclose(res_oos.llf, res.llf)
    cov = res.smoothed_state_cov.transpose(2, 0, 1)
    desired_acov1 = cov[:, 2:4, :2]
    desired_acov2 = cov[:, 4:6, :2]
    desired_acov3 = cov[:, 6:8, :2]
    oos_cov = np.concatenate(
        (res_oos.smoothed_state_cov, res_oos.predicted_state_cov[..., -1:]), axis=2
    ).transpose(2, 0, 1)
    acov1 = res.smoothed_state_autocovariance(-1).transpose(2, 0, 1)
    assert_allclose(acov1[:-1, :2, :2], desired_acov1[1:])
    assert_allclose(acov1[-2:, :2, :2], oos_cov[-5:-3, 2:4, :2])
    acov2 = res.smoothed_state_autocovariance(
        -2, extend_kwargs=extend_kwargs1
    ).transpose(2, 0, 1)
    assert_allclose(acov2[:-2, :2, :2], desired_acov2[2:])
    assert_allclose(acov2[-2:, :2, :2], oos_cov[-4:-2, 4:6, :2])
    acov3 = res.smoothed_state_autocovariance(
        -3, extend_kwargs=extend_kwargs2
    ).transpose(2, 0, 1)
    assert_allclose(acov3[:-3, :2, :2], desired_acov3[3:], atol=1e-10, rtol=1e-06)
    assert_allclose(acov3[-3:, :2, :2], oos_cov[-4:-1, 6:8, :2], atol=1e-10, rtol=1e-06)
    acov1 = res.smoothed_state_autocovariance(
        -1, t=mod.nobs, extend_kwargs=extend_kwargs1
    )
    assert_allclose(acov1[:2, :2], oos_cov[-3, 2:4, :2], atol=1e-10, rtol=1e-06)
    acov1 = res.smoothed_state_autocovariance(-1, t=0)
    assert_allclose(acov1[:2, :2], desired_acov1[0 + 1], atol=1e-10, rtol=1e-06)
    acov1 = res.smoothed_state_autocovariance(-1, start=8, end=9).transpose(2, 0, 1)
    assert_allclose(
        acov1[:, :2, :2], desired_acov1[8 + 1 : 9 + 1], atol=1e-10, rtol=1e-06
    )
    acov2 = res.smoothed_state_autocovariance(
        -2, t=mod.nobs, extend_kwargs=extend_kwargs2
    )
    assert_allclose(acov2[:2, :2], oos_cov[-2, 4:6, :2], atol=1e-10, rtol=1e-06)
    acov2 = res.smoothed_state_autocovariance(
        -2, t=mod.nobs - 1, extend_kwargs=extend_kwargs1
    )
    assert_allclose(acov2[:2, :2], oos_cov[-3, 4:6, :2], atol=1e-10, rtol=1e-06)
    acov2 = res.smoothed_state_autocovariance(-2, t=0)
    assert_allclose(acov2[:2, :2], desired_acov2[0 + 2], atol=1e-10, rtol=1e-06)
    acov2 = res.smoothed_state_autocovariance(-2, start=8, end=9).transpose(2, 0, 1)
    assert_allclose(
        acov2[:, :2, :2], desired_acov2[8 + 2 : 9 + 2], atol=1e-10, rtol=1e-06
    )


@pytest.mark.parametrize("missing", ["all", "partial", "mixed", None])
@pytest.mark.parametrize("filter_univariate", [True, False])
@pytest.mark.parametrize("tvp", [True, False])
def test_smoothed_state_autocovariances_forwards_oos(missing, filter_univariate, tvp):
    mod_oos, res_oos = get_acov_model(missing, filter_univariate, tvp, oos=5)
    names = [
        "obs_intercept",
        "design",
        "obs_cov",
        "transition",
        "selection",
        "state_cov",
    ]
    if not tvp:
        mod, res = get_acov_model(
            missing, filter_univariate, tvp, params=mod_oos.start_params
        )
    else:
        mod, _ = get_acov_model(missing, filter_univariate, tvp)
        for name in names:
            mod[name] = mod_oos[name, ..., :-5]
        res = mod.ssm.smooth()
    assert_allclose(res_oos.llf, res.llf)
    cov = np.concatenate(
        (res_oos.smoothed_state_cov, res_oos.predicted_state_cov[..., -1:]), axis=2
    ).transpose(2, 0, 1)
    desired_acov1 = cov[:, 2:4, :2]
    desired_acov2 = cov[:, 4:6, :2]
    desired_acov3 = cov[:, 6:8, :2]
    extend_kwargs = {}
    if tvp:
        extend_kwargs = {
            "obs_intercept": mod_oos["obs_intercept", ..., -5:],
            "design": mod_oos["design", ..., -5:],
            "obs_cov": mod_oos["obs_cov", ..., -5:],
            "transition": mod_oos["transition", ..., -5:],
            "selection": mod_oos["selection", ..., -5:],
            "state_cov": mod_oos["state_cov", ..., -5:],
        }
    acov1 = res.smoothed_state_autocovariance(
        -1, end=mod_oos.nobs, extend_kwargs=extend_kwargs
    ).transpose(2, 0, 1)
    assert_equal(acov1.shape, (mod_oos.nobs, mod.k_states, mod.k_states))
    assert_allclose(acov1[:, :2, :2], desired_acov1[1:])
    acov2 = res.smoothed_state_autocovariance(
        -2, end=mod_oos.nobs - 1, extend_kwargs=extend_kwargs
    ).transpose(2, 0, 1)
    assert_equal(acov2.shape, (mod_oos.nobs - 1, mod.k_states, mod.k_states))
    assert_allclose(acov2[:, :2, :2], desired_acov2[2:])
    acov3 = res.smoothed_state_autocovariance(
        -3, end=mod_oos.nobs - 2, extend_kwargs=extend_kwargs
    ).transpose(2, 0, 1)
    assert_equal(acov3.shape, (mod_oos.nobs - 2, mod.k_states, mod.k_states))
    assert_allclose(acov3[:, :2, :2], desired_acov3[3:])


@pytest.mark.parametrize("missing", ["all", "partial", "mixed", None])
@pytest.mark.parametrize("filter_univariate", [True, False])
@pytest.mark.parametrize("tvp", [True, False])
def test_smoothed_state_autocovariances_backwards_oos(missing, filter_univariate, tvp):
    mod_oos, res_oos = get_acov_model(missing, filter_univariate, tvp, oos=5)
    names = [
        "obs_intercept",
        "design",
        "obs_cov",
        "transition",
        "selection",
        "state_cov",
    ]
    if not tvp:
        mod, res = get_acov_model(
            missing, filter_univariate, tvp, params=mod_oos.start_params
        )
    else:
        mod, _ = get_acov_model(missing, filter_univariate, tvp)
        for name in names:
            mod[name] = mod_oos[name, ..., :-5]
        res = mod.ssm.smooth()
    assert_allclose(res_oos.llf, res.llf)
    cov = np.concatenate(
        (res_oos.smoothed_state_cov, res_oos.predicted_state_cov[..., -1:]), axis=2
    ).transpose(2, 0, 1)
    desired_acov1 = cov[:, :2, 2:4]
    desired_acov2 = cov[:, :2, 4:6]
    desired_acov3 = cov[:, :2, 6:8]
    end = mod_oos.nobs + 1
    extend_kwargs = {}
    if tvp:
        extend_kwargs = {
            "obs_intercept": mod_oos["obs_intercept", ..., -5:],
            "design": mod_oos["design", ..., -5:],
            "obs_cov": mod_oos["obs_cov", ..., -5:],
            "transition": mod_oos["transition", ..., -5:],
            "selection": mod_oos["selection", ..., -5:],
            "state_cov": mod_oos["state_cov", ..., -5:],
        }
    acov1 = res.smoothed_state_autocovariance(
        1, end=end, extend_kwargs=extend_kwargs
    ).transpose(2, 0, 1)
    assert_equal(acov1.shape, (mod_oos.nobs + 1, mod.k_states, mod.k_states))
    assert_allclose(acov1[1:, :2, :2], desired_acov1[1:])
    assert_equal(acov1[:1], np.nan)
    acov2 = res.smoothed_state_autocovariance(
        2, end=end, extend_kwargs=extend_kwargs
    ).transpose(2, 0, 1)
    assert_allclose(acov2[2:, :2, :2], desired_acov2[2:])
    assert_equal(acov2[:2], np.nan)
    acov3 = res.smoothed_state_autocovariance(
        3, end=end, extend_kwargs=extend_kwargs
    ).transpose(2, 0, 1)
    assert_allclose(acov3[3:, :2, :2], desired_acov3[3:])
    assert_equal(acov3[:3], np.nan)


def test_smoothed_state_autocovariances_invalid():
    _, res = get_acov_model(missing=False, filter_univariate=False, tvp=False)
    with pytest.raises(ValueError, match="Cannot specify both `t`"):
        res.smoothed_state_autocovariance(1, t=1, start=1)
    with pytest.raises(ValueError, match="Negative `t`"):
        res.smoothed_state_autocovariance(1, t=-1)
    with pytest.raises(ValueError, match="Negative `t`"):
        res.smoothed_state_autocovariance(1, start=-1)
    with pytest.raises(ValueError, match="Negative `t`"):
        res.smoothed_state_autocovariance(1, end=-1)
    with pytest.raises(ValueError, match="`end` must be after `start`"):
        res.smoothed_state_autocovariance(1, start=5, end=4)


@pytest.mark.parametrize("missing", ["all", "partial", "mixed", None])
@pytest.mark.parametrize("filter_univariate", [True, False])
@pytest.mark.parametrize("tvp", [True, False])
def test_news_basic(missing, filter_univariate, tvp):
    mod, res = get_acov_model(missing, filter_univariate, tvp)
    params = [] if tvp else mod.start_params
    append = np.zeros((10, 2)) * np.nan
    append[0] = [0.1, -0.2]
    endog2 = np.concatenate((mod.endog, append), axis=0)
    mod2 = mod.clone(endog2)
    res2 = mod2.smooth(params, return_ssm=True)
    endog3 = endog2.copy()
    endog3[-10:] = np.nan
    mod3 = mod2.clone(endog3)
    res3 = mod3.smooth(params, return_ssm=True)
    for t in [0, 1, 150, mod.nobs - 1, mod.nobs, mod.nobs + 1, mod.nobs + 9]:
        out = res2.news(res, t=t)
        desired = res2.smoothed_forecasts[..., t] - res3.smoothed_forecasts[..., t]
        assert_allclose(out.update_impacts, desired, atol=1e-14)
        assert_equal(out.revision_impacts, None)
        out = res2.news(res, start=t, end=t + 1)
        assert_allclose(out.update_impacts, desired[None, ...], atol=1e-14)


@pytest.mark.parametrize("missing", ["all", "partial", "mixed", None])
@pytest.mark.parametrize("filter_univariate", [True, False])
@pytest.mark.parametrize("tvp", [True, False])
def test_news_revisions(missing, filter_univariate, tvp):
    mod, res = get_acov_model(missing, filter_univariate, tvp, oos=10)
    params = [] if tvp else mod.start_params
    endog2 = mod.endog.copy()
    endog2[-11] = [0.0, 0.0]
    endog2[-10] = [-0.3, -0.4]
    mod2 = mod.clone(endog2)
    res2 = mod2.smooth(params, return_ssm=True)
    nobs = mod.nobs - 10
    for t in [0, 1, 150, nobs - 1, nobs, nobs + 1, nobs + 9]:
        out = res2.news(res, t=t)
        desired = (
            res2.smoothed_forecasts[..., t]
            - out.revision_results.smoothed_forecasts[..., t]
        )
        assert_allclose(out.update_impacts, desired, atol=1e-10)
        desired = (
            out.revision_results.smoothed_forecasts[..., t]
            - res.smoothed_forecasts[..., t]
        )
        assert_allclose(out.revision_impacts, desired, atol=1e-10)


@pytest.mark.parametrize("missing", ["all", "partial", "mixed", None])
@pytest.mark.parametrize("filter_univariate", [True, False])
@pytest.mark.parametrize("tvp", [True, False])
def test_news_invalid(missing, filter_univariate, tvp):
    error_ss = "This results object has %s and so it does not appear to by an extension of `previous`. Can only compute the news by comparing this results set to previous results objects."
    mod, res = get_acov_model(missing, filter_univariate, tvp, oos=1)
    params = [] if tvp else mod.start_params
    endog2 = mod.endog.copy()
    endog2[-1] = [0.2, 0.5]
    mod2 = mod.clone(endog2)
    res2_filtered = mod2.filter(params, return_ssm=True)
    res2_smoothed = mod2.smooth(params, return_ssm=True)
    res2_smoothed.news(res, t=mod.nobs - 1)
    msg = "Cannot compute news without having applied the Kalman smoother first."
    with pytest.raises(ValueError, match=msg):
        res2_filtered.news(res, t=mod.nobs - 1)
    if tvp:
        msg = "Cannot compute the impacts of news on periods outside of the sample in time-varying models."
        with pytest.raises(RuntimeError, match=msg):
            res2_smoothed.news(res, t=mod.nobs + 2)
    mod, res = get_acov_model(missing, filter_univariate, tvp)
    params = [] if tvp else mod.start_params
    endog2 = mod.endog.copy()[: mod.nobs - 1]
    mod2 = mod.clone(endog2)
    res2 = mod2.smooth(params, return_ssm=True)
    msg = error_ss % "fewer observations than `previous`"
    with pytest.raises(ValueError, match=msg):
        res2.news(res, t=mod.nobs - 1)
    mod2 = sarimax.SARIMAX(np.zeros(mod.nobs))
    res2 = mod2.smooth([0.5, 1.0], return_ssm=True)
    msg = error_ss % "different state space dimensions than `previous`"
    with pytest.raises(ValueError, match=msg):
        res2.news(res, t=mod.nobs - 1)
    mod2, res2 = get_acov_model(missing, filter_univariate, not tvp, oos=1)
    if tvp:
        msg = "time-invariant design while `previous` does not"
    else:
        msg = "time-varying design while `previous` does not"
    with pytest.raises(ValueError, match=msg):
        res2.news(res, t=mod.nobs - 1)
