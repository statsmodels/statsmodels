r"""
Tests for smoothing and estimation of unobserved states and disturbances

- Predicted states: :math:`E(\alpha_t | Y_{t-1})`
- Filtered states: :math:`E(\alpha_t | Y_t)`
- Smoothed states: :math:`E(\alpha_t | Y_n)`
- Smoothed disturbances :math:`E(\varepsilon_t | Y_n), E(\eta_t | Y_n)`

Tested against R (FKF, KalmanRun / KalmanSmooth), Stata (sspace), and
MATLAB (ssm toolbox)

Author: Chad Fulton
License: Simplified-BSD
"""
import contextlib
import itertools
from pathlib import Path
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pandas as pd
import pytest
from scipy import linalg

from statsmodels import datasets
from statsmodels.tools.sm_exceptions import OutputWarning
from statsmodels.tsa.statespace import kalman_smoother, mlemodel, sarimax, varmax
from statsmodels.tsa.statespace.kalman_filter import (
    FILTER_UNIVARIATE,
    INVERT_UNIVARIATE,
    SOLVE_LU,
)
from statsmodels.tsa.statespace.kalman_smoother import (
    SMOOTH_ALTERNATIVE,
    SMOOTH_CLASSICAL,
    SMOOTH_UNIVARIATE,
    SMOOTHER_DISTURBANCE,
    SMOOTHER_STATE,
    SMOOTHER_STATE_COV,
    KalmanSmoother,
    SmootherResults,
)
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS

current_path = Path(__file__).resolve().parent

# The diagonalized observation covariance must be expected where it is emitted
pytestmark = pytest.mark.filterwarnings(
    "error:The univariate filtering approach diagonalized"
    ":statsmodels.tools.sm_exceptions.OutputWarning"
)


class TestStatesAR3:
    @classmethod
    def setup_class(cls, *args, alternate_timing=False, **kwargs):
        # Dataset / Stata comparison
        path = Path(current_path).joinpath("results", "results_wpi1_ar3_stata.csv")
        cls.stata = pd.read_csv(path)
        cls.stata.index = pd.date_range(start="1960-01-01", periods=124, freq="QS")
        # Matlab comparison
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

        # Parameters from from Stata's sspace MLE estimation
        params = np.r_[0.5270715, 0.0952613, 0.2580355, 0.5307459]
        cls.results = cls.model.smooth(params, cov_type="none")

        # Calculate the determinant of the covariance matrices (for easy
        # comparison to other languages without having to store 2-dim arrays)
        cls.results.det_predicted_state_cov = np.zeros((1, cls.model.nobs))
        cls.results.det_smoothed_state_cov = np.zeros((1, cls.model.nobs))
        for i in range(cls.model.nobs):
            cls.results.det_predicted_state_cov[0, i] = np.linalg.det(
                cls.results.filter_results.predicted_state_cov[:, :, i]
            )
            cls.results.det_smoothed_state_cov[0, i] = np.linalg.det(
                cls.results.smoother_results.smoothed_state_cov[:, :, i]
            )

        # Perform simulation smoothing
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
        # Initialization issues can change the first few smoothed states
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
        # Dataset
        path = Path(current_path).joinpath("results", "results_wpi1_ar3_stata.csv")
        cls.stata = pd.read_csv(path)
        cls.stata.index = pd.date_range(start="1960-01-01", periods=124, freq="QS")
        # Matlab comparison
        path = Path(current_path).joinpath("results", "results_wpi1_missing_ar3_matlab_ssm.csv")
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
        # KFAS comparison
        path = Path(current_path).joinpath("results", "results_smoothing3_R.csv")
        cls.R_ssm = pd.read_csv(path)

        # Create missing observations
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

        # Parameters from from Stata's sspace MLE estimation
        params = np.r_[0.5270715, 0.0952613, 0.2580355, 0.5307459]
        cls.results = cls.model.smooth(params, return_ssm=True)

        # Calculate the determinant of the covariance matrices (for easy
        # comparison to other languages without having to store 2-dim arrays)
        cls.results.det_predicted_state_cov = np.zeros((1, cls.model.nobs))
        cls.results.det_smoothed_state_cov = np.zeros((1, cls.model.nobs))
        for i in range(cls.model.nobs):
            cls.results.det_predicted_state_cov[0, i] = np.linalg.det(
                cls.results.predicted_state_cov[:, :, i]
            )
            cls.results.det_smoothed_state_cov[0, i] = np.linalg.det(
                cls.results.smoothed_state_cov[:, :, i]
            )
        # store smoothed forecasts for thread-safety
        cls.smoothed_forecasts = cls.results.smoothed_forecasts

        # Perform simulation smoothing
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

    # There is a discrepancy between MATLAB ssm toolbox and
    # statsmodels.tsa.statespace on the following variables in the case of
    # missing data. Tests against the R package KFAS confirm our results

    def test_smoothed_state_disturbance(self):
        # See note above about why this assertion is invalid
        # assert_almost_equal(
        #     self.results.smoothed_state_disturbance.T,
        #     self.matlab_ssm[['eta']], 4
        # )
        assert_almost_equal(
            self.results.smoothed_state_disturbance.T, self.R_ssm[["etahat"]], 9
        )

    def test_smoothed_state_disturbance_cov(self):
        # See note above about why this assertion is invalid
        # assert_almost_equal(
        #     self.results.smoothed_state_disturbance_cov[0].T,
        #     self.matlab_ssm[['etavar']], 4
        # )
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
    measurement equation (e.g., forecasts, etc.). In this case, although the
    model is multivariate, each of the series is truly independent so the
    values will be the same regardless of whether the univariate approach
    is used or not.
    """

    @classmethod
    def setup_class(cls, **kwargs):
        # Results
        path = Path(current_path).joinpath("results", "results_smoothing_R.csv")
        cls.desired = pd.read_csv(path)

        # Data
        dta = datasets.macrodata.load_pandas().data
        dta.index = pd.date_range(start="1959-01-01", end="2009-7-01", freq="QS")
        obs = dta[["realgdp", "realcons", "realinv"]].diff().iloc[1:]
        obs.iloc[0:50, 0] = np.nan
        obs.iloc[19:70, 1] = np.nan
        obs.iloc[39:90, 2] = np.nan
        obs.iloc[119:130, 0] = np.nan
        obs.iloc[119:130, 2] = np.nan

        # Create the model
        mod = mlemodel.MLEModel(obs, k_states=3, k_posdef=3, **kwargs)
        mod["design"] = np.eye(3)
        mod["obs_cov"] = np.eye(3)
        mod["transition"] = np.eye(3)
        mod["selection"] = np.eye(3)
        mod["state_cov"] = np.eye(3)
        mod.initialize_approximate_diffuse(1e6)
        cls.model = mod
        cls.results = mod.smooth([], return_ssm=True)

        # Calculate the determinant of the covariance matrices (for easy
        # comparison to other languages without having to store 2-dim arrays)
        cls.results.det_scaled_smoothed_estimator_cov = np.zeros((1, cls.model.nobs))
        cls.results.det_predicted_state_cov = np.zeros((1, cls.model.nobs))
        cls.results.det_smoothed_state_cov = np.zeros((1, cls.model.nobs))
        cls.results.det_smoothed_state_disturbance_cov = np.zeros((1, cls.model.nobs))
        # store smoothed forecasts for thread-safety
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
    measurement equation (e.g., forecasts, etc.). In this case, although the
    model is multivariate, each of the series is truly independent so the
    values will be the same regardless of whether the univariate approach is
    used or not.
    """

    @classmethod
    def setup_class(cls, *args, **kwargs):
        # Results
        path = Path(current_path).joinpath("results", "results_smoothing2_R.csv")
        cls.desired = pd.read_csv(path)

        # Data
        dta = datasets.macrodata.load_pandas().data
        dta.index = pd.date_range(start="1959-01-01", end="2009-7-01", freq="QS")
        obs = np.log(dta[["realgdp", "realcons", "realinv"]]).diff().iloc[1:]

        # Create the model
        mod = mlemodel.MLEModel(obs, k_states=3, k_posdef=3, **kwargs)
        mod["design"] = np.eye(3)
        mod["obs_cov"] = np.array(
            [
                [0.0000640649, 0.0, 0.0],
                [0.0, 0.0000572802, 0.0],
                [0.0, 0.0, 0.0017088585],
            ]
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
                [0.0000640649, 0.0000388496, 0.0002148769],
                [0.0000388496, 0.0000572802, 0.000001555],
                [0.0002148769, 0.000001555, 0.0017088585],
            ]
        )
        mod.initialize_approximate_diffuse(1e6)
        cls.model = mod
        cls.results = mod.smooth([], return_ssm=True)

        # Calculate the determinant of the covariance matrices (for easy
        # comparison to other languages without having to store 2-dim arrays)
        cls.results.det_scaled_smoothed_estimator_cov = np.zeros((1, cls.model.nobs))
        cls.results.det_predicted_state_cov = np.zeros((1, cls.model.nobs))
        cls.results.det_smoothed_state_cov = np.zeros((1, cls.model.nobs))
        cls.results.det_smoothed_state_disturbance_cov = np.zeros((1, cls.model.nobs))
        # store smoothed forecasts for thread-safety
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
            atol=1e-4,
        )

    def test_scaled_smoothed_estimator_cov(self):
        # Last obs is zero, so exclude it
        assert_allclose(
            np.log(self.results.det_scaled_smoothed_estimator_cov.T[:-1]),
            np.log(self.desired[["detN"]][:-1]),
            atol=1e-6,
        )

    def test_forecasts(self):
        assert_allclose(
            self.results.forecasts.T, self.desired[["m1", "m2", "m3"]], atol=1e-6
        )

    def test_forecasts_error(self):
        assert_allclose(
            self.results.forecasts_error.T[:, 0], self.desired["v1"], atol=1e-6
        )

    def test_forecasts_error_cov(self):
        assert_allclose(
            self.results.forecasts_error_cov.diagonal()[:, 0],
            self.desired["F1"],
            atol=1e-6,
        )

    def test_predicted_states(self):
        assert_allclose(
            self.results.predicted_state[:, 1:].T,
            self.desired[["a1", "a2", "a3"]],
            atol=1e-6,
        )

    def test_predicted_states_cov(self):
        assert_allclose(
            self.results.det_predicted_state_cov.T, self.desired[["detP"]], atol=1e-16
        )

    def test_smoothed_states(self):
        assert_allclose(
            self.results.smoothed_state.T,
            self.desired[["alphahat1", "alphahat2", "alphahat3"]],
            atol=1e-6,
        )

    def test_smoothed_states_cov(self):
        assert_allclose(
            self.results.det_smoothed_state_cov.T, self.desired[["detV"]], atol=1e-16
        )

    def test_smoothed_forecasts(self):
        assert_allclose(
            self.smoothed_forecasts.T,
            self.desired[["muhat1", "muhat2", "muhat3"]],
            atol=1e-6,
        )

    def test_smoothed_state_disturbance(self):
        assert_allclose(
            self.results.smoothed_state_disturbance.T,
            self.desired[["etahat1", "etahat2", "etahat3"]],
            atol=1e-6,
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
            atol=1e-6,
        )

    def test_smoothed_measurement_disturbance_cov(self):
        assert_allclose(
            self.results.smoothed_measurement_disturbance_cov.diagonal(),
            self.desired[["Veps1", "Veps2", "Veps3"]],
            atol=1e-6,
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
    measurement equation (e.g., forecasts, etc.). In this case, although the
    model is multivariate, each of the series is truly independent so the
    values will be the same regardless of whether the univariate approach is
    used or not.
    """

    @classmethod
    def setup_class(cls, *args, **kwargs):
        # Results
        path = Path(current_path).joinpath("results", "results_smoothing2_R.csv")
        cls.desired = pd.read_csv(path)

        # Data
        dta = datasets.macrodata.load_pandas().data
        dta.index = pd.date_range(start="1959-01-01", end="2009-7-01", freq="QS")
        obs = np.log(dta[["realgdp", "realcons", "realinv"]]).diff().iloc[1:]

        # Create the model
        mod = mlemodel.MLEModel(obs, k_states=3, k_posdef=3, **kwargs)
        mod.ssm.filter_univariate = True
        mod["design"] = np.eye(3)
        mod["obs_cov"] = np.array(
            [
                [0.0000640649, 0.0, 0.0],
                [0.0, 0.0000572802, 0.0],
                [0.0, 0.0, 0.0017088585],
            ]
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
                [0.0000640649, 0.0000388496, 0.0002148769],
                [0.0000388496, 0.0000572802, 0.000001555],
                [0.0002148769, 0.000001555, 0.0017088585],
            ]
        )
        mod.initialize_approximate_diffuse(1e6)
        cls.model = mod
        cls.results = mod.smooth([], return_ssm=True)

        # Calculate the determinant of the covariance matrices (for easy
        # comparison to other languages without having to store 2-dim arrays)
        cls.results.det_scaled_smoothed_estimator_cov = np.zeros((1, cls.model.nobs))
        cls.results.det_predicted_state_cov = np.zeros((1, cls.model.nobs))
        cls.results.det_smoothed_state_cov = np.zeros((1, cls.model.nobs))
        cls.results.det_smoothed_state_disturbance_cov = np.zeros((1, cls.model.nobs))
        # store smoothed forecasts for thread-safety
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
            atol=1e-4,
        )

    def test_scaled_smoothed_estimator_cov(self):
        # Last obs is zero, so exclude it
        assert_allclose(
            np.log(self.results.det_scaled_smoothed_estimator_cov.T[:-1]),
            np.log(self.desired[["detN"]][:-1]),
        )

    def test_forecasts(self):
        assert_allclose(self.results.forecasts.T[:, 0], self.desired["m1"], atol=1e-6)

    def test_forecasts_error(self):
        assert_allclose(
            self.results.forecasts_error.T, self.desired[["v1", "v2", "v3"]], atol=1e-6
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
            atol=1e-8,
        )

    def test_predicted_states_cov(self):
        assert_allclose(
            self.results.det_predicted_state_cov.T, self.desired[["detP"]], atol=1e-18
        )

    def test_smoothed_states(self):
        assert_allclose(
            self.results.smoothed_state.T,
            self.desired[["alphahat1", "alphahat2", "alphahat3"]],
            atol=1e-6,
        )

    def test_smoothed_states_cov(self):
        assert_allclose(
            self.results.det_smoothed_state_cov.T, self.desired[["detV"]], atol=1e-18
        )

    def test_smoothed_forecasts(self):
        assert_allclose(
            self.smoothed_forecasts.T,
            self.desired[["muhat1", "muhat2", "muhat3"]],
            atol=1e-6,
        )

    def test_smoothed_state_disturbance(self):
        assert_allclose(
            self.results.smoothed_state_disturbance.T,
            self.desired[["etahat1", "etahat2", "etahat3"]],
            atol=1e-6,
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
            atol=1e-6,
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
        # Data
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

        # Create the model with typical state space
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
        mod.initialize_approximate_diffuse(1e6)
        cls.model = mod
        cls.results = mod.smooth([], return_ssm=True)
        # store smoothed forecasts for thread-safety
        cls.smoothed_forecasts = cls.results.smoothed_forecasts

        # Create the model with augmented state space
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

        mod.initialize_approximate_diffuse(1e6)
        cls.augmented_model = mod
        cls.augmented_results = mod.smooth([], return_ssm=True)

    def test_smoothed_state_autocov(self):
        # Cov(\alpha_{t+1}, \alpha_t)
        # Initialization makes these two methods slightly different for the
        # first few observations
        assert_allclose(
            self.results.smoothed_state_autocov[:, :, 0:5],
            self.augmented_results.smoothed_state_cov[:3, 3:, 1:6],
            atol=1e-4,
        )
        assert_allclose(
            self.results.smoothed_state_autocov[:, :, 5:-1],
            self.augmented_results.smoothed_state_cov[:3, 3:, 6:],
            atol=1e-7,
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


def expect_diagonalized(expected):
    # The univariate filter diagonalizes the non-diagonal obs_cov of TVSS
    if expected:
        return pytest.warns(
            OutputWarning, match="The univariate filtering approach diagonalized"
        )
    return contextlib.nullcontext()


class TVSSWithLags(TVSS):
    def __init__(self, endog):
        # TVSS has 2 states, here we will add in 3 lags of those
        super().__init__(endog, _k_states=8)
        self["transition", 2:, :6] = np.eye(6)[..., None]
        # Can't use exact diffuse filtering
        self.ssm.initialize_approximate_diffuse(1e-4)


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
        with expect_diagonalized(filter_univariate):
            res = mod.smooth([], return_ssm=return_ssm)

    return mod, res


@pytest.mark.parametrize("missing", ["all", "partial", "mixed", None])
@pytest.mark.parametrize("filter_univariate", [True, False])
@pytest.mark.parametrize("tvp", [True, False])
def test_smoothed_state_autocovariances_backwards(missing, filter_univariate, tvp):
    r"""
    Test for Cov(t, t - lag)
    """
    _, res = get_acov_model(missing, filter_univariate, tvp)

    cov = res.smoothed_state_cov.transpose(2, 0, 1)
    desired_acov1 = cov[:, :2, 2:4]
    desired_acov2 = cov[:, :2, 4:6]
    desired_acov3 = cov[:, :2, 6:8]

    # Test all "backward" autocovariances: Cov(t, t-lag)
    acov1 = res.smoothed_state_autocovariance(1).transpose(2, 0, 1)
    assert_allclose(acov1[1:, :2, :2], desired_acov1[1:], rtol=1e-6, atol=1e-6)
    assert_equal(acov1[:1], np.nan)

    acov2 = res.smoothed_state_autocovariance(2).transpose(2, 0, 1)
    assert_allclose(acov2[2:, :2, :2], desired_acov2[2:], rtol=1e-6, atol=1e-6)
    assert_equal(acov2[:2], np.nan)

    acov3 = res.smoothed_state_autocovariance(3).transpose(2, 0, 1)
    assert_allclose(acov3[3:, :2, :2], desired_acov3[3:], rtol=1e-6, atol=1e-6)
    assert_equal(acov3[:3], np.nan)

    # Test for specific autocovariances
    acov1 = res.smoothed_state_autocovariance(1, t=0)
    assert_allclose(acov1, np.nan)
    acov1 = res.smoothed_state_autocovariance(1, t=1)
    assert_allclose(acov1[:2, :2], desired_acov1[1], rtol=1e-6, atol=1e-6)
    acov1 = res.smoothed_state_autocovariance(1, start=8, end=9).transpose(2, 0, 1)
    assert_allclose(acov1[:, :2, :2], desired_acov1[8:9], rtol=1e-6, atol=1e-6)

    acov2 = res.smoothed_state_autocovariance(2, t=0)
    assert_allclose(acov2, np.nan)
    acov2 = res.smoothed_state_autocovariance(2, t=1)
    assert_allclose(acov2, np.nan)
    acov2 = res.smoothed_state_autocovariance(2, t=2)
    assert_allclose(acov2[:2, :2], desired_acov2[2], rtol=1e-6, atol=1e-6)
    acov2 = res.smoothed_state_autocovariance(2, start=8, end=9).transpose(2, 0, 1)
    assert_allclose(acov2[:, :2, :2], desired_acov2[8:9], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("missing", ["all", "partial", "mixed", None])
@pytest.mark.parametrize("filter_univariate", [True, False])
@pytest.mark.parametrize("tvp", [True, False])
def test_smoothed_state_autocovariances_forwards(missing, filter_univariate, tvp):
    r"""
    Test for Cov(t, t + lag)
    """
    # Out-of-sample model
    # Note: in TVP case, we need to first generate the larger model, and then
    # create the smaller model with the system matrices from the larger model
    # (otherwise they will be different, since the matrices are randomly
    # generated)
    mod_oos, res_oos = get_acov_model(missing, filter_univariate, tvp, oos=3)

    # Basic model
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
        with expect_diagonalized(filter_univariate):
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

    # Test all "forwards" autocovariances: Cov(t, t+lag)
    # For Cov(t, t+lag), the first out-of-sample forward covariance,
    # Cov(T, T+1), is already available, so we dno't need extend kwaargs
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
    assert_allclose(acov3[:-3, :2, :2], desired_acov3[3:], atol=1e-10, rtol=1e-6)
    assert_allclose(acov3[-3:, :2, :2], oos_cov[-4:-1, 6:8, :2], atol=1e-10, rtol=1e-6)

    # Test for specific autocovariances
    acov1 = res.smoothed_state_autocovariance(
        -1, t=mod.nobs, extend_kwargs=extend_kwargs1
    )
    assert_allclose(acov1[:2, :2], oos_cov[-3, 2:4, :2], atol=1e-10, rtol=1e-6)
    acov1 = res.smoothed_state_autocovariance(-1, t=0)
    assert_allclose(acov1[:2, :2], desired_acov1[0 + 1], atol=1e-10, rtol=1e-6)
    acov1 = res.smoothed_state_autocovariance(-1, start=8, end=9).transpose(2, 0, 1)
    assert_allclose(
        acov1[:, :2, :2], desired_acov1[8 + 1 : 9 + 1], atol=1e-10, rtol=1e-6
    )

    acov2 = res.smoothed_state_autocovariance(
        -2, t=mod.nobs, extend_kwargs=extend_kwargs2
    )
    assert_allclose(acov2[:2, :2], oos_cov[-2, 4:6, :2], atol=1e-10, rtol=1e-6)
    acov2 = res.smoothed_state_autocovariance(
        -2, t=mod.nobs - 1, extend_kwargs=extend_kwargs1
    )
    assert_allclose(acov2[:2, :2], oos_cov[-3, 4:6, :2], atol=1e-10, rtol=1e-6)
    acov2 = res.smoothed_state_autocovariance(-2, t=0)
    assert_allclose(acov2[:2, :2], desired_acov2[0 + 2], atol=1e-10, rtol=1e-6)
    acov2 = res.smoothed_state_autocovariance(-2, start=8, end=9).transpose(2, 0, 1)
    assert_allclose(
        acov2[:, :2, :2], desired_acov2[8 + 2 : 9 + 2], atol=1e-10, rtol=1e-6
    )


@pytest.mark.parametrize("missing", ["all", "partial", "mixed", None])
@pytest.mark.parametrize("filter_univariate", [True, False])
@pytest.mark.parametrize("tvp", [True, False])
def test_smoothed_state_autocovariances_forwards_oos(missing, filter_univariate, tvp):
    # Out-of-sample model
    # Note: in TVP case, we need to first generate the larger model, and then
    # create the smaller model with the system matrices from the larger model
    # (otherwise they will be different, since the matrices are randomly
    # generated)
    mod_oos, res_oos = get_acov_model(missing, filter_univariate, tvp, oos=5)

    # Basic model
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
        with expect_diagonalized(filter_univariate):
            res = mod.ssm.smooth()

    assert_allclose(res_oos.llf, res.llf)

    cov = np.concatenate(
        (res_oos.smoothed_state_cov, res_oos.predicted_state_cov[..., -1:]), axis=2
    ).transpose(2, 0, 1)
    desired_acov1 = cov[:, 2:4, :2]
    desired_acov2 = cov[:, 4:6, :2]
    desired_acov3 = cov[:, 6:8, :2]

    # Test all "forwards" autocovariances: Cov(t, t+lag)
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

    # Note: we can compute up to Cov(mod_oos.nobs, mod_oos.nobs + 1) using
    # a model that has state space matrices defined up to mod_oos.nobs. Since
    # mod_oos.nobs = mod.nobs + 5, we need to pass in 5 additional time points,
    # and that is what extend_kwargs, above, does.
    acov1 = res.smoothed_state_autocovariance(
        -1, end=mod_oos.nobs, extend_kwargs=extend_kwargs
    ).transpose(2, 0, 1)
    assert_equal(acov1.shape, (mod_oos.nobs, mod.k_states, mod.k_states))
    assert_allclose(acov1[:, :2, :2], desired_acov1[1:], rtol=1e-6)

    # Note: now we can compute up to Cov(mod_oos.nobs - 1, mod_oos.nobs + 1)
    # using a model that has state space matrices defined up to mod_oos.nobs.
    # We still need to pass in 5 additional time points for the extend kwargs.
    # This is why we have end = mod_oos.nobs - 1, because this function returns
    # values through Cov(end, end + 2). Because start=0 (the default), we
    # will have values for Cov(0, 2), Cov(1, 3), ...,
    # Cov(mod_oos.nobs - 1, mod_oos.nobs + 1), and that is a set of
    # mod_oos.nobs - 1 matrices.
    acov2 = res.smoothed_state_autocovariance(
        -2, end=mod_oos.nobs - 1, extend_kwargs=extend_kwargs
    ).transpose(2, 0, 1)
    assert_equal(acov2.shape, (mod_oos.nobs - 1, mod.k_states, mod.k_states))
    assert_allclose(acov2[:, :2, :2], desired_acov2[2:], rtol=1e-6)

    # Note: now we can compute up to Cov(mod_oos.nobs - 2, mod_oos.nobs + 1)
    # using a model that has state space matrices defined up to mod_oos.nobs.
    # We still need to pass in 5 additional time points for the extend kwargs.
    acov3 = res.smoothed_state_autocovariance(
        -3, end=mod_oos.nobs - 2, extend_kwargs=extend_kwargs
    ).transpose(2, 0, 1)
    assert_equal(acov3.shape, (mod_oos.nobs - 2, mod.k_states, mod.k_states))
    assert_allclose(acov3[:, :2, :2], desired_acov3[3:], rtol=1e-6)


@pytest.mark.parametrize("missing", ["all", "partial", "mixed", None])
@pytest.mark.parametrize("filter_univariate", [True, False])
@pytest.mark.parametrize("tvp", [True, False])
def test_smoothed_state_autocovariances_backwards_oos(missing, filter_univariate, tvp):
    # Out-of-sample model
    # Note: in TVP case, we need to first generate the larger model, and then
    # create the smaller model with the system matrices from the larger model
    # (otherwise they will be different, since the matrices are randomly
    # generated)
    mod_oos, res_oos = get_acov_model(missing, filter_univariate, tvp, oos=5)

    # Basic model
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
        with expect_diagonalized(filter_univariate):
            res = mod.ssm.smooth()

    assert_allclose(res_oos.llf, res.llf)

    cov = np.concatenate(
        (res_oos.smoothed_state_cov, res_oos.predicted_state_cov[..., -1:]), axis=2
    ).transpose(2, 0, 1)
    desired_acov1 = cov[:, :2, 2:4]
    desired_acov2 = cov[:, :2, 4:6]
    desired_acov3 = cov[:, :2, 6:8]

    # Test all "backwards" autocovariances: Cov(t, t - lag)
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

    # Note: we can compute up to Cov(mod_oos.nobs + 1, mod_oos.nobs) using
    # a model that has state space matrices defined up to mod_oos.nobs. Since
    # mod_oos.nobs = mod.nobs + 5, we need to pass in 5 additional time points,
    # and that is what extend_kwargs, above, does.
    acov1 = res.smoothed_state_autocovariance(
        1, end=end, extend_kwargs=extend_kwargs
    ).transpose(2, 0, 1)
    assert_equal(acov1.shape, (mod_oos.nobs + 1, mod.k_states, mod.k_states))
    assert_allclose(acov1[1:, :2, :2], desired_acov1[1:])
    # We cannot compute Cov(1, 0), so this is always NaNs
    assert_equal(acov1[:1], np.nan)

    # Note: we can compute up to Cov(mod_oos.nobs + 1, mod_oos.nobs - 1) using
    # a model that has state space matrices defined up to mod_oos.nobs, which
    # is why we don't need to change `end` here relative to the lag=1 case
    acov2 = res.smoothed_state_autocovariance(
        2, end=end, extend_kwargs=extend_kwargs
    ).transpose(2, 0, 1)
    assert_allclose(acov2[2:, :2, :2], desired_acov2[2:])
    # We cannot compute Cov(1, -1) or Cov(2, 0), so this is always NaNs
    assert_equal(acov2[:2], np.nan)

    # Note: we can compute up to Cov(mod_oos.nobs + 1, mod_oos.nobs - 2) using
    # a model that has state space matrices defined up to mod_oos.nobs, which
    # is why we don't need to change `end` here relative to the lag=1 or lag=2
    # cases
    acov3 = res.smoothed_state_autocovariance(
        3, end=end, extend_kwargs=extend_kwargs
    ).transpose(2, 0, 1)
    assert_allclose(acov3[3:, :2, :2], desired_acov3[3:])
    # We cannot compute Cov(1, -2), Cov(2, -1), or Cov(3, 0), so this is always
    # NaNs
    assert_equal(acov3[:3], np.nan)


def test_smoothed_state_autocovariances_invalid():
    # Tests for invalid calls of `smoothed_state_autocovariance`
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
    # Basic tests for news

    # Get the basic model
    mod, res = get_acov_model(missing, filter_univariate, tvp)
    params = [] if tvp else mod.start_params

    # Get an expanded model with one new observation and 9 additional NaN
    # datapoints (so that we can compute the desired value using the
    # `smoothed_forecasts` attribute).
    append = np.zeros((10, 2)) * np.nan
    append[0] = [0.1, -0.2]
    endog2 = np.concatenate((mod.endog, append), axis=0)
    mod2 = mod.clone(endog2)
    res2 = mod2.smooth(params, return_ssm=True)

    # Get an expanded model with only 10 additional NaN datapoints, to compute
    # the baseline `smoothed_forecasts`.
    endog3 = endog2.copy()
    endog3[-10:] = np.nan
    mod3 = mod2.clone(endog3)
    res3 = mod3.smooth(params, return_ssm=True)

    # Test the news computation at the start, middle, and end of the sample, as
    # well as out-of-sample.
    for t in [0, 1, 150, mod.nobs - 1, mod.nobs, mod.nobs + 1, mod.nobs + 9]:
        # Test with a time argument
        out = res2.news(res, t=t)
        desired = res2.smoothed_forecasts[..., t] - res3.smoothed_forecasts[..., t]
        # The "news" about the t=0 smoothed forecast from new data at
        # observation t=202 is almost identically zero, so we need to set an
        # "atol" to avoid problems with comparing floating point versions of
        # zero.
        assert_allclose(out.update_impacts, desired, atol=1e-14)
        assert_equal(out.revision_impacts, None)

        # Test with start/end arguments
        out = res2.news(res, start=t, end=t + 1)
        assert_allclose(out.update_impacts, desired[None, ...], atol=1e-14)


@pytest.mark.parametrize("missing", ["all", "partial", "mixed", None])
@pytest.mark.parametrize("filter_univariate", [True, False])
@pytest.mark.parametrize("tvp", [True, False])
def test_news_revisions(missing, filter_univariate, tvp):
    # Tests for news when there are revisions in the model

    # Get the basic model
    mod, res = get_acov_model(missing, filter_univariate, tvp, oos=10)
    params = [] if tvp else mod.start_params

    endog2 = mod.endog.copy()
    # Revise the last datapoint
    endog2[-11] = [0.0, 0.0]
    # Add a new datapoint
    endog2[-10] = [-0.3, -0.4]
    mod2 = mod.clone(endog2)
    res2 = mod2.smooth(params, return_ssm=True)

    # Test the news computation at the start, middle, and end of the sample, as
    # well as out-of-sample.
    nobs = mod.nobs - 10
    for t in [0, 1, 150, nobs - 1, nobs, nobs + 1, nobs + 9]:
        with expect_diagonalized(tvp and filter_univariate):
            out = res2.news(res, t=t)

        # Test for the news
        desired = (
            res2.smoothed_forecasts[..., t]
            - out.revision_results.smoothed_forecasts[..., t]
        )
        # Relaxed tolerance to 1e-10 after random failures
        assert_allclose(out.update_impacts, desired, atol=1e-10)

        # Test for the revisions
        desired = (
            out.revision_results.smoothed_forecasts[..., t]
            - res.smoothed_forecasts[..., t]
        )
        # Relaxed tolerance to 1e-10 after random failures
        assert_allclose(out.revision_impacts, desired, atol=1e-10)


@pytest.mark.parametrize("missing", ["all", "partial", "mixed", None])
@pytest.mark.parametrize("filter_univariate", [True, False])
@pytest.mark.parametrize("tvp", [True, False])
def test_news_invalid(missing, filter_univariate, tvp):
    # Tests for invalid calls to news

    # (generic error message used below)
    error_ss = (
        "This results object has %s and so it does not appear to"
        " by an extension of `previous`. Can only compute the"
        " news by comparing this results set to previous results"
        " objects."
    )

    # Basic model / results setup
    mod, res = get_acov_model(missing, filter_univariate, tvp, oos=1)
    params = [] if tvp else mod.start_params

    endog2 = mod.endog.copy()
    endog2[-1] = [0.2, 0.5]
    mod2 = mod.clone(endog2)
    res2_filtered = mod2.filter(params, return_ssm=True)
    res2_smoothed = mod2.smooth(params, return_ssm=True)

    # Test that news works with smoothing, but not with only filtering
    res2_smoothed.news(res, t=mod.nobs - 1)
    msg = "Cannot compute news without having applied the Kalman smoother first."
    with pytest.raises(ValueError, match=msg):
        res2_filtered.news(res, t=mod.nobs - 1)

    # Test that if we want to request news for an out-of-sample period in a
    # time-varying model, then we need to provide a new design matrix
    if tvp:
        msg = (
            "Cannot compute the impacts of news on periods outside of the"
            " sample in time-varying models."
        )
        with pytest.raises(RuntimeError, match=msg):
            res2_smoothed.news(res, t=mod.nobs + 2)

    # Test that news won't work when the calling model is smaller
    mod, res = get_acov_model(missing, filter_univariate, tvp)
    params = [] if tvp else mod.start_params

    endog2 = mod.endog.copy()[: mod.nobs - 1]
    mod2 = mod.clone(endog2)
    res2 = mod2.smooth(params, return_ssm=True)
    msg = error_ss % "fewer observations than `previous`"
    with pytest.raises(ValueError, match=msg):
        res2.news(res, t=mod.nobs - 1)

    # Test that news won't work when the state dimensions are different
    mod2 = sarimax.SARIMAX(np.zeros(mod.nobs))
    res2 = mod2.smooth([0.5, 1.0], return_ssm=True)
    msg = error_ss % "different state space dimensions than `previous`"
    with pytest.raises(ValueError, match=msg):
        res2.news(res, t=mod.nobs - 1)

    # Test that news won't work when one of the models is time-varying and one
    # is time-invariant
    mod2, res2 = get_acov_model(missing, filter_univariate, not tvp, oos=1)
    if tvp:
        msg = "time-invariant design while `previous` does not"
    else:
        msg = "time-varying design while `previous` does not"
    with pytest.raises(ValueError, match=msg):
        res2.news(res, t=mod.nobs - 1)


@pytest.mark.parametrize("extend_first", [False, True])
@pytest.mark.parametrize("lag", [-2, -3])
def test_smoothed_state_autocovariances_cache_extend_kwargs(lag, extend_first):
    # For lag < -1 the default end depends on whether extend_kwargs is given
    # (nobs + lag + 1 without, nobs with), also for a time-invariant model,
    # and the cached results do not depend on the order of the calls
    rng = np.random.default_rng(20261122)
    nobs = 30
    mod = mlemodel.MLEModel(rng.standard_normal((nobs, 2)), k_states=2, k_posdef=2)
    mod["design"] = np.array([[1.0, 0.0], [0.5, 1.0]])
    mod["obs_cov"] = np.diag([1.0, 0.7])
    mod["transition"] = np.array([[0.9, 0.1], [0.0, 0.8]])
    mod["selection"] = np.eye(2)
    mod["state_cov"] = np.array([[0.5, 0.2], [0.2, 0.4]])
    mod.ssm.initialize_known(np.zeros(2), np.eye(2))
    res = mod.ssm.smooth()
    if extend_first:
        extended = res.smoothed_state_autocovariance(lag, extend_kwargs={})
        default = res.smoothed_state_autocovariance(lag)
    else:
        default = res.smoothed_state_autocovariance(lag)
        extended = res.smoothed_state_autocovariance(lag, extend_kwargs={})
    assert default.shape == (2, 2, nobs + lag + 1)
    assert extended.shape == (2, 2, nobs)

    # Reference: Cov(alpha_t, alpha_s) from the joint posterior covariance of
    # the states (CFA simulation smoother) in the sample and, beyond it, from
    # the transition: Cov(alpha_t, alpha_{s + 1}) = Cov(alpha_t, alpha_s) T'
    cfa = mod.simulation_smoother(method="cfa")
    cfa.simulate(variates=np.zeros((2, nobs)))
    posterior_cov = cfa.posterior_cov
    desired = np.zeros((2, 2, nobs))
    for t in range(nobs):
        s = t - lag
        last = min(s, nobs - 1)
        block = posterior_cov[2 * t : 2 * t + 2, 2 * last : 2 * last + 2]
        transition = np.linalg.matrix_power(mod["transition"].T, s - last)
        desired[..., t] = block @ transition
    # atol: some autocovariances are close to zero
    assert_allclose(extended, desired, rtol=1e-8, atol=1e-12)
    assert_allclose(default, desired[..., : nobs + lag + 1], rtol=1e-8, atol=1e-12)


SMOOTH_ARGUMENTS_RESULTS = [
    "scaled_smoothed_estimator",
    "scaled_smoothed_estimator_cov",
    "smoothing_error",
    "smoothed_state",
    "smoothed_state_cov",
    "smoothed_state_autocov",
    "smoothed_measurement_disturbance",
    "smoothed_state_disturbance",
    "smoothed_measurement_disturbance_cov",
    "smoothed_state_disturbance_cov",
]


def _smooth_arguments_model():
    # Bivariate model with correlated states and a diagonal observation
    # covariance (so the univariate filter does not diagonalize it)
    rng = np.random.default_rng(20261122)
    endog = rng.standard_normal((40, 2)).cumsum(0) * 0.1 + rng.standard_normal((40, 2))
    mod = mlemodel.MLEModel(endog, k_states=2, k_posdef=2)
    mod["design"] = np.array([[1.0, 0.0], [0.5, 1.0]])
    mod["obs_intercept"] = np.array([0.1, -0.2])
    mod["obs_cov"] = np.diag([1.0, 0.7])
    mod["transition"] = np.array([[0.9, 0.1], [0.0, 0.8]])
    mod["state_intercept"] = np.array([0.05, 0.02])
    mod["selection"] = np.eye(2)
    mod["state_cov"] = np.array([[0.5, 0.2], [0.2, 0.4]])
    mod.ssm.initialize_known(np.array([0.3, -0.1]), np.eye(2))
    return mod


@pytest.mark.parametrize("use_ssm", [True, False])
def test_smooth_complex_step(use_ssm):
    # complex_step is passed on to the Kalman filter, so the stationary
    # initialization of the AR(2) is complex-step safe and the derivative of
    # the loglikelihood and of the smoothed state agree with central finite
    # differences (the loglikelihood derivative also with filter())
    rng = np.random.default_rng(20261122)
    mod = sarimax.SARIMAX(rng.standard_normal(50), order=(2, 0, 0))
    params = np.array([0.5, 0.2, 1.0])
    # The step of MLEModel.score. The stationary initialization uses a
    # complex Schur decomposition, whose intermediate results can have
    # imaginary parts of order one (depending on the LAPACK library, e.g. on
    # macOS), so their rounding error (1e-16) dominates a tiny step such as
    # 1e-20; with this step it is 1e-8 relative
    h = 1e-8
    cs_params = params + np.array([h * 1j, 0, 0])
    if use_ssm:
        # As in MLEModel.smooth, the complex step needs an inversion method
        # without the (conjugating) Cholesky factorization
        mod.update(cs_params, complex_step=True)
        res = mod.ssm.smooth(
            complex_step=True, inversion_method=INVERT_UNIVARIATE | SOLVE_LU
        )
    else:
        res = mod.smooth(cs_params, complex_step=True)
    llf_derivative = np.imag(res.llf) / h
    state_derivative = np.imag(res.smoothed_state) / h

    filtered = mod.filter(cs_params, complex_step=True)
    assert_allclose(llf_derivative, np.imag(filtered.llf) / h, rtol=1e-10)

    # Central differences; their error is about 1e-9 relative here, so the
    # tolerance catches a wrong derivative (it was 11% off) but not rounding
    eps = 1e-6
    step = np.array([eps, 0, 0])
    llf_fd = (mod.loglike(params + step) - mod.loglike(params - step)) / (2 * eps)
    assert_allclose(llf_derivative, llf_fd, rtol=1e-6)
    state_plus = mod.smooth(params + step).smoothed_state
    state_minus = mod.smooth(params - step).smoothed_state
    state_fd = (state_plus - state_minus) / (2 * eps)
    # atol: some derivatives are close to zero, so scale by the largest one.
    # The first period is included: the results re-initialized the state
    # without the complex step before smoothing, and the smoother reads the
    # initial state covariance again in the first period.
    atol = 1e-6 * np.max(np.abs(state_fd))
    assert_allclose(state_derivative, state_fd, rtol=1e-6, atol=atol)


def test_smooth_smooth_method_argument():
    # smooth_method selects the smoothing method for this call only. The
    # univariate smoother needs the univariate filter, and the univariate
    # filter needs it: a method that cannot be used was ignored on main (the
    # model's method was used), and still is, with a FutureWarning
    conventional = _smooth_arguments_model().ssm.smooth()
    mod = _smooth_arguments_model()
    with pytest.warns(FutureWarning, match="smooth_method cannot be used"):
        res = mod.ssm.smooth(smooth_method=SMOOTH_UNIVARIATE)
    for name in SMOOTH_ARGUMENTS_RESULTS:
        assert_allclose(
            getattr(res, name), getattr(conventional, name), rtol=1e-13
        )
    univariate = _smooth_arguments_model()
    univariate.ssm.filter_univariate = True
    with pytest.warns(FutureWarning, match="smooth_method cannot be used"):
        res = univariate.ssm.smooth(
            smooth_method=kalman_smoother.SMOOTH_CONVENTIONAL
        )
    desired = _smooth_arguments_model()
    desired.ssm.filter_univariate = True
    desired = desired.ssm.smooth()
    for name in SMOOTH_ARGUMENTS_RESULTS:
        assert_allclose(getattr(res, name), getattr(desired, name), rtol=1e-13)

    # The methods that can be used do not warn
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for method in [0, SMOOTH_ALTERNATIVE, SMOOTH_CLASSICAL]:
            res = _smooth_arguments_model().ssm.smooth(smooth_method=method)
            assert_allclose(
                res.smoothed_state, conventional.smoothed_state, rtol=1e-10
            )
        for method in [0, SMOOTH_UNIVARIATE]:
            univariate = _smooth_arguments_model()
            univariate.ssm.filter_univariate = True
            res = univariate.ssm.smooth(smooth_method=method)
            assert_allclose(
                res.smoothed_state, desired.smoothed_state, rtol=1e-13
            )

    # A model set to the conventional smoother cannot smooth the output of the
    # univariate filter, unless the argument selects the univariate smoother
    mod = _smooth_arguments_model()
    mod.ssm.filter_univariate = True
    mod.ssm.smooth_method = kalman_smoother.SMOOTH_CONVENTIONAL
    with pytest.raises(ValueError, match="Invalid smoothing method"):
        mod.ssm.smooth()
    res = mod.ssm.smooth(smooth_method=SMOOTH_UNIVARIATE)
    assert mod.ssm.smooth_method == kalman_smoother.SMOOTH_CONVENTIONAL

    # The default method for the univariate filter is the univariate smoother
    default = _smooth_arguments_model()
    default.ssm.filter_univariate = True
    desired = default.ssm.smooth()
    for name in SMOOTH_ARGUMENTS_RESULTS:
        assert_allclose(getattr(res, name), getattr(desired, name), rtol=1e-13)

    # and it agrees with the conventional (multivariate) filter and smoother
    conventional = _smooth_arguments_model().ssm.smooth()
    assert_allclose(res.smoothed_state, conventional.smoothed_state, rtol=1e-10)
    assert_allclose(
        res.smoothed_state_cov, conventional.smoothed_state_cov, rtol=1e-10
    )


def test_smooth_filter_method_argument():
    # filter_method is passed to the Kalman filter, as the other keyword
    # arguments are (it raised a TypeError: multiple values for filter_method)
    mod = _smooth_arguments_model()
    res = mod.ssm.smooth(filter_method=FILTER_UNIVARIATE)
    assert np.all(res.univariate_filter == 1)

    desired_mod = _smooth_arguments_model()
    desired_mod.ssm.filter_univariate = True
    desired = desired_mod.ssm.smooth()
    for name in SMOOTH_ARGUMENTS_RESULTS:
        assert_allclose(getattr(res, name), getattr(desired, name), rtol=1e-13)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"results": kalman_smoother.SmootherResults},
        {"run_filter": False},
        {"run_filter": True},
        {"prefix": "d"},
        {"prefix": "s"},
    ],
)
def test_smooth_unused_arguments_deprecated(kwargs):
    # results, run_filter and prefix have no effect: they warn and the output
    # is that of a call without them
    mod = _smooth_arguments_model()
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        desired = mod.ssm.smooth()
    name = next(iter(kwargs))
    with pytest.warns(FutureWarning, match=f"{name} has no effect") as record:
        res = mod.ssm.smooth(**kwargs)
    # The warning points to the caller
    assert record[0].filename == __file__
    assert type(res) is kalman_smoother.SmootherResults
    assert res.smoothed_state.dtype == np.float64
    for name in SMOOTH_ARGUMENTS_RESULTS:
        assert_allclose(getattr(res, name), getattr(desired, name), rtol=1e-14)


@pytest.mark.parametrize("method", ["filter", "smooth"])
def test_complex_step_initial_state_cov(method):
    # The results re-initialized the state without the complex step after
    # filtering, so the complex-step derivative of their initial state
    # covariance was wrong (and of the first period of the smoothed state
    # covariance, which the smoother computes from it). With measurement
    # error, the first observation does not pin down the state
    rng = np.random.default_rng(20261122)
    mod = sarimax.SARIMAX(
        rng.standard_normal(50), order=(2, 0, 0), measurement_error=True
    )
    params = np.array([0.5, 0.2, 0.5, 1.0])
    # The step of MLEModel.score (a tiny step is dominated by rounding; see
    # test_smooth_complex_step)
    h = 1e-8
    cs_params = params + np.array([h * 1j, 0, 0, 0])
    res = getattr(mod, method)(cs_params, complex_step=True)
    results = res.smoother_results if method == "smooth" else res.filter_results

    # The stationary covariance of the AR(2) state from scipy, and its central
    # differences (error about 1e-10 relative here; the derivative was wrong
    # in every element, by up to 100%)
    def stationary_cov(phi1):
        transition = np.array([[phi1, 1.0], [params[1], 0.0]])
        selected_cov = np.array([[params[3], 0.0], [0.0, 0.0]])
        return linalg.solve_discrete_lyapunov(transition, selected_cov)

    eps = 1e-6
    cov_plus = stationary_cov(params[0] + eps)
    cov_minus = stationary_cov(params[0] - eps)
    assert_allclose(
        np.real(results.initial_state_cov), stationary_cov(params[0]), rtol=1e-12
    )
    assert_allclose(
        np.imag(results.initial_state_cov) / h,
        (cov_plus - cov_minus) / (2 * eps),
        rtol=1e-6,
    )

    if method == "smooth":
        step = np.array([eps, 0, 0, 0])
        plus = mod.smooth(params + step).smoothed_state_cov[:, :, 0]
        minus = mod.smooth(params - step).smoothed_state_cov[:, :, 0]
        assert_allclose(
            np.imag(res.smoothed_state_cov[:, :, 0]) / h,
            (plus - minus) / (2 * eps),
            rtol=1e-6,
        )


def test_results_of_uninitialized_statespace():
    # A results object for a model whose compiled statespace object exists
    # (simulation_smoother creates it) but has not been initialized yet still
    # initializes it, and stores the stationary covariance of the AR(2) state
    rng = np.random.default_rng(20261122)
    mod = sarimax.SARIMAX(rng.standard_normal(50), order=(2, 0, 0))
    params = np.array([0.5, 0.2, 1.0])
    mod.update(params)
    mod.simulation_smoother()
    res = mod.ssm.results_class(mod.ssm)

    transition = np.array([[params[0], 1.0], [params[1], 0.0]])
    selected_cov = np.array([[params[2], 0.0], [0.0, 0.0]])
    desired = linalg.solve_discrete_lyapunov(transition, selected_cov)
    assert_allclose(res.initial_state_cov, desired, rtol=1e-12)


def correlated_model(diffuse=False, state_cov=None, initial_state_cov=None):
    # Bivariate model with correlated states and non-zero intercepts, so that
    # no entry of the smoother output is zero by construction. The diagonal
    # observation covariance keeps the univariate (diffuse) filter from
    # diagonalizing it.
    rng = np.random.default_rng(20261122)
    endog = rng.standard_normal((40, 2)).cumsum(0) * 0.1 + rng.standard_normal((40, 2))
    mod = mlemodel.MLEModel(endog, k_states=2, k_posdef=2)
    mod["design"] = np.array([[1.0, 0.0], [0.5, 1.0]])
    mod["obs_intercept"] = np.array([0.1, -0.2])
    mod["obs_cov"] = np.diag([1.0, 0.7])
    mod["transition"] = np.array([[0.9, 0.1], [0.0, 0.8]])
    mod["state_intercept"] = np.array([0.05, 0.02])
    mod["selection"] = np.eye(2)
    if state_cov is None:
        state_cov = np.array([[0.5, 0.2], [0.2, 0.4]])
    mod["state_cov"] = state_cov
    if diffuse:
        mod.ssm.initialize_diffuse()
    else:
        if initial_state_cov is None:
            initial_state_cov = np.eye(2)
        mod.ssm.initialize_known(np.array([0.3, -0.1]), initial_state_cov)
    return mod


SMOOTHER_RESULTS = [
    "scaled_smoothed_estimator",
    "scaled_smoothed_estimator_cov",
    "smoothing_error",
    "smoothed_state",
    "smoothed_state_cov",
    "smoothed_state_autocov",
    "smoothed_measurement_disturbance",
    "smoothed_state_disturbance",
    "smoothed_measurement_disturbance_cov",
    "smoothed_state_disturbance_cov",
]


@pytest.mark.parametrize(
    "smooth_method, flag",
    [(SMOOTH_ALTERNATIVE, "smooth_alternative"), (SMOOTH_CLASSICAL, "smooth_classical")],
)
def test_exact_diffuse_alternative_classical_smoothers(smooth_method, flag):
    # In the exact diffuse periods the alternative and classical smoothers use
    # the diffuse recursions; all output equals the conventional smoother
    desired = correlated_model(diffuse=True).ssm.smooth()
    assert desired.nobs_diffuse > 0

    mod = correlated_model(diffuse=True)
    mod.ssm.set_smooth_method(**{flag: True})
    assert mod.ssm.smooth_method == smooth_method
    res = mod.ssm.smooth()
    assert res.nobs_diffuse == desired.nobs_diffuse
    for name in SMOOTHER_RESULTS:
        actual = getattr(res, name)
        expected = getattr(desired, name)
        if name == "smoothing_error":
            # The diffuse smoother does not define the smoothing error in the
            # diffuse periods (the classical smoother fills them)
            actual = actual[:, desired.nobs_diffuse :]
            expected = expected[:, desired.nobs_diffuse :]
        # The recursions differ in rounding, about 1e-11 here for entries of
        # order one; atol covers entries that are zero in exact arithmetic
        # (e.g. the scaled smoothed estimator after the last period)
        assert_allclose(actual, expected, rtol=1e-8, atol=1e-9)


def test_classical_smoother_singular_predicted_state_cov():
    # The second state is deterministic, so its predicted variance is exactly
    # zero and the classical smoother cannot invert the predicted state cov
    kwargs = dict(
        state_cov=np.diag([0.5, 0.0]), initial_state_cov=np.diag([1.0, 0.0])
    )
    mod = correlated_model(**kwargs)
    # The conventional smoother handles it: the deterministic state is known
    res = mod.ssm.smooth()
    assert_allclose(res.smoothed_state_cov[1, 1], 0.0, atol=1e-12)

    mod = correlated_model(**kwargs)
    mod.ssm.smooth_classical = True
    with pytest.raises(
        np.linalg.LinAlgError, match="Singular predicted state covariance"
    ):
        mod.ssm.smooth()


@pytest.mark.parametrize(
    "option, match",
    [
        ("memory_no_predicted", "without all predicted states"),
        ("memory_no_gain", "without all Kalman gains"),
        ("memory_no_smoothing", "without all smoothing variables"),
    ],
)
def test_smooth_conserve_memory_raises(option, match):
    mod = correlated_model()
    mod.ssm.set_conserve_memory(**{option: True})
    # Filtering alone works
    assert np.isfinite(mod.ssm.loglike())
    with pytest.raises(ValueError, match=match):
        mod.ssm.smooth()


def test_smooth_univariate_requires_univariate_filter():
    mod = correlated_model()
    mod.ssm.smooth_univariate = True
    with pytest.raises(ValueError, match="can only use univariate smoothing"):
        mod.ssm.smooth()


def test_smooth_invalid_smooth_method():
    mod = correlated_model()
    mod.ssm.smooth_method = 0x10
    with pytest.raises(NotImplementedError, match="Smoother method not available"):
        mod.ssm.smooth()


def test_kalman_smoother_results_class():
    class CustomSmootherResults(SmootherResults):
        pass

    mod = correlated_model()
    desired = mod.ssm.smooth()
    names = [
        "design",
        "obs_intercept",
        "obs_cov",
        "transition",
        "state_intercept",
        "selection",
        "state_cov",
    ]
    ssm = KalmanSmoother(
        2,
        2,
        results_class=CustomSmootherResults,
        **{name: mod[name] for name in names},
    )
    ssm.bind(np.ascontiguousarray(mod.endog))
    ssm.initialize_known(np.array([0.3, -0.1]), np.eye(2))
    res = ssm.smooth()
    assert type(res) is CustomSmootherResults
    assert_allclose(res.smoothed_state, desired.smoothed_state, rtol=1e-12)


SMOOTHER_OUTPUT_ATTRIBUTES = {
    SMOOTHER_STATE: ["scaled_smoothed_estimator", "smoothed_state"],
    SMOOTHER_STATE_COV: ["scaled_smoothed_estimator_cov", "smoothed_state_cov"],
    SMOOTHER_DISTURBANCE: [
        "scaled_smoothed_estimator",
        "smoothing_error",
        "smoothed_measurement_disturbance",
        "smoothed_state_disturbance",
    ],
}


@pytest.mark.parametrize(
    "smoother_output", list(SMOOTHER_OUTPUT_ATTRIBUTES), ids=["state", "cov", "dist"]
)
def test_smoother_output_subset(smoother_output):
    # Only the requested output is computed and stored; it equals the output
    # of the full smoother, and everything else is None
    desired = correlated_model().ssm.smooth()

    mod = correlated_model()
    mod.ssm.smoother_output = smoother_output
    res = mod.ssm.smooth()
    computed = SMOOTHER_OUTPUT_ATTRIBUTES[smoother_output]
    for name in SMOOTHER_RESULTS:
        if name in computed:
            assert_allclose(getattr(res, name), getattr(desired, name), rtol=1e-12)
        else:
            assert getattr(res, name) is None
    if "scaled_smoothed_estimator" in computed:
        assert_allclose(
            res.scaled_smoothed_estimator_presample,
            desired.scaled_smoothed_estimator_presample,
            rtol=1e-12,
        )
    else:
        assert not hasattr(res, "scaled_smoothed_estimator_presample")

    if smoother_output == SMOOTHER_STATE_COV:
        assert_allclose(
            res.smoothed_state_autocovariance(0),
            desired.smoothed_state_cov,
            rtol=1e-12,
        )
    else:
        with pytest.raises(RuntimeError, match="Cannot return smoothed state cov"):
            res.smoothed_state_autocovariance(0)


def test_smoothed_state_autocovariance_forward_default_end():
    # In a time-varying model without `extend_kwargs`, Cov(t, t + 2) is only
    # available through period nobs - 2, so the default `end` is nobs - 1
    mod, res = get_acov_model(None, False, True)
    desired = res.smoothed_state_cov.transpose(2, 0, 1)[:, 4:6, :2]

    acov = res.smoothed_state_autocovariance(-2)
    assert_equal(acov.shape, (mod.k_states, mod.k_states, mod.nobs - 1))
    acov = acov.transpose(2, 0, 1)
    assert_allclose(acov[:-1, :2, :2], desired[2:], rtol=1e-6, atol=1e-10)
    # The last one, Cov(nobs - 2, nobs), uses the predicted state cov at nobs
    explicit = res.smoothed_state_autocovariance(-2, start=mod.nobs - 2, end=mod.nobs - 1)
    assert_allclose(acov[-1], explicit[..., 0], rtol=1e-12)


def get_news_models(oos=10, revise=False):
    # Results before and after a data release (one new observation of both
    # variables, and optionally a revision of the last published one), and
    # the results with the new observation missing
    mod, res = get_acov_model(None, False, False, oos=oos)
    params = mod.start_params
    nobs = mod.nobs - oos

    endog2 = mod.endog.copy()
    if revise:
        endog2[nobs - 1] = [0.0, 0.0]
    endog2[nobs] = [-0.3, -0.4]
    res2 = mod.clone(endog2).smooth(params, return_ssm=True)

    endog3 = endog2.copy()
    endog3[nobs] = np.nan
    res3 = mod.clone(endog3).smooth(params, return_ssm=True)
    return nobs, res, res2, res3


def test_news_default_periods():
    # Without `t`, `start` and `end`, the news are for the last period of the
    # sample; `end` defaults to the end of the sample
    nobs, res, res2, res3 = get_news_models()
    desired = (res2.smoothed_forecasts - res3.smoothed_forecasts).T

    out = res2.news(res)
    assert_allclose(out.update_impacts, desired[-1:], rtol=1e-8)

    out = res2.news(res, start=nobs - 2)
    assert_equal(out.update_impacts.shape, (res2.nobs - nobs + 2, 2))
    # The impact on the period before the release is small, but not zero
    assert_allclose(out.update_impacts, desired[nobs - 2 :], rtol=1e-8, atol=1e-14)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(t=1, start=1), "Cannot specify both `t`"),
        (dict(t=1, end=2), "Cannot specify both `t`"),
        (dict(t=-1), "Negative `t`"),
        (dict(start=-2, end=3), "Negative `t`"),
        (dict(start=5, end=5), "`end` must be after `start`"),
    ],
)
def test_news_invalid_periods(kwargs, match):
    _, res, res2, _ = get_news_models(oos=1)
    with pytest.raises(ValueError, match=match):
        res2.news(res, **kwargs)


def test_news_state_index_partition():
    # The impacts are linear in the states, so the impacts computed from a
    # partition of the states add up to the impacts from all states
    nobs, res, res2, _ = get_news_models(revise=True)
    kwargs = dict(start=nobs - 3, end=nobs + 3)
    desired = res2.news(res, **kwargs)

    part1 = res2.news(res, state_index=[2, 0, 1], **kwargs)
    part2 = res2.news(res, state_index=[3, 4, 5, 6, 7], **kwargs)
    assert_equal(part1.state_index, [0, 1, 2])
    assert_equal(desired.state_index, None)
    for name in ["update_impacts", "revision_detailed_impacts"]:
        actual = getattr(part1, name) + getattr(part2, name)
        # atol: impacts on periods before the release are close to zero
        assert_allclose(actual, getattr(desired, name), rtol=1e-8, atol=1e-14)
    # The gain of the news is also linear in the states
    assert_allclose(part1.gain + part2.gain, desired.gain, rtol=1e-8, atol=1e-14)
    # The total revision impacts do not depend on the states used
    assert_allclose(part1.revision_impacts, desired.revision_impacts, rtol=1e-12)


def test_smoothed_state_gain():
    # The gain maps the news (the forecast error of the new observation) to
    # the change of the smoothed state: E[a_s | post] - E[a_s | previous]
    # The previous results include the (missing) period of the release
    nobs, res, res2, res3 = get_news_models(oos=1)
    assert res.nobs == nobs + 1
    updates_ix = [(nobs, 0), (nobs, 1)]
    news = res2.endog[:, nobs] - res3.smoothed_forecasts[:, nobs]
    desired = (res2.smoothed_state - res3.smoothed_state).T

    # Default: the last period of the previous sample, here the release
    gain = res.smoothed_state_gain(updates_ix)
    assert_equal(gain.shape, (1, res.k_states, 2))
    assert_allclose(gain @ news, desired[nobs:], rtol=1e-8)
    # `end` defaults to the end of the previous sample
    gain = res.smoothed_state_gain(updates_ix, start=nobs - 3)
    assert_equal(gain.shape, (4, res.k_states, 2))
    assert_allclose(gain @ news, desired[nobs - 3 :], rtol=1e-8)
    # A single period
    gain = res.smoothed_state_gain(updates_ix, t=nobs - 1)
    assert_equal(gain.shape, (res.k_states, 2))
    assert_allclose(gain @ news, desired[nobs - 1], rtol=1e-8)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(t=1, start=1), "Cannot specify both `t`"),
        (dict(t=-1), "Negative `t`"),
        (dict(start=5, end=5), "`end` must be after `start`"),
    ],
)
def test_smoothed_state_gain_invalid(kwargs, match):
    _, res = get_acov_model(None, False, False)
    with pytest.raises(ValueError, match=match):
        res.smoothed_state_gain([(res.nobs, 0)], **kwargs)


def test_smoothed_state_gain_time_varying_requires_extension():
    _, res = get_acov_model(None, False, True)
    with pytest.raises(ValueError, match="time-varying design matrix"):
        res.smoothed_state_gain([(res.nobs, 0)])


def test_smoothed_decomposition_signal_state_index_partition():
    # The contributions to the smoothed signal are linear in the states, so
    # those from a partition of the states add up to those from all states
    # (The state intercept contributions are left out: they do not use
    # `state_index`.)
    res = correlated_model().ssm.smooth()
    desired = res.get_smoothed_decomposition("smoothed_signal")
    part1 = res.get_smoothed_decomposition("smoothed_signal", state_index=[0])
    part2 = res.get_smoothed_decomposition("smoothed_signal", state_index=[1])
    for i in [0, 1, 3]:
        # atol: the weights of observations far from t are close to zero
        assert_allclose(part1[i] + part2[i], desired[i], rtol=1e-10, atol=1e-14)
    # All contributions together give the smoothed signal Z a_t
    data, obs_intercept, state_intercept, prior = desired
    signal = (
        data.sum(axis=(2, 3))
        + obs_intercept.sum(axis=(2, 3))
        + state_intercept.sum(axis=(2, 3))
        + prior.sum(axis=2)
    )
    assert_allclose(
        signal, (res.design[..., 0] @ res.smoothed_state).T, rtol=1e-10, atol=1e-12
    )


# Results attributes of each smoother output
SUBSET_ATTRIBUTES = {
    kalman_smoother.SMOOTHER_STATE: ["smoothed_state", "scaled_smoothed_estimator"],
    kalman_smoother.SMOOTHER_STATE_COV: [
        "smoothed_state_cov",
        "scaled_smoothed_estimator_cov",
    ],
    kalman_smoother.SMOOTHER_STATE_AUTOCOV: ["smoothed_state_autocov"],
    kalman_smoother.SMOOTHER_DISTURBANCE: [
        "scaled_smoothed_estimator",
        "smoothing_error",
        "smoothed_measurement_disturbance",
        "smoothed_state_disturbance",
    ],
    kalman_smoother.SMOOTHER_DISTURBANCE_COV: [
        "scaled_smoothed_estimator_cov",
        "smoothed_measurement_disturbance_cov",
        "smoothed_state_disturbance_cov",
    ],
}
SUBSET_FLAGS = {
    "smoother_state": kalman_smoother.SMOOTHER_STATE,
    "smoother_state_cov": kalman_smoother.SMOOTHER_STATE_COV,
    "smoother_state_autocov": kalman_smoother.SMOOTHER_STATE_AUTOCOV,
    "smoother_disturbance": kalman_smoother.SMOOTHER_DISTURBANCE,
    "smoother_disturbance_cov": kalman_smoother.SMOOTHER_DISTURBANCE_COV,
}
# All subsets, including no output (0)
SUBSET_MASKS = [
    sum(subset)
    for r in range(6)
    for subset in itertools.combinations(SUBSET_FLAGS.values(), r)
]


def _output_subset_model(smooth_method, diffuse):
    # Bivariate model with intercepts, correlated states, a diagonal
    # observation covariance and missing data
    rng = np.random.default_rng(20261122)
    endog = rng.standard_normal((40, 2)).cumsum(0) * 0.1 + rng.standard_normal((40, 2))
    endog[10, 0] = np.nan
    endog[20] = np.nan
    mod = mlemodel.MLEModel(endog, k_states=2, k_posdef=2)
    mod["design"] = np.array([[1.0, 0.0], [0.5, 1.0]])
    mod["obs_intercept"] = np.array([0.1, -0.2])
    mod["obs_cov"] = np.diag([1.0, 0.7])
    mod["transition"] = np.array([[0.9, 0.1], [0.0, 0.8]])
    mod["state_intercept"] = np.array([0.05, 0.02])
    mod["selection"] = np.eye(2)
    mod["state_cov"] = np.array([[0.5, 0.2], [0.2, 0.4]])
    if diffuse:
        mod.ssm.initialize_diffuse()
    else:
        mod.ssm.initialize_known(np.array([0.3, -0.1]), np.eye(2))
    if smooth_method == SMOOTH_UNIVARIATE:
        mod.ssm.filter_univariate = True
    mod.ssm.smooth_method = smooth_method
    return mod


def _expected_smoother_output(smoother_output, smooth_method):
    # The documented dependencies of the smoother output
    expected = smoother_output
    if expected & kalman_smoother.SMOOTHER_STATE_AUTOCOV:
        expected |= kalman_smoother.SMOOTHER_STATE_COV
    if expected & kalman_smoother.SMOOTHER_DISTURBANCE_COV:
        expected |= kalman_smoother.SMOOTHER_DISTURBANCE
        if smooth_method == SMOOTH_CLASSICAL:
            expected |= kalman_smoother.SMOOTHER_STATE_COV
    if smooth_method == SMOOTH_CLASSICAL and (
        expected & kalman_smoother.SMOOTHER_DISTURBANCE
    ):
        expected |= kalman_smoother.SMOOTHER_STATE
    return expected


def _check_smoother_output(res, desired, expected_output, rtol=1e-12):
    assert res.smoother_output == expected_output
    for name, flag in SUBSET_FLAGS.items():
        assert getattr(res, name) is bool(expected_output & flag)
    # smoother_all is set if any output is computed
    assert res.smoother_all is bool(expected_output)
    computed = set()
    for flag, names in SUBSET_ATTRIBUTES.items():
        if expected_output & flag:
            computed.update(names)
    for names in SUBSET_ATTRIBUTES.values():
        for name in names:
            actual = getattr(res, name)
            if name in computed:
                # atol: entries that are zero in the full run (for example
                # the smoothing error of missing observations)
                assert_allclose(actual, getattr(desired, name), rtol=rtol, atol=1e-14)
            else:
                assert actual is None
    presample = {
        "scaled_smoothed_estimator_presample": "scaled_smoothed_estimator",
        "scaled_smoothed_estimator_cov_presample": "scaled_smoothed_estimator_cov",
    }
    for name, attribute in presample.items():
        if attribute in computed:
            assert_allclose(getattr(res, name), getattr(desired, name), rtol=rtol)
        else:
            assert not hasattr(res, name)


@pytest.mark.parametrize("via", ["argument", "option"])
@pytest.mark.parametrize("diffuse", [False, True])
@pytest.mark.parametrize(
    "smooth_method",
    [
        kalman_smoother.SMOOTH_CONVENTIONAL,
        SMOOTH_ALTERNATIVE,
        SMOOTH_CLASSICAL,
        SMOOTH_UNIVARIATE,
    ],
)
def test_smoother_output_dependencies(smooth_method, diffuse, via):
    # Every subset of the smoother output agrees with the full smoother run,
    # with the output it depends on computed as well; the results report the
    # computed output and have None for output that was not computed
    desired = _output_subset_model(smooth_method, diffuse).ssm.smooth()
    assert desired.smoother_output == kalman_smoother.SMOOTHER_ALL
    if diffuse:
        assert desired.nobs_diffuse > 0
    for smoother_output in SUBSET_MASKS:
        mod = _output_subset_model(smooth_method, diffuse)
        if via == "argument":
            res = mod.ssm.smooth(smoother_output=smoother_output)
            # The options of the model do not change
            assert mod.ssm.smoother_output == kalman_smoother.SMOOTHER_ALL
        else:
            mod.ssm.smoother_output = smoother_output
            res = mod.ssm.smooth()
            assert mod.ssm.smoother_output == smoother_output
        expected = _expected_smoother_output(smoother_output, smooth_method)
        _check_smoother_output(res, desired, expected)


def test_smoother_output_dependencies_concentrated_scale():
    # With a concentrated scale the computed covariances and scaled smoothed
    # estimators are rescaled to those of the model with that scale, and
    # output that was not computed is None
    rng = np.random.default_rng(20261122)
    endog = rng.standard_normal(60).cumsum() * 0.1 + rng.standard_normal(60)
    mod = sarimax.SARIMAX(
        endog, order=(1, 0, 1), measurement_error=True, concentrate_scale=True
    )
    # The measurement variance is relative to the concentrated scale
    assert mod.param_names == ["ar.L1", "ma.L1", "snr.measurement_error"]
    params = [0.5, 0.2, 0.4]
    scale = mod.smooth(params).scale
    ref_mod = sarimax.SARIMAX(endog, order=(1, 0, 1), measurement_error=True)
    desired = ref_mod.smooth([0.5, 0.2, 0.4 * scale, scale])
    for smoother_output in SUBSET_MASKS:
        res = mod.smooth(params, smoother_output=smoother_output)
        assert res.smoother_results.filter_concentrated
        expected = _expected_smoother_output(
            smoother_output, kalman_smoother.SMOOTH_CONVENTIONAL
        )
        # rtol: the two models compute the output in different orders
        _check_smoother_output(
            res.smoother_results, desired.smoother_results, expected, rtol=1e-10
        )
        if expected & kalman_smoother.SMOOTHER_STATE_COV:
            # atol: the second state of the ARMA(1, 1) has covariances that
            # are zero up to rounding
            assert_allclose(
                res.smoothed_state_cov,
                desired.smoothed_state_cov,
                rtol=1e-10,
                atol=1e-14,
            )
        else:
            assert res.smoothed_state_cov is None
