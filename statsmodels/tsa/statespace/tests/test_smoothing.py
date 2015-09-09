"""
Tests for smoothing and estimation of unobserved states and disturbances

- Predicted states: :math:`E(\alpha_t | Y_{t-1})`
- Filtered states: :math:`E(\alpha_t | Y_t)`
- Smoothed states: :math:`E(\alpha_t | Y_n)`
- Smoothed disturbances :math:`E(\varepsilon_t | Y_n), E(\eta_t | Y_n)`
- Simulation smoothing

Tested against R (FKF, KalmanRun / KalmanSmooth), Stata (sspace), and
MATLAB (ssm toolbox)

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import pandas as pd
import os

from statsmodels.tsa.statespace import sarimax
from numpy.testing import assert_almost_equal, assert_raises
from nose.exc import SkipTest

current_path = os.path.dirname(os.path.abspath(__file__))


class TestStatesAR3(object):
    def __init__(self, *args, **kwargs):
        # Dataset / Stata comparison
        path = current_path + os.sep + 'results/results_wpi1_ar3_stata.csv'
        self.stata = pd.read_csv(path)
        self.stata.index = pd.date_range(start='1960-01-01', periods=124,
                                         freq='QS')
        # Matlab comparison
        path = current_path + os.sep+'results/results_wpi1_ar3_matlab_ssm.csv'
        matlab_names = [
            'a1','a2','a3','detP','alphahat1','alphahat2','alphahat3',
            'detV','eps','epsvar','eta','etavar'
        ]
        self.matlab_ssm = pd.read_csv(path, header=None, names=matlab_names)
        # Regression tests data
        path = current_path + os.sep+'results/results_wpi1_ar3_regression.csv'
        self.regression = pd.read_csv(path)

        self.model = sarimax.SARIMAX(
            self.stata['wpi'], order=(3, 1, 0), simple_differencing=True,
            hamilton_representation=True, *args, **kwargs
        )

        # Parameters from from Stata's sspace MLE estimation
        params = np.r_[.5270715, .0952613, .2580355, .5307459]
        self.results = self.model.smooth(params)

        # Calculate the determinant of the covariance matrices (for easy
        # comparison to other languages without having to store 2-dim arrays)
        self.results.det_predicted_state_cov = np.zeros((1, self.model.nobs))
        self.results.det_smoothed_state_cov = np.zeros((1, self.model.nobs))
        for i in range(self.model.nobs):
            self.results.det_predicted_state_cov[0,i] = np.linalg.det(
                self.results.filter_results.predicted_state_cov[:,:,i])
            self.results.det_smoothed_state_cov[0,i] = np.linalg.det(
                self.results.smoother_results.smoothed_state_cov[:,:,i])

    def test_predict_obs(self):
        assert_almost_equal(
            self.results.filter_results.predict().forecasts[0],
            self.stata.ix[1:, 'dep1'], 4
        )

    def test_standardized_residuals(self):
        assert_almost_equal(
            self.results.filter_results.standardized_forecasts_error[0],
            self.stata.ix[1:, 'sr1'], 4
        )

    def test_predicted_states(self):
        assert_almost_equal(
            self.results.filter_results.predicted_state[:,:-1].T,
            self.stata.ix[1:, ['sp1', 'sp2', 'sp3']], 4
        )
        assert_almost_equal(
            self.results.filter_results.predicted_state[:,:-1].T,
            self.matlab_ssm[['a1', 'a2', 'a3']], 4
        )

    def test_predicted_states_cov(self):
        assert_almost_equal(
            self.results.det_predicted_state_cov.T,
            self.matlab_ssm[['detP']], 4
        )

    def test_filtered_states(self):
        assert_almost_equal(
            self.results.filter_results.filtered_state.T,
            self.stata.ix[1:, ['sf1', 'sf2', 'sf3']], 4
        )

    def test_smoothed_states(self):
        assert_almost_equal(
            self.results.smoother_results.smoothed_state.T,
            self.stata.ix[1:, ['sm1', 'sm2', 'sm3']], 4
        )
        assert_almost_equal(
            self.results.smoother_results.smoothed_state.T,
            self.matlab_ssm[['alphahat1', 'alphahat2', 'alphahat3']], 4
        )

    def test_smoothed_states_cov(self):
        assert_almost_equal(
            self.results.det_smoothed_state_cov.T,
            self.matlab_ssm[['detV']], 4
        )

    def test_smoothed_measurement_disturbance(self):
        assert_almost_equal(
            self.results.smoother_results.smoothed_measurement_disturbance.T,
            self.matlab_ssm[['eps']], 4
        )

    def test_smoothed_measurement_disturbance_cov(self):
        res = self.results.smoother_results
        assert_almost_equal(
            res.smoothed_measurement_disturbance_cov[0].T,
            self.matlab_ssm[['epsvar']], 4
        )

    def test_smoothed_state_disturbance(self):
        assert_almost_equal(
            self.results.smoother_results.smoothed_state_disturbance.T,
            self.matlab_ssm[['eta']], 4
        )

    def test_smoothed_state_disturbance_cov(self):
        assert_almost_equal(
            self.results.smoother_results.smoothed_state_disturbance_cov[0].T,
            self.matlab_ssm[['etavar']], 4
        )


class TestStatesMissingAR3(object):
    def __init__(self, alternate_timing=True, *args, **kwargs):
        # Dataset
        path = current_path + os.sep + 'results/results_wpi1_ar3_stata.csv'
        self.stata = pd.read_csv(path)
        self.stata.index = pd.date_range(start='1960-01-01', periods=124,
                                         freq='QS')
        # Matlab comparison
        path = current_path + os.sep+'results/results_wpi1_missing_ar3_matlab_ssm.csv'
        matlab_names = [
            'a1','a2','a3','detP','alphahat1','alphahat2','alphahat3',
            'detV','eps','epsvar','eta','etavar'
        ]
        self.matlab_ssm = pd.read_csv(path, header=None, names=matlab_names)
        # Regression tests data
        path = current_path + os.sep+'results/results_wpi1_missing_ar3_regression.csv'
        self.regression = pd.read_csv(path)

        # Create missing observations
        self.stata['dwpi'] = self.stata['wpi'].diff()
        self.stata.ix[10:21, 'dwpi'] = np.nan

        self.model = sarimax.SARIMAX(
            self.stata.ix[1:,'dwpi'], order=(3, 0, 0),
            hamilton_representation=True, *args, **kwargs
        )

        # Parameters from from Stata's sspace MLE estimation
        self.params = np.r_[.5270715, .0952613, .2580355, .5307459]
        
    def test_missing_smooth(self):
        assert_raises(RuntimeError, self.model.smooth, self.params)
