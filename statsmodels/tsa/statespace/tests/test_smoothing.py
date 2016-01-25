"""
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
from __future__ import division, absolute_import, print_function

import numpy as np
import pandas as pd
import os

from statsmodels import datasets
from statsmodels.tsa.statespace import mlemodel, sarimax
from numpy.testing import assert_allclose, assert_almost_equal, assert_raises
from nose.exc import SkipTest

current_path = os.path.dirname(os.path.abspath(__file__))


class TestStatesAR3(object):
    @classmethod
    def setup_class(cls, *args, **kwargs):
        # Dataset / Stata comparison
        path = current_path + os.sep + 'results/results_wpi1_ar3_stata.csv'
        cls.stata = pd.read_csv(path)
        cls.stata.index = pd.date_range(start='1960-01-01', periods=124,
                                         freq='QS')
        # Matlab comparison
        path = current_path + os.sep+'results/results_wpi1_ar3_matlab_ssm.csv'
        matlab_names = [
            'a1','a2','a3','detP','alphahat1','alphahat2','alphahat3',
            'detV','eps','epsvar','eta','etavar'
        ]
        cls.matlab_ssm = pd.read_csv(path, header=None, names=matlab_names)
        # Regression tests data
        path = current_path + os.sep+'results/results_wpi1_ar3_regression.csv'
        cls.regression = pd.read_csv(path)

        cls.model = sarimax.SARIMAX(
            cls.stata['wpi'], order=(3, 1, 0), simple_differencing=True,
            hamilton_representation=True, *args, **kwargs
        )

        # Parameters from from Stata's sspace MLE estimation
        params = np.r_[.5270715, .0952613, .2580355, .5307459]
        cls.results = cls.model.smooth(params)

        # Calculate the determinant of the covariance matrices (for easy
        # comparison to other languages without having to store 2-dim arrays)
        cls.results.det_predicted_state_cov = np.zeros((1, cls.model.nobs))
        cls.results.det_smoothed_state_cov = np.zeros((1, cls.model.nobs))
        for i in range(cls.model.nobs):
            cls.results.det_predicted_state_cov[0,i] = np.linalg.det(
                cls.results.filter_results.predicted_state_cov[:,:,i])
            cls.results.det_smoothed_state_cov[0,i] = np.linalg.det(
                cls.results.smoother_results.smoothed_state_cov[:,:,i])

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
    @classmethod
    def setup_class(cls, *args, **kwargs):
        # Dataset
        path = current_path + os.sep + 'results/results_wpi1_ar3_stata.csv'
        cls.stata = pd.read_csv(path)
        cls.stata.index = pd.date_range(start='1960-01-01', periods=124,
                                         freq='QS')
        # Matlab comparison
        path = current_path + os.sep+'results/results_wpi1_missing_ar3_matlab_ssm.csv'
        matlab_names = [
            'a1','a2','a3','detP','alphahat1','alphahat2','alphahat3',
            'detV','eps','epsvar','eta','etavar'
        ]
        cls.matlab_ssm = pd.read_csv(path, header=None, names=matlab_names)
        # Regression tests data
        path = current_path + os.sep+'results/results_wpi1_missing_ar3_regression.csv'
        cls.regression = pd.read_csv(path)

        # Create missing observations
        cls.stata['dwpi'] = cls.stata['wpi'].diff()
        cls.stata.ix[10:21, 'dwpi'] = np.nan

        cls.model = sarimax.SARIMAX(
            cls.stata.ix[1:,'dwpi'], order=(3, 0, 0),
            hamilton_representation=True, *args, **kwargs
        )

        # Parameters from from Stata's sspace MLE estimation
        params = np.r_[.5270715, .0952613, .2580355, .5307459]
        cls.results = cls.model.smooth(params, return_ssm=True)

        # Calculate the determinant of the covariance matrices (for easy
        # comparison to other languages without having to store 2-dim arrays)
        cls.results.det_predicted_state_cov = np.zeros((1, cls.model.nobs))
        cls.results.det_smoothed_state_cov = np.zeros((1, cls.model.nobs))
        for i in range(cls.model.nobs):
            cls.results.det_predicted_state_cov[0,i] = np.linalg.det(
                cls.results.predicted_state_cov[:,:,i])
            cls.results.det_smoothed_state_cov[0,i] = np.linalg.det(
                cls.results.smoothed_state_cov[:,:,i])

    def test_predicted_states(self):
        assert_almost_equal(
            self.results.predicted_state[:,:-1].T,
            self.matlab_ssm[['a1', 'a2', 'a3']], 4
        )

    def test_predicted_states_cov(self):
        assert_almost_equal(
            self.results.det_predicted_state_cov.T,
            self.matlab_ssm[['detP']], 4
        )

    def test_smoothed_states(self):
        assert_almost_equal(
            self.results.smoothed_state.T,
            self.matlab_ssm[['alphahat1', 'alphahat2', 'alphahat3']], 4
        )

    def test_smoothed_states_cov(self):
        assert_almost_equal(
            self.results.det_smoothed_state_cov.T,
            self.matlab_ssm[['detV']], 4
        )

    def test_smoothed_measurement_disturbance(self):
        assert_almost_equal(
            self.results.smoothed_measurement_disturbance.T,
            self.matlab_ssm[['eps']], 4
        )

    def test_smoothed_measurement_disturbance_cov(self):
        assert_almost_equal(
            self.results.smoothed_measurement_disturbance_cov[0].T,
            self.matlab_ssm[['epsvar']], 4
        )


class TestMultivariateMissing(object):
    """
    Tests for most filtering and smoothing variables against output from the
    R library KFAS.
    
    Note that KFAS uses the univariate approach which generally will result in
    different predicted values and covariance matrices associated with the
    measurement equation (e.g. forecasts, etc.). In this case, although the
    model is multivariate, each of the series is truly independent so the values
    will be the same regardless of whether the univariate approach is used or
    not.
    """
    @classmethod
    def setup_class(cls):
        # Results
        path = current_path + os.sep + 'results/results_smoothing_R.csv'
        cls.desired = pd.read_csv(path)
        
        # Data
        dta = datasets.macrodata.load_pandas().data
        dta.index = pd.date_range(start='1959-01-01', end='2009-7-01', freq='QS')
        obs = dta[['realgdp','realcons','realinv']].diff().ix[1:]
        obs.ix[0:50, 0] = np.nan
        obs.ix[19:70, 1] = np.nan
        obs.ix[39:90, 2] = np.nan
        obs.ix[119:130, 0] = np.nan
        obs.ix[119:130, 2] = np.nan
        
        # Create the model
        mod = mlemodel.MLEModel(obs, k_states=3, k_posdef=3)
        mod['design'] = np.eye(3)
        mod['obs_cov'] = np.eye(3)
        mod['transition'] = np.eye(3)
        mod['selection'] = np.eye(3)
        mod['state_cov'] = np.eye(3)
        mod.initialize_approximate_diffuse(1e6)
        cls.model = mod
        cls.results = mod.smooth([], return_ssm=True)
        
        # Calculate the determinant of the covariance matrices (for easy
        # comparison to other languages without having to store 2-dim arrays)
        cls.results.det_scaled_smoothed_estimator_cov = (
            np.zeros((1, cls.model.nobs)))
        cls.results.det_predicted_state_cov = np.zeros((1, cls.model.nobs))
        cls.results.det_smoothed_state_cov = np.zeros((1, cls.model.nobs))
        cls.results.det_smoothed_state_disturbance_cov = (
            np.zeros((1, cls.model.nobs)))

        for i in range(cls.model.nobs):
            cls.results.det_scaled_smoothed_estimator_cov[0,i] = (
                np.linalg.det(
                    cls.results.scaled_smoothed_estimator_cov[:,:,i]))
            cls.results.det_predicted_state_cov[0,i] = np.linalg.det(
                cls.results.predicted_state_cov[:,:,i+1])
            cls.results.det_smoothed_state_cov[0,i] = np.linalg.det(
                cls.results.smoothed_state_cov[:,:,i])
            cls.results.det_smoothed_state_disturbance_cov[0,i] = (
                np.linalg.det(
                    cls.results.smoothed_state_disturbance_cov[:,:,i]))
                
    def test_loglike(self):
        assert_allclose(np.sum(self.results.llf_obs), -205310.9767)

    def test_scaled_smoothed_estimator(self):
        assert_allclose(
            self.results.scaled_smoothed_estimator.T,
            self.desired[['r1', 'r2', 'r3']]
        )
        
    def test_scaled_smoothed_estimator_cov(self):
        assert_allclose(
            self.results.det_scaled_smoothed_estimator_cov.T,
            self.desired[['detN']]
        )
        
    def test_forecasts(self):
        assert_allclose(
            self.results.forecasts.T,
            self.desired[['m1', 'm2', 'm3']]
        )
    
    def test_forecasts_error(self):
        assert_allclose(
            self.results.forecasts_error.T,
            self.desired[['v1', 'v2', 'v3']]
        )

    def test_forecasts_error_cov(self):
        assert_allclose(
            self.results.forecasts_error_cov.diagonal(),
            self.desired[['F1', 'F2', 'F3']]
        )

    def test_predicted_states(self):
        assert_allclose(
            self.results.predicted_state[:,1:].T,
            self.desired[['a1', 'a2', 'a3']]
        )

    def test_predicted_states_cov(self):
        assert_allclose(
            self.results.det_predicted_state_cov.T,
            self.desired[['detP']]
        )

    def test_smoothed_states(self):
        assert_allclose(
            self.results.smoothed_state.T,
            self.desired[['alphahat1', 'alphahat2', 'alphahat3']]
        )

    def test_smoothed_states_cov(self):
        assert_allclose(
            self.results.det_smoothed_state_cov.T,
            self.desired[['detV']]
        )
        
    def test_smoothed_forecasts(self):
        assert_allclose(
            self.results.smoothed_forecasts.T,
            self.desired[['muhat1','muhat2','muhat3']]
        )
        
    def test_smoothed_state_disturbance(self):
        assert_allclose(
            self.results.smoothed_state_disturbance.T,
            self.desired[['etahat1','etahat2','etahat3']]
        )

    def test_smoothed_state_disturbance_cov(self):
        assert_allclose(
            self.results.det_smoothed_state_disturbance_cov.T,
            self.desired[['detVeta']]
        )

    def test_smoothed_measurement_disturbance(self):
        assert_allclose(
            self.results.smoothed_measurement_disturbance.T,
            self.desired[['epshat1','epshat2','epshat3']]
        )

    def test_smoothed_measurement_disturbance_cov(self):
        assert_allclose(
            self.results.smoothed_measurement_disturbance_cov.diagonal(),
            self.desired[['Veps1','Veps2','Veps3']]
        )
