"""
Tests for exact diffuse initialization

Notes
-----

These tests are against four sources:

- Koopman (1997)
- The R package KFAS (v1.3.1): test_exact_diffuse_filtering.R
- Stata: test_exact_diffuse_filtering_stata.do
- Statsmodels state space models using approximate diffuse filtering

Koopman (1997) provides analytic results for a few cases that we can test
against. More comprehensive tests are available against the R package KFAS,
which also uses the Durbin and Koopman (2012) univariate diffuse filtering
method. However, there are apparently some bugs in the KFAS output (see notes
below), so some tests are run against Stata.

KFAS v1.3.1 appears to have the following bugs:

- Incorrect filtered covariance matrix (in their syntax, kf$Ptt). These
  matrices are not even symmetric, so they are clearly wrong.
- Loglikelihood computation appears to be incorrect for the diffuse part of
  the state. See the section with "Note: Apparent loglikelihood discrepancy"
  in the R file. It appears that KFAS does not include the constant term
  (-0.5 * log(2 pi)) for the diffuse observations, whereas the loglikelihood
  function as given in e.g. section 7.2.5 of Durbin and Koopman (2012) shows
  that it should be included. To confirm this, we also check against the
  loglikelihood value computed by Stata.

Stata uses the DeJong diffuse filtering method, which gives almost identical
results but does imply some numerical differences for output at the 6th or 7th
decimal place.

Finally, we have tests against the same model using approximate (rather than
exact) diffuse filtering. These will by definition have some discrepancies in
the diffuse observations.

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function
from statsmodels.compat.testing import SkipTest, skip, skipif

import numpy as np
import pandas as pd
import os

from statsmodels.tools.tools import Bunch
from statsmodels import datasets
from statsmodels.tsa.statespace.initialization import Initialization
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from statsmodels.tsa.statespace.structural import UnobservedComponents
from numpy.testing import assert_equal, assert_allclose
import pytest

current_path = os.path.dirname(os.path.abspath(__file__))
macrodata = datasets.macrodata.load_pandas().data
macrodata.index = pd.PeriodIndex(start='1959Q1', end='2009Q3', freq='Q')

def test_local_level_analytic():
    # Analytic test using results from Koopman (1997), section 5.1
    y1 = 10.2394
    sigma2_y = 1.993
    sigma2_mu = 8.253

    # Construct the basic representation
    mod = KalmanFilter(k_endog=1, k_states=1, k_posdef=1)
    endog = np.r_[y1, [1] * 9]
    mod.bind(endog)
    mod.initialize(Initialization(mod.k_states, initialization_type='diffuse'))
    # mod.filter_univariate = True  # should not be required

    # Fill in the system matrices for a local level model
    mod['design', :] = 1
    mod['obs_cov', :] = sigma2_y
    mod['transition', :] = 1
    mod['selection', :] = 1
    mod['state_cov', :] = sigma2_mu

    # Perform filtering
    res = mod.filter()

    # Basic initialization variables
    assert_allclose(res.predicted_state_cov[0, 0, 0], 0)
    assert_allclose(res.predicted_diffuse_state_cov[0, 0, 0], 1)

    # Output of the exact diffuse initialization, see Koopman (1997)
    assert_allclose(res.forecasts_error[0, 0], y1)
    assert_allclose(res.forecasts_error_cov[0, 0, 0], sigma2_y)
    assert_allclose(res.forecasts_error_diffuse_cov[0, 0, 0], 1)
    assert_allclose(res.kalman_gain[0, 0, 0], 1)
    assert_allclose(res.predicted_state[0, 1], y1)
    assert_allclose(res.predicted_state_cov[0, 0, 1], sigma2_y + sigma2_mu)
    assert_allclose(res.predicted_diffuse_state_cov[0, 0, 1], 0)

    # Miscellaneous
    assert_equal(res.nobs_diffuse, 1)

def test_local_linear_trend_analytic():
    # Analytic test using results from Koopman (1997), section 5.2
    y1 = 10.2394
    y2 = 4.2039
    y3 = 6.123123
    sigma2_y = 1.993
    sigma2_mu = 8.253
    sigma2_beta = 2.334

    # Construct the basic representation
    mod = KalmanFilter(k_endog=1, k_states=2, k_posdef=2)
    endog = np.r_[y1, y2, y3, [1] * 7]
    mod.bind(endog)
    mod.initialize(Initialization(mod.k_states, initialization_type='diffuse'))
    # mod.filter_univariate = True  # should not be required

    # Fill in the system matrices for a local level model
    mod['design', 0, 0] = 1
    mod['obs_cov', 0, 0] = sigma2_y
    mod['transition'] = np.array([[1, 1],
                                  [0, 1]])
    mod['selection'] = np.eye(2)
    mod['state_cov'] = np.diag([sigma2_mu, sigma2_beta])

    # Perform filtering
    res = mod.filter()

    # Basic initialization variables
    assert_allclose(res.predicted_state_cov[..., 0], np.zeros((2, 2)))
    assert_allclose(res.predicted_diffuse_state_cov[..., 0], np.eye(2))

    # Output of the exact diffuse initialization, see Koopman (1997)
    q_mu = sigma2_mu / sigma2_y
    q_beta = sigma2_beta / sigma2_y
    assert_allclose(res.forecasts_error[0, 0], y1)
    assert_allclose(res.kalman_gain[:, 0, 0], [1, 0])
    assert_allclose(res.predicted_state[:, 1], [y1, 0])
    P2 = sigma2_y * np.array([[1 + q_mu, 0],
                              [0, q_beta]])
    assert_allclose(res.predicted_state_cov[:, :, 1], P2)
    assert_allclose(res.predicted_diffuse_state_cov[0, 0, 1], np.ones((2, 2)))

    # assert_allclose(res.kalman_gain[:, 0, 1], [2, 1])
    assert_allclose(res.predicted_state[:, 2], [2 * y2 - y1, y2 - y1])
    P3 = sigma2_y * np.array([[5 + 2 * q_mu + q_beta, 3 + q_mu + q_beta],
                              [3 + q_mu + q_beta, 2 + q_mu + 2 * q_beta]])
    assert_allclose(res.predicted_state_cov[:, :, 2], P3)
    assert_allclose(res.predicted_diffuse_state_cov[:, :, 2], np.zeros((2, 2)))

    # Miscellaneous
    assert_equal(res.nobs_diffuse, 2)


def test_common_level_analytic():
    # Analytic test using results from Koopman (1997), section 5.3
    y11 = 10.2394
    y21 = 8.2304
    theta = 0.1111
    sigma2_1 = 1
    sigma_12 = 0
    sigma2_2 = 1
    sigma2_mu = 3.2324

    # Construct the basic representation
    mod = KalmanFilter(k_endog=2, k_states=2, k_posdef=1)
    mod.filter_univariate = True
    endog = np.column_stack([np.r_[y11, [1] * 9], np.r_[y21, [1] * 9]])
    mod.bind(endog.T)
    mod.initialize(Initialization(mod.k_states, initialization_type='diffuse'))

    # Fill in the system matrices for a common trend model
    mod['design'] = np.array([[1, 0],
                              [theta, 1]])
    mod['obs_cov'] = np.eye(2)
    mod['transition'] = np.eye(2)
    mod['selection', 0, 0] = 1
    mod['state_cov', 0, 0] = sigma2_mu

    # Perform filtering
    res = mod.filter()

    # Basic initialization variables
    assert_allclose(res.predicted_state_cov[..., 0], np.zeros((2, 2)))
    assert_allclose(res.predicted_diffuse_state_cov[..., 0], np.eye(2))

    # Output of the exact diffuse initialization, see Koopman (1997)

    # Note: since Koopman (1997) did not apply the univariate method,
    # forecast errors and covariances, and the Kalman gain won't match
    # assert_allclose(res.forecasts_error[:, 0], [y11, y21])
    # assert_allclose(res.forecasts_error_cov[:, :, 0], np.eye(2))
    # F_inf1 = np.array([[1, theta],
    #                    [theta, 1 + theta**2]])
    # assert_allclose(res.forecasts_error_diffuse_cov[:, :, 0], F_inf1)
    # K0 = np.array([[1, 0],
    #                [-theta, 1]])
    # assert_allclose(res.kalman_gain[..., 0], K0)
    assert_allclose(res.predicted_state[:, 1], [y11, y21 - theta * y11])
    P2 = np.array([[1 + sigma2_mu, -theta],
                   [-theta, 1 + theta**2]])
    assert_allclose(res.predicted_state_cov[..., 1], P2)
    assert_allclose(res.predicted_diffuse_state_cov[..., 1], np.zeros((2, 2)))

    # Miscellaneous
    assert_equal(res.nobs_diffuse, 1)


def test_common_level_restricted_analytic():
    # Analytic test using results from Koopman (1997), section 5.3,
    # with the restriction mu_bar = 0
    y11 = 10.2394
    y21 = 8.2304
    theta = 0.1111
    sigma2_1 = 1
    sigma_12 = 0
    sigma2_2 = 1
    sigma2_mu = 3.2324

    # Construct the basic representation
    mod = KalmanFilter(k_endog=2, k_states=1, k_posdef=1)
    endog = np.column_stack([np.r_[y11, [1] * 9], np.r_[y21, [1] * 9]])
    mod.bind(endog.T)
    mod.initialize(Initialization(mod.k_states, initialization_type='diffuse'))
    # mod.filter_univariate = True  # should not be required

    # Fill in the system matrices for a local level model
    mod['design'] = np.array([[1, theta]]).T
    mod['obs_cov'] = np.eye(2)
    mod['transition', :] = 1
    mod['selection', :] = 1
    mod['state_cov', :] = sigma2_mu

    # Perform filtering
    res = mod.filter()

    # Basic initialization variables
    assert_allclose(res.predicted_state_cov[..., 0], 0)
    assert_allclose(res.predicted_diffuse_state_cov[..., 0], 1)

    # Output of the exact diffuse initialization, see Koopman (1997)
    phi = 1 / (1 + theta**2)
    # Note: since Koopman (1997) did not apply the univariate method,
    # forecast errors and covariances, and the Kalman gain won't match
    # assert_allclose(res.forecasts_error[:, 0], [y11, y21])
    # assert_allclose(res.forecasts_error_cov[0, 0, 0], np.eye(2))
    # F_inf1 = np.array([[1, theta],
    #                    [theta, theta**2]])
    # assert_allclose(res.forecasts_error_diffuse_cov[0, 0, 0], F_inf1)
    # assert_allclose(res.kalman_gain[..., 0], phi * np.array([1, theta]))
    assert_allclose(res.predicted_state[:, 1], phi * (y11 + theta * y21))
    # Note: Koopman (1997) actually has phi + sigma2_mu**0.5, but that appears
    # to be a typo
    assert_allclose(res.predicted_state_cov[..., 1], phi + sigma2_mu)
    assert_allclose(res.predicted_diffuse_state_cov[..., 1], 0)

    # Miscellaneous
    assert_equal(res.nobs_diffuse, 1)

def test_local_linear_trend_missing_analytic():
    # Analytic test using results from Koopman (1997), section 6.2
    y1 = 10.2394
    y2 = np.nan
    y3 = 6.123123
    sigma2_y = 1.993
    sigma2_mu = 8.253
    sigma2_beta = 2.334

    # Construct the basic representation
    mod = KalmanFilter(k_endog=1, k_states=2, k_posdef=2)
    endog = np.r_[y1, y2, y3, [1] * 7]
    mod.bind(endog)
    mod.initialize(Initialization(mod.k_states, initialization_type='diffuse'))
    # mod.filter_univariate = True  # should not be required

    # Fill in the system matrices for a local level model
    mod['design', 0, 0] = 1
    mod['obs_cov', 0, 0] = sigma2_y
    mod['transition'] = np.array([[1, 1],
                                  [0, 1]])
    mod['selection'] = np.eye(2)
    mod['state_cov'] = np.diag([sigma2_mu, sigma2_beta])

    res = mod.filter()

    # Test output
    q_mu = sigma2_mu / sigma2_y
    q_beta = sigma2_beta / sigma2_y
    a4 = [1.5 * y3 - 0.5 * y1, 0.5 * y3 - 0.5 * y1]
    assert_allclose(res.predicted_state[:, 3], a4)
    P4 = sigma2_y * np.array([
        [2.5 + 1.5 * q_mu + 1.25 * q_beta, 1 + 0.5 * q_mu + 1.25 * q_beta],
        [1 + 0.5 * q_mu + 1.25 * q_beta, 0.5 + 0.5 * q_mu + 2.25 * q_beta]])
    assert_allclose(res.predicted_state_cov[:, :, 3], P4)

    # Miscellaneous
    assert_equal(res.nobs_diffuse, 3)


class Check(object):
    def test_forecasts(self, d=None, rtol_diffuse=1e-5):
        if d is not None and rtol_diffuse != np.inf:
            assert_allclose(
                self.results_a.forecasts.T[:d],
                self.results_b.forecasts.T[:d], rtol=rtol_diffuse)
        assert_allclose(
            self.results_a.forecasts.T[d:],
            self.results_b.forecasts.T[d:])

    def test_forecasts_error(self, d=None, rtol_diffuse=1e-5):
        if d is not None and rtol_diffuse != np.inf:
            assert_allclose(
                self.results_a.forecasts_error.T[:d],
                self.results_b.forecasts_error.T[:d], rtol=rtol_diffuse)
        assert_allclose(
            self.results_a.forecasts_error.T[d:],
            self.results_b.forecasts_error.T[d:])

    def test_forecasts_error_cov(self, d=None, rtol_diffuse=1e-5):
        actual = self.results_a.forecasts_error_cov.T
        desired = self.results_b.forecasts_error_cov.T
        if desired.ndim == 1:
            actual = np.sum(actual, axis=(1, 2))
        if d is not None and rtol_diffuse != np.inf:
            assert_allclose(actual[:d], desired[d:], rtol=rtol_diffuse)
        assert_allclose(actual[d:], desired[d:])

    def test_filtered_state(self, d=None, rtol_diffuse=1e-5):
        if d is not None and rtol_diffuse != np.inf:
            assert_allclose(
                self.results_a.filtered_state.T[:d],
                self.results_b.filtered_state.T[:d], rtol=rtol_diffuse)
        assert_allclose(
            self.results_a.filtered_state.T[d:],
            self.results_b.filtered_state.T[d:])

    def test_filtered_state_cov(self, d=None, rtol_diffuse=1e-5):
        actual = self.results_a.filtered_state_cov.T
        desired = self.results_b.filtered_state_cov.T
        if desired.ndim == 1:
            actual = np.sum(actual, axis=(1, 2))
        if d is not None and rtol_diffuse != np.inf:
            assert_allclose(actual[:d], desired[:d], rtol=rtol_diffuse)
        assert_allclose(actual[d:], desired[d:], atol=1e-15)  # TODO atol necessary?

    def test_predicted_state(self, d=None, rtol_diffuse=1e-5):
        if d is not None and rtol_diffuse != np.inf:
            assert_allclose(
                self.results_a.predicted_state.T[:d],
                self.results_b.predicted_state.T[:d])
        assert_allclose(
            self.results_a.predicted_state.T[d:],
            self.results_b.predicted_state.T[d:])

    def test_predicted_state_cov(self, d=None, rtol_diffuse=1e-5):
        actual = self.results_a.predicted_state_cov.T
        desired = self.results_b.predicted_state_cov.T
        if desired.ndim == 1:
            actual = np.sum(actual, axis=(1, 2))
        if d is not None and rtol_diffuse != np.inf:
            assert_allclose(actual[:d], desired[:d], rtol=rtol_diffuse)
        assert_allclose(actual[d:], desired[d:])

    def test_loglike(self, d=None, rtol_diffuse=1e-5):
        if np.isscalar(self.results_b.llf_obs):
            assert_allclose(np.sum(self.results_a.llf_obs),
                                   self.results_b.llf_obs)
        else:
            if d is not None and rtol_diffuse != np.inf:
                assert_allclose(
                    self.results_a.llf_obs[:d],
                    self.results_b.llf_obs[:d], rtol=rtol_diffuse)
            assert_allclose(
                self.results_a.llf_obs[d:],
                self.results_b.llf_obs[d:])

    # def test_smoothed_states(self):
    #     assert_allclose(
    #         self.results_a.smoothed_state,
    #         self.results_b.smoothed_state)

    # def test_smoothed_states_cov(self):
    #     assert_allclose(
    #         self.results_a.smoothed_state_cov,
    #         self.results_b.smoothed_state_cov)

    # def test_smoothed_states_autocov(self):
    #     assert_allclose(
    #         self.results_a.smoothed_state_autocov,
    #         self.results_b.smoothed_state_autocov)

    # def test_smoothed_measurement_disturbance(self):
    #     assert_allclose(
    #         self.results_a.smoothed_measurement_disturbance,
    #         self.results_b.smoothed_measurement_disturbance)

    # def test_smoothed_measurement_disturbance_cov(self):
    #     assert_allclose(
    #         self.results_a.smoothed_measurement_disturbance_cov,
    #         self.results_b.smoothed_measurement_disturbance_cov)

    # def test_smoothed_state_disturbance(self):
    #     assert_allclose(
    #         self.results_a.smoothed_state_disturbance,
    #         self.results_b.smoothed_state_disturbance)

    # def test_smoothed_state_disturbance_cov(self):
    #     assert_allclose(
    #         self.results_a.smoothed_state_disturbance_cov,
    #         self.results_b.smoothed_state_disturbance_cov)

    # def test_simulation_smoothed_state(self):
    #     assert_allclose(
    #         self.sim_a.simulated_state,
    #         self.sim_a.simulated_state)

    # def test_simulation_smoothed_measurement_disturbance(self):
    #     assert_allclose(
    #         self.sim_a.simulated_measurement_disturbance,
    #         self.sim_a.simulated_measurement_disturbance)

    # def test_simulation_smoothed_state_disturbance(self):
    #     assert_allclose(
    #         self.sim_a.simulated_state_disturbance,
    #         self.sim_a.simulated_state_disturbance)


class CheckApproximateDiffuse(Check):
    """
    Test the exact diffuse initialization against the approximate diffuse
    initialization. By definition, the first few observations will be quite
    different between the exact and approximate approach for many quantities,
    so we do not test them here.
    """

    def test_forecasts(self, d=None, rtol_diffuse=np.inf):
        if d is None:
            d = self.results_a.nobs_diffuse
        super(CheckApproximateDiffuse, self).test_forecasts(d, rtol_diffuse)

    def test_forecasts_error(self, d=None, rtol_diffuse=np.inf):
        if d is None:
            d = self.results_a.nobs_diffuse
        super(CheckApproximateDiffuse, self).test_forecasts_error(d, rtol_diffuse)

    def test_forecasts_error_cov(self, d=None, rtol_diffuse=np.inf):
        if d is None:
            d = self.results_a.nobs_diffuse
        super(CheckApproximateDiffuse, self).test_forecasts_error_cov(d, rtol_diffuse)

    def test_filtered_state(self, d=None, rtol_diffuse=np.inf):
        if d is None:
            d = self.results_a.nobs_diffuse
        super(CheckApproximateDiffuse, self).test_filtered_state(d, rtol_diffuse)

    def test_filtered_state_cov(self, d=None, rtol_diffuse=np.inf):
        if d is None:
            d = self.results_a.nobs_diffuse
        super(CheckApproximateDiffuse, self).test_filtered_state_cov(d, rtol_diffuse)

    def test_predicted_state(self, d=None, rtol_diffuse=np.inf):
        if d is None:
            d = self.results_a.nobs_diffuse
        super(CheckApproximateDiffuse, self).test_predicted_state(d, rtol_diffuse)

    def test_predicted_state_cov(self, d=None, rtol_diffuse=np.inf):
        if d is None:
            d = self.results_a.nobs_diffuse
        super(CheckApproximateDiffuse, self).test_predicted_state_cov(d, rtol_diffuse)

    def test_loglike(self, d=None, rtol_diffuse=np.inf):
        if d is None:
            d = self.results_a.nobs_diffuse
        super(CheckApproximateDiffuse, self).test_loglike(d, rtol_diffuse)


class TestVAR1(CheckApproximateDiffuse):
    @classmethod
    def setup_class(cls):
        # Dataset
        endog = (np.log(macrodata[['realgdp', 'realcons']]).diff().iloc[1:] * 400)

        # Model
        cls.model = VARMAX(endog, order=(1, 0), trend='nc')
        cls.model.update(cls.model.start_params)
        cls.ssm = cls.model.ssm

        # Exact diffuse
        init1 = Initialization(cls.ssm.k_states, 'diffuse')
        cls.ssm.initialize(init1)
        cls.results_a = cls.ssm.filter()
        cls.d = cls.results_a.nobs_diffuse

        # Approximate diffuse
        init2 = Initialization(cls.ssm.k_states, 'approximate_diffuse')
        cls.ssm.initialize(init2)
        cls.results_b = cls.ssm.filter()

    def test_initialization(self):
        assert_allclose(self.results_a.initial_state_cov, 0)
        assert_allclose(self.results_a.initial_diffuse_state_cov, np.eye(2))

        assert_allclose(self.results_b.initial_state_cov, np.eye(2) * 1e6)
        assert_allclose(self.results_b.initial_diffuse_state_cov, 0)

    def test_nobs_diffuse(self):
        assert_allclose(self.d, 1)


class TestVAR1_Missing(CheckApproximateDiffuse):
    @classmethod
    def setup_class(cls):
        # Dataset
        endog = (np.log(macrodata[['realgdp', 'realcons']]).diff().iloc[1:] * 400)

        endog.iloc[0, 0:5] = np.nan
        endog.iloc[:, 8:12] = np.nan

        # Model
        cls.model = VARMAX(endog, order=(1, 0), trend='nc')
        cls.model.update(cls.model.start_params)
        cls.ssm = cls.model.ssm
        cls.ssm.filter_univariate = True

        # Exact diffuse
        init1 = Initialization(cls.ssm.k_states, 'diffuse')
        cls.ssm.initialize(init1)
        cls.results_a = cls.ssm.filter()
        cls.d = cls.results_a.nobs_diffuse

        # Approximate diffuse
        init2 = Initialization(cls.ssm.k_states, 'approximate_diffuse')
        cls.ssm.initialize(init2)
        cls.results_b = cls.ssm.filter()

    def test_initialization(self):
        assert_allclose(self.results_a.initial_state_cov, 0)
        assert_allclose(self.results_a.initial_diffuse_state_cov, np.eye(2))

        assert_allclose(self.results_b.initial_state_cov, np.eye(2) * 1e6)
        assert_allclose(self.results_b.initial_diffuse_state_cov, 0)

    def test_nobs_diffuse(self):
        assert_allclose(self.d, 2)


class TestVAR1_Mixed(CheckApproximateDiffuse):
    @classmethod
    def setup_class(cls):
        # Dataset
        endog = (np.log(macrodata[['realgdp', 'realcons']]).diff().iloc[1:] * 400)

        # Model
        cls.model = VARMAX(endog, order=(1, 0), trend='nc')
        cls.model.update(cls.model.start_params)
        cls.ssm = cls.model.ssm
        cls.ssm.filter_univariate = True

        # Exact diffuse
        init1 = Initialization(cls.ssm.k_states)
        init1.set(0, 'diffuse')
        init1.set(1, 'stationary')
        cls.ssm.initialize(init1)
        cls.results_a = cls.ssm.filter()
        cls.d = cls.results_a.nobs_diffuse

        # Approximate diffuse
        init2 = Initialization(cls.ssm.k_states)
        init2.set(0, 'approximate_diffuse')
        init2.set(1, 'stationary')
        cls.ssm.initialize(init2)
        cls.results_b = cls.ssm.filter()

    def test_initialization(self):
        assert_allclose(self.results_a.initial_state_cov, np.diag([0, 13.474315]))
        assert_allclose(self.results_a.initial_diffuse_state_cov, np.diag([1, 0]))

        assert_allclose(self.results_b.initial_state_cov, np.diag([1e6, 13.474315]))
        assert_allclose(self.results_b.initial_diffuse_state_cov, 0)

    def test_nobs_diffuse(self):
        assert_allclose(self.d, 1)


class TestDFM(CheckApproximateDiffuse):
    @classmethod
    def setup_class(cls):
        # Dataset
        endog = (np.log(macrodata[['realgdp', 'realcons']]).diff().iloc[1:] * 400)

        # Model
        cls.model = DynamicFactor(endog, k_factors=1, factor_order=2)
        cls.model.update(cls.model.start_params)
        cls.ssm = cls.model.ssm
        cls.ssm.filter_univariate = True

        # Exact diffuse
        init1 = Initialization(cls.ssm.k_states, 'diffuse')
        cls.ssm.initialize(init1)
        cls.results_a = cls.ssm.filter()
        cls.d = cls.results_a.nobs_diffuse

        # Approximate diffuse
        init2 = Initialization(cls.ssm.k_states, 'approximate_diffuse')
        cls.ssm.initialize(init2)
        cls.results_b = cls.ssm.filter()

    def test_initialization(self):
        assert_allclose(self.results_a.initial_state_cov, 0)
        assert_allclose(self.results_a.initial_diffuse_state_cov, np.eye(2))

        assert_allclose(self.results_b.initial_state_cov, np.eye(2) * 1e6)
        assert_allclose(self.results_b.initial_diffuse_state_cov, 0)

    def test_nobs_diffuse(self):
        assert_allclose(self.d, 2)


class TestDFMCollapsed(CheckApproximateDiffuse):
    @classmethod
    def setup_class(cls):
        # Dataset
        endog = (np.log(macrodata[['realgdp', 'realcons']]).diff().iloc[1:] * 400)

        # Model
        cls.model = DynamicFactor(endog, k_factors=1, factor_order=1)
        cls.model.update(cls.model.start_params)
        cls.ssm = cls.model.ssm
        cls.ssm.filter_univariate = True
        cls.ssm.filter_collapsed = True

        # Exact diffuse
        init1 = Initialization(cls.ssm.k_states, 'diffuse')
        cls.ssm.initialize(init1)
        cls.results_a = cls.ssm.filter()
        cls.d = cls.results_a.nobs_diffuse

        # Approximate diffuse
        init2 = Initialization(cls.ssm.k_states, 'approximate_diffuse')
        cls.ssm.initialize(init2)
        cls.results_b = cls.ssm.filter()

    def test_initialization(self):
        assert_allclose(self.results_a.initial_state_cov, 0)
        assert_allclose(self.results_a.initial_diffuse_state_cov, 1)

        assert_allclose(self.results_b.initial_state_cov, 1e6)
        assert_allclose(self.results_b.initial_diffuse_state_cov, 0)

    def test_nobs_diffuse(self):
        assert_allclose(self.d, 1)


class CheckKFAS(Check):
    @classmethod
    def setup_class(cls, results_path):
        # Dimensions
        ssm = cls.ssm
        n = ssm.nobs
        p = ssm.k_endog
        m = ssm.k_states
        r = ssm.k_posdef

        # Extract the different pieces of output from KFAS
        kfas = pd.read_csv(results_path)
        components = [('r', m), ('r0', m), ('r1', m), ('N', 4), ('m', p),
                      ('v', p), ('F', p), ('Finf', p), ('a', m), ('P', 1),
                      ('Pinf', 1), ('att', m), ('Ptt', 1),
                      ('alphahat', m), ('V', 1), ('muhat', p),
                      ('V_mu', 1), ('etahat', r), ('V_eta', 1), ('epshat', p),
                      ('V_eps', p), ('llf', 1)]
        dta = {}
        ix = 0
        for key, length in components:
            dta[key] = kfas.iloc[:, ix:ix + length].fillna(0)
            dta[key].name = None
            ix += length

        # Reformat the KFAS output to compare with statsmodels output
        res = Bunch()
        d = len(dta['Pinf'].dropna())
        
        # forecasts
        res['forecasts'] = dta['m'].values[:n].T
        res['forecasts_error'] = dta['v'].values[:n].T
        res['forecasts_error_cov'] = np.c_[
            [np.diag(x) for y, x in dta['F'].iloc[:n].iterrows()]].T
        res['forecasts_error_diffuse_cov'] = np.c_[
            [np.diag(x) for y, x in dta['Finf'].iloc[:n].iterrows()]].T

        # filtered
        res['filtered_state'] = dta['att'].values[:n].T
        # (this is actually a 1-dimension array with the sum of each matrix)
        res['filtered_state_cov'] = dta['Ptt'].values[:n].squeeze()
        # predicted
        res['predicted_state'] = dta['a'].values.T
        # (this is actually a 1-dimension array with the sum of each matrix)
        res['predicted_state_cov'] = dta['P'].values.squeeze()
        res['predicted_diffuse_state_cov'] = dta['Pinf'].values
        # loglike
        # Note: KFAS only gives the total loglikelihood
        res['llf_obs'] = dta['llf'].values[0, 0]

        # smoothed
        res['smoothed_state'] = dta['alphahat'].values[:n].T
        # (this is actually a 1-dimension array with the sum of each matrix)
        res['smoothed_state_cov'] = dta['V'].values[:n].squeeze()

        res['smoothed_measurement_disturbance'] = dta['epshat'].values[:n].T
        res['smoothed_measurement_disturbance_cov'] = np.c_[
            [np.diag(x) for y, x in dta['V_eps'].iloc[:n].iterrows()]].T
        res['smoothed_state_disturbance'] = dta['etahat'].T
        # (this is actually a 1-dimension array with the sum of each matrix)
        res['smoothed_state_disturbance_cov'] = dta['V_eta'].squeeze()

        # scaled smoothed estimator
        # Note: we store both r and r0 together as "scaled smoothed estimator"
        # while "scaled smoothed diffuse estimator" corresponds to r1
        res['scaled_smoothed_estimator'] = np.c_[dta['r0'][:d].T,
                                                 dta['r'][d:].T].T
        res['scaled_smoothed_diffuse_estimator'] = dta['r1'].values.T
        # Note: we store N and N0 together as "scaled smoothed estimator cov"
        # while N1 is "scaled smoothed diffuse1 estimator cov"
        # and N2 is "scaled smoothed diffuse2 estimator cov"
        # dta['N'] has columns [N, N0, N1, N2]
        N, N0, N1, N2 = [v for k, v in dta['N'].items()]
        # (these are actually 1-dimension arrays with the sum of each matrix)
        res['scaled_smoothed_estimator_cov'] = np.r_[N0[:d], N1[d:]]
        res['scaled_smoothed_diffuse1_estimator_cov'] = N1.values.squeeze()
        res['scaled_smoothed_diffuse2_estimator_cov'] = N2.values.squeeze()

        # Save the results object for the tests
        cls.results_b = res

    def test_forecasts_error_diffuse_cov(self, d=None, rtol_diffuse=1e-5):
        actual = self.results_a.forecasts_error_diffuse_cov.T
        desired = self.results_b.forecasts_error_diffuse_cov.T
        if desired.ndim == 1:
            actual = np.sum(actual, axis=0)
        if d is not None and rtol_diffuse != np.inf:
            assert_allclose(actual[:d], desired[d:], rtol=rtol_diffuse)
        assert_allclose(actual[d:], desired[d:])

    # Skipped because KFAS v1.3.1 has a bug for these matrices (they are not
    # even symmetric)
    @skip
    def test_filtered_state_cov(self):
        pass

    # KFAS v1.3.1 seems to compute the loglikelihood incorrectly, so we correct
    # for it here
    def test_loglike(self):
        kfas = self.results_b.llf_obs
        # We need to add back in the constant term for all of the non-missing
        # diffuse observations
        nonmissing = self.ssm.k_endog - np.isnan(self.ssm.endog).sum(axis=0)
        constant = -0.5 * np.log(2 * np.pi) * nonmissing
        desired = kfas + constant[:self.results_a.nobs_diffuse].sum()
        assert_allclose(np.sum(self.results_a.llf_obs), desired)


class TestLocalLevel(CheckKFAS):
    """
    This is the same model as in test_local_level_analytic, above, but the test
    is against the output of the R package KFAS
    """
    @classmethod
    def setup_class(cls):
        # Dataset
        y1 = 10.2394
        sigma2_y = 1.993
        sigma2_mu = 8.253
        endog = np.r_[y1, [1] * 9]

        # Model
        cls.model = UnobservedComponents(endog, 'llevel')
        cls.model.update([sigma2_y, sigma2_mu])
        cls.ssm = cls.model.ssm

        # Exact diffuse
        init1 = Initialization(cls.ssm.k_states, 'diffuse')
        cls.ssm.initialize(init1)
        cls.results_a = cls.ssm.filter()
        cls.d = cls.results_a.nobs_diffuse

        # Setup the base class with the results object
        results_path = os.path.join(current_path, 'results',
                                    'results_exact_initial_local_level_R.csv')

        super(TestLocalLevel, cls).setup_class(results_path)

    def test_loglike(self):
        # Test directly against Stata output
        # See results/test_exact_diffuse_filtering_stata.do for code to
        # reproduce these figures
        desired = -23.9352603142740605
        assert_allclose(self.results_a.llf_obs.sum(), desired)

        # Also call the parent, which tests against a corrected term from KFAS
        super(TestLocalLevel, self).test_loglike()


class TestLocalLinearTrend(CheckKFAS):
    """
    This is the same model as in test_local_linear_trend_analytic, above, but
    the test is against the output of the R package KFAS
    """
    @classmethod
    def setup_class(cls, missing=False):
        y1 = 10.2394
        y2 = 4.2039 if not missing else np.nan
        y3 = 6.123123
        sigma2_y = 1.993
        sigma2_mu = 8.253
        sigma2_beta = 2.334
        endog = np.r_[y1, y2, y3, [1] * 7]

        # Model
        cls.model = UnobservedComponents(endog, 'lltrend')
        cls.model.update([sigma2_y, sigma2_mu, sigma2_beta])
        cls.ssm = cls.model.ssm

        # Exact diffuse
        init1 = Initialization(cls.ssm.k_states, 'diffuse')
        cls.ssm.initialize(init1)
        cls.results_a = cls.ssm.filter()
        cls.d = cls.results_a.nobs_diffuse

        # Setup the base class with the results object
        if not missing:
            results_path = os.path.join(
                current_path, 'results',
                'results_exact_initial_local_linear_trend_R.csv')
        else:
            results_path = os.path.join(
                current_path, 'results',
                'results_exact_initial_local_linear_trend_missing_R.csv')

        super(TestLocalLinearTrend, cls).setup_class(results_path)

    def test_loglike(self):
        # Test directly against Stata output
        # See results/test_exact_diffuse_filtering_stata.do for code to
        # reproduce these figures
        desired = -22.9743755748041529
        assert_allclose(self.results_a.llf_obs.sum(), desired)

        # Also call the parent, which tests against a corrected term from KFAS
        super(TestLocalLinearTrend, self).test_loglike()


class TestLocalLinearTrendMissing(TestLocalLinearTrend):
    @classmethod
    def setup_class(cls):
        super(TestLocalLinearTrendMissing, cls).setup_class(missing=True)

        # Replace 0 with NaN in the KFAS forecast error associated with the
        # missing observation
        cls.results_b.forecasts_error[0, 1] = np.nan

    def test_loglike(self):
        # We cannot test against output from Stata since they don't allow
        # missing values in either ucm or sspace.

        # Call the parent, which tests against a corrected term from KFAS
        super(TestLocalLinearTrend, self).test_loglike()


class TestCommonLevel(CheckKFAS):
    """
    This is the same model as in test_common_level_analytic, above, but
    the test is against the output of the R package KFAS
    """
    @classmethod
    def setup_class(cls):
        # Analytic test using results from Koopman (1997), section 5.3
        y11 = 10.2394
        y21 = 8.2304
        theta = 0.1111
        sigma2_1 = 1
        sigma_12 = 0
        sigma2_2 = 1
        sigma2_mu = 3.2324

        # Construct the basic representation
        mod = KalmanFilter(k_endog=2, k_states=2, k_posdef=1)
        mod.filter_univariate = True
        endog = np.column_stack([np.r_[y11, [1] * 9], np.r_[y21, [1] * 9]])
        mod.bind(endog.T)
        mod.initialize(Initialization(mod.k_states, initialization_type='diffuse'))

        # Fill in the system matrices for a common trend model
        mod['design'] = np.array([[1, 0],
                                  [theta, 1]])
        mod['obs_cov'] = np.eye(2)
        mod['transition'] = np.eye(2)
        mod['selection', 0, 0] = 1
        mod['state_cov', 0, 0] = sigma2_mu

        # Save model, results
        cls.ssm = mod
        cls.results_a = mod.filter()

        # Setup the base class with the results object
        results_path = os.path.join(
            current_path, 'results',
            'results_exact_initial_common_level_R.csv')

        super(TestCommonLevel, cls).setup_class(results_path)

    # KFAS stores forecasts from the multivariate filter even in the univariate
    # case, so we can't check against them here
    @skip
    def test_forecasts(self, d=None, rtol_diffuse=np.inf):
        pass

    def test_loglike(self):
        # Test directly against Stata output
        # See results/test_exact_diffuse_filtering_stata.do for code to
        # reproduce these figures
        desired = -53.7830389463984773
        assert_allclose(self.results_a.llf_obs.sum(), desired)

        # Also call the parent, which tests against a corrected term from KFAS
        super(TestCommonLevel, self).test_loglike()
