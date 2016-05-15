"""
Tests for univariate treatment of multivariate models

TODO skips the tests for measurement disturbance and measurement disturbance
covariance, which do not pass. The univariate smoother *appears* to be
correctly implemented against Durbin and Koopman (2012) chapter 6, yet still
gives a different answer from the conventional smoother. It's not clear if
this is intended (i.e. it has to be at least slightly different, since the
conventional smoother can return a non-diagonal covariance matrix whereas the
univariate smoother must return a diagonal covariance matrix).

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import pandas as pd
import os

from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.statespace.tools import compatibility_mode
from statsmodels.tsa.statespace.tests.results import results_kalman_filter
from numpy.testing import assert_almost_equal, assert_allclose
from nose.exc import SkipTest

current_path = os.path.dirname(os.path.abspath(__file__))


class TestClark1989(object):
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
        if compatibility_mode:
            raise SkipTest('Collapsed methods not available.')

        cls.true = results_kalman_filter.uc_bi
        cls.true_states = pd.DataFrame(cls.true['states'])

        # GDP and Unemployment, Quarterly, 1948.1 - 1995.3
        data = pd.DataFrame(
            cls.true['data'],
            index=pd.date_range('1947-01-01', '1995-07-01', freq='QS'),
            columns=['GDP', 'UNEMP']
        )[4:]
        data['GDP'] = np.log(data['GDP'])
        data['UNEMP'] = (data['UNEMP']/100)

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
            cls.true['parameters'],
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
        n_disturbance_variates = (
            (cls.model.k_endog + cls.model.k_posdef) * cls.model.nobs
        )
        cls.conventional_sim = cls.model.simulation_smoother(
            disturbance_variates=np.zeros(n_disturbance_variates),
            initial_state_variates=np.zeros(cls.model.k_states)
        )

        # Univariate filtering, smoothing, and simulation smoothing
        cls.model.filter_univariate = True
        cls.univariate_results = cls.model.smooth()
        cls.univariate_sim = cls.model.simulation_smoother(
            disturbance_variates=np.zeros(n_disturbance_variates),
            initial_state_variates=np.zeros(cls.model.k_states)
        )

    def test_using_univariate(self):
        # Regression test to make sure the univariate_results actually
        # used the univariate Kalman filtering approach (i.e. that the flag
        # being set actually caused the filter to not use the conventional
        # filter)
        assert not self.conventional_results.filter_univariate
        assert self.univariate_results.filter_univariate

        assert_allclose(
            self.conventional_results.forecasts_error_cov[1,1,0],
            143.03724478030821
        )
        assert_allclose(
            self.univariate_results.forecasts_error_cov[1,1,0],
            120.66208525029386
        )

    def test_forecasts(self):
        assert_almost_equal(
            self.conventional_results.forecasts[0,:],
            self.univariate_results.forecasts[0,:], 9
        )

    def test_forecasts_error(self):
        assert_almost_equal(
            self.conventional_results.forecasts_error[0,:],
            self.univariate_results.forecasts_error[0,:], 9
        )

    def test_forecasts_error_cov(self):
        assert_almost_equal(
            self.conventional_results.forecasts_error_cov[0,0,:],
            self.univariate_results.forecasts_error_cov[0,0,:], 9
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

    @SkipTest
    def test_smoothed_measurement_disturbance(self):
        assert_almost_equal(
            self.conventional_results.smoothed_measurement_disturbance,
            self.univariate_results.smoothed_measurement_disturbance, 9
        )

    @SkipTest
    def test_smoothed_measurement_disturbance_cov(self):
        assert_almost_equal(
            self.conventional_results.smoothed_measurement_disturbance_cov,
            self.univariate_results.smoothed_measurement_disturbance_cov, 9
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
        assert_almost_equal(
            self.conventional_sim.simulated_state,
            self.univariate_sim.simulated_state, 9
        )

    def test_simulation_smoothed_measurement_disturbance(self):
        assert_almost_equal(
            self.conventional_sim.simulated_measurement_disturbance,
            self.univariate_sim.simulated_measurement_disturbance, 9
        )

    def test_simulation_smoothed_state_disturbance(self):
        assert_almost_equal(
            self.conventional_sim.simulated_state_disturbance,
            self.univariate_sim.simulated_state_disturbance, 9
        )


class TestClark1989Alternate(TestClark1989):
    @classmethod
    def setup_class(cls, *args, **kwargs):
        super(TestClark1989Alternate, cls).setup_class(alternate_timing=True, *args, **kwargs)

    def test_using_alterate(self):
        assert(self.model._kalman_filter.filter_timing == 1)
