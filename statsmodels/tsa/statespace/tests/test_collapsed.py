"""
Tests for collapsed observation vector

These tests cannot be run for the Clark 1989 model since the dimension of
observations (2) is smaller than the number of states (6).

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
from numpy.testing import assert_equal, assert_allclose
from nose.exc import SkipTest

current_path = os.path.dirname(os.path.abspath(__file__))


class Trivariate(object):
    """
    Tests collapsing three-dimensional observation data to two-dimensional
    """
    @classmethod
    def setup_class(cls, dtype=float, alternate_timing=False, **kwargs):
        cls.results = results_kalman_filter.uc_bi

        if compatibility_mode:
            raise SkipTest('Collapsed methods not available.')

        # GDP and Unemployment, Quarterly, 1948.1 - 1995.3
        data = pd.DataFrame(
            cls.results['data'],
            index=pd.date_range('1947-01-01', '1995-07-01', freq='QS'),
            columns=['GDP', 'UNEMP']
        )[4:]
        data['GDP'] = np.log(data['GDP'])
        data['UNEMP'] = (data['UNEMP']/100)
        data['X'] = np.exp(data['GDP']) * data['UNEMP']

        k_states = 2
        cls.mlemodel = MLEModel(data, k_states=k_states, **kwargs)
        cls.model = cls.mlemodel.ssm
        if alternate_timing:
            cls.model.timing_init_filtered = True

        # Statespace representation
        cls.model['selection'] = np.eye(cls.model.k_states)

        # Update matrices with test parameters
        cls.model['design'] = np.array([[0.5, 0.2],
                                         [0,   0.8],
                                         [1,  -0.5]])
        cls.model['transition'] = np.array([[0.4, 0.5],
                                             [1,   0]])
        cls.model['obs_cov'] = np.diag([0.2, 1.1, 0.5])
        cls.model['state_cov'] = np.diag([2., 1])

        # Initialization
        cls.model.initialize_approximate_diffuse()

    def test_using_collapsed(self):
        # Test to make sure the results_b actually used a collapsed Kalman
        # filtering approach (i.e. that the flag being set actually caused the
        # filter to not use the conventional filter)

        assert not self.results_a.filter_collapsed
        assert self.results_b.filter_collapsed

        assert self.results_a.collapsed_forecasts is None
        assert self.results_b.collapsed_forecasts is not None

        assert_equal(self.results_a.forecasts.shape[0], 3)
        assert_equal(self.results_b.collapsed_forecasts.shape[0], 2)

    def test_forecasts(self):
        assert_allclose(
            self.results_a.forecasts[0,:],
            self.results_b.forecasts[0,:],
        )

    def test_forecasts_error(self):
        assert_allclose(
            self.results_a.forecasts_error[0,:],
            self.results_b.forecasts_error[0,:]
        )

    def test_forecasts_error_cov(self):
        assert_allclose(
            self.results_a.forecasts_error_cov[0,0,:],
            self.results_b.forecasts_error_cov[0,0,:]
        )

    def test_filtered_state(self):
        assert_allclose(
            self.results_a.filtered_state,
            self.results_b.filtered_state
        )

    def test_filtered_state_cov(self):
        assert_allclose(
            self.results_a.filtered_state_cov,
            self.results_b.filtered_state_cov
        )

    def test_predicted_state(self):
        assert_allclose(
            self.results_a.predicted_state,
            self.results_b.predicted_state
        )

    def test_predicted_state_cov(self):
        assert_allclose(
            self.results_a.predicted_state_cov,
            self.results_b.predicted_state_cov
        )

    def test_loglike(self):
        assert_allclose(
            self.results_a.llf_obs,
            self.results_b.llf_obs
        )

    def test_smoothed_states(self):
        assert_allclose(
            self.results_a.smoothed_state,
            self.results_b.smoothed_state
        )

    def test_smoothed_states_cov(self):
        assert_allclose(
            self.results_a.smoothed_state_cov,
            self.results_b.smoothed_state_cov,
            rtol=1e-4
        )

    @SkipTest
    # Skipped because "measurement" refers to different things; even different
    # dimensions
    def test_smoothed_measurement_disturbance(self):
        assert_allclose(
            self.results_a.smoothed_measurement_disturbance,
            self.results_b.smoothed_measurement_disturbance
        )

    @SkipTest
    # Skipped because "measurement" refers to different things; even different
    # dimensions
    def test_smoothed_measurement_disturbance_cov(self):
        assert_allclose(
            self.results_a.smoothed_measurement_disturbance_cov,
            self.results_b.smoothed_measurement_disturbance_cov
        )

    def test_smoothed_state_disturbance(self):
        assert_allclose(
            self.results_a.smoothed_state_disturbance,
            self.results_b.smoothed_state_disturbance
        )

    def test_smoothed_state_disturbance_cov(self):
        assert_allclose(
            self.results_a.smoothed_state_disturbance_cov,
            self.results_b.smoothed_state_disturbance_cov
        )

    def test_simulation_smoothed_state(self):
        assert_allclose(
            self.sim_a.simulated_state,
            self.sim_a.simulated_state
        )

    def test_simulation_smoothed_measurement_disturbance(self):
        assert_allclose(
            self.sim_a.simulated_measurement_disturbance,
            self.sim_a.simulated_measurement_disturbance
        )

    def test_simulation_smoothed_state_disturbance(self):
        assert_allclose(
            self.sim_a.simulated_state_disturbance,
            self.sim_a.simulated_state_disturbance
        )

class TestTrivariateConventional(Trivariate):

    @classmethod
    def setup_class(cls, dtype=float, **kwargs):
        super(TestTrivariateConventional, cls).setup_class(dtype, **kwargs)
        n_disturbance_variates = (
            (cls.model.k_endog + cls.model.k_posdef) * cls.model.nobs
        )

        # Collapsed filtering, smoothing, and simulation smoothing
        cls.model.filter_conventional = True
        cls.model.filter_collapsed = True
        cls.results_b = cls.model.smooth()
        cls.sim_b = cls.model.simulation_smoother(
            disturbance_variates=np.zeros(n_disturbance_variates),
            initial_state_variates=np.zeros(cls.model.k_states)
        )

        # Conventional filtering, smoothing, and simulation smoothing
        cls.model.filter_collapsed = False
        cls.results_a = cls.model.smooth()
        cls.sim_a = cls.model.simulation_smoother(
            disturbance_variates=np.zeros(n_disturbance_variates),
            initial_state_variates=np.zeros(cls.model.k_states)
        )

class TestTrivariateConventionalAlternate(TestTrivariateConventional):
    @classmethod
    def setup_class(cls, *args, **kwargs):
        super(TestTrivariateConventionalAlternate, cls).setup_class(alternate_timing=True, *args, **kwargs)

    def test_using_alterate(self):
        assert(self.model._kalman_filter.filter_timing == 1)


class TestTrivariateConventionalPartialMissing(Trivariate):
    @classmethod
    def setup_class(cls, dtype=float, **kwargs):
        super(TestTrivariateConventionalPartialMissing, cls).setup_class(dtype, **kwargs)
        n_disturbance_variates = (
            (cls.model.k_endog + cls.model.k_posdef) * cls.model.nobs
        )

        # Set partially missing data
        cls.model.endog[:2, 10:180] = np.nan

        # Collapsed filtering, smoothing, and simulation smoothing
        cls.model.filter_conventional = True
        cls.model.filter_collapsed = True
        cls.results_b = cls.model.smooth()
        cls.sim_b = cls.model.simulation_smoother(
            disturbance_variates=np.zeros(n_disturbance_variates),
            initial_state_variates=np.zeros(cls.model.k_states)
        )

        # Conventional filtering, smoothing, and simulation smoothing
        cls.model.filter_collapsed = False
        cls.results_a = cls.model.smooth()
        cls.sim_a = cls.model.simulation_smoother(
            disturbance_variates=np.zeros(n_disturbance_variates),
            initial_state_variates=np.zeros(cls.model.k_states)
        )


class TestTrivariateConventionalPartialMissingAlternate(TestTrivariateConventionalPartialMissing):
    @classmethod
    def setup_class(cls, *args, **kwargs):
        super(TestTrivariateConventionalPartialMissingAlternate, cls).setup_class(alternate_timing=True, *args, **kwargs)

    def test_using_alterate(self):
        assert(self.model._kalman_filter.filter_timing == 1)


class TestTrivariateConventionalAllMissing(Trivariate):
    @classmethod
    def setup_class(cls, dtype=float, **kwargs):
        super(TestTrivariateConventionalAllMissing, cls).setup_class(dtype, **kwargs)
        n_disturbance_variates = (
            (cls.model.k_endog + cls.model.k_posdef) * cls.model.nobs
        )

        # Set partially missing data
        cls.model.endog[:, 10:180] = np.nan

        # Collapsed filtering, smoothing, and simulation smoothing
        cls.model.filter_conventional = True
        cls.model.filter_collapsed = True
        cls.results_b = cls.model.smooth()
        cls.sim_b = cls.model.simulation_smoother(
            disturbance_variates=np.zeros(n_disturbance_variates),
            initial_state_variates=np.zeros(cls.model.k_states)
        )

        # Conventional filtering, smoothing, and simulation smoothing
        cls.model.filter_collapsed = False
        cls.results_a = cls.model.smooth()
        cls.sim_a = cls.model.simulation_smoother(
            disturbance_variates=np.zeros(n_disturbance_variates),
            initial_state_variates=np.zeros(cls.model.k_states)
        )


class TestTrivariateConventionalAllMissingAlternate(TestTrivariateConventionalAllMissing):
    @classmethod
    def setup_class(cls, *args, **kwargs):
        super(TestTrivariateConventionalAllMissingAlternate, cls).setup_class(alternate_timing=True, *args, **kwargs)

    def test_using_alterate(self):
        assert(self.model._kalman_filter.filter_timing == 1)


class TestTrivariateUnivariate(Trivariate):
    @classmethod
    def setup_class(cls, dtype=float, **kwargs):
        super(TestTrivariateUnivariate, cls).setup_class(dtype, **kwargs)
        n_disturbance_variates = (
            (cls.model.k_endog + cls.model.k_posdef) * cls.model.nobs
        )

        # Collapsed filtering, smoothing, and simulation smoothing
        cls.model.filter_univariate = True
        cls.model.filter_collapsed = True
        cls.results_b = cls.model.smooth()
        cls.sim_b = cls.model.simulation_smoother(
            disturbance_variates=np.zeros(n_disturbance_variates),
            initial_state_variates=np.zeros(cls.model.k_states)
        )

        # Univariate filtering, smoothing, and simulation smoothing
        cls.model.filter_collapsed = False
        cls.results_a = cls.model.smooth()
        cls.sim_a = cls.model.simulation_smoother(
            disturbance_variates=np.zeros(n_disturbance_variates),
            initial_state_variates=np.zeros(cls.model.k_states)
        )


class TestTrivariateUnivariateAlternate(TestTrivariateUnivariate):
    @classmethod
    def setup_class(cls, *args, **kwargs):
        super(TestTrivariateUnivariateAlternate, cls).setup_class(alternate_timing=True, *args, **kwargs)

    def test_using_alterate(self):
        assert(self.model._kalman_filter.filter_timing == 1)


class TestTrivariateUnivariatePartialMissing(Trivariate):
    @classmethod
    def setup_class(cls, dtype=float, **kwargs):
        super(TestTrivariateUnivariatePartialMissing, cls).setup_class(dtype, **kwargs)
        n_disturbance_variates = (
            (cls.model.k_endog + cls.model.k_posdef) * cls.model.nobs
        )

        # Set partially missing data
        cls.model.endog[:2, 10:180] = np.nan

        # Collapsed filtering, smoothing, and simulation smoothing
        cls.model.filter_univariate = True
        cls.model.filter_collapsed = True
        cls.results_b = cls.model.smooth()
        cls.sim_b = cls.model.simulation_smoother(
            disturbance_variates=np.zeros(n_disturbance_variates),
            initial_state_variates=np.zeros(cls.model.k_states)
        )

        # Univariate filtering, smoothing, and simulation smoothing
        cls.model.filter_collapsed = False
        cls.results_a = cls.model.smooth()
        cls.sim_a = cls.model.simulation_smoother(
            disturbance_variates=np.zeros(n_disturbance_variates),
            initial_state_variates=np.zeros(cls.model.k_states)
        )


class TestTrivariateUnivariatePartialMissingAlternate(TestTrivariateUnivariatePartialMissing):
    @classmethod
    def setup_class(cls, *args, **kwargs):
        super(TestTrivariateUnivariatePartialMissingAlternate, cls).setup_class(alternate_timing=True, *args, **kwargs)

    def test_using_alterate(self):
        assert(self.model._kalman_filter.filter_timing == 1)


class TestTrivariateUnivariateAllMissing(Trivariate):
    @classmethod
    def setup_class(cls, dtype=float, **kwargs):
        super(TestTrivariateUnivariateAllMissing, cls).setup_class(dtype, **kwargs)
        n_disturbance_variates = (
            (cls.model.k_endog + cls.model.k_posdef) * cls.model.nobs
        )

        # Set partially missing data
        cls.model.endog[:, 10:180] = np.nan

        # Univariate filtering, smoothing, and simulation smoothing
        cls.model.filter_univariate = True
        cls.model.filter_collapsed = True
        cls.results_b = cls.model.smooth()
        cls.sim_b = cls.model.simulation_smoother(
            disturbance_variates=np.zeros(n_disturbance_variates),
            initial_state_variates=np.zeros(cls.model.k_states)
        )

        # Conventional filtering, smoothing, and simulation smoothing
        cls.model.filter_collapsed = False
        cls.results_a = cls.model.smooth()
        cls.sim_a = cls.model.simulation_smoother(
            disturbance_variates=np.zeros(n_disturbance_variates),
            initial_state_variates=np.zeros(cls.model.k_states)
        )

class TestTrivariateUnivariateAllMissingAlternate(TestTrivariateUnivariateAllMissing):
    @classmethod
    def setup_class(cls, *args, **kwargs):
        super(TestTrivariateUnivariateAllMissingAlternate, cls).setup_class(alternate_timing=True, *args, **kwargs)

    def test_using_alterate(self):
        assert(self.model._kalman_filter.filter_timing == 1)
