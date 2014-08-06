"""
Tests for representation module

Author: Chad Fulton
License: Simplified-BSD

References
----------

Kim, Chang-Jin, and Charles R. Nelson. 1999.
"State-Space Models with Regime Switching:
Classical and Gibbs-Sampling Approaches with Applications".
MIT Press Books. The MIT Press.
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import pandas as pd
import os

import statsmodels.tsa.statespace as ss
from .results import results_kalman_filter
from numpy.testing import assert_almost_equal
from nose.exc import SkipTest

current_path = os.path.dirname(os.path.abspath(__file__))


class Clark1987(object):
    """
    Clark's (1987) univariate unobserved components model of real GDP (as
    presented in Kim and Nelson, 1999)

    Test data produced using GAUSS code described in Kim and Nelson (1999) and
    found at http://econ.korea.ac.kr/~cjkim/SSMARKOV.htm

    See `results.results_kalman_filter` for more information.
    """
    def __init__(self, dtype=float, **kwargs):
        self.true = results_kalman_filter.uc_uni
        self.true_states = pd.DataFrame(self.true['states'])

        # GDP, Quarterly, 1947.1 - 1995.3
        data = pd.DataFrame(
            self.true['data'],
            index=pd.date_range('1947-01-01', '1995-07-01', freq='QS'),
            columns=['GDP']
        )
        data['lgdp'] = np.log(data['GDP'])

        # Construct the statespace representation
        k_states = 4
        self.model = ss.Representation(data['lgdp'], k_states=k_states,
                                       **kwargs)

        self.model.design[:, :, 0] = [1, 1, 0, 0]
        self.model.transition[([0, 0, 1, 1, 2, 3],
                               [0, 3, 1, 2, 1, 3],
                               [0, 0, 0, 0, 0, 0])] = [1, 1, 0, 0, 1, 1]
        self.model.selection = np.eye(self.model.k_states)

        # Update matrices with given parameters
        (sigma_v, sigma_e, sigma_w, phi_1, phi_2) = np.array(
            self.true['parameters']
        )
        self.model.transition[([1, 1], [1, 2], [0, 0])] = [phi_1, phi_2]
        self.model.state_cov[
            np.diag_indices(k_states)+(np.zeros(k_states, dtype=int),)] = [
            sigma_v**2, sigma_e**2, 0, sigma_w**2
        ]

        # Initialization
        initial_state = np.zeros((k_states,))
        initial_state_cov = np.eye(k_states)*100

        # Initialization: modification
        initial_state_cov = np.dot(
            np.dot(self.model.transition[:, :, 0], initial_state_cov),
            self.model.transition[:, :, 0].T
        )
        self.model.initialize_known(initial_state, initial_state_cov)

    def run_filter(self):
        # Filter the data
        self.results = self.model.filter()

    def test_loglike(self):
        assert_almost_equal(
            self.results.loglikelihood[self.true['start']:].sum(),
            self.true['loglike'], 5
        )

    def test_filtered_state(self):
        assert_almost_equal(
            self.results.filtered_state[0][self.true['start']:],
            self.true_states.iloc[:, 0], 4
        )
        assert_almost_equal(
            self.results.filtered_state[1][self.true['start']:],
            self.true_states.iloc[:, 1], 4
        )
        assert_almost_equal(
            self.results.filtered_state[3][self.true['start']:],
            self.true_states.iloc[:, 2], 4
        )


class TestClark1987Single(Clark1987):
    """
    Basic single precision test for the loglikelihood and filtered states.
    """
    def __init__(self):
        raise SkipTest('Not implemented')
        super(TestClark1987Single, self).__init__(
            dtype=np.float32, conserve_memory=0
        )
        self.run_filter()


class TestClark1987Double(Clark1987):
    """
    Basic double precision test for the loglikelihood and filtered states.
    """
    def __init__(self):
        super(TestClark1987Double, self).__init__(
            dtype=float, conserve_memory=0
        )
        self.run_filter()


class TestClark1987SingleComplex(Clark1987):
    """
    Basic single precision complex test for the loglikelihood and filtered
    states.
    """
    def __init__(self):
        raise SkipTest('Not implemented')
        super(TestClark1987SingleComplex, self).__init__(
            dtype=np.complex64, conserve_memory=0
        )
        self.run_filter()


class TestClark1987DoubleComplex(Clark1987):
    """
    Basic double precision complex test for the loglikelihood and filtered
    states.
    """
    def __init__(self):
        super(TestClark1987DoubleComplex, self).__init__(
            dtype=complex, conserve_memory=0
        )
        self.run_filter()


class TestClark1987Conserve(Clark1987):
    """
    Memory conservation test for the loglikelihood and filtered states.
    """
    def __init__(self):
        super(TestClark1987Conserve, self).__init__(
            dtype=float, conserve_memory=0x01 | 0x02
        )
        self.run_filter()


class Clark1987Forecast(Clark1987):
    """
    Forecasting test for the loglikelihood and filtered states.
    """
    def __init__(self, dtype=float, nforecast=100, conserve_memory=0):
        super(Clark1987Forecast, self).__init__(
            dtype=dtype, conserve_memory=conserve_memory
        )
        self.nforecast = nforecast

        # Add missing observations to the end (to forecast)
        self.model.endog = np.array(
            np.r_[self.model.endog[0, :], [np.nan]*nforecast],
            ndmin=2, dtype=dtype, order="F"
        )
        self.model.nobs = self.model.endog.shape[1]

    def test_filtered_state(self):
        assert_almost_equal(
            self.results.filtered_state[0][self.true['start']:-self.nforecast],
            self.true_states.iloc[:, 0], 4
        )
        assert_almost_equal(
            self.results.filtered_state[1][self.true['start']:-self.nforecast],
            self.true_states.iloc[:, 1], 4
        )
        assert_almost_equal(
            self.results.filtered_state[3][self.true['start']:-self.nforecast],
            self.true_states.iloc[:, 2], 4
        )


class TestClark1987ForecastDouble(Clark1987Forecast):
    """
    Basic double forecasting test for the loglikelihood and filtered states.
    """
    def __init__(self):
        super(TestClark1987ForecastDouble, self).__init__()
        self.run_filter()


class TestClark1987ForecastDoubleComplex(Clark1987Forecast):
    """
    Basic double complex forecasting test for the loglikelihood and filtered
    states.
    """
    def __init__(self):
        super(TestClark1987ForecastDoubleComplex, self).__init__(
            dtype=complex
        )
        self.run_filter()


class TestClark1987ForecastConserve(Clark1987Forecast):
    """
    Memory conservation forecasting test for the loglikelihood and filtered
    states.
    """
    def __init__(self):
        super(TestClark1987ForecastConserve, self).__init__(
            dtype=float, conserve_memory=0x01 | 0x02
        )
        self.run_filter()


class TestClark1987ConserveAll(Clark1987):
    """
    Memory conservation forecasting test for the loglikelihood and filtered
    states.
    """
    def __init__(self):
        super(TestClark1987ConserveAll, self).__init__(
            dtype=float, conserve_memory=0x01 | 0x02 | 0x04 | 0x08
        )
        self.model.loglikelihood_burn = self.true['start']
        self.run_filter()

    def test_loglike(self):
        assert_almost_equal(
            self.results.loglikelihood[0], self.true['loglike'], 5
        )

    def test_filtered_state(self):
        end = self.true_states.shape[0]
        assert_almost_equal(
            self.results.filtered_state[0][-1],
            self.true_states.iloc[end-1, 0], 4
        )
        assert_almost_equal(
            self.results.filtered_state[1][-1],
            self.true_states.iloc[end-1, 1], 4
        )


class Clark1989(object):
    """
    Clark's (1989) bivariate unobserved components model of real GDP (as
    presented in Kim and Nelson, 1999)

    Tests two-dimensional observation data.

    Test data produced using GAUSS code described in Kim and Nelson (1999) and
    found at http://econ.korea.ac.kr/~cjkim/SSMARKOV.htm

    See `results.results_kalman_filter` for more information.
    """
    def __init__(self, dtype=float, **kwargs):
        self.true = results_kalman_filter.uc_bi
        self.true_states = pd.DataFrame(self.true['states'])

        # GDP and Unemployment, Quarterly, 1948.1 - 1995.3
        data = pd.DataFrame(
            self.true['data'],
            index=pd.date_range('1947-01-01', '1995-07-01', freq='QS'),
            columns=['GDP', 'UNEMP']
        )[4:]
        data['GDP'] = np.log(data['GDP'])
        data['UNEMP'] = (data['UNEMP']/100)

        k_states = 6
        self.model = ss.Representation(data, k_states=k_states, **kwargs)

        # Statespace representation
        self.model.design[:, :, 0] = [[1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]]
        self.model.transition[
            ([0, 0, 1, 1, 2, 3, 4, 5],
             [0, 4, 1, 2, 1, 2, 4, 5],
             [0, 0, 0, 0, 0, 0, 0, 0])
        ] = [1, 1, 0, 0, 1, 1, 1, 1]
        self.model.selection = np.eye(self.model.k_states)

        # Update matrices with given parameters
        (sigma_v, sigma_e, sigma_w, sigma_vl, sigma_ec,
         phi_1, phi_2, alpha_1, alpha_2, alpha_3) = np.array(
            self.true['parameters'],
        )
        self.model.design[([1, 1, 1], [1, 2, 3], [0, 0, 0])] = [
            alpha_1, alpha_2, alpha_3
        ]
        self.model.transition[([1, 1], [1, 2], [0, 0])] = [phi_1, phi_2]
        self.model.obs_cov[1, 1, 0] = sigma_ec**2
        self.model.state_cov[
            np.diag_indices(k_states)+(np.zeros(k_states, dtype=int),)] = [
            sigma_v**2, sigma_e**2, 0, 0, sigma_w**2, sigma_vl**2
        ]

        # Initialization
        initial_state = np.zeros((k_states,))
        initial_state_cov = np.eye(k_states)*100

        # Initialization: self.modelification
        initial_state_cov = np.dot(
            np.dot(self.model.transition[:, :, 0], initial_state_cov),
            self.model.transition[:, :, 0].T
        )
        self.model.initialize_known(initial_state, initial_state_cov)

    def run_filter(self):
        # Filter the data
        self.results = self.model.filter()

    def test_loglike(self):
        assert_almost_equal(
            self.results.loglikelihood[self.true['start']:].sum(),
            self.true['loglike'], 2
        )

    def test_filtered_state(self):
        assert_almost_equal(
            self.results.filtered_state[0][self.true['start']:],
            self.true_states.iloc[:, 0], 4
        )
        assert_almost_equal(
            self.results.filtered_state[1][self.true['start']:],
            self.true_states.iloc[:, 1], 4
        )
        assert_almost_equal(
            self.results.filtered_state[4][self.true['start']:],
            self.true_states.iloc[:, 2], 4
        )
        assert_almost_equal(
            self.results.filtered_state[5][self.true['start']:],
            self.true_states.iloc[:, 3], 4
        )


class TestClark1989(Clark1989):
    """
    Basic double precision test for the loglikelihood and filtered
    states with two-dimensional observation vector.
    """
    def __init__(self):
        super(TestClark1989, self).__init__(dtype=float, conserve_memory=0)
        self.run_filter()


class TestClark1989Conserve(Clark1989):
    """
    Memory conservation test for the loglikelihood and filtered states with
    two-dimensional observation vector.
    """
    def __init__(self):
        super(TestClark1989Conserve, self).__init__(
            dtype=float, conserve_memory=0x01 | 0x02
        )
        self.run_filter()


class Clark1989Forecast(Clark1989):
    """
    Memory conservation test for the loglikelihood and filtered states with
    two-dimensional observation vector.
    """
    def __init__(self, dtype=float, nforecast=100, conserve_memory=0):
        super(Clark1989Forecast, self).__init__(
            dtype=dtype, conserve_memory=conserve_memory
        )
        self.nforecast = nforecast

        # Add missing observations to the end (to forecast)
        self.model.endog = np.array(
            np.c_[
                self.model.endog,
                np.r_[[np.nan, np.nan]*nforecast].reshape(2, nforecast)
            ],
            ndmin=2, dtype=dtype, order="F"
        )
        self.model.nobs = self.model.endog.shape[1]

        self.run_filter()

    def test_filtered_state(self):
        assert_almost_equal(
            self.results.filtered_state[0][self.true['start']:-self.nforecast],
            self.true_states.iloc[:, 0], 4
        )
        assert_almost_equal(
            self.results.filtered_state[1][self.true['start']:-self.nforecast],
            self.true_states.iloc[:, 1], 4
        )
        assert_almost_equal(
            self.results.filtered_state[4][self.true['start']:-self.nforecast],
            self.true_states.iloc[:, 2], 4
        )
        assert_almost_equal(
            self.results.filtered_state[5][self.true['start']:-self.nforecast],
            self.true_states.iloc[:, 3], 4
        )


class TestClark1989ForecastDouble(Clark1989Forecast):
    """
    Basic double forecasting test for the loglikelihood and filtered states.
    """
    def __init__(self):
        super(TestClark1989ForecastDouble, self).__init__()
        self.run_filter()


class TestClark1989ForecastDoubleComplex(Clark1989Forecast):
    """
    Basic double complex forecasting test for the loglikelihood and filtered
    states.
    """
    def __init__(self):
        super(TestClark1989ForecastDoubleComplex, self).__init__(
            dtype=complex
        )
        self.run_filter()


class TestClark1989ForecastConserve(Clark1989Forecast):
    """
    Memory conservation forecasting test for the loglikelihood and filtered
    states.
    """
    def __init__(self):
        super(TestClark1989ForecastConserve, self).__init__(
            dtype=float, conserve_memory=0x01 | 0x02
        )
        self.run_filter()


class TestClark1989ConserveAll(Clark1989):
    """
    Memory conservation forecasting test for the loglikelihood and filtered
    states.
    """
    def __init__(self):
        super(TestClark1989ConserveAll, self).__init__(
            dtype=float, conserve_memory=0x01 | 0x02 | 0x04 | 0x08
        )
        self.model.loglikelihood_burn = self.true['start']
        self.run_filter()

    def test_loglike(self):
        assert_almost_equal(
            self.results.loglikelihood[0], self.true['loglike'], 2
        )

    def test_filtered_state(self):
        end = self.true_states.shape[0]
        assert_almost_equal(
            self.results.filtered_state[0][-1],
            self.true_states.iloc[end-1, 0], 4
        )
        assert_almost_equal(
            self.results.filtered_state[1][-1],
            self.true_states.iloc[end-1, 1], 4
        )
        assert_almost_equal(
            self.results.filtered_state[4][-1],
            self.true_states.iloc[end-1, 2], 4
        )
        assert_almost_equal(
            self.results.filtered_state[5][-1],
            self.true_states.iloc[end-1, 3], 4
        )