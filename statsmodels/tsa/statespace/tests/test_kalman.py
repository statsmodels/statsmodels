"""
Tests for _statespace module

Author: Chad Fulton
License: Simplified-BSD

References
----------

Kim, Chang-Jin, and Charles R. Nelson. 1999.
"State-Space Models with Regime Switching:
Classical and Gibbs-Sampling Approaches with Applications".
MIT Press Books. The MIT Press.

Hamilton, James D. 1994.
Time Series Analysis.
Princeton, N.J.: Princeton University Press.
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import pandas as pd
import os

try:
    from scipy.linalg.blas import find_best_blas_type
except ImportError:
    # Shim for SciPy 0.11, derived from tag=0.11 scipy.linalg.blas
    _type_conv = {'f': 's', 'd': 'd', 'F': 'c', 'D': 'z', 'G': 'z'}

    def find_best_blas_type(arrays):
        dtype, index = max(
            [(ar.dtype, i) for i, ar in enumerate(arrays)])
        prefix = _type_conv.get(dtype.char, 'd')
        return (prefix, dtype, None)


from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace import _statespace as ss
from .results import results_kalman_filter
from numpy.testing import assert_almost_equal, assert_allclose
from nose.exc import SkipTest

prefix_statespace_map = {
    's': ss.sStatespace, 'd': ss.dStatespace,
    'c': ss.cStatespace, 'z': ss.zStatespace
}
prefix_kalman_filter_map = {
    's': ss.sKalmanFilter, 'd': ss.dKalmanFilter,
    'c': ss.cKalmanFilter, 'z': ss.zKalmanFilter
}

current_path = os.path.dirname(os.path.abspath(__file__))


class Clark1987(object):
    """
    Clark's (1987) univariate unobserved components model of real GDP (as
    presented in Kim and Nelson, 1999)

    Test data produced using GAUSS code described in Kim and Nelson (1999) and
    found at http://econ.korea.ac.kr/~cjkim/SSMARKOV.htm

    See `results.results_kalman_filter` for more information.
    """
    def __init__(self, dtype=float, conserve_memory=0, loglikelihood_burn=0):
        self.true = results_kalman_filter.uc_uni
        self.true_states = pd.DataFrame(self.true['states'])

        # GDP, Quarterly, 1947.1 - 1995.3
        data = pd.DataFrame(
            self.true['data'],
            index=pd.date_range('1947-01-01', '1995-07-01', freq='QS'),
            columns=['GDP']
        )
        data['lgdp'] = np.log(data['GDP'])

        # Parameters
        self.conserve_memory = conserve_memory
        self.loglikelihood_burn = loglikelihood_burn

        # Observed data
        self.obs = np.array(data['lgdp'], ndmin=2, dtype=dtype, order="F")

        # Measurement equation
        self.k_endog = k_endog = 1  # dimension of observed data
        # design matrix
        self.design = np.zeros((k_endog, 4, 1), dtype=dtype, order="F")
        self.design[:, :, 0] = [1, 1, 0, 0]
        # observation intercept
        self.obs_intercept = np.zeros((k_endog, 1), dtype=dtype, order="F")
        # observation covariance matrix
        self.obs_cov = np.zeros((k_endog, k_endog, 1), dtype=dtype, order="F")

        # Transition equation
        self.k_states = k_states = 4  # dimension of state space
        # transition matrix
        self.transition = np.zeros((k_states, k_states, 1),
                                   dtype=dtype, order="F")
        self.transition[([0, 0, 1, 1, 2, 3],
                         [0, 3, 1, 2, 1, 3],
                         [0, 0, 0, 0, 0, 0])] = [1, 1, 0, 0, 1, 1]
        # state intercept
        self.state_intercept = np.zeros((k_states, 1), dtype=dtype, order="F")
        # selection matrix
        self.selection = np.asfortranarray(np.eye(k_states)[:, :, None],
                                           dtype=dtype)
        # state covariance matrix
        self.state_cov = np.zeros((k_states, k_states, 1),
                                  dtype=dtype, order="F")

        # Initialization: Diffuse priors
        self.initial_state = np.zeros((k_states,), dtype=dtype, order="F")
        self.initial_state_cov = np.asfortranarray(np.eye(k_states)*100,
                                                   dtype=dtype)

        # Update matrices with given parameters
        (sigma_v, sigma_e, sigma_w, phi_1, phi_2) = np.array(
            self.true['parameters'], dtype=dtype
        )
        self.transition[([1, 1], [1, 2], [0, 0])] = [phi_1, phi_2]
        self.state_cov[
            np.diag_indices(k_states)+(np.zeros(k_states, dtype=int),)] = [
            sigma_v**2, sigma_e**2, 0, sigma_w**2
        ]

        # Initialization: modification
        # Due to the difference in the way Kim and Nelson (1999) and Durbin
        # and Koopman (2012) define the order of the Kalman filter routines,
        # we need to modify the initial state covariance matrix to match
        # Kim and Nelson's results, since the *Statespace models follow Durbin
        # and Koopman.
        self.initial_state_cov = np.asfortranarray(
            np.dot(
                np.dot(self.transition[:, :, 0], self.initial_state_cov),
                self.transition[:, :, 0].T
            )
        )

    def init_filter(self):
        # Use the appropriate Statespace model
        prefix = find_best_blas_type((self.obs,))
        cls = prefix_statespace_map[prefix[0]]

        # Instantiate the statespace model
        self.model = cls(
            self.obs, self.design, self.obs_intercept, self.obs_cov,
            self.transition, self.state_intercept, self.selection,
            self.state_cov
        )
        self.model.initialize_known(self.initial_state, self.initial_state_cov)

        # Initialize the appropriate Kalman filter
        cls = prefix_kalman_filter_map[prefix[0]]
        self.filter = cls(self.model, conserve_memory=self.conserve_memory,
                          loglikelihood_burn=self.loglikelihood_burn)

    def run_filter(self):
        # Filter the data
        self.filter()

        # Get results
        self.result = {
            'loglike': lambda burn: np.sum(self.filter.loglikelihood[burn:]),
            'state': np.array(self.filter.filtered_state),
        }

    def test_loglike(self):
        assert_almost_equal(
            self.result['loglike'](self.true['start']), self.true['loglike'], 5
        )

    def test_filtered_state(self):
        assert_almost_equal(
            self.result['state'][0][self.true['start']:],
            self.true_states.iloc[:, 0], 4
        )
        assert_almost_equal(
            self.result['state'][1][self.true['start']:],
            self.true_states.iloc[:, 1], 4
        )
        assert_almost_equal(
            self.result['state'][3][self.true['start']:],
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
        self.init_filter()
        self.run_filter()

    def test_loglike(self):
        assert_allclose(
            self.result['loglike'](self.true['start']), self.true['loglike'],
            rtol=1e-3
        )

    def test_filtered_state(self):
        assert_allclose(
            self.result['state'][0][self.true['start']:],
            self.true_states.iloc[:, 0],
            atol=1e-2
        )
        assert_allclose(
            self.result['state'][1][self.true['start']:],
            self.true_states.iloc[:, 1],
            atol=1e-2
        )
        assert_allclose(
            self.result['state'][3][self.true['start']:],
            self.true_states.iloc[:, 2],
            atol=1e-2
        )


class TestClark1987Double(Clark1987):
    """
    Basic double precision test for the loglikelihood and filtered states.
    """
    def __init__(self):
        super(TestClark1987Double, self).__init__(
            dtype=float, conserve_memory=0
        )
        self.init_filter()
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
        self.init_filter()
        self.run_filter()

    def test_loglike(self):
        assert_allclose(
            self.result['loglike'](self.true['start']), self.true['loglike'],
            rtol=1e-3
        )

    def test_filtered_state(self):
        assert_allclose(
            self.result['state'][0][self.true['start']:],
            self.true_states.iloc[:, 0],
            atol=1e-2
        )
        assert_allclose(
            self.result['state'][1][self.true['start']:],
            self.true_states.iloc[:, 1],
            atol=1e-2
        )
        assert_allclose(
            self.result['state'][3][self.true['start']:],
            self.true_states.iloc[:, 2],
            atol=1e-2
        )


class TestClark1987DoubleComplex(Clark1987):
    """
    Basic double precision complex test for the loglikelihood and filtered
    states.
    """
    def __init__(self):
        super(TestClark1987DoubleComplex, self).__init__(
            dtype=complex, conserve_memory=0
        )
        self.init_filter()
        self.run_filter()


class TestClark1987Conserve(Clark1987):
    """
    Memory conservation test for the loglikelihood and filtered states.
    """
    def __init__(self):
        super(TestClark1987Conserve, self).__init__(
            dtype=float, conserve_memory=0x01 | 0x02
        )
        self.init_filter()
        self.run_filter()


class Clark1987Forecast(Clark1987):
    """
    Forecasting test for the loglikelihood and filtered states.
    """
    def __init__(self, dtype=float, nforecast=100, conserve_memory=0):
        super(Clark1987Forecast, self).__init__(
            dtype, conserve_memory
        )
        self.nforecast = nforecast

        # Add missing observations to the end (to forecast)
        self._obs = self.obs
        self.obs = np.array(np.r_[self.obs[0, :], [np.nan]*nforecast],
                            ndmin=2, dtype=dtype, order="F")

    def test_filtered_state(self):
        assert_almost_equal(
            self.result['state'][0][self.true['start']:-self.nforecast],
            self.true_states.iloc[:, 0], 4
        )
        assert_almost_equal(
            self.result['state'][1][self.true['start']:-self.nforecast],
            self.true_states.iloc[:, 1], 4
        )
        assert_almost_equal(
            self.result['state'][3][self.true['start']:-self.nforecast],
            self.true_states.iloc[:, 2], 4
        )


class TestClark1987ForecastDouble(Clark1987Forecast):
    """
    Basic double forecasting test for the loglikelihood and filtered states.
    """
    def __init__(self):
        super(TestClark1987ForecastDouble, self).__init__()
        self.init_filter()
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
        self.init_filter()
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
        self.init_filter()
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
        self.loglikelihood_burn = self.true['start']
        self.init_filter()
        self.run_filter()

    def test_loglike(self):
        assert_almost_equal(
            self.result['loglike'](0), self.true['loglike'], 5
        )

    def test_filtered_state(self):
        end = self.true_states.shape[0]
        assert_almost_equal(
            self.result['state'][0][-1],
            self.true_states.iloc[end-1, 0], 4
        )
        assert_almost_equal(
            self.result['state'][1][-1],
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
    def __init__(self, dtype=float, conserve_memory=0, loglikelihood_burn=0):
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

        # Observed data
        self.obs = np.array(data, ndmin=2, dtype=dtype, order="C").T

        # Parameters
        self.k_endog = k_endog = 2  # dimension of observed data
        self.k_states = k_states = 6  # dimension of state space
        self.conserve_memory = conserve_memory
        self.loglikelihood_burn = loglikelihood_burn

        # Measurement equation

        # design matrix
        self.design = np.zeros((k_endog, k_states, 1), dtype=dtype, order="F")
        self.design[:, :, 0] = [[1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]]
        # observation intercept
        self.obs_intercept = np.zeros((k_endog, 1), dtype=dtype, order="F")
        # observation covariance matrix
        self.obs_cov = np.zeros((k_endog, k_endog, 1), dtype=dtype, order="F")

        # Transition equation

        # transition matrix
        self.transition = np.zeros((k_states, k_states, 1),
                                   dtype=dtype, order="F")
        self.transition[([0, 0, 1, 1, 2, 3, 4, 5],
                         [0, 4, 1, 2, 1, 2, 4, 5],
                         [0, 0, 0, 0, 0, 0, 0, 0])] = [1, 1, 0, 0, 1, 1, 1, 1]
        # state intercept
        self.state_intercept = np.zeros((k_states, 1), dtype=dtype, order="F")
        # selection matrix
        self.selection = np.asfortranarray(np.eye(k_states)[:, :, None],
                                           dtype=dtype)
        # state covariance matrix
        self.state_cov = np.zeros((k_states, k_states, 1),
                                  dtype=dtype, order="F")

        # Initialization: Diffuse priors
        self.initial_state = np.zeros((k_states,), dtype=dtype)
        self.initial_state_cov = np.asfortranarray(np.eye(k_states)*100,
                                                   dtype=dtype)

        # Update matrices with given parameters
        (sigma_v, sigma_e, sigma_w, sigma_vl, sigma_ec,
         phi_1, phi_2, alpha_1, alpha_2, alpha_3) = np.array(
            self.true['parameters'], dtype=dtype
        )
        self.design[([1, 1, 1], [1, 2, 3], [0, 0, 0])] = [
            alpha_1, alpha_2, alpha_3
        ]
        self.transition[([1, 1], [1, 2], [0, 0])] = [phi_1, phi_2]
        self.obs_cov[1, 1, 0] = sigma_ec**2
        self.state_cov[
            np.diag_indices(k_states)+(np.zeros(k_states, dtype=int),)] = [
            sigma_v**2, sigma_e**2, 0, 0, sigma_w**2, sigma_vl**2
        ]

        # Initialization: modification
        # Due to the difference in the way Kim and Nelson (1999) and Drubin
        # and Koopman (2012) define the order of the Kalman filter routines,
        # we need to modify the initial state covariance matrix to match
        # Kim and Nelson's results, since the *Statespace models follow Durbin
        # and Koopman.
        self.initial_state_cov = np.asfortranarray(
            np.dot(
                np.dot(self.transition[:, :, 0], self.initial_state_cov),
                self.transition[:, :, 0].T
            )
        )

    def init_filter(self):
        # Use the appropriate Statespace model
        prefix = find_best_blas_type((self.obs,))
        cls = prefix_statespace_map[prefix[0]]

        # Instantiate the statespace model
        self.model = cls(
            self.obs, self.design, self.obs_intercept, self.obs_cov,
            self.transition, self.state_intercept, self.selection,
            self.state_cov
        )
        self.model.initialize_known(self.initial_state, self.initial_state_cov)

        # Initialize the appropriate Kalman filter
        cls = prefix_kalman_filter_map[prefix[0]]
        self.filter = cls(self.model, conserve_memory=self.conserve_memory,
                          loglikelihood_burn=self.loglikelihood_burn)

    def run_filter(self):
        # Filter the data
        self.filter()

        # Get results
        self.result = {
            'loglike': lambda burn: np.sum(self.filter.loglikelihood[burn:]),
            'state': np.array(self.filter.filtered_state),
        }

    def test_loglike(self):
        assert_almost_equal(
            # self.result['loglike'](self.true['start']),
            self.result['loglike'](0),
            self.true['loglike'], 2
        )

    def test_filtered_state(self):
        assert_almost_equal(
            self.result['state'][0][self.true['start']:],
            self.true_states.iloc[:, 0], 4
        )
        assert_almost_equal(
            self.result['state'][1][self.true['start']:],
            self.true_states.iloc[:, 1], 4
        )
        assert_almost_equal(
            self.result['state'][4][self.true['start']:],
            self.true_states.iloc[:, 2], 4
        )
        assert_almost_equal(
            self.result['state'][5][self.true['start']:],
            self.true_states.iloc[:, 3], 4
        )


class TestClark1989(Clark1989):
    """
    Basic double precision test for the loglikelihood and filtered
    states with two-dimensional observation vector.
    """
    def __init__(self):
        super(TestClark1989, self).__init__(dtype=float, conserve_memory=0)
        self.init_filter()
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
        self.init_filter()
        self.run_filter()


class Clark1989Forecast(Clark1989):
    """
    Memory conservation test for the loglikelihood and filtered states with
    two-dimensional observation vector.
    """
    def __init__(self, dtype=float, nforecast=100, conserve_memory=0):
        super(Clark1989Forecast, self).__init__(dtype, conserve_memory)
        self.nforecast = nforecast

        # Add missing observations to the end (to forecast)
        self._obs = self.obs
        self.obs = np.array(
            np.c_[
                self._obs,
                np.r_[[np.nan, np.nan]*nforecast].reshape(2, nforecast)
            ],
            ndmin=2, dtype=dtype, order="F"
        )

        self.init_filter()
        self.run_filter()

    def test_filtered_state(self):
        assert_almost_equal(
            self.result['state'][0][self.true['start']:-self.nforecast],
            self.true_states.iloc[:, 0], 4
        )
        assert_almost_equal(
            self.result['state'][1][self.true['start']:-self.nforecast],
            self.true_states.iloc[:, 1], 4
        )
        assert_almost_equal(
            self.result['state'][4][self.true['start']:-self.nforecast],
            self.true_states.iloc[:, 2], 4
        )
        assert_almost_equal(
            self.result['state'][5][self.true['start']:-self.nforecast],
            self.true_states.iloc[:, 3], 4
        )


class TestClark1989ForecastDouble(Clark1989Forecast):
    """
    Basic double forecasting test for the loglikelihood and filtered states.
    """
    def __init__(self):
        super(TestClark1989ForecastDouble, self).__init__()
        self.init_filter()
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
        self.init_filter()
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
        self.init_filter()
        self.run_filter()


class TestClark1989ConserveAll(Clark1989):
    """
    Memory conservation forecasting test for the loglikelihood and filtered
    states.
    """
    def __init__(self):
        super(TestClark1989ConserveAll, self).__init__(
            dtype=float, conserve_memory=0x01 | 0x02 | 0x04 | 0x08,
        )
        # self.loglikelihood_burn = self.true['start']
        self.loglikelihood_burn = 0
        self.init_filter()
        self.run_filter()

    def test_loglike(self):
        assert_almost_equal(
            self.result['loglike'](0), self.true['loglike'], 2
        )

    def test_filtered_state(self):
        end = self.true_states.shape[0]
        assert_almost_equal(
            self.result['state'][0][-1],
            self.true_states.iloc[end-1, 0], 4
        )
        assert_almost_equal(
            self.result['state'][1][-1],
            self.true_states.iloc[end-1, 1], 4
        )
        assert_almost_equal(
            self.result['state'][4][-1],
            self.true_states.iloc[end-1, 2], 4
        )
        assert_almost_equal(
            self.result['state'][5][-1],
            self.true_states.iloc[end-1, 3], 4
        )
