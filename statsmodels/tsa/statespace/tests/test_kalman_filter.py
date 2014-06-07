"""
Tests for Kalman Filter

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
        return prefix


from statsmodels.tsa.statespace.kalman_filter import (
    skalman_filter, dkalman_filter, ckalman_filter, zkalman_filter
)
from .results import results_kalman_filter
from numpy.testing import assert_almost_equal
from nose.exc import SkipTest

prefix_kalman_filter_map = {
    's': skalman_filter, 'd': dkalman_filter,
    'c': ckalman_filter, 'z': zkalman_filter
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
    def __init__(self, dtype=float):
        self.true = results_kalman_filter.uc_uni
        self.true_states = pd.DataFrame(self.true['states'])

        # GDP, Quarterly, 1947.1 - 1995.3
        data = pd.DataFrame(
            self.true['data'],
            index=pd.date_range('1947-01-01', '1995-07-01', freq='QS'),
            columns=['GDP']
        )
        data['lgdp'] = np.log(data['GDP'])

        # Observed data
        y = np.array(data['lgdp'], ndmin=2, dtype=dtype, order="F")

        # Measurement equation
        n = 1  # dimension of observed data
        H = np.zeros((n, 4, 1), dtype=dtype, order="F")
        H[:, :, 0] = [1, 1, 0, 0]  # link state to observations
        R = np.zeros((n, n), dtype=dtype, order="F")  # var/cov matrix

        # Transition equation
        k = 4  # dimension of state space
        mu = np.zeros((k,), dtype=dtype)  # state mean
        F = np.zeros((k, k), dtype=dtype, order="F")  # state AR matrix
        F[([0, 0, 1, 1, 2, 3], [0, 3, 1, 2, 1, 3])] = [1, 1, 0, 0, 1, 1]
        # optional matrix so that Q_star is always positive definite
        G = np.asfortranarray(np.eye(k), dtype=dtype)
        Q_star = np.zeros((k, k), dtype=dtype, order="F")

        # Initialization: Diffuse priors
        initial_state = np.zeros((k,), dtype=dtype)
        initial_state_cov = np.asfortranarray(np.eye(k)*100, dtype=dtype)

        # Update matrices with given parameters
        (sigma_v, sigma_e, sigma_w, phi_1, phi_2) = np.array(
            self.true['parameters'], dtype=dtype
        )
        F[([1, 1], [1, 2])] = [phi_1, phi_2]
        Q_star[np.diag_indices(k)] = [
            sigma_v**2, sigma_e**2, 0, sigma_w**2
        ]

        # Use the appropriate Kalman filter
        prefix = find_best_blas_type((y,))
        kalman_filter = prefix_kalman_filter_map[prefix[0]]

        # Filter the data
        args = (y, H, mu, F, R, G, Q_star,
                None, None, initial_state, initial_state_cov)
        (state, state_cov, est_state, est_state_cov, forecast,
         prediction_error, prediction_error_cov, inverse_prediction_error_cov,
         gain, loglikelihood) = kalman_filter(*args)

        # Save the output
        self.state = np.asarray(state[:, 1:])
        self.state_cov = np.asarray(state_cov[:, :, 1:])
        self.est_state = np.asarray(est_state[:, 1:])
        self.est_state_cov = np.asarray(est_state_cov[:, :, 1:])
        self.forecast = np.asarray(forecast[:, 1:])
        self.prediction_error = np.asarray(prediction_error[:, 1:])
        self.prediction_error_cov = np.asarray(prediction_error_cov[:, :, 1:])
        self.inverse_prediction_error_cov = np.asarray(
            inverse_prediction_error_cov[:, :, 1:]
        )
        self.gain = np.asarray(gain[:, :, 1:])
        self.loglikelihood = np.asarray(loglikelihood[1:])

        # Get results
        self.result = {
            'loglike': lambda burn: np.sum(self.loglikelihood[burn:]),
            'state': self.state,
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
        super(TestClark1987Single, self).__init__(dtype=np.float32)


class TestClark1987Double(Clark1987):
    """
    Basic double precision test for the loglikelihood and filtered states.
    """
    def __init__(self):
        super(TestClark1987Double, self).__init__(dtype=float)


class TestClark1987SingleComplex(Clark1987):
    """
    Basic single precision complex test for the loglikelihood and filtered
    states.
    """
    def __init__(self):
        raise SkipTest('Not implemented')
        super(TestClark1987SingleComplex, self).__init__(dtype=np.complex64)


class TestClark1987DoubleComplex(Clark1987):
    """
    Basic double precision complex test for the loglikelihood and filtered
    states.
    """
    def __init__(self):
        super(TestClark1987DoubleComplex, self).__init__(dtype=complex)


class TestClark1989(object):
    """
    Clark's (1989) bivariate unobserved components model of real GDP (as
    presented in Kim and Nelson, 1999)

    Tests two-dimensional observation data.

    Test data produced using GAUSS code described in Kim and Nelson (1999) and
    found at http://econ.korea.ac.kr/~cjkim/SSMARKOV.htm

    See `results.results_kalman_filter` for more information.
    """
    def __init__(self, dtype=float):
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
        y = np.array(data, ndmin=2, dtype=dtype, order="C").T

        # Parameters
        n = 2  # dimension of observed data
        k = 6  # dimension of state space

        # Measurement equation
        H = np.zeros((n, k, 1), dtype=dtype, order="F")
        H[:, :, 0] = [[1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]]
        R = np.zeros((n, n), dtype=dtype, order="F")  # var/cov matrix

        # Transition equation
        mu = np.zeros((k,), dtype=dtype)  # state mean
        F = np.zeros((k, k), dtype=dtype, order="F")  # state AR matrix
        F[([0, 0, 1, 1, 2, 3, 4, 5],
           [0, 4, 1, 2, 1, 2, 4, 5])] = [1, 1, 0, 0, 1, 1, 1, 1]
        # optional matrix so that Q_star is always positive definite
        G = np.asfortranarray(np.eye(k), dtype=dtype)
        Q_star = np.zeros((k, k), dtype=dtype, order="F")

        # Initialization: Diffuse priors
        initial_state = np.zeros((k,), dtype=dtype)
        initial_state_cov = np.asfortranarray(np.eye(k)*100, dtype=dtype)

        # Update matrices with given parameters
        (sigma_v, sigma_e, sigma_w, sigma_vl, sigma_ec,
         phi_1, phi_2, alpha_1, alpha_2, alpha_3) = np.array(
            self.true['parameters'], dtype=dtype
        )
        H[([1, 1, 1], [1, 2, 3], [0, 0, 0])] = [alpha_1, alpha_2, alpha_3]
        F[([1, 1], [1, 2])] = [phi_1, phi_2]
        R[1, 1] = sigma_ec**2
        Q_star[np.diag_indices(k)] = [
            sigma_v**2, sigma_e**2, 0, 0, sigma_w**2, sigma_vl**2
        ]

        # Use the appropriate Kalman filter
        prefix = find_best_blas_type((y,))
        kalman_filter = prefix_kalman_filter_map[prefix[0]]

        # Filter the data
        args = (y, H, mu, F, R, G, Q_star,
                None, None, initial_state, initial_state_cov)
        (state, state_cov, est_state, est_state_cov, forecast,
         prediction_error, prediction_error_cov, inverse_prediction_error_cov,
         gain, loglikelihood) = kalman_filter(*args)

        # Save the output
        self.state = np.asarray(state[:, 1:])
        self.state_cov = np.asarray(state_cov[:, :, 1:])
        self.est_state = np.asarray(est_state[:, 1:])
        self.est_state_cov = np.asarray(est_state_cov[:, :, 1:])
        self.forecast = np.asarray(forecast[:, 1:])
        self.prediction_error = np.asarray(prediction_error[:, 1:])
        self.prediction_error_cov = np.asarray(prediction_error_cov[:, :, 1:])
        self.inverse_prediction_error_cov = np.asarray(
            inverse_prediction_error_cov[:, :, 1:]
        )
        self.gain = np.asarray(gain[:, :, 1:])
        self.loglikelihood = np.asarray(loglikelihood[1:])

        # Get results
        self.result = {
            'loglike': lambda burn: np.sum(self.loglikelihood[burn:]),
            'state': self.state,
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
            self.result['state'][4][self.true['start']:],
            self.true_states.iloc[:, 2], 4
        )
        assert_almost_equal(
            self.result['state'][5][self.true['start']:],
            self.true_states.iloc[:, 3], 4
        )


class TestKimNelson1989(object):
    """
    Kim and Nelson's (1989) time-varying parameters model of monetary growth
    (as presented in Kim and Nelson, 1999)

    Tests time-varying observation equation.

    Test data produced using GAUSS code described in Kim and Nelson (1999) and
    found at http://econ.korea.ac.kr/~cjkim/SSMARKOV.htm

    See `results.results_kalman_filter` for more information.
    """
    def __init__(self, dtype=float):
        self.true = results_kalman_filter.tvp
        self.true_states = pd.DataFrame(self.true['states'])

        # Quarterly, 1959.3--1985.4
        data = pd.DataFrame(
            self.true['data'],
            index=pd.date_range(
                start='1959-07-01', end='1985-10-01', freq='QS'),
            columns=['Qtr', 'm1', 'dint', 'inf', 'surpl', 'm1lag']
        )

        # Observed data
        y = np.array(data['m1'], ndmin=2, dtype=dtype, order="F")

        # Parameters
        n = 1  # dimension of observed data
        k = 5  # dimension of state space

        # Measurement equation
        H = np.asfortranarray(np.c_[
            np.ones(data['dint'].shape),
            data['dint'],
            data['inf'],
            data['surpl'],
            data['m1lag']
        ].T[None, :], dtype=dtype)
        R = np.zeros((n, n), dtype=dtype, order="F")  # var/cov matrix

        # Transition equation
        mu = np.zeros((k,), dtype=dtype)  # state mean
        F = np.asfortranarray(np.eye(k), dtype=dtype)
        # optional matrix so that Q_star is always positive definite
        G = np.asfortranarray(np.eye(k), dtype=dtype)
        Q_star = np.zeros((k, k), dtype=dtype, order="F")

        # Initialization: Diffuse priors
        initial_state = np.zeros((k,), dtype=dtype)
        initial_state_cov = np.asfortranarray(np.eye(k)*50, dtype=dtype)

        # Update matrices with given parameters
        (sigma_e, sigma_v0, sigma_v1, sigma_v2, sigma_v3, sigma_v4) = np.array(
            self.true['parameters'], dtype=dtype
        )
        R[0, 0] = sigma_e**2
        Q_star[np.diag_indices(k)] = [
            sigma_v0**2, sigma_v1**2,
            sigma_v2**2, sigma_v3**2,
            sigma_v4**2
        ]

        # Use the appropriate Kalman filter
        prefix = find_best_blas_type((y,))
        kalman_filter = prefix_kalman_filter_map[prefix[0]]

        # Filter the data
        args = (y, H, mu, F, R, G, Q_star,
                None, None, initial_state, initial_state_cov)
        (state, state_cov, est_state, est_state_cov, forecast,
         prediction_error, prediction_error_cov, inverse_prediction_error_cov,
         gain, loglikelihood) = kalman_filter(*args)

        # Save the output
        self.state = np.asarray(state[:, 1:])
        self.state_cov = np.asarray(state_cov[:, :, 1:])
        self.est_state = np.asarray(est_state[:, 1:])
        self.est_state_cov = np.asarray(est_state_cov[:, :, 1:])
        self.forecast = np.asarray(forecast[:, 1:])
        self.prediction_error = np.asarray(prediction_error[:, 1:])
        self.prediction_error_cov = np.asarray(prediction_error_cov[:, :, 1:])
        self.inverse_prediction_error_cov = np.asarray(
            inverse_prediction_error_cov[:, :, 1:]
        )
        self.gain = np.asarray(gain[:, :, 1:])
        self.loglikelihood = np.asarray(loglikelihood[1:])

        # Get results
        self.result = {
            'loglike': lambda burn: np.sum(self.loglikelihood[burn:]),
            'state': self.state,
        }

    def test_loglike(self):
        assert_almost_equal(
            self.result['loglike'](self.true['start']),
            self.true['loglike'], 5
        )

    def test_filtered_state(self):
        assert_almost_equal(
            self.result['state'][0][self.true['start']-1:-1],
            self.true_states.iloc[:, 0], 3
        )
        assert_almost_equal(
            self.result['state'][1][self.true['start']-1:-1],
            self.true_states.iloc[:, 1], 3
        )
        assert_almost_equal(
            self.result['state'][2][self.true['start']-1:-1],
            self.true_states.iloc[:, 2], 3
        )
        assert_almost_equal(
            self.result['state'][3][self.true['start']-1:-1],
            self.true_states.iloc[:, 3], 3
        )
        assert_almost_equal(
            self.result['state'][4][self.true['start']-1:-1],
            self.true_states.iloc[:, 4], 3
        )


class TestRealGDPAR(object):
    """
    Test fitting an AR(12) via a state-space model to the FRED GDPC1 series.

    Tests a higher dimensional state-space.

    Results set created using Stata sspace model, and also verified with the
    FKF (Fast Kalman Filter) library in R.

    See results/test_realgdpar_stata.do and results/test_realgdpar_r.R
    for more information.
    """

    def __init__(self, dtype=float):
        self.stata_output = pd.read_csv(
            current_path + '/results/results_realgdpar_stata.csv')
        self.true = results_kalman_filter.gdp

        # GDP, Quarterly, 1947.1 - 2014.1
        dlgdp = np.log(self.stata_output['value']).diff()[1:]

        # Parameters
        n = 1   # dimension of observed data
        k = 12  # dimension of state space

        # Observed data
        y = np.array(dlgdp, ndmin=2, dtype=dtype, order="F")

        # Measurement equation
        H = np.zeros((n, k, 1), dtype=dtype, order="F")
        H[0, 0, 0] = 1  # link state to observations
        R = np.zeros((n, n), dtype=dtype, order="F")  # var/cov matrix

        # Transition equation
        mu = np.zeros((k,), dtype=dtype)  # state mean
        F = np.zeros((k, k), dtype=dtype, order="F")  # state AR matrix
        idx = np.diag_indices(k-1)
        F[(idx[0]+1, idx[1])] = 1
        # optional matrix so that Q_star is always positive definite
        G = np.zeros((k, 1), dtype=dtype, order="F")
        G[0, 0] = 1
        Q_star = np.zeros((1, 1), dtype=dtype, order="F")

        # Initialization: done within Kalman filter

        # Update matrices with given parameters
        F[0, :] = self.true['stata_params'][:12]
        Q_star[0, 0] = self.true['stata_params'][12]

        # Use the appropriate Kalman filter
        prefix = find_best_blas_type((y,))
        kalman_filter = prefix_kalman_filter_map[prefix[0]]

        # Filter the data
        args = (y, H, mu, F, R, G, Q_star,
                None, None, None, None)
        (state, state_cov, est_state, est_state_cov, forecast,
         prediction_error, prediction_error_cov, inverse_prediction_error_cov,
         gain, loglikelihood) = kalman_filter(*args)

        # Save the output
        self.state = np.asarray(state[:, 1:])
        self.state_cov = np.asarray(state_cov[:, :, 1:])
        self.est_state = np.asarray(est_state[:, 1:])
        self.est_state_cov = np.asarray(est_state_cov[:, :, 1:])
        self.forecast = np.asarray(forecast[:, 1:])
        self.prediction_error = np.asarray(prediction_error[:, 1:])
        self.prediction_error_cov = np.asarray(prediction_error_cov[:, :, 1:])
        self.inverse_prediction_error_cov = np.asarray(
            inverse_prediction_error_cov[:, :, 1:]
        )
        self.gain = np.asarray(gain[:, :, 1:])
        self.loglikelihood = np.asarray(loglikelihood[1:])

        # Get results
        self.result = {
            'loglike': np.sum(self.loglikelihood),
            'state': self.state,
            'est_state': self.est_state,
        }

    def test_loglike(self):
        assert_almost_equal(
            self.result['loglike'], self.true['stata_loglike'], 3
        )

    def test_filtered_state(self):
        for i in range(12):
            assert_almost_equal(
                self.result['state'][i, :],
                self.stata_output['u%d' % (i+1)][1:], 6
            )

    def test_predicted_state(self):
        for i in range(12):
            assert_almost_equal(
                self.result['est_state'][i, :],
                self.stata_output['est_u%d' % (i+1)][1:], 6
            )