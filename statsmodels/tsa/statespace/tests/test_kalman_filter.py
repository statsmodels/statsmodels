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
from statsmodels.tsa.statespace.kalman_filter import dkalman_filter
from .results import results_kalman_filter
from numpy.testing import assert_almost_equal


class TestClark1987(object):
    """
    Clark's (1987) univariate unobserved components model of real GDP (as
    presented in Kim and Nelson, 1999)

    Test data produced using GAUSS code described in Kim and Nelson (1999) and
    found at http://econ.korea.ac.kr/~cjkim/SSMARKOV.htm

    See `results.results_kalman` for more information.
    """
    def __init__(self):
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
        y = np.array(data['lgdp'], ndmin=2, dtype=float, order="F")

        # Measurement equation
        n = 1  # dimension of observed data
        H = np.zeros((n, 4, 1), dtype=float, order="F")
        H[:, :, 0] = [1, 1, 0, 0]  # link state to observations
        R = np.zeros((n, n), dtype=float, order="F")  # var/cov matrix

        # Transition equation
        k = 4  # dimension of state space
        mu = np.zeros((k,), dtype=float)  # state mean
        F = np.zeros((k, k), dtype=float, order="F")  # state AR matrix
        F[([0, 0, 1, 1, 2, 3], [0, 3, 1, 2, 1, 3])] = [1, 1, 0, 0, 1, 1]
        # optional matrix so that Q_star is always positive definite
        G = np.asfortranarray(np.eye(k), dtype=float)
        Q_star = np.zeros((k, k), dtype=float, order="F")

        # Initialization: Diffuse priors
        initial_state = np.zeros((k,), dtype=float)
        initial_state_cov = np.asfortranarray(np.eye(k)*100, dtype=float)

        # Update matrices with given parameters
        (sigma_v, sigma_e, sigma_w, phi_1, phi_2) = self.true['parameters']
        F[([1, 1], [1, 2])] = [phi_1, phi_2]
        Q_star[np.diag_indices(k)] = [
            sigma_v**2, sigma_e**2, 0, sigma_w**2
        ]

        # Filter the data
        args = (y, H, mu, F, R, G, Q_star,
                None, None, initial_state, initial_state_cov)
        (state, state_cov, est_state, est_state_cov, forecast,
         prediction_error, prediction_error_cov, inverse_prediction_error_cov,
         gain, loglikelihood) = dkalman_filter(*args)

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
