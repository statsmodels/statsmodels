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
        return prefix


from statsmodels.tsa.statespace import _statespace as ss
from .results import results_kalman_filter
from numpy.testing import assert_almost_equal
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
        obs = np.array(data['lgdp'], ndmin=2, dtype=dtype, order="F")

        # Measurement equation
        nendog = 1  # dimension of observed data
        # design matrix
        design = np.zeros((nendog, 4, 1), dtype=dtype, order="F")
        design[:, :, 0] = [1, 1, 0, 0]
        # observation intercept
        obs_intercept = np.zeros((nendog, 1), dtype=dtype, order="F")
        # observation covariance matrix
        obs_cov = np.zeros((nendog, nendog, 1), dtype=dtype, order="F")

        # Transition equation
        nstates = 4  # dimension of state space
        # transition matrix
        transition = np.zeros((nstates, nstates, 1), dtype=dtype, order="F")
        transition[([0, 0, 1, 1, 2, 3],
                    [0, 3, 1, 2, 1, 3],
                    [0, 0, 0, 0, 0, 0])] = [1, 1, 0, 0, 1, 1]
        # state intercept
        state_intercept = np.zeros((nstates, 1), dtype=dtype, order="F")
        # selection matrix
        selection = np.asfortranarray(np.eye(nstates)[:, :, None], dtype=dtype)
        # state covariance matrix
        state_cov = np.zeros((nstates, nstates, 1), dtype=dtype, order="F")

        # Initialization: Diffuse priors
        initial_state = np.zeros((nstates,), dtype=dtype, order="F")
        initial_state_cov = np.asfortranarray(np.eye(nstates)*100, dtype=dtype)

        # Update matrices with given parameters
        (sigma_v, sigma_e, sigma_w, phi_1, phi_2) = np.array(
            self.true['parameters'], dtype=dtype
        )
        transition[([1, 1], [1, 2], [0, 0])] = [phi_1, phi_2]
        state_cov[np.diag_indices(nstates)+(np.zeros(nstates, dtype=int),)] = [
            sigma_v**2, sigma_e**2, 0, sigma_w**2
        ]

        # Initialization: modification
        # Due to the difference in the way Kim and Nelson (1999) and Drubin
        # and Koopman (2012) define the order of the Kalman filter routines,
        # we need to modify the initial state covariance matrix to match
        # Kim and Nelson's results, since the *Statespace models follow Durbin
        # and Koopman.
        initial_state_cov = np.asfortranarray(
            np.dot(
                np.dot(transition[:, :, 0], initial_state_cov),
                transition[:, :, 0].T
            )
        )

        # Use the appropriate Statespace model
        prefix = find_best_blas_type((obs,))
        cls = prefix_statespace_map[prefix[0]]

        # Instantiate the statespace model
        self.model = cls(obs, design, obs_intercept, obs_cov, transition,
                         state_intercept, selection, state_cov)
        self.model.initialize_known(initial_state, initial_state_cov)

        # Initialize the appropriate Kalman filter
        cls = prefix_kalman_filter_map[prefix[0]]
        self.filter = cls(self.model)

        # Filter the data
        self.filter()

        # Get results
        self.result = {
            'loglike': lambda burn: np.sum(self.model.loglikelihood[burn:]),
            'state': np.array(self.model.filtered_state),
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
        obs = np.array(data, ndmin=2, dtype=dtype, order="C").T

        # Parameters
        nendog = 2  # dimension of observed data
        nstates = 6  # dimension of state space

        # Measurement equation

        # design matrix
        design = np.zeros((nendog, nstates, 1), dtype=dtype, order="F")
        design[:, :, 0] = [[1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]]
        # observation intercept
        obs_intercept = np.zeros((nendog, 1), dtype=dtype, order="F")
        # observation covariance matrix
        obs_cov = np.zeros((nendog, nendog, 1), dtype=dtype, order="F")

        # Transition equation

        # transition matrix
        transition = np.zeros((nstates, nstates, 1), dtype=dtype, order="F")
        transition[([0, 0, 1, 1, 2, 3, 4, 5],
                    [0, 4, 1, 2, 1, 2, 4, 5],
                    [0, 0, 0, 0, 0, 0, 0, 0])] = [1, 1, 0, 0, 1, 1, 1, 1]
        # state intercept
        state_intercept = np.zeros((nstates, 1), dtype=dtype, order="F")
        # selection matrix
        selection = np.asfortranarray(np.eye(nstates)[:, :, None], dtype=dtype)
        # state covariance matrix
        state_cov = np.zeros((nstates, nstates, 1), dtype=dtype, order="F")

        # Initialization: Diffuse priors
        initial_state = np.zeros((nstates,), dtype=dtype)
        initial_state_cov = np.asfortranarray(np.eye(nstates)*100, dtype=dtype)

        # Update matrices with given parameters
        (sigma_v, sigma_e, sigma_w, sigma_vl, sigma_ec,
         phi_1, phi_2, alpha_1, alpha_2, alpha_3) = np.array(
            self.true['parameters'], dtype=dtype
        )
        design[([1, 1, 1], [1, 2, 3], [0, 0, 0])] = [alpha_1, alpha_2, alpha_3]
        transition[([1, 1], [1, 2], [0, 0])] = [phi_1, phi_2]
        obs_cov[1, 1, 0] = sigma_ec**2
        state_cov[np.diag_indices(nstates)+(np.zeros(nstates, dtype=int),)] = [
            sigma_v**2, sigma_e**2, 0, 0, sigma_w**2, sigma_vl**2
        ]

        # Initialization: modification
        # Due to the difference in the way Kim and Nelson (1999) and Durbin
        # and Koopman (2012) define the order of the Kalman filter routines,
        # we need to modify the initial state covariance matrix to match
        # Kim and Nelson's results, since the *Statespace models follow Durbin
        # and Koopman.
        initial_state_cov = np.asfortranarray(
            np.dot(
                np.dot(transition[:, :, 0], initial_state_cov),
                transition[:, :, 0].T
            )
        )

        # Use the appropriate Statespace model
        prefix = find_best_blas_type((obs,))
        cls = prefix_statespace_map[prefix[0]]

        # Instantiate the statespace model
        self.model = cls(obs, design, obs_intercept, obs_cov, transition,
                         state_intercept, selection, state_cov)
        self.model.initialize_known(initial_state, initial_state_cov)

        # Initialize the appropriate Kalman filter
        cls = prefix_kalman_filter_map[prefix[0]]
        self.filter = cls(self.model)

        # Filter the data
        self.filter()

        # Get results
        self.result = {
            'loglike': lambda burn: np.sum(self.model.loglikelihood[burn:]),
            'state': np.array(self.model.filtered_state),
        }

    def test_loglike(self):
        assert_almost_equal(
            self.result['loglike'](self.true['start']), self.true['loglike'], 2
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
