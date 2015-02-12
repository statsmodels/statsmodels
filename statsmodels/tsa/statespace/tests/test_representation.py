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

from statsmodels.tsa.statespace.representation import Representation
from statsmodels.tsa.statespace.model import Model
from .results import results_kalman_filter
from numpy.testing import assert_equal, assert_almost_equal, assert_raises, assert_allclose
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
        self.model = Model(data['lgdp'], k_states=k_states, **kwargs)

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
            self.results.llf_obs[self.true['start']:].sum(),
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
            self.results.llf_obs[0], self.true['loglike'], 5
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
        self.model = Model(data, k_states=k_states, **kwargs)

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
            # self.results.llf_obs[self.true['start']:].sum(),
            self.results.llf_obs[0:].sum(),
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
        # self.model.loglikelihood_burn = self.true['start']
        self.model.loglikelihood_burn = 0
        self.run_filter()

    def test_loglike(self):
        assert_almost_equal(
            self.results.llf_obs[0], self.true['loglike'], 2
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

# Miscellaneous coverage-related tests
def test_slice_notation():
    endog = np.arange(10)*1.0
    mod = Model(endog, k_states=2)

    # Test invalid __setitem__
    def set_designs():
        mod['designs'] = 1
    def set_designs2():
        mod['designs',0,0] = 1
    def set_designs3():
        mod[0] = 1
    assert_raises(IndexError, set_designs)
    assert_raises(IndexError, set_designs2)
    assert_raises(IndexError, set_designs3)

    # Test invalid __getitem__
    assert_raises(IndexError, lambda: mod['designs'])
    assert_raises(IndexError, lambda: mod['designs',0,0,0])
    assert_raises(IndexError, lambda: mod[0])

    # Test valid __setitem__, __getitem__
    assert_equal(mod.design[0,0,0], 0)
    mod['design',0,0,0] = 1
    assert_equal(mod['design'].sum(), 1)
    assert_equal(mod.design[0,0,0], 1)
    assert_equal(mod['design',0,0,0], 1)

    # Test valid __setitem__, __getitem__ with unspecified time index
    mod['design'] = np.zeros(mod['design'].shape)
    assert_equal(mod.design[0,0], 0)
    mod['design',0,0] = 1
    assert_equal(mod.design[0,0], 1)
    assert_equal(mod['design',0,0], 1)

def test_representation():
    # Test an invalid number of states
    def zero_kstates():
        mod = Representation(1, 0)
    assert_raises(ValueError, zero_kstates)

    # Test an invalid endogenous array
    def empty_endog():
        endog = np.zeros((0,0))
        mod = Representation(endog, k_states=2)
    assert_raises(ValueError, empty_endog)

    # Test a Fortran-ordered endogenous array (which will be assumed to be in
    # wide format: k_endog x nobs)
    nobs = 10
    k_endog = 2
    endog = np.asfortranarray(np.arange(nobs*k_endog).reshape(k_endog,nobs)*1.)
    mod = Representation(endog, k_states=2)
    assert_equal(mod.nobs, nobs)
    assert_equal(mod.k_endog, k_endog)

    # Test a C-ordered endogenous array (which will be assumed to be in
    # tall format: nobs x k_endog)
    nobs = 10
    k_endog = 2
    endog = np.arange(nobs*k_endog).reshape(nobs,k_endog)*1.
    mod = Representation(endog, k_states=2)
    assert_equal(mod.nobs, nobs)
    assert_equal(mod.k_endog, k_endog)

    # Test getting the statespace representation
    assert_equal(mod._statespace, None)
    mod._initialize_representation()
    assert(mod._statespace is not None)

def test_bind():
    mod = Representation(1, k_states=2)

    # Test invalid endogenous array (it must be ndarray)
    assert_raises(ValueError, lambda: mod.bind([1,2,3,4]))

    # Test valid (nobs x 1) endogenous array
    mod.bind(np.arange(10)*1.)
    assert_equal(mod.nobs, 10)

    # Test valid (k_endog x 0) endogenous array
    mod.bind(np.zeros(0,dtype=np.float64))

    # Test invalid (3-dim) endogenous array
    assert_raises(ValueError, lambda: mod.bind(np.arange(12).reshape(2,2,3)*1.))

    # Test valid F-contiguous
    mod.bind(np.asfortranarray(np.arange(10).reshape(1,10)))
    assert_equal(mod.nobs, 10)

    # Test valid C-contiguous
    mod.bind(np.arange(10).reshape(10,1))
    assert_equal(mod.nobs, 10)

    # Test invalid F-contiguous
    assert_raises(ValueError, lambda: mod.bind(np.asfortranarray(np.arange(10).reshape(10,1))))

    # Test invalid C-contiguous
    assert_raises(ValueError, lambda: mod.bind(np.arange(10).reshape(1,10)))

def test_initialization():
    mod = Representation(1, k_states=2)

    # Test invalid state initialization
    assert_raises(RuntimeError, lambda: mod._initialize_state())

    # Test valid initialization
    initial_state = np.zeros(2,) + 1.5
    initial_state_cov = np.eye(2) * 3.
    mod.initialize_known(initial_state, initial_state_cov)
    assert_equal(mod._initial_state.sum(), 3)
    assert_equal(mod._initial_state_cov.diagonal().sum(), 6)

    # Test invalid initial_state
    initial_state = np.zeros(10,)
    assert_raises(ValueError, lambda: mod.initialize_known(initial_state, initial_state_cov))
    initial_state = np.zeros((10,10))
    assert_raises(ValueError, lambda: mod.initialize_known(initial_state, initial_state_cov))

    # Test invalid initial_state_cov
    initial_state = np.zeros(2,) + 1.5
    initial_state_cov = np.eye(3)
    assert_raises(ValueError, lambda: mod.initialize_known(initial_state, initial_state_cov))
