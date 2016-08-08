"""
Tests for Markov switching time varying parameters model

Author: Valery Likhosherstov
License: Simplified-BSD

References
----------

Kim, Chang-Jin, and Charles R. Nelson. 1999.
"State-Space Models with Regime Switching:
Classical and Gibbs-Sampling Approaches with Applications".
MIT Press Books. The MIT Press.
"""
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from statsmodels.tsa.statespace.regime_switching.api import SwitchingTVPModel
from .results import results_kim1993
from statsmodels.tsa.statespace.tools import compatibility_mode
from nose.exc import SkipTest

if compatibility_mode:
    raise SkipTest


class Kim1993(object):
    """
    Kim's (1993) time-varying-parameter model with heteroskedastic disturbances
    for U.S. monetary growth uncertainty (chapter 5.5.1 of Kim and Nelson,
    1999).

    Test data produced using GAUSS code described in Kim and Nelson (1999) and
    found at http://econ.korea.ac.kr/~cjkim/MARKOV/programs/tvpmrkf.opt

    See `results.results_kim1993` for more information.
    """

    @classmethod
    def setup_class(cls):

        cls.dtype = np.float64
        dtype = cls.dtype

        cls.true = results_kim1993.tvpmrkf
        start = cls.true['start']

        # Model attributes
        cls.k_regimes = 2
        k_regimes = cls.k_regimes
        cls.k_exog = 5
        k_exog = cls.k_exog

        # Preparing test data

        cls.endog = np.array(cls.true['data']['m1'], dtype=dtype)

        cls.exog = np.zeros((cls.endog.shape[0], k_exog), dtype=dtype)

        cls.exog[:, 0] = 1
        cls.exog[:, 1] = cls.true['data']['dint']
        cls.exog[:, 2] = cls.true['data']['inf']
        cls.exog[:, 3] = cls.true['data']['surp']
        cls.exog[:, 4] = cls.true['data']['m1lag']

        # Instantiate the model
        cls.model = SwitchingTVPModel(k_regimes, cls.endog, cls.exog,
                dtype=dtype, loglikelihood_burn=start)

        # Set model initial states
        cls.model.initialize_known(np.zeros(k_exog, dtype=dtype),
                np.identity(k_exog, dtype=dtype) * 100)


class TestKim1993_Filtering(Kim1993):
    """
    Basic test for the loglikelihood and forecast precision.
    """

    @classmethod
    def setup_class(cls):

        super(TestKim1993_Filtering, cls).setup_class()

        dtype = cls.dtype
        start = cls.true['start']

        results = cls.model.filter(np.array(cls.true['parameters'],
                dtype=dtype), return_ssm=True)

        cls.result = {
                'loglike': results.loglike(),
                'f_cast': results.forecasts_error.ravel()[start:],
                'ss': results.forecasts_error_cov.ravel()[start:]
        }

    def test_loglike(self):
        assert_allclose(self.result['loglike'], self.true['loglike'], rtol=1e-5)

    def test_forecast(self):
        assert_allclose(self.result['f_cast'], self.true['f_cast'], rtol=3e-2)
        assert_allclose(self.result['ss'], self.true['ss'], rtol=3e-2)


class TestKim1993_MLE(Kim1993):
    """
    Basic test for MLE correct convergence.
    """

    @classmethod
    def setup_class(cls):

        super(TestKim1993_MLE, cls).setup_class()

        dtype = cls.dtype

        params = cls.model.fit(start_params=np.array(
                cls.true['start_parameters'], dtype=dtype), return_params=True)

        cls.result = {
                'loglike': cls.model.loglike(params),
                'params': params
        }

    def test_loglike(self):
        assert_allclose(self.result['loglike'], self.true['loglike'], rtol=1e-5)

    def test_params(self):
        assert_allclose(self.result['params'], self.true['parameters'],
                rtol=1e-3, atol=1e-5)


class TestKim1993_MLEFitNonswitchingFirst(Kim1993):
    """
    Basic test for correct convergence of MLE from the start provided by
    non-switching model.
    """

    @classmethod
    def setup_class(cls):

        super(TestKim1993_MLEFitNonswitchingFirst, cls).setup_class()

        params = cls.model.fit(fit_nonswitching_first=True, return_params=True)

        params = cls.model.normalize_params(params)

        cls.result = {
                'loglike': cls.model.loglike(params),
                'params': params
        }

    def test_loglike(self):
        assert_allclose(self.result['loglike'], self.true['loglike'], rtol=1e-2)

    def test_params(self):
        assert_allclose(self.result['params'], self.true['parameters'],
                rtol=5e-2, atol=5e-2)

class TestKim1993_SwitchingTVP(Kim1993):
    """
    Smoke test to check if switching tvp cov option works fine and doesn't
    throw any errors.
    """

    @classmethod
    def setup_class(cls):

        super(TestKim1993_SwitchingTVP, cls).setup_class()

        dtype = cls.dtype
        k_exog = cls.k_exog
        k_regimes = cls.k_regimes

        # Redefine the model - add switching tvp
        cls.model = SwitchingTVPModel(k_regimes, cls.endog, cls.exog,
                switching_obs_cov=True, switching_tvp_cov=True, dtype=dtype)

        # Set model initial states
        cls.model.initialize_known(np.zeros(k_exog, dtype=dtype),
                np.identity(k_exog, dtype=dtype) * 100)

        params = cls.model.fit(fit_nonswitching_first=True, return_params=True)

        params = cls.model.normalize_params(params)

        cls.result_params = params

    def test_params(self):
        # Check if result parameters are finite and not Nan
        assert_equal(np.isfinite(self.result_params), True)
