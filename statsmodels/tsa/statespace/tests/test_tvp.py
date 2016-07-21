"""
Tests for Time varying parameters model

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
from numpy.testing import assert_allclose
from statsmodels.tsa.statespace.tvp import TVPModel
from .results import results_tvp


class KimNelson1989(object):
    """
    Kim and Nelson's (1989) time-varying-parameter model for modelling changing
    conditional variance of uncertainty in the U.S. monetary growth (chapter
    3.4 of Kim and Nelson, 1999).

    Test data produced using GAUSS code described in Kim and Nelson (1999) and
    found at http://econ.korea.ac.kr/~cjkim/MARKOV/programs/tvp.opt

    See `results.results_tvp` for more information.
    """

    @classmethod
    def setup_class(cls):
        cls.dtype = np.float64
        dtype = cls.dtype

        cls.true = results_tvp.tvp
        start = cls.true['start']

        # Model attributes
        cls.k_exog = 5
        k_exog = cls.k_exog

        # Preparing endog and exog data

        cls.endog = np.array(cls.true['data']['m1'], dtype=dtype)

        cls.exog = np.zeros((cls.endog.shape[0], k_exog), dtype=dtype)

        cls.exog[:, 0] = 1
        cls.exog[:, 1] = cls.true['data']['dint']
        cls.exog[:, 2] = cls.true['data']['inf']
        cls.exog[:, 3] = cls.true['data']['surpl']
        cls.exog[:, 4] = cls.true['data']['m1lag']

        # Instantiate the model

        cls.model = TVPModel(cls.endog, exog=cls.exog, dtype=dtype,
                alternate_timing=True, loglikelihood_burn=start)

        cls.model.initialize_known(np.zeros(k_exog, dtype=dtype),
                np.identity(k_exog, dtype=dtype) * 50)


class TestKimNelson1989_Filtering(KimNelson1989):
    """
    Basic test for the loglikelihood, forecast errors and predicted states
    precision.
    """

    @classmethod
    def setup_class(cls):

        super(TestKimNelson1989_Filtering, cls).setup_class()

        dtype = cls.dtype
        start = cls.true['start']

        results = cls.model.filter(np.array(cls.true['parameters'],
                dtype=dtype), return_ssm=True)

        cls.result = {
                'loglike': results.llf_obs[start:].sum(),
                'f_cast': results.forecasts_error.ravel()[start:],
                'ss': results.forecasts_error_cov.ravel()[start:],
                'beta_tl': results.predicted_state.T[start:-1]
        }

    def test_loglike(self):
        assert_allclose(self.result['loglike'], self.true['loglike'], rtol=1e-5)

    def test_forecast(self):
        assert_allclose(self.result['f_cast'], self.true['f_cast'], rtol=1e-2)
        assert_allclose(self.result['ss'], self.true['ss'], rtol=1e-2)

    def test_predicted(self):
        assert_allclose(self.result['beta_tl'], self.true['beta_tl'], rtol=2e-2)


class TestKimNelson1989_MLE(KimNelson1989):
    """
    Basic test for MLE correct convergence.
    """

    @classmethod
    def setup_class(cls):

        super(TestKimNelson1989_MLE, cls).setup_class()

        dtype = cls.dtype

        params = cls.model.fit(np.array(np.array(
                cls.true['untransformed_start_parameters']), dtype=dtype),
                return_params=True)

        cls.result = {
                'loglike' : cls.model.loglike(params),
                'params': params
        }

    def test_loglike(self):
        assert_allclose(self.result['loglike'], self.true['loglike'], rtol=1e-5)

    def test_params(self):
        assert_allclose(self.result['params'], self.true['parameters'], rtol=1e-3)


class TestKimNelson1989_MLEDefaultStart(KimNelson1989):
    """
    Basic test for MLE correct convergence from default start.
    """

    @classmethod
    def setup_class(cls):

        super(TestKimNelson1989_MLEDefaultStart, cls).setup_class()

        dtype = cls.dtype

        params = cls.model.fit(return_params=True)

        cls.result = {
                'loglike' : cls.model.loglike(params),
                'params': params
        }

    def test_loglike(self):
        assert_allclose(self.result['loglike'], self.true['loglike'], rtol=1e-5)

    def test_params(self):
        assert_allclose(self.result['params'], self.true['parameters'], rtol=1e-2)