"""
Tests for Markov switching autoregressive model

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
from numpy.testing import assert_allclose, assert_array_equal
from statsmodels.tsa.statespace.regime_switching.api import \
        MarkovAutoregression, RegimePartition
from .results import results_hamilton1989


class Hamilton1989(object):
    """
    Hamilton's (1989) Markov-Switching Model of Business Fluctuations (chapter
    4.4 of Kim and Nelson, 1999)

    Test data produced using GAUSS code described in Kim and Nelson (1999) and
    found at http://econ.korea.ac.kr/~cjkim/MARKOV/programs/hmt4_kim.opt

    See `results.results_hamilton1989` for more information.
    """

    @classmethod
    def setup_class(cls):
        cls.dtype = np.float64
        dtype = cls.dtype

        # Model attributes
        cls.k_ar_regimes = 2
        cls.order = 4

        cls.true = results_hamilton1989.hmt4_kim

        # Preparing observations
        data = np.array(cls.true['data'], dtype=dtype)
        data = np.log(data) * 100

        cls.obs = data[20:152] - data[19:151]

        cls.model = MarkovAutoregression(cls.k_ar_regimes, cls.order, cls.obs,
                switching_mean=True, dtype=dtype)


class TestHamilton1989_Filtering(Hamilton1989):
    """
    Basic test for the loglikelihood, predicted and filtered regime
    probabilities precision.
    """

    @classmethod
    def setup_class(cls):

        super(TestHamilton1989_Filtering, cls).setup_class()

        params = np.array(cls.true['parameters'], dtype=cls.dtype)

        results = cls.model.filter(params)

        cls.result = {
                'loglike': results.loglike(),
                'pr_tt0': results.filtered_regime_probs[:, \
                        ::2].sum(axis=1),
                'pr_tl0': results.predicted_regime_probs[:, \
                        ::2].sum(axis=1)
        }

    def test_loglike(self):
        assert_allclose(self.result['loglike'], self.true['loglike'],
                rtol=1e-3)

    def test_probs(self):
        assert_allclose(self.result['pr_tt0'], self.true['pr_tt0'], rtol=1e-2)
        assert_allclose(self.result['pr_tl0'], self.true['pr_tl0'], rtol=1e-2)


class TestHamilton1989_Smoothing(Hamilton1989):
    """
    Basic test for the smoothed regime probabilities precision.
    """

    @classmethod
    def setup_class(cls):

        super(TestHamilton1989_Smoothing, cls).setup_class()

        params = np.array(cls.true['parameters'], dtype=cls.dtype)

        partition = RegimePartition([0, 1] * 16)

        results = cls.model.smooth(params, return_extended_probs=True,
                regime_partition=partition)

        cls.result = {
                'smooth0': results.smoothed_regime_probs[:, 0]
        }

    def test_probs(self):
        assert_allclose(self.result['smooth0'], self.true['smooth0'],
                rtol=1e-2)


class TestHamilton1989_MLE(Hamilton1989):
    """
    Basic test for MLE correct convergence
    """

    @classmethod
    def setup_class(cls):

        super(TestHamilton1989_MLE, cls).setup_class()

        params = cls.model.fit(
                start_params=np.array(
                cls.true['untransformed_start_parameters'], dtype=cls.dtype),
                transformed=False)

        cls.result = {
                'loglike': cls.model.loglike(params),
                'params': params
        }

    def test_loglike(self):
        assert_allclose(self.result['loglike'], self.true['loglike'],
                rtol=1e-3)

    def test_params(self):
        assert_allclose(self.result['params'], self.true['parameters'],
                rtol=5e-2)


class TestHamilton1989_EM(Hamilton1989):
    """
    Test for EM algorithm convergence to near-optimal solution
    """

    @classmethod
    def setup_class(cls):

        super(TestHamilton1989_EM, cls).setup_class()

        # It takes some time to run 50 sessions of EM-algorithm
        params = cls.model.fit_em_with_random_starts()

        params = cls.model.normalize_params(params)

        cls.result = {
                'loglike': cls.model.loglike(params),
                'params': params
        }

    def test_loglike(self):
        assert_allclose(self.result['loglike'], self.true['loglike'],
                rtol=2e-2)

    def test_params(self):
        # Test that EM algorithm produces sensible result (difference is
        # significant in only one parameter)
        is_close = np.isclose(self.result['params'], self.true['parameters'],
                atol=0.15, rtol=0.1)
        true_is_close = [True] * 9
        true_is_close[6] = False
        assert_array_equal(is_close, true_is_close)
