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
from numpy.testing import assert_allclose, assert_equal
from statsmodels.tsa.statespace.regime_switching.api import \
        MarkovAutoregression, RegimePartition
from .results import results_garcia_perron1996
from statsmodels.tsa.statespace.tools import compatibility_mode
from nose.exc import SkipTest

if compatibility_mode:
    raise SkipTest


class GarciaPerron1996(object):
    """
    Garcia and Perron's (1996) 3-state Markov-switching mean and variance model
    of Real Interest Rate (chapter 4.5 of Kim and Nelson, 1999).

    Test data produced using GAUSS code described in Kim and Nelson (1999) and
    found at http://econ.korea.ac.kr/~cjkim/MARKOV/programs/intr_s3.opt

    See `results.results_garcia_perron1996` for more information.
    """

    @classmethod
    def setup_class(cls):

        cls.dtype = np.float64
        dtype = cls.dtype

        # Model attributes
        cls.k_ar_regimes = 3
        cls.order = 2

        cls.true = results_garcia_perron1996.intr_s3

        data = np.array(cls.true['data'], dtype=dtype)

        # Preparing observations
        ex_r = data[1:176, 1]
        inf = np.log(data[1:176, 2] / data[0:175, 2]) * 100 * 4
        cls.obs = ex_r[49:175] - inf[49:175]

        cls.model = MarkovAutoregression(cls.k_ar_regimes, cls.order, cls.obs,
                switching_mean=True, switching_variance=True)


class TestGarciaPerron1996_Filtering(GarciaPerron1996):
    """
    Basic test for the loglikelihood and predicted regime probabilities
    precision.
    """

    @classmethod
    def setup_class(cls):

        super(TestGarciaPerron1996_Filtering, cls).setup_class()

        params = np.array(cls.true['parameters'], dtype=cls.dtype)

        results = cls.model.filter(params, return_ssm=True)

        predicted_regime_probs = results.predicted_regime_probs

        pr_probs = np.zeros((3, predicted_regime_probs.shape[1]),
                dtype=cls.dtype)

        for i in range(3):
            pr_probs[i, :] = predicted_regime_probs[i::3, :].sum(axis=0)

        cls.result = {
                'loglike': results.loglike(),
                'pr_probs': pr_probs.T
        }

    def test_loglike(self):
        assert_allclose(self.result['loglike'], self.true['loglike'], rtol=1e-3)

    def test_probs(self):
        assert_allclose(self.result['pr_probs'], self.true['pr_probs'],
                atol=1e-3)


class TestGarciaPerron1996_MLE(GarciaPerron1996):
    """
    Basic test for MLE correct convergence.
    """

    @classmethod
    def setup_class(cls):

        super(TestGarciaPerron1996_MLE, cls).setup_class()

        params = cls.model.fit(start_params=np.array(
                cls.true['start_parameters'], dtype=cls.dtype),
                return_params=True)

        cls.result = {
                'loglike': cls.model.loglike(params),
                'params': params
        }

    def test_loglike(self):
        assert_allclose(self.result['loglike'], self.true['loglike'], rtol=1e-3)

    def test_params(self):
        assert_allclose(self.result['params'], self.true['parameters'],
                rtol=1e-2, atol=1e-7)


class TestGarciaPerron1996_EM(GarciaPerron1996):
    """
    Test for EM algorithm convergence to near-optimal solution
    """

    @classmethod
    def setup_class(cls):

        super(TestGarciaPerron1996_EM, cls).setup_class()

        # It takes some time to run 50 sessions of EM-algorithm

        params = cls.model.fit_em_with_random_starts()

        params = cls.model.normalize_params(params)

        cls.result = {
                'loglike': cls.model.loglike(params),
                'params': params
        }

    def test_loglike(self):
        assert_allclose(self.result['loglike'], self.true['loglike'], rtol=2e-2)

    def test_params(self):
        # Test that EM algorithm produces sensible result
        assert_allclose(self.result['params'], self.true['parameters'],
                rtol=2e-1, atol=0.15)


class TestGarciaPerron1996_MLEFitNonswitchingFirst(GarciaPerron1996):
    """
    Smoke test to check if non-switching start parameters feature doesn't throw
    errors and returns something (due to complexity of MS-AR model, EM-algorithm
    is prefered as a more sophisticated tool)
    """

    @classmethod
    def setup_class(cls):

        super(TestGarciaPerron1996_MLEFitNonswitchingFirst, cls).setup_class()

        params = cls.model.fit(fit_nonswitching_first=True, return_params=True)

        cls.result = {
                'loglike': cls.model.loglike(params),
                'params': params
        }

    def test_loglike(self):
        # Check if calculated likelihood makes sense
        assert_allclose(self.result['loglike'], self.true['loglike'], rtol=5e-2)

    def test_params(self):
        # Check if parameters are finite and not NaN
        assert_equal(np.isfinite(self.result['params']), True)
