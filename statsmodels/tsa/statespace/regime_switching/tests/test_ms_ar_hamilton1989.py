import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from statsmodels.tsa.statespace.regime_switching.api import \
        MarkovAutoregression, RegimePartition
from .results import results_hamilton1989


class Hamilton1989(object):

    @classmethod
    def setup_class(cls):
        cls.dtype = np.float64
        dtype = cls.dtype

        cls.k_ar_regimes = 2
        cls.order = 4

        cls.true = results_hamilton1989.hmt4_kim
        cls.true_pr_tt0 = np.array(cls.true['pr_tt0'], dtype=dtype)
        cls.true_pr_tl0 = np.array(cls.true['pr_tl0'], dtype=dtype)
        cls.true_smooth0 = np.array(cls.true['smooth0'], dtype=dtype)

        data = np.array(cls.true['data'], dtype=dtype)
        data = np.log(data) * 100

        cls.obs = data[20:152] - data[19:151]

        cls.model = MarkovAutoregression(cls.k_ar_regimes, cls.order, cls.obs,
                switching_mean=True, dtype=dtype)


class TestHamilton1989_Filtering(Hamilton1989):

    @classmethod
    def setup_class(cls):

        super(TestHamilton1989_Filtering, cls).setup_class()

        params = np.array(cls.true['parameters'], dtype=cls.dtype)

        cls.model.filter(params)

        cls.result = {
                'loglike': cls.model.ssm.loglike(filter_first=False),
                'pr_tt0': cls.model.ssm.filtered_regime_probs[:, \
                        ::2].sum(axis=1),
                'pr_tl0': cls.model.ssm.predicted_regime_probs[:, \
                        ::2].sum(axis=1)
        }

    def test_loglike(self):
        assert_allclose(self.result['loglike'], self.true['loglike'],
                rtol=1e-3)

    def test_probs(self):
        assert_allclose(self.result['pr_tt0'], self.true['pr_tt0'], rtol=1e-2)
        assert_allclose(self.result['pr_tl0'], self.true['pr_tl0'], rtol=1e-2)


class TestHamilton1989_Smoothing(Hamilton1989):

    @classmethod
    def setup_class(cls):

        super(TestHamilton1989_Smoothing, cls).setup_class()

        params = np.array(cls.true['parameters'], dtype=cls.dtype)

        partition = RegimePartition([0, 1] * 16)

        smoothed_regime_probs, smoothed_curr_and_next_regime_probs = \
                cls.model.get_smoothed_regime_probs(params,
                return_extended_probs=True, regime_partition=partition)

        cls.result = {
                'smooth0': smoothed_regime_probs[:, 0]
        }

    def test_probs(self):
        assert_allclose(self.result['smooth0'], self.true['smooth0'],
                rtol=1e-2)


class TestHamilton1989_MLE(Hamilton1989):

    @classmethod
    def setup_class(cls):

        super(TestHamilton1989_MLE, cls).setup_class()

        params = cls.model.fit(
                start_params=np.array(
                cls.true['untransformed_start_parameters'], dtype=cls.dtype),
                transformed=False)

        cls.model.filter(params)

        cls.result = {
                'loglike': cls.model.ssm.loglike(filter_first=False),
                'params': params
        }

    def test_loglike(self):
        assert_allclose(self.result['loglike'], self.true['loglike'],
                rtol=1e-3)

    def test_params(self):
        assert_allclose(self.result['params'], self.true['parameters'],
                rtol=5e-2)


class TestHamilton1989_EM(Hamilton1989):

    @classmethod
    def setup_class(cls):

        super(TestHamilton1989_EM, cls).setup_class()

        # It takes some time to run 50 sessions of EM-algorithm
        params = cls.model.fit_em_with_random_starts()

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
