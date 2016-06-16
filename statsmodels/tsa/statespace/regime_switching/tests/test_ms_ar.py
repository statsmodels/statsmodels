import numpy as np
from numpy.testing import assert_allclose
from statsmodels.tsa.statespace.regime_switching.api import \
        MarkovAutoregression
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

        smoothed_regime_probs, smoothed_curr_and_next_regime_probs = \
                cls.model.get_smoothed_regime_probs(params)

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
                rtol=1e-2)


class TestHamilton1989_EM(Hamilton1989):

    @classmethod
    def setup_class(cls):

        super(TestHamilton1989_EM, cls).setup_class()

        np.random.seed(seed=1)

        random_start_params = np.random.normal(size=9)

        params = cls.model.fit_em_algorithm(start_params=random_start_params,
                transformed=False)

    def test_loglike(self):
        assert_allclose(self.result['loglike'], self.true['loglike'],
                rtol=1e-3)
