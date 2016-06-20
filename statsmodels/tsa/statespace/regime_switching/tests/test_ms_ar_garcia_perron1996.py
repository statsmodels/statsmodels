import numpy as np
from numpy.testing import assert_allclose
from statsmodels.tsa.statespace.regime_switching.api import \
        MarkovAutoregression, RegimePartition
from .results import results_garcia_perron1996


class GarciaPerron1996(object):

    @classmethod
    def setup_class(cls):

        cls.dtype = np.float64
        dtype = cls.dtype

        cls.k_ar_regimes = 3
        cls.order = 2

        cls.true = results_garcia_perron1996.intr_s3

        data = np.array(cls.true['data'], dtype=dtype)

        ex_r = data[1:176, 1]
        inf = np.log(data[1:176, 2] / data[0:175, 2]) * 100 * 4
        cls.obs = ex_r[49:175] - inf[49:175]

        cls.model = MarkovAutoregression(cls.k_ar_regimes, cls.order, cls.obs,
                switching_mean=True, switching_variance=True)


class TestGarciaPerron1996_Filtering(GarciaPerron1996):

    @classmethod
    def setup_class(cls):

        super(TestGarciaPerron1996_Filtering, cls).setup_class()

        params = np.array(cls.true['parameters'], dtype=cls.dtype)

        cls.model.filter(params)

        predicted_regime_probs = cls.model.ssm.predicted_regime_probs

        pr_probs = np.zeros((predicted_regime_probs.shape[0], 3),
                dtype=cls.dtype)

        for i in range(3):
            pr_probs[:, i] = predicted_regime_probs[:, i::3].sum(axis=1)

        cls.result = {
                'loglike': cls.model.ssm.loglike(filter_first=False),
                'pr_probs': pr_probs
        }

    def test_loglike(self):
        assert_allclose(self.result['loglike'], self.true['loglike'], rtol=1e-3)

    def test_probs(self):
        assert_allclose(self.result['pr_probs'], self.true['pr_probs'],
                atol=1e-3)


class TestGarciaPerron1996_MLE(GarciaPerron1996):

    @classmethod
    def setup_class(cls):

        super(TestGarciaPerron1996_MLE, cls).setup_class()

        params = cls.model.fit(start_params=np.array(
                cls.true['start_parameters'], dtype=cls.dtype))

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
