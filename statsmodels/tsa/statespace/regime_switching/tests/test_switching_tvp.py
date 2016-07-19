import numpy as np
from numpy.testing import assert_allclose
from statsmodels.tsa.statespace.regime_switching.api import SwitchingTVPModel
from .results import results_kim1993


class Kim1993(object):

    @classmethod
    def setup_class(cls):

        cls.dtype = np.float64
        dtype = cls.dtype

        cls.true = results_kim1993.tvpmrkf
        start = cls.true['start']

        cls.k_regimes = 2
        k_regimes = cls.k_regimes
        cls.k_exog = 5
        k_exog = cls.k_exog

        cls.endog = np.array(cls.true['data']['m1'], dtype=dtype)

        cls.exog = np.zeros((cls.endog.shape[0], k_exog), dtype=dtype)

        cls.exog[:, 0] = 1
        cls.exog[:, 1] = cls.true['data']['dint']
        cls.exog[:, 2] = cls.true['data']['inf']
        cls.exog[:, 3] = cls.true['data']['surp']
        cls.exog[:, 4] = cls.true['data']['m1lag']

        cls.model = SwitchingTVPModel(k_regimes, cls.endog, exog=cls.exog,
                dtype=dtype, loglikelihood_burn=start)

        cls.model.initialize_known(np.zeros(k_exog, dtype=dtype),
                np.identity(k_exog, dtype=dtype) * 100)


class TestKim1993_Filtering(Kim1993):

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
