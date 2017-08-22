import os
import numpy as np
from numpy.testing import (assert_, assert_raises, assert_almost_equal,
                           assert_equal, assert_array_equal, assert_allclose,
                           assert_array_less)

import statsmodels.api as sm
from .results.results_discrete import RandHIE

class CheckResults(object):
    def test_params(self):
        assert_allclose(self.res1.params, self.res2.params, atol=1e-5, rtol=1e-5)

    def test_llf(self):
        assert_allclose(self.res1.llf, self.res2.llf, atol=1e-5, rtol=1e-5)

    def test_conf_int(self):
        assert_allclose(self.res1.conf_int(), self.res2.conf_int, atol=1e-3, rtol=1e-5)

    def test_bse(self):
        assert_allclose(self.res1.bse, self.res2.bse, atol=1e-3)

    def test_aic(self):
        assert_allclose(self.res1.aic, self.res2.aic, atol=1e-2, rtol=1e-2)

    def test_bic(self):
        assert_allclose(self.res1.bic, self.res2.bic, atol=1e-2, rtol=1e-2)

    def test_fit_regularized(self):
        model = self.res1.model

        alpha = np.ones(len(self.res1.params))
        res_reg = model.fit_regularized(alpha=alpha*0.01, disp=0)

        assert_allclose(res_reg.params, self.res1.params,
            rtol=1e-3, atol=5e-3)
        assert_allclose(res_reg.bse, self.res1.bse,
            rtol=1e-3, atol=5e-3)

class TestTruncatedPoissonModel(CheckResults):
    @classmethod
    def setup_class(cls):
        data = sm.datasets.randhie.load()
        exog = sm.add_constant(data.exog[:,:4], prepend=False)
        cls.res1 = sm.TruncatedPoisson(data.endog, exog, truncation=5).fit(maxiter=500)
        res2 = RandHIE()
        res2.truncated_poisson()
        cls.res2 = res2

class TestZeroTruncatedPoissonModel(CheckResults):
    @classmethod
    def setup_class(cls):
        data = sm.datasets.randhie.load()
        exog = sm.add_constant(data.exog[:,:4], prepend=False)
        cls.res1 = sm.TruncatedPoisson(data.endog, exog, truncation=0).fit(maxiter=500)
        res2 = RandHIE()
        res2.zero_truncated_poisson()
        cls.res2 = res2

class TestZeroTruncatedNBPModel(CheckResults):
    @classmethod
    def setup_class(cls):
        data = sm.datasets.randhie.load()
        exog = sm.add_constant(data.exog[:,:3], prepend=False)
        cls.res1 = sm.TruncatedNegativeBinomialP(data.endog, exog,
            truncation=0).fit(maxiter=500)
        res2 = RandHIE()
        res2.zero_truncted_nbp()
        cls.res2 = res2    

    def test_conf_int(self):
        pass


class TestTruncatedPoisson_predict(object):
    @classmethod
    def setup_class(cls):
        cls.expected_params = [1, 0.5]
        np.random.seed(123)
        nobs = 200
        exog = np.ones((nobs, 2))
        exog[:nobs//2, 1] = 2
        mu_true = exog.dot(cls.expected_params)
        cls.endog = sm.distributions.truncatedpoisson.rvs(mu_true, 0, size=mu_true.shape)
        model = sm.TruncatedPoisson(cls.endog, exog, truncation=0)
        cls.res = model.fit(method='bfgs', maxiter=5000, maxfun=5000)

    def test_mean(self):
        assert_allclose(self.res.predict().mean(), self.endog.mean(),
                        atol=2e-1, rtol=2e-1)

    def test_var(self):
        assert_allclose((self.res.predict().mean() *
                        self.res._dispersion_factor.mean()),
                        self.endog.var(), atol=5e-2, rtol=5e-2)

    def test_predict_prob(self):
        res = self.res
        endog = res.model.endog

        pr = res.predict(which='prob')
        pr2 = sm.distributions.truncatedpoisson.pmf(np.arange(6),
                                        res.predict().mean(), 0).T
        assert_allclose(pr2, pr2, rtol=1e-10, atol=1e-10)

class TestTruncatedNBP_predict(object):
    @classmethod
    def setup_class(cls):
        cls.expected_params = [1, 0.5, 0.5]
        np.random.seed(1234)
        nobs = 200
        exog = np.ones((nobs, 2))
        exog[:nobs//2, 1] = 2
        mu_true = np.exp(exog.dot(cls.expected_params[:-1]))
        cls.endog = sm.distributions.truncatednegbin.rvs(mu_true,
            cls.expected_params[-1], 2, 0, size=mu_true.shape)
        model = sm.TruncatedNegativeBinomialP(cls.endog, exog, truncation=0, p=2)
        cls.res = model.fit(method='nm', maxiter=5000, maxfun=5000)

    def test_mean(self):
        assert_allclose(self.res.predict().mean(), self.endog.mean(),
                        atol=2e-1, rtol=2e-1)

    def test_var(self):
        #assert_allclose((self.res.predict().mean() *
        #                self.res._dispersion_factor.mean()),
        #                self.endog.var(), atol=5e-2, rtol=5e-2)

    def test_predict_prob(self):
        res = self.res
        endog = res.model.endog

        pr = res.predict(which='prob')
        pr2 = sm.distributions.truncatednegbin.pmf(np.arange(6),
            res.predict().mean(), self.expected_params[-1], 2, 0).T
        assert_allclose(pr2, pr2, rtol=1e-10, atol=1e-10)  


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, '-vvs', '-x', '--pdb'])