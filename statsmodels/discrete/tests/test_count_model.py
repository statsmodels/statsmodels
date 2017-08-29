import os
import numpy as np
from numpy.testing import (assert_, assert_raises, assert_almost_equal,
                           assert_equal, assert_array_equal, assert_allclose,
                           assert_array_less)

import statsmodels.api as sm
from .results.results_discrete import RandHIE

class CheckGeneric(object):
    def test_params(self):
        assert_allclose(self.res1.params, self.res2.params, atol=1e-5, rtol=1e-5)

    def test_llf(self):
        assert_allclose(self.res1.llf, self.res2.llf, atol=1e-5, rtol=1e-5)

    def test_conf_int(self):
        assert_allclose(self.res1.conf_int(), self.res2.conf_int, atol=1e-3, rtol=1e-5)

    def test_bse(self):
        assert_allclose(self.res1.bse, self.res2.bse, atol=1e-3, rtol=1e-3)

    def test_aic(self):
        assert_allclose(self.res1.aic, self.res2.aic, atol=1e-2, rtol=1e-2)

    def test_bic(self):
        assert_allclose(self.res1.aic, self.res2.aic, atol=1e-1, rtol=1e-1)

    def test_t(self):
         unit_matrix = np.identity(self.res1.params.size)
         t_test = self.res1.t_test(unit_matrix)
         assert_allclose(self.res1.tvalues, t_test.tvalue)

    def test_fit_regularized(self):
        model = self.res1.model

        alpha = np.ones(len(self.res1.params))
        alpha[-2:] = 0
        res_reg = model.fit_regularized(alpha=alpha*0.01, disp=0, maxiter=500)

        assert_allclose(res_reg.params[2:], self.res1.params[2:],
            atol=5e-2, rtol=5e-2)

class TestZeroInflatedModel_logit(CheckGeneric):
    @classmethod
    def setup_class(cls):
        data = sm.datasets.randhie.load()
        cls.endog = data.endog
        exog = sm.add_constant(data.exog[:,1:4], prepend=False)
        exog_infl = sm.add_constant(data.exog[:,0], prepend=False)
        cls.res1 = sm.ZeroInflatedPoisson(data.endog, exog, 
            exog_infl=exog_infl, inflation='logit').fit(method='newton', maxiter=500)
        res2 = RandHIE()
        res2.zero_inflated_poisson_logit()
        cls.res2 = res2

class TestZeroInflatedModel_probit(CheckGeneric):
    @classmethod
    def setup_class(cls):
        data = sm.datasets.randhie.load()
        cls.endog = data.endog
        exog = sm.add_constant(data.exog[:,1:4], prepend=False)
        exog_infl = sm.add_constant(data.exog[:,0], prepend=False)
        cls.res1 = sm.ZeroInflatedPoisson(data.endog, exog,
            exog_infl=exog_infl, inflation='probit').fit(method='newton', maxiter=500)
        res2 = RandHIE()
        res2.zero_inflated_poisson_probit()
        cls.res2 = res2

class TestZeroInflatedModel_offset(CheckGeneric):
    @classmethod
    def setup_class(cls):
        data = sm.datasets.randhie.load()
        cls.endog = data.endog
        exog = sm.add_constant(data.exog[:,1:4], prepend=False)
        exog_infl = sm.add_constant(data.exog[:,0], prepend=False)
        cls.res1 = sm.ZeroInflatedPoisson(data.endog, exog,
            exog_infl=exog_infl, offset=data.exog[:,7]).fit(method='newton', maxiter=500)
        res2 = RandHIE()
        res2.zero_inflated_poisson_offset()
        cls.res2 = res2

class TestZeroInflatedPoisson_predict(object):
    @classmethod
    def setup_class(cls):
        expected_params = [1, 0.5]
        np.random.seed(123)
        nobs = 200
        exog = np.ones((nobs, 2))
        exog[:nobs//2, 1] = 2
        mu_true = exog.dot(expected_params)
        cls.endog = sm.distributions.zipoisson.rvs(mu_true, 0.05,
                                                   size=mu_true.shape)
        model = sm.ZeroInflatedPoisson(cls.endog, exog)
        cls.res = model.fit(method='bfgs', maxiter=5000, maxfun=5000)

    def test_mean(self):
        assert_allclose(self.res.predict().mean(), self.endog.mean(),
                        atol=1e-2, rtol=1e-2)

    def test_var(self):
        assert_allclose((self.res.predict().mean() *
                        self.res._dispersion_factor.mean()),
                        self.endog.var(), atol=5e-2, rtol=5e-2)

    def test_predict_prob(self):
        res = self.res
        endog = res.model.endog

        pr = res.predict(which='prob')
        pr2 = sm.distributions.zipoisson.pmf(np.arange(7)[:,None],
            res.predict(), 0.05).T
        assert_allclose(pr, pr2, rtol=0.05, atol=0.05)

class TestZeroInflatedGeneralizedPoisson(CheckGeneric):
    @classmethod
    def setup_class(cls):
        data = sm.datasets.randhie.load()
        cls.endog = data.endog
        exog = sm.add_constant(data.exog[:,1:4], prepend=False)
        exog_infl = sm.add_constant(data.exog[:,0], prepend=False)
        cls.res1 = sm.ZeroInflatedGeneralizedPoisson(data.endog, exog,
            exog_infl=exog_infl, p=1).fit(method='newton', maxiter=500)
        res2 = RandHIE()
        res2.zero_inflated_generalized_poisson()
        cls.res2 = res2

    def test_bse(self):
        pass

    def test_conf_int(self):
        pass

    def test_bic(self):
        pass

    def test_t(self):
         unit_matrix = np.identity(self.res1.params.size)
         t_test = self.res1.t_test(unit_matrix)
         assert_allclose(self.res1.tvalues, t_test.tvalue)

class TestZeroInflatedGeneralizedPoisson_predict(object):
    @classmethod
    def setup_class(cls):
        expected_params = [1, 0.5, 0.5]
        np.random.seed(1234)
        nobs = 200
        exog = np.ones((nobs, 2))
        exog[:nobs//2, 1] = 2
        mu_true = exog.dot(expected_params[:-1])
        cls.endog = sm.distributions.zigenpoisson.rvs(mu_true, expected_params[-1],
                                                      2, 0.5, size=mu_true.shape)
        model = sm.ZeroInflatedGeneralizedPoisson(cls.endog, exog, p=2)
        cls.res = model.fit(method='bfgs', maxiter=5000, maxfun=5000)

    def test_mean(self):
        assert_allclose(self.res.predict().mean(), self.endog.mean(),
                        atol=1e-2, rtol=1e-2)

    def test_var(self):
        assert_allclose((self.res.predict().mean() *
                        self.res._dispersion_factor.mean()),
                        self.endog.var(), atol=0.1, rtol=0.1)

    def test_predict_prob(self):
        res = self.res
        endog = res.model.endog

        pr = res.predict(which='prob')
        pr2 = sm.distributions.zinegbin.pmf(np.arange(12)[:,None],
            res.predict(), 0.5, 2, 0.5).T
        assert_allclose(pr, pr2, rtol=0.1, atol=0.1)

class TestZeroInflatedNegativeBinomialP(CheckGeneric):
    @classmethod
    def setup_class(cls):
        data = sm.datasets.randhie.load()
        cls.endog = data.endog
        exog = sm.add_constant(data.exog[:,1], prepend=False)
        exog_infl = sm.add_constant(data.exog[:,0], prepend=False)
        cls.res1 = sm.ZeroInflatedNegativeBinomialP(data.endog, exog,
            exog_infl=exog_infl, p=2).fit(method='bfgs', maxiter=500)
        res2 = RandHIE()
        res2.zero_inflated_negative_binomial()
        cls.res2 = res2

    def test_params(self):
        assert_allclose(self.res1.params, self.res2.params,
                        atol=1e-3, rtol=1e-3)

    def test_conf_int(self):
        pass

    def test_bic(self):
        pass

    def test_fit_regularized(self):
        model = self.res1.model

        alpha = np.ones(len(self.res1.params))
        alpha[-2:] = 0
        res_reg = model.fit_regularized(alpha=alpha*0.01, disp=0, maxiter=500)

        assert_allclose(res_reg.params[2:], self.res1.params[2:],
            atol=1e-1, rtol=1e-1)
    
class TestZeroInflatedNegativeBinomialP_predict(object):
    @classmethod
    def setup_class(cls):
        expected_params = [1, 0.5, 0.5]
        np.random.seed(123)
        nobs = 200
        exog = np.ones((nobs, 2))
        exog[:nobs//2, 1] = 2
        mu_true = exog.dot(expected_params[:-1])
        cls.endog = sm.distributions.zinegbin.rvs(mu_true, expected_params[-1],
                                                  2, 0.5, size=mu_true.shape)
        model = sm.ZeroInflatedNegativeBinomialP(cls.endog, exog, p=2)
        cls.res = model.fit(method='bfgs', maxiter=5000, maxfun=5000)

    def test_mean(self):
        assert_allclose(self.res.predict().mean(), self.endog.mean(),
                        atol=1e-2, rtol=1e-2)

    def test_var(self):
        assert_allclose((self.res.predict().mean() *
                        self.res._dispersion_factor.mean()),
                        self.endog.var(), atol=5e-2, rtol=5e-2)

    def test_predict_prob(self):
        res = self.res
        endog = res.model.endog

        pr = res.predict(which='prob')
        pr2 = sm.distributions.zinegbin.pmf(np.arange(10)[:,None],
            res.predict(), 0.5, 2, 0.5).T
        assert_allclose(pr, pr2, rtol=0.1, atol=0.1)

class TestZeroInflatedNegativeBinomialP_predict2(object):
        @classmethod
        def setup_class(cls):
            data = sm.datasets.randhie.load()

            cls.endog = data.endog
            exog = data.exog
            res = sm.ZeroInflatedNegativeBinomialP(
                cls.endog, exog, exog_infl=exog, p=2).fit(method="bfgs",
                                                          maxiter=1000)

            cls.res = res

        def test_mean(self):
            assert_allclose(self.res.predict().mean(), self.endog.mean(),
                            atol=0.02)

        def test_zero_nonzero_mean(self):
            mean1 = self.endog.mean()
            mean2 = ((1 - self.res.predict(which='prob-zero').mean()) *
                     self.res.predict(which='mean-nonzero').mean())
            assert_allclose(mean1, mean2, atol=0.2)

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, '-vvs', '-x', '--pdb'])
