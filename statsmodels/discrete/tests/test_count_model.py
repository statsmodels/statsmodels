import os
import numpy as np
from numpy.testing import (assert_, assert_raises, assert_almost_equal,
                           assert_equal, assert_array_equal, assert_allclose,
                           assert_array_less)

import statsmodels.api as sm
from .results.results_discrete import RandHIE

class TestZeroInflatedModel_logit(object):
    @classmethod
    def setup_class(cls):
        data = sm.datasets.randhie.load()
        cls.endog = data.endog
        exog = sm.add_constant(data.exog[:,1:4], prepend=False)
        exog_infl = sm.add_constant(data.exog[:,0], prepend=False)
        cls.res1 = sm.PoissonZeroInflated(data.endog, exog, 
            exog_infl=exog_infl, inflation='logit').fit(method='newton', maxiter=500)
        res2 = RandHIE()
        res2.zero_inflated_poisson_logit()
        cls.res2 = res2

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
        assert_allclose(self.res1.aic, self.res2.aic, atol=1e-1, rtol=1e-1)

    def test_fit_regularized(self):
        model = self.res1.model

        alpha = np.ones(len(self.res1.params))
        alpha[-2:] = 0
        res_reg = model.fit_regularized(alpha=alpha*0.01, disp=0)

        assert_allclose(res_reg.params, self.res1.params, atol=5e-5)
        assert_allclose(res_reg.bse, self.res1.bse, atol=5e-5)

class TestZeroInflatedModel_probit(object):
    @classmethod
    def setup_class(cls):
        data = sm.datasets.randhie.load()
        cls.endog = data.endog
        exog = sm.add_constant(data.exog[:,1:4], prepend=False)
        exog_infl = sm.add_constant(data.exog[:,0], prepend=False)
        cls.res1 = sm.PoissonZeroInflated(data.endog, exog,
            exog_infl=exog_infl, inflation='probit').fit(method='newton', maxiter=500)
        res2 = RandHIE()
        res2.zero_inflated_poisson_probit()
        cls.res2 = res2

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
        assert_allclose(self.res1.aic, self.res2.aic, atol=1e-1, rtol=1e-1)

class TestZeroInflatedModel_offset(object):
    @classmethod
    def setup_class(cls):
        data = sm.datasets.randhie.load()
        cls.endog = data.endog
        exog = sm.add_constant(data.exog[:,1:4], prepend=False)
        exog_infl = sm.add_constant(data.exog[:,0], prepend=False)
        cls.res1 = sm.PoissonZeroInflated(data.endog, exog,
            exog_infl=exog_infl, offset=data.exog[:,7]).fit(method='newton', maxiter=500)
        res2 = RandHIE()
        res2.zero_inflated_poisson_offset()
        cls.res2 = res2

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
        assert_allclose(self.res1.aic, self.res2.aic, atol=1e-1, rtol=1e-1)

class TestZeroInflatedModel_predict(object):
    @classmethod
    def setup_class(cls):
        expected_params = [1, 0.5]
        np.random.seed(123)
        nobs = 200
        exog = np.ones((nobs, 2))
        exog[:nobs//2, 1] = 2
        mu_true = exog.dot(expected_params)
        cls.endog = sm.distributions.zipoisson.rvs(mu_true, expected_params[-1],
                                                   size=mu_true.shape)
        model = sm.PoissonZeroInflated(cls.endog, exog, p=1)
        cls.res = model.fit(method='bfgs', maxiter=5000, maxfun=5000)

    def test_mean(self):
        assert_allclose(self.res.predict().mean(), self.endog.mean(),
                        atol=1e-2, rtol=1e-2)

    def test_var(self):
        assert_allclose((self.res.predict().mean() *
                        self.res._dispersion_factor.mean()),
                        self.endog.var(), atol=5e-2, rtol=5e-2)


if __name__ == "__main__":
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb'],
                   exit=False)
