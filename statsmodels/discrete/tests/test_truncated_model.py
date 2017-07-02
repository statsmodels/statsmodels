import os
import numpy as np
from numpy.testing import (assert_, assert_raises, assert_almost_equal,
                           assert_equal, assert_array_equal, assert_allclose,
                           assert_array_less)

import statsmodels.api as sm
from .results.results_discrete import RandHIE

class TestTruncatedPoissonModel(object):
    @classmethod
    def setup_class(cls):
        data = sm.datasets.randhie.load()
        endog = data.endog
        exog = sm.add_constant(data.exog[:,:4], prepend=False)
        cls.res1 = sm.TruncatedPoisson(data.endog, exog, truncation=5).fit(maxiter=500)
        res2 = RandHIE()
        res2.truncated_poisson()
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
        assert_allclose(self.res1.bic, self.res2.bic, atol=1e-2, rtol=1e-2)

    def test_fit_regularized(self):
        model = self.res1.model

        alpha = np.ones(len(self.res1.params))
        res_reg = model.fit_regularized(alpha=alpha*0.01, disp=0)

        assert_allclose(res_reg.params, self.res1.params, atol=5e-5)
        assert_allclose(res_reg.bse, self.res1.bse, atol=5e-5)

class TestZeroTruncatedPoissonModel(object):
    @classmethod
    def setup_class(cls):
        data = sm.datasets.randhie.load()
        endog = data.endog
        exog = sm.add_constant(data.exog[:,:4], prepend=False)
        cls.res1 = sm.TruncatedPoisson(data.endog, exog, truncation=0).fit(maxiter=500)
        res2 = RandHIE()
        res2.zero_truncated_poisson()
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
        assert_allclose(self.res1.bic, self.res2.bic, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, '-vvs', '-x', '--pdb'])