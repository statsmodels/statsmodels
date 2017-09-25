from unittest import TestCase

import numpy as np
from numpy.testing import assert_allclose

from statsmodels.regression.linear_model import WLS
from statsmodels.regression._tools import _MinimalWLS

class TestMinimalWLS(TestCase):
    @classmethod
    def setup_class(cls):
        rs = np.random.RandomState(1234)
        cls.exog1 = rs.randn(200,5)
        cls.endog1 = cls.exog1.sum(1) + rs.randn(200)
        cls.weights1 = 1.0 + np.sin(np.arange(200.0)/100.0*np.pi)
        cls.exog2 = rs.randn(50,1)
        cls.endog2 = 0.3 * cls.exog2.ravel() + rs.randn(50)
        cls.weights2 = 1.0 + np.log(np.arange(1.0,51.0))

    def test_equivalence_with_wls(self):
        res = WLS(self.endog1, self.exog1).fit()
        minres = _MinimalWLS(self.endog1, self.exog1).fit()
        assert_allclose(res.params, minres.params)
        assert_allclose(res.resid, minres.resid)

        res = WLS(self.endog2, self.exog2).fit()
        minres = _MinimalWLS(self.endog2, self.exog2).fit()
        assert_allclose(res.params, minres.params)
        assert_allclose(res.resid, minres.resid)

        res = WLS(self.endog1, self.exog1, weights=self.weights1).fit()
        minres = _MinimalWLS(self.endog1, self.exog1, weights=self.weights1).fit()
        assert_allclose(res.params, minres.params)
        assert_allclose(res.resid, minres.resid)

        res = WLS(self.endog2, self.exog2, weights=self.weights2).fit()
        minres = _MinimalWLS(self.endog2, self.exog2, weights=self.weights2).fit()
        assert_allclose(res.params, minres.params)
        assert_allclose(res.resid, minres.resid)
