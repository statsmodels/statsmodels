__author__ = 'kevin.sheppard'

from numpy.testing import TestCase, assert_almost_equal, assert_raises
import numpy.random as rnd
import numpy as np
import statsmodels.tsa.arima_model as ar
from statsmodels.regression.linear_model import OLS

# T = 1000
# burn = 500
# e = rnd.standard_normal(T + burn)
# y = np.zeros_like(e)
# for t in xrange(T + burn):
#     y[t] = 0.9 * y[t - 1] + e[t]
# y = y[-T:]
# reload(ar)
# arma_model = ar.ARMA(y, order=(0, 0))
# #res_nc = arma_model.fit(trend='nc')
#
# arma_model = ar.ARMA(y, order=(0, 0))
# res_c = arma_model.fit(trend='c')
#
# res_11 = ar.ARMA(y, order=(1, 1)).fit()
# res_11.summary()
#
# #res_c.summary2()
# res_c.predict()


class TestARMA00(TestCase):
    @classmethod
    def setup_class(cls):
        T = 1000
        burn = 500
        e = rnd.standard_normal(T + burn)
        y = np.zeros_like(e)
        for t in xrange(T + burn):
            y[t] = 0.9 * y[t - 1] + e[t]
        y = y[-T:]
        cls.y = y
        cls.arma_00_model = ar.ARMA(y, order=(0, 0))
        cls.arma_00_res = cls.arma_00_model.fit()


    def test_parameters(self):
        params = self.arma_00_res.params
        assert_almost_equal(self.y.mean(), params)

    def test_predictions(self):
        predictions = self.arma_00_res.predict()
        assert_almost_equal(self.y.mean() * np.ones_like(predictions), predictions)

    def test_information_criteria(self):
        res = self.arma_00_res
        y = self.y
        ols_res = OLS(y, np.ones_like(y)).fit()
        ols_ic = np.array([ols_res.aic, ols_res.bic])
        arma_ic = np.array([res.aic, res.bic])
        assert_almost_equal(ols_ic, arma_ic)

    def test_arma_00_nc(self):
        arma_00 = ar.ARMA(self.y, order=(0, 0))
        assert_raises(ValueError, arma_00.fit, trend='nc')


    def test_css(self):
        arma = ar.ARMA(self.y, order=(0, 0))
        fit = arma.fit(method='css')
        predictions = fit.predict()
        assert_almost_equal(self.y.mean() * np.ones_like(predictions), predictions)

    def test_arima(self):
        yi = np.cumsum(self.y)
        arima = ar.ARIMA(yi, order=(0, 1, 0))
        fit = arima.fit()
        assert_almost_equal(np.diff(yi).mean(), fit.params)


if __name__ == "__main__":
    import nose

    nose.runmodule(argv=[__file__, '-vvs', '-x'], exit=False) #, '--pdb'
