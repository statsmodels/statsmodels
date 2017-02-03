from statsmodels.compat.python import lrange, BytesIO, cPickle

import os
import warnings

from nose.tools import nottest
import numpy as np
from numpy.testing import (assert_almost_equal, assert_, assert_allclose,
                           assert_raises, dec, TestCase)
import pandas as pd
from pandas import PeriodIndex, DatetimeIndex

from statsmodels.datasets.macrodata import load as load_macrodata
from statsmodels.datasets.macrodata import load_pandas as load_macrodata_pandas
import statsmodels.sandbox.tsa.fftarma as fa
from statsmodels.tools.testing import assert_equal
from statsmodels.tsa.arma_mle import Arma
from statsmodels.tsa.arima_model import ARMA, ARIMA
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.base.datetools import dates_from_range
from statsmodels.tsa.tests.results import results_arma, results_arima
from statsmodels.tsa.arima_process import arma_generate_sample

try:
    import matplotlib.pyplot as plt
    have_matplotlib = True
except:
    have_matplotlib = False

DECIMAL_4 = 4
DECIMAL_3 = 3
DECIMAL_2 = 2
DECIMAL_1 = 1

current_path = os.path.dirname(os.path.abspath(__file__))
y_arma = np.genfromtxt(open(current_path + '/results/y_arma_data.csv', "rb"),
        delimiter=",", skip_header=1, dtype=float)

cpi_dates = PeriodIndex(start='1959q1', end='2009q3', freq='Q')
sun_dates = PeriodIndex(start='1700', end='2008', freq='A')
cpi_predict_dates = PeriodIndex(start='2009q3', end='2015q4', freq='Q')
sun_predict_dates = PeriodIndex(start='2008', end='2033', freq='A')


def test_compare_arma():
    #this is a preliminary test to compare arma_kf, arma_cond_ls and arma_cond_mle
    #the results returned by the fit methods are incomplete
    #for now without random.seed

    np.random.seed(9876565)
    x = fa.ArmaFft([1, -0.5], [1., 0.4], 40).generate_sample(nsample=200,
            burnin=1000)

    # this used kalman filter through descriptive
    #d = ARMA(x)
    #d.fit((1,1), trend='nc')
    #dres = d.res

    modkf = ARMA(x, (1,1))
    ##rkf = mkf.fit((1,1))
    ##rkf.params
    reskf = modkf.fit(trend='nc', disp=-1)
    dres = reskf

    modc = Arma(x)
    resls = modc.fit(order=(1,1))
    rescm = modc.fit_mle(order=(1,1), start_params=[0.4,0.4, 1.], disp=0)

    #decimal 1 corresponds to threshold of 5% difference
    #still different sign  corrcted
    #assert_almost_equal(np.abs(resls[0] / d.params), np.ones(d.params.shape), decimal=1)
    assert_almost_equal(resls[0] / dres.params, np.ones(dres.params.shape),
        decimal=1)
    #rescm also contains variance estimate as last element of params

    #assert_almost_equal(np.abs(rescm.params[:-1] / d.params), np.ones(d.params.shape), decimal=1)
    assert_almost_equal(rescm.params[:-1] / dres.params,
                        np.ones(dres.params.shape), decimal=1)
    #return resls[0], d.params, rescm.params


class CheckArmaResultsMixin(object):
    """
    res2 are the results from gretl.  They are in results/results_arma.
    res1 are from statsmodels
    """
    decimal_params = DECIMAL_4
    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params,
                self.decimal_params)

    decimal_aic = DECIMAL_4
    def test_aic(self):
        assert_almost_equal(self.res1.aic, self.res2.aic, self.decimal_aic)

    decimal_bic = DECIMAL_4
    def test_bic(self):
        assert_almost_equal(self.res1.bic, self.res2.bic, self.decimal_bic)

    decimal_arroots = DECIMAL_4
    def test_arroots(self):
        assert_almost_equal(self.res1.arroots, self.res2.arroots,
                    self.decimal_arroots)

    decimal_maroots = DECIMAL_4
    def test_maroots(self):
        assert_almost_equal(self.res1.maroots, self.res2.maroots,
                    self.decimal_maroots)

    decimal_bse = DECIMAL_2
    def test_bse(self):
        assert_almost_equal(self.res1.bse, self.res2.bse, self.decimal_bse)

    decimal_cov_params = DECIMAL_4
    def test_covparams(self):
        assert_almost_equal(self.res1.cov_params(), self.res2.cov_params,
                self.decimal_cov_params)

    decimal_hqic = DECIMAL_4
    def test_hqic(self):
        assert_almost_equal(self.res1.hqic, self.res2.hqic, self.decimal_hqic)

    decimal_llf = DECIMAL_4
    def test_llf(self):
        assert_almost_equal(self.res1.llf, self.res2.llf, self.decimal_llf)

    decimal_resid = DECIMAL_4
    def test_resid(self):
        assert_almost_equal(self.res1.resid, self.res2.resid,
                self.decimal_resid)

    decimal_fittedvalues = DECIMAL_4
    def test_fittedvalues(self):
        assert_almost_equal(self.res1.fittedvalues, self.res2.fittedvalues,
                self.decimal_fittedvalues)

    decimal_pvalues = DECIMAL_2
    def test_pvalues(self):
        assert_almost_equal(self.res1.pvalues, self.res2.pvalues,
                    self.decimal_pvalues)

    decimal_t = DECIMAL_2 # only 2 decimal places in gretl output
    def test_tvalues(self):
        assert_almost_equal(self.res1.tvalues, self.res2.tvalues,
                            self.decimal_t)

    decimal_sigma2 = DECIMAL_4
    def test_sigma2(self):
        assert_almost_equal(self.res1.sigma2, self.res2.sigma2,
                self.decimal_sigma2)

    def test_summary(self):
        # smoke tests
        table = self.res1.summary()



class CheckForecastMixin(object):
    decimal_forecast = DECIMAL_4
    def test_forecast(self):
        assert_almost_equal(self.res1.forecast_res, self.res2.forecast,
                self.decimal_forecast)

    decimal_forecasterr = DECIMAL_4
    def test_forecasterr(self):
        assert_almost_equal(self.res1.forecast_err, self.res2.forecasterr,
                self.decimal_forecasterr)


class CheckDynamicForecastMixin(object):
    decimal_forecast_dyn = 4
    def test_dynamic_forecast(self):
        assert_almost_equal(self.res1.forecast_res_dyn, self.res2.forecast_dyn,
                            self.decimal_forecast_dyn)

    #def test_forecasterr(self):
    #    assert_almost_equal(self.res1.forecast_err_dyn,
    #                        self.res2.forecasterr_dyn,
    #                        DECIMAL_4)


class CheckArimaResultsMixin(CheckArmaResultsMixin):
    def test_order(self):
        assert self.res1.k_diff == self.res2.k_diff
        assert self.res1.k_ar == self.res2.k_ar
        assert self.res1.k_ma == self.res2.k_ma

    decimal_predict_levels = DECIMAL_4
    def test_predict_levels(self):
        assert_almost_equal(self.res1.predict(typ='levels'), self.res2.linear,
                self.decimal_predict_levels)


class Test_Y_ARMA11_NoConst(CheckArmaResultsMixin, CheckForecastMixin):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,0]
        cls.res1 = ARMA(endog, order=(1,1)).fit(trend='nc', disp=-1)
        (cls.res1.forecast_res, cls.res1.forecast_err,
                confint) = cls.res1.forecast(10)
        cls.res2 = results_arma.Y_arma11()

    def test_pickle(self):
        fh = BytesIO()
        #test wrapped results load save pickle
        self.res1.save(fh)
        fh.seek(0,0)
        res_unpickled = self.res1.__class__.load(fh)
        assert_(type(res_unpickled) is type(self.res1))


class Test_Y_ARMA14_NoConst(CheckArmaResultsMixin):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,1]
        cls.res1 = ARMA(endog, order=(1,4)).fit(trend='nc', disp=-1)
        cls.res2 = results_arma.Y_arma14()


@dec.slow
class Test_Y_ARMA41_NoConst(CheckArmaResultsMixin, CheckForecastMixin):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,2]
        cls.res1 = ARMA(endog, order=(4,1)).fit(trend='nc', disp=-1)
        (cls.res1.forecast_res, cls.res1.forecast_err,
                confint) = cls.res1.forecast(10)
        cls.res2 = results_arma.Y_arma41()
        cls.decimal_maroots = DECIMAL_3


class Test_Y_ARMA22_NoConst(CheckArmaResultsMixin):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,3]
        cls.res1 = ARMA(endog, order=(2,2)).fit(trend='nc', disp=-1)
        cls.res2 = results_arma.Y_arma22()


class Test_Y_ARMA50_NoConst(CheckArmaResultsMixin, CheckForecastMixin):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,4]
        cls.res1 = ARMA(endog, order=(5,0)).fit(trend='nc', disp=-1)
        (cls.res1.forecast_res, cls.res1.forecast_err,
                confint) = cls.res1.forecast(10)
        cls.res2 = results_arma.Y_arma50()


class Test_Y_ARMA02_NoConst(CheckArmaResultsMixin):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,5]
        cls.res1 = ARMA(endog, order=(0,2)).fit(trend='nc', disp=-1)
        cls.res2 = results_arma.Y_arma02()


class Test_Y_ARMA11_Const(CheckArmaResultsMixin, CheckForecastMixin):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,6]
        cls.res1 = ARMA(endog, order=(1,1)).fit(trend="c", disp=-1)
        (cls.res1.forecast_res, cls.res1.forecast_err,
                confint) = cls.res1.forecast(10)
        cls.res2 = results_arma.Y_arma11c()


class Test_Y_ARMA14_Const(CheckArmaResultsMixin):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,7]
        cls.res1 = ARMA(endog, order=(1,4)).fit(trend="c", disp=-1)
        cls.res2 = results_arma.Y_arma14c()


class Test_Y_ARMA41_Const(CheckArmaResultsMixin, CheckForecastMixin):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,8]
        cls.res2 = results_arma.Y_arma41c()
        cls.res1 = ARMA(endog, order=(4,1)).fit(trend="c", disp=-1,
                                                start_params=cls.res2.params)
        (cls.res1.forecast_res, cls.res1.forecast_err,
                confint) = cls.res1.forecast(10)
        cls.decimal_cov_params = DECIMAL_3
        cls.decimal_fittedvalues = DECIMAL_3
        cls.decimal_resid = DECIMAL_3
        cls.decimal_params = DECIMAL_3


class Test_Y_ARMA22_Const(CheckArmaResultsMixin):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,9]
        cls.res1 = ARMA(endog, order=(2,2)).fit(trend="c", disp=-1)
        cls.res2 = results_arma.Y_arma22c()


class Test_Y_ARMA50_Const(CheckArmaResultsMixin, CheckForecastMixin):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,10]
        cls.res1 = ARMA(endog, order=(5,0)).fit(trend="c", disp=-1)
        (cls.res1.forecast_res, cls.res1.forecast_err,
                confint) = cls.res1.forecast(10)
        cls.res2 = results_arma.Y_arma50c()


class Test_Y_ARMA02_Const(CheckArmaResultsMixin):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,11]
        cls.res1 = ARMA(endog, order=(0,2)).fit(trend="c", disp=-1)
        cls.res2 = results_arma.Y_arma02c()


# cov_params and tvalues are off still but not as much vs. R
class Test_Y_ARMA11_NoConst_CSS(CheckArmaResultsMixin):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,0]
        cls.res1 = ARMA(endog, order=(1,1)).fit(method="css", trend='nc',
                            disp=-1)
        cls.res2 = results_arma.Y_arma11("css")
        cls.decimal_t = DECIMAL_1


# better vs. R
class Test_Y_ARMA14_NoConst_CSS(CheckArmaResultsMixin):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,1]
        cls.res1 = ARMA(endog, order=(1,4)).fit(method="css", trend='nc',
                            disp=-1)
        cls.res2 = results_arma.Y_arma14("css")
        cls.decimal_fittedvalues = DECIMAL_3
        cls.decimal_resid = DECIMAL_3
        cls.decimal_t = DECIMAL_1


# bse, etc. better vs. R
# maroot is off because maparams is off a bit (adjust tolerance?)
class Test_Y_ARMA41_NoConst_CSS(CheckArmaResultsMixin):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,2]
        cls.res1 = ARMA(endog, order=(4,1)).fit(method="css", trend='nc',
                        disp=-1)
        cls.res2 = results_arma.Y_arma41("css")
        cls.decimal_t = DECIMAL_1
        cls.decimal_pvalues = 0
        cls.decimal_cov_params = DECIMAL_3
        cls.decimal_maroots = DECIMAL_1


#same notes as above
class Test_Y_ARMA22_NoConst_CSS(CheckArmaResultsMixin):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,3]
        cls.res1 = ARMA(endog, order=(2,2)).fit(method="css", trend='nc',
                            disp=-1)
        cls.res2 = results_arma.Y_arma22("css")
        cls.decimal_t = DECIMAL_1
        cls.decimal_resid = DECIMAL_3
        cls.decimal_pvalues = DECIMAL_1
        cls.decimal_fittedvalues = DECIMAL_3


#NOTE: gretl just uses least squares for AR CSS
# so BIC, etc. is
# -2*res1.llf + np.log(nobs)*(res1.q+res1.p+res1.k)
# with no adjustment for p and no extra sigma estimate
#NOTE: so our tests use x-12 arima results which agree with us and are
# consistent with the rest of the models
class Test_Y_ARMA50_NoConst_CSS(CheckArmaResultsMixin):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,4]
        cls.res1 = ARMA(endog, order=(5,0)).fit(method="css", trend='nc',
                            disp=-1)
        cls.res2 = results_arma.Y_arma50("css")
        cls.decimal_t = 0
        cls.decimal_llf = DECIMAL_1 # looks like rounding error?


class Test_Y_ARMA02_NoConst_CSS(CheckArmaResultsMixin):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,5]
        cls.res1 = ARMA(endog, order=(0,2)).fit(method="css", trend='nc',
                            disp=-1)
        cls.res2 = results_arma.Y_arma02("css")


#NOTE: our results are close to --x-12-arima option and R
class Test_Y_ARMA11_Const_CSS(CheckArmaResultsMixin):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,6]
        cls.res1 = ARMA(endog, order=(1,1)).fit(trend="c", method="css",
                        disp=-1)
        cls.res2 = results_arma.Y_arma11c("css")
        cls.decimal_params = DECIMAL_3
        cls.decimal_cov_params = DECIMAL_3
        cls.decimal_t = DECIMAL_1


class Test_Y_ARMA14_Const_CSS(CheckArmaResultsMixin):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,7]
        cls.res1 = ARMA(endog, order=(1,4)).fit(trend="c", method="css",
                        disp=-1)
        cls.res2 = results_arma.Y_arma14c("css")
        cls.decimal_t = DECIMAL_1
        cls.decimal_pvalues = DECIMAL_1


class Test_Y_ARMA41_Const_CSS(CheckArmaResultsMixin):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,8]
        cls.res1 = ARMA(endog, order=(4,1)).fit(trend="c", method="css",
                        disp=-1)
        cls.res2 = results_arma.Y_arma41c("css")
        cls.decimal_t = DECIMAL_1
        cls.decimal_cov_params = DECIMAL_1
        cls.decimal_maroots = DECIMAL_3
        cls.decimal_bse = DECIMAL_1


class Test_Y_ARMA22_Const_CSS(CheckArmaResultsMixin):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,9]
        cls.res1 = ARMA(endog, order=(2,2)).fit(trend="c", method="css",
                        disp=-1)
        cls.res2 = results_arma.Y_arma22c("css")
        cls.decimal_t = 0
        cls.decimal_pvalues = DECIMAL_1


class Test_Y_ARMA50_Const_CSS(CheckArmaResultsMixin):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,10]
        cls.res1 = ARMA(endog, order=(5,0)).fit(trend="c", method="css",
                        disp=-1)
        cls.res2 = results_arma.Y_arma50c("css")
        cls.decimal_t = DECIMAL_1
        cls.decimal_params = DECIMAL_3
        cls.decimal_cov_params = DECIMAL_2


class Test_Y_ARMA02_Const_CSS(CheckArmaResultsMixin):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,11]
        cls.res1 = ARMA(endog, order=(0,2)).fit(trend="c", method="css",
                        disp=-1)
        cls.res2 = results_arma.Y_arma02c("css")


def test_reset_trend():
    endog = y_arma[:,0]
    mod = ARMA(endog, order=(1,1))
    res1 = mod.fit(trend="c", disp=-1)
    res2 = mod.fit(trend="nc", disp=-1)
    assert_equal(len(res1.params), len(res2.params)+1)


@dec.slow
def test_start_params_bug():
    data = np.array([1368., 1187, 1090, 1439, 2362, 2783, 2869, 2512, 1804,
    1544, 1028, 869, 1737, 2055, 1947, 1618, 1196, 867, 997, 1862, 2525,
    3250, 4023, 4018, 3585, 3004, 2500, 2441, 2749, 2466, 2157, 1847, 1463,
    1146, 851, 993, 1448, 1719, 1709, 1455, 1950, 1763, 2075, 2343, 3570,
    4690, 3700, 2339, 1679, 1466, 998, 853, 835, 922, 851, 1125, 1299, 1105,
    860, 701, 689, 774, 582, 419, 846, 1132, 902, 1058, 1341, 1551, 1167,
    975, 786, 759, 751, 649, 876, 720, 498, 553, 459, 543, 447, 415, 377,
    373, 324, 320, 306, 259, 220, 342, 558, 825, 994, 1267, 1473, 1601,
    1896, 1890, 2012, 2198, 2393, 2825, 3411, 3406, 2464, 2891, 3685, 3638,
    3746, 3373, 3190, 2681, 2846, 4129, 5054, 5002, 4801, 4934, 4903, 4713,
    4745, 4736, 4622, 4642, 4478, 4510, 4758, 4457, 4356, 4170, 4658, 4546,
    4402, 4183, 3574, 2586, 3326, 3948, 3983, 3997, 4422, 4496, 4276, 3467,
    2753, 2582, 2921, 2768, 2789, 2824, 2482, 2773, 3005, 3641, 3699, 3774,
    3698, 3628, 3180, 3306, 2841, 2014, 1910, 2560, 2980, 3012, 3210, 3457,
    3158, 3344, 3609, 3327, 2913, 2264, 2326, 2596, 2225, 1767, 1190, 792,
    669, 589, 496, 354, 246, 250, 323, 495, 924, 1536, 2081, 2660, 2814, 2992,
    3115, 2962, 2272, 2151, 1889, 1481, 955, 631, 288, 103, 60, 82, 107, 185,
    618, 1526, 2046, 2348, 2584, 2600, 2515, 2345, 2351, 2355, 2409, 2449,
    2645, 2918, 3187, 2888, 2610, 2740, 2526, 2383, 2936, 2968, 2635, 2617,
    2790, 3906, 4018, 4797, 4919, 4942, 4656, 4444, 3898, 3908, 3678, 3605,
    3186, 2139, 2002, 1559, 1235, 1183, 1096, 673, 389, 223, 352, 308, 365,
    525, 779, 894, 901, 1025, 1047, 981, 902, 759, 569, 519, 408, 263, 156,
    72, 49, 31, 41, 192, 423, 492, 552, 564, 723, 921, 1525, 2768, 3531, 3824,
    3835, 4294, 4533, 4173, 4221, 4064, 4641, 4685, 4026, 4323, 4585, 4836,
    4822, 4631, 4614, 4326, 4790, 4736, 4104, 5099, 5154, 5121, 5384, 5274,
    5225, 4899, 5382, 5295, 5349, 4977, 4597, 4069, 3733, 3439, 3052, 2626,
    1939, 1064, 713, 916, 832, 658, 817, 921, 772, 764, 824, 967, 1127, 1153,
    824, 912, 957, 990, 1218, 1684, 2030, 2119, 2233, 2657, 2652, 2682, 2498,
    2429, 2346, 2298, 2129, 1829, 1816, 1225, 1010, 748, 627, 469, 576, 532,
    475, 582, 641, 605, 699, 680, 714, 670, 666, 636, 672, 679, 446, 248, 134,
    160, 178, 286, 413, 676, 1025, 1159, 952, 1398, 1833, 2045, 2072, 1798,
    1799, 1358, 727, 353, 347, 844, 1377, 1829, 2118, 2272, 2745, 4263, 4314,
    4530, 4354, 4645, 4547, 5391, 4855, 4739, 4520, 4573, 4305, 4196, 3773,
    3368, 2596, 2596, 2305, 2756, 3747, 4078, 3415, 2369, 2210, 2316, 2263,
    2672, 3571, 4131, 4167, 4077, 3924, 3738, 3712, 3510, 3182, 3179, 2951,
    2453, 2078, 1999, 2486, 2581, 1891, 1997, 1366, 1294, 1536, 2794, 3211,
    3242, 3406, 3121, 2425, 2016, 1787, 1508, 1304, 1060, 1342, 1589, 2361,
    3452, 2659, 2857, 3255, 3322, 2852, 2964, 3132, 3033, 2931, 2636, 2818,
    3310, 3396, 3179, 3232, 3543, 3759, 3503, 3758, 3658, 3425, 3053, 2620,
    1837, 923, 712, 1054, 1376, 1556, 1498, 1523, 1088, 728, 890, 1413, 2524,
    3295, 4097, 3993, 4116, 3874, 4074, 4142, 3975, 3908, 3907, 3918, 3755,
    3648, 3778, 4293, 4385, 4360, 4352, 4528, 4365, 3846, 4098, 3860, 3230,
    2820, 2916, 3201, 3721, 3397, 3055, 2141, 1623, 1825, 1716, 2232, 2939,
    3735, 4838, 4560, 4307, 4975, 5173, 4859, 5268, 4992, 5100, 5070, 5270,
    4760, 5135, 5059, 4682, 4492, 4933, 4737, 4611, 4634, 4789, 4811, 4379,
    4689, 4284, 4191, 3313, 2770, 2543, 3105, 2967, 2420, 1996, 2247, 2564,
    2726, 3021, 3427, 3509, 3759, 3324, 2988, 2849, 2340, 2443, 2364, 1252,
    623, 742, 867, 684, 488, 348, 241, 187, 279, 355, 423, 678, 1375, 1497,
    1434, 2116, 2411, 1929, 1628, 1635, 1609, 1757, 2090, 2085, 1790, 1846,
    2038, 2360, 2342, 2401, 2920, 3030, 3132, 4385, 5483, 5865, 5595, 5485,
    5727, 5553, 5560, 5233, 5478, 5159, 5155, 5312, 5079, 4510, 4628, 4535,
    3656, 3698, 3443, 3146, 2562, 2304, 2181, 2293, 1950, 1930, 2197, 2796,
    3441, 3649, 3815, 2850, 4005, 5305, 5550, 5641, 4717, 5131, 2831, 3518,
    3354, 3115, 3515, 3552, 3244, 3658, 4407, 4935, 4299, 3166, 3335, 2728,
    2488, 2573, 2002, 1717, 1645, 1977, 2049, 2125, 2376, 2551, 2578, 2629,
    2750, 3150, 3699, 4062, 3959, 3264, 2671, 2205, 2128, 2133, 2095, 1964,
    2006, 2074, 2201, 2506, 2449, 2465, 2064, 1446, 1382, 983, 898, 489, 319,
    383, 332, 276, 224, 144, 101, 232, 429, 597, 750, 908, 960, 1076, 951,
    1062, 1183, 1404, 1391, 1419, 1497, 1267, 963, 682, 777, 906, 1149, 1439,
    1600, 1876, 1885, 1962, 2280, 2711, 2591, 2411])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ARMA(data, order=(4,1)).fit(disp=-1)


class Test_ARIMA101(CheckArmaResultsMixin):
    # just make sure this works
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,6]
        cls.res1 = ARIMA(endog, (1,0,1)).fit(trend="c", disp=-1)
        (cls.res1.forecast_res, cls.res1.forecast_err,
                confint) = cls.res1.forecast(10)
        cls.res2 = results_arma.Y_arma11c()
        cls.res2.k_diff = 0
        cls.res2.k_ar = 1
        cls.res2.k_ma = 1


class Test_ARIMA111(CheckArimaResultsMixin, CheckForecastMixin,
                    CheckDynamicForecastMixin):
    @classmethod
    def setupClass(cls):
        cpi = load_macrodata().data['cpi']
        cls.res1 = ARIMA(cpi, (1,1,1)).fit(disp=-1)
        cls.res2 = results_arima.ARIMA111()
        # make sure endog names changes to D.cpi
        cls.decimal_llf = 3
        cls.decimal_aic = 3
        cls.decimal_bic = 3
        cls.decimal_cov_params = 2 # this used to be better?
        cls.decimal_t = 0
        (cls.res1.forecast_res,
         cls.res1.forecast_err,
         conf_int)              = cls.res1.forecast(25)
        #cls.res1.forecast_res_dyn = cls.res1.predict(start=164, end=226, typ='levels', dynamic=True)
        #TODO: fix the indexing for the end here, I don't think this is right
        # if we're going to treat it like indexing
        # the forecast from 2005Q1 through 2009Q4 is indices
        # 184 through 227 not 226
        # note that the first one counts in the count so 164 + 64 is 65
        # predictions
        cls.res1.forecast_res_dyn = cls.res1.predict(start=164, end=164+63,
                                            typ='levels', dynamic=True)

    def test_freq(self):
        assert_almost_equal(self.res1.arfreq, [0.0000], 4)
        assert_almost_equal(self.res1.mafreq, [0.0000], 4)


class Test_ARIMA111CSS(CheckArimaResultsMixin, CheckForecastMixin,
                       CheckDynamicForecastMixin):
    @classmethod
    def setupClass(cls):
        cpi = load_macrodata().data['cpi']
        cls.res1 = ARIMA(cpi, (1,1,1)).fit(disp=-1, method='css')
        cls.res2 = results_arima.ARIMA111(method='css')
        cls.res2.fittedvalues = - cpi[1:-1] + cls.res2.linear
        # make sure endog names changes to D.cpi
        (cls.res1.forecast_res,
         cls.res1.forecast_err,
         conf_int)              = cls.res1.forecast(25)
        cls.decimal_forecast = 2
        cls.decimal_forecast_dyn = 2
        cls.decimal_forecasterr = 3
        cls.res1.forecast_res_dyn = cls.res1.predict(start=164, end=164+63,
                                                 typ='levels', dynamic=True)

        # precisions
        cls.decimal_arroots = 3
        cls.decimal_cov_params = 3
        cls.decimal_hqic = 3
        cls.decimal_maroots = 3
        cls.decimal_t = 1
        cls.decimal_fittedvalues = 2 # because of rounding when copying
        cls.decimal_resid = 2
        #cls.decimal_llf = 3
        #cls.decimal_aic = 3
        #cls.decimal_bic = 3
        cls.decimal_predict_levels = DECIMAL_2


class Test_ARIMA112CSS(CheckArimaResultsMixin):
    @classmethod
    def setupClass(cls):
        cpi = load_macrodata().data['cpi']
        cls.res1 = ARIMA(cpi, (1,1,2)).fit(disp=-1, method='css',
                                start_params = [.905322, -.692425, 1.07366,
                                                0.172024])
        cls.res2 = results_arima.ARIMA112(method='css')
        cls.res2.fittedvalues = - cpi[1:-1] + cls.res2.linear
        # make sure endog names changes to D.cpi
        cls.decimal_llf = 3
        cls.decimal_aic = 3
        cls.decimal_bic = 3
        #(cls.res1.forecast_res,
        # cls.res1.forecast_err,
        # conf_int)              = cls.res1.forecast(25)
        #cls.res1.forecast_res_dyn = cls.res1.predict(start=164, end=226, typ='levels', dynamic=True)
        #TODO: fix the indexing for the end here, I don't think this is right
        # if we're going to treat it like indexing
        # the forecast from 2005Q1 through 2009Q4 is indices
        # 184 through 227 not 226
        # note that the first one counts in the count so 164 + 64 is 65
        # predictions
        #cls.res1.forecast_res_dyn = self.predict(start=164, end=164+63,
        #                                         typ='levels', dynamic=True)
        # since we got from gretl don't have linear prediction in differences
        cls.decimal_arroots = 3
        cls.decimal_maroots = 2
        cls.decimal_t = 1
        cls.decimal_resid = 2
        cls.decimal_fittedvalues = 3
        cls.decimal_predict_levels = DECIMAL_3

    def test_freq(self):
        assert_almost_equal(self.res1.arfreq, [0.5000], 4)
        assert_almost_equal(self.res1.mafreq, [0.5000, 0.5000], 4)

#class Test_ARIMADates(CheckArmaResults, CheckForecast, CheckDynamicForecast):
#    @classmethod
#    def setupClass(cls):
#        from statsmodels.tsa.datetools import dates_from_range
#
#        cpi = load_macrodata().data['cpi']
#        dates = dates_from_range('1959q1', length=203)
#        cls.res1 = ARIMA(cpi, dates=dates, freq='Q').fit(order=(1,1,1), disp=-1)
#        cls.res2 = results_arima.ARIMA111()
#        # make sure endog names changes to D.cpi
#        cls.decimal_llf = 3
#        cls.decimal_aic = 3
#        cls.decimal_bic = 3
#        (cls.res1.forecast_res,
#         cls.res1.forecast_err,
#         conf_int)              = cls.res1.forecast(25)


def test_arima_predict_mle_dates():
    cpi = load_macrodata().data['cpi']
    res1 = ARIMA(cpi, (4,1,1), dates=cpi_dates, freq='Q').fit(disp=-1)

    with open(current_path + '/results/results_arima_forecasts_all_mle.csv', "rb") as test_data:
        arima_forecasts = np.genfromtxt(test_data, delimiter=",", skip_header=1, dtype=float)

    fc = arima_forecasts[:,0]
    fcdyn = arima_forecasts[:,1]
    fcdyn2 = arima_forecasts[:,2]

    start, end = 2, 51
    fv = res1.predict('1959Q3', '1971Q4', typ='levels')
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    assert_equal(res1.data.predict_dates, cpi_dates[start:end+1])

    start, end = 202, 227
    fv = res1.predict('2009Q3', '2015Q4', typ='levels')
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    assert_equal(res1.data.predict_dates, cpi_predict_dates)

    # make sure dynamic works

    start, end = '1960q2', '1971q4'
    fv = res1.predict(start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn[5:51+1], DECIMAL_4)

    start, end = '1965q1', '2015q4'
    fv = res1.predict(start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn2[24:227+1], DECIMAL_4)


def test_arma_predict_mle_dates():
    from statsmodels.datasets.sunspots import load
    sunspots = load().data['SUNACTIVITY']
    mod = ARMA(sunspots, (9,0), dates=sun_dates, freq='A')
    mod.method = 'mle'

    assert_raises(ValueError, mod._get_predict_start, *('1701', True))

    start, end = 2, 51
    _ = mod._get_predict_start('1702', False)
    _ = mod._get_predict_end('1751')
    assert_equal(mod.data.predict_dates, sun_dates[start:end+1])

    start, end = 308, 333
    _ = mod._get_predict_start('2008', False)
    _ = mod._get_predict_end('2033')
    assert_equal(mod.data.predict_dates, sun_predict_dates)


def test_arima_predict_css_dates():
    cpi = load_macrodata().data['cpi']
    res1 = ARIMA(cpi, (4,1,1), dates=cpi_dates, freq='Q').fit(disp=-1,
            method='css', trend='nc')

    params = np.array([ 1.231272508473910,
                       -0.282516097759915,
                        0.170052755782440,
                       -0.118203728504945,
                       -0.938783134717947])

    with open(current_path + '/results/results_arima_forecasts_all_css.csv', "rb") as test_data:
        arima_forecasts = np.genfromtxt(test_data, delimiter=",", skip_header=1, dtype=float)

    fc = arima_forecasts[:,0]
    fcdyn = arima_forecasts[:,1]
    fcdyn2 = arima_forecasts[:,2]

    start, end = 5, 51
    fv = res1.model.predict(params, '1960Q2', '1971Q4', typ='levels')
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    assert_equal(res1.data.predict_dates, cpi_dates[start:end+1])

    start, end = 202, 227
    fv = res1.model.predict(params, '2009Q3', '2015Q4', typ='levels')
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    assert_equal(res1.data.predict_dates, cpi_predict_dates)

    # make sure dynamic works
    start, end = 5, 51
    fv = res1.model.predict(params, '1960Q2', '1971Q4', typ='levels',
                                                        dynamic=True)
    assert_almost_equal(fv, fcdyn[start:end+1], DECIMAL_4)

    start, end = '1965q1', '2015q4'
    fv = res1.model.predict(params, start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn2[24:227+1], DECIMAL_4)


def test_arma_predict_css_dates():
    from statsmodels.datasets.sunspots import load
    sunspots = load().data['SUNACTIVITY']
    mod = ARMA(sunspots, (9,0), dates=sun_dates, freq='A')
    mod.method = 'css'
    assert_raises(ValueError, mod._get_predict_start, *('1701', False))


def test_arima_predict_mle():
    cpi = load_macrodata().data['cpi']
    res1 = ARIMA(cpi, (4,1,1)).fit(disp=-1)
    # fit the model so that we get correct endog length but use
    with open(current_path + '/results/results_arima_forecasts_all_mle.csv', "rb") as test_data:
        arima_forecasts = np.genfromtxt(test_data, delimiter=",", skip_header=1, dtype=float)
    fc = arima_forecasts[:,0]
    fcdyn = arima_forecasts[:,1]
    fcdyn2 = arima_forecasts[:,2]
    fcdyn3 = arima_forecasts[:,3]
    fcdyn4 = arima_forecasts[:,4]

    # 0 indicates the first sample-observation below
    # ie., the index after the pre-sample, these are also differenced once
    # so the indices are moved back once from the cpi in levels
    # start < p, end <p 1959q2 - 1959q4
    start, end = 1,3
    fv = res1.predict(start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # start < p, end 0 1959q3 - 1960q1
    start, end = 2, 4
    fv = res1.predict(start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # start < p, end >0 1959q3 - 1971q4
    start, end = 2, 51
    fv = res1.predict(start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # start < p, end nobs 1959q3 - 2009q3
    start, end = 2, 202
    fv = res1.predict(start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # start < p, end >nobs 1959q3 - 2015q4
    start, end = 2, 227
    fv = res1.predict(start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # start 0, end >0 1960q1 - 1971q4
    start, end = 4, 51
    fv = res1.predict(start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # start 0, end nobs 1960q1 - 2009q3
    start, end = 4, 202
    fv = res1.predict(start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # start 0, end >nobs 1960q1 - 2015q4
    start, end = 4, 227
    fv = res1.predict(start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # start >p, end >0 1965q1 - 1971q4
    start, end = 24, 51
    fv = res1.predict(start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # start >p, end nobs 1965q1 - 2009q3
    start, end = 24, 202
    fv = res1.predict(start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # start >p, end >nobs 1965q1 - 2015q4
    start, end = 24, 227
    fv = res1.predict(start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # start nobs, end nobs 2009q3 - 2009q3
    #NOTE: raises
    #start, end = 202, 202
    #fv = res1.predict(start, end, typ='levels')
    #assert_almost_equal(fv, [])
    # start nobs, end >nobs 2009q3 - 2015q4
    start, end = 202, 227
    fv = res1.predict(start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_3)
    # start >nobs, end >nobs 2009q4 - 2015q4
    #NOTE: this raises but shouldn't, dynamic forecasts could start
    #one period out
    start, end = 203, 227
    fv = res1.predict(start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # defaults
    start, end = None, None
    fv = res1.predict(start, end, typ='levels')
    assert_almost_equal(fv, fc[1:203], DECIMAL_4)

    #### Dynamic #####

    # start < p, end <p 1959q2 - 1959q4
    #NOTE: should raise
    #start, end = 1,3
    #fv = res1.predict(start, end, dynamic=True, typ='levels')
    #assert_almost_equal(fv, arima_forecasts[:,15])
    # start < p, end 0 1959q3 - 1960q1

    #NOTE: below should raise an error
    #start, end = 2, 4
    #fv = res1.predict(start, end, dynamic=True, typ='levels')
    #assert_almost_equal(fv, fcdyn[5:end+1], DECIMAL_4)
    # start < p, end >0 1959q3 - 1971q4
    #start, end = 2, 51
    #fv = res1.predict(start, end, dynamic=True, typ='levels')
    #assert_almost_equal(fv, fcdyn[5:end+1], DECIMAL_4)
    ## start < p, end nobs 1959q3 - 2009q3
    #start, end = 2, 202
    #fv = res1.predict(start, end, dynamic=True, typ='levels')
    #assert_almost_equal(fv, fcdyn[5:end+1], DECIMAL_4)
    ## start < p, end >nobs 1959q3 - 2015q4
    #start, end = 2, 227
    #fv = res1.predict(start, end, dynamic=True, typ='levels')
    #assert_almost_equal(fv, fcdyn[5:end+1], DECIMAL_4)
    # start 0, end >0 1960q1 - 1971q4
    start, end = 5, 51
    fv = res1.predict(start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn[start:end+1], DECIMAL_4)
    # start 0, end nobs 1960q1 - 2009q3
    start, end = 5, 202
    fv = res1.predict(start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn[start:end+1], DECIMAL_4)
    # start 0, end >nobs 1960q1 - 2015q4
    start, end = 5, 227
    fv = res1.predict(start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn[start:end+1], DECIMAL_4)
    # start >p, end >0 1965q1 - 1971q4
    start, end = 24, 51
    fv = res1.predict(start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn2[start:end+1], DECIMAL_4)
    # start >p, end nobs 1965q1 - 2009q3
    start, end = 24, 202
    fv = res1.predict(start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn2[start:end+1], DECIMAL_4)
    # start >p, end >nobs 1965q1 - 2015q4
    start, end = 24, 227
    fv = res1.predict(start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn2[start:end+1], DECIMAL_4)
    # start nobs, end nobs 2009q3 - 2009q3
    start, end = 202, 202
    fv = res1.predict(start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn3[start:end+1], DECIMAL_4)
    # start nobs, end >nobs 2009q3 - 2015q4
    start, end = 202, 227
    fv = res1.predict(start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn3[start:end+1], DECIMAL_4)
    # start >nobs, end >nobs 2009q4 - 2015q4
    start, end = 203, 227
    fv = res1.predict(start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn4[start:end+1], DECIMAL_4)
    # defaults
    start, end = None, None
    fv = res1.predict(start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn[5:203], DECIMAL_4)


def _check_start(model, given, expected, dynamic):
    start = model._get_predict_start(given, dynamic)
    assert_equal(start, expected)


def _check_end(model, given, end_expect, out_of_sample_expect):
    end, out_of_sample = model._get_predict_end(given)
    assert_equal((end, out_of_sample), (end_expect, out_of_sample_expect))


def test_arma_predict_indices():
    from statsmodels.datasets.sunspots import load
    sunspots = load().data['SUNACTIVITY']
    model = ARMA(sunspots, (9,0), dates=sun_dates, freq='A')
    model.method = 'mle'

    # raises - pre-sample + dynamic
    assert_raises(ValueError, model._get_predict_start, *(0, True))
    assert_raises(ValueError, model._get_predict_start, *(8, True))
    assert_raises(ValueError, model._get_predict_start, *('1700', True))
    assert_raises(ValueError, model._get_predict_start, *('1708', True))

    # raises - start out of sample
    assert_raises(ValueError, model._get_predict_start, *(311, True))
    assert_raises(ValueError, model._get_predict_start, *(311, False))
    assert_raises(ValueError, model._get_predict_start, *('2010', True))
    assert_raises(ValueError, model._get_predict_start, *('2010', False))

    # works - in-sample
    # None
                  # given, expected, dynamic
    start_test_cases = [
                  (None, 9, True),
                  # all start get moved back by k_diff
                  (9, 9, True),
                  (10, 10, True),
                  # what about end of sample start - last value is first
                  # forecast
                  (309, 309, True),
                  (308, 308, True),
                  (0, 0, False),
                  (1, 1, False),
                  (4, 4, False),

                  # all start get moved back by k_diff
                  ('1709', 9, True),
                  ('1710', 10, True),
                  # what about end of sample start - last value is first
                  # forecast
                  ('2008', 308, True),
                  ('2009', 309, True),
                  ('1700', 0, False),
                  ('1708', 8, False),
                  ('1709', 9, False),
                  ]

    for case in start_test_cases:
        _check_start(*((model,) + case))

    # the length of sunspot is 309, so last index is 208
    end_test_cases = [(None, 308, 0),
                      (307, 307, 0),
                      (308, 308, 0),
                      (309, 308, 1),
                      (312, 308, 4),
                      (51, 51, 0),
                      (333, 308, 25),

                      ('2007', 307, 0),
                      ('2008', 308, 0),
                      ('2009', 308, 1),
                      ('2012', 308, 4),
                      ('1815', 115, 0),
                      ('2033', 308, 25),
                      ]

    for case in end_test_cases:
        _check_end(*((model,)+case))


def test_arima_predict_indices():
    cpi = load_macrodata().data['cpi']
    model = ARIMA(cpi, (4,1,1), dates=cpi_dates, freq='Q')
    model.method = 'mle'

    # starting indices

    # raises - pre-sample + dynamic
    assert_raises(ValueError, model._get_predict_start, *(0, True))
    assert_raises(ValueError, model._get_predict_start, *(4, True))
    assert_raises(ValueError, model._get_predict_start, *('1959Q1', True))
    assert_raises(ValueError, model._get_predict_start, *('1960Q1', True))

    # raises - index differenced away
    assert_raises(ValueError, model._get_predict_start, *(0, False))
    assert_raises(ValueError, model._get_predict_start, *('1959Q1', False))

    # raises - start out of sample
    assert_raises(ValueError, model._get_predict_start, *(204, True))
    assert_raises(ValueError, model._get_predict_start, *(204, False))
    assert_raises(ValueError, model._get_predict_start, *('2010Q1', True))
    assert_raises(ValueError, model._get_predict_start, *('2010Q1', False))

    # works - in-sample
    # None
                  # given, expected, dynamic
    start_test_cases = [
                  (None, 4, True),
                  # all start get moved back by k_diff
                  (5, 4, True),
                  (6, 5, True),
                  # what about end of sample start - last value is first
                  # forecast
                  (203, 202, True),
                  (1, 0, False),
                  (4, 3, False),
                  (5, 4, False),
                  # all start get moved back by k_diff
                  ('1960Q2', 4, True),
                  ('1960Q3', 5, True),
                  # what about end of sample start - last value is first
                  # forecast
                  ('2009Q4', 202, True),
                  ('1959Q2', 0, False),
                  ('1960Q1', 3, False),
                  ('1960Q2', 4, False),
                  ]

    for case in start_test_cases:
        _check_start(*((model,) + case))

    # check raises
    #TODO: make sure dates are passing through unmolested
    #assert_raises(ValueError, model._get_predict_end, ("2001-1-1",))


    # the length of diff(cpi) is 202, so last index is 201
    end_test_cases = [(None, 201, 0),
                      (201, 200, 0),
                      (202, 201, 0),
                      (203, 201, 1),
                      (204, 201, 2),
                      (51, 50, 0),
                      (164+63, 201, 25),

                      ('2009Q2', 200, 0),
                      ('2009Q3', 201, 0),
                      ('2009Q4', 201, 1),
                      ('2010Q1', 201, 2),
                      ('1971Q4', 50, 0),
                      ('2015Q4', 201, 25),
                      ]

    for case in end_test_cases:
        _check_end(*((model,)+case))

    # check higher k_diff

    model.k_diff = 2
    # raises - pre-sample + dynamic
    assert_raises(ValueError, model._get_predict_start, *(0, True))
    assert_raises(ValueError, model._get_predict_start, *(5, True))
    assert_raises(ValueError, model._get_predict_start, *('1959Q1', True))
    assert_raises(ValueError, model._get_predict_start, *('1960Q1', True))

    # raises - index differenced away
    assert_raises(ValueError, model._get_predict_start, *(1, False))
    assert_raises(ValueError, model._get_predict_start, *('1959Q2', False))

    start_test_cases = [(None, 4, True),
                  # all start get moved back by k_diff
                  (6, 4, True),
                  # what about end of sample start - last value is first
                  # forecast
                  (203, 201, True),
                  (2, 0, False),
                  (4, 2, False),
                  (5, 3, False),
                  ('1960Q3', 4, True),
                  # what about end of sample start - last value is first
                  # forecast
                  ('2009Q4', 201, True),
                  ('2009Q4', 201, True),
                  ('1959Q3', 0, False),
                  ('1960Q1', 2, False),
                  ('1960Q2', 3, False),
                  ]

    for case in start_test_cases:
        _check_start(*((model,)+case))

    end_test_cases = [(None, 200, 0),
                      (201, 199, 0),
                      (202, 200, 0),
                      (203, 200, 1),
                      (204, 200, 2),
                      (51, 49, 0),
                      (164+63, 200, 25),

                      ('2009Q2', 199, 0),
                      ('2009Q3', 200, 0),
                      ('2009Q4', 200, 1),
                      ('2010Q1', 200, 2),
                      ('1971Q4', 49, 0),
                      ('2015Q4', 200, 25),
                      ]

    for case in end_test_cases:
        _check_end(*((model,)+case))


def test_arima_predict_indices_css():
    cpi = load_macrodata().data['cpi']
    #NOTE: Doing no-constant for now to kick the conditional exogenous
    #issue 274 down the road
    # go ahead and git the model to set up necessary variables
    model = ARIMA(cpi, (4,1,1))
    model.method = 'css'

    assert_raises(ValueError, model._get_predict_start, *(0, False))
    assert_raises(ValueError, model._get_predict_start, *(0, True))
    assert_raises(ValueError, model._get_predict_start, *(2, False))
    assert_raises(ValueError, model._get_predict_start, *(2, True))


def test_arima_predict_css():
    cpi = load_macrodata().data['cpi']
    #NOTE: Doing no-constant for now to kick the conditional exogenous
    #issue 274 down the road
    # go ahead and git the model to set up necessary variables
    res1 = ARIMA(cpi, (4,1,1)).fit(disp=-1, method="css",
                            trend="nc")
    # but use gretl parameters to predict to avoid precision problems
    params = np.array([ 1.231272508473910,
                       -0.282516097759915,
                       0.170052755782440,
                      -0.118203728504945,
                      -0.938783134717947])

    with open(current_path + '/results/results_arima_forecasts_all_css.csv', "rb") as test_data:
        arima_forecasts = np.genfromtxt(test_data, delimiter=",", skip_header=1, dtype=float)
    fc = arima_forecasts[:,0]
    fcdyn = arima_forecasts[:,1]
    fcdyn2 = arima_forecasts[:,2]
    fcdyn3 = arima_forecasts[:,3]
    fcdyn4 = arima_forecasts[:,4]

    #NOTE: should raise
    #start, end = 1,3
    #fv = res1.model.predict(params, start, end)
    ## start < p, end 0 1959q3 - 1960q1
    #start, end = 2, 4
    #fv = res1.model.predict(params, start, end)
    ## start < p, end >0 1959q3 - 1971q4
    #start, end = 2, 51
    #fv = res1.model.predict(params, start, end)
    ## start < p, end nobs 1959q3 - 2009q3
    #start, end = 2, 202
    #fv = res1.model.predict(params, start, end)
    ## start < p, end >nobs 1959q3 - 2015q4
    #start, end = 2, 227
    #fv = res1.model.predict(params, start, end)
    # start 0, end >0 1960q1 - 1971q4
    start, end = 5, 51
    fv = res1.model.predict(params, start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # start 0, end nobs 1960q1 - 2009q3
    start, end = 5, 202
    fv = res1.model.predict(params, start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # start 0, end >nobs 1960q1 - 2015q4
    #TODO: why detoriating precision?
    fv = res1.model.predict(params, start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # start >p, end >0 1965q1 - 1971q4
    start, end = 24, 51
    fv = res1.model.predict(params, start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # start >p, end nobs 1965q1 - 2009q3
    start, end = 24, 202
    fv = res1.model.predict(params, start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # start >p, end >nobs 1965q1 - 2015q4
    start, end = 24, 227
    fv = res1.model.predict(params, start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # start nobs, end nobs 2009q3 - 2009q3
    start, end = 202, 202
    fv = res1.model.predict(params, start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # start nobs, end >nobs 2009q3 - 2015q4
    start, end = 202, 227
    fv = res1.model.predict(params, start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # start >nobs, end >nobs 2009q4 - 2015q4
    start, end = 203, 227
    fv = res1.model.predict(params, start, end, typ='levels')
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # defaults
    start, end = None, None
    fv = res1.model.predict(params, start, end, typ='levels')
    assert_almost_equal(fv, fc[5:203], DECIMAL_4)

    #### Dynamic #####

    #NOTE: should raise
    # start < p, end <p 1959q2 - 1959q4
    #start, end = 1,3
    #fv = res1.predict(start, end, dynamic=True)
    # start < p, end 0 1959q3 - 1960q1
    #start, end = 2, 4
    #fv = res1.predict(start, end, dynamic=True)
    ## start < p, end >0 1959q3 - 1971q4
    #start, end = 2, 51
    #fv = res1.predict(start, end, dynamic=True)
    ## start < p, end nobs 1959q3 - 2009q3
    #start, end = 2, 202
    #fv = res1.predict(start, end, dynamic=True)
    ## start < p, end >nobs 1959q3 - 2015q4
    #start, end = 2, 227
    #fv = res1.predict(start, end, dynamic=True)
    # start 0, end >0 1960q1 - 1971q4
    start, end = 5, 51
    fv = res1.model.predict(params, start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn[start:end+1], DECIMAL_4)
    # start 0, end nobs 1960q1 - 2009q3
    start, end = 5, 202
    fv = res1.model.predict(params, start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn[start:end+1], DECIMAL_4)
    # start 0, end >nobs 1960q1 - 2015q4
    start, end = 5, 227
    fv = res1.model.predict(params, start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn[start:end+1], DECIMAL_4)
    # start >p, end >0 1965q1 - 1971q4
    start, end = 24, 51
    fv = res1.model.predict(params, start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn2[start:end+1], DECIMAL_4)
    # start >p, end nobs 1965q1 - 2009q3
    start, end = 24, 202
    fv = res1.model.predict(params, start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn2[start:end+1], DECIMAL_4)
    # start >p, end >nobs 1965q1 - 2015q4
    start, end = 24, 227
    fv = res1.model.predict(params, start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn2[start:end+1], DECIMAL_4)
    # start nobs, end nobs 2009q3 - 2009q3
    start, end = 202, 202
    fv = res1.model.predict(params, start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn3[start:end+1], DECIMAL_4)
    # start nobs, end >nobs 2009q3 - 2015q4
    start, end = 202, 227
    fv = res1.model.predict(params, start, end, dynamic=True, typ='levels')
    # start >nobs, end >nobs 2009q4 - 2015q4
    start, end = 203, 227
    fv = res1.model.predict(params, start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn4[start:end+1], DECIMAL_4)
    # defaults
    start, end = None, None
    fv = res1.model.predict(params, start, end, dynamic=True, typ='levels')
    assert_almost_equal(fv, fcdyn[5:203], DECIMAL_4)


def test_arima_predict_css_diffs():

    cpi = load_macrodata().data['cpi']
    #NOTE: Doing no-constant for now to kick the conditional exogenous
    #issue 274 down the road
    # go ahead and git the model to set up necessary variables
    res1 = ARIMA(cpi, (4,1,1)).fit(disp=-1, method="css",
                            trend="c")
    # but use gretl parameters to predict to avoid precision problems
    params = np.array([0.78349893861244,
                      -0.533444105973324,
                       0.321103691668809,
                       0.264012463189186,
                       0.107888256920655,
                       0.920132542916995])
    # we report mean, should we report constant?
    params[0] = params[0] / (1 - params[1:5].sum())

    with open(current_path + '/results/results_arima_forecasts_all_css_diff.csv', "rb") as test_data:
        arima_forecasts = np.genfromtxt(test_data, delimiter=",", skip_header=1, dtype=float)
    fc = arima_forecasts[:,0]
    fcdyn = arima_forecasts[:,1]
    fcdyn2 = arima_forecasts[:,2]
    fcdyn3 = arima_forecasts[:,3]
    fcdyn4 = arima_forecasts[:,4]

    #NOTE: should raise
    #start, end = 1,3
    #fv = res1.model.predict(params, start, end)
    ## start < p, end 0 1959q3 - 1960q1
    #start, end = 2, 4
    #fv = res1.model.predict(params, start, end)
    ## start < p, end >0 1959q3 - 1971q4
    #start, end = 2, 51
    #fv = res1.model.predict(params, start, end)
    ## start < p, end nobs 1959q3 - 2009q3
    #start, end = 2, 202
    #fv = res1.model.predict(params, start, end)
    ## start < p, end >nobs 1959q3 - 2015q4
    #start, end = 2, 227
    #fv = res1.model.predict(params, start, end)
    # start 0, end >0 1960q1 - 1971q4
    start, end = 5, 51
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # start 0, end nobs 1960q1 - 2009q3
    start, end = 5, 202
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # start 0, end >nobs 1960q1 - 2015q4
    #TODO: why detoriating precision?
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # start >p, end >0 1965q1 - 1971q4
    start, end = 24, 51
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # start >p, end nobs 1965q1 - 2009q3
    start, end = 24, 202
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # start >p, end >nobs 1965q1 - 2015q4
    start, end = 24, 227
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # start nobs, end nobs 2009q3 - 2009q3
    start, end = 202, 202
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # start nobs, end >nobs 2009q3 - 2015q4
    start, end = 202, 227
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # start >nobs, end >nobs 2009q4 - 2015q4
    start, end = 203, 227
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # defaults
    start, end = None, None
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[5:203], DECIMAL_4)

    #### Dynamic #####

    #NOTE: should raise
    # start < p, end <p 1959q2 - 1959q4
    #start, end = 1,3
    #fv = res1.predict(start, end, dynamic=True)
    # start < p, end 0 1959q3 - 1960q1
    #start, end = 2, 4
    #fv = res1.predict(start, end, dynamic=True)
    ## start < p, end >0 1959q3 - 1971q4
    #start, end = 2, 51
    #fv = res1.predict(start, end, dynamic=True)
    ## start < p, end nobs 1959q3 - 2009q3
    #start, end = 2, 202
    #fv = res1.predict(start, end, dynamic=True)
    ## start < p, end >nobs 1959q3 - 2015q4
    #start, end = 2, 227
    #fv = res1.predict(start, end, dynamic=True)
    # start 0, end >0 1960q1 - 1971q4
    start, end = 5, 51
    fv = res1.model.predict(params, start, end, dynamic=True)
    assert_almost_equal(fv, fcdyn[start:end+1], DECIMAL_4)
    # start 0, end nobs 1960q1 - 2009q3
    start, end = 5, 202
    fv = res1.model.predict(params, start, end, dynamic=True)
    assert_almost_equal(fv, fcdyn[start:end+1], DECIMAL_4)
    # start 0, end >nobs 1960q1 - 2015q4
    start, end = 5, 227
    fv = res1.model.predict(params, start, end, dynamic=True)
    assert_almost_equal(fv, fcdyn[start:end+1], DECIMAL_4)
    # start >p, end >0 1965q1 - 1971q4
    start, end = 24, 51
    fv = res1.model.predict(params, start, end, dynamic=True)
    assert_almost_equal(fv, fcdyn2[start:end+1], DECIMAL_4)
    # start >p, end nobs 1965q1 - 2009q3
    start, end = 24, 202
    fv = res1.model.predict(params, start, end, dynamic=True)
    assert_almost_equal(fv, fcdyn2[start:end+1], DECIMAL_4)
    # start >p, end >nobs 1965q1 - 2015q4
    start, end = 24, 227
    fv = res1.model.predict(params, start, end, dynamic=True)
    assert_almost_equal(fv, fcdyn2[start:end+1], DECIMAL_4)
    # start nobs, end nobs 2009q3 - 2009q3
    start, end = 202, 202
    fv = res1.model.predict(params, start, end, dynamic=True)
    assert_almost_equal(fv, fcdyn3[start:end+1], DECIMAL_4)
    # start nobs, end >nobs 2009q3 - 2015q4
    start, end = 202, 227
    fv = res1.model.predict(params, start, end, dynamic=True)
    # start >nobs, end >nobs 2009q4 - 2015q4
    start, end = 203, 227
    fv = res1.model.predict(params, start, end, dynamic=True)
    assert_almost_equal(fv, fcdyn4[start:end+1], DECIMAL_4)
    # defaults
    start, end = None, None
    fv = res1.model.predict(params, start, end, dynamic=True)
    assert_almost_equal(fv, fcdyn[5:203], DECIMAL_4)


def test_arima_predict_mle_diffs():

    cpi = load_macrodata().data['cpi']
    #NOTE: Doing no-constant for now to kick the conditional exogenous
    #issue 274 down the road
    # go ahead and git the model to set up necessary variables
    res1 = ARIMA(cpi, (4,1,1)).fit(disp=-1, trend="c")
    # but use gretl parameters to predict to avoid precision problems
    params = np.array([0.926875951549299,
        -0.555862621524846,
        0.320865492764400,
        0.252253019082800,
        0.113624958031799,
        0.939144026934634])

    with open(current_path + '/results/results_arima_forecasts_all_mle_diff.csv', "rb") as test_data:
        arima_forecasts = np.genfromtxt(test_data, delimiter=",", skip_header=1, dtype=float)
    fc = arima_forecasts[:,0]
    fcdyn = arima_forecasts[:,1]
    fcdyn2 = arima_forecasts[:,2]
    fcdyn3 = arima_forecasts[:,3]
    fcdyn4 = arima_forecasts[:,4]

    #NOTE: should raise
    start, end = 1,3
    fv = res1.model.predict(params, start, end)
    ## start < p, end 0 1959q3 - 1960q1
    start, end = 2, 4
    fv = res1.model.predict(params, start, end)
    ## start < p, end >0 1959q3 - 1971q4
    start, end = 2, 51
    fv = res1.model.predict(params, start, end)
    ## start < p, end nobs 1959q3 - 2009q3
    start, end = 2, 202
    fv = res1.model.predict(params, start, end)
    ## start < p, end >nobs 1959q3 - 2015q4
    start, end = 2, 227
    fv = res1.model.predict(params, start, end)
    # start 0, end >0 1960q1 - 1971q4
    start, end = 5, 51
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # start 0, end nobs 1960q1 - 2009q3
    start, end = 5, 202
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # start 0, end >nobs 1960q1 - 2015q4
    #TODO: why detoriating precision?
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # start >p, end >0 1965q1 - 1971q4
    start, end = 24, 51
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # start >p, end nobs 1965q1 - 2009q3
    start, end = 24, 202
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # start >p, end >nobs 1965q1 - 2015q4
    start, end = 24, 227
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # start nobs, end nobs 2009q3 - 2009q3
    start, end = 202, 202
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # start nobs, end >nobs 2009q3 - 2015q4
    start, end = 202, 227
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # start >nobs, end >nobs 2009q4 - 2015q4
    start, end = 203, 227
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[start:end+1], DECIMAL_4)
    # defaults
    start, end = None, None
    fv = res1.model.predict(params, start, end)
    assert_almost_equal(fv, fc[1:203], DECIMAL_4)

    #### Dynamic #####

    #NOTE: should raise
    # start < p, end <p 1959q2 - 1959q4
    #start, end = 1,3
    #fv = res1.predict(start, end, dynamic=True)
    # start < p, end 0 1959q3 - 1960q1
    #start, end = 2, 4
    #fv = res1.predict(start, end, dynamic=True)
    ## start < p, end >0 1959q3 - 1971q4
    #start, end = 2, 51
    #fv = res1.predict(start, end, dynamic=True)
    ## start < p, end nobs 1959q3 - 2009q3
    #start, end = 2, 202
    #fv = res1.predict(start, end, dynamic=True)
    ## start < p, end >nobs 1959q3 - 2015q4
    #start, end = 2, 227
    #fv = res1.predict(start, end, dynamic=True)
    # start 0, end >0 1960q1 - 1971q4
    start, end = 5, 51
    fv = res1.model.predict(params, start, end, dynamic=True)
    assert_almost_equal(fv, fcdyn[start:end+1], DECIMAL_4)
    # start 0, end nobs 1960q1 - 2009q3
    start, end = 5, 202
    fv = res1.model.predict(params, start, end, dynamic=True)
    assert_almost_equal(fv, fcdyn[start:end+1], DECIMAL_4)
    # start 0, end >nobs 1960q1 - 2015q4
    start, end = 5, 227
    fv = res1.model.predict(params, start, end, dynamic=True)
    assert_almost_equal(fv, fcdyn[start:end+1], DECIMAL_4)
    # start >p, end >0 1965q1 - 1971q4
    start, end = 24, 51
    fv = res1.model.predict(params, start, end, dynamic=True)
    assert_almost_equal(fv, fcdyn2[start:end+1], DECIMAL_4)
    # start >p, end nobs 1965q1 - 2009q3
    start, end = 24, 202
    fv = res1.model.predict(params, start, end, dynamic=True)
    assert_almost_equal(fv, fcdyn2[start:end+1], DECIMAL_4)
    # start >p, end >nobs 1965q1 - 2015q4
    start, end = 24, 227
    fv = res1.model.predict(params, start, end, dynamic=True)
    assert_almost_equal(fv, fcdyn2[start:end+1], DECIMAL_4)
    # start nobs, end nobs 2009q3 - 2009q3
    start, end = 202, 202
    fv = res1.model.predict(params, start, end, dynamic=True)
    assert_almost_equal(fv, fcdyn3[start:end+1], DECIMAL_4)
    # start nobs, end >nobs 2009q3 - 2015q4
    start, end = 202, 227
    fv = res1.model.predict(params, start, end, dynamic=True)
    # start >nobs, end >nobs 2009q4 - 2015q4
    start, end = 203, 227
    fv = res1.model.predict(params, start, end, dynamic=True)
    assert_almost_equal(fv, fcdyn4[start:end+1], DECIMAL_4)
    # defaults
    start, end = None, None
    fv = res1.model.predict(params, start, end, dynamic=True)
    assert_almost_equal(fv, fcdyn[5:203], DECIMAL_4)


def test_arima_wrapper():
    cpi = load_macrodata_pandas().data['cpi']
    cpi.index = pd.Index(cpi_dates)
    res = ARIMA(cpi, (4,1,1), freq='Q').fit(disp=-1)
    assert_equal(res.params.index, pd.Index(['const', 'ar.L1.D.cpi', 'ar.L2.D.cpi',
                                    'ar.L3.D.cpi', 'ar.L4.D.cpi',
                                    'ma.L1.D.cpi']))
    assert_equal(res.model.endog_names, 'D.cpi')


def test_1dexog():
    # smoke test, this will raise an error if broken
    dta = load_macrodata_pandas().data
    endog = dta['realcons'].values
    exog = dta['m1'].values.squeeze()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = ARMA(endog, (1,1), exog).fit(disp=-1)
        mod.predict(193, 203, exog[-10:])

        # check for dynamic is true and pandas Series  see #2589
        mod.predict(193, 202, exog[-10:], dynamic=True)

        dta.index = pd.Index(cpi_dates)
        mod = ARMA(dta['realcons'], (1,1), dta['m1']).fit(disp=-1)
        mod.predict(dta.index[-10], dta.index[-1], exog=dta['m1'][-10:], dynamic=True)

        mod = ARMA(dta['realcons'], (1,1), dta['m1']).fit(trend='nc', disp=-1)
        mod.predict(dta.index[-10], dta.index[-1], exog=dta['m1'][-10:], dynamic=True)


def test_arima_predict_bug():
    #predict_start_date wasn't getting set on start = None
    from statsmodels.datasets import sunspots
    dta = sunspots.load_pandas().data.SUNACTIVITY
    dta.index = pd.Index(dates_from_range('1700', '2008'))
    arma_mod20 = ARMA(dta, (2,0)).fit(disp=-1)
    arma_mod20.predict(None, None)

    # test prediction with time stamp, see #2587
    predict = arma_mod20.predict(dta.index[-20], dta.index[-1])
    assert_(predict.index.equals(dta.index[-20:]))
    predict = arma_mod20.predict(dta.index[-20], dta.index[-1], dynamic=True)
    assert_(predict.index.equals(dta.index[-20:]))
    # partially out of sample
    predict_dates = pd.Index(dates_from_range('2000', '2015'))
    predict = arma_mod20.predict(predict_dates[0], predict_dates[-1])
    assert_(predict.index.equals(predict_dates))
    #assert_(1 == 0)


def test_arima_predict_q2():
    # bug with q > 1 for arima predict
    inv = load_macrodata().data['realinv']
    arima_mod = ARIMA(np.log(inv), (1,1,2)).fit(start_params=[0,0,0,0], disp=-1)
    fc, stderr, conf_int = arima_mod.forecast(5)
    # values copy-pasted from gretl
    assert_almost_equal(fc,
                        [7.306320, 7.313825, 7.321749, 7.329827, 7.337962],
                        5)


def test_arima_predict_pandas_nofreq():
    # this is issue 712
    dates = ["2010-01-04", "2010-01-05", "2010-01-06", "2010-01-07",
             "2010-01-08", "2010-01-11", "2010-01-12", "2010-01-11",
             "2010-01-12", "2010-01-13", "2010-01-17"]
    close = [626.75, 623.99, 608.26, 594.1, 602.02, 601.11, 590.48, 587.09,
             589.85, 580.0,587.62]
    data = pd.DataFrame(close, index=DatetimeIndex(dates), columns=["close"])

    #TODO: fix this names bug for non-string names names
    arma = ARMA(data, order=(1,0)).fit(disp=-1)

    # first check that in-sample prediction works
    predict = arma.predict()
    assert_(predict.index.equals(data.index))

    # check that this raises an exception when date not on index
    assert_raises(ValueError, arma.predict, start="2010-1-9", end=10)
    assert_raises(ValueError, arma.predict, start="2010-1-9", end="2010-1-17")

    # raise because end not on index
    assert_raises(ValueError, arma.predict, start="2010-1-4", end="2010-1-10")
    # raise because end not on index
    assert_raises(ValueError, arma.predict, start=3, end="2010-1-10")

    predict = arma.predict(start="2010-1-7", end=10) # should be of length 10
    assert_(len(predict) == 8)
    assert_(predict.index.equals(data.index[3:10+1]))

    predict = arma.predict(start="2010-1-7", end=14)
    assert_(predict.index.equals(pd.Index(lrange(3, 15))))

    predict = arma.predict(start=3, end=14)
    assert_(predict.index.equals(pd.Index(lrange(3, 15))))

    # end can be a date if it's in the sample and on the index
    # predict dates is just a slice of the dates index then
    predict = arma.predict(start="2010-1-6", end="2010-1-13")
    assert_(predict.index.equals(data.index[2:10]))
    predict = arma.predict(start=2, end="2010-1-13")
    assert_(predict.index.equals(data.index[2:10]))


def test_arima_predict_exog():
    # check 625 and 626
    #from statsmodels.tsa.arima_process import arma_generate_sample
    #arparams = np.array([1, -.45, .25])
    #maparams = np.array([1, .15])
    #nobs = 100
    #np.random.seed(123)
    #y = arma_generate_sample(arparams, maparams, nobs, burnin=100)

    ## make an exogenous trend
    #X = np.array(lrange(nobs)) / 20.0
    ## add a constant
    #y += 2.5

    from pandas import read_csv
    arima_forecasts = read_csv(current_path + "/results/"
                            "results_arima_exog_forecasts_mle.csv")
    y = arima_forecasts["y"].dropna()
    X = np.arange(len(y) + 25)/20.
    predict_expected = arima_forecasts["predict"]
    arma_res = ARMA(y.values, order=(2,1), exog=X[:100]).fit(trend="c",
                                                             disp=-1)
    # params from gretl
    params = np.array([2.786912485145725, -0.122650190196475,
                       0.533223846028938, -0.319344321763337,
                       0.132883233000064])
    assert_almost_equal(arma_res.params, params, 5)
    # no exog for in-sample
    predict = arma_res.predict()
    assert_almost_equal(predict, predict_expected.values[:100], 5)

    # check 626
    assert_(len(arma_res.model.exog_names) == 5)

    # exog for out-of-sample and in-sample dynamic
    predict = arma_res.model.predict(params, end=124, exog=X[100:])
    assert_almost_equal(predict, predict_expected.values, 6)

    # conditional sum of squares
    #arima_forecasts = read_csv(current_path + "/results/"
    #                        "results_arima_exog_forecasts_css.csv")
    #predict_expected = arima_forecasts["predict"].dropna()
    #arma_res = ARMA(y.values, order=(2,1), exog=X[:100]).fit(trend="c",
    #                                                         method="css",
    #                                                         disp=-1)

    #params = np.array([2.152350033809826, -0.103602399018814,
    #                   0.566716580421188, -0.326208009247944,
    #                   0.102142932143421])
    #predict = arma_res.model.predict(params)
    ## in-sample
    #assert_almost_equal(predict, predict_expected.values[:98], 6)


    #predict = arma_res.model.predict(params, end=124, exog=X[100:])
    ## exog for out-of-sample and in-sample dynamic
    #assert_almost_equal(predict, predict_expected.values, 3)


def test_arima_no_diff():
    # issue 736
    # smoke test, predict will break if we have ARIMAResults but
    # ARMA model, need ARIMA(p, 0, q) to return an ARMA in init.
    ar = [1, -.75, .15, .35]
    ma = [1, .25, .9]
    y = arma_generate_sample(ar, ma, 100)
    mod = ARIMA(y, (3, 0, 2))
    assert_(type(mod) is ARMA)
    res = mod.fit(disp=-1)
    # smoke test just to be sure
    res.predict()


def test_arima_predict_noma():
    # issue 657
    # smoke test
    ar = [1, .75]
    ma = [1]
    data = arma_generate_sample(ar, ma, 100)
    arma = ARMA(data, order=(0,1))
    arma_res = arma.fit(disp=-1)
    arma_res.forecast(1)


def test_arimax():
    dta = load_macrodata_pandas().data
    dates = dates_from_range("1959Q1", length=len(dta))
    dta.index = cpi_dates
    dta = dta[["realdpi", "m1", "realgdp"]]
    y = dta.pop("realdpi")

    # 1 exog
    #X = dta.ix[1:]["m1"]
    #res = ARIMA(y, (2, 1, 1), X).fit(disp=-1)
    #params = [23.902305009084373, 0.024650911502790, -0.162140641341602,
    #          0.165262136028113, -0.066667022903974]
    #assert_almost_equal(res.params.values, params, 6)


    # 2 exog
    X = dta
    res = ARIMA(y, (2, 1, 1), X).fit(disp=False, solver="nm", maxiter=1000,
                ftol=1e-12, xtol=1e-12)

    # from gretl
    #params = [13.113976653926638, -0.003792125069387,  0.004123504809217,
    #          -0.199213760940898,  0.151563643588008, -0.033088661096699]
    # from stata using double
    stata_llf = -1076.108614859121
    params = [13.1259220104, -0.00376814509403812, 0.00411970083135622,
              -0.19921477896158524, 0.15154396192855729, -0.03308400760360837]
    # we can get close
    assert_almost_equal(res.params.values, params, 4)

    # This shows that it's an optimizer problem and not a problem in the code
    assert_almost_equal(res.model.loglike(np.array(params)), stata_llf, 6)

    X = dta.diff()
    X.iloc[0] = 0
    res = ARIMA(y, (2, 1, 1), X).fit(disp=False)

    # gretl won't estimate this - looks like maybe a bug on their part,
    # but we can just fine, we're close to Stata's answer
    # from Stata
    params = [19.5656863783347, 0.32653841355833396198,
              0.36286527042965188716, -1.01133792126884,
              -0.15722368379307766206, 0.69359822544092153418]

    assert_almost_equal(res.params.values, params, 3)


def test_bad_start_params():
    endog = np.array([820.69093, 781.0103028, 785.8786988, 767.64282267,
         778.9837648 ,   824.6595702 ,   813.01877867,   751.65598567,
         753.431091  ,   746.920813  ,   795.6201904 ,   772.65732833,
         793.4486454 ,   868.8457766 ,   823.07226547,   783.09067747,
         791.50723847,   770.93086347,   835.34157333,   810.64147947,
         738.36071367,   776.49038513,   822.93272333,   815.26461227,
         773.70552987,   777.3726522 ,   811.83444853,   840.95489133,
         777.51031933,   745.90077307,   806.95113093,   805.77521973,
         756.70927733,   749.89091773,  1694.2266924 ,  2398.4802244 ,
        1434.6728516 ,   909.73940427,   929.01291907,   769.07561453,
         801.1112548 ,   796.16163313,   817.2496376 ,   857.73046447,
         838.849345  ,   761.92338873,   731.7842242 ,   770.4641844 ])
    mod = ARMA(endog, (15, 0))
    assert_raises(ValueError, mod.fit)

    inv = load_macrodata().data['realinv']
    arima_mod = ARIMA(np.log(inv), (1,1,2))
    assert_raises(ValueError, mod.fit)


def test_arima_small_data_bug():
    # Issue 1038, too few observations with given order
    from datetime import datetime
    import statsmodels.api as sm

    vals = [96.2, 98.3, 99.1, 95.5, 94.0, 87.1, 87.9, 86.7402777504474]

    dr = dates_from_range("1990q1", length=len(vals))
    ts = pd.Series(vals, index=dr)
    df = pd.DataFrame(ts)
    mod = sm.tsa.ARIMA(df, (2, 0, 2))
    assert_raises(ValueError, mod.fit)


def test_arima_dataframe_integer_name():
    # Smoke Test for Issue 1038
    from datetime import datetime
    import statsmodels.api as sm

    vals = [96.2, 98.3, 99.1, 95.5, 94.0, 87.1, 87.9, 86.7402777504474,
            94.0, 96.5, 93.3, 97.5, 96.3, 92.]

    dr = dates_from_range("1990q1", length=len(vals))
    ts = pd.Series(vals, index=dr)
    df = pd.DataFrame(ts)
    mod = sm.tsa.ARIMA(df, (2, 0, 2))


def test_arima_exog_predict_1d():
    # test 1067
    np.random.seed(12345)
    y = np.random.random(100)
    x = np.random.random(100)
    mod = ARMA(y, (2, 1), x).fit(disp=-1)
    newx = np.random.random(10)
    results = mod.forecast(steps=10, alpha=0.05, exog=newx)


def test_arima_1123():
    # test ARMAX predict when trend is none
    np.random.seed(12345)
    arparams = np.array([.75, -.25])
    maparams = np.array([.65, .35])

    arparam = np.r_[1, -arparams]
    maparam = np.r_[1, maparams]

    nobs = 20

    dates = dates_from_range('1980',length=nobs)

    y = arma_generate_sample(arparams, maparams, nobs)

    X = np.random.randn(nobs)
    y += 5*X
    mod = ARMA(y[:-1], order=(1,0), exog=X[:-1])
    res = mod.fit(trend='nc', disp=False)
    fc = res.forecast(exog=X[-1:])
    # results from gretl
    assert_almost_equal(fc[0], 2.200393, 6)
    assert_almost_equal(fc[1], 1.030743, 6)
    assert_almost_equal(fc[2][0,0], 0.180175, 6)
    assert_almost_equal(fc[2][0,1], 4.220611, 6)

    mod = ARMA(y[:-1], order=(1,1), exog=X[:-1])
    res = mod.fit(trend='nc', disp=False)
    fc = res.forecast(exog=X[-1:])
    assert_almost_equal(fc[0], 2.765688, 6)
    assert_almost_equal(fc[1], 0.835048, 6)
    assert_almost_equal(fc[2][0,0], 1.129023, 6)
    assert_almost_equal(fc[2][0,1], 4.402353, 6)

    # make sure this works to. code looked fishy.
    mod = ARMA(y[:-1], order=(1,0), exog=X[:-1])
    res = mod.fit(trend='c', disp=False)
    fc = res.forecast(exog=X[-1:])
    assert_almost_equal(fc[0], 2.481219, 6)
    assert_almost_equal(fc[1], 0.968759, 6)
    assert_almost_equal(fc[2][0], [0.582485, 4.379952], 6)


def test_small_data():
    # 1146
    y = [-1214.360173, -1848.209905, -2100.918158, -3647.483678, -4711.186773]

    # refuse to estimate these
    assert_raises(ValueError, ARIMA, y, (2, 0, 3))
    assert_raises(ValueError, ARIMA, y, (1, 1, 3))
    mod = ARIMA(y, (1, 0, 3))
    assert_raises(ValueError, mod.fit, trend="c")

    # try to estimate these...leave it up to the user to check for garbage
    # and be clear, these are garbage parameters.
    # X-12 arima will estimate, gretl refuses to estimate likely a problem
    # in start params regression.
    res = mod.fit(trend="nc", disp=0, start_params=[.1,.1,.1,.1])
    mod = ARIMA(y, (1, 0, 2))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = mod.fit(disp=0, start_params=[np.mean(y), .1, .1, .1])


class TestARMA00(TestCase):

    @classmethod
    def setup_class(cls):
        from statsmodels.datasets.sunspots import load

        sunspots = load().data['SUNACTIVITY']
        cls.y = y = sunspots
        cls.arma_00_model = ARMA(y, order=(0, 0))
        cls.arma_00_res = cls.arma_00_model.fit(disp=-1)

    def test_parameters(self):
        params = self.arma_00_res.params
        assert_almost_equal(self.y.mean(), params)

    def test_predictions(self):
        predictions = self.arma_00_res.predict()
        assert_almost_equal(self.y.mean() * np.ones_like(predictions), predictions)

    @nottest
    def test_information_criteria(self):
        # This test is invalid since the ICs differ due to df_model differences
        # between OLS and ARIMA
        res = self.arma_00_res
        y = self.y
        ols_res = OLS(y, np.ones_like(y)).fit(disp=-1)
        ols_ic = np.array([ols_res.aic, ols_res.bic])
        arma_ic = np.array([res.aic, res.bic])
        assert_almost_equal(ols_ic, arma_ic, DECIMAL_4)

    def test_arma_00_nc(self):
        arma_00 = ARMA(self.y, order=(0, 0))
        assert_raises(ValueError, arma_00.fit, trend='nc', disp=-1)

    def test_css(self):
        arma = ARMA(self.y, order=(0, 0))
        fit = arma.fit(method='css', disp=-1)
        predictions = fit.predict()
        assert_almost_equal(self.y.mean() * np.ones_like(predictions), predictions)

    def test_arima(self):
        yi = np.cumsum(self.y)
        arima = ARIMA(yi, order=(0, 1, 0))
        fit = arima.fit(disp=-1)
        assert_almost_equal(np.diff(yi).mean(), fit.params, DECIMAL_4)

    def test_arma_ols(self):
        y = self.y
        y_lead = y[1:]
        y_lag = y[:-1]
        T = y_lag.shape[0]
        X = np.hstack((np.ones((T,1)), y_lag[:,None]))
        ols_res = OLS(y_lead, X).fit()
        arma_res = ARMA(y_lead,order=(0,0),exog=y_lag).fit(trend='c', disp=-1)
        assert_almost_equal(ols_res.params, arma_res.params)

    def test_arma_exog_no_constant(self):
        y = self.y
        y_lead = y[1:]
        y_lag = y[:-1]
        X = y_lag[:,None]
        ols_res = OLS(y_lead, X).fit()
        arma_res = ARMA(y_lead,order=(0,0),exog=y_lag).fit(trend='nc', disp=-1)
        assert_almost_equal(ols_res.params, arma_res.params)
        pass


def test_arima_dates_startatend():
    # bug
    np.random.seed(18)
    x = pd.Series(np.random.random(36),
                  index=pd.DatetimeIndex(start='1/1/1990',
                                         periods=36, freq='M'))
    res = ARIMA(x, (1, 0, 0)).fit(disp=0)
    pred = res.predict(start=len(x), end=len(x))
    assert_(pred.index[0] == x.index.shift(1)[-1])
    fc = res.forecast()[0]
    assert_almost_equal(pred.values[0], fc)


def test_arma_missing():
    from statsmodels.tools.sm_exceptions import MissingDataError
    # bug 1343
    y = np.random.random(40)
    y[-1] = np.nan
    assert_raises(MissingDataError, ARMA, y, (1, 0), missing='raise')


@dec.skipif(not have_matplotlib)
def test_plot_predict():
    from statsmodels.datasets.sunspots import load_pandas

    dta = load_pandas().data[['SUNACTIVITY']]
    dta.index = DatetimeIndex(start='1700', end='2009', freq='A')
    res = ARMA(dta, (3, 0)).fit(disp=-1)
    fig = res.plot_predict('1990', '2012', dynamic=True, plot_insample=False)
    plt.close(fig)

    res = ARIMA(dta, (3, 1, 0)).fit(disp=-1)
    fig = res.plot_predict('1990', '2012', dynamic=True, plot_insample=False)
    plt.close(fig)



def test_arima_diff2():
    dta = load_macrodata_pandas().data['cpi']
    dates = dates_from_range("1959Q1", length=len(dta))
    dta.index = cpi_dates
    mod = ARIMA(dta, (3, 2, 1)).fit(disp=-1)
    fc, fcerr, conf_int = mod.forecast(10)
    # forecasts from gretl
    conf_int_res = [ (216.139,  219.231),
                     (216.472,  221.520),
                     (217.064,  223.649),
                     (217.586,  225.727),
                     (218.119,  227.770),
                     (218.703,  229.784),
                     (219.306,  231.777),
                     (219.924,  233.759),
                     (220.559,  235.735),
                     (221.206,  237.709)]


    fc_res = [217.685, 218.996, 220.356, 221.656, 222.945, 224.243, 225.541,
          226.841, 228.147, 229.457]
    fcerr_res = [0.7888, 1.2878, 1.6798, 2.0768,  2.4620, 2.8269, 3.1816,
                 3.52950, 3.8715, 4.2099]

    assert_almost_equal(fc, fc_res, 3)
    assert_almost_equal(fcerr, fcerr_res, 3)
    assert_almost_equal(conf_int, conf_int_res, 3)

    predicted = mod.predict('2008Q1', '2012Q1', typ='levels')

    predicted_res = [214.464, 215.478, 221.277, 217.453, 212.419, 213.530,
                     215.087, 217.685 , 218.996 , 220.356 , 221.656 ,
                     222.945 , 224.243 , 225.541 , 226.841 , 228.147 ,
                     229.457]
    assert_almost_equal(predicted, predicted_res, 3)


def test_arima111_predict_exog_2127():
    # regression test for issue #2127
    ef =  [ 0.03005,  0.03917,  0.02828,  0.03644,  0.03379,  0.02744,
            0.03343,  0.02621,  0.0305 ,  0.02455,  0.03261,  0.03507,
            0.02734,  0.05373,  0.02677,  0.03443,  0.03331,  0.02741,
            0.03709,  0.02113,  0.03343,  0.02011,  0.03675,  0.03077,
            0.02201,  0.04844,  0.05518,  0.03765,  0.05433,  0.03049,
            0.04829,  0.02936,  0.04421,  0.02457,  0.04007,  0.03009,
            0.04504,  0.05041,  0.03651,  0.02719,  0.04383,  0.02887,
            0.0344 ,  0.03348,  0.02364,  0.03496,  0.02549,  0.03284,
            0.03523,  0.02579,  0.0308 ,  0.01784,  0.03237,  0.02078,
            0.03508,  0.03062,  0.02006,  0.02341,  0.02223,  0.03145,
            0.03081,  0.0252 ,  0.02683,  0.0172 ,  0.02225,  0.01579,
            0.02237,  0.02295,  0.0183 ,  0.02356,  0.02051,  0.02932,
            0.03025,  0.0239 ,  0.02635,  0.01863,  0.02994,  0.01762,
            0.02837,  0.02421,  0.01951,  0.02149,  0.02079,  0.02528,
            0.02575,  0.01634,  0.02563,  0.01719,  0.02915,  0.01724,
            0.02804,  0.0275 ,  0.02099,  0.02522,  0.02422,  0.03254,
            0.02095,  0.03241,  0.01867,  0.03998,  0.02212,  0.03034,
            0.03419,  0.01866,  0.02623,  0.02052]
    ue =  [  4.9,   5. ,   5. ,   5. ,   4.9,   4.7,   4.8,   4.7,   4.7,
             4.6,   4.6,   4.7,   4.7,   4.5,   4.4,   4.5,   4.4,   4.6,
             4.5,   4.4,   4.5,   4.4,   4.6,   4.7,   4.6,   4.7,   4.7,
             4.7,   5. ,   5. ,   4.9,   5.1,   5. ,   5.4,   5.6,   5.8,
             6.1,   6.1,   6.5,   6.8,   7.3,   7.8,   8.3,   8.7,   9. ,
             9.4,   9.5,   9.5,   9.6,   9.8,  10. ,   9.9,   9.9,   9.7,
             9.8,   9.9,   9.9,   9.6,   9.4,   9.5,   9.5,   9.5,   9.5,
             9.8,   9.4,   9.1,   9. ,   9. ,   9.1,   9. ,   9.1,   9. ,
             9. ,   9. ,   8.8,   8.6,   8.5,   8.2,   8.3,   8.2,   8.2,
             8.2,   8.2,   8.2,   8.1,   7.8,   7.8,   7.8,   7.9,   7.9,
             7.7,   7.5,   7.5,   7.5,   7.5,   7.3,   7.2,   7.2,   7.2,
             7. ,   6.7,   6.6,   6.7,   6.7,   6.3,   6.3]

    # rescaling results in convergence failure
    #model = sm.tsa.ARIMA(np.array(ef)*100, (1,1,1), exog=ue)
    model = ARIMA(ef, (1,1,1), exog=ue)
    res = model.fit(transparams=False, iprint=0, disp=0)

    predicts = res.predict(start=len(ef), end = len(ef)+10,
                           exog=ue[-11:], typ = 'levels')

    # regression test, not verified numbers
    # if exog=ue in predict, which values are used ?
    predicts_res = np.array(
          [ 0.02612291,  0.02361929,  0.024966  ,  0.02448193,  0.0248772 ,
            0.0248762 ,  0.02506319,  0.02516542,  0.02531214,  0.02544654,
            0.02559099,  0.02550931])

    # if exog=ue[-11:] in predict
    predicts_res = np.array(
          [ 0.02591112,  0.02321336,  0.02436593,  0.02368773,  0.02389767,
            0.02372018,  0.02374833,  0.02367407,  0.0236443 ,  0.02362868,
            0.02362312])

    assert_allclose(predicts, predicts_res, atol=1e-6)


def test_ARIMA_exog_predict():
    # test forecasting and dynamic prediction with exog against Stata

    dta = load_macrodata_pandas().data
    dates = dates_from_range("1959Q1", length=len(dta))
    cpi_dates = dates_from_range('1959Q1', '2009Q3')
    dta.index = cpi_dates

    data = dta
    data['loginv'] = np.log(data['realinv'])
    data['loggdp'] = np.log(data['realgdp'])
    data['logcons'] = np.log(data['realcons'])

    forecast_period = dates_from_range('2008Q2', '2009Q3')
    end = forecast_period[0]
    data_sample = data.ix[dta.index < end]

    exog_full = data[['loggdp', 'logcons']]

    # pandas

    mod = ARIMA(data_sample['loginv'], (1,0,1), exog=data_sample[['loggdp', 'logcons']])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = mod.fit(disp=0, solver='bfgs', maxiter=5000)

    predicted_arma_fp = res.predict(start=197, end=202, exog=exog_full.values[197:]).values
    predicted_arma_dp = res.predict(start=193, end=202, exog=exog_full[197:], dynamic=True)


    # numpy
    mod2 = ARIMA(np.asarray(data_sample['loginv']), (1,0,1),
                   exog=np.asarray(data_sample[['loggdp', 'logcons']]))
    res2 = mod2.fit(start_params=res.params, disp=0, solver='bfgs', maxiter=5000)

    exog_full = data[['loggdp', 'logcons']]
    predicted_arma_f = res2.predict(start=197, end=202, exog=exog_full.values[197:])
    predicted_arma_d = res2.predict(start=193, end=202, exog=exog_full[197:], dynamic=True)

    #ARIMA(1, 1, 1)
    ex = np.asarray(data_sample[['loggdp', 'logcons']].diff())
    # The first obsevation is not (supposed to be) used, but I get a Lapack problem
    # Intel MKL ERROR: Parameter 5 was incorrect on entry to DLASCL.
    ex[0] = 0
    mod111 = ARIMA(np.asarray(data_sample['loginv']), (1,1,1),
                       # Stata differences also the exog
                       exog=ex)

    res111 = mod111.fit(disp=0, solver='bfgs', maxiter=5000)
    exog_full_d = data[['loggdp', 'logcons']].diff()
    res111.predict(start=197, end=202, exog=exog_full_d.values[197:])

    predicted_arima_f = res111.predict(start=196, end=202, exog=exog_full_d.values[197:], typ='levels')
    predicted_arima_d = res111.predict(start=193, end=202, exog=exog_full_d.values[197:], typ='levels', dynamic=True)

    res_f101 = np.array([ 7.73975859954,  7.71660108543,  7.69808978329,  7.70872117504,
             7.6518392758 ,  7.69784279784,  7.70290907856,  7.69237782644,
             7.65017785174,  7.66061689028,  7.65980022857,  7.61505314129,
             7.51697158428,  7.5165760663 ,  7.5271053284 ])
    res_f111 = np.array([ 7.74460013693,  7.71958207517,  7.69629561172,  7.71208186737,
             7.65758850178,  7.69223472572,  7.70411775588,  7.68896109499,
             7.64016249001,  7.64871881901,  7.62550283402,  7.55814609462,
             7.44431310053,  7.42963968062,  7.43554675427])
    res_d111 = np.array([ 7.74460013693,  7.71958207517,  7.69629561172,  7.71208186737,
             7.65758850178,  7.69223472572,  7.71870821151,  7.7299430215 ,
             7.71439447355,  7.72544001101,  7.70521902623,  7.64020040524,
             7.5281927191 ,  7.5149442694 ,  7.52196378005])
    res_d101 = np.array([ 7.73975859954,  7.71660108543,  7.69808978329,  7.70872117504,
             7.6518392758 ,  7.69784279784,  7.72522142662,  7.73962377858,
             7.73245950636,  7.74935432862,  7.74449584691,  7.69589103679,
             7.5941274688 ,  7.59021764836,  7.59739267775])

    assert_allclose(predicted_arma_dp, res_d101[-len(predicted_arma_d):], atol=1e-4)
    assert_allclose(predicted_arma_fp, res_f101[-len(predicted_arma_f):], atol=1e-4)
    assert_allclose(predicted_arma_d, res_d101[-len(predicted_arma_d):], atol=1e-4)
    assert_allclose(predicted_arma_f, res_f101[-len(predicted_arma_f):], atol=1e-4)
    assert_allclose(predicted_arima_d, res_d111[-len(predicted_arima_d):], rtol=1e-4, atol=1e-4)
    assert_allclose(predicted_arima_f, res_f111[-len(predicted_arima_f):], rtol=1e-4, atol=1e-4)


    # test for forecast with 0 ar fix in #2457 numbers again from Stata

    res_f002 = np.array([ 7.70178181209,  7.67445481224,  7.6715373765 ,  7.6772915319 ,
         7.61173201163,  7.67913499878,  7.6727609212 ,  7.66275451925,
         7.65199799315,  7.65149983741,  7.65554131408,  7.62213286298,
         7.53795983357,  7.53626130154,  7.54539963934])
    res_d002 = np.array([ 7.70178181209,  7.67445481224,  7.6715373765 ,  7.6772915319 ,
         7.61173201163,  7.67913499878,  7.67306697759,  7.65287924998,
         7.64904451605,  7.66580449603,  7.66252081172,  7.62213286298,
         7.53795983357,  7.53626130154,  7.54539963934])


    mod_002 = ARIMA(np.asarray(data_sample['loginv']), (0,0,2),
                   exog=np.asarray(data_sample[['loggdp', 'logcons']]))

    # doesn't converge with default starting values
    res_002 = mod_002.fit(start_params=np.concatenate((res.params[[0, 1, 2, 4]], [0])),
                          disp=0, solver='bfgs', maxiter=5000)

    # forecast
    fpredict_002 = res_002.predict(start=197, end=202, exog=exog_full.values[197:])
    forecast_002 = res_002.forecast(steps=len(exog_full.values[197:]),
                                    exog=exog_full.values[197:])
    forecast_002 = forecast_002[0]  # TODO we are not checking the other results
    assert_allclose(fpredict_002, res_f002[-len(fpredict_002):], rtol=1e-4, atol=1e-6)
    assert_allclose(forecast_002, res_f002[-len(forecast_002):], rtol=1e-4, atol=1e-6)

    # dynamic predict
    dpredict_002 = res_002.predict(start=193, end=202, exog=exog_full.values[197:],
                                   dynamic=True)
    assert_allclose(dpredict_002, res_d002[-len(dpredict_002):], rtol=1e-4, atol=1e-6)


def test_arima_fit_mutliple_calls():
    y = [-1214.360173, -1848.209905, -2100.918158, -3647.483678, -4711.186773]
    mod = ARIMA(y, (1, 0, 2))
    # Make multiple calls to fit
    with warnings.catch_warnings(record=True) as w:
        mod.fit(disp=0, start_params=[np.mean(y), .1, .1, .1])
    assert_equal(mod.exog_names,  ['const', 'ar.L1.y', 'ma.L1.y', 'ma.L2.y'])
    with warnings.catch_warnings(record=True) as w:
        mod.fit(disp=0, start_params=[np.mean(y), .1, .1, .1])
    assert_equal(mod.exog_names,  ['const', 'ar.L1.y', 'ma.L1.y', 'ma.L2.y'])

def test_long_ar_start_params():
    np.random.seed(12345)
    arparams = np.array([1, -.75, .25])
    maparams = np.array([1, .65, .35])

    nobs = 30

    y = arma_generate_sample(arparams, maparams, nobs)

    model = ARMA(y, order=(2, 2))

    res = model.fit(method='css',start_ar_lags=10, disp=0)
    res = model.fit(method='css-mle',start_ar_lags=10, disp=0)
    res = model.fit(method='mle',start_ar_lags=10, disp=0)
    assert_raises(ValueError, model.fit, start_ar_lags=nobs+5, disp=0)


def test_arma_pickle():
    np.random.seed(9876565)
    x = fa.ArmaFft([1, -0.5], [1., 0.4], 40).generate_sample(nsample=200,
                                                             burnin=1000)
    mod = ARMA(x, (1, 1))
    pkl_mod = cPickle.loads(cPickle.dumps(mod))

    res = mod.fit(trend="c", disp=-1)
    pkl_res = pkl_mod.fit(trend="c", disp=-1)

    assert_allclose(res.params, pkl_res.params)
    assert_allclose(res.llf, pkl_res.llf)
    assert_almost_equal(res.resid, pkl_res.resid)
    assert_almost_equal(res.fittedvalues, pkl_res.fittedvalues)
    assert_almost_equal(res.pvalues, pkl_res.pvalues)


def test_arima_pickle():
    endog = y_arma[:, 6]
    mod = ARIMA(endog, (1, 0, 1))
    pkl_mod = cPickle.loads(cPickle.dumps(mod))

    res = mod.fit(trend="c", disp=-1)
    pkl_res = pkl_mod.fit(trend="c", disp=-1)

    assert_allclose(res.params, pkl_res.params)
    assert_allclose(res.llf, pkl_res.llf)
    assert_almost_equal(res.resid, pkl_res.resid)
    assert_almost_equal(res.fittedvalues, pkl_res.fittedvalues)
    assert_almost_equal(res.pvalues, pkl_res.pvalues)


if __name__ == "__main__":
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb'], exit=False)
