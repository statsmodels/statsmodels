import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_
import statsmodels.sandbox.tsa.fftarma as fa
from statsmodels.tsa.descriptivestats import TsaDescriptive
from statsmodels.tsa.arma_mle import Arma
from statsmodels.tsa.arima_model import ARMA, ARIMA
from results import results_arma, results_arima
import os
from statsmodels.tsa.base import datetools
import pandas
try:
    from statsmodels.tsa.kalmanf import kalman_loglike
    fast_kalman = 1
except:
    fast_kalman = 0
    #NOTE: the KF with complex input returns a different precision for
    # the hessian imaginary part, so we use approx_hess and the the
    # resulting stats are slightly different.

DECIMAL_4 = 4
DECIMAL_3 = 3
DECIMAL_2 = 2
DECIMAL_1 = 1

current_path = os.path.dirname(os.path.abspath(__file__))
y_arma = np.genfromtxt(open(current_path + '/results/y_arma_data.csv', "rb"),
        delimiter=",", skip_header=1, dtype=float)


def test_compare_arma():
    #this is a preliminary test to compare arma_kf, arma_cond_ls and arma_cond_mle
    #the results returned by the fit methods are incomplete
    #for now without random.seed

    #np.random.seed(9876565)
    x = fa.ArmaFft([1, -0.5], [1., 0.4], 40).generate_sample(size=200,
            burnin=1000)

# this used kalman filter through descriptive
#    d = ARMA(x)
#    d.fit((1,1), trend='nc')
#    dres = d.res

    modkf = ARMA(x)
    ##rkf = mkf.fit((1,1))
    ##rkf.params
    reskf = modkf.fit((1,1), trend='nc', disp=-1)
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
    assert_almost_equal(rescm.params[:-1] / dres.params, np.ones(dres.params.shape), decimal=1)
    #return resls[0], d.params, rescm.params


class CheckArmaResults(object):
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

    def test_arroots(self):
        assert_almost_equal(self.res1.arroots, self.res2.arroots, DECIMAL_4)

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
        assert_almost_equal(self.res1.tvalues, self.res2.tvalues, self.decimal_t)

    decimal_sigma2 = DECIMAL_4
    def test_sigma2(self):
        assert_almost_equal(self.res1.sigma2, self.res2.sigma2,
                self.decimal_sigma2)

class CheckForecast(object):
    def test_forecast(self):
        assert_almost_equal(self.res1.forecast_res, self.res2.forecast,
                DECIMAL_4)

    def test_forecasterr(self):
        assert_almost_equal(self.res1.forecast_err, self.res2.forecasterr,
                DECIMAL_4)

class CheckArimaResults(CheckArmaResults):
    def test_order(self):
        assert self.res1.k_diff == self.res2.k_diff
        assert self.res1.k_ar == self.res2.k_ar
        assert self.res1.k_ma == self.res2.k_ma

    def test_predict_levels(self):
        assert_almost_equal(self.res1.predict(typ='levels'), self.res2.linear)

    def test_forecast(self):
        fc, fcerr, conf_int = self.res1.forecast(20)
        assert_almost_equal(fc, self.res2.forecast)
        assert_almost_equal(fcerr, self.res2.fcerr)
        assert_almost_equal(conf_int[:,0], self.res2.fc_conf_int[:,0])
        assert_almost_equal(conf_int[:,1], self.res2.fc_conf_int[:,1])

#NOTE: Ok
class Test_Y_ARMA11_NoConst(CheckArmaResults, CheckForecast):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,0]
        cls.res1 = ARMA(endog).fit(order=(1,1), trend='nc', disp=-1)
        (cls.res1.forecast_res, cls.res1.forecast_err,
                confint) = cls.res1.forecast(10)
        cls.res2 = results_arma.Y_arma11()

    def test_pickle(self):
        from statsmodels.compatnp.py3k import BytesIO
        fh = BytesIO()
        #test wrapped results load save pickle
        self.res1.save(fh)
        fh.seek(0,0)
        res_unpickled = self.res1.__class__.load(fh)
        assert_(type(res_unpickled) is type(self.res1))

#NOTE: Ok
class Test_Y_ARMA14_NoConst(CheckArmaResults):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,1]
        cls.res1 = ARMA(endog).fit(order=(1,4), trend='nc', disp=-1)
        cls.res2 = results_arma.Y_arma14()
        if fast_kalman:
            cls.decimal_t = 0


#NOTE: Ok
class Test_Y_ARMA41_NoConst(CheckArmaResults, CheckForecast):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,2]
        cls.res1 = ARMA(endog).fit(order=(4,1), trend='nc', disp=-1)
        (cls.res1.forecast_res, cls.res1.forecast_err,
                confint) = cls.res1.forecast(10)
        cls.res2 = results_arma.Y_arma41()
        cls.decimal_maroots = DECIMAL_3

#NOTE: Ok
class Test_Y_ARMA22_NoConst(CheckArmaResults):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,3]
        cls.res1 = ARMA(endog).fit(order=(2,2), trend='nc', disp=-1)
        cls.res2 = results_arma.Y_arma22()
        if fast_kalman:
            cls.decimal_t -= 1

#NOTE: Ok
class Test_Y_ARMA50_NoConst(CheckArmaResults, CheckForecast):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,4]
        cls.res1 = ARMA(endog).fit(order=(5,0), trend='nc', disp=-1)
        (cls.res1.forecast_res, cls.res1.forecast_err,
                confint) = cls.res1.forecast(10)
        cls.res2 = results_arma.Y_arma50()

#NOTE: Ok
class Test_Y_ARMA02_NoConst(CheckArmaResults):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,5]
        cls.res1 = ARMA(endog).fit(order=(0,2), trend='nc', disp=-1)
        cls.res2 = results_arma.Y_arma02()
        if fast_kalman:
            cls.decimal_t -= 1

#NOTE: Ok
class Test_Y_ARMA11_Const(CheckArmaResults, CheckForecast):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,6]
        cls.res1 = ARMA(endog).fit(order=(1,1), trend="c", disp=-1)
        (cls.res1.forecast_res, cls.res1.forecast_err,
                confint) = cls.res1.forecast(10)
        cls.res2 = results_arma.Y_arma11c()

#NOTE: OK
class Test_Y_ARMA14_Const(CheckArmaResults):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,7]
        cls.res1 = ARMA(endog).fit(order=(1,4), trend="c", disp=-1)
        cls.res2 = results_arma.Y_arma14c()
        if fast_kalman:
            cls.decimal_t = 0
            cls.decimal_cov_params -= 1

#NOTE: Ok
class Test_Y_ARMA41_Const(CheckArmaResults, CheckForecast):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,8]
        cls.res1 = ARMA(endog).fit(order=(4,1), trend="c", disp=-1)
        (cls.res1.forecast_res, cls.res1.forecast_err,
                confint) = cls.res1.forecast(10)
        cls.res2 = results_arma.Y_arma41c()
        cls.decimal_cov_params = DECIMAL_3
        cls.decimal_fittedvalues = DECIMAL_3
        cls.decimal_resid = DECIMAL_3
        cls.decimal_params = DECIMAL_3
        if fast_kalman:
            cls.decimal_cov_params -= 2
            cls.decimal_bse -= 1

#NOTE: Ok
class Test_Y_ARMA22_Const(CheckArmaResults):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,9]
        cls.res1 = ARMA(endog).fit(order=(2,2), trend="c", disp=-1)
        cls.res2 = results_arma.Y_arma22c()
        if fast_kalman:
            cls.decimal_t = 0

#NOTE: Ok
class Test_Y_ARMA50_Const(CheckArmaResults, CheckForecast):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,10]
        cls.res1 = ARMA(endog).fit(order=(5,0), trend="c", disp=-1)
        (cls.res1.forecast_res, cls.res1.forecast_err,
                confint) = cls.res1.forecast(10)
        cls.res2 = results_arma.Y_arma50c()

#NOTE: Ok
class Test_Y_ARMA02_Const(CheckArmaResults):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,11]
        cls.res1 = ARMA(endog).fit(order=(0,2), trend="c", disp=-1)
        cls.res2 = results_arma.Y_arma02c()
        if fast_kalman:
            cls.decimal_t -= 1

#NOTE:
# cov_params and tvalues are off still but not as much vs. R
class Test_Y_ARMA11_NoConst_CSS(CheckArmaResults):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,0]
        cls.res1 = ARMA(endog).fit(order=(1,1), method="css", trend='nc',
                            disp=-1)
        cls.res2 = results_arma.Y_arma11("css")
        cls.decimal_t = DECIMAL_1

# better vs. R
class Test_Y_ARMA14_NoConst_CSS(CheckArmaResults):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,1]
        cls.res1 = ARMA(endog).fit(order=(1,4), method="css", trend='nc',
                            disp=-1)
        cls.res2 = results_arma.Y_arma14("css")
        cls.decimal_fittedvalues = DECIMAL_3
        cls.decimal_resid = DECIMAL_3
        cls.decimal_t = DECIMAL_1

#NOTE: Ok
#NOTE:
# bse, etc. better vs. R
# maroot is off because maparams is off a bit (adjust tolerance?)
class Test_Y_ARMA41_NoConst_CSS(CheckArmaResults):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,2]
        cls.res1 = ARMA(endog).fit(order=(4,1), method="css", trend='nc',
                        disp=-1)
        cls.res2 = results_arma.Y_arma41("css")
        cls.decimal_t = DECIMAL_1
        cls.decimal_pvalues = 0
        cls.decimal_cov_params = DECIMAL_3
        cls.decimal_maroots = DECIMAL_1

#NOTE: Ok
#same notes as above
class Test_Y_ARMA22_NoConst_CSS(CheckArmaResults):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,3]
        cls.res1 = ARMA(endog).fit(order=(2,2), method="css", trend='nc',
                            disp=-1)
        cls.res2 = results_arma.Y_arma22("css")
        cls.decimal_t = DECIMAL_1
        cls.decimal_resid = DECIMAL_3
        cls.decimal_pvalues = DECIMAL_1
        cls.decimal_fittedvalues = DECIMAL_3

#NOTE: Ok
#NOTE: gretl just uses least squares for AR CSS
# so BIC, etc. is
# -2*res1.llf + np.log(nobs)*(res1.q+res1.p+res1.k)
# with no adjustment for p and no extra sigma estimate
#NOTE: so our tests use x-12 arima results which agree with us and are
# consistent with the rest of the models
class Test_Y_ARMA50_NoConst_CSS(CheckArmaResults):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,4]
        cls.res1 = ARMA(endog).fit(order=(5,0), method="css", trend='nc',
                            disp=-1)
        cls.res2 = results_arma.Y_arma50("css")
        cls.decimal_t = 0
        cls.decimal_llf = DECIMAL_1 # looks like rounding error?

#NOTE: ok
class Test_Y_ARMA02_NoConst_CSS(CheckArmaResults):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,5]
        cls.res1 = ARMA(endog).fit(order=(0,2), method="css", trend='nc',
                            disp=-1)
        cls.res2 = results_arma.Y_arma02("css")

#NOTE: Ok
#NOTE: our results are close to --x-12-arima option and R
class Test_Y_ARMA11_Const_CSS(CheckArmaResults):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,6]
        cls.res1 = ARMA(endog).fit(order=(1,1), trend="c", method="css",
                        disp=-1)
        cls.res2 = results_arma.Y_arma11c("css")
        cls.decimal_params = DECIMAL_3
        cls.decimal_cov_params = DECIMAL_3
        cls.decimal_t = DECIMAL_1

#NOTE: Ok
class Test_Y_ARMA14_Const_CSS(CheckArmaResults):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,7]
        cls.res1 = ARMA(endog).fit(order=(1,4), trend="c", method="css",
                        disp=-1)
        cls.res2 = results_arma.Y_arma14c("css")
        cls.decimal_t = DECIMAL_1
        cls.decimal_pvalues = DECIMAL_1

#NOTE: Ok
class Test_Y_ARMA41_Const_CSS(CheckArmaResults):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,8]
        cls.res1 = ARMA(endog).fit(order=(4,1), trend="c", method="css",
                        disp=-1)
        cls.res2 = results_arma.Y_arma41c("css")
        cls.decimal_t = DECIMAL_1
        cls.decimal_cov_params = DECIMAL_1
        cls.decimal_maroots = DECIMAL_3
        cls.decimal_bse = DECIMAL_1

#NOTE: Ok
class Test_Y_ARMA22_Const_CSS(CheckArmaResults):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,9]
        cls.res1 = ARMA(endog).fit(order=(2,2), trend="c", method="css",
                        disp=-1)
        cls.res2 = results_arma.Y_arma22c("css")
        cls.decimal_t = 0
        cls.decimal_pvalues = DECIMAL_1

#NOTE: Ok
class Test_Y_ARMA50_Const_CSS(CheckArmaResults):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,10]
        cls.res1 = ARMA(endog).fit(order=(5,0), trend="c", method="css",
                        disp=-1)
        cls.res2 = results_arma.Y_arma50c("css")
        cls.decimal_t = DECIMAL_1
        cls.decimal_params = DECIMAL_3
        cls.decimal_cov_params = DECIMAL_2

#NOTE: Ok
class Test_Y_ARMA02_Const_CSS(CheckArmaResults):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,11]
        cls.res1 = ARMA(endog).fit(order=(0,2), trend="c", method="css",
                        disp=-1)
        cls.res2 = results_arma.Y_arma02c("css")

def test_reset_trend():
    endog = y_arma[:,0]
    mod = ARMA(endog)
    res1 = mod.fit(order=(1,1), trend="c", disp=-1)
    res2 = mod.fit(order=(1,1), trend="nc", disp=-1)
    assert_equal(len(res1.params), len(res2.params)+1)

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
    res = ARMA(data).fit(order=(4,1), disp=-1)

class Test_ARIMA101(CheckArmaResults):
    # just make sure this works
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,6]
        cls.res1 = ARIMA(endog).fit(order=(1,0,1), trend="c", disp=-1)
        (cls.res1.forecast_res, cls.res1.forecast_err,
                confint) = cls.res1.forecast(10)
        cls.res2 = results_arma.Y_arma11c()
        cls.res2.k_diff = 0
        cls.res2.k_ar = 1
        cls.res2.k_ma = 1

class Test_ARIMA111(CheckArmaResults):
    @classmethod
    def setupClass(cls):
        from statsmodels.datasets.macrodata import load

        cpi = load().data['cpi']
        cls.res1 = ARIMA(cpi).fit(order=(1,1,1), disp=-1)
        cls.res2 = results_arima.ARIMA111()
        # make sure endog names changes to D.cpi
        cls.decimal_llf = 3
        cls.decimal_aic = 3
        cls.decimal_bic = 3


if __name__ == "__main__":
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb'], exit=False)
