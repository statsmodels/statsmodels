import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import scikits.statsmodels.sandbox.tsa.fftarma as fa
from scikits.statsmodels.tsa.descriptivestats import TsaDescriptive
from scikits.statsmodels.tsa.arma_mle import Arma
from scikits.statsmodels.tsa.arima_model import ARMA
from results import results_arma
import os
try:
    from scikits.statsmodels.tsa.kalmanf import kalman_loglike
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

#NOTE: Ok
class Test_Y_ARMA11_NoConst(CheckArmaResults, CheckForecast):
    @classmethod
    def setupClass(cls):
        endog = y_arma[:,0]
        cls.res1 = ARMA(endog).fit(order=(1,1), trend='nc', disp=-1)
        (cls.res1.forecast_res, cls.res1.forecast_err,
                confint) = cls.res1.forecast(10)
        cls.res2 = results_arma.Y_arma11()

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


if __name__ == "__main__":
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb'], exit=False)
