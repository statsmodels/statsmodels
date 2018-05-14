from statsmodels.compat.numpy import recarray_select

from statsmodels.compat.python import lrange
from statsmodels.tools.sm_exceptions import ColinearityWarning
from statsmodels.tsa.stattools import (adfuller, acf, pacf_ols, pacf_yw,
                                               pacf, grangercausalitytests,
                                               coint, acovf, kpss, ResultsStore,
                                               arma_order_select_ic)
import numpy as np
import pandas as pd
import pytest
from numpy.testing import (assert_almost_equal, assert_equal, assert_warns,
                           assert_raises, assert_, assert_allclose)
from statsmodels.datasets import macrodata, sunspots
from pandas import Series, DatetimeIndex, DataFrame
import os
import warnings
from statsmodels.tools.sm_exceptions import MissingDataError

DECIMAL_8 = 8
DECIMAL_6 = 6
DECIMAL_5 = 5
DECIMAL_4 = 4
DECIMAL_3 = 3
DECIMAL_2 = 2
DECIMAL_1 = 1

class CheckADF(object):
    """
    Test Augmented Dickey-Fuller

    Test values taken from Stata.
    """
    levels = ['1%', '5%', '10%']
    data = macrodata.load()
    x = data.data['realgdp']
    y = data.data['infl']

    def test_teststat(self):
        assert_almost_equal(self.res1[0], self.teststat, DECIMAL_5)

    def test_pvalue(self):
        assert_almost_equal(self.res1[1], self.pvalue, DECIMAL_5)

    def test_critvalues(self):
        critvalues = [self.res1[4][lev] for lev in self.levels]
        assert_almost_equal(critvalues, self.critvalues, DECIMAL_2)

class TestADFConstant(CheckADF):
    """
    Dickey-Fuller test for unit root
    """
    @classmethod
    def setup_class(cls):
        cls.res1 = adfuller(cls.x, regression="c", autolag=None,
                maxlag=4)
        cls.teststat = .97505319
        cls.pvalue = .99399563
        cls.critvalues = [-3.476, -2.883, -2.573]

class TestADFConstantTrend(CheckADF):
    """
    """
    @classmethod
    def setup_class(cls):
        cls.res1 = adfuller(cls.x, regression="ct", autolag=None,
                maxlag=4)
        cls.teststat = -1.8566374
        cls.pvalue = .67682968
        cls.critvalues = [-4.007, -3.437, -3.137]

#class TestADFConstantTrendSquared(CheckADF):
#    """
#    """
#    pass
#TODO: get test values from R?

class TestADFNoConstant(CheckADF):
    """
    """
    @classmethod
    def setup_class(cls):
        cls.res1 = adfuller(cls.x, regression="nc", autolag=None,
                maxlag=4)
        cls.teststat = 3.5227498
        cls.pvalue = .99999 # Stata does not return a p-value for noconstant.
                        # Tau^max in MacKinnon (1994) is missing, so it is
                        # assumed that its right-tail is well-behaved
        cls.critvalues = [-2.587, -1.950, -1.617]

# No Unit Root

class TestADFConstant2(CheckADF):
    @classmethod
    def setup_class(cls):
        cls.res1 = adfuller(cls.y, regression="c", autolag=None,
                maxlag=1)
        cls.teststat = -4.3346988
        cls.pvalue = .00038661
        cls.critvalues = [-3.476, -2.883, -2.573]

class TestADFConstantTrend2(CheckADF):
    @classmethod
    def setup_class(cls):
        cls.res1 = adfuller(cls.y, regression="ct", autolag=None,
                maxlag=1)
        cls.teststat = -4.425093
        cls.pvalue = .00199633
        cls.critvalues = [-4.006, -3.437, -3.137]

class TestADFNoConstant2(CheckADF):
    @classmethod
    def setup_class(cls):
        cls.res1 = adfuller(cls.y, regression="nc", autolag=None,
                maxlag=1)
        cls.teststat = -2.4511596
        cls.pvalue = 0.013747 # Stata does not return a p-value for noconstant
                               # this value is just taken from our results
        cls.critvalues = [-2.587,-1.950,-1.617]
        _, _1, _2, cls.store = adfuller(cls.y, regression="nc", autolag=None,
                                         maxlag=1, store=True)

    def test_store_str(self):
        assert_equal(self.store.__str__(), 'Augmented Dickey-Fuller Test Results')

class CheckCorrGram(object):
    """
    Set up for ACF, PACF tests.
    """
    data = macrodata.load_pandas()
    x = data.data['realgdp']
    filename = os.path.dirname(os.path.abspath(__file__))+\
            "/results/results_corrgram.csv"
    results = pd.read_csv(filename, delimiter=',')

    #not needed: add 1. for lag zero
    #self.results['acvar'] = np.concatenate(([1.], self.results['acvar']))


class TestACF(CheckCorrGram):
    """
    Test Autocorrelation Function
    """
    @classmethod
    def setup_class(cls):
        cls.acf = cls.results['acvar']
        #cls.acf = np.concatenate(([1.], cls.acf))
        cls.qstat = cls.results['Q1']
        cls.res1 = acf(cls.x, nlags=40, qstat=True, alpha=.05)
        cls.confint_res = cls.results[['acvar_lb','acvar_ub']].as_matrix()

    def test_acf(self):
        assert_almost_equal(self.res1[0][1:41], self.acf, DECIMAL_8)

    def test_confint(self):
        centered = self.res1[1] - self.res1[1].mean(1)[:,None]
        assert_almost_equal(centered[1:41], self.confint_res, DECIMAL_8)

    def test_qstat(self):
        assert_almost_equal(self.res1[2][:40], self.qstat, DECIMAL_3)
        # 3 decimal places because of stata rounding

#    def pvalue(self):
#        pass
#NOTE: shouldn't need testing if Q stat is correct


class TestACF_FFT(CheckCorrGram):
    # Test Autocorrelation Function using FFT
    @classmethod
    def setup_class(cls):
        cls.acf = cls.results['acvarfft']
        cls.qstat = cls.results['Q1']
        cls.res1 = acf(cls.x, nlags=40, qstat=True, fft=True)

    def test_acf(self):
        assert_almost_equal(self.res1[0][1:], self.acf, DECIMAL_8)

    def test_qstat(self):
        #todo why is res1/qstat 1 short
        assert_almost_equal(self.res1[1], self.qstat, DECIMAL_3)

class TestACFMissing(CheckCorrGram):
    # Test Autocorrelation Function using Missing
    @classmethod
    def setup_class(cls):
        cls.x = np.concatenate((np.array([np.nan]),cls.x))
        cls.acf = cls.results['acvar'] # drop and conservative
        cls.qstat = cls.results['Q1']
        cls.res_drop = acf(cls.x, nlags=40, qstat=True, alpha=.05,
                            missing='drop')
        cls.res_conservative = acf(cls.x, nlags=40, qstat=True, alpha=.05,
                                    missing='conservative')
        cls.acf_none = np.empty(40) * np.nan # lags 1 to 40 inclusive
        cls.qstat_none = np.empty(40) * np.nan
        cls.res_none = acf(cls.x, nlags=40, qstat=True, alpha=.05,
                        missing='none')

    def test_raise(self):
        assert_raises(MissingDataError, acf, self.x, nlags=40,
                      qstat=True, alpha=.05, missing='raise')

    def test_acf_none(self):
        assert_almost_equal(self.res_none[0][1:41], self.acf_none, DECIMAL_8)

    def test_acf_drop(self):
        assert_almost_equal(self.res_drop[0][1:41], self.acf, DECIMAL_8)

    def test_acf_conservative(self):
        assert_almost_equal(self.res_conservative[0][1:41], self.acf,
                            DECIMAL_8)

    def test_qstat_none(self):
        #todo why is res1/qstat 1 short
        assert_almost_equal(self.res_none[2], self.qstat_none, DECIMAL_3)

# how to do this test? the correct q_stat depends on whether nobs=len(x) is
# used when x contains NaNs or whether nobs<len(x) when x contains NaNs
#    def test_qstat_drop(self):
#        assert_almost_equal(self.res_drop[2][:40], self.qstat, DECIMAL_3)


class TestPACF(CheckCorrGram):
    @classmethod
    def setup_class(cls):
        cls.pacfols = cls.results['PACOLS']
        cls.pacfyw = cls.results['PACYW']

    def test_ols(self):
        pacfols, confint = pacf(self.x, nlags=40, alpha=.05, method="ols")
        assert_almost_equal(pacfols[1:], self.pacfols, DECIMAL_6)
        centered = confint - confint.mean(1)[:,None]
        # from edited Stata ado file
        res = [[-.1375625, .1375625]] * 40
        assert_almost_equal(centered[1:41], res, DECIMAL_6)
        # check lag 0
        assert_equal(centered[0], [0., 0.])
        assert_equal(confint[0], [1, 1])
        assert_equal(pacfols[0], 1)


    def test_yw(self):
        pacfyw = pacf_yw(self.x, nlags=40, method="mle")
        assert_almost_equal(pacfyw[1:], self.pacfyw, DECIMAL_8)

    def test_ld(self):
        pacfyw = pacf_yw(self.x, nlags=40, method="mle")
        pacfld = pacf(self.x, nlags=40, method="ldb")
        assert_almost_equal(pacfyw, pacfld, DECIMAL_8)

        pacfyw = pacf(self.x, nlags=40, method="yw")
        pacfld = pacf(self.x, nlags=40, method="ldu")
        assert_almost_equal(pacfyw, pacfld, DECIMAL_8)

class CheckCoint(object):
    """
    Test Cointegration Test Results for 2-variable system

    Test values taken from Stata
    """
    levels = ['1%', '5%', '10%']
    data = macrodata.load()
    y1 = data.data['realcons']
    y2 = data.data['realgdp']

    def test_tstat(self):
        assert_almost_equal(self.coint_t,self.teststat, DECIMAL_4)

# this doesn't produce the old results anymore
class TestCoint_t(CheckCoint):
    """
    Get AR(1) parameter on residuals
    """
    @classmethod
    def setup_class(cls):
        #cls.coint_t = coint(cls.y1, cls.y2, trend="c")[0]
        cls.coint_t = coint(cls.y1, cls.y2, trend="c", maxlag=0, autolag=None)[0]
        cls.teststat = -1.8208817
        cls.teststat = -1.830170986148


def test_coint():
    nobs = 200
    scale_e = 1
    const = [1, 0, 0.5, 0]
    np.random.seed(123)
    unit = np.random.randn(nobs).cumsum()
    y = scale_e * np.random.randn(nobs, 4)
    y[:, :2] += unit[:, None]
    y += const
    y = np.round(y, 4)

    for trend in []:#['c', 'ct', 'ctt', 'nc']:
        print('\n', trend)
        print(coint(y[:, 0], y[:, 1], trend=trend, maxlag=4, autolag=None))
        print(coint(y[:, 0], y[:, 1:3], trend=trend, maxlag=4, autolag=None))
        print(coint(y[:, 0], y[:, 2:], trend=trend, maxlag=4, autolag=None))
        print(coint(y[:, 0], y[:, 1:], trend=trend, maxlag=4, autolag=None))

    # results from Stata egranger
    res_egranger = {}
    # trend = 'ct'
    res = res_egranger['ct'] = {}
    res[0]  = [-5.615251442239, -4.406102369132,  -3.82866685109, -3.532082997903]
    res[1]  = [-5.63591313706, -4.758609717199, -4.179130554708, -3.880909696863]
    res[2]  = [-2.892029275027, -4.758609717199, -4.179130554708, -3.880909696863]
    res[3]  = [-5.626932544079,  -5.08363327039, -4.502469783057,   -4.2031051091]

    # trend = 'c'
    res = res_egranger['c'] = {}
    # first critical value res[0][1] has a discrepancy starting at 4th decimal
    res[0]  = [-5.760696844656, -3.952043522638, -3.367006313729, -3.065831247948]
    # manually adjusted to have higher precision as in other cases
    res[0][1] = -3.952321293401682
    res[1]  = [-5.781087068772, -4.367111915942, -3.783961136005, -3.483501524709]
    res[2]  = [-2.477444137366, -4.367111915942, -3.783961136005, -3.483501524709]
    res[3]  = [-5.778205811661, -4.735249216434, -4.152738973763, -3.852480848968]

    # trend = 'ctt'
    res = res_egranger['ctt'] = {}
    res[0]  = [-5.644431269946, -4.796038299708, -4.221469431008, -3.926472577178]
    res[1]  = [-5.665691609506, -5.111158174219,  -4.53317278104,  -4.23601008516]
    res[2]  = [-3.161462374828, -5.111158174219,  -4.53317278104,  -4.23601008516]
    res[3]  = [-5.657904558563, -5.406880189412, -4.826111619543, -4.527090164875]

    # The following for 'nc' are only regression test numbers
    # trend = 'nc' not allowed in egranger
    # trend = 'nc'
    res = res_egranger['nc'] = {}
    nan = np.nan  # shortcut for table
    res[0]  = [-3.7146175989071137, nan, nan, nan]
    res[1]  = [-3.8199323012888384, nan, nan, nan]
    res[2]  = [-1.6865000791270679, nan, nan, nan]
    res[3]  = [-3.7991270451873675, nan, nan, nan]

    for trend in ['c', 'ct', 'ctt', 'nc']:
        res1 = {}
        res1[0] = coint(y[:, 0], y[:, 1], trend=trend, maxlag=4, autolag=None)
        res1[1] = coint(y[:, 0], y[:, 1:3], trend=trend, maxlag=4,
                        autolag=None)
        res1[2] = coint(y[:, 0], y[:, 2:], trend=trend, maxlag=4, autolag=None)
        res1[3] = coint(y[:, 0], y[:, 1:], trend=trend, maxlag=4, autolag=None)

        for i in range(4):
            res = res_egranger[trend]

            assert_allclose(res1[i][0], res[i][0], rtol=1e-11)
            r2 = res[i][1:]
            r1 = res1[i][2]
            assert_allclose(r1, r2, rtol=0, atol=6e-7)

    # use default autolag #4490
    res1_0 = coint(y[:, 0], y[:, 1], trend='ct', maxlag=4)
    assert_allclose(res1_0[2], res_egranger['ct'][0][1:], rtol=0, atol=6e-7)
    # the following is just a regression test
    assert_allclose(res1_0[:2], [-13.992946638547112, 2.270898990540678e-27],
                    rtol=1e-10, atol=1e-27)


def test_coint_identical_series():
    nobs = 200
    scale_e = 1
    np.random.seed(123)
    y = scale_e * np.random.randn(nobs)
    warnings.simplefilter('always', ColinearityWarning)
    with warnings.catch_warnings(record=True) as w:
        c = coint(y, y, trend="c", maxlag=0, autolag=None)
    assert_equal(len(w), 1)
    assert_equal(c[1], 0.0)
    assert_(np.isneginf(c[0]))


def test_coint_perfect_collinearity():
    # test uses nearly perfect collinearity
    nobs = 200
    scale_e = 1
    np.random.seed(123)
    x = scale_e * np.random.randn(nobs, 2)
    y = 1 + x.sum(axis=1) + 1e-7 * np.random.randn(nobs)
    warnings.simplefilter('always', ColinearityWarning)
    with warnings.catch_warnings(record=True) as w:
        c = coint(y, x, trend="c", maxlag=0, autolag=None)
    assert_equal(c[1], 0.0)
    assert_(np.isneginf(c[0]))


class TestGrangerCausality(object):

    def test_grangercausality(self):
        # some example data
        mdata = macrodata.load().data
        mdata = recarray_select(mdata, ['realgdp', 'realcons'])
        data = mdata.view((float, 2))
        data = np.diff(np.log(data), axis=0)

        #R: lmtest:grangertest
        r_result = [0.243097, 0.7844328, 195, 2]  # f_test
        gr = grangercausalitytests(data[:, 1::-1], 2, verbose=False)
        assert_almost_equal(r_result, gr[2][0]['ssr_ftest'], decimal=7)
        assert_almost_equal(gr[2][0]['params_ftest'], gr[2][0]['ssr_ftest'], decimal=7)

    def test_granger_fails_on_nobs_check(self):
        # Test that if maxlag is too large, Granger Test raises a clear error.
        X = np.random.rand(10, 2)
        grangercausalitytests(X, 2, verbose=False)  # This should pass.
        assert_raises(ValueError, grangercausalitytests, X, 3, verbose=False)


class SetupKPSS(object):
    data = macrodata.load()
    x = data.data['realgdp']


class TestKPSS(SetupKPSS):
    """
    R-code
    ------
    library(tseries)
    kpss.stat(x, "Level")
    kpss.stat(x, "Trend")

    In this context, x is the vector containing the
    macrodata['realgdp'] series.
    """

    def test_fail_nonvector_input(self):
        with warnings.catch_warnings(record=True) as w:
            kpss(self.x)  # should be fine

        x = np.random.rand(20, 2)
        assert_raises(ValueError, kpss, x)

    def test_fail_unclear_hypothesis(self):
        # these should be fine,
        with warnings.catch_warnings(record=True) as w:
            kpss(self.x, 'c')
            kpss(self.x, 'C')
            kpss(self.x, 'ct')
            kpss(self.x, 'CT')

        assert_raises(ValueError, kpss, self.x, "unclear hypothesis")

    def test_teststat(self):
        with warnings.catch_warnings(record=True) as w:
            kpss_stat, pval, lags, crits = kpss(self.x, 'c', 3)
        assert_almost_equal(kpss_stat, 5.0169, DECIMAL_3)

        with warnings.catch_warnings(record=True) as w:
            kpss_stat, pval, lags, crits = kpss(self.x, 'ct', 3)
        assert_almost_equal(kpss_stat, 1.1828, DECIMAL_3)

    def test_pval(self):
        with warnings.catch_warnings(record=True) as w:
            kpss_stat, pval, lags, crits = kpss(self.x, 'c', 3)
        assert_equal(pval, 0.01)

        with warnings.catch_warnings(record=True) as w:
            kpss_stat, pval, lags, crits = kpss(self.x, 'ct', 3)
        assert_equal(pval, 0.01)

    def test_store(self):
        with warnings.catch_warnings(record=True) as w:
            kpss_stat, pval, crit, store = kpss(self.x, 'c', 3, True)

        # assert attributes, and make sure they're correct
        assert_equal(store.nobs, len(self.x))
        assert_equal(store.lags, 3)

    def test_lags(self):
        with warnings.catch_warnings(record=True) as w:
            kpss_stat, pval, lags, crits = kpss(self.x, 'c')
        assert_equal(lags, int(np.ceil(12. * np.power(len(self.x) / 100., 1 / 4.))))
        # assert_warns(UserWarning, kpss, self.x)



def test_pandasacovf():
    s = Series(lrange(1, 11))
    assert_almost_equal(acovf(s), acovf(s.values))


def test_acovf2d():
    dta = sunspots.load_pandas().data
    dta.index = DatetimeIndex(start='1700', end='2009', freq='A')[:309]
    del dta["YEAR"]
    res = acovf(dta)
    assert_equal(res, acovf(dta.values))
    X = np.random.random((10,2))
    assert_raises(ValueError, acovf, X)

def test_acovf_fft_vs_convolution():
    np.random.seed(1)
    q = np.random.normal(size=100)

    for demean in [True, False]:
        for unbiased in [True, False]:
            F1 = acovf(q, demean=demean, unbiased=unbiased, fft=True)
            F2 = acovf(q, demean=demean, unbiased=unbiased, fft=False)
            assert_almost_equal(F1, F2, decimal=7)

@pytest.mark.slow
def test_arma_order_select_ic():
    # smoke test, assumes info-criteria are right
    from statsmodels.tsa.arima_process import arma_generate_sample
    import statsmodels.api as sm

    arparams = np.array([.75, -.25])
    maparams = np.array([.65, .35])
    arparams = np.r_[1, -arparams]
    maparam = np.r_[1, maparams]
    nobs = 250
    np.random.seed(2014)
    y = arma_generate_sample(arparams, maparams, nobs)
    res = arma_order_select_ic(y, ic=['aic', 'bic'], trend='nc')
    # regression tests in case we change algorithm to minic in sas
    aic_x = np.array([[       np.nan,  552.7342255 ,  484.29687843],
                      [ 562.10924262,  485.5197969 ,  480.32858497],
                      [ 507.04581344,  482.91065829,  481.91926034],
                      [ 484.03995962,  482.14868032,  483.86378955],
                      [ 481.8849479 ,  483.8377379 ,  485.83756612]])
    bic_x = np.array([[       np.nan,  559.77714733,  494.86126118],
                      [ 569.15216446,  496.08417966,  494.41442864],
                      [ 517.61019619,  496.99650196,  499.52656493],
                      [ 498.12580329,  499.75598491,  504.99255506],
                      [ 499.49225249,  504.96650341,  510.48779255]])
    aic = DataFrame(aic_x , index=lrange(5), columns=lrange(3))
    bic = DataFrame(bic_x , index=lrange(5), columns=lrange(3))
    assert_almost_equal(res.aic.values, aic.values, 5)
    assert_almost_equal(res.bic.values, bic.values, 5)
    assert_equal(res.aic_min_order, (1, 2))
    assert_equal(res.bic_min_order, (1, 2))
    assert_(res.aic.index.equals(aic.index))
    assert_(res.aic.columns.equals(aic.columns))
    assert_(res.bic.index.equals(bic.index))
    assert_(res.bic.columns.equals(bic.columns))

    res = arma_order_select_ic(y, ic='aic', trend='nc')
    assert_almost_equal(res.aic.values, aic.values, 5)
    assert_(res.aic.index.equals(aic.index))
    assert_(res.aic.columns.equals(aic.columns))
    assert_equal(res.aic_min_order, (1, 2))

def test_arma_order_select_ic_failure():
    # this should trigger an SVD convergence failure, smoke test that it
    # returns, likely platform dependent failure...
    # looks like AR roots may be cancelling out for 4, 1?
    y = np.array([ 0.86074377817203640006,  0.85316549067906921611,
        0.87104653774363305363,  0.60692382068987393851,
        0.69225941967301307667,  0.73336177248909339976,
        0.03661329261479619179,  0.15693067239962379955,
        0.12777403512447857437, -0.27531446294481976   ,
       -0.24198139631653581283, -0.23903317951236391359,
       -0.26000241325906497947, -0.21282920015519238288,
       -0.15943768324388354896,  0.25169301564268781179,
        0.1762305709151877342 ,  0.12678133368791388857,
        0.89755829086753169399,  0.82667068795350151511])
    import warnings
    with warnings.catch_warnings():
        # catch a hessian inversion and convergence failure warning
        warnings.simplefilter("ignore")
        res = arma_order_select_ic(y)


def test_acf_fft_dataframe():
    # regression test #322

    result = acf(sunspots.load_pandas().data[['SUNACTIVITY']], fft=True)
    assert_equal(result.ndim, 1)

if __name__=="__main__":
    import pytest
    pytest.main([__file__, '-vvs', '-x', '--pdb'])
