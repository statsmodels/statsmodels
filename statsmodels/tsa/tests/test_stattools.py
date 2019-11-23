import os
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import (assert_almost_equal, assert_equal, assert_raises,
                           assert_, assert_allclose)
from pandas import Series, date_range, DataFrame

from statsmodels.compat.numpy import lstsq
from statsmodels.compat.pandas import assert_index_equal
from statsmodels.compat.platform import PLATFORM_WIN
from statsmodels.compat.python import lrange
from statsmodels.datasets import macrodata, sunspots, nile, randhie, modechoice
from statsmodels.tools.sm_exceptions import (CollinearityWarning,
                                             MissingDataError,
                                             InterpolationWarning)
from statsmodels.tsa.arima_process import arma_acovf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import (adfuller, acf, pacf_yw, pacf_ols,
                                       pacf, grangercausalitytests,
                                       coint, acovf, kpss,
                                       arma_order_select_ic, levinson_durbin,
                                       levinson_durbin_pacf, pacf_burg,
                                       innovations_algo, innovations_filter)

DECIMAL_8 = 8
DECIMAL_6 = 6
DECIMAL_5 = 5
DECIMAL_4 = 4
DECIMAL_3 = 3
DECIMAL_2 = 2
DECIMAL_1 = 1


@pytest.fixture('module')
def acovf_data():
    rnd = np.random.RandomState(12345)
    return rnd.randn(250)


class CheckADF(object):
    """
    Test Augmented Dickey-Fuller

    Test values taken from Stata.
    """
    levels = ['1%', '5%', '10%']
    data = macrodata.load_pandas()
    x = data.data['realgdp'].values
    y = data.data['infl'].values

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

        cls.pvalue = .99999
        # Stata does not return a p-value for noconstant.
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
        cls.pvalue = 0.013747
        # Stata does not return a p-value for noconstant
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
        cls.res1 = acf(cls.x, nlags=40, qstat=True, alpha=.05, fft=False)
        cls.confint_res = cls.results[['acvar_lb','acvar_ub']].values

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
                           missing='drop', fft=False)
        cls.res_conservative = acf(cls.x, nlags=40, qstat=True, alpha=.05,
                                   fft=False, missing='conservative')
        cls.acf_none = np.empty(40) * np.nan  # lags 1 to 40 inclusive
        cls.qstat_none = np.empty(40) * np.nan
        cls.res_none = acf(cls.x, nlags=40, qstat=True, alpha=.05,
                           missing='none', fft=False)

    def test_raise(self):
        with pytest.raises(MissingDataError):
            acf(self.x, nlags=40, qstat=True, fft=False, alpha=.05,
                missing='raise')

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

    def test_ols_inefficient(self):
        lag_len = 5
        pacfols = pacf_ols(self.x, nlags=lag_len, efficient=False)
        x = self.x.copy()
        x -= x.mean()
        n = x.shape[0]
        lags = np.zeros((n - 5, 5))
        lead = x[5:]
        direct = np.empty(lag_len + 1)
        direct[0] = 1.0
        for i in range(lag_len):
            lags[:, i] = x[5 - (i + 1):-(i + 1)]
            direct[i + 1] = lstsq(lags[:, :(i + 1)], lead, rcond=None)[0][-1]
        assert_allclose(pacfols, direct, atol=1e-8)

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
    data = macrodata.load_pandas()
    y1 = data.data['realcons'].values
    y2 = data.data['realgdp'].values

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
    warnings.simplefilter('always', CollinearityWarning)
    with pytest.warns(CollinearityWarning):
        c = coint(y, y, trend="c", maxlag=0, autolag=None)
    assert_equal(c[1], 0.0)
    assert_(np.isneginf(c[0]))


def test_coint_perfect_collinearity():
    # test uses nearly perfect collinearity
    nobs = 200
    scale_e = 1
    np.random.seed(123)
    x = scale_e * np.random.randn(nobs, 2)
    y = 1 + x.sum(axis=1) + 1e-7 * np.random.randn(nobs)
    warnings.simplefilter('always', CollinearityWarning)
    with warnings.catch_warnings(record=True) as w:
        c = coint(y, x, trend="c", maxlag=0, autolag=None)
    assert_equal(c[1], 0.0)
    assert_(np.isneginf(c[0]))


class TestGrangerCausality(object):

    def test_grangercausality(self):
        # some example data
        mdata = macrodata.load_pandas().data
        mdata = mdata[['realgdp', 'realcons']].values
        data = mdata.astype(float)
        data = np.diff(np.log(data), axis=0)

        #R: lmtest:grangertest
        r_result = [0.243097, 0.7844328, 195, 2]  # f_test
        gr = grangercausalitytests(data[:, 1::-1], 2, verbose=False)
        assert_almost_equal(r_result, gr[2][0]['ssr_ftest'], decimal=7)
        assert_almost_equal(gr[2][0]['params_ftest'], gr[2][0]['ssr_ftest'],
                            decimal=7)

    def test_granger_fails_on_nobs_check(self, reset_randomstate):
        # Test that if maxlag is too large, Granger Test raises a clear error.
        X = np.random.rand(10, 2)
        grangercausalitytests(X, 2, verbose=False)  # This should pass.
        assert_raises(ValueError, grangercausalitytests, X, 3, verbose=False)


class SetupKPSS(object):
    data = macrodata.load_pandas()
    x = data.data['realgdp'].values


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

    def test_fail_nonvector_input(self, reset_randomstate):
        # should be fine
        with pytest.warns(InterpolationWarning):
            kpss(self.x, lags='legacy')

        x = np.random.rand(20, 2)
        assert_raises(ValueError, kpss, x)

    def test_fail_unclear_hypothesis(self):
        # these should be fine,
        with pytest.warns(InterpolationWarning):
            kpss(self.x, 'c', lags='legacy')
        with pytest.warns(InterpolationWarning):
            kpss(self.x, 'C', lags='legacy')
        with pytest.warns(InterpolationWarning):
            kpss(self.x, 'ct', lags='legacy')
        with pytest.warns(InterpolationWarning):
            kpss(self.x, 'CT', lags='legacy')

        assert_raises(ValueError, kpss, self.x, "unclear hypothesis",
                      lags='legacy')

    def test_teststat(self):
        with pytest.warns(InterpolationWarning):
            kpss_stat, pval, lags, crits = kpss(self.x, 'c', 3)
        assert_almost_equal(kpss_stat, 5.0169, DECIMAL_3)

        with pytest.warns(InterpolationWarning):
            kpss_stat, pval, lags, crits = kpss(self.x, 'ct', 3)
        assert_almost_equal(kpss_stat, 1.1828, DECIMAL_3)

    def test_pval(self):
        with pytest.warns(InterpolationWarning):
            kpss_stat, pval, lags, crits = kpss(self.x, 'c', 3)
        assert_equal(pval, 0.01)

        with pytest.warns(InterpolationWarning):
            kpss_stat, pval, lags, crits = kpss(self.x, 'ct', 3)
        assert_equal(pval, 0.01)

    def test_store(self):
        with pytest.warns(InterpolationWarning):
            kpss_stat, pval, crit, store = kpss(self.x, 'c', 3, True)

        # assert attributes, and make sure they're correct
        assert_equal(store.nobs, len(self.x))
        assert_equal(store.lags, 3)

    # test autolag function _kpss_autolag against SAS 9.3
    def test_lags(self):
        # real GDP from macrodata data set
        with pytest.warns(InterpolationWarning):
            res = kpss(self.x, 'c', lags='auto')
        assert_equal(res[2], 9)
        # real interest rates from macrodata data set
        res = kpss(sunspots.load(True).data['SUNACTIVITY'], 'c', lags='auto')
        assert_equal(res[2], 7)
        # volumes from nile data set
        with pytest.warns(InterpolationWarning):
            res = kpss(nile.load(True).data['volume'], 'c', lags='auto')
        assert_equal(res[2], 5)
        # log-coinsurance from randhie data set
        with pytest.warns(InterpolationWarning):
            res = kpss(randhie.load(True).data['lncoins'], 'ct', lags='auto')
        assert_equal(res[2], 75)
        # in-vehicle time from modechoice data set
        with pytest.warns(InterpolationWarning):
            res = kpss(modechoice.load(True).data['invt'], 'ct', lags='auto')
        assert_equal(res[2], 18)

    def test_kpss_fails_on_nobs_check(self):
        # Test that if lags exceeds number of observations KPSS raises a
        # clear error
        # GH5925
        nobs = len(self.x)
        msg = (r"lags \({}\) must be < number of observations \({}\)"
               .format(nobs, nobs))
        with pytest.raises(ValueError, match=msg):
            kpss(self.x, 'c', lags=nobs)

    def test_kpss_autolags_does_not_assign_lags_equal_to_nobs(self):
        # Test that if *autolags* exceeds number of observations, we set suitable lags
        # GH5925
        data_which_breaks_autolag = np.array(
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0,
             0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0,
             0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
             0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
             0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
             1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1,
             1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1,
             0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0,
             0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0,
             0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
             0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
             0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
             1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1,
             1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]).astype(float)

        kpss(data_which_breaks_autolag, lags="auto")

    def test_legacy_lags(self):
        # Test legacy lags are the same
        with pytest.warns(InterpolationWarning):
            res = kpss(self.x, 'c', lags='legacy')
        assert_equal(res[2], 15)

    def test_unknown_lags(self):
        # Test legacy lags are the same
        with pytest.raises(ValueError):
            kpss(self.x, 'c', lags='unknown')

    def test_deprecation(self):
        with pytest.warns(FutureWarning):
            kpss(self.x, 'c')


def test_pandasacovf():
    s = Series(lrange(1, 11))
    assert_almost_equal(acovf(s, fft=False), acovf(s.values, fft=False))


def test_acovf2d(reset_randomstate):
    dta = sunspots.load_pandas().data
    dta.index = date_range(start='1700', end='2009', freq='A')[:309]
    del dta["YEAR"]
    res = acovf(dta, fft=False)
    assert_equal(res, acovf(dta.values, fft=False))
    x = np.random.random((10, 2))
    with pytest.raises(ValueError):
        acovf(x, fft=False)


@pytest.mark.parametrize('demean', [True, False])
@pytest.mark.parametrize('unbiased', [True, False])
def test_acovf_fft_vs_convolution(demean, unbiased):
    np.random.seed(1)
    q = np.random.normal(size=100)

    F1 = acovf(q, demean=demean, unbiased=unbiased, fft=True)
    F2 = acovf(q, demean=demean, unbiased=unbiased, fft=False)
    assert_almost_equal(F1, F2, decimal=7)


@pytest.mark.smoke
@pytest.mark.slow
def test_arma_order_select_ic():
    # smoke test, assumes info-criteria are right
    from statsmodels.tsa.arima_process import arma_generate_sample

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
    aic = DataFrame(aic_x, index=lrange(5), columns=lrange(3))
    bic = DataFrame(bic_x, index=lrange(5), columns=lrange(3))
    assert_almost_equal(res.aic.values, aic.values, 5)
    assert_almost_equal(res.bic.values, bic.values, 5)
    assert_equal(res.aic_min_order, (1, 2))
    assert_equal(res.bic_min_order, (1, 2))
    assert_(res.aic.index.equals(aic.index))
    assert_(res.aic.columns.equals(aic.columns))
    assert_(res.bic.index.equals(bic.index))
    assert_(res.bic.columns.equals(bic.columns))

    index = pd.date_range('2000-1-1', freq='M', periods=len(y))
    y_series = pd.Series(y, index=index)
    res_pd = arma_order_select_ic(y_series, max_ar=2, max_ma=1,
                                  ic=['aic', 'bic'], trend='nc')
    assert_almost_equal(res_pd.aic.values, aic.values[:3, :2], 5)
    assert_almost_equal(res_pd.bic.values, bic.values[:3, :2], 5)
    assert_equal(res_pd.aic_min_order, (2, 1))
    assert_equal(res_pd.bic_min_order, (1, 1))

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


def test_levinson_durbin_acov():
    rho = 0.9
    m = 20
    acov = rho**np.arange(200)
    sigma2_eps, ar, pacf, _, _ = levinson_durbin(acov, m, isacov=True)
    assert_allclose(sigma2_eps, 1 - rho ** 2)
    assert_allclose(ar, np.array([rho] + [0] * (m - 1)), atol=1e-8)
    assert_allclose(pacf, np.array([1, rho] + [0] * (m - 1)), atol=1e-8)



@pytest.mark.parametrize("missing", ['conservative', 'drop', 'raise', 'none'])
@pytest.mark.parametrize("fft", [False, True])
@pytest.mark.parametrize("demean", [True, False])
@pytest.mark.parametrize("unbiased", [True, False])
def test_acovf_nlags(acovf_data, unbiased, demean, fft, missing):
    full = acovf(acovf_data, unbiased=unbiased, demean=demean, fft=fft,
                 missing=missing)
    limited = acovf(acovf_data, unbiased=unbiased, demean=demean, fft=fft,
                    missing=missing, nlag=10)
    assert_allclose(full[:11], limited)


@pytest.mark.parametrize("missing", ['conservative', 'drop'])
@pytest.mark.parametrize("fft", [False, True])
@pytest.mark.parametrize("demean", [True, False])
@pytest.mark.parametrize("unbiased", [True, False])
def test_acovf_nlags_missing(acovf_data, unbiased, demean, fft, missing):
    acovf_data = acovf_data.copy()
    acovf_data[1:3] = np.nan
    full = acovf(acovf_data, unbiased=unbiased, demean=demean, fft=fft,
                 missing=missing)
    limited = acovf(acovf_data, unbiased=unbiased, demean=demean, fft=fft,
                    missing=missing, nlag=10)
    assert_allclose(full[:11], limited)


def test_acovf_error(acovf_data):
    with pytest.raises(ValueError):
        acovf(acovf_data, nlag=250, fft=False)


def test_acovf_warns(acovf_data):
    with pytest.warns(FutureWarning):
        acovf(acovf_data)


def test_acf_warns(acovf_data):
    with pytest.warns(FutureWarning):
        acf(acovf_data, nlags=40)


def test_pacf2acf_ar():
    pacf = np.zeros(10)
    pacf[0] = 1
    pacf[1] = 0.9
    ar, acf = levinson_durbin_pacf(pacf)
    assert_allclose(acf, 0.9 ** np.arange(10.))
    assert_allclose(ar, pacf[1:], atol=1e-8)

    ar, acf = levinson_durbin_pacf(pacf, nlags=5)
    assert_allclose(acf, 0.9 ** np.arange(6.))
    assert_allclose(ar, pacf[1:6], atol=1e-8)


def test_pacf2acf_levinson_durbin():
    pacf = -0.9 ** np.arange(11.)
    pacf[0] = 1
    ar, acf = levinson_durbin_pacf(pacf)
    _, ar_ld, pacf_ld, _, _ = levinson_durbin(acf, 10, isacov=True)
    assert_allclose(ar, ar_ld, atol=1e-8)
    assert_allclose(pacf, pacf_ld, atol=1e-8)

    # From R, FitAR, PacfToAR
    ar_from_r = [-4.1609, -9.2549, -14.4826, -17.6505, -17.5012, -14.2969, -9.5020, -4.9184,
                 -1.7911, -0.3486]
    assert_allclose(ar, ar_from_r, atol=1e-4)


def test_pacf2acf_errors():
    pacf = -0.9 ** np.arange(11.)
    pacf[0] = 1
    with pytest.raises(ValueError):
        levinson_durbin_pacf(pacf, nlags=20)
    with pytest.raises(ValueError):
        levinson_durbin_pacf(pacf[:1])
    with pytest.raises(ValueError):
        levinson_durbin_pacf(np.zeros(10))
    with pytest.raises(ValueError):
        levinson_durbin_pacf(np.zeros((10, 2)))


def test_pacf_burg():
    rnd = np.random.RandomState(12345)
    e = rnd.randn(10001)
    y = e[1:] + 0.5 * e[:-1]
    pacf, sigma2 = pacf_burg(y, 10)
    yw_pacf = pacf_yw(y, 10)
    assert_allclose(pacf, yw_pacf, atol=5e-4)
    # Internal consistency check between pacf and sigma2
    ye = y - y.mean()
    s2y = ye.dot(ye) / 10000
    pacf[0] = 0
    sigma2_direct = s2y * np.cumprod(1 - pacf ** 2)
    assert_allclose(sigma2, sigma2_direct, atol=1e-3)


def test_pacf_burg_error():
    with pytest.raises(ValueError):
        pacf_burg(np.empty((20,2)), 10)
    with pytest.raises(ValueError):
        pacf_burg(np.empty(100), 101)


def test_innovations_algo_brockwell_davis():
    ma = -0.9
    acovf = np.array([1 + ma ** 2, ma])
    theta, sigma2 = innovations_algo(acovf, nobs=4)
    exp_theta = np.array([[0], [-.4972], [-.6606], [-.7404]])
    assert_allclose(theta, exp_theta, rtol=1e-4)
    assert_allclose(sigma2, [1.81, 1.3625, 1.2155, 1.1436], rtol=1e-4)

    theta, sigma2 = innovations_algo(acovf, nobs=500)
    assert_allclose(theta[-1, 0], ma)
    assert_allclose(sigma2[-1], 1.0)


def test_innovations_algo_rtol():
    ma = np.array([-0.9, 0.5])
    acovf = np.array([1 + (ma ** 2).sum(), ma[0] + ma[1] * ma[0], ma[1]])
    theta, sigma2 = innovations_algo(acovf, nobs=500)
    theta_2, sigma2_2 = innovations_algo(acovf, nobs=500, rtol=1e-8)
    assert_allclose(theta, theta_2)
    assert_allclose(sigma2, sigma2_2)


def test_innovations_errors():
    ma = -0.9
    acovf = np.array([1 + ma ** 2, ma])
    with pytest.raises(ValueError):
        innovations_algo(acovf, nobs=2.2)
    with pytest.raises(ValueError):
        innovations_algo(acovf, nobs=-1)
    with pytest.raises(ValueError):
        innovations_algo(np.empty((2, 2)))
    with pytest.raises(ValueError):
        innovations_algo(acovf, rtol='none')


def test_innovations_filter_brockwell_davis(reset_randomstate):
    ma = -0.9
    acovf = np.array([1 + ma ** 2, ma])
    theta, _ = innovations_algo(acovf, nobs=4)
    e = np.random.randn(5)
    endog = e[1:] + ma * e[:-1]
    resid = innovations_filter(endog, theta)
    expected = [endog[0]]
    for i in range(1, 4):
        expected.append(endog[i] - theta[i, 0] * expected[-1])
    expected = np.array(expected)
    assert_allclose(resid, expected)


def test_innovations_filter_pandas(reset_randomstate):
    ma = np.array([-0.9, 0.5])
    acovf = np.array([1 + (ma ** 2).sum(), ma[0] + ma[1] * ma[0], ma[1]])
    theta, _ = innovations_algo(acovf, nobs=10)
    endog = np.random.randn(10)
    endog_pd = pd.Series(endog,
                         index=pd.date_range('2000-01-01', periods=10))
    resid = innovations_filter(endog, theta)
    resid_pd = innovations_filter(endog_pd, theta)
    assert_allclose(resid, resid_pd.values)
    assert_index_equal(endog_pd.index, resid_pd.index)


def test_innovations_filter_errors():
    ma = -0.9
    acovf = np.array([1 + ma ** 2, ma])
    theta, _ = innovations_algo(acovf, nobs=4)
    with pytest.raises(ValueError):
        innovations_filter(np.empty((2, 2)), theta)
    with pytest.raises(ValueError):
        innovations_filter(np.empty(4), theta[:-1])
    with pytest.raises(ValueError):
        innovations_filter(pd.DataFrame(np.empty((1, 4))), theta)


def test_innovations_algo_filter_kalman_filter(reset_randomstate):
    # Test the innovations algorithm and filter against the Kalman filter
    # for exact likelihood evaluation of an ARMA process
    ar_params = np.array([0.5])
    ma_params = np.array([0.2])
    # TODO could generalize to sigma2 != 1, if desired, after #5324 is merged
    # and there is a sigma2 argument to arma_acovf
    # (but maybe this is not really necessary for the point of this test)
    sigma2 = 1

    endog = np.random.normal(size=10)

    # Innovations algorithm approach
    acovf = arma_acovf(np.r_[1, -ar_params], np.r_[1, ma_params],
                       nobs=len(endog))

    theta, v = innovations_algo(acovf)
    u = innovations_filter(endog, theta)
    llf_obs = -0.5 * u**2 / (sigma2 * v) - 0.5 * np.log(2 * np.pi * v)

    # Kalman filter apparoach
    mod = SARIMAX(endog, order=(len(ar_params), 0, len(ma_params)))
    res = mod.filter(np.r_[ar_params, ma_params, sigma2])

    # Test that the two approaches are identical
    atol = 1e-6 if PLATFORM_WIN else 0.0
    assert_allclose(u, res.forecasts_error[0], atol=atol)
    assert_allclose(theta[1:, 0], res.filter_results.kalman_gain[0, 0, :-1],
                    atol=atol)
    assert_allclose(llf_obs, res.llf_obs, atol=atol)


def test_adfuller_short_series(reset_randomstate):
    y = np.random.standard_normal(7)
    res = adfuller(y, store=True)
    assert res[-1].maxlag == 1
    y = np.random.standard_normal(2)
    with pytest.raises(ValueError, match='sample size is too short'):
        adfuller(y)
    y = np.random.standard_normal(3)
    with pytest.raises(ValueError, match='sample size is too short'):
        adfuller(y, regression='ct')


def test_adfuller_maxlag_too_large(reset_randomstate):
    y = np.random.standard_normal(100)
    with pytest.raises(ValueError, match='maxlag must be less than'):
        adfuller(y, maxlag=51)
