# TODO: Tests for features that are just called
# TODO: Test for trend='ctt'
from __future__ import print_function, division
from statsmodels.compat import iteritems

import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_raises)
from numpy.testing.decorators import skipif
from numpy import log, polyval, diff, ceil, ones

from statsmodels.tsa.unitroot import (ADF, adfuller, DFGLS, ensure_1d,
                                      PhillipsPerron, KPSS, UnitRootTest,
                                      VarianceRatio)
from statsmodels.tsa.critical_values.dickey_fuller import tau_2010
from statsmodels.datasets import macrodata
import warnings
import sys

DECIMAL_5 = 5
DECIMAL_4 = 4
DECIMAL_3 = 3
DECIMAL_2 = 2
DECIMAL_1 = 1

PYTHON_LT_27 = sys.version_info[0] == 2 and sys.version_info[1] <= 6

def test_adf_autolag():
    #see issue #246
    #this is mostly a unit test
    d2 = macrodata.load().data
    x = log(d2['realgdp'])
    xd = diff(x)

    for k_trend, tr in enumerate(['nc', 'c', 'ct', 'ctt']):
        #check exog
        adf3 = adfuller(x, maxlag=None, autolag='aic',
                        regression=tr, store=True, regresults=True)
        st2 = adf3[-1]

        assert_equal(len(st2.autolag_results), 15 + 1)   # +1 for lagged level
        for l, res in sorted(iteritems(st2.autolag_results))[:5]:
            lag = l - k_trend
            #assert correct design matrices in _autolag
            assert_equal(res.model.exog[-10:, k_trend], x[-11:-1])
            assert_equal(res.model.exog[-1, k_trend + 1:], xd[-lag:-1][::-1])
            #min-ic lag of dfgls in Stata is also 2, or 9 for maic with notrend
            assert_equal(st2.usedlag, 2)

        #same result with lag fixed at usedlag of autolag
        adf2 = adfuller(x, maxlag=2, autolag=None, regression=tr)
        assert_almost_equal(adf3[:2], adf2[:2], decimal=12)

    tr = 'c'
    #check maxlag with autolag
    adf3 = adfuller(x, maxlag=5, autolag='aic',
                    regression=tr, store=True, regresults=True)
    assert_equal(len(adf3[-1].autolag_results), 5 + 1)
    adf3 = adfuller(x, maxlag=0, autolag='aic',
                    regression=tr, store=True, regresults=True)
    assert_equal(len(adf3[-1].autolag_results), 0 + 1)


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

    def __init__(self):
        self.res1 = adfuller(self.x, regression="c", autolag=None,
                             maxlag=4)
        self.teststat = .97505319
        self.pvalue = .99399563
        self.critvalues = [-3.476, -2.883, -2.573]


class TestADFConstantTrend(CheckADF):
    """
    """

    def __init__(self):
        self.res1 = adfuller(self.x, regression="ct", autolag=None,
                             maxlag=4)
        self.teststat = -1.8566374
        self.pvalue = .67682968
        self.critvalues = [-4.007, -3.437, -3.137]


#class TestADFConstantTrendSquared(CheckADF):
#    """
#    """
#    pass
#TODO: get test values from R?

class TestADFNoConstant(CheckADF):
    """
    """

    def __init__(self):
        self.res1 = adfuller(self.x, regression="nc", autolag=None,
                             maxlag=4)
        self.teststat = 3.5227498
        self.pvalue = .99999  # Stata does not return a p-value for noconstant.
        # Tau^max in MacKinnon (1994) is missing, so it is
        # assumed that its right-tail is well-behaved
        self.critvalues = [-2.587, -1.950, -1.617]


# No Unit Root

class TestADFConstant2(CheckADF):
    def __init__(self):
        self.res1 = adfuller(self.y, regression="c", autolag=None,
                             maxlag=1)
        self.teststat = -4.3346988
        self.pvalue = .00038661
        self.critvalues = [-3.476, -2.883, -2.573]


class TestADFConstantTrend2(CheckADF):
    def __init__(self):
        self.res1 = adfuller(self.y, regression="ct", autolag=None,
                             maxlag=1)
        self.teststat = -4.425093
        self.pvalue = .00199633
        self.critvalues = [-4.006, -3.437, -3.137]


class TestADFNoConstant2(CheckADF):
    def __init__(self):
        self.res1 = adfuller(self.y, regression="nc", autolag=None,
                             maxlag=1)
        self.teststat = -2.4511596
        self.pvalue = 0.013747  # Stata does not return a p-value for noconstant
        # this value is just taken from our results
        self.critvalues = [-2.587, -1.950, -1.617]


class TestUnitRoot(object):
    @classmethod
    def setupClass(cls):
        from statsmodels.datasets.macrodata import load

        cls.cpi = log(load().data['cpi'])
        cls.inflation = diff(cls.cpi)
        cls.inflation_change = diff(cls.inflation)

    def test_adf_no_options(self):
        adf = ADF(self.inflation)
        assert_almost_equal(adf.stat, -3.09310, DECIMAL_4)
        assert_equal(adf.lags, 2)
        assert_almost_equal(adf.pvalue, .027067, DECIMAL_4)
        adf.regression.summary()

    def test_adf_no_lags(self):
        adf = ADF(self.inflation, lags=0).stat
        assert_almost_equal(adf, -6.56880, DECIMAL_4)

    def test_adf_nc_no_lags(self):
        adf = ADF(self.inflation, trend='nc', lags=0)
        assert_almost_equal(adf.stat, -3.88845, DECIMAL_4)
        # 16.239

    def test_adf_c_no_lags(self):
        adf = ADF(self.inflation, trend='c', lags=0)
        assert_almost_equal(adf.stat, -6.56880, DECIMAL_4)
        assert_equal(adf.nobs, self.inflation.shape[0] - adf.lags - 1)

    def test_adf_ct_no_lags(self):
        adf = ADF(self.inflation, trend='ct', lags=0)
        assert_almost_equal(adf.stat, -6.66705, DECIMAL_4)

    def test_adf_lags_10(self):
        adf = ADF(self.inflation, lags=10)
        assert_almost_equal(adf.stat, -2.28375, DECIMAL_4)
        adf.summary()

    def test_adf_auto_bic(self):
        adf = ADF(self.inflation, method='BIC')
        assert_equal(adf.lags, 2)

    def test_adf_critical_value(self):
        adf = ADF(self.inflation, trend='c', lags=3)
        adf_cv = adf.critical_values
        temp = polyval(tau_2010['c'][0, :, ::-1].T, 1. / adf.nobs)
        cv = {'1%': temp[0], '5%': temp[1], '10%': temp[2]}
        for k, v in iteritems(cv):
            assert_almost_equal(v, adf_cv[k])

    def test_adf_auto_t_stat(self):
        adf = ADF(self.inflation, method='t-stat')
        assert_equal(adf.lags, 10)
        old_stat = adf.stat
        adf.lags += 1
        assert adf.stat != old_stat
        old_stat = adf.stat
        assert_equal(adf.y, self.inflation)
        adf.trend = 'ctt'
        assert adf.stat != old_stat
        assert adf.trend == 'ctt'
        assert len(adf.valid_trends) == len(('nc', 'c', 'ct', 'ctt'))
        for d in adf.valid_trends:
            assert d in ('nc', 'c', 'ct', 'ctt')
        assert adf.null_hypothesis == 'The process contains a unit root.'
        assert adf.alternative_hypothesis == 'The process is weakly stationary.'

    def test_kpss_auto(self):
        kpss = KPSS(self.inflation)
        m = self.inflation.shape[0]
        lags = np.ceil(12.0 * (m / 100) ** (1.0 / 4))
        assert_equal(kpss.lags, lags)

    def test_kpss(self):
        kpss = KPSS(self.inflation, trend='ct', lags=12)
        assert_almost_equal(kpss.stat, .235581902996454, DECIMAL_4)
        assert_equal(self.inflation.shape[0], kpss.nobs)
        kpss.summary()

    def test_kpss_c(self):
        kpss = KPSS(self.inflation, trend='c', lags=12)
        assert_almost_equal(kpss.stat, .3276290340191141, DECIMAL_4)

    def test_pp(self):
        pp = PhillipsPerron(self.inflation, lags=12)
        assert_almost_equal(pp.stat, -7.8076512, DECIMAL_4)
        assert pp.test_type == 'tau'
        pp.test_type = 'rho'
        assert_almost_equal(pp.stat, -108.1552688, DECIMAL_2)
        pp.summary()

    @skipif(PYTHON_LT_27)
    def test_pp_bad_type(self):
        pp = PhillipsPerron(self.inflation, lags=12)
        with assert_raises(ValueError):
            pp.test_type = 'unknown'

    def test_pp_auto(self):
        pp = PhillipsPerron(self.inflation)
        n = self.inflation.shape[0] - 1
        lags = ceil(12.0 * ((n / 100.0) ** (1.0 / 4.0)))
        assert_equal(pp.lags, lags)
        assert_almost_equal(pp.stat, -8.135547778, DECIMAL_4)
        pp.test_type = 'rho'
        assert_almost_equal(pp.stat, -118.7746451, DECIMAL_2)

    def test_dfgls_c(self):
        dfgls = DFGLS(self.inflation, trend='c', lags=0)
        assert_almost_equal(dfgls.stat, -6.017304, DECIMAL_4)
        dfgls.summary()
        dfgls.regression.summary()

    def test_dfgls(self):
        dfgls = DFGLS(self.inflation, trend='ct', lags=0)
        assert_almost_equal(dfgls.stat, -6.300927, DECIMAL_4)
        dfgls.summary()
        dfgls.regression.summary()

    def test_dfgls_auto(self):
        dfgls = DFGLS(self.inflation, trend='ct', method='BIC', max_lags=3)
        assert_equal(dfgls.lags, 2)
        assert_almost_equal(dfgls.stat, -2.9035369, DECIMAL_4)

    @skipif(PYTHON_LT_27)
    def test_dfgls_bad_trend(self):
        dfgls = DFGLS(self.inflation, trend='ct', method='BIC', max_lags=3)
        with assert_raises(ValueError):
            dfgls.trend = 'nc'

        assert dfgls != 0.0

    @skipif(PYTHON_LT_27)
    def test_ensure_1d(self):
        assert_raises(ValueError, ensure_1d, *(ones((2, 2)),))

    @skipif(PYTHON_LT_27)
    def test_negative_lag(self):
        adf = ADF(self.inflation)
        with assert_raises(ValueError):
            adf.lags = -1

    @skipif(PYTHON_LT_27)
    def test_invalid_determinstic(self):
        adf = ADF(self.inflation)
        with assert_raises(ValueError):
            adf.trend = 'bad-value'

    def test_variance_ratio(self):
        vr = VarianceRatio(self.inflation, debiased=False)
        y = self.inflation
        dy = np.diff(y)
        mu = dy.mean()
        dy2 = y[2:] - y[:-2]
        nq = dy.shape[0]
        denom = np.sum((dy - mu) ** 2.0) / (nq)
        num = np.sum((dy2 - 2 * mu) ** 2.0) / (nq * 2)
        ratio = num / denom

        assert_almost_equal(ratio, vr.vr)

    def test_variance_ratio_no_overlap(self):
        vr = VarianceRatio(self.inflation, overlap=False)

        with warnings.catch_warnings(record=True) as w:
            computed_value = vr.vr
            assert_equal(len(w), 1)

        y = self.inflation
        # Adjust due ot sample size
        y = y[:-1]
        dy = np.diff(y)
        mu = dy.mean()
        dy2 = y[2::2] - y[:-2:2]
        nq = dy.shape[0]
        denom = np.sum((dy - mu) ** 2.0) / nq
        num = np.sum((dy2 - 2 * mu) ** 2.0) / nq
        ratio = num / denom
        assert_equal(ratio, computed_value)

        vr.overlap = True
        assert_equal(vr.overlap, True)
        vr2 = VarianceRatio(self.inflation)
        assert_almost_equal(vr.stat, vr2.stat)

    def test_variance_ratio_non_robust(self):
        vr = VarianceRatio(self.inflation, robust=False, debiased=False)
        y = self.inflation
        dy = np.diff(y)
        mu = dy.mean()
        dy2 = y[2:] - y[:-2]
        nq = dy.shape[0]
        denom = np.sum((dy - mu) ** 2.0) / nq
        num = np.sum((dy2 - 2 * mu) ** 2.0) / (nq * 2)
        ratio = num / denom
        variance = 3.0 / 2.0
        stat = np.sqrt(nq) * (ratio - 1) / np.sqrt(variance)
        assert_almost_equal(stat, vr.stat)
        orig_stat = vr.stat
        vr.robust = True
        assert_equal(vr.robust, True)
        assert vr.stat != orig_stat

    def test_variance_ratio_no_constant(self):
        y = np.random.randn(100)
        vr = VarianceRatio(y, trend='nc', debiased=False)
        dy = np.diff(y)
        mu = 0.0
        dy2 = y[2:] - y[:-2]
        nq = dy.shape[0]
        denom = np.sum((dy - mu) ** 2.0) / nq
        num = np.sum((dy2 - 2 * mu) ** 2.0) / (nq * 2)
        ratio = num / denom
        assert_almost_equal(ratio, vr.vr)
        assert_equal(vr.debiased, False)

    @skipif(PYTHON_LT_27)
    def test_variance_ratio_invalid_lags(self):
        y = self.inflation
        assert_raises(ValueError, VarianceRatio, y, lags=1)

    def test_variance_ratio_generic(self):
        # TODO: Currently not a test, just makes sure code runs at all
        vr = VarianceRatio(self.inflation, lags=24)


if __name__ == "__main__":
    import nose

    nose.runmodule(argv=[__file__, '-vvs', '-x'], exit=False)

