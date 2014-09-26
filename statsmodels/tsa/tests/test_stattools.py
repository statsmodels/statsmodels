from statsmodels.compat.python import lrange
from statsmodels.tsa.stattools import (adfuller, acf, pacf_ols, pacf_yw,
                                               pacf, grangercausalitytests,
                                               coint, acovf,
                                               arma_order_select_ic)
from statsmodels.tsa.base.datetools import dates_from_range
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_raises,
                           dec, assert_)
from numpy import genfromtxt#, concatenate
from statsmodels.datasets import macrodata, sunspots
from pandas import Series, Index, DataFrame
import os


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
        self.pvalue = .99999 # Stata does not return a p-value for noconstant.
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
        self.pvalue = 0.013747 # Stata does not return a p-value for noconstant
                               # this value is just taken from our results
        self.critvalues = [-2.587,-1.950,-1.617]

class CheckCorrGram(object):
    """
    Set up for ACF, PACF tests.
    """
    data = macrodata.load()
    x = data.data['realgdp']
    filename = os.path.dirname(os.path.abspath(__file__))+\
            "/results/results_corrgram.csv"
    results = genfromtxt(open(filename, "rb"), delimiter=",", names=True,dtype=float)

    #not needed: add 1. for lag zero
    #self.results['acvar'] = np.concatenate(([1.], self.results['acvar']))


class TestACF(CheckCorrGram):
    """
    Test Autocorrelation Function
    """
    def __init__(self):
        self.acf = self.results['acvar']
        #self.acf = np.concatenate(([1.], self.acf))
        self.qstat = self.results['Q1']
        self.res1 = acf(self.x, nlags=40, qstat=True, alpha=.05)
        self.confint_res = self.results[['acvar_lb','acvar_ub']].view((float,
                                                                            2))

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
    """
    Test Autocorrelation Function using FFT
    """
    def __init__(self):
        self.acf = self.results['acvarfft']
        self.qstat = self.results['Q1']
        self.res1 = acf(self.x, nlags=40, qstat=True, fft=True)

    def test_acf(self):
        assert_almost_equal(self.res1[0][1:], self.acf, DECIMAL_8)

    def test_qstat(self):
        #todo why is res1/qstat 1 short
        assert_almost_equal(self.res1[1], self.qstat, DECIMAL_3)


class TestPACF(CheckCorrGram):
    def __init__(self):
        self.pacfols = self.results['PACOLS']
        self.pacfyw = self.results['PACYW']

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

class TestCoint_t(CheckCoint):
    """
    Get AR(1) parameter on residuals
    """
    def __init__(self):
        self.coint_t = coint(self.y1, self.y2, regression ="c")[0]
        self.teststat = -1.8208817


class TestGrangerCausality(object):

    def test_grangercausality(self):
        # some example data
        mdata = macrodata.load().data
        mdata = mdata[['realgdp', 'realcons']]
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



def test_pandasacovf():
    s = Series(lrange(1, 11))
    assert_almost_equal(acovf(s), acovf(s.values))


def test_acovf2d():
    dta = sunspots.load_pandas().data
    dta.index = Index(dates_from_range('1700', '2008'))
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

@dec.slow
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
    import nose
#    nose.runmodule(argv=[__file__, '-vvs','-x','-pdb'], exit=False)
    import numpy as np
    np.testing.run_module_suite()
