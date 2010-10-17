from scikits.statsmodels.tsa.stattools import (adfuller, acf, pacf_ols, pacf_yw)

from numpy.testing import assert_almost_equal
from numpy import genfromtxt#, concatenate
from scikits.statsmodels.datasets import macrodata
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
    results = genfromtxt(filename, delimiter=",", names=True,dtype=float)

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
        self.res1 = acf(self.x, nlags=40, qstat=True)

    def test_acf(self):
        assert_almost_equal(self.res1[0][1:41], self.acf, DECIMAL_8)

#    def test_confint(self):
#        pass
#NOTE: need to figure out how to center confidence intervals

    def test_qstat(self):
        assert_almost_equal(self.res1[1][:40], self.qstat, DECIMAL_3)
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
        pacfols = pacf_ols(self.x, nlags=40)
        assert_almost_equal(pacfols[1:], self.pacfols, DECIMAL_6)

    def test_yw(self):
        pacfyw = pacf_yw(self.x, nlags=40, method="mle")
        assert_almost_equal(pacfyw[1:], self.pacfyw, DECIMAL_8)

if __name__=="__main__":
    import nose
#    nose.runmodule(argv=[__file__, '-vvs','-x','-pdb'], exit=False)
    import numpy as np
    np.testing.run_module_suite()
