from scikits.statsmodels.sandbox.tsa.stattools import adfuller
from numpy.testing import assert_almost_equal
from scikits.statsmodels.datasets import macrodata

DECIMAL_6 = 6
DECIMAL_5 = 5
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

if __name__=="__main__":
    import nose
#    nose.runmodule(argv=[__file__, '-vvs','-x','-pdb'], exit=False)
    import numpy as np
    np.testing.run_module_suite()
