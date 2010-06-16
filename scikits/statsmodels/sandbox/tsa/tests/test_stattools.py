from scikits.statsmodels.sandbox.tsa.stattools import dfuller
from numpy.testing import array_almost_equal
from scikits.statsmodels.datasets import macrodata

class TestADF(object):
    def setup(self):
        data = macrodata.load
        x = data['realgdp']

class TestADFConstantNoTrend(TestADF):
    """
    Dickey-Fuller test for unit root
    """
    return

class TestADFNoConstant(TestADF):
    """
    """
    return

def TestADFTrendNoConstant(TestADF):
    """
    """
    return

def TestADFTrendConstant(TestADF):
    """
    """
    return
