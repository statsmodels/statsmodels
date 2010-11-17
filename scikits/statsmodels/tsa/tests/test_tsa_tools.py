'''tests for some time series analysis functions

'''

import numpy as np
from numpy.testing import assert_array_almost_equal
import scikits.statsmodels as sm
import scikits.statsmodels.tsa.stattools as tsa

from results import savedrvs
from results.datamlw_tls import mlacf, mlccf, mlpacf, mlywar

xo = savedrvs.rvsdata.xar2
x100 = xo[-100:]/1000.
x1000 = xo/1000.


def test_acf():
    acf_x = tsa.acf(x100, unbiased=False)[:21]
    assert_array_almost_equal(mlacf.acf100.ravel(), acf_x, 8) #why only dec=8
    acf_x = tsa.acf(x1000, unbiased=False)[:21]
    assert_array_almost_equal(mlacf.acf1000.ravel(), acf_x, 8) #why only dec=9

def test_ccf():
    ccf_x = tsa.ccf(x100[4:], x100[:-4], unbiased=False)[:21]
    assert_array_almost_equal(mlccf.ccf100.ravel()[:21][::-1], ccf_x, 8)
    ccf_x = tsa.ccf(x1000[4:], x1000[:-4], unbiased=False)[:21]
    assert_array_almost_equal(mlccf.ccf1000.ravel()[:21][::-1], ccf_x, 8)

def test_pacf_yw():
    pacfyw = tsa.pacf_yw(x100, 20, method='mle')
    assert_array_almost_equal(mlpacf.pacf100.ravel(), pacfyw, 1)
    pacfyw = tsa.pacf_yw(x1000, 20, method='mle')
    assert_array_almost_equal(mlpacf.pacf1000.ravel(), pacfyw, 2)
    #assert False

def test_pacf_ols():
    pacfols = tsa.pacf_ols(x100, 20)
    assert_array_almost_equal(mlpacf.pacf100.ravel(), pacfols, 2)
    pacfols = tsa.pacf_ols(x1000, 20)
    assert_array_almost_equal(mlpacf.pacf1000.ravel(), pacfols, 5)
    #assert False

def test_ywcoef():
    assert_array_almost_equal(mlywar.arcoef100[1:],
                    -sm.regression.yule_walker(x100, 10, method='mle')[0], 8)
    assert_array_almost_equal(mlywar.arcoef1000[1:],
                    -sm.regression.yule_walker(x1000, 20, method='mle')[0], 8)

if __name__ == '__main__':
    #running them directly
    test_acf()
    test_ccf()
    test_pacf_yw()
    test_pacf_ols()
    test_ywcoef()

