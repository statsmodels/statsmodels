
import numpy as np
from numpy.testing import assert_array_almost_equal

from scikits.statsmodels.tsa.arima_process import (arma_impulse_response,
                        lpol_fiar, lpol_fima)

from results.results_process import armarep

def test_fi():
    #test identity of ma and ar representation of fi lag polynomial
    n = 100
    mafromar = arma_impulse_response(lpol_fiar(0.4, n=n), [1], n)
    assert_array_almost_equal(mafromar, lpol_fima(0.4, n=n), 13)


def test_arma_impulse_response():
    arrep = arma_impulse_response(armarep.ma, armarep.ar, nobs=21)[1:]
    marep = arma_impulse_response(armarep.ar, armarep.ma, nobs=21)[1:]
    assert_array_almost_equal(armarep.marep.ravel(), marep, 14)
    #difference in sign convention to matlab for AR term
    assert_array_almost_equal(-armarep.arrep.ravel(), arrep, 14)

if __name__ == '__main__':
    test_fi()
    test_arma_impulse_response()
