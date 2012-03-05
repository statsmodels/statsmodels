
import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_almost_equal,
                           assert_equal)


from statsmodels.tsa.arima_process import (arma_impulse_response,
                        lpol_fiar, lpol_fima)
from statsmodels.sandbox.tsa.fftarma import ArmaFft

from results.results_process import armarep  #benchmarkdata

arlist = [[1.],
          [1, -0.9],  #ma representation will need many terms to get high precision
          [1,  0.9],
          [1, -0.9, 0.3]]

malist = [[1.],
          [1,  0.9],
          [1, -0.9],
          [1,  0.9, -0.3]]



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


def test_spectrum():
    nfreq = 20
    w = np.linspace(0, np.pi, nfreq, endpoint=False)
    for ar in arlist:
        for ma in malist:
            arma = ArmaFft(ar, ma, 20)
            spdr, wr = arma.spdroots(w)
            spdp, wp = arma.spdpoly(w, 200)
            spdd, wd = arma.spddirect(nfreq*2)
            assert_equal(w, wr)
            assert_equal(w, wp)
            assert_almost_equal(w, wd[:nfreq], decimal=14)
            assert_almost_equal(spdr, spdp, decimal=7,
                                err_msg='spdr spdp not equal for %s, %s' % (ar, ma))
            assert_almost_equal(spdr, spdd[:nfreq], decimal=7,
                                err_msg='spdr spdd not equal for %s, %s' % (ar, ma))

def test_armafft():
    #test other methods
    nfreq = 20
    w = np.linspace(0, np.pi, nfreq, endpoint=False)
    for ar in arlist:
        for ma in malist:
            arma = ArmaFft(ar, ma, 20)
            ac1 = arma.invpowerspd(1024)[:10]
            ac2 = arma.acovf(10)[:10]
            assert_almost_equal(ac1, ac2, decimal=7,
                                err_msg='acovf not equal for %s, %s' % (ar, ma))


if __name__ == '__main__':
    test_fi()
    test_arma_impulse_response()
    test_spectrum()
    test_armafft()
