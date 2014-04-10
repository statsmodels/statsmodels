
from statsmodels.compat.python import range
import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_almost_equal,
                           assert_equal)


from statsmodels.tsa.arima_process import (arma_generate_sample, arma_acovf,
                        arma_acf, arma_impulse_response, lpol_fiar, lpol_fima)
from statsmodels.sandbox.tsa.fftarma import ArmaFft

from .results.results_process import armarep  #benchmarkdata

arlist = [[1.],
          [1, -0.9],  #ma representation will need many terms to get high precision
          [1,  0.9],
          [1, -0.9, 0.3]]

malist = [[1.],
          [1,  0.9],
          [1, -0.9],
          [1,  0.9, -0.3]]


def test_arma_acovf():
    # Check for specific AR(1)
    N = 20;
    phi = 0.9;
    sigma = 1;
    # rep 1: from module function
    rep1 = arma_acovf([1, -phi], [1], N);
    # rep 2: manually
    rep2 = [1.*sigma*phi**i/(1-phi**2) for i in range(N)];
    assert_almost_equal(rep1, rep2, 7); # 7 is max precision here


def test_arma_acf():
    # Check for specific AR(1)
    N = 20;
    phi = 0.9;
    sigma = 1;
    # rep 1: from module function
    rep1 = arma_acf([1, -phi], [1], N);
    # rep 2: manually
    acovf = np.array([1.*sigma*phi**i/(1-phi**2) for i in range(N)])
    rep2 = acovf / (1./(1-phi**2));
    assert_almost_equal(rep1, rep2, 8); # 8 is max precision here


def _manual_arma_generate_sample(ar, ma, eta):
    T = len(eta);
    ar = ar[::-1];
    ma = ma[::-1];
    p,q = len(ar), len(ma);
    rep2 = [0]*max(p,q); # initialize with zeroes
    for t in range(T):
        yt = eta[t];
        if p:
            yt += np.dot(rep2[-p:], ar);
        if q:
            # left pad shocks with zeros
            yt += np.dot([0]*(q-t) + list(eta[max(0,t-q):t]), ma);
        rep2.append(yt);
    return np.array(rep2[max(p,q):]);

def test_arma_generate_sample():
    # Test that this generates a true ARMA process
    # (amounts to just a test that scipy.signal.lfilter does what we want)
    T = 100;
    dists = [np.random.randn]
    for dist in dists:
        np.random.seed(1234);
        eta = dist(T);
        for ar in arlist:
            for ma in malist:
                # rep1: from module function
                np.random.seed(1234);
                rep1 = arma_generate_sample(ar, ma, T, distrvs=dist);
                # rep2: "manually" create the ARMA process
                ar_params = -1*np.array(ar[1:]);
                ma_params = np.array(ma[1:]);
                rep2 = _manual_arma_generate_sample(ar_params, ma_params, eta)
                assert_array_almost_equal(rep1, rep2, 13);


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
    test_arma_acovf()
    test_arma_acf()
    test_arma_generate_sample()
    test_fi()
    test_arma_impulse_response()
    test_spectrum()
    test_armafft()
