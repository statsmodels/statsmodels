from statsmodels.compat.python import range

from distutils.version import LooseVersion

from statsmodels.tsa.arima_model import ARMA
from unittest import TestCase

import pytest
import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_almost_equal,
                           assert_allclose,
                           assert_equal, assert_raises, assert_)

from statsmodels.tsa.arima_process import (arma_generate_sample, arma_acovf,
                                           arma_acf, arma_impulse_response, lpol_fiar, lpol_fima,
                                           ArmaProcess, lpol2index, index2lpol)
from statsmodels.sandbox.tsa.fftarma import ArmaFft

from statsmodels.tsa.tests.results.results_process import armarep  # benchmarkdata

arlist = [[1.],
          [1, -0.9],  # ma representation will need many terms to get high precision
          [1, 0.9],
          [1, -0.9, 0.3]]

malist = [[1.],
          [1, 0.9],
          [1, -0.9],
          [1, 0.9, -0.3]]

DECIMAL_4 = 4
NP16 = LooseVersion(np.__version__) < '1.7'


def test_arma_acovf():
    # Check for specific AR(1)
    N = 20
    phi = 0.9
    sigma = 1
    # rep 1: from module function
    rep1 = arma_acovf([1, -phi], [1], N)
    # rep 2: manually
    rep2 = [1. * sigma * phi ** i / (1 - phi ** 2) for i in range(N)]
    assert_almost_equal(rep1, rep2, 7)  # 7 is max precision here

def test_arma_acovf_persistent():
    # Test arma_acovf in case where there is a near-unit root.
    # .999 is high enough to trigger the "while ir[-1] > 5*1e-5:" clause,
    # but not high enough to trigger the "nobs_ir > 50000" clause.
    ar = np.array([1, -.9995])
    ma = np.array([1])
    process = ArmaProcess(ar, ma)
    res = process.acovf(10)

    # Theoretical variance sig2 given by:
    # sig2 = .9995**2 * sig2 + 1
    sig2 = 1/(1-.9995**2)

    corrs = .9995**np.arange(10)
    expected = sig2*corrs
    assert_equal(res.ndim, 1)
    assert_allclose(res, expected, atol=1e-6)
    # atol=7 breaks at .999, worked at .995

def test_arma_acf():
    # Check for specific AR(1)
    N = 20
    phi = 0.9
    sigma = 1
    # rep 1: from module function
    rep1 = arma_acf([1, -phi], [1], N)
    # rep 2: manually
    acovf = np.array([1. * sigma * phi ** i / (1 - phi ** 2) for i in range(N)])
    rep2 = acovf / (1. / (1 - phi ** 2))
    assert_almost_equal(rep1, rep2, 8)  # 8 is max precision here


def _manual_arma_generate_sample(ar, ma, eta):
    T = len(eta)
    ar = ar[::-1]
    ma = ma[::-1]
    p, q = len(ar), len(ma)
    rep2 = [0] * max(p, q)  # initialize with zeroes
    for t in range(T):
        yt = eta[t]
        if p:
            yt += np.dot(rep2[-p:], ar)
        if q:
            # left pad shocks with zeros
            yt += np.dot([0] * (q - t) + list(eta[max(0, t - q):t]), ma)
        rep2.append(yt)
    return np.array(rep2[max(p, q):])


def test_arma_generate_sample():
    # Test that this generates a true ARMA process
    # (amounts to just a test that scipy.signal.lfilter does what we want)
    T = 100
    dists = [np.random.randn]
    for dist in dists:
        np.random.seed(1234)
        eta = dist(T)
        for ar in arlist:
            for ma in malist:
                # rep1: from module function
                np.random.seed(1234)
                rep1 = arma_generate_sample(ar, ma, T, distrvs=dist)
                # rep2: "manually" create the ARMA process
                ar_params = -1 * np.array(ar[1:])
                ma_params = np.array(ma[1:])
                rep2 = _manual_arma_generate_sample(ar_params, ma_params, eta)
                assert_array_almost_equal(rep1, rep2, 13)


def test_fi():
    # test identity of ma and ar representation of fi lag polynomial
    n = 100
    mafromar = arma_impulse_response(lpol_fiar(0.4, n=n), [1], n)
    assert_array_almost_equal(mafromar, lpol_fima(0.4, n=n), 13)


def test_arma_impulse_response():
    arrep = arma_impulse_response(armarep.ma, armarep.ar, leads=21)[1:]
    marep = arma_impulse_response(armarep.ar, armarep.ma, leads=21)[1:]
    assert_array_almost_equal(armarep.marep.ravel(), marep, 14)
    # difference in sign convention to matlab for AR term
    assert_array_almost_equal(-armarep.arrep.ravel(), arrep, 14)


def test_spectrum():
    nfreq = 20
    w = np.linspace(0, np.pi, nfreq, endpoint=False)
    for ar in arlist:
        for ma in malist:
            arma = ArmaFft(ar, ma, 20)
            spdr, wr = arma.spdroots(w)
            spdp, wp = arma.spdpoly(w, 200)
            spdd, wd = arma.spddirect(nfreq * 2)
            assert_equal(w, wr)
            assert_equal(w, wp)
            assert_almost_equal(w, wd[:nfreq], decimal=14)
            assert_almost_equal(spdr, spdd[:nfreq], decimal=7,
                                err_msg='spdr spdd not equal for %s, %s' % (ar, ma))
            assert_almost_equal(spdr, spdp, decimal=7,
                                err_msg='spdr spdp not equal for %s, %s' % (ar, ma))


def test_armafft():
    # test other methods
    nfreq = 20
    w = np.linspace(0, np.pi, nfreq, endpoint=False)
    for ar in arlist:
        for ma in malist:
            arma = ArmaFft(ar, ma, 20)
            ac1 = arma.invpowerspd(1024)[:10]
            ac2 = arma.acovf(10)[:10]
            assert_almost_equal(ac1, ac2, decimal=7,
                                err_msg='acovf not equal for %s, %s' % (ar, ma))

def test_lpol2index_index2lpol():
    process = ArmaProcess([1, 0, 0, -0.8])
    coefs, locs = lpol2index(process.arcoefs)
    assert_almost_equal(coefs, [0.8])
    assert_equal(locs, [2])

    process = ArmaProcess([1, .1, .1, -0.8])
    coefs, locs = lpol2index(process.arcoefs)
    assert_almost_equal(coefs, [-.1, -.1, 0.8])
    assert_equal(locs, [0, 1, 2])
    ar = index2lpol(coefs, locs)
    assert_equal(process.arcoefs, ar)


class TestArmaProcess(TestCase):
    def test_empty_coeff(self):
        process = ArmaProcess()
        assert_equal(process.arcoefs, np.array([]))
        assert_equal(process.macoefs, np.array([]))

        process = ArmaProcess([1, -0.8])
        assert_equal(process.arcoefs, np.array([0.8]))
        assert_equal(process.macoefs, np.array([]))

        process = ArmaProcess(ma=[1, -0.8])
        assert_equal(process.arcoefs, np.array([]))
        assert_equal(process.macoefs, np.array([-0.8]))

    def test_from_coeff(self):
        ar = [1.8, -0.9]
        ma = [0.3]
        process = ArmaProcess.from_coeffs(np.array(ar), np.array(ma))

        ar.insert(0, -1)
        ma.insert(0, 1)
        ar_p = -1 * np.array(ar)
        ma_p = ma
        process_direct = ArmaProcess(ar_p, ma_p)

        assert_equal(process.arcoefs, process_direct.arcoefs)
        assert_equal(process.macoefs, process_direct.macoefs)
        assert_equal(process.nobs, process_direct.nobs)
        assert_equal(process.maroots, process_direct.maroots)
        assert_equal(process.arroots, process_direct.arroots)
        assert_equal(process.isinvertible, process_direct.isinvertible)
        assert_equal(process.isstationary, process_direct.isstationary)

    def test_from_model(self):
        process = ArmaProcess([1, -.8], [1, .3], 1000)
        t = 1000
        rs = np.random.RandomState(12345)
        y = process.generate_sample(t, burnin=100, distrvs=rs.standard_normal)
        res = ARMA(y, (1, 1)).fit(disp=False)
        process_model = ArmaProcess.from_estimation(res)
        process_coef = ArmaProcess.from_coeffs(res.arparams, res.maparams, t)

        assert_equal(process_model.arcoefs, process_coef.arcoefs)
        assert_equal(process_model.macoefs, process_coef.macoefs)
        assert_equal(process_model.nobs, process_coef.nobs)
        assert_equal(process_model.isinvertible, process_coef.isinvertible)
        assert_equal(process_model.isstationary, process_coef.isstationary)

    def test_process_multiplication(self):
        process1 = ArmaProcess.from_coeffs([.9])
        process2 = ArmaProcess.from_coeffs([.7])
        process3 = process1 * process2
        assert_equal(process3.arcoefs, np.array([1.6, -0.7 * 0.9]))
        assert_equal(process3.macoefs, np.array([]))

        process1 = ArmaProcess.from_coeffs([.9], [.2])
        process2 = ArmaProcess.from_coeffs([.7])
        process3 = process1 * process2

        assert_equal(process3.arcoefs, np.array([1.6, -0.7 * 0.9]))
        assert_equal(process3.macoefs, np.array([0.2]))

        process1 = ArmaProcess.from_coeffs([.9], [.2])
        process2 = process1 * (np.array([1.0, -0.7]), np.array([1.0]))
        assert_equal(process2.arcoefs, np.array([1.6, -0.7 * 0.9]))

        assert_raises(TypeError, process1.__mul__, [3])

    @pytest.mark.skipif(NP16, reason='numpy<1.7')
    def test_str_repr(self):
        process1 = ArmaProcess.from_coeffs([.9], [.2])
        out = process1.__str__()
        print(out)
        assert_(out.find('AR: [1.0, -0.9]') != -1)
        assert_(out.find('MA: [1.0, 0.2]') != -1)

        out = process1.__repr__()
        assert_(out.find('nobs=100') != -1)
        assert_(out.find('at ' + str(hex(id(process1)))) != -1)

    def test_acf(self):
        process1 = ArmaProcess.from_coeffs([.9])
        acf = process1.acf(10)
        expected = np.array(0.9) ** np.arange(10.0)
        assert_array_almost_equal(acf, expected)

        acf = process1.acf()
        assert_(acf.shape[0] == process1.nobs)

    def test_pacf(self):
        process1 = ArmaProcess.from_coeffs([.9])
        pacf = process1.pacf(10)
        expected = np.array([1, 0.9] + [0] * 8)
        assert_array_almost_equal(pacf, expected)

        pacf = process1.pacf()
        assert_(pacf.shape[0] == process1.nobs)

    def test_isstationary(self):
        process1 = ArmaProcess.from_coeffs([1.1])
        assert_equal(process1.isstationary, False)

        process1 = ArmaProcess.from_coeffs([1.8, -0.9])
        assert_equal(process1.isstationary, True)

        process1 = ArmaProcess.from_coeffs([1.5, -0.5])
        print(np.abs(process1.arroots))
        assert_equal(process1.isstationary, False)

    def test_arma2ar(self):
        process1 = ArmaProcess.from_coeffs([], [0.8])
        vals = process1.arma2ar(100)
        assert_almost_equal(vals, (-0.8) ** np.arange(100.0))

    def test_invertroots(self):
        process1 = ArmaProcess.from_coeffs([], [2.5])
        process2 = process1.invertroots(True)
        assert_almost_equal(process2.ma, np.array([1.0, 0.4]))

        process1 = ArmaProcess.from_coeffs([], [0.4])
        process2 = process1.invertroots(True)
        assert_almost_equal(process2.ma, np.array([1.0, 0.4]))

        process1 = ArmaProcess.from_coeffs([], [2.5])
        roots, invertable = process1.invertroots(False)
        assert_equal(invertable, False)
        assert_almost_equal(roots, np.array([1, 0.4]))

    def test_generate_sample(self):
        process = ArmaProcess.from_coeffs([0.9])
        np.random.seed(12345)
        sample = process.generate_sample()
        np.random.seed(12345)
        expected = np.random.randn(100)
        for i in range(1, 100):
            expected[i] = 0.9 * expected[i - 1] + expected[i]
        assert_almost_equal(sample, expected)

        process = ArmaProcess.from_coeffs([1.6, -0.9])
        np.random.seed(12345)
        sample = process.generate_sample()
        np.random.seed(12345)
        expected = np.random.randn(100)
        expected[1] = 1.6 * expected[0] + expected[1]
        for i in range(2, 100):
            expected[i] = 1.6 * expected[i - 1] - 0.9 * expected[i - 2] + expected[i]
        assert_almost_equal(sample, expected)

        process = ArmaProcess.from_coeffs([1.6, -0.9])
        np.random.seed(12345)
        sample = process.generate_sample(burnin=100)
        np.random.seed(12345)
        expected = np.random.randn(200)
        expected[1] = 1.6 * expected[0] + expected[1]
        for i in range(2, 200):
            expected[i] = 1.6 * expected[i - 1] - 0.9 * expected[i - 2] + expected[i]
        assert_almost_equal(sample, expected[100:])


        np.random.seed(12345)
        sample = process.generate_sample(nsample=(100,5))
        assert_equal(sample.shape, (100,5))

    def test_impulse_response(self):
        process = ArmaProcess.from_coeffs([0.9])
        ir = process.impulse_response(10)
        assert_almost_equal(ir, 0.9 ** np.arange(10))

    def test_periodogram(self):
        process = ArmaProcess()
        pg = process.periodogram()
        assert_almost_equal(pg[0], np.linspace(0,np.pi,100,False))
        assert_almost_equal(pg[1], np.sqrt(2 / np.pi) / 2 * np.ones(100))

