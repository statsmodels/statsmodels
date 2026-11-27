from statsmodels.compat.pandas import QUARTER_END

import datetime as dt
import re
import warnings

import numpy as np
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
)
import pandas as pd
import pytest

from statsmodels.sandbox.tsa.fftarma import ArmaFft
from statsmodels.tools.docstring import NumpyDocString
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import (
    ArmaProcess,
    ar2arma,
    arma2ar,
    arma2ma,
    arma_acf,
    arma_acovf,
    arma_generate_sample,
    arma_impulse_response,
    arma_pacf,
    arma_periodogram,
    deconvolve,
    index2lpol,
    lpol2index,
    lpol_fiar,
    lpol_fima,
    lpol_sdiff,
)
from statsmodels.tsa.tests.results import results_arma_acf
from statsmodels.tsa.tests.results.results_process import armarep  # benchmarkdata

arlist = [
    [1.0],
    [1, -0.9],  # ma representation will need many terms to get high precision
    [1, 0.9],
    [1, -0.9, 0.3],
]

malist = [[1.0], [1, 0.9], [1, -0.9], [1, 0.9, -0.3]]

DECIMAL_4 = 4


def test_arma_acovf():
    # Check for specific AR(1)
    N = 20
    phi = 0.9
    sigma = 1
    # rep 1: from module function
    rep1 = arma_acovf([1, -phi], [1], N)
    # rep 2: manually
    rep2 = [1.0 * sigma * phi**i / (1 - phi**2) for i in range(N)]
    assert_allclose(rep1, rep2)


def test_arma_acovf_persistent():
    # Test arma_acovf in case where there is a near-unit root.
    # .999 is high enough to trigger the "while ir[-1] > 5*1e-5:" clause,
    # but not high enough to trigger the "nobs_ir > 50000" clause.
    ar = np.array([1, -0.9995])
    ma = np.array([1])
    process = ArmaProcess(ar, ma)
    res = process.acovf(10)

    # Theoretical variance sig2 given by:
    # sig2 = .9995**2 * sig2 + 1
    sig2 = 1 / (1 - 0.9995**2)

    corrs = 0.9995 ** np.arange(10)
    expected = sig2 * corrs
    assert_equal(res.ndim, 1)
    assert_allclose(res, expected)


def test_arma_acf():
    # Check for specific AR(1)
    N = 20
    phi = 0.9
    sigma = 1
    # rep 1: from module function
    rep1 = arma_acf([1, -phi], [1], N)
    # rep 2: manually
    acovf = np.array([1.0 * sigma * phi**i / (1 - phi**2) for i in range(N)])
    rep2 = acovf / (1.0 / (1 - phi**2))
    assert_allclose(rep1, rep2)


def test_arma_acf_compare_R_ARMAacf():
    # Test specific cases against output from R's ARMAacf
    bd_example_3_3_2 = arma_acf([1, -1, 0.25], [1, 1])
    assert_allclose(bd_example_3_3_2, results_arma_acf.bd_example_3_3_2)
    example_1 = arma_acf([1, -1, 0.25], [1, 1, 0.2])
    assert_allclose(example_1, results_arma_acf.custom_example_1)
    example_2 = arma_acf([1, -1, 0.25], [1, 1, 0.2, 0.3])
    assert_allclose(example_2, results_arma_acf.custom_example_2)
    example_3 = arma_acf([1, -1, 0.25], [1, 1, 0.2, 0.3, -0.35])
    assert_allclose(example_3, results_arma_acf.custom_example_3)
    example_4 = arma_acf([1, -1, 0.25], [1, 1, 0.2, 0.3, -0.35, -0.1])
    assert_allclose(example_4, results_arma_acf.custom_example_4)
    example_5 = arma_acf([1, -1, 0.25, -0.1], [1, 1, 0.2])
    assert_allclose(example_5, results_arma_acf.custom_example_5)
    example_6 = arma_acf([1, -1, 0.25, -0.1, 0.05], [1, 1, 0.2])
    assert_allclose(example_6, results_arma_acf.custom_example_6)
    example_7 = arma_acf([1, -1, 0.25, -0.1, 0.05, -0.02], [1, 1, 0.2])
    assert_allclose(example_7, results_arma_acf.custom_example_7)


def test_arma_acov_compare_theoretical_arma_acov():
    # Test against the older version of this function, which used a different
    # approach that nicely shows the theoretical relationship
    # See GH:5324 when this was removed for full version of the function
    # including documentation and inline comments

    def arma_acovf_historical(ar, ma, nobs=10):
        if np.abs(np.sum(ar) - 1) > 0.9:
            nobs_ir = max(1000, 2 * nobs)
        else:
            nobs_ir = max(100, 2 * nobs)
        ir = arma_impulse_response(ar, ma, leads=nobs_ir)
        while ir[-1] > 5 * 1e-5:
            nobs_ir *= 10
            ir = arma_impulse_response(ar, ma, leads=nobs_ir)
        if nobs_ir > 50000 and nobs < 1001:
            end = len(ir)
            acovf = np.array(
                [np.dot(ir[: end - nobs - t], ir[t : end - nobs]) for t in range(nobs)]
            )
        else:
            acovf = np.correlate(ir, ir, "full")[len(ir) - 1 :]
        return acovf[:nobs]

    assert_allclose(
        arma_acovf([1, -0.5], [1, 0.2]),
        arma_acovf_historical([1, -0.5], [1, 0.2]),
    )
    assert_allclose(
        arma_acovf([1, -0.99], [1, 0.2]),
        arma_acovf_historical([1, -0.99], [1, 0.2]),
    )


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
            yt += np.dot([0] * (q - t) + list(eta[max(0, t - q) : t]), ma)
        rep2.append(yt)
    return np.array(rep2[max(p, q) :])


@pytest.mark.parametrize("ar", arlist)
@pytest.mark.parametrize("ma", malist)
def test_arma_generate_sample(ar, ma):
    # Test that this generates a true ARMA process
    # (amounts to just a test that scipy.signal.lfilter does what we want)
    T = 100
    rg = np.random.RandomState(1234)
    eta = rg.standard_normal(T)

    # rep1: from module function
    rg2 = np.random.RandomState(1234)
    rep1 = arma_generate_sample(ar, ma, T, distrvs=rg2.standard_normal)
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


def test_fi_d_zero():
    # d == 0 is the identity operator: (1-L)^0 == 1, so the lag polynomial is
    # 1 followed by zeros.  The gammaln formula is indeterminate at lag zero
    # when d == 0 (inf - inf) and used to produce a NaN coefficient in
    # lpol_fima and a RuntimeWarning in both helpers.
    n = 10
    expected = np.r_[1.0, np.zeros(n - 1)]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert_array_almost_equal(lpol_fima(0.0, n=n), expected, 14)
        assert_array_almost_equal(lpol_fiar(0.0, n=n), expected, 14)
        # n == 1 only has the lag-zero coefficient
        assert_array_almost_equal(lpol_fima(0.0, n=1), [1.0], 14)
        assert_array_almost_equal(lpol_fiar(0.0, n=1), [1.0], 14)
        # nonzero d is unaffected; the lag-zero coefficient is still exactly 1
        assert lpol_fima(0.4, n=n)[0] == 1.0
        assert lpol_fiar(-0.3, n=n)[0] == 1.0


@pytest.mark.parametrize("d", [-0.45, -0.25, -1e-8, 0.0, 1e-8, 0.3, 0.45])
def test_fi_leading_terms(d):
    # (1 - L)^(-d) = 1 + d L + d (d + 1) / 2 L^2 + d (d + 1) (d + 2) / 6 L^3 + ...
    # (1 - L)^d = 1 - d L + d (d - 1) / 2 L^2 - d (d - 1) (d - 2) / 6 L^3 + ...
    # the signs of the coefficients for negative d are the reverse of the signs
    # for positive d
    ma = [1, d, d * (d + 1) / 2, d * (d + 1) * (d + 2) / 6]
    ar = [1, -d, d * (d - 1) / 2, -d * (d - 1) * (d - 2) / 6]
    assert_allclose(lpol_fima(d, n=4), ma, rtol=1e-8, atol=1e-15)
    assert_allclose(lpol_fiar(d, n=4), ar, rtol=1e-8, atol=1e-15)


@pytest.mark.parametrize("func", [lpol_fima, lpol_fiar])
@pytest.mark.parametrize("d", [-0.3, 0.0, 0.3])
def test_fi_short(func, d):
    assert func(d, n=0).shape == (0,)
    assert_allclose(func(d, n=1), [1.0])


def test_arma_impulse_response():
    arrep = arma_impulse_response(armarep.ma, armarep.ar, leads=21)[1:]
    marep = arma_impulse_response(armarep.ar, armarep.ma, leads=21)[1:]
    assert_array_almost_equal(armarep.marep.ravel(), marep, 14)
    # difference in sign convention to matlab for AR term
    assert_array_almost_equal(-armarep.arrep.ravel(), arrep, 14)


@pytest.mark.parametrize("ar", arlist)
@pytest.mark.parametrize("ma", malist)
def test_spectrum(ar, ma):
    nfreq = 20
    w = np.linspace(0, np.pi, nfreq, endpoint=False)

    arma = ArmaFft(ar, ma, 20)
    spdr, wr = arma.spdroots(w)
    spdp, wp = arma.spdpoly(w, 200)
    spdd, wd = arma.spddirect(nfreq * 2)
    assert_equal(w, wr)
    assert_equal(w, wp)
    assert_almost_equal(w, wd[:nfreq], decimal=14)
    assert_almost_equal(
        spdr,
        spdd[:nfreq],
        decimal=7,
        err_msg=f"spdr spdd not equal for {ar}, {ma}",
    )
    assert_almost_equal(
        spdr,
        spdp,
        decimal=7,
        err_msg=f"spdr spdp not equal for {ar}, {ma}",
    )


@pytest.mark.parametrize("ar", arlist)
@pytest.mark.parametrize("ma", malist)
def test_armafft(ar, ma):
    # test other methods
    arma = ArmaFft(ar, ma, 20)
    ac1 = arma.invpowerspd(1024)[:10]
    ac2 = arma.acovf(10)[:10]
    assert_allclose(ac1, ac2, atol=1e-15, err_msg=f"acovf not equal for {ar}, {ma}")


def test_lpol2index_index2lpol():
    process = ArmaProcess([1, 0, 0, -0.8])
    coefs, locs = lpol2index(process.arcoefs)
    assert_almost_equal(coefs, [0.8])
    assert_equal(locs, [2])

    process = ArmaProcess([1, 0.1, 0.1, -0.8])
    coefs, locs = lpol2index(process.arcoefs)
    assert_almost_equal(coefs, [-0.1, -0.1, 0.8])
    assert_equal(locs, [0, 1, 2])
    ar = index2lpol(coefs, locs)
    assert_equal(process.arcoefs, ar)


@pytest.mark.parametrize("form", ["list", "tuple", "series"])
@pytest.mark.parametrize("arg", ["coeffs", "index"])
def test_index2lpol_input_forms(arg, form):
    # A tuple index used to raise an IndexError
    kwargs = {"coeffs": [0.8, -0.1], "index": [2, 5]}
    kwargs[arg] = _coefficient_form(kwargs[arg], form)
    res = index2lpol(**kwargs)
    assert_allclose(res, [0.0, 0.0, 0.8, 0.0, 0.0, -0.1], rtol=1e-15)


class TestArmaProcess:
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

    def test_from_roots(self):
        ar = [1.8, -0.9]
        ma = [0.3]

        ar.insert(0, -1)
        ma.insert(0, 1)
        ar_p = -1 * np.array(ar)
        ma_p = ma
        process_direct = ArmaProcess(ar_p, ma_p)

        process = ArmaProcess.from_roots(
            np.array(process_direct.maroots), np.array(process_direct.arroots)
        )

        assert_almost_equal(process.arcoefs, process_direct.arcoefs)
        assert_almost_equal(process.macoefs, process_direct.macoefs)
        assert_almost_equal(process.nobs, process_direct.nobs)
        assert_almost_equal(process.maroots, process_direct.maroots)
        assert_almost_equal(process.arroots, process_direct.arroots)
        assert_almost_equal(process.isinvertible, process_direct.isinvertible)
        assert_almost_equal(process.isstationary, process_direct.isstationary)

        process_direct = ArmaProcess(ar=ar_p)
        process = ArmaProcess.from_roots(arroots=np.array(process_direct.arroots))

        assert_almost_equal(process.arcoefs, process_direct.arcoefs)
        assert_almost_equal(process.macoefs, process_direct.macoefs)
        assert_almost_equal(process.nobs, process_direct.nobs)
        assert_almost_equal(process.maroots, process_direct.maroots)
        assert_almost_equal(process.arroots, process_direct.arroots)
        assert_almost_equal(process.isinvertible, process_direct.isinvertible)
        assert_almost_equal(process.isstationary, process_direct.isstationary)

        process_direct = ArmaProcess(ma=ma_p)
        process = ArmaProcess.from_roots(maroots=np.array(process_direct.maroots))

        assert_almost_equal(process.arcoefs, process_direct.arcoefs)
        assert_almost_equal(process.macoefs, process_direct.macoefs)
        assert_almost_equal(process.nobs, process_direct.nobs)
        assert_almost_equal(process.maroots, process_direct.maroots)
        assert_almost_equal(process.arroots, process_direct.arroots)
        assert_almost_equal(process.isinvertible, process_direct.isinvertible)
        assert_almost_equal(process.isstationary, process_direct.isstationary)

        process_direct = ArmaProcess()
        process = ArmaProcess.from_roots()

        assert_almost_equal(process.arcoefs, process_direct.arcoefs)
        assert_almost_equal(process.macoefs, process_direct.macoefs)
        assert_almost_equal(process.nobs, process_direct.nobs)
        assert_almost_equal(process.maroots, process_direct.maroots)
        assert_almost_equal(process.arroots, process_direct.arroots)
        assert_almost_equal(process.isinvertible, process_direct.isinvertible)
        assert_almost_equal(process.isstationary, process_direct.isstationary)

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

    def test_process_multiplication(self):
        process1 = ArmaProcess.from_coeffs([0.9])
        process2 = ArmaProcess.from_coeffs([0.7])
        process3 = process1 * process2
        assert_equal(process3.arcoefs, np.array([1.6, -0.7 * 0.9]))
        assert_equal(process3.macoefs, np.array([]))

        process1 = ArmaProcess.from_coeffs([0.9], [0.2])
        process2 = ArmaProcess.from_coeffs([0.7])
        process3 = process1 * process2

        assert_equal(process3.arcoefs, np.array([1.6, -0.7 * 0.9]))
        assert_equal(process3.macoefs, np.array([0.2]))

        process1 = ArmaProcess.from_coeffs([0.9], [0.2])
        process2 = process1 * (np.array([1.0, -0.7]), np.array([1.0]))
        assert_equal(process2.arcoefs, np.array([1.6, -0.7 * 0.9]))

        with pytest.raises(TypeError):
            process1.__mul__([3])

    def test_str_repr(self):
        process1 = ArmaProcess.from_coeffs([0.9], [0.2])
        out = process1.__str__()
        print(out)
        assert out.find("AR: [1.0, -0.9]") != -1
        assert out.find("MA: [1.0, 0.2]") != -1

        out = process1.__repr__()
        assert out.find("nobs=100") != -1
        assert out.find("at " + str(hex(id(process1)))) != -1

    def test_acf(self):
        process1 = ArmaProcess.from_coeffs([0.9])
        acf = process1.acf(10)
        expected = np.array(0.9) ** np.arange(10.0)
        assert_array_almost_equal(acf, expected)

        acf = process1.acf()
        assert acf.shape[0] == process1.nobs

    def test_pacf(self):
        process1 = ArmaProcess.from_coeffs([0.9])
        pacf = process1.pacf(10)
        expected = np.array([1, 0.9] + [0] * 8)
        assert_array_almost_equal(pacf, expected)

        pacf = process1.pacf()
        assert pacf.shape[0] == process1.nobs

    def test_arma2ma_arma2ar_default_lags(self):
        # GH 4464
        # Reference values for phi=0.9, theta=0.2 (R 4.6.1, stats::ARMAtoMA):
        #   c(1, ARMAtoMA(ar = 0.9, ma = 0.2, lag.max = 3))
        #       -> 1, 1.1, 0.99, 0.891
        #   c(1, ARMAtoMA(ar = -0.2, ma = -0.9, lag.max = 3))
        #       -> 1, -1.1, 0.22, -0.044
        # They are the expansions of (1 + 0.2L) / (1 - 0.9L) and
        # (1 - 0.9L) / (1 + 0.2L)
        ma_expected = [1, 1.1, 0.99, 0.891]
        ar_expected = [1, -1.1, 0.22, -0.044]

        process1 = ArmaProcess.from_coeffs([0.9], [0.2])
        assert process1.nobs == 100
        ma_rep = process1.arma2ma()
        assert ma_rep.shape == (100,)
        assert_allclose(ma_rep[:4], ma_expected, rtol=1e-12)
        assert_allclose(process1.arma2ma(10), ma_rep[:10])
        ar_rep = process1.arma2ar()
        assert ar_rep.shape == (100,)
        assert_allclose(ar_rep[:4], ar_expected, rtol=1e-12)
        assert_allclose(process1.arma2ar(10), ar_rep[:10])

        # The default is nobs, not the default of the module level functions
        process2 = ArmaProcess.from_coeffs([0.9], [0.2], nobs=37)
        assert process2.nobs == 37
        ma_rep = process2.arma2ma()
        assert ma_rep.shape == (37,)
        assert_allclose(ma_rep[:4], ma_expected, rtol=1e-12)
        assert_allclose(ma_rep, arma2ma(process2.ar, process2.ma, lags=37))
        ar_rep = process2.arma2ar()
        assert ar_rep.shape == (37,)
        assert_allclose(ar_rep[:4], ar_expected, rtol=1e-12)
        assert_allclose(ar_rep, arma2ar(process2.ar, process2.ma, lags=37))

    LAGS_METHODS = [
        "acovf",
        "acf",
        "pacf",
        "periodogram",
        "impulse_response",
        "arma2ma",
        "arma2ar",
    ]

    @pytest.mark.parametrize("method", LAGS_METHODS)
    @pytest.mark.parametrize("value", [0, -1])
    def test_lags_not_positive(self, method, value):
        # 0 selected nobs, as omitting the value does, and a negative value
        # raised a different error in each method
        process = ArmaProcess.from_coeffs([0.6], [0.3], nobs=7)
        with pytest.raises(ValueError, match="must be a positive integer, got"):
            getattr(process, method)(value)

    @pytest.mark.parametrize("method", LAGS_METHODS)
    def test_lags_default(self, method):
        process = ArmaProcess.from_coeffs([0.6], [0.3], nobs=7)
        kwargs = {"normalization": "2pi"} if method == "periodogram" else {}
        default = np.asarray(getattr(process, method)(**kwargs))
        assert default.shape[-1] == 7
        # Any integer value, including an integral float, is accepted
        for value in (7, np.int64(7), 7.0):
            res = np.asarray(getattr(process, method)(value, **kwargs))
            assert_allclose(res, default, rtol=1e-15)
        for value in (7.5, True):
            with pytest.raises(TypeError, match="must be integer_like"):
                getattr(process, method)(value, **kwargs)

    def test_lags_one(self):
        # One value, against the closed forms of an ARMA(1, 1) with unit
        # innovation variance
        phi, theta = 0.6, 0.3
        process = ArmaProcess.from_coeffs([phi], [theta], nobs=7)
        gamma0 = (1 + 2 * phi * theta + theta**2) / (1 - phi**2)
        assert_allclose(process.acovf(1), [gamma0], rtol=1e-12)
        for method in ("acf", "pacf", "impulse_response", "arma2ma", "arma2ar"):
            assert_allclose(getattr(process, method)(1), [1.0], rtol=1e-15)
        freq, sd = process.periodogram(1, normalization="sqrt(2pi)")
        assert_allclose(freq, [0.0], rtol=1e-15)
        expected = (1 + theta) ** 2 / (1 - phi) ** 2 / np.sqrt(2 * np.pi)
        assert_allclose(sd, [expected], rtol=1e-12)

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
        rs = np.random.RandomState(12345)
        sample = process.generate_sample(distrvs=rs.standard_normal)
        rs = np.random.RandomState(12345)
        expected = rs.randn(100)
        for i in range(1, 100):
            expected[i] = 0.9 * expected[i - 1] + expected[i]
        assert_almost_equal(sample, expected)

        process = ArmaProcess.from_coeffs([1.6, -0.9])
        rs = np.random.RandomState(12345)
        sample = process.generate_sample(distrvs=rs.standard_normal)
        rs = np.random.RandomState(12345)
        expected = rs.randn(100)
        expected[1] = 1.6 * expected[0] + expected[1]
        for i in range(2, 100):
            expected[i] = 1.6 * expected[i - 1] - 0.9 * expected[i - 2] + expected[i]
        assert_almost_equal(sample, expected)

        process = ArmaProcess.from_coeffs([1.6, -0.9])
        rs = np.random.RandomState(12345)
        sample = process.generate_sample(burnin=100, distrvs=rs.standard_normal)
        rs = np.random.RandomState(12345)
        expected = rs.randn(200)
        expected[1] = 1.6 * expected[0] + expected[1]
        for i in range(2, 200):
            expected[i] = 1.6 * expected[i - 1] - 0.9 * expected[i - 2] + expected[i]
        assert_almost_equal(sample, expected[100:])

        rs = np.random.RandomState(12345)
        sample = process.generate_sample(nsample=(100, 5), distrvs=rs.standard_normal)
        assert_equal(sample.shape, (100, 5))

    @pytest.mark.parametrize(
        "make",
        [
            lambda nobs: ArmaProcess([1.0, -0.9], nobs=nobs),
            lambda nobs: ArmaProcess.from_coeffs([0.9], nobs=nobs),
            lambda nobs: ArmaProcess.from_roots(arroots=[1 / 0.9], nobs=nobs),
        ],
        ids=["init", "from_coeffs", "from_roots"],
    )
    @pytest.mark.parametrize("nsample", [{}, {"nsample": None}])
    def test_generate_sample_default_nobs(self, make, nsample):
        # The default sample size is nobs. It used to be 100, whatever nobs
        process = make(37)
        rng = np.random.default_rng(12345)
        sample = process.generate_sample(distrvs=rng.standard_normal, **nsample)
        # The AR(1) recursion with the same innovations
        expected = np.random.default_rng(12345).standard_normal(37)
        for i in range(1, 37):
            expected[i] += 0.9 * expected[i - 1]
        assert sample.shape == (37,)
        assert_allclose(sample, expected, rtol=1e-12)

        rng = np.random.default_rng(12345)
        sample = process.generate_sample(
            burnin=5, distrvs=rng.standard_normal, **nsample
        )
        expected = np.random.default_rng(12345).standard_normal(42)
        for i in range(1, 42):
            expected[i] += 0.9 * expected[i - 1]
        assert sample.shape == (37,)
        assert_allclose(sample, expected[5:], rtol=1e-12)

    def test_generate_sample_default_nobs_from_estimation(self):
        # from_estimation takes nobs from the results
        rng = np.random.default_rng(0)
        data = arma_generate_sample([1.0, -0.6], [1.0, 0.3], 80, distrvs=rng.normal)
        res = ARIMA(data, order=(1, 0, 1), trend="n").fit()
        process = ArmaProcess.from_estimation(res)
        sample = process.generate_sample(distrvs=np.random.default_rng(1).normal)
        expected = arma_generate_sample(
            res.polynomial_reduced_ar,
            res.polynomial_reduced_ma,
            80,
            distrvs=np.random.default_rng(1).normal,
        )
        assert sample.shape == (80,)
        assert_allclose(sample, expected, rtol=1e-15)

    def test_impulse_response(self):
        process = ArmaProcess.from_coeffs([0.9])
        ir = process.impulse_response(10)
        assert_almost_equal(ir, 0.9 ** np.arange(10))

    def test_periodogram(self):
        process = ArmaProcess()
        pg = process.periodogram(normalization="sqrt(2pi)")
        assert_almost_equal(pg[0], np.linspace(0, np.pi, 100, False))
        assert_almost_equal(pg[1], np.sqrt(2 / np.pi) / 2 * np.ones(100))


@pytest.mark.parametrize("d", [0, 1])
@pytest.mark.parametrize("seasonal", [True])
def test_from_estimation(d, seasonal):
    ar = [0.8] if not seasonal else [0.8, 0, 0, 0.2, -0.16]
    ma = [0.4] if not seasonal else [0.4, 0, 0, 0.2, -0.08]
    ap = ArmaProcess.from_coeffs(ar, ma, 500)
    idx = pd.date_range(dt.datetime(1900, 1, 1), periods=500, freq=QUARTER_END)
    rs = np.random.RandomState(12345111)
    data = ap.generate_sample(500, distrvs=rs.standard_normal)
    if d == 1:
        data = np.cumsum(data)
    data = pd.Series(data, index=idx)
    seasonal_order = (1, 0, 1, 4) if seasonal else None
    mod = ARIMA(data, order=(1, d, 1), seasonal_order=seasonal_order)
    res = mod.fit()
    ap_from = ArmaProcess.from_estimation(res)
    shape = (5,) if seasonal else (1,)
    assert ap_from.arcoefs.shape == shape
    assert ap_from.macoefs.shape == shape


def test_from_estimation_nobs():
    # nobs=0 used to select the nobs of the results, as omitting it does
    rng = np.random.default_rng(0)
    data = arma_generate_sample([1.0, -0.6], [1.0, 0.3], 80, distrvs=rng.normal)
    res = ARIMA(data, order=(1, 0, 1), trend="n").fit()
    assert res.nobs == 80
    assert ArmaProcess.from_estimation(res).nobs == 80
    for nobs in (0, 7):
        process = ArmaProcess.from_estimation(res, nobs=nobs)
        assert process.nobs == nobs
        assert_allclose(process.ar, res.polynomial_reduced_ar, rtol=1e-15)
        assert_allclose(process.ma, res.polynomial_reduced_ma, rtol=1e-15)


@pytest.mark.parametrize("s", [4, 12])
def test_lpol_sdiff(s):
    # (1 - L^s) has coefficient 1 at lag 0, -1 at lag s, and 0 elsewhere.
    coeffs = lpol_sdiff(s)
    expected = np.zeros(s + 1)
    expected[0] = 1.0
    expected[s] = -1.0
    assert_array_equal(coeffs, expected)

    # Applying the polynomial as a filter must reproduce a direct seasonal
    # difference x_t - x_{t-s}, independently of how the coefficients were
    # derived. np.convolve(coeffs, x, "full")[t] == sum_j coeffs[j] * x[t-j]
    # by the definition of discrete convolution, with x implicitly
    # zero-padded outside its bounds.
    rng = np.random.default_rng(0)
    x = rng.standard_normal(30)
    conv = np.convolve(coeffs, x, mode="full")
    assert_allclose(conv[s : len(x)], x[s:] - x[:-s])


def test_deconvolve_roundtrip():
    # Polynomial division has an exact answer: if num == den * quot exactly,
    # deconvolve must recover quot exactly with a zero remainder.
    den = np.array([1.0, -0.5, 0.25])
    quot_true = np.array([2.0, -1.0, 0.5, 3.0])
    num = np.convolve(den, quot_true)
    quot, rem = deconvolve(num, den)
    assert_allclose(quot, quot_true, atol=1e-10)
    assert_allclose(rem, np.zeros_like(num), atol=1e-10)


def test_deconvolve_den_longer_than_num():
    # When den is longer than num and n is not given, deconvolve reports
    # the trivial "quotient is empty, remainder is the whole signal" result.
    num = np.array([1.0, 2.0])
    den = np.array([1.0, 2.0, 3.0, 4.0])
    quot, rem = deconvolve(num, den)
    assert len(quot) == 0
    assert_array_equal(rem, num)


@pytest.mark.parametrize(
    "num_dtype, den_dtype, expected",
    [
        (int, int, np.float64),
        (np.float32, np.float32, np.float64),
        (np.float64, np.float64, np.float64),
        (np.complex64, np.complex64, np.complex128),
        (np.complex128, np.complex128, np.complex128),
        (np.float64, np.complex128, np.complex128),
    ],
)
@pytest.mark.parametrize("form", ["ndarray", "list"])
def test_deconvolve_den_longer_than_num_empty_ndarray(
    num_dtype, den_dtype, expected, form
):
    # The empty quotient was a list, and the remainder was num with its
    # dtype, the same object for an ndarray. Both now have the dtype of the
    # other case, n given, and the remainder is a copy.
    num = np.array([1, 2], dtype=num_dtype)
    den = np.array([1, 2, 3, 4], dtype=den_dtype)
    num_arg = num if form == "ndarray" else num.tolist()
    quot, rem = deconvolve(num_arg, den)
    assert isinstance(quot, np.ndarray)
    assert quot.shape == np.array([]).shape
    assert quot.dtype == expected
    assert isinstance(rem, np.ndarray)
    assert rem.dtype == expected
    assert_array_equal(rem, num)
    assert not np.shares_memory(rem, num)
    # The dtypes of the other case
    quot_n, rem_n = deconvolve(num_arg, den, n=1)
    assert quot_n.dtype == expected
    assert rem_n.dtype == expected


def test_ar2arma_recovers_correctly_specified_ar():
    # When the fitted ARMA(p, 1) nests the true AR(p-1) process exactly
    # (trivial MA), ar2arma should recover it (near) exactly, and the
    # ordinary impulse response of the approximation -- computed
    # independently via ArmaProcess -- should match the true process's own
    # impulse response closely.
    n = 40
    for ar_true, p in [([1.0, -0.8], 2), ([1.0, -0.5, 0.25], 3)]:
        ar_des = np.zeros(n)
        ar_des[: len(ar_true)] = ar_true
        ar_app, ma_app, _ = ar2arma(ar_des, p=p, q=1, n=n)
        assert_allclose(ar_app, ar_true, atol=1e-5)
        assert_allclose(ma_app, [1.0], atol=1e-5)

        true_irf = ArmaProcess(ar_true, [1.0]).impulse_response(leads=n)
        app_irf = ArmaProcess(ar_app, ma_app).impulse_response(leads=n)
        assert_allclose(app_irf, true_irf, atol=1e-4)


def test_ar2arma_approximates_higher_order_ar():
    # When under-parameterized (fitting an ARMA(1, 1) to an AR(3) process),
    # ar2arma cannot recover the true process exactly, but the impulse
    # response of the fit -- independently computed via ArmaProcess --
    # should still closely track the true process's impulse response over a
    # reasonable horizon.
    n = 40
    ar_true = [1.0, -0.6, 0.2, -0.1]
    ar_des = np.zeros(n)
    ar_des[: len(ar_true)] = ar_true
    ar_app, ma_app, _ = ar2arma(ar_des, p=2, q=2, n=n)

    true_irf = ArmaProcess(ar_true, [1.0]).impulse_response(leads=15)
    app_irf = ArmaProcess(ar_app, ma_app).impulse_response(leads=15)
    assert_allclose(app_irf, true_irf, atol=0.1)


def test_arma_acovf_nobs_validation():
    # a negative nobs used to return a silently truncated acovf on the
    # innovations path and leak a bare numpy error on the white-noise path
    with pytest.raises(ValueError, match="nobs must be a positive integer"):
        arma_acovf([1.0, -0.5], [1.0], nobs=-1)
    with pytest.raises(ValueError, match="nobs must be a positive integer"):
        arma_acovf([1.0], [1.0], nobs=-1)
    with pytest.raises(ValueError, match="nobs must be a positive integer"):
        arma_acf([1.0, -0.5], [1.0], lags=0)


LAGS_FUNCTIONS = [
    (arma_pacf, "lags"),
    (arma_impulse_response, "leads"),
    (arma2ma, "lags"),
    (arma2ar, "lags"),
]


@pytest.mark.parametrize("value", [0, -1, np.int64(0)])
@pytest.mark.parametrize("func, name", LAGS_FUNCTIONS)
def test_arma_functions_lags_not_positive(func, name, value):
    # 0 raised an IndexError, and a negative value a numpy error about
    # negative dimensions
    with pytest.raises(ValueError, match=f"{name} must be a positive integer, got"):
        func([1.0, -0.6], [1.0, 0.3], **{name: value})


@pytest.mark.parametrize("func, name", LAGS_FUNCTIONS)
def test_arma_functions_lags_integer_like(func, name):
    ar, ma = [1.0, -0.6], [1.0, 0.3]
    # Lag zero of each is 1
    assert_allclose(func(ar, ma, **{name: 1}), [1.0], rtol=1e-15)
    # An integral float used to raise a TypeError from numpy
    expected = func(ar, ma, **{name: 7})
    for value in (np.int64(7), 7.0):
        assert_allclose(func(ar, ma, **{name: value}), expected, rtol=1e-15)
    for value in (7.5, True):
        with pytest.raises(TypeError, match="must be integer_like"):
            func(ar, ma, **{name: value})


def test_arma_acovf_negative_sigma2():
    with pytest.raises(ValueError, match="positive innovation variance"):
        arma_acovf([1.0, -0.5], [1.0], nobs=5, sigma2=-1.0)


@pytest.mark.parametrize("nobs", [1, 2, 6])
def test_arma11_acovf_closed_form(nobs):
    # ARMA(1, 1), see Brockwell and Davis (2009), Section 3.3:
    # gamma(0) = s2 (1 + 2 phi theta + theta^2) / (1 - phi^2),
    # gamma(1) = s2 (1 + phi theta) (phi + theta) / (1 - phi^2) and
    # gamma(k) = phi gamma(k - 1) for k > 1. The first two are solved for
    # jointly, so nobs = 1 and 2 return part of that solution.
    phi, theta, sigma2 = 0.6, 0.3, 2.5
    g0 = sigma2 * (1 + 2 * phi * theta + theta**2) / (1 - phi**2)
    g1 = sigma2 * (1 + phi * theta) * (phi + theta) / (1 - phi**2)
    expected = np.r_[g0, g1 * phi ** np.arange(nobs - 1)]
    acovf = arma_acovf([1.0, -phi], [1.0, theta], nobs=nobs, sigma2=sigma2)
    assert_allclose(acovf, expected, rtol=1e-12)


def test_ma1_acf_pacf_closed_form():
    # MA(1), see Brockwell and Davis (2009), Sections 3.3 and 3.4:
    # rho(1) = theta / (1 + theta^2), rho(k) = 0 for k > 1, and the partial
    # autocorrelation of lag k is
    # -(-theta)^k (1 - theta^2) / (1 - theta^(2 (k + 1))).
    theta, lags = 0.5, 8
    k = np.arange(1, lags)
    expected_acf = np.r_[1.0, theta / (1 + theta**2), np.zeros(lags - 2)]
    pacf_k = -((-theta) ** k) * (1 - theta**2) / (1 - theta ** (2 * (k + 1)))
    expected_pacf = np.r_[1.0, pacf_k]
    process = ArmaProcess(ma=[1.0, theta])
    # atol: the autocorrelations beyond lag 1 are zero
    acf = arma_acf([1.0], [1.0, theta], lags)
    assert_allclose(acf, expected_acf, rtol=1e-12, atol=1e-15)
    assert_allclose(process.acf(lags), expected_acf, rtol=1e-12, atol=1e-15)
    assert_allclose(arma_pacf([1.0], [1.0, theta], lags), expected_pacf, rtol=1e-10)
    assert_allclose(process.pacf(lags), expected_pacf, rtol=1e-10)


def test_arma11_representations_closed_form():
    # (1 + theta L) / (1 - phi L) = 1 + sum_k (phi + theta) phi^(k - 1) L^k
    # (1 - phi L) / (1 + theta L) = 1 - sum_k (phi + theta) (-theta)^(k - 1) L^k
    phi, theta, n = 0.7, 0.4, 10
    k = np.arange(1, n)
    ma_rep = np.r_[1.0, (phi + theta) * phi ** (k - 1)]
    ar_rep = np.r_[1.0, -(phi + theta) * (-theta) ** (k - 1)]
    ar, ma = [1.0, -phi], [1.0, theta]
    process = ArmaProcess(ar, ma)
    assert_allclose(arma_impulse_response(ar, ma, leads=n), ma_rep, rtol=1e-12)
    assert_allclose(arma2ma(ar, ma, lags=n), ma_rep, rtol=1e-12)
    assert_allclose(arma2ar(ar, ma, lags=n), ar_rep, rtol=1e-12)
    assert_allclose(process.impulse_response(n), ma_rep, rtol=1e-12)
    assert_allclose(process.arma2ma(n), ma_rep, rtol=1e-12)
    assert_allclose(process.arma2ar(n), ar_rep, rtol=1e-12)


def test_lag_polynomial_identities():
    # ar(L) psi(L) = ma(L) and ma(L) pi(L) = ar(L) for the MA and AR
    # representations psi and pi, up to the truncation after n terms
    ar = np.array([1.0, -0.5, 0.25])
    ma = np.array([1.0, 0.4, -0.2])
    n = 12
    psi = arma2ma(ar, ma, lags=n)
    pi = arma2ar(ar, ma, lags=n)
    # atol: the coefficients after lag 2 are zero
    expected_ma = np.r_[ma, np.zeros(n - 3)]
    expected_ar = np.r_[ar, np.zeros(n - 3)]
    assert_allclose(np.convolve(ar, psi)[:n], expected_ma, rtol=1e-12, atol=1e-14)
    assert_allclose(np.convolve(ma, pi)[:n], expected_ar, rtol=1e-12, atol=1e-14)
    # The product of two processes multiplies their lag polynomials
    product = ArmaProcess(ar, ma) * ArmaProcess([1.0, 0.3], [1.0, -0.6])
    assert_allclose(product.ar, np.convolve(ar, [1.0, 0.3]), rtol=1e-15)
    assert_allclose(product.ma, np.convolve(ma, [1.0, -0.6]), rtol=1e-15)
    # (1 - z / 2) (1 + z / 4) = 1 - z / 4 - z^2 / 8 and 1 + z / 5
    process = ArmaProcess.from_roots(maroots=[-5.0], arroots=[2.0, -4.0])
    assert_allclose(process.ar, [1.0, -0.25, -0.125], rtol=1e-15)
    assert_allclose(process.ma, [1.0, 0.2], rtol=1e-15)
    assert process.isstationary
    assert process.isinvertible


def test_arma_periodogram_closed_form():
    # |ma(exp(-iw))|^2 = 1 + 2 theta cos(w) + theta^2 and
    # |ar(exp(-iw))|^2 = 1 - 2 phi cos(w) + phi^2, see the Notes for the
    # division by sqrt(2 pi)
    phi, theta = 0.5, 0.4
    w = np.linspace(0, np.pi, 7)
    freq, sd = arma_periodogram(
        [1.0, -phi], [1.0, theta], worN=w, normalization="sqrt(2pi)"
    )
    num = 1 + 2 * theta * np.cos(w) + theta**2
    den = 1 - 2 * phi * np.cos(w) + phi**2
    assert_allclose(freq, w, rtol=1e-15)
    assert_allclose(sd, num / den / np.sqrt(2 * np.pi), rtol=1e-12)
    process = ArmaProcess([1.0, -phi], [1.0, theta])
    freq, sd = process.periodogram(7, normalization="sqrt(2pi)")
    assert_allclose(freq, np.linspace(0, np.pi, 7, endpoint=False), rtol=1e-15)
    den = 1 - 2 * phi * np.cos(freq) + phi**2
    num = 1 + 2 * theta * np.cos(freq) + theta**2
    assert_allclose(sd, num / den / np.sqrt(2 * np.pi), rtol=1e-12)


# AR(1), MA(1), ARMA(1, 1) and ARMA(2, 2), all stationary
PERIODOGRAM_PROCESSES = [
    ([1.0, -0.5], [1.0]),
    ([1.0], [1.0, 0.4]),
    ([1.0, -0.5], [1.0, 0.4]),
    ([1.0, -0.75, 0.25], [1.0, 0.65, 0.35]),
]


@pytest.mark.parametrize("ar, ma", PERIODOGRAM_PROCESSES)
def test_arma_periodogram_normalization_2pi(ar, ma):
    # The spectral density of an ARMA process with unit innovation variance,
    # f(w) = |ma(exp(-iw))|^2 / |ar(exp(-iw))|^2 / (2 pi)
    w = np.linspace(0, np.pi, 9)
    z = np.exp(-1j * w)
    poly = np.polynomial.polynomial.polyval
    expected = np.abs(poly(z, ma)) ** 2 / np.abs(poly(z, ar)) ** 2 / (2 * np.pi)
    freq, sd = arma_periodogram(ar, ma, worN=w, normalization="2pi")
    assert_allclose(freq, w, rtol=1e-15)
    assert_allclose(sd, expected, rtol=1e-12)

    process = ArmaProcess(ar, ma)
    freq, sd = process.periodogram(8, normalization="2pi")
    assert_allclose(freq, np.linspace(0, np.pi, 8, endpoint=False), rtol=1e-15)
    z = np.exp(-1j * freq)
    expected = np.abs(poly(z, ma)) ** 2 / np.abs(poly(z, ar)) ** 2 / (2 * np.pi)
    assert_allclose(sd, expected, rtol=1e-12)


@pytest.mark.parametrize("ar, ma", PERIODOGRAM_PROCESSES)
def test_arma_periodogram_normalization_2pi_autocovariance(ar, ma):
    # The autocovariances are the Fourier coefficients of the spectral
    # density, gamma(k) = int_{-pi}^{pi} f(w) cos(k w) dw. The mean over n
    # equally spaced frequencies on [0, 2 pi) integrates the periodic,
    # smooth integrand with an error that decays geometrically in n.
    n = 512
    w, sd = arma_periodogram(ar, ma, worN=n, whole=True, normalization="2pi")
    lags = np.arange(5)
    gamma = 2 * np.pi * np.mean(sd * np.cos(np.outer(lags, w)), axis=1)
    # atol: the autocovariances of the MA(1) after lag 1 are zero
    assert_allclose(gamma, arma_acovf(ar, ma, nobs=5), rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("method", [False, True])
def test_arma_periodogram_normalization_default_deprecated(method):
    # Omitting normalization keeps the division by sqrt(2 pi), with a warning
    # that points to the caller
    ar, ma = [1.0, -0.5], [1.0, 0.4]
    process = ArmaProcess(ar, ma)

    def periodogram(**kwargs):
        if method:
            return process.periodogram(8, **kwargs)
        return arma_periodogram(ar, ma, 8, **kwargs)

    name = "ArmaProcess.periodogram" if method else "arma_periodogram"
    match = (
        f"^The spectral density returned by {name} is divided by "
        r"sqrt\(2 \* pi\), not by 2 \* pi as in the usual definition\. The "
        r"default will change to 2 \* pi after statsmodels 0\.17 is released\."
    )
    with pytest.warns(FutureWarning, match=match) as record:
        w, sd = periodogram()
    assert len(record) == 1
    assert record[0].filename == __file__
    z = np.exp(-1j * w)
    poly = np.polynomial.polynomial.polyval
    expected = np.abs(poly(z, ma)) ** 2 / np.abs(poly(z, ar)) ** 2
    assert_allclose(sd, expected / np.sqrt(2 * np.pi), rtol=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert_allclose(periodogram(normalization="sqrt(2pi)")[1], sd, rtol=1e-15)
        sd_2pi = periodogram(normalization="2pi")[1]
    assert_allclose(sd_2pi, expected / (2 * np.pi), rtol=1e-12)


@pytest.mark.parametrize("method", [False, True])
def test_arma_periodogram_normalization_invalid(method):
    process = ArmaProcess([1.0, -0.5], [1.0, 0.4])

    def periodogram(normalization):
        if method:
            return process.periodogram(8, normalization=normalization)
        return arma_periodogram(process.ar, process.ma, 8, normalization=normalization)

    with pytest.raises(ValueError, match="normalization must be one of"):
        periodogram("pi")
    with pytest.raises(TypeError, match="normalization must be a string"):
        periodogram(2.0)


@pytest.mark.skipif(
    ArmaProcess.periodogram.__doc__ is None, reason="docstrings are stripped"
)
def test_arma_process_periodogram_docstring():
    # The method documents normalization as the function does
    def parameters(func):
        return {param.name: param for param in NumpyDocString(func.__doc__)["Parameters"]}

    method = parameters(ArmaProcess.periodogram)
    assert list(method) == ["nobs", "normalization"]
    assert method["normalization"] == parameters(arma_periodogram)["normalization"]


def test_arma_periodogram_unit_root():
    # The frequency response of a unit root is not defined at frequency 0
    with np.errstate(divide="ignore", invalid="ignore"):
        with pytest.warns(RuntimeWarning, match="nan in frequency response") as rec:
            w, sd = arma_periodogram(
                [1.0, -1.0], [1.0], worN=8, normalization="sqrt(2pi)"
            )
    # The warning points at the caller, not at the wrapper of _require_1d
    assert [r.filename for r in rec if r.category is RuntimeWarning] == [__file__]
    expected = 1 / (2 - 2 * np.cos(w[1:])) / np.sqrt(2 * np.pi)
    assert_allclose(sd[1:], expected, rtol=1e-10)


@pytest.mark.parametrize("worN", [0, -1, np.int64(0)])
def test_arma_periodogram_worn_not_positive(worN):
    # 0 returned empty arrays, and a negative integer raised an error from
    # scipy that said that worN must be nonnegative
    with pytest.raises(ValueError, match="worN must be a positive integer, got"):
        arma_periodogram([1.0, -0.6], [1.0, 0.3], worN=worN)


def test_arma_periodogram_worn_float_is_frequency():
    # A float is a frequency, 0.0 too, not the number of frequencies
    phi, theta = 0.6, 0.3
    w, sd = arma_periodogram(
        [1.0, -phi], [1.0, theta], worN=0.0, normalization="sqrt(2pi)"
    )
    assert_allclose(w, [0.0], rtol=1e-15)
    expected = (1 + theta) ** 2 / (1 - phi) ** 2 / np.sqrt(2 * np.pi)
    assert_allclose(sd, [expected], rtol=1e-12)


def test_ar2arma_start():
    # A correctly specified AR(1) is fit exactly, so the coefficients are
    # found from a start other than the default
    n = 40
    ar_true = [1.0, -0.8]
    ar_des = np.r_[ar_true, np.zeros(n - 2)]
    ar_app, ma_app, _ = ar2arma(ar_des, p=2, q=2, n=n, start=np.array([-0.2, 0.3]))
    assert_allclose(ar_app, ar_true, rtol=1e-6)
    # atol: the MA coefficient of the process is zero
    assert_allclose(ma_app, [1.0, 0.0], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("form", ["list", "tuple", "series"])
def test_ar2arma_input_forms(form):
    # A tuple used to be unpacked into the arguments of the objective and
    # raised a TypeError
    n = 40
    ar_true = [1.0, -0.8]
    ar_des = np.r_[ar_true, np.zeros(n - 2)]
    ar_app, ma_app, _ = ar2arma(_coefficient_form(ar_des, form), p=2, q=1, n=n)
    expected_ar, expected_ma, _ = ar2arma(ar_des, p=2, q=1, n=n)
    assert_allclose(ar_app, expected_ar, rtol=1e-12)
    assert_allclose(ma_app, expected_ma, rtol=1e-12)
    # The AR(1) is correctly specified, so it is recovered
    assert_allclose(ar_app, ar_true, rtol=1e-5)


def test_deconvolve_n_terms():
    # 1 / (1 - L / 2) = sum_k L^k / 2^k. With n terms of the quotient, the
    # remainder is 1 - (1 - L / 2) sum_{k<n} L^k / 2^k = L^n / 2^n. All of the
    # values are powers of 2, so the arithmetic is exact.
    n = 6
    quot, rem = deconvolve([1.0], [1.0, -0.5], n=n)
    assert_allclose(quot, 0.5 ** np.arange(n), rtol=1e-15)
    assert_allclose(rem, np.r_[np.zeros(n), 0.5**n], rtol=1e-15)


COEFFICIENT_FORMS = ("list", "tuple", "series")


def _coefficient_form(value, form):
    return {"list": list, "tuple": tuple, "series": pd.Series}[form](
        np.asarray(value).tolist()
    )


@pytest.mark.parametrize("form", COEFFICIENT_FORMS)
@pytest.mark.parametrize(
    "func", [arma_acovf, arma_acf, arma_pacf, arma_impulse_response, arma2ma, arma2ar]
)
def test_lag_polynomial_input_forms(func, form):
    # A lag polynomial is array_like, and 1-d
    ar = np.array([1.0, -0.5, 0.25])
    ma = np.array([1.0, 0.4])
    res = func(_coefficient_form(ar, form), _coefficient_form(ma, form), 8)
    assert isinstance(res, np.ndarray)
    assert_allclose(res, func(ar, ma, 8), rtol=1e-14)


@pytest.mark.parametrize(
    "func", [arma_acovf, arma_acf, arma_pacf, arma_impulse_response, arma2ma, arma2ar]
)
def test_lag_polynomial_int_dtype(func):
    expected = func(np.array([1.0]), np.array([1.0, 2.0, -1.0]), 6)
    assert_allclose(func(np.array([1]), np.array([1, 2, -1]), 6), expected, rtol=1e-14)


TWO_D_SHAPES = ["column", "row", "1x1", "two columns", "dataframe"]


def _as_2d(value, shape):
    value = np.asarray(value, dtype=float)
    return {
        "column": value[:, None],
        "row": value[None, :],
        "1x1": value[None, :1],
        "two columns": np.c_[value, value],
        "dataframe": pd.DataFrame({"x": value}),
    }[shape]


def _raises_2d(arg, shape):
    # The message for input with more than one dimension that raised before
    return f"^{arg} must be 1-D, got an array with shape {re.escape(str(shape))}$"


def _warns_2d(arg):
    # The message for input with more than one dimension that returned before
    return (
        f"^{arg} must be 1-D. Passing an array with 2 dimensions is deprecated "
        "and will raise a ValueError after statsmodels 0.17 is released.$"
    )


def _spectral_density(ar, ma, w):
    z = np.exp(-1j * w)
    poly = np.polynomial.polynomial.polyval
    return np.abs(poly(z, ma) / poly(z, ar)) ** 2 / np.sqrt(2 * np.pi)


LAG_POLYNOMIAL_FUNCTIONS = {
    "arma_acovf": arma_acovf,
    "arma_acf": arma_acf,
    "arma_pacf": arma_pacf,
    "arma_impulse_response": arma_impulse_response,
    "arma2ma": arma2ma,
    "arma2ar": arma2ar,
    "arma_periodogram": arma_periodogram,
    "arma_generate_sample": lambda ar, ma, n: arma_generate_sample(
        ar, ma, n, distrvs=np.random.default_rng(0).standard_normal
    ),
}
# The 2-D input of these functions that returned a result before, with
# ar = [1, -0.5, 0.25] and ma = [1, 0.4]
LAG_POLYNOMIAL_2D_RETURNED = [
    ("arma_periodogram", "ar", "column"),
    ("arma_periodogram", "ar", "dataframe"),
    ("arma_periodogram", "ma", "column"),
    ("arma_periodogram", "ma", "1x1"),
    ("arma_periodogram", "ma", "dataframe"),
]
LAG_POLYNOMIAL_2D_RAISED = [
    (name, arg, shape)
    for name in LAG_POLYNOMIAL_FUNCTIONS
    for arg in ("ar", "ma")
    for shape in TWO_D_SHAPES
    if (name, arg, shape) not in LAG_POLYNOMIAL_2D_RETURNED
]


def _unchecked(func):
    # The function without the 1-D check of _require_1d
    return getattr(func, "__wrapped__", func)


# The functions without the 1-D check, which tell what 2-D input did before
LAG_POLYNOMIAL_UNCHECKED = {
    "arma_acovf": _unchecked(arma_acovf),
    "arma_acf": _unchecked(arma_acf),
    "arma_pacf": _unchecked(arma_pacf),
    "arma_impulse_response": _unchecked(arma_impulse_response),
    "arma2ma": lambda ar, ma, n: _unchecked(arma_impulse_response)(ar, ma, leads=n),
    "arma2ar": _unchecked(arma2ar),
    "arma_periodogram": _unchecked(arma_periodogram),
    "arma_generate_sample": lambda ar, ma, n: _unchecked(arma_generate_sample)(
        ar, ma, n, distrvs=np.random.default_rng(0).standard_normal
    ),
}


def _as_tuple(value):
    return value if isinstance(value, tuple) else (value,)


@pytest.mark.parametrize("name, arg, shape", LAG_POLYNOMIAL_2D_RAISED)
def test_lag_polynomial_2d_raises(name, arg, shape):
    # NumPy and SciPy raised errors that did not name the argument, and
    # arma2ar named the other one
    polynomials = {"ar": [1.0, -0.5, 0.25], "ma": [1.0, 0.4]}
    polynomials[arg] = _as_2d(polynomials[arg], shape)
    args = (polynomials["ar"], polynomials["ma"], 8)
    func = LAG_POLYNOMIAL_FUNCTIONS[name]
    if shape == "1x1":
        # A (1, 1) polynomial can reach a conversion of an array to a scalar,
        # which raises in recent NumPy but returned in older NumPy (with a
        # DeprecationWarning from NumPy 1.25). What the function without the
        # check does decides what is expected.
        try:
            with warnings.catch_warnings(record=True):
                expected = LAG_POLYNOMIAL_UNCHECKED[name](*args)
        except Exception:
            pass
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                with pytest.warns(FutureWarning, match=_warns_2d(arg)):
                    result = func(*args)
            for res, exp in zip(_as_tuple(result), _as_tuple(expected), strict=True):
                assert_allclose(res, exp, rtol=1e-12)
            return
    match = _raises_2d(arg, np.shape(polynomials[arg]))
    with pytest.raises(ValueError, match=match) as exc_info:
        func(*args)
    # The error of the function without the check is the cause
    assert exc_info.value.__cause__ is not None


@pytest.mark.parametrize("name, arg, shape", LAG_POLYNOMIAL_2D_RETURNED)
def test_arma_periodogram_2d_deprecated(name, arg, shape):
    # A column, or a (1, 1) ma, returned the result of the 1-D polynomial
    polynomials = {"ar": [1.0, -0.5, 0.25], "ma": [1.0, 0.4]}
    one_d = {"ar": [1.0, -0.5, 0.25], "ma": [1.0] if shape == "1x1" else [1.0, 0.4]}
    polynomials[arg] = _as_2d(polynomials[arg], shape)
    with pytest.warns(FutureWarning, match=_warns_2d(arg)):
        w, sd = arma_periodogram(
            polynomials["ar"], polynomials["ma"], 8, normalization="sqrt(2pi)"
        )
    assert_allclose(w, np.linspace(0, np.pi, 8, endpoint=False), rtol=1e-15)
    assert_allclose(sd, _spectral_density(one_d["ar"], one_d["ma"], w), rtol=1e-12)


@pytest.mark.parametrize("func", [arma_acovf, arma_acf, arma_pacf])
@pytest.mark.parametrize("arg", ["ar", "ma"])
@pytest.mark.parametrize("shape", ["1x1", "row"])
def test_white_noise_2d_deprecated(func, arg, shape):
    # If the other polynomial has one element, a (1, 1) polynomial or a row
    # gave the white noise result. For a row this ignores its coefficients
    # after the first.
    polynomials = {"ar": [1.0], "ma": [1.0]}
    polynomials[arg] = _as_2d([1.0, -0.5] if arg == "ar" else [1.0, 0.4], shape)
    with pytest.warns(FutureWarning, match=_warns_2d(arg)) as record:
        res = func(polynomials["ar"], polynomials["ma"], 4)
    assert_allclose(res, [1.0, 0.0, 0.0, 0.0], rtol=1e-15)
    # arma_acf and arma_pacf call arma_acovf, which does not warn again, and
    # the warning points to the caller
    assert len(record) == 1
    assert record[0].filename == __file__


@pytest.mark.parametrize("func", [arma_acovf, arma_acf, arma_pacf])
@pytest.mark.parametrize("arg", ["ar", "ma"])
def test_white_noise_column_raises(func, arg):
    polynomials = {"ar": [1.0], "ma": [1.0]}
    polynomials[arg] = _as_2d([1.0, 0.5], "column")
    with pytest.raises(ValueError, match=_raises_2d(arg, (2, 1))):
        func(polynomials["ar"], polynomials["ma"], 4)


def test_lag_polynomial_2d_keyword():
    ar, ma = [1.0, -0.5], [1.0, 0.4]
    with pytest.warns(FutureWarning, match=_warns_2d("ma")):
        w, sd = arma_periodogram(
            ar=ar, ma=_as_2d(ma, "column"), worN=4, normalization="sqrt(2pi)"
        )
    assert_allclose(sd, _spectral_density(ar, ma, w), rtol=1e-12)
    with pytest.raises(ValueError, match=_raises_2d("ma", (2, 1))):
        arma_impulse_response(ar=ar, ma=_as_2d(ma, "column"), leads=4)


def test_lag_polynomial_2d_both():
    # Both polynomials 2-D: the error names the first, and if the function
    # returns, each one warns
    ar, ma = [1.0, -0.5], [1.0, 0.4]
    with pytest.raises(ValueError, match=_raises_2d("ar", (1, 2))):
        arma_impulse_response(_as_2d(ar, "row"), _as_2d(ma, "column"), 4)
    with pytest.warns(FutureWarning) as record:
        w, sd = arma_periodogram(
            _as_2d(ar, "column"), _as_2d(ma, "column"), 4, normalization="sqrt(2pi)"
        )
    assert [str(item.message)[:2] for item in record] == ["ar", "ma"]
    assert_allclose(sd, _spectral_density(ar, ma, w), rtol=1e-12)


@pytest.mark.parametrize("form", COEFFICIENT_FORMS)
def test_arma_process_input_forms(form):
    ar = np.array([1.0, -0.5, 0.25])
    ma = np.array([1.0, 0.4])
    expected = ArmaProcess(ar, ma)
    arroots = _coefficient_form(expected.arroots, form)
    maroots = _coefficient_form(expected.maroots, form)
    processes = [
        ArmaProcess(_coefficient_form(ar, form), _coefficient_form(ma, form)),
        ArmaProcess.from_coeffs(
            _coefficient_form(-ar[1:], form), _coefficient_form(ma[1:], form)
        ),
        ArmaProcess.from_roots(maroots, arroots),
    ]
    for process in processes:
        assert_allclose(process.ar, ar, rtol=1e-12)
        assert_allclose(process.ma, ma, rtol=1e-12)
        assert_allclose(process.acf(8), expected.acf(8), rtol=1e-12)
    ar_form, ma_form = _coefficient_form(ar, form), _coefficient_form(ma, form)
    _, sd = arma_periodogram(ar_form, ma_form, 8, normalization="2pi")
    expected_sd = arma_periodogram(ar, ma, 8, normalization="2pi")[1]
    assert_allclose(sd, expected_sd, rtol=1e-14)
    rng = np.random.default_rng(0)
    sample = arma_generate_sample(ar_form, ma_form, 20, distrvs=rng.standard_normal)
    rng = np.random.default_rng(0)
    expected_sample = arma_generate_sample(ar, ma, 20, distrvs=rng.standard_normal)
    assert_allclose(sample, expected_sample, rtol=1e-14)


ARMA_PROCESS_2D_CASES = {
    "ar": lambda value: ArmaProcess(value, [1.0, 0.4]),
    "ma": lambda value: ArmaProcess([1.0, -0.5], value),
    "arcoefs": lambda value: ArmaProcess.from_coeffs(value, [0.4]),
    "macoefs": lambda value: ArmaProcess.from_coeffs([0.5], value),
    "arroots": lambda value: ArmaProcess.from_roots(arroots=value),
    "maroots": lambda value: ArmaProcess.from_roots(maroots=value),
}


@pytest.mark.parametrize("shape", TWO_D_SHAPES)
@pytest.mark.parametrize("arg", list(ARMA_PROCESS_2D_CASES))
def test_arma_process_2d_raises(arg, shape):
    # NumPy raised errors that did not name the argument
    value = _as_2d([1.0, -0.5] if arg in ("ar", "ma") else [2.0, -3.0], shape)
    with pytest.raises(ValueError, match=_raises_2d(arg, np.shape(value))):
        ARMA_PROCESS_2D_CASES[arg](value)


def test_from_roots_empty_2d_deprecated():
    # No roots, with shape (0, 2), returned the process without a polynomial
    with pytest.warns(FutureWarning, match=_warns_2d("arroots")):
        process = ArmaProcess.from_roots(arroots=np.zeros((0, 2)))
    assert_allclose(process.ar, [1.0], rtol=1e-15)
    assert_allclose(process.ma, [1.0], rtol=1e-15)


@pytest.mark.parametrize("form", COEFFICIENT_FORMS)
def test_lpol2index_deconvolve_input_forms(form):
    coeffs, index = lpol2index(_coefficient_form([1.0, 0.0, -0.5, 0.0, 0.2], form))
    assert_allclose(coeffs, [1.0, -0.5, 0.2], rtol=1e-15)
    assert_equal(index, [0, 2, 4])
    # (1 - L / 2) (2 + L + L^2 / 2) = 2 - L^3 / 4, exact in floating point
    num = _coefficient_form([2.0, 0.0, 0.0, -0.25], form)
    quot, rem = deconvolve(num, _coefficient_form([1.0, -0.5], form))
    assert_allclose(quot, [2.0, 1.0, 0.5], rtol=1e-15)
    assert_allclose(rem, np.zeros(4), rtol=1e-15)


@pytest.mark.parametrize("shape", TWO_D_SHAPES)
def test_lpol2index_2d_deprecated(shape):
    # 2-D input returned the rows with a non-zero element and their index
    expected = {
        "column": ([[1.0], [-0.5]], [0, 2]),
        "row": ([[1.0, 0.0, -0.5], [1.0, 0.0, -0.5]], [0, 0]),
        "1x1": ([[1.0]], [0]),
        "two columns": (
            [[1.0, 1.0], [1.0, 1.0], [-0.5, -0.5], [-0.5, -0.5]],
            [0, 0, 2, 2],
        ),
        "dataframe": ([[1.0], [-0.5]], [0, 2]),
    }[shape]
    with pytest.warns(FutureWarning, match=_warns_2d("ar")):
        coeffs, index = lpol2index(_as_2d([1.0, 0.0, -0.5], shape))
    assert_allclose(coeffs, expected[0], rtol=1e-15)
    assert_equal(index, expected[1])


@pytest.mark.parametrize("shape", ["row", "1x1"])
def test_deconvolve_num_2d_deprecated(shape):
    # A num with one row is shorter than den, so it was returned as the
    # remainder, with an empty quotient
    num = _as_2d([2.0, 0.0, 0.0, -0.25], shape)
    with pytest.warns(FutureWarning, match=_warns_2d("num")):
        quot, rem = deconvolve(num, [1.0, -0.5])
    assert quot == []
    assert_allclose(rem, num, rtol=1e-15)
    assert rem.shape == num.shape


def test_deconvolve_den_2d_deprecated():
    # den is not used when it is longer than num and n is not given
    with pytest.warns(FutureWarning, match=_warns_2d("den")):
        quot, rem = deconvolve([2.0], _as_2d([1.0, -0.5], "column"))
    assert quot == []
    assert_allclose(rem, [2.0], rtol=1e-15)


@pytest.mark.parametrize("arg", ["num", "den"])
def test_deconvolve_2d_result_unchanged(arg):
    # The deprecated 2-D input still returns an empty list and num itself,
    # with its dtype, while 1-D input returns float arrays
    num = np.array([2, 0, 0, -1])
    den = np.array([1, 2, 3, 4, 5])
    if arg == "num":
        num = num[None, :]
    else:
        den = den[:, None]
    with pytest.warns(FutureWarning, match=_warns_2d(arg)):
        quot, rem = deconvolve(num, den)
    assert isinstance(quot, list)
    assert quot == []
    assert rem is num


@pytest.mark.parametrize(
    "arg, shape",
    [("num", "column"), ("num", "two columns"), ("num", "dataframe")]
    + [("den", shape) for shape in TWO_D_SHAPES],
)
def test_deconvolve_2d_raises(arg, shape):
    polynomials = {"num": [2.0, 0.0, 0.0, -0.25], "den": [1.0, -0.5]}
    polynomials[arg] = _as_2d(polynomials[arg], shape)
    with pytest.raises(ValueError, match=_raises_2d(arg, np.shape(polynomials[arg]))):
        deconvolve(polynomials["num"], polynomials["den"])


def test_index2lpol_2d_deprecated():
    # A row coeffs and a (1, 1) coeffs broadcast to the index, and a (1, 1)
    # index worked with one coefficient
    with pytest.warns(FutureWarning, match=_warns_2d("coeffs")):
        res = index2lpol(_as_2d([0.8, -0.1], "row"), [2, 5])
    assert_allclose(res, [0.0, 0.0, 0.8, 0.0, 0.0, -0.1], rtol=1e-15)
    with pytest.warns(FutureWarning, match=_warns_2d("coeffs")):
        res = index2lpol(_as_2d([0.8], "1x1"), [2, 5])
    assert_allclose(res, [0.0, 0.0, 0.8, 0.0, 0.0, 0.8], rtol=1e-15)
    with pytest.warns(FutureWarning, match=_warns_2d("index")):
        res = index2lpol([0.8], np.array([[5]]))
    assert_allclose(res, [0.0, 0.0, 0.0, 0.0, 0.0, 0.8], rtol=1e-15)


@pytest.mark.parametrize(
    "arg, shape",
    [("coeffs", "column"), ("coeffs", "two columns"), ("coeffs", "dataframe")]
    + [("index", shape) for shape in TWO_D_SHAPES],
)
def test_index2lpol_2d_raises(arg, shape):
    values = {"coeffs": [0.8, -0.1], "index": [2, 5]}
    values[arg] = _as_2d(values[arg], shape)
    if arg == "index":
        values[arg] = values[arg].astype(int)
    with pytest.raises(ValueError, match=_raises_2d(arg, np.shape(values[arg]))):
        index2lpol(values["coeffs"], values["index"])


@pytest.mark.parametrize("shape", TWO_D_SHAPES)
def test_ar2arma_2d_raises(shape):
    # scipy.optimize.leastsq raised for 2-D input
    ar_des = _as_2d(np.r_[1.0, -0.8, np.zeros(38)], shape)
    with pytest.raises(ValueError, match=_raises_2d("ar_des", np.shape(ar_des))):
        ar2arma(ar_des, p=2, q=1, n=40)


@pytest.mark.parametrize(
    "worN",
    [np.array(0), np.array(-1), np.int64(0)],
    ids=["0d-array-0", "0d-array-neg", "int64-0"],
)
def test_arma_periodogram_worN_int_like_not_positive(worN):
    # A 0-d integer array is a number of frequencies, as scipy.signal.freqz
    # takes it, so 0 or a negative value raises. np.array(0) gave empty
    # arrays.
    with pytest.raises(ValueError, match="worN must be a positive integer"):
        arma_periodogram([1.0, -0.5], [1.0, 0.4], worN=worN, normalization="2pi")


@pytest.mark.parametrize(
    "worN, freq",
    [
        (np.array(4), np.linspace(0, np.pi, 4, endpoint=False)),
        (np.int32(4), np.linspace(0, np.pi, 4, endpoint=False)),
        (2.0, np.array([2.0])),
        (0.0, np.array([0.0])),
    ],
    ids=["0d-array", "int32", "float", "float-zero"],
)
def test_arma_periodogram_worN_int_like(worN, freq):
    # An integer is a number of frequencies. A float is a frequency (worN is
    # strict), also an integral float such as 2.0 or 0.0.
    ar, ma = [1.0, -0.5], [1.0, 0.4]
    w, sd = arma_periodogram(ar, ma, worN=worN, normalization="2pi")
    assert_allclose(w, freq, rtol=1e-15)
    z = np.exp(-1j * freq)
    poly = np.polynomial.polynomial.polyval
    expected = np.abs(poly(z, ma)) ** 2 / np.abs(poly(z, ar)) ** 2 / (2 * np.pi)
    assert_allclose(sd, expected, rtol=1e-12)
