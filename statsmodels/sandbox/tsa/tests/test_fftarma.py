"""
Tests corresponding to sandbox.tsa.fftarma
"""
import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal
import pytest

from statsmodels.sandbox.tsa.fftarma import ArmaFft, spdar1


def test_padarr_atend():
    arma = ArmaFft([1, -0.5], [1.0, 0.4], 40)
    result = arma.padarr(np.array([1.0, 2.0, 3.0]), 5)
    assert_array_almost_equal(result, [1.0, 2.0, 3.0, 0.0, 0.0])


def test_padarr_atstart():
    arma = ArmaFft([1, -0.5], [1.0, 0.4], 40)
    result = arma.padarr(np.array([1.0, 2.0, 3.0]), 5, atend=False)
    assert_array_almost_equal(result, [0.0, 0.0, 1.0, 2.0, 3.0])


def test_pad_extends_both_polynomials():
    arma = ArmaFft([1, -0.5, 0.1], [1.0, 0.4], 40)
    arpad, mapad = arma.pad(6)
    assert_array_almost_equal(arpad, [1.0, -0.5, 0.1, 0.0, 0.0, 0.0])
    assert_array_almost_equal(mapad, [1.0, 0.4, 0.0, 0.0, 0.0, 0.0])


def test_fftar_default_length_matches_ar():
    arma = ArmaFft([1, -0.5, 0.1], [1.0, 0.4], 40)
    result = arma.fftar()
    assert len(result) == len(arma.ar)


def test_fftma_explicit_length():
    arma = ArmaFft([1, -0.5, 0.1], [1.0, 0.4], 40)
    result = arma.fftma(len(arma.ma))
    assert len(result) == len(arma.ma)


@pytest.mark.xfail(
    reason=(
        "BUG: fftma(n)'s `if n is None: n = len(self.ar)` pads/pads the MA "
        "polynomial to the *AR* polynomial's length by default -- almost "
        "certainly copy-pasted from fftar() without updating self.ar to "
        "self.ma. Whenever len(ar) != len(ma), fftma(None) returns an "
        "array of the wrong length (matching ar, not ma)."
    ),
    raises=AssertionError,
    strict=True,
)
def test_fftma_default_length_should_match_ma_not_ar():
    arma = ArmaFft([1, -0.5, 0.1], [1.0, 0.4], 40)
    assert len(arma.ar) != len(arma.ma)
    result = arma.fftma(None)
    assert len(result) == len(arma.ma)


def test_fftarma_equals_fftma_over_fftar():
    arma = ArmaFft([1, -0.5], [1.0, 0.4], 40)
    n = 16
    assert_allclose(arma.fftarma(n), arma.fftma(n) / arma.fftar(n))


def test_fftarma_default_n_is_nobs():
    arma = ArmaFft([1, -0.5], [1.0, 0.4], 40)
    assert_allclose(arma.fftarma(), arma.fftarma(arma.nobs))


def test_invpowerspd_matches_inherited_acovf():
    # this is the exact relationship asserted in the function's own
    # docstring example
    arma = ArmaFft([1, -0.5], [1.0, 0.4], 40)
    assert_allclose(arma.invpowerspd(2**8)[:10], arma.acovf(10), atol=1e-10)


def test_spdroots_matches_closed_form_ar1_formula():
    # cross-check the roots-based spectral density against the textbook
    # closed-form AR(1) spectral density for a pure AR(1) process
    ar1 = ArmaFft([1, -0.5], [1.0], 100)
    w = np.linspace(0.05, np.pi - 0.05, 15)
    spd_roots, _ = ar1.spdroots(w)
    spd_formula = spdar1(ar1.ar, w)
    assert_allclose(spd_roots, spd_formula, rtol=1e-10)


def test_spdar1_scalar_and_array_ar_are_consistent():
    w = np.linspace(0.1, np.pi - 0.1, 5)
    from_array = spdar1(np.array([1, -0.6]), w)
    from_scalar = spdar1(0.6, w)
    assert_allclose(from_array, from_scalar)


def test_spd_shape_is_2n():
    arma = ArmaFft([1, -0.5], [1.0, 0.4], 40)
    spd, w = arma.spd(16)
    assert spd.shape == (32,)
    assert w.shape == (32,)


def test_spdshift_shape_and_nonnegative():
    arma = ArmaFft([1, -0.5], [1.0, 0.4], 40)
    spd, w = arma.spdshift(16)
    assert spd.shape == (16,)
    assert w.shape == (16,)
    assert np.all(spd >= -1e-10)


def test_spddirect_is_nonnegative():
    arma = ArmaFft([1, -0.5], [1.0, 0.4], 40)
    spd, w = arma.spddirect(256)
    assert np.all(spd >= 0)


def test_spdpoly_approximates_spdroots():
    # spdpoly builds a high-order MA approximation of the ARMA spectral
    # density; with enough MA terms it should be close to the exact
    # roots-based density
    arma = ArmaFft([1, -0.5], [1.0, 0.4], 40)
    w = np.linspace(0.1, np.pi - 0.1, 10)
    spd_poly, _ = arma.spdpoly(w, nma=200)
    spd_roots, _ = arma.spdroots(w)
    assert_allclose(spd_poly, spd_roots, rtol=1e-3)


def test_acf2spdfreq_shape():
    arma = ArmaFft([1, -0.5], [1.0, 0.4], 40)
    acovf = arma.acovf(10)
    hw = arma.acf2spdfreq(acovf, nfreq=25)
    assert hw.shape == (25,)


@pytest.mark.xfail(
    reason=(
        "BUG: spdmapoly(w, ...) references a bare name `nfreq` in its "
        "`if w is None:` branch, but `nfreq` is not a parameter, local, "
        "or module-level name anywhere -- calling spdmapoly(None) always "
        "raises NameError."
    ),
    raises=NameError,
    strict=True,
)
def test_spdmapoly_with_default_w():
    arma = ArmaFft([1, -0.5], [1.0, 0.4], 40)
    arma.spdmapoly(w=None)


def test_filter_matches_direct_fft_ratio():
    arma = ArmaFft([1, -0.5], [1.0, 0.4], 40)
    x = np.random.RandomState(0).normal(size=20)
    result = arma.filter(x)
    from numpy import fft

    expected = fft.ifft(arma.fftma(len(x)) / arma.fftar(len(x)) * fft.fft(x))
    assert_allclose(result, expected)


def test_filter_n_equals_fftarma_branch_is_unreachable():
    # BUG (dead code, not a crash): filter() has
    #     if n == self.fftarma:
    #         fftarma = self.fftarma
    #     else:
    #         fftarma = self.fftma(n) / self.fftar(n)
    # `self.fftarma` is a bound method, so `n == self.fftarma` (comparing
    # an int to a method object) is always False -- the special-cased
    # branch can never execute, and if it somehow did, `fftarma` would be
    # rebound to a method object and the next line
    # (`fftarma * fft.fft(x)`) would raise a TypeError. filter() happens
    # to still return correct results today only because it always falls
    # through to the else branch.
    arma = ArmaFft([1, -0.5], [1.0, 0.4], 40)
    x = np.arange(5.0)
    assert not (x.shape[0] == arma.fftarma)


@pytest.mark.xfail(
    reason=(
        "BUG: filter2() does `from statsmodels.tsa.filters import "
        "fftconvolve3`, but fftconvolve3 is not (no longer?) re-exported "
        "from that package's __init__ -- it now only lives in "
        "statsmodels.tsa.filters.filtertools. filter2() always raises "
        "ImportError. Separately, even fftconvolve3 itself currently "
        "raises AttributeError on modern numpy (`np.complex` was removed "
        "as a deprecated alias), so fixing the import alone would not be "
        "enough to make filter2() work."
    ),
    raises=ImportError,
    strict=True,
)
def test_filter2():
    arma = ArmaFft([1, -0.5], [1.0, 0.4], 40)
    x = np.random.RandomState(0).normal(size=20)
    arma.filter2(x)
