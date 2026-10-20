"""
Tests for the deprecated ``TsaDescriptive`` collection of statistics.
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest

from statsmodels.tsa.descriptivestats import TsaDescriptive


def _tsa_descriptive(data):
    """Build a TsaDescriptive, silencing the deprecation warning of the class."""
    with pytest.warns(FutureWarning, match="TsaDescriptive is deprecated"):
        return TsaDescriptive(data)


def test_periodogram_definition():
    """The periodogram of the series matches the definition in the docstring."""
    n = 64
    rng = np.random.default_rng(20261020)
    x = rng.standard_normal(n)

    pergr = _tsa_descriptive(x).periodogram()

    # direct evaluation of the sum, independent of the FFT used by the method
    t = np.arange(n)
    k = np.arange(n)
    dft = np.exp(-2j * np.pi * np.outer(k, t) / n) @ x
    expected = np.abs(dft) ** 2 / n
    expected[0] = 0.0

    assert pergr.shape == (n,)
    assert_allclose(pergr, expected, rtol=1e-10, atol=1e-12)


def test_periodogram_of_cosine():
    """A cosine at a Fourier frequency gives one pair of periodogram spikes."""
    n = 64
    freq = 8
    x = np.cos(2 * np.pi * freq * np.arange(n) / n)

    pergr = _tsa_descriptive(x).periodogram()

    expected = np.zeros(n)
    expected[freq] = expected[n - freq] = n / 4.0  # (n / 2) ** 2 / n
    assert_allclose(pergr, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.matplotlib
def test_plot4(close_figures):
    """plot4 needs a working periodogram for its power spectrum panel."""
    rng = np.random.default_rng(20261020)
    x = np.cos(2 * np.pi * 8 * np.arange(64) / 64)
    x += 0.1 * rng.standard_normal(x.shape)

    fig = _tsa_descriptive(x).plot4(nacf=10, nfreq=16)

    assert len(fig.axes) == 4
