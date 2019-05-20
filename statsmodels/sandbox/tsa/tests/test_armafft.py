import numpy as np
from numpy.testing import (assert_almost_equal,
                           assert_allclose,
                           assert_equal)

import pytest

from statsmodels.sandbox.tsa.fftarma import ArmaFft


arlist = [
    [1.],
    [1, -0.9],  # ma representation will need many terms to get high precision
    [1, 0.9],
    [1, -0.9, 0.3]
]

malist = [
    [1.],
    [1, 0.9],
    [1, -0.9],
    [1, 0.9, -0.3]
]


@pytest.mark.parametrize('ar', arlist)
@pytest.mark.parametrize('ma', malist)
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
    assert_almost_equal(spdr, spdd[:nfreq], decimal=7,
                        err_msg='spdr spdd not equal for %s, %s' % (ar, ma))
    assert_almost_equal(spdr, spdp, decimal=7,
                        err_msg='spdr spdp not equal for %s, %s' % (ar, ma))


@pytest.mark.parametrize('ar', arlist)
@pytest.mark.parametrize('ma', malist)
def test_armafft(ar, ma):
    # test other methods
    arma = ArmaFft(ar, ma, 20)
    ac1 = arma.invpowerspd(1024)[:10]
    ac2 = arma.acovf(10)[:10]
    assert_allclose(ac1, ac2, atol=1e-15,
                    err_msg='acovf not equal for %s, %s' % (ar, ma))
