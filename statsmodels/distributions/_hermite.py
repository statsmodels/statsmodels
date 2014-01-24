from __future__ import division, print_function, absolute_import

import numpy as np
from numpy import poly1d

""" A stub for the Hermite polynomials for older numpy versions.
This uses _hermnorm recursion, a direct copy-paste from 
scipy/stats/morestats.py
"""

def _hermnorm(N):
    # return the negatively normalized hermite polynomials up to order N-1
    #  (inclusive)
    #  using the recursive relationship
    #  p_n+1 = p_n(x)' - x*p_n(x)
    #   and p_0(x) = 1
    plist = [None]*N
    plist[0] = poly1d(1)
    for n in range(1,N):
        plist[n] = plist[n-1].deriv() - poly1d([1,0])*plist[n-1]
    return plist


class HermiteEStub(object):
    """Cook up an HermiteE replacement from _hermnorm."""
    def __init__(self, coef):
        self.coef = list(coef)
        h = _hermnorm(len(coef))
        self._func = np.poly1d([0])
        for j in range(len(coef)):
            self._func += (1 - 2*(j%2)) * coef[j] * h[j]

    def __call__(self, x):
        return self._func(x)

    def roots(self):
        return np.roots(self._func.coeffs)

try:
    from numpy.polynomial.hermite_e import HermiteE
except ImportError:  # numpy < 1.6
    HermiteE = HermiteEStub


if __name__ == "__main__":
    np.random.seed(12345)
    coef = np.random.random(3)
    xx = np.random.random(1000) * 20 - 10

    np.testing.assert_allclose(HermiteE(coef)(xx),
                               HermiteEStub(coef)(xx))
    np.testing.assert_allclose(np.sort(HermiteE(coef).roots()),
                               np.sort(HermiteEStub(coef).roots()))

