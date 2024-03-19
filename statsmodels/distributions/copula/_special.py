"""

Special functions for copulas not available in scipy

Created on Jan. 27, 2023
"""

import numpy as np
from scipy.special import factorial
from mpmath import polylog


class Sterling1us:
    """Stirling numbers of the first kind (unsigned)
    """
    # based on
    # https://rosettacode.org/wiki/Stirling_numbers_of_the_first_kind#Python

    def __init__(self):
        self._cache = {}

    def __call__(self, n, k):
        key = (n, k)
        if key in self._cache:
            return self._cache[key]
        if n == k:
            return 1
        if (n > 0 and k == 0) or k > n:
            return 0
        result = self(n - 1, k - 1) + (n - 1) * self(n - 1, k)
        self._cache[key] = result
        return result

    def clear_cache(self):
        """clear cache of Sterling numbers
        """
        self._cache = {}


sterling1us = Sterling1us()


def sterling1s(n, k):
    """Stirling numbers of the first kind (signed)
    """
    return sterling1us(n, k) * (-1) ** (n - k)


class Sterling2:
    """Stirling numbers of the second kind
    """
    # based on
    # https://rosettacode.org/wiki/Stirling_numbers_of_the_second_kind#Python

    def __init__(self):
        self._cache = {}

    def __call__(self, n, k):
        key = (n, k)
        if key in self._cache:
            return self._cache[key]
        if n == k:
            return 1
        if (n > 0 and k == 0) or k > n:
            return 0
        result = k * self(n - 1, k) + self(n - 1, k - 1)
        self._cache[key] = result
        return result

    def clear_cache(self):
        """clear cache of Sterling numbers
        """
        self._cache = {}


sterling2 = Sterling2()


def li3(z):
    """Polylogarithm for negative integer order -3

    Li(-3, z)
    """
    return z * (1 + 4 * z + z**2) / (1 - z)**4


def li4(z):
    """Polylogarithm for negative integer order -4

    Li(-4, z)
    """
    return z * (1 + z) * (1 + 10 * z + z**2) / (1 - z)**5


def lin(n, z):
    """Polylogarithm for negative integer order -n

    Li(-n, z)

    https://en.wikipedia.org/wiki/Polylogarithm#Particular_values
    """
    if np.size(z) > 1:
        z = np.array(z)[..., None]

    k = range(n + 1)
    st2 = np.array([sterling2(n + 1, ki + 1) for ki in k])
    res = (-1)**(n+1) * np.sum(factorial(k) * st2
                                * (-1 / (1 - z))**(np.array(k)+1), axis=-1)
    # this formula is numerically unstable for large n and low z
    # e.g. lin(25,1e-9)
    # better use mpmath.polylog
    return res


"""Polylogarithm from mpmath, vectorized
"""
polylog_vec = np.vectorize(lambda n, z: float(polylog(n, z)))
