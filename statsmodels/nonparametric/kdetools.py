# ### Convenience Functions to be moved to kerneltools ####
import numpy as np


def forrt(X, m=None):
    """
    RFFT with order like Munro (1976) FORTT routine

    Parameters
    ----------
    X : ndarray
        The array to transform.
    m : int, optional
        The length used for the FFT. If None, ``len(X)`` is used.

    Returns
    -------
    ndarray
        The real and imaginary parts of the RFFT, concatenated in the
        Munro (1976) FORTT order.
    """
    if m is None:
        m = len(X)
    y = np.fft.rfft(X, m) / m
    return np.r_[y.real, y[1:-1].imag]


def revrt(X, m=None):
    """
    Inverse of forrt, equivalent to Munro (1976) REVRT routine

    Parameters
    ----------
    X : ndarray
        The array, in the Munro (1976) FORTT order, to invert.
    m : int, optional
        The length used for the inverse FFT. If None, ``len(X)`` is used.

    Returns
    -------
    ndarray
        The inverse real FFT of `X`.
    """
    if m is None:
        m = len(X)
    i = int(m // 2 + 1)
    y = X[:i] + np.r_[0, X[i:], 0] * 1j
    return np.fft.irfft(y) * m


def silverman_transform(bw, M, RANGE):
    """
    FFT of Gaussian kernel following to Silverman AS 176

    Parameters
    ----------
    bw : float
        The bandwidth used in the Gaussian kernel.
    M : int
        The number of grid points.
    RANGE : float
        The range of the grid over which the kernel is evaluated.

    Returns
    -------
    ndarray
        The FFT of the Gaussian kernel evaluated at the grid points.

    Notes
    -----
    Underflow is intentional as a dampener.
    """
    J = np.arange(M / 2 + 1)
    FAC1 = 2 * (np.pi * bw / RANGE) ** 2
    JFAC = J**2 * FAC1
    BC = 1 - 1.0 / 3 * (J * 1.0 / M * np.pi) ** 2
    FAC = np.exp(-JFAC) / BC
    kern_est = np.r_[FAC, FAC[1:-1]]
    return kern_est


def counts(x, v):
    """
    Counts the number of elements of x that fall within the grid points v

    Parameters
    ----------
    x : array_like
        The data whose elements are counted.
    v : array_like
        The grid points (bin edges) used to bin `x`.

    Returns
    -------
    ndarray
        The number of elements of `x` falling in each bin defined by `v`.

    Notes
    -----
    Using np.digitize and np.bincount
    """
    idx = np.digitize(x, v)
    return np.bincount(idx, minlength=len(v))


def kdesum(x, axis=0):
    """
    Computes the sum of pairwise differences of x along the given axis

    Parameters
    ----------
    x : array_like
        The data for which pairwise differences are summed.
    axis : int, optional
        The axis along which the differences ``x[i] - x`` are summed.

    Returns
    -------
    ndarray
        The array of summed pairwise differences, one entry per element
        of `x`.
    """
    return np.asarray([np.sum(x[i] - x, axis) for i in range(len(x))])
