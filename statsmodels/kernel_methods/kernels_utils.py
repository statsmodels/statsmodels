import numpy as np
from scipy.special import erf
from scipy import fftpack, integrate
from . import _cy_kernels


def rfftfreq(n, d=1.0):
    """
    Return the Discrete Fourier Transform sample frequencies
    (for usage with rfft, irfft).

    The returned float array `f` contains the frequency bin centers in cycles
    per unit of the sample spacing (with zero at the start).  For instance, if
    the sample spacing is in seconds, then the frequency unit is cycles/second.

    Given a window length `n` and a sample spacing `d`::

      f = [0, 1, ...,     n/2-1,     n/2] / (d*n)   if n is even
      f = [0, 1, ..., (n-1)/2-1, (n-1)/2] / (d*n)   if n is odd

    Unlike `fftfreq` (but like `scipy.fftpack.rfftfreq`)
    the Nyquist frequency component is considered to be positive.

    Parameters
    ----------
    n : int
        Window length.
    d : scalar, optional
        Sample spacing (inverse of the sampling rate). Defaults to 1.

    Returns
    -------
    f : ndarray
        Array of length ``n//2 + 1`` containing the sample frequencies.

    Notes
    -----
    This function has been copied from the source code of numpy because it has
    been added only on version 1.8 while statsmodels requires support of numpy
    1.7.

    Examples
    --------
    >>> signal = np.array([-2, 8, 6, 4, 1, 0, 3, 5, -3, 4], dtype=float)
    >>> fourier = np.fft.rfft(signal)
    >>> n = signal.size
    >>> sample_rate = 100
    >>> freq = np.fft.fftfreq(n, d=1./sample_rate)
    >>> freq
    array([  0.,  10.,  20.,  30.,  40., -50., -40., -30., -20., -10.])
    >>> freq = np.fft.rfftfreq(n, d=1./sample_rate)
    >>> freq
    array([  0.,  10.,  20.,  30.,  40.,  50.])

    """
    if not isinstance(n, int):
        raise ValueError("n should be an integer")
    val = 1.0 / (n * d)
    N = n // 2 + 1
    results = np.arange(0, N, dtype=int)
    return results * val


def rfftsize(N):
    """
    Returns the number of elements in the result of :py:func:`numpy.fft.rfft`.
    """
    return (N // 2) + 1


def rfftnsize(Ns):
    """
    Returns the number of elements in the result of :py:func:`numpy.fft.rfft`.
    """
    return tuple(Ns[:-1]) + ((Ns[-1] // 2) + 1, )


def rfftnfreq(Ns, dx=None):
    """
    Return the Discrete Fourier Transform sample frequencies
    (for usage with :py:func:`numpy.fft.rfftn`, :py:func:`numpy.fft.irfftn`).

    See :py:func:`scipy.fftpack.rfftfreq` and :py:func:`numpy.fft.rfftn` for
    details.

    Parameters
    ----------
    Ns: list of int
        Number of samples for each dimension
    dx: None, 1D or 2D array
        If not None, this must be of same length as Ns and is the space between
        samples along that axis

    Returns
    -------
    list of ndarray
        Sparse grid for the frequencies or dx is 1D, a full grid if dx is 2D.

    Notes
    -----
    If dx is a 2D array, this corresponds to the bandwidth matrix.
    """
    ndim = len(Ns)
    if dx is None:
        dx = np.ones((ndim, ), dtype=float)
    else:
        dx = np.asarray(dx)
        if dx.ndim == 1 and dx.shape != (ndim, ):
            raise ValueError("If 1D, dx must be of same length as Ns")
        elif dx.ndim == 2 and dx.shape != (ndim, ndim):
            raise ValueError(
                "If 2D, dx must be of a square matrix with as many dimensions "
                + "as Ns")
    trans = None
    if dx.ndim == 2:
        trans = dx
        dx = np.ones((ndim, ), dtype=float)
    fs = []
    for d in range(ndim - 1):
        fs.append(np.fft.fftfreq(Ns[d], dx[d]))
    fs.append(rfftfreq(Ns[-1], dx[-1]))
    if trans is None:
        return np.meshgrid(*fs, indexing='ij', sparse=True, copy=False)
    grid = np.asarray(np.meshgrid(*fs, indexing='ij', sparse=False))
    return np.tensordot(trans, grid, axes=([1], [0]))


def fftsamples(N, dx=1.0):
    """
    Returns the array of sample positions needed to comput the FFT with N
    samples.
    (for usage with :py:func:`scipy.fftpack.fft`, :py:func:`numpy.fft.rfft`).

    Parameters
    ----------
    N: int
        Number of samples for the FFT
    dx: float or None
        Distance between sample points. If None, dx = 1.0.

    Returns
    -------
    ndarray
        Array of positions
    """
    if N % 2 == 1:
        n = (N - 1) // 2
        return dx * (np.concatenate([np.arange(n + 1),
                                     np.arange(-n, 0)]) + 0.5)
    else:
        n = N // 2
        return dx * np.concatenate([np.arange(n), np.arange(-n, 0)])


def fftnsamples(Ns, dx=None):
    """
    Returns the array of sample positions needed to comput the FFT with N
    samples.
    (for usage with :py:func:`numpy.fft.fftn`, :py:func:`numpy.fft.rfftn`).

    Parameters
    ----------
    N: list of int
        Number of samples for the FFT for each dimension
    dx: float or None
        Distance between sample points for each dimension. If None, dx = 1.0
        for each dimension.

    Returns
    -------
    Grid
        Grid for the samples
    """
    ndim = len(Ns)
    if dx is None:
        dx = [1.0] * ndim
    elif len(dx) != ndim:
        raise ValueError("Error, dx must be of same length as Ns")
    fs = [fftsamples(Ns[d], dx[d]) for d in range(ndim)]
    return np.meshgrid(*fs, indexing='ij', sparse=True, copy=False)


def dctfreq(N, dx=1.0):
    """
    Return the Discrete Cosine Transform sample frequencies
    (for usage with :py:func:`scipy.fftpack.dct`,
    :py:func:`scipy.fftpack.idct`).

    Parameters
    ----------
    n : int
        Window length.
    d : scalar, optional
        Sample spacing (inverse of the sampling rate). Defaults to 1.

    Returns
    -------
    f : ndarray
        Array of length ``N`` containing the sample frequencies.
    """
    dz = 1 / (2 * N * dx)
    return np.arange(N) * dz


def dctnfreq(Ns, dx=None):
    """
    Return the Discrete Cosine Transform sample frequencies
    (for usage with :py:func:`scipy.fftpack.dct`,
    :py:func:`scipy.fftpack.idct`).

    Parameters
    ----------
    Ns: list of int
        Number of samples for each dimension
    dx: None of list of float
        If not None, this must be of same length as Ns and is the space between
        samples along that axis

    Returns
    -------
    list of ndarray
        Sparse grid for the frequencies
    """
    ndim = len(Ns)
    if dx is None:
        dx = [1.0] * ndim
    elif len(dx) != ndim:
        raise ValueError("Error, dx must be of same length as Ns")
    fs = [dctfreq(Ns[d], dx[d]) for d in range(ndim)]
    return np.meshgrid(*fs, indexing='ij', sparse=True, copy=False)


def dctsamples(N, dx=1.0):
    """
    Returns the array of sample positions needed to comput the DCT with N
    samples.
    (for usage with :py:func:`scipy.fftpack.dct`)

    Parameters
    ----------
    N: int
        Number of samples for the DCT
    dx: float or None
        Distance between sample points. If None, dx = 1.0.

    Returns
    -------
    ndarray
        Array of positions
    """
    return np.arange(0.5, N) * dx


def dctnsamples(Ns, dx=None):
    """
    Returns the array of sample positions needed to comput the DCT with N
    samples.
    (for usage with :py:func:`scipy.fftpack.dctn`)

    Parameters
    ----------
    N: list of int
        Number of samples for the DCT for each dimension
    dx: float or None
        Distance between sample points for each dimension. If None, dx = 1.0
        for each dimension.

    Returns
    -------
    Grid
        Grid for the samples
    """
    ndim = len(Ns)
    if dx is None:
        dx = [1.0] * ndim
    elif len(dx) != ndim:
        raise ValueError("Error, dx must be of same length as Ns")
    fs = [dctsamples(Ns[d], dx[d]) for d in range(ndim)]
    return np.meshgrid(*fs, indexing='ij', sparse=True, copy=False)
