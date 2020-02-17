r"""
:Author: Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>

Module providing a set of kernels for use within the kernel_methods package.
"""

import numpy as np
from scipy.special import erf
from scipy import fftpack, integrate
from .kde_utils import make_ufunc, numpy_trans1d_method
from . import _cy_kernels
from copy import copy as shallowcopy

S2PI = np.sqrt(2 * np.pi)

S2 = np.sqrt(2)


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


class Kernel1D(object):
    r"""
    A 1D kernel :math:`K(x)` is a function with the following properties:

    .. math::

        \begin{array}{rcl}
        \int_\mathbb{R} K(x) &=& 1 \\
        \int_\mathbb{R} zK(x)dz &=& 0 \\
        \int_\mathbb{R} x^2K(x) dz &<& \infty \quad (\approx 1)
        \end{array}

    Which translates into the function should have:

    - a sum of 1 (i.e. a valid density of probability);
    - an average of 0 (i.e. centered);
    - a finite variance. It is even recommanded that the variance is close to 1
      to give a uniform meaning to the bandwidth.
    """

    #: Interval containing most of the kernel:
    # :math:`\int_{-c}^c p(x) dx \approx 1`
    cut = 3.
    #: Lower bound of the kernel domain:
    # :math:`\int_{-\infty}^l p(x) dx = 0`
    lower = -np.inf
    #: Upper bound of the kernel domain: :math:`\int_u^{-\infty} p(x) dx = 0`
    upper = np.inf

    @property
    def ndim(self):
        """
        Dimension of the kernel (always 1 for this one)
        """
        return 1

    def for_ndim(self, ndim):
        """
        Returns an equivalent kernel but for `ndim` dimensions.
        """
        assert ndim == 1, "Error, this kernel only works in 1D"
        return self

    def pdf(self, xs, out=None):
        r"""
        Returns the density of the kernel on the points `xs`. This is the
        funtion :math:`K(xs)` itself.

        Parameters
        ----------
        xs : ndarray
            Array of points to evaluate the function on. The method should
            accept any shape of array.
        out: ndarray
            If provided, it will be of the same shape as `xs` and the result
            should be stored in it. Ideally, it should be used for as many
            intermediate computation as possible.
        """
        raise NotImplementedError()

    def __call__(self, xs, out=None):
        """
        Alias for :py:meth:`Kernel1D.pdf`
        """
        return self.pdf(xs, out=out)

    @numpy_trans1d_method()
    def cdf(self, xs, out):
        r"""
        Returns the cumulative density function on the points `xs`, i.e.:

        .. math::

            K_0(x) = \int_{-\infty}^x K(t) dt
        """
        try:
            comp_cdf = self.__comp_cdf
        except AttributeError:
            lower = self.lower
            upper = self.upper
            pdf = self.pdf

            @make_ufunc()
            def comp_cdf(x):
                if x <= lower:
                    return 0
                if x >= upper:
                    x = upper
                return integrate.quad(pdf, lower, x)[0]

            self.__comp_cdf = comp_cdf
        return comp_cdf(xs, out=out)

    @numpy_trans1d_method()
    def pm1(self, xs, out):
        r"""
        Returns the first moment of the density function, i.e.:

        .. math::

            K_1(x) = \int_{-\infty}^x x K(t) dt
        """
        try:
            comp_pm1 = self.__comp_pm1
        except AttributeError:
            lower = self.lower
            upper = self.upper

            def pm1(x):
                return x * self.pdf(x)

            @make_ufunc()
            def comp_pm1(x):
                if x <= lower:
                    return 0
                if x > upper:
                    x = upper
                return integrate.quad(pm1, lower, x)[0]

            self.__comp_pm1 = comp_pm1
        return comp_pm1(xs, out=out)

    @numpy_trans1d_method()
    def pm2(self, xs, out):
        r"""
        Returns the second moment of the density function, i.e.:

        .. math::

            K_2(x) = \int_{-\infty}^x x^2 K(t) dt
        """
        try:
            comp_pm2 = self.__comp_pm2
        except AttributeError:
            lower = self.lower
            upper = self.upper

            def pm2(x):
                return x * x * self.pdf(x)

            @make_ufunc()
            def comp_pm2(x):
                if x <= lower:
                    return 0
                if x > upper:
                    x = upper
                return integrate.quad(pm2, lower, x)[0]

            self.__comp_pm2 = comp_pm2
        return comp_pm2(xs, out=out)

    def rfft(self, N, dx, out=None):
        """
        FFT of the kernel on the points of ``x``. The points will always be
        provided as a regular grid spanning the frequency range to be explored.
        """
        samples = fftsamples(N, dx)
        pdf = self.pdf(samples)
        pdf *= dx
        if out is None:
            out = np.empty(rfftsize(N), dtype=complex)
        out[:] = np.fft.rfft(pdf)
        return out

    def rfft_xfx(self, N, dx, out=None):
        """
        FFT of the function :math:`x k(x)`. The points are given as for the fft
        function.
        """
        samples = fftsamples(N, dx)
        pdf = self.pdf(samples)
        pdf *= samples
        pdf *= dx
        if out is None:
            out = np.empty(rfftsize(N), dtype=complex)
        out[:] = np.fft.rfft(pdf)
        return out

    def dct(self, N, dx, out=None):
        r"""
        DCT of the kernel on the points of ``x``. The points will always be
        provided as a regular grid spanning the frequency range to be explored.
        """
        samples = dctsamples(N, dx)
        out = self.pdf(samples, out=out)
        out *= dx
        out[...] = fftpack.dct(out, overwrite_x=True)
        return out

    @numpy_trans1d_method()
    def _convolution(self, xs, out):
        try:
            comp_conv = self.__comp_conv
        except AttributeError:
            pdf = self.pdf

            def comp_conv(x, support, support_out, out):
                sup_pdf = pdf(support)
                dx = support[1] - support[0]

                @make_ufunc(1, 1)
                def comp(x):
                    np.subtract(x, support, out=support_out)
                    pdf(support_out, out=support_out)
                    np.multiply(support_out, sup_pdf, out=support_out)
                    return np.sum(support_out) * dx

                return comp(x, out=out)

            self.__comp_conv = comp_conv
        sup = np.linspace(-2.5 * self.cut, 2.5 * self.cut, 2**16)
        sup_out = np.empty(sup.shape, sup.dtype)
        return comp_conv(xs, sup, sup_out, out=out)

    @property
    def convolution(self):
        r"""
        Convolution kernel.

        The definition of a convolution kernel is:

        .. math::

            \bar{K(x)} = (K \otimes K)(x) = \int_{\mathcal{R}} K(y) K(x-y) dy

        Notes
        -----

        The computation of the convolution is, by default, very expensive. Most
        kernels should define this methods in addition to the PDF.
        """
        if not hasattr(self, '_convolve_kernel'):
            self._convolve_kernel = From1DPDF(self._convolution)
        return self._convolve_kernel


class From1DPDF(Kernel1D):
    """
    This class creates a kernel from a single function computing the PDF.
    """
    def __init__(self, pdf):
        self._pdf = pdf

    def pdf(self, xs, out=None):
        """
        Call the pdf function set at construction time
        """
        return self._pdf(xs, out)

    __call__ = pdf


class Gaussian1D(Kernel1D):
    """
    1D Gaussian density kernel with extra integrals for 1D bounded kernel
    estimation.
    """
    cut = 5.

    def for_ndim(self, ndim):
        """
        Return an equivalent kernel, but for `ndim` dimensions
        """
        if ndim == 1:
            return self
        return Gaussian(ndim)

    def pdf(self, xs, out=None):
        r"""
        Return the probability density of the function. The formula used is:

        .. math::

            \phi(x) = \frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}

        :param ndarray xs: Array of any shape
        :returns: an array of shape identical to ``xs``
        """
        return _cy_kernels.norm1d_pdf(xs, out)

    def convolution(self, xs, out=None):
        r"""
        Return the PDF of the Gaussian convolution kernel, given by:

        .. math::

            \bar{K}(x) = \frac{1}{2\sqrt{\pi}} e^{-\frac{x^2}{4}}
        """
        return _cy_kernels.norm1d_convolution(xs, out)

    def _pdf(self, xs, out=None):
        """
        Full-python implementation of :py:func:`Gaussian1D.pdf`
        """
        xs = np.asarray(xs)
        if out is None:
            out = np.empty(xs.shape, dtype=xs.dtype)
        np.multiply(xs, xs, out)
        out *= -0.5
        np.exp(out, out)
        out /= S2PI
        return out

    __call__ = pdf

    def _ft(self, xs, out):
        out = np.multiply(xs, xs, out)
        out *= -2 * np.pi**2
        np.exp(out, out)
        return out

    def rfft(self, N, dx, out=None):
        """
        Returns the FFT of the Gaussian distribution
        """
        xs = rfftfreq(N, dx)
        return self._ft(xs, out)

    def rfft_xfx(self, N, dx, out=None):
        r"""
        The FFT of :math:`x\mathcal{N}(x)` which is:

        .. math::

            \text{FFT}(x \mathcal{N}(x)) = -e^{-\frac{\omega^2}{2}}\omega i
        """
        xs = rfftfreq(N, dx)
        if out is None:
            out = np.empty(xs.shape, dtype=complex)
        np.multiply(xs, xs, out)
        out *= -2 * np.pi**2
        np.exp(out, out)
        out *= xs
        out *= -2j * np.pi
        return out

    def dct(self, N, dx, out=None):
        """
        Returns the FFT of the Gaussian distribution
        """
        xs = dctfreq(N, dx)
        return self._ft(xs, out)

    def cdf(self, xs, out=None):
        r"""
        Cumulative density of probability. The formula used is:

        .. math::

            \text{cdf}(x) \triangleq \int_{-\infty}^x \phi(x)
                dz = \frac{1}{2}\text{erf}\left(\frac{x}{\sqrt{2}}\right) +
                    \frac{1}{2}
        """
        return _cy_kernels.norm1d_cdf(xs, out)

    def _cdf(self, xs, out=None):
        """
        Full-python implementation of :py:func:`Gaussian1D.cdf`
        """
        xs = np.asarray(xs)
        if out is None:
            out = np.empty(xs.shape, dtype=xs.dtype)
        np.divide(xs, S2, out)
        erf(out, out)
        out *= 0.5
        out += 0.5
        return out

    def pm1(self, xs, out=None):
        r"""
        Partial moment of order 1:

        .. math::

            \text{pm1}(x) \triangleq \int_{-\infty}^x x\phi(x) dz
                = -\frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}
        """
        return _cy_kernels.norm1d_pm1(xs, out)

    def _pm1(self, xs, out=None):
        """
        Full-python implementation of :py:func:`Gaussian1D.pm1`
        """
        xs = np.asarray(xs)
        if out is None:
            out = np.empty(xs.shape, dtype=xs.dtype)
        np.multiply(xs, xs, out)
        out *= -0.5
        np.exp(out, out)
        out /= -S2PI
        return out

    def pm2(self, xs, out=None):
        r"""
        Partial moment of order 2:

        .. math::

            \text{pm2}(x) \triangleq \int_{-\infty}^x x^2\phi(x) dz
                = \frac{1}{2}\text{erf}\left(\frac{x}{2}\right) -
                \frac{x}{\sqrt{2\pi}} e^{-\frac{x^2}{2}} + \frac{1}{2}
        """
        return _cy_kernels.norm1d_pm2(xs, out)

    def _pm2(self, xs, out=None):
        """
        Full-python implementation of :py:func:`Gaussian1D.pm2`
        """
        xs = np.asarray(xs, dtype=float)
        if out is None:
            out = np.empty(xs.shape)
        np.divide(xs, S2, out)
        erf(out, out)
        out /= 2
        if xs.shape:
            zz = np.isfinite(xs)
            sz = xs[zz]
            out[zz] -= sz * np.exp(-0.5 * sz * sz) / S2PI
        elif np.isfinite(xs):
            out -= xs * np.exp(-0.5 * xs * xs) / S2PI
        out += 0.5
        return out


class KernelnD(object):
    """
    This class is the base class for nD kernels.

    It provides various services, such as numerical approximations for the CDF,
    FFT and DCT for the kernel.
    """
    #: Interval containing most of the kernel
    cut = 3.
    #: Lower bound of the kernel domain for each axis
    lower = -np.inf
    #: Upper bound of the kernel domain for each axis
    upper = np.inf

    def __init__(self, ndim=2):
        self._ndim = ndim

    @property
    def ndim(self):
        """
        Number of dimension of the kernel
        """
        return self._ndim

    def for_ndim(self, ndim):
        """
        Create a version of the same kernel, but for dimension ``ndim``

        Notes
        -----
        The default version copies the object, and changed the :py:attr:`ndim`
        attribute. If this is not sufficient, you need to override this method.
        """
        if ndim == self.ndim:
            return self
        new_ker = shallowcopy(self)
        new_ker._ndim = ndim
        return new_ker

    def pdf(self, y, out=None):
        r"""
        Returns the density of the kernel on the points `xs`. This is the
        funtion :math:`K(x)` itself.

        Parameters
        ----------
        xs: ndarray
            Array of points to evaluate the function on. This should be at
            least a 2D array, with the last dimension corresponding to the
            dimension of the problem.
        out: ndarray
            If provided, it will be of the same shape as `xs` and the result
            should be stored in it. Ideally, it should be used for as many
            intermediate computation as possible.
        """
        raise NotImplementedError()

    def __call__(self, xs, out=None):
        """
        Alias for :py:meth:`KernelnD.pdf`
        """
        return self.pdf(xs, out=out)

    def cdf(self, xs, out=None):
        """
        CDF of the kernel.

        By default, use :py:func:`scipy.integrate.nquad` to integrate the PDF
        from the lower bounds to the upper bound.
        """
        try:
            comp_cdf = self.__comp_cdf
        except AttributeError:

            def pdf(*xs):
                return self.pdf(xs)

            lower = self.lower
            upper = self.upper
            ndim = self.ndim

            @make_ufunc(ndim)
            def comp_cdf(*xs):
                if any(x <= lower for x in xs):
                    return 0
                xs = np.minimum(xs, upper)
                return integrate.nquad(pdf, [(lower, x) for x in xs])[0]

            self.__comp_cdf = comp_cdf
        xs = np.atleast_2d(xs)
        if out is None:
            out = np.empty(xs.shape[:-1], dtype=float)
        return comp_cdf(*xs.T, out=out)

    def rfft(self, N, dx, out=None):
        """
        FFT of the kernel on the points of ``xs``. The points will always be
        provided as a regular grid spanning the frequency range to be explored.
        """
        samples = (s[..., None] for s in fftnsamples(N, dx))
        samples = np.concatenate(np.broadcast_arrays(*samples), axis=-1)
        pdf = self.pdf(samples)
        pdf *= np.prod(dx)
        if out is None:
            out = np.empty(rfftnsize(N), dtype=complex)
        out[:] = np.fft.rfftn(pdf)
        return out

    def dct(self, N, dx, out=None):
        """
        DCT of the kernel on the points of ``xs``. The points will always be
        provided as a regular grid spanning the frequency range to be explored.
        """
        samples = (s[..., None] for s in dctnsamples(N, dx))
        samples = np.concatenate(np.broadcast_arrays(*samples), axis=-1)
        pdf = self.pdf(samples)
        pdf *= np.prod(dx)
        if out is None:
            out = np.empty(N, dtype=float)
        out[:] = fftpack.dct(pdf, axis=0)
        for a in range(1, out.ndim):
            out[:] = fftpack.dct(out, axis=a)
        return out


class Gaussian(KernelnD):
    """
    Returns a function-object for the PDF of a Normal kernel of variance
    identity and average 0 in dimension ``dim``.
    """
    cut = 5.

    def for_ndim(self, ndim):
        """
        Return an equivalent kernel, but for `ndim` dimensions
        """
        if ndim == 1:
            return Gaussian1D()
        return Gaussian(ndim)

    def __init__(self, dim=2):
        super(Gaussian, self).__init__(dim)
        self.factor = 1 / np.sqrt(2 * np.pi)**dim

    def pdf(self, xs, out=None):
        """
        Return the probability density of the function.

        :param ndarray xs: Array of shape (...,D) where D is the dimension of
        the kernel :returns: an array of shape (...) with the density on each
        point of ``xs``
        """
        xs = np.atleast_2d(xs)
        if out is None:
            out = np.empty(xs.shape[:-1], dtype=xs.dtype)
        np.sum(xs * xs, axis=-1, out=out)
        out *= -0.5
        np.exp(out, out=out)
        out *= self.factor
        return out

    def cdf(self, xs, out=None):
        """
        Return the CDF of the Gaussian kernel
        """
        tmp = erf(xs / np.sqrt(2))
        tmp += 1
        out = np.prod(tmp, axis=-1, out=out)
        out /= 2**self.ndim
        return out

    def _ft(self, fs, out):
        cst = -2 * np.pi**2
        fs = [np.exp(cst * f**2) for f in fs]
        res = fs[0]
        for i in range(1, len(fs) - 1):
            res = res * fs[i]
        if out is None:
            out = np.multiply(res, fs[-1])
        else:
            np.multiply(res, fs[-1], out=out)
        return out

    def rfft(self, N, dx, out=None):
        fs = rfftnfreq(N, dx)
        return self._ft(fs, out)

    def dct(self, N, dx, out=None):
        fs = dctnfreq(N, dx)
        return self._ft(fs, out)

    __call__ = pdf


from ._kernels1d import TriCube, Epanechnikov, EpanechnikovOrder4, GaussianOrder4
# from ._kernelsnd import Gaussian
from ._kernelsnc import AitchisonAitken, WangRyzin
""" List of 1D kernels """
kernels1D = [
    Gaussian1D, TriCube, Epanechnikov, EpanechnikovOrder4, GaussianOrder4
]
""" List of nD kernels """
kernelsnD = [Gaussian]
""" List of non-continuous kernels """
kernelsNC = [AitchisonAitken, WangRyzin]
