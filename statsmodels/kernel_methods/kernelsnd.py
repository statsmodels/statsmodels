"""
This module defines the base class for nD kernels and the nD Gaussian kernel.
"""

import numpy as np
from scipy import integrate, fftpack
from .kernels_utils import (rfftnfreq, rfftnsize, dctnfreq, dctnsamples,
                            fftnsamples)
from copy import copy as shallowcopy
from scipy.special import erf
from .kde_utils import make_ufunc


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
