from .kernels_utils import rfftfreq, dctfreq, fftsamples, dctsamples, rfftsize
from . import _cy_kernels
import numpy as np
from scipy import integrate, fftpack
from scipy.special import erf
from .kde_utils import make_ufunc, numpy_trans1d_method

S2PI = np.sqrt(2 * np.pi)

S2 = np.sqrt(2)


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


class TriCube(Kernel1D):
    r"""
    Return the kernel corresponding to a tri-cube distribution, whose
    expression is. The tri-cube function is given by:

    .. math::

        f_r(x) = \left\{\begin{array}{ll}
                        \left(1-|x|^3\right)^3 & \text{, if } x \in [-1;1]\\
                                0 & \text{, otherwise}
                        \end{array}\right.

    As :math:`f_r` is not a probability and is not of variance 1, we use a
    normalized function:

    .. math::

        f(x) = a b f_r(ax)

        a = \sqrt{\frac{35}{243}}

        b = \frac{70}{81}

    """
    def pdf(self, xs, out=None):
        return _cy_kernels.tricube_pdf(xs, out)

    __call__ = pdf

    upper = 1. / _cy_kernels.tricube_width
    lower = -upper
    cut = upper

    def cdf(self, xs, out=None):
        r"""
        CDF of the distribution:

        .. math::

            \text{cdf}(x) = \left\{\begin{array}{ll}
                \frac{1}{162} {\left(60 (ax)^{7} - 7 {\left(2 (ax)^{10}
                + 15 (ax)^{4}\right)} \mathrm{sgn}\left(ax\right) +
                140 ax + 81\right)} & \text{, if}x\in[-1/a;1/a]\\
                0 & \text{, if} x < -1/a \\
                1 & \text{, if} x > 1/a
                \end{array}\right.
        """
        return _cy_kernels.tricube_cdf(xs, out)

    def pm1(self, xs, out=None):
        r"""
        Partial moment of order 1:

        .. math::

            \text{pm1}(x) = \left\{\begin{array}{ll}
                \frac{7}{3564a} {\left(165 (ax)^{8} -
                8 {\left(5 (ax)^{11} + 33 (ax)^{5}\right)}
                \mathrm{sgn}\left(ax\right) + 220 (ax)^{2} - 81\right)}
                & \text{, if} x\in [-1/a;1/a]\\
                0 & \text{, otherwise}
                \end{array}\right.
        """
        return _cy_kernels.tricube_pm1(xs, out)

    def pm2(self, xs, out=None):
        r"""
        Partial moment of order 2:

        .. math::

            \text{pm2}(x) = \left\{\begin{array}{ll}
                \frac{35}{486a^2} {\left(4 (ax)^{9} + 4 (ax)^{3} -
                {\left((ax)^{12} + 6 (ax)^{6}\right)}
                \mathrm{sgn}\left(ax\right) + 1\right)}
                & \text{, if} x\in[-1/a;1/a] \\
                0 & \text{, if } x < -1/a \\
                1 & \text{, if } x > 1/a
            \end{array}\right.
        """
        return _cy_kernels.tricube_pm2(xs, out)

    def convolution(self, xs, out=None):
        r"""
        Tricube convolution kernel:

        .. math::

             \left\{
             \begin{array}{cc}
             \frac{(x+2)^7 \left(14 x^{12}-196 x^{11}+1568 x^{10}-8439 x^9+
                    33474 x^8-98448 x^7+213558 x^6-334740 x^5+561120 x^4-
                    453722 x^3+558880 x^2-206822 x+151470\right)}{12932920}
                    & -2<x\leq -1 \\
             -\frac{(x-2)^7 \left(14 x^{12}+196 x^{11}+1568 x^{10}+8439 x^9+
                    33474 x^8+98448 x^7+213558 x^6+334740 x^5+561120 x^4+
                    453722 x^3+558880 x^2+206822 x+151470\right)}{12932920}
                    & 1\leq x<2 \\
             -\frac{3 x^{19}}{923780}-\frac{3 x^{16}}{40040}-
                    \frac{111 x^{13}}{20020}-\frac{31 x^{10}}{140}-
                    \frac{81 x^9}{70}-\frac{729 x^8}{220}-
                    \frac{747 x^7}{140}-\frac{729 x^6}{182}+\frac{9
                    x^4}{5}-\frac{19683 x^2}{13090}+\frac{6561}{6916}
                    & -1<x\leq 0 \\
             \frac{3 x^{19}}{923780}-\frac{3 x^{16}}{40040}+
                    \frac{111 x^{13}}{20020}-\frac{31 x^{10}}{140}+
                    \frac{81 x^9}{70}-\frac{729 x^8}{220}+
                    \frac{747 x^7}{140}-\frac{729 x^6}{182}+
                    \frac{9 x^4}{5}-\frac{19683 x^2}{13090}+\frac{6561}{6916}
                    & 0<x<1
             \end{array} \right.
        """
        return _cy_kernels.tricube_convolution(xs, out)


class Epanechnikov(Kernel1D):
    r"""
    1D Epanechnikov density kernel with extra integrals for 1D bounded kernel
    estimation.
    """
    def pdf(self, xs, out=None):
        r"""
        The PDF of the kernel is usually given by:

        .. math::

            f_r(x) = \left\{\begin{array}{ll}
                    \frac{3}{4} \left(1-x^2\right) & \text{, if} x \in [-1:1]\\
                            0 & \text{, otherwise}
                    \end{array}\right.

        As :math:`f_r` is not of variance 1 (and therefore would need
        adjustments for the bandwidth selection), we use a normalized function:

        .. math::

            f(x) = \frac{1}{\sqrt{5}}f_r\left(\frac{x}{\sqrt{5}}\right)
        """
        return _cy_kernels.epanechnikov_pdf(xs, out)

    __call__ = pdf

    upper = 1. / _cy_kernels.epanechnikov_width
    lower = -upper
    cut = upper

    def convolution(self, xs, out=None):
        r"""
        Epanechnikov convolution kernel.

        The convolution of the non-normalized kernel is:

        .. math::

            \bar{f_r}(x) = \frac{3}{160} \left( (2-|x|)^3 (4 + 6|x| + x^2)
                \right) \qquad \text{, if } |x| < 2

        But of course, we need to normalize it in the same way:

        .. math::

            \bar{f}(x) = \frac{1}{\sqrt{5}}
                \bar{f_r}\left(\frac{x}{\sqrt{5}}\right)

        """
        return _cy_kernels.epanechnikov_convolution(xs, out)

    def cdf(self, xs, out=None):
        r"""
        CDF of the distribution. The CDF is defined on the interval
        :math:`[-\sqrt{5}:\sqrt{5}]` as:

        .. math::

            \text{cdf}(x) = \left\{\begin{array}{ll}
                    \frac{1}{2} + \frac{3}{4\sqrt{5}} x -
                        \frac{3}{20\sqrt{5}}x^3
                    & \text{, if } x\in[-\sqrt{5}:\sqrt{5}] \\
                    0 & \text{, if } x < -\sqrt{5} \\
                    1 & \text{, if } x > \sqrt{5}
                    \end{array}\right.
        """
        return _cy_kernels.epanechnikov_cdf(xs, out)

    def pm1(self, xs, out=None):
        r"""
        First partial moment of the distribution:

        .. math::

            \text{pm1}(x) = \left\{\begin{array}{ll}
                    -\frac{3\sqrt{5}}{16}\left(1-\frac{2}{5}x^2+\frac{1}{25}x^4\right)
                    & \text{, if } x\in[-\sqrt{5}:\sqrt{5}] \\
                    0 & \text{, otherwise}
                    \end{array}\right.
        """
        return _cy_kernels.epanechnikov_pm1(xs, out)

    def pm2(self, xs, out=None):
        r"""
        Second partial moment of the distribution:

        .. math::

            \text{pm2}(x) = \left\{\begin{array}{ll}
                    \frac{5}{20}\left(2 + \frac{1}{\sqrt{5}}x^3 -
                                      \frac{3}{5^{5/2}}x^5 \right)
                    & \text{, if } x\in[-\sqrt{5}:\sqrt{5}] \\
                    0 & \text{, if } x < -\sqrt{5} \\
                    1 & \text{, if } x > \sqrt{5}
                    \end{array}\right.
        """
        return _cy_kernels.epanechnikov_pm2(xs, out)

    def rfft(self, N, dx, out=None):
        r"""
        FFT of the Epanechnikov kernel:

        .. math::

            \text{FFT}(w) = \frac{3}{w'^3}\left( \sin w' - w' \cos w' \right)

        where :math:`w' = w\sqrt{5}`
        """
        z = rfftfreq(N, dx)
        return _cy_kernels.epanechnikov_fft(z, out)

    def rfft_xfx(self, N, dx, out=None):
        r"""
        .. math::

            \text{FFT}(w E(w)) = \frac{3\sqrt{5}i}{w'^4}\left( 3 w' \cos w' -
                3 \sin w' + w'^2 \sin w' \right)

        where :math:`w' = w\sqrt{5}`
        """
        z = rfftfreq(N, dx)
        return _cy_kernels.epanechnikov_fft_xfx(z, out)

    def dct(self, N, dx, out=None):
        z = dctfreq(N, dx)
        return _cy_kernels.epanechnikov_fft(z, out)


class EpanechnikovOrder4(Kernel1D):
    r"""
    Order 4 Epanechnikov kernel. That is:

    .. math::

        K_{[4]}(x) = \frac{3}{2} K(x) + \frac{1}{2} x K'(x) =
            -\frac{15}{8}x^2+\frac{9}{8}

    where :math:`K` is the non-normalized Epanechnikov kernel.
    """

    upper = 1
    lower = -upper
    cut = upper

    def pdf(self, xs, out=None):
        return _cy_kernels.epanechnikov_o4_pdf(xs, out)

    __call__ = pdf

    def cdf(self, xs, out=None):
        return _cy_kernels.epanechnikov_o4_cdf(xs, out)

    def pm1(self, xs, out=None):
        return _cy_kernels.epanechnikov_o4_pm1(xs, out)

    def pm2(self, xs, out=None):
        return _cy_kernels.epanechnikov_o4_pm2(xs, out)


class GaussianOrder4(Kernel1D):
    r"""
    Order 4 Normal kernel. That is:

    .. math::

        \phi_{[4]}(x) = \frac{3}{2} \phi(x) + \frac{1}{2} x \phi'(x) =
            \frac{1}{2}(3-x^2)\phi(x)

    where :math:`\phi` is the Gaussian kernel.

    """

    lower = -np.inf
    upper = np.inf
    cut = 3.

    def pdf(self, xs, out=None):
        return _cy_kernels.normal_o4_pdf(xs, out)

    __call__ = pdf

    def cdf(self, xs, out=None):
        return _cy_kernels.normal_o4_cdf(xs, out)

    def pm1(self, xs, out=None):
        return _cy_kernels.normal_o4_pm1(xs, out)

    def pm2(self, xs, out=None):
        return _cy_kernels.normal_o4_pm2(xs, out)
