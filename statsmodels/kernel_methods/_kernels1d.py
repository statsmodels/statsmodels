from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from .kernels import Kernel1D, rfftfreq, dctfreq
from . import _cy_kernels


class tricube(Kernel1D):
    r"""
    Return the kernel corresponding to a tri-cube distribution, whose
    expression is.

    The tri-cube function is given by:

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

    def pdf(self, z, out=None):
        return _cy_kernels.tricube_pdf(z, out)

    __call__ = pdf

    upper = 1. / _cy_kernels.tricube_width
    lower = -upper
    cut = upper

    def cdf(self, z, out=None):
        r"""
        CDF of the distribution:

        .. math::

            \text{cdf}(x) = \left\{\begin{array}{ll}
                \frac{1}{162} {\left(60 (ax)^{7} - 7 {\left(2 (ax)^{10} +
                    15 (ax)^{4}\right)} \mathrm{sgn}\left(ax\right) + 140 ax +
                    81\right)} & \text{, if}x\in[-1/a;1/a] \\
                0 & \text{, if} x < -1/a \\
                1 & \text{, if} x > 1/a
                \end{array}\right.
        """
        return _cy_kernels.tricube_cdf(z, out)

    def pm1(self, z, out=None):
        r"""
        Partial moment of order 1:

        .. math::

            \text{pm1}(x) = \left\{\begin{array}{ll}
                \frac{7}{3564a} {\left(165 (ax)^{8} - 8 {\left(5 (ax)^{11} +
                    33 (ax)^{5}\right)} \mathrm{sgn}\left(ax\right) +
                    220 (ax)^{2} - 81\right)} & \text{, if} x\in [-1/a;1/a] \\
                0 & \text{, otherwise}
                \end{array}\right.
        """
        return _cy_kernels.tricube_pm1(z, out)

    def pm2(self, z, out=None):
        r"""
        Partial moment of order 2:

        .. math::

            \text{pm2}(x) = \left\{\begin{array}{ll}
                \frac{35}{486a^2} {\left(4 (ax)^{9} + 4 (ax)^{3} -
                    {\left((ax)^{12} + 6 (ax)^{6}\right)} \mathrm{sgn}
                    \left(ax\right) + 1\right)} & \text{, if} x\in[-1/a;1/a] \\
                0 & \text{, if } x < -1/a \\
                1 & \text{, if } x > 1/a
            \end{array}\right.
        """
        return _cy_kernels.tricube_pm2(z, out)

    def convolution(self, z, out=None):
        r"""
        Tricube convolution kernel:

        .. math::

            \left\{
            \begin{array}{cc}
                \frac{(x+2)^7 \left(
                    14 x^{12}-196 x^{11}+1568 x^{10}-8439 x^9+33474 x^8-98448
                    x^7+213558 x^6-334740 x^5+561120 x^4-453722 x^3+558880
                    x^2-206822 x+151470
                \right)}{12932920} & -2<x\leq -1 \\

                -\frac{(x-2)^7 \left(
                    14 x^{12}+196 x^{11}+1568 x^{10}+8439 x^9+33474 x^8+98448
                    x^7+213558 x^6+334740 x^5+561120 x^4+453722 x^3+558880
                    x^2+206822 x+151470
                \right)}{12932920} & 1\leq x<2 \\

                -\frac{3 x^{19}}{923780}-\frac{3 x^{16}}{40040}-
                \frac{111 x^{13}}{20020}-\frac{31 x^{10}}{140}-
                \frac{81 x^9}{70}-\frac{729 x^8}{220}-\frac{747 x^7}{140}-
                \frac{729 x^6}{182}+\frac{9 x^4}{5}-\frac{19683 x^2}{13090}+
                \frac{6561}{6916} & -1<x\leq 0 \\

                \frac{3 x^{19}}{923780}-\frac{3 x^{16}}{40040}+
                \frac{111 x^{13}}{20020}-\frac{31 x^{10}}{140}+
                \frac{81 x^9}{70}-\frac{729 x^8}{220}+\frac{747 x^7}{140}-
                \frac{729 x^6}{182}+\frac{9 x^4}{5}-\frac{19683 x^2}{13090}+
                \frac{6561}{6916} & 0<x<1
            \end{array} \right.
        """
        return _cy_kernels.tricube_convolution(z, out)


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
                \frac{3}{4} \left(1-x^2\right) & \text{, if} x \in [-1:1] \\
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

            \bar{f_r}(x) = \frac{3}{160}
                \left( (2-|x|)^3 (4 + 6|x| + x^2) \right)
                \qquad \text{, if } |x| < 2

        But of course, we need to normalize it in the same way:

        .. math::

            \bar{f}(x) = \frac{1}{\sqrt{5}}
                \bar{f_r}\left(\frac{x}{\sqrt{5}}\right)

        """
        return _cy_kernels.epanechnikov_convolution(xs, out)

    def cdf(self, xs, out=None):
        r"""
        CDF of the distribution.

        The CDF is defined on the interval :math:`[-\sqrt{5}:\sqrt{5}]` as:

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
                -\frac{3\sqrt{5}}{16}
                    \left(1-\frac{2}{5}x^2+\frac{1}{25}x^4\right)
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

            \text{FFT}(w E(w)) = \frac{3\sqrt{5}i}{w'^4}
                \left( 3 w' \cos w' - 3 \sin w' + w'^2 \sin w' \right)

        where :math:`w' = w\sqrt{5}`
        """
        z = rfftfreq(N, dx)
        return _cy_kernels.epanechnikov_fft_xfx(z, out)

    def dct(self, N, dx, out=None):
        z = dctfreq(N, dx)
        return _cy_kernels.epanechnikov_fft(z, out)


class Epanechnikov_order4(Kernel1D):
    r"""
    Order 4 Epanechnikov kernel. That is:

    .. math::

        K_{[4]}(x) = \frac{3}{2} K(x) + \frac{1}{2} x K'(x)
                   = -\frac{15}{8}x^2+\frac{9}{8}

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


class normal_order4(Kernel1D):
    r"""
    Order 4 Normal kernel. That is:

    .. math::

        \phi_{[4]}(x) = \frac{3}{2} \phi(x) + \frac{1}{2} x \phi'(x)
                      = \frac{1}{2}(3-x^2)\phi(x)

    where :math:`\phi` is the normal kernel.

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
