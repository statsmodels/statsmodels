"""
This module implements the Linear Combination KDE estimation method, which is a
1st order approximation of the KDE at the boundaries.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from .kde_utils import numpy_trans1d_method, finite, Grid
from ._kde1d_cyclic import Cyclic1D
from ._kde1d_methods import KDE1DMethod, fftdensity
from .kernels import Kernel1D


class _LinearCombinationKernel(Kernel1D):
    def __init__(self, ker):
        self._kernel = ker

    def pdf(self, x, out=None):
        """
        Compute the PDF of the estimated distribution.

        Parameters
        ----------
        points: ndarray
            Points to evaluate the distribution on
        out: ndarray
            Result object. If must have the same shapes as ``points``

        Returns
        -------
        out: ndarray
            Returns the PDF for each point. The default is to use the formula
            for unbounded pdf computation using the :py:func:`convolve`
            function.
        """
        out = self._kernel(x, out)
        out *= x
        return out

    __call__ = pdf


class LinearCombination(Cyclic1D):
    r"""
    This method uses the linear combination correction published in [KM1]_.

    The estimation is done with a modified kernel given by:

    .. math::

        \hat{K}(x;X,h,L,U) \triangleq \frac{a_2(l,u) -
        a_1(-u, -l) z}{a_2(l,u)a_0(l,u) - a_1(-u,-l)^2} K(z)

    where:

    .. math::

        z = \frac{x-X}{h} \qquad l = \frac{L-x}{h} \qquad u = \frac{U-x}{h}

    .. [KM1] Jones, M. C. 1993. Simple boundary correction for kernel density
        estimation. Statistics and Computing 3: 135--146.
    """

    #: Name of the method, for presentation purposes
    name = 'linear combination1d'

    def __init__(self):
        super(LinearCombination, self).__init__()

    @numpy_trans1d_method()
    def pdf(self, points, out):
        """
        Compute the PDF of the estimated distribution.

        Parameters
        ----------
        points: ndarray
            Points to evaluate the distribution on
        out: ndarray
            Result object. If must have the same shapes as ``points``

        Returns
        -------
        out: ndarray
            Returns the PDF for each point. The default is to use the formula
            for unbounded pdf computation using the :py:func:`convolve`
            function.
        """
        if not self.bounded:
            return KDE1DMethod.pdf(self, points, out)

        exog = self.exog
        points = points[..., np.newaxis]

        bw = self.bandwidth * self.adjust

        lower = (self.lower - points) / bw
        upper = (self.upper - points) / bw
        z = (points - exog) / bw

        kernel = self.kernel

        a0 = kernel.cdf(upper) - kernel.cdf(lower)
        a1 = kernel.pm1(-lower) - kernel.pm1(-upper)
        a2 = kernel.pm2(upper) - kernel.pm2(lower)

        denom = a2 * a0 - a1 * a1
        upper = a2 - a1 * z

        upper /= denom
        upper *= (self.weights / bw) * kernel(z)

        upper.sum(axis=-1, out=out)
        out /= self.total_weights

        return out

    @property
    def bin_type(self):
        """
        Type of the bins adapted for the method
        """
        return 'b'

    def cdf(self, points, out=None):
        r"""
        Compute the CDF of the estimated distribution, defined as:

        .. math::

            cdf(x) = P(X \leq x) = \int_l^x p(t) dt

        where :math:`l` is the lower bound of the distribution domain and
        :math:`p` the density of probability

        Parameters
        ----------
        points: ndarray
            Points to evaluate the CDF on
        out: ndarray
            Result object. If must have the same shapes as ``points``

        Returns
        -------
        out: ndarray
            The CDF for the points parameters
        """
        if not self.bounded:
            return super(LinearCombination, self).cdf(points, out)
        return self.numeric_cdf(points, out)

    def grid(self, N=None, cut=None, span=None):
        """
        Evaluate the PDF of the distribution on a regular grid with at least
        ``N`` elements.

        Parameters
        ----------
        N: int
            minimum number of element in the returned grid. Most
            methods will want to round it to the next power of 2.
        cut: float
            for unbounded domains, how far from the last data
            point should the grid go, as a fraction of the bandwidth.
        span: (float, float)
            If specified, fix the lower and upper bounds of the grid on which
            the PDF is computer. *If the KDE is bounded, you should always use
            the bounds as border*.

        Returns
        -------
        mesh : :py:class:`Grid`
            Grid on which the PDF has bin evaluated
        values : ndarray
            Values of the PDF for each position of the grid.
        """
        if self.adjust.shape:
            return KDE1DMethod.grid(self, N, cut)
        if not self.bounded:
            return super(LinearCombination, self).grid(N, cut)

        if cut is None:
            cut = self.kernel.cut
        N = self.grid_size(N)

        bw = self.bandwidth * self.adjust
        exog = self.exog
        weights = self.weights
        kernel = self.kernel

        # Range on which the density is to be estimated
        lower = self.lower
        upper = self.upper
        if span is None:
            est_lower, est_upper = lower, upper
        else:
            est_lower, est_upper = span
        if not finite(lower):
            est_lower = exog.min() - cut * self.bandwidth
        if not finite(upper):
            est_upper = exog.max() + cut * self.bandwidth
        est_R = est_upper - est_lower

        # Compute the FFT with enough margin to avoid side effects
        # Here we assume that bw << est_R / 8 otherwise the FFT approximation
        # is bad anyway
        shift_N = N // 8
        comp_N = N + N // 4
        comp_lower = est_lower - est_R / 8
        comp_upper = est_upper + est_R / 8
        total_weights = self.total_weights

        mesh, density = fftdensity(exog, kernel.rfft, bw, comp_lower,
                                   comp_upper, comp_N, weights, total_weights)
        _, z_density = fftdensity(exog, kernel.rfft_xfx, bw, comp_lower,
                                  comp_upper, comp_N, weights, total_weights)

        grid = mesh.full()
        grid = grid[shift_N:shift_N + N]
        density = density[shift_N:shift_N + N]
        z_density = z_density[shift_N:shift_N + N]

        # Apply linear combination approximation
        lower = (lower - grid) / bw
        upper = (upper - grid) / bw
        a0 = kernel.cdf(upper) - kernel.cdf(lower)
        a1 = kernel.pm1(-lower) - kernel.pm1(-upper)
        a2 = kernel.pm2(upper) - kernel.pm2(lower)

        density *= a2
        density -= a1 * z_density
        density /= a2 * a0 - a1 * a1

        return Grid(grid), density
