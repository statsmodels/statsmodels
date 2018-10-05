"""
This module implements the Renormalization KDE estimation method, which is a 0-order correction at the boundaries.
"""
from __future__ import division, absolute_import, print_function

import numpy as np
from .kde_utils import numpy_trans1d_method, finite, Grid
from ._kde1d_cyclic import Cyclic1D
from ._kde1d_methods import KDE1DMethod, fftdensity


class Renormalization(Cyclic1D):
    r"""
    This method consists in using the normal kernel method, but renormalize
    to only take into account the part of the kernel within the domain of the
    density.

    The kernel is then replaced with:

    .. math::

        \hat{K}(x;X,h,L,U) \triangleq \frac{1}{a_0(u,l)} K(z)

    where:

    .. math::

        z = \frac{x-X}{h} \qquad l = \frac{L-x}{h} \qquad u = \frac{U-x}{h}

    """

    #: Name of the method, for presentation purposes
    name = 'renormalization1d'

    def __init__(self):
        super(Renormalization, self).__init__()

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
            for unbounded pdf computation using the :py:func:`convolve` function.
        """
        if not self.bounded:
            return Cyclic1D.pdf(self, points, out)

        exog = self.exog
        points = points[..., np.newaxis]

        bw = self.bandwidth * self.adjust

        l = (points - self.lower) / bw
        u = (points - self.upper) / bw
        z = (points - exog) / bw

        kernel = self.kernel

        a1 = (kernel.cdf(l) - kernel.cdf(u))

        terms = kernel(z) * ((self.weights / bw) / a1)

        terms.sum(axis=-1, out=out)
        out /= self.total_weights

        return out

    @property
    def bin_type(self):
        """
        Type of the bins adapted for the method
        """
        return 'b'

    @numpy_trans1d_method()
    def cdf(self, points, out):
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
            return super(Renormalization, self).cdf(points, out)
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
            return super(Renormalization, self).grid(N, cut)

        if cut is None:
            cut = self.kernel.cut
        N = self.grid_size(N)

        bw = self.bandwidth * self.adjust
        exog = self.exog

        if span is None:
            lower = self.lower
            upper = self.upper
        else:
            lower, upper = span

        if not finite(lower):
            lower = exog.min() - cut * self.bandwidth
        if not finite(upper):
            upper = exog.max() + cut * self.bandwidth

        R = upper - lower
        kernel = self.kernel

        # Compute the FFT with enough margin to avoid side effects
        # here we assume that bw << est_R / 8 otherwise the FFT approximation is bad anyway
        shift_N = N // 8
        comp_N = N + N // 4
        comp_lower = lower - R / 8
        comp_upper = upper + R / 8

        weights = self.weights

        mesh, density = fftdensity(exog, kernel.rfft, bw, comp_lower, comp_upper, comp_N, weights, self.total_weights)

        mesh = mesh.full()
        mesh = mesh[shift_N:shift_N + N]
        density = density[shift_N:shift_N + N]

        # Apply renormalization
        l = (mesh - lower) / bw
        u = (mesh - upper) / bw
        a1 = (kernel.cdf(l) - kernel.cdf(u))

        density /= a1

        return Grid(mesh, bounds=[lower, upper]), density
