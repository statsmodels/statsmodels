"""
This module implements the Reflection1D KDE estimation method, using DCT to
speed up computation on grids.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from .kde_utils import numpy_trans1d_method, finite
from ._kde1d_methods import KDE1DMethod, dctdensity, dctdensity_from_binned


class Reflection1D(KDE1DMethod):
    r"""
    This method consist in simulating the reflection of the data left and
    right of the boundaries. If one of the boundary is infinite, then the
    data is not reflected in that direction. To this purpose, the kernel is
    replaced with:

    .. math::

        \hat{K}(x; X, h, L, U) \triangleq K(z)
        + K\left(\frac{x+X-2L}{h}\right)
        + K\left(\frac{x+X-2U}{h}\right)

    where:

    .. math::

        z = \frac{x-X}{h}


    See the :py:mod:`pyqt_fit.kde1d_methods` for a description of the various
    symbols.

    When computing grids, if the bandwidth is constant, the result is computing
    using CDT.
    """

    #: Name of the method, for presentation purposes
    name = 'reflection1d'

    def __init__(self):
        super(Reflection1D, self).__init__()

    @property
    def bin_type(self):
        """
        Type of the bins adapted for the method
        """
        return 'r'

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

        # Make sure points are between the bounds, with reflection if needed
        if any(points < self.lower) or any(points > self.upper):
            span = self.upper - self.lower
            points = points - (self.lower + span)
            points %= 2 * span
            points -= self.lower + span
            points = np.abs(points)

        bw = self.bandwidth * self.adjust

        z = (points - exog) / bw
        z1 = (points + exog) / bw
        L = self.lower
        U = self.upper

        kernel = self.kernel

        terms = kernel(z)

        if L > -np.inf:
            terms += kernel(z1 - (2 * L / bw))

        if U < np.inf:
            terms += kernel(z1 - (2 * U / bw))

        terms *= self.weights / bw
        terms.sum(axis=-1, out=out)
        out /= self.total_weights

        return out

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
            return KDE1DMethod.cdf(self, points, out)

        exog = self.exog
        points = points[..., np.newaxis]

        # Make sure points are between the bounds, with reflection if needed
        if any(points < self.lower) or any(points > self.upper):
            span = self.upper - self.lower
            points = points - (self.lower + span)
            points %= 2 * span
            points -= self.lower + span
            points = np.abs(points)

        bw = self.bandwidth * self.adjust

        z = (points - exog) / bw
        z1 = (points + exog) / bw
        L = self.lower
        U = self.upper

        kernel = self.kernel

        terms = kernel.cdf(z)

        if L > -np.inf:
            # Remove the truncated part on the left
            terms -= kernel.cdf((L - exog) / bw)
            # Add the reflected part
            terms += kernel.cdf(z1 - (2 * L / bw))
            # Remove the truncated part from the reflection
            terms -= kernel.cdf((exog - L) / bw)

        if U < np.inf:
            # Add the reflected part
            terms += kernel.cdf(z1 - (2 * U / bw))

        terms *= self.weights
        terms.sum(axis=-1, out=out)
        out /= self.total_weights

        out[points[..., 0] >= self.upper] = 1
        out[points[..., 0] <= self.lower] = 0

        return out

    def grid(self, N=None, cut=None, span=None):
        """
        DCT-based estimation of KDE estimation, i.e. with reflection boundary
        conditions. This works only for fixed bandwidth (i.e. adjust = 1) and
        gaussian kernel.

        For open domains, the grid is taken with 3 times the bandwidth as extra
        space to remove the boundary problems.
        """
        if self.adjust.shape:
            return KDE1DMethod.grid(self, N, cut)

        bw = self.bandwidth * self.adjust
        exog = self.exog
        N = self.grid_size(N)

        if span is None:
            lower = self.lower
            upper = self.upper
        else:
            lower, upper = span

        if cut is None:
            cut = self.kernel.cut

        if not finite(lower):
            lower = np.min(exog) - cut * self.bandwidth
        if not finite(upper):
            upper = np.max(exog) + cut * self.bandwidth

        weights = self.weights

        return dctdensity(exog, self.kernel.dct, bw, lower, upper, N, weights,
                          self.total_weights)

    def from_binned(self, mesh, binned, normed=False, dim=-1):
        """
        Evaluate the PDF from data already binned. The binning might have been
        high-dimensional but must be of the same data.

        Parameters
        ----------
        mesh: grid.Grid
            Grid of the binning
        bins: ndarray
            Array of the same shape as the mesh with the values per bin
        normed: bool
            If true, the result will be normed w.r.t. the total weight of the
            exog
        dim: int
            Dimension along which the estimation must be done

        Returns
        -------
        ndarray
            Array of same size as bins, but with the estimated of the PDF for
            each line along the dimension `dim`
        """
        if self.adjust.ndim:
            raise ValueError("Error, cannot use binned data with non-constant "
                             "adjustment.")
        return dctdensity_from_binned(mesh, binned, self.kernel.dct,
                                      self.bandwidth*self.adjust, normed,
                                      self.total_weights, dim=dim)

    def grid_size(self, N=None):
        """
        Returns a valid grid size.
        """
        if N is None:
            if self.adjust.shape:
                return 2 ** 10
            return 2 ** 16
        return N  # 2 ** int(np.ceil(np.log2(N)))
