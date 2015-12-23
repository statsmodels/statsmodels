"""
This module implements the Cyclic1D KDE estimation method, using FFT to speed up computation on grids.
"""
from __future__ import division, absolute_import, print_function

import numpy as np
from .kde_utils import numpy_trans1d_method, finite
from ._kde1d_methods import KDE1DMethod, fftdensity, fftdensity_from_binned


class Cyclic1D(KDE1DMethod):
    r"""
    This method assumes cyclic boundary conditions and works only for closed
    boundaries.

    The estimation is done with a modified kernel given by:

    .. math::

        \hat{K}(x; X, h, L, U) \triangleq K(z)
        + K\left(z - \frac{U-L}{h}\right)
        + K\left(z + \frac{U-L}{h}\right)

    where:

    .. math::

        z = \frac{x-X}{h}

    When computing grids, if the bandwidth is constant, the result is computing
    using FFT.
    """

    #: Name of the method, for presentation purposes
    name = 'cyclic1d'

    def __init__(self):
        super(Cyclic1D, self).__init__()

    @property
    def bin_type(self):
        """
        Type of the bins adapted for the method
        """
        return 'c'

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
            return KDE1DMethod.pdf(self, points, out)
        if not self.closed:
            raise ValueError("Cyclic boundary conditions can only be used with "
                             "closed or un-bounded domains.")

        exog = self.exog
        points = points[..., np.newaxis]

        # Make sure points are between the bounds
        if any(points < self.lower) or any(points > self.upper):
            points = points - self.lower
            points %= self.upper - self.lower
            points += self.lower

        bw = self.bandwidth * self.adjust

        z = (points - exog) / bw
        L = self.lower
        U = self.upper

        span = (U - L) / bw

        kernel = self.kernel

        terms = kernel(z)
        terms += kernel(z + span)  # Add points to the left
        terms += kernel(z - span)  # Add points to the right

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
        if not self.closed:
            raise ValueError("Cyclic boundary conditions can only be used with "
                             "closed or unbounded domains.")

        exog = self.exog
        points = np.atleast_1d(points)[..., np.newaxis]

        # Make sure points are between the bounds
        if any(points < self.lower) or any(points > self.upper):
            points = points - self.lower
            points %= self.upper - self.lower
            points += self.lower

        bw = self.bandwidth * self.adjust

        z = (points - exog) / bw
        L = self.lower
        U = self.upper

        span = (U - L) / bw

        kernel = self.kernel

        terms = kernel.cdf(z)
        terms -= kernel.cdf((L - exog) / bw)  # Remove the parts left of the lower bound

        terms += kernel.cdf(z + span)  # Repeat on the left
        terms -= kernel.cdf((L - exog) / bw + span)  # Remove parts left of lower bounds

        terms += kernel.cdf(z - span)  # Repeat on the right

        terms *= self.weights
        terms.sum(axis=-1, out=out)
        out /= self.total_weights

        out[points[..., 0] >= self.upper] = 1
        out[points[..., 0] <= self.lower] = 0

        return out

    def grid(self, N=None, cut=None, span=None):
        """
        FFT-based estimation of KDE estimation, i.e. with cyclic boundary
        conditions. This works only for closed domains, fixed bandwidth
        (i.e. adjust = 1) and gaussian kernel.
        """
        if self.adjust.shape:
            return KDE1DMethod.grid(self, N, cut)
        if self.bounded and not self.closed:
            raise ValueError("Error, cyclic boundary conditions require "
                             "a closed or un-bounded domain.")
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
        if not finite(upper):
            upper = np.max(exog) + cut * self.bandwidth
        if not finite(lower):
            lower = np.min(exog) - cut * self.bandwidth

        return fftdensity(exog, self.kernel.rfft, bw, lower, upper, N, self.weights, self.total_weights)

    def from_binned(self, mesh, binned, normed=False, dim=-1):
        """
        Evaluate the PDF from data already binned. The binning might have been high-dimensional but must be of the same
        data.

        Parameters
        ----------
        mesh: grid.Grid
            Grid of the binning
        bins: ndarray
            Array of the same shape as the mesh with the values per bin
        normed: bool
            If true, the result will be normed w.r.t. the total weight of the exog
        dim: int
            Dimension along which the estimation must be done

        Returns
        -------
        ndarray
            Array of same size as bins, but with the estimated of the PDF for each line along the dimension `dim`
        """
        if self.adjust.ndim:
            raise ValueError("Error, cannot use binned data with non-constant adjustment.")
        return fftdensity_from_binned(mesh, binned, self.kernel.rfft, self.adjust*self.bandwidth,
                                      normed, self.total_weights, dim)

    def grid_size(self, N=None):
        """
        Returns a valid grid size.
        """
        if N is None:
            if self.adjust.shape:
                return 2 ** 10
            return 2 ** 16
        return N  # 2 ** int(np.ceil(np.log2(N)))
