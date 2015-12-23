"""
:Author: Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>

This modules contains a set of methods to compute KDEs on non-continuous data.
"""

from __future__ import division, absolute_import, print_function
import numpy as np
from .kde_utils import numpy_trans1d_method, finite
from .fast_linbin import fast_linbin as fast_bin
from copy import copy as shallow_copy
from ._kde_methods import KDEMethod, filter_exog
from . import kernels


def _compute_bandwidth(kde):
    """
    Compute the bandwidth and covariance for the estimated model, based of its exog attribute
    """
    if kde.bandwidth is not None:
        if callable(kde.bandwidth):
            bw = float(kde.bandwidth(kde))
        else:
            bw = float(kde.bandwidth)
        return bw
    raise ValueError("Bandwidth needs to be specified")


class Unordered(KDEMethod):
    """
    Discrete, unordered, univariate method.
    """
    def __init__(self):
        KDEMethod.__init__(self)
        self._exog = None
        self._num_levels = None
        self._weights = None
        self._total_weights = None
        self._bw = None
        self._kernel = kernels.AitchisonAitken()
        self._epsilon = 1e-6

    name = 'unordered'

    @property
    def axis_type(self):
        return 'u'

    @axis_type.setter
    def axis_type(self, value):
        if value != 'u':
            raise ValueError('Error, this method can only be used for discrete unordered axis')

    @property
    def bin_type(self):
        return 'd'

    @property
    def to_bin(self):
        return None

    @property
    def epsilon(self):
        """
        Precision required for the computation of the bounds and number of levels.

        This is used to compute the upper and lower bound while fitting the
        KDE, if they are not already provided.
        """
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        if self._fitted:
            raise RuntimeError("You cannot change epsilon after fitting")
        value = float(value)
        if value <= 0:
            raise ValueError('epsilon must be strictly positive')
        self._epsilon = value

    def fit(self, kde, compute_bandwidth=True):
        if kde.ndim != 1:
            raise ValueError("Error, this method only accepts one variable problem")
        if kde.axis_type != self.axis_type:
            raise ValueError("Error, this method only accepts an unordered discrete axis")
        kde = filter_exog(kde, self.bin_type)
        fitted = self.copy()
        fitted._fitted = True
        fitted._exog = kde.exog.reshape((kde.npts,))
        if compute_bandwidth:
            fitted._bw = _compute_bandwidth(kde)
        if kde.kernel is not None:
            fitted._kernel = kde.kernel.for_ndim(1)
        fitted._weights = kde.weights
        fitted._adjust = kde.adjust
        fitted._total_weights = kde.total_weights
        cut = fitted.cut()
        if not finite(kde.upper):
            upper = int(np.round(fitted._exog.max())) + cut
        else:
            upper = int(np.round(kde.upper))
        if not finite(kde.lower):
            lower = int(np.round(fitted._exog.min())) - cut
        else:
            lower = int(np.round(kde.lower))
        fitted._lower = lower
        fitted.upper = upper
        return fitted

    def copy(self):
        return shallow_copy(self)

    @property
    def adjust(self):
        return self._adjust

    @adjust.setter
    def adjust(self, val):
        try:
            self._adjust = np.asarray(float(val))
        except TypeError:
            val = np.atleast_1d(val).astype(float)
            assert val.shape == (self.npts,), "Adjust must be a single values or a 1D array with value per input point"
            self._adjust = val

    @adjust.deleter
    def adjust(self):
        self._adjust = np.asarray(1.)

    @property
    def ndim(self):
        """
        Dimension of the problem
        """
        return 1

    @property
    def npts(self):
        """
        Number of points in the setup
        """
        return self._exog.shape[0]

    @property
    def bandwidth(self):
        """
        Selected bandwidth.

        Unlike the bandwidth for the KDE, this must be an actual value and not a method.
        """
        return self._bw

    @bandwidth.setter
    def bandwidth(self, val):
        val = float(val)
        assert val > 0, "The bandwidth must be strictly positive"
        self._bw = val

    def update_inputs(self, exog, weights=1., adjust=1.):
        """
        Update all the variable lengths inputs at once to ensure consistency
        """
        exog = np.atleast_1d(exog)
        if exog.ndim != 1:
            raise ValueError("Error, exog must be a 1D array (nb dimensions: {})".format(exog.ndim))
        weights = np.asarray(weights)
        adjust = np.asarray(adjust)
        if weights.ndim != 0 and weights.shape != exog.shape:
            raise ValueError("Error, weights must be either a single number, or an array the same shape as exog")
        if adjust.ndim != 0 and adjust.shape != exog.shape:
            raise ValueError("Error, adjust must be either a single number, or an array the same shape as exog")
        self._exog = exog
        self._weights = weights
        self._adjust = adjust
        if weights.ndim > 0:
            self._total_weights = weights.sum()
        else:
            self._total_weights = self.npts

    @property
    def exog(self):
        """
        Input points.

        Notes
        -----
        At that point, you are not allowed to change the number of exogenous points.
        """
        return self._exog

    @exog.setter
    def exog(self, value):
        if self._fitted:
            value = np.atleast_1d(value).reshape(self._exog.shape)
        else:
            value = np.atleast_1d(value).squeeze()
            if value.ndim > 1:
                raise ValueError("Error, the exog data points must be 1D")
        self._exog = value

    @property
    def kernel(self):
        """
        Kernel used for the estimation
        """
        return self._kernel

    @kernel.setter
    def kernel(self, ker):
        self._kernel = ker

    @property
    def weights(self):
        """
        Weights for the points in ``KDE1DMethod.exog``
        """
        return self._weights

    @weights.setter
    def weights(self, val):
        val = np.asarray(val)
        if val.shape:
            val = val.reshape(self._exog.shape)
            self._weights = val
            self._total_weights = val.sum()
        else:
            self._weights = np.asarray(1.)
            self._total_weights = None

    @property
    def total_weights(self):
        """
        Sum of the point weights
        """
        return self._total_weights

    @property
    def num_levels(self):
        return self._num_levels

    @num_levels.setter
    def num_levels(self, nl):
        nl = int(nl)
        if nl <= 1:
            raise ValueError("Error, there must be at least 2 levels")
        self._num_levels = nl

    @property
    def lower(self):
        return self._lower

    @lower.setter
    def lower(self, value):
        value = int(value)
        self._lower = value

    @property
    def upper(self):
        if self.num_levels:
            return self.num_levels - 1 + self._lower
        return np.inf

    @upper.setter
    def upper(self, value):
        if finite(value):
            self.num_levels = int(value) + 1 - self._lower
        else:
            self._num_levels = None

    @numpy_trans1d_method()
    def pdf(self, points, out):
        """
        Compute the PDF of the estimated distributiom

        Parameters
        ----------
        """
        if self.adjust.ndim:
            raise NotImplemented("This method cannot handle adjustments")
        points = points[:, None]
        kpdf = self.kernel.pdf(points, self.exog, self.bandwidth, self.num_levels)
        kpdf *= self.weights
        kpdf.sum(axis=-1, out=out)
        out /= self.total_weights
        return out

    def __call__(self, points, out=None):
        return self.pdf(points, out)

    def grid(self, N=None, cut=None):
        """
        Create a grid with all the values, in this implementation N and cut are ignored and are present only for
        compatibility with the continuous version.
        """
        if self.adjust.ndim:
            raise NotImplemented("This method cannot handle adjustments")
        weights = self.weights
        mesh, bins = fast_bin(self._exog, [self.lower, self.upper], self.num_levels, weights=weights, bin_type='d')
        return mesh, self.from_binned(mesh, bins, True)

    def cut(self):
        return self._kernel.cut(self.bandwidth*self.adjust.max(), self.epsilon)

    def from_binned(self, mesh, bins, normed=False, dim=-1):
        result = self.kernel.from_binned(mesh, bins, self.bandwidth, dim)
        if normed:
            result /= self.total_weights
        return result

    transform_axis = None
    restore_axis = None
    transform_bins = None


class Ordered(Unordered):
    """
    Discrete, ordered, univariate method.
    """
    def __init__(self):
        Unordered.__init__(self)
        self._kernel = kernels.WangRyzin()

    name = "ordered"

    @property
    def axis_type(self):
        return 'o'

    @axis_type.setter
    def axis_type(self, value):
        if value != 'o':
            raise ValueError('Error, this method can only be used for discrete ordered axis')

    @property
    def bin_type(self):
        return 'd'

    def grid_cdf(self, N=None, cut=None):
        """
        Compute the CDF of the distribution on a grid

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
            Grid on which the CDF has bin evaluated
        values : ndarray
            Values of the CDF for each position of the grid.

        """
        mesh, bins = self.grid()
        return mesh, np.cumsum(bins, out=bins)

    @numpy_trans1d_method()
    def cdf(self, points, out):
        """
        Compute the CDF of the distribution.

        Parameters
        ----------
        points: ndarray
            Points to evaluate the CDF on
        out: ndarray
            Result object. If must have the same shapes as ``points``

        Returns
        -------
        out: ndarray
            The default implementation uses the formula for unbounded CDF
            computation.
        """
        _, bins = self.grid_cdf()
        out[...] = bins[points]
        return out
