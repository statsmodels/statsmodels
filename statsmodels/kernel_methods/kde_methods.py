"""
:Author: Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>

This module contains the multi-variate KDE meta-method.
"""

import numpy as np
from .kde_utils import atleast_2df, AxesType, Grid
from . import bandwidths


def _array_arg(value, value_name, ndim, dtype=float):
    """
    Simple function returning an array with ndim values.

    If value is a single value, it is duplicated as needed.
    If value is a list of the wrong size, a ValueError is thrown.
    """
    value = np.asarray(value, dtype=dtype)
    if value.ndim == 0:
        return value * np.ones((ndim, ), dtype=dtype)
    if value.shape != (ndim, ):
        raise ValueError(
            "Error, '{0}' must be a scalar or a 1D array with {1} elements".
            format(value_name, ndim))
    return value


def filter_exog(kde, bin_type):
    """
    Filter the data to remove anything that is outside the bounds

    Parameters
    ----------
    kde: object with the fields exog, lower, upper, weights and adjust and a
         copy method This object must behave as a KDE object for those fields.
    bin_type: str
        String of length D with the bin types for each dimension

    Returns
    -------
    Either the kde object itself, or a copy with modified exog, weights and
    adjust properties
    """
    if any(b not in 'CRBD' for b in bin_type):
        raise ValueError(
            "bin_type must be one of 'C', 'R', 'B' or 'D'. Current value: {}".
            format(bin_type))
    exog = atleast_2df(kde.exog)
    sel = np.ones(exog.shape[0], dtype=bool)
    ndim = exog.shape[1]
    lower = np.atleast_1d(kde.lower)
    upper = np.atleast_1d(kde.upper)
    if lower.shape != (exog.shape[1], ) or upper.shape != (exog.shape[1], ):
        raise ValueError('Lower and upper bound must be at most a 1D ' +
                         'array with one value per dimension.')
    if len(bin_type) == 1:
        bin_type = bin_type * ndim
    for i in range(ndim):
        if bin_type[i] == 'B' or bin_type[i] == 'D':
            sel &= (exog[:, i] >= lower[i]) & (exog[:, i] <= upper[i])
    if np.all(sel):
        return kde
    k = kde.copy()
    k.exog = exog[sel]
    if kde.weights.shape:
        if kde.weights.shape != (exog.shape[0], ):
            raise ValueError("The weights must be either a single value or " +
                             "an array of shape (npts,)")
        k.weights = kde.weights[sel]
    if kde.adjust.shape:
        if kde.weights.shape != (exog.shape[0], ):
            raise ValueError("The adjustments must be either a single " +
                             "value or an array of shape (npts,)")
        k.adjust = kde.adjust[sel]
    return k


class KDEMethod(object):
    """
    This is the base class for KDE methods.

    Although inheriting from it is not required, it is recommended as it will
    provide quite a few useful services.

    Notes
    -----
    The kernel must be an object modeled on :py:class:`kernels.Kernels` or on
    :py:class:`kernels.Kernel1D` for 1D kernels. It is recommended to inherit
    one of these classes to provide numerical approximation for all methods.

    By default, the kernel class is :py:class:`pyqt_fit.kernels.Gaussian`
    """
    def __init__(self):
        self._exog = None
        self._upper = None
        self._lower = None
        self._axis_type = AxesType()
        self._kernel = None
        self._bandwidth = bandwidths.Multivariate()
        self._weights = None
        self._adjust = None
        self._total_weights = None
        self._fitted = False
        self._mask = slice(None)

    def __str__(self):
        return self.name

    @property
    def fitted(self):
        """
        Whether this is the output of the `KDE.fit` method or not.
        """
        return self._fitted

    @property
    def exog(self):
        """
        Exogenous data set. Its shape is NxD for N points in D dimension.
        """
        return self._exog

    @exog.setter
    def exog(self, value):
        value = atleast_2df(value).astype(float)
        if value.shape != self._exog.shape:
            raise ValueError(
                "Bad input change, you cannot change it after fitting")
        self._exog = value.reshape(self._exog.shape)

    @property
    def ndim(self):
        """
        Number of dimensions of the problem
        """
        if self._exog is None:
            return 1
        return self._exog.shape[1]

    @property
    def npts(self):
        """
        Number of points in the exogenous dataset.
        """
        return self._exog.shape[0]

    @property
    def kernel(self):
        r"""
        Kernel used for the density estimation.
        """
        return self._kernel

    @kernel.setter
    def kernel(self, k):
        self._kernel = k

    @property
    def axis_type(self):
        """
        AxesType
            Type of each axis. Each axis type is defined by a letter:
                - C for continuous
                - U for unordered (discrete)
                - O for ordered (discrete)
        """
        return self._axis_type

    @axis_type.setter
    def axis_type(self, value):
        self._axis_type.set(value)

    @axis_type.deleter
    def axis_type(self):
        self._axis_type[:] = 'C'

    @property
    def bandwidth(self):
        """
        Selected bandwidth.

        Unlike the bandwidth for the KDE, this must be an actual value and not
        a method.
        """
        return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, value):
        value = np.asarray(value)
        self._bandwidth = value.reshape(self._bandwidth.shape)

    @property
    def weights(self):
        """
        Weigths associated to each data point. It can be either a single value,
        or a 1D-array with a value per data point. If a single value is
        provided, the weights will always be set to 1.
        """
        return self._weights

    @weights.setter
    def weights(self, ws):
        try:
            ws = float(ws)
            self._weights = np.asarray(1.)
            if self._fitted:
                self._total_weights = self.npts
        except TypeError:
            ws = np.atleast_1d(ws).astype(float)
            ws = ws.reshape((self.npts, ))
            self._weights = ws
            if self._fitted:
                self._total_weights = sum(ws)

    @weights.deleter
    def weights(self):
        self._weights = np.asarray(1.)
        self._total_weights = self.npts

    @property
    def total_weights(self):
        """
        Sum of the weights of the data
        """
        return self._total_weights

    @property
    def adjust(self):
        """
        Adjustment of the bandwidth, per data point. It can be either a single
        value or an array with one value per data point. The real bandwidth
        then becomes: bandwidth * adjust

        When deleted, the adjustment is reset to 1.
        """
        return self._adjust

    @adjust.setter
    def adjust(self, ls):
        try:
            self._adjust = np.asarray(float(ls))
        except TypeError:
            ls = np.atleast_1d(ls).astype(float)
            ls = ls.reshape((self.npts, ))
            self._adjust = ls

    @adjust.deleter
    def adjust(self):
        self._adjust = np.asarray(1.)

    @property
    def lower(self):
        r"""
        Lower bound of the density domain. If deleted, becomes :math:`-\infty`
        on all dimension.

        Note that for discrete dimensions, the lower bounds will also be reset
        to 0.
        """
        return self._lower

    @lower.setter
    def lower(self, val):
        self._lower = np.atleast_1d(val)

    @lower.deleter
    def lower(self):
        self._lower = -np.inf * np.ones((self.ndim, ), dtype=float)

    @property
    def upper(self):
        r"""
        Upper bound of the density domain. If deleted, becomes :math:`\infty`
        on all dimensions

        Note that for discrete dimensions, if the upper dimension is 0, it will
        be set to the maximum observed element.
        """
        return self._upper

    @upper.setter
    def upper(self, val):
        self._upper = np.atleast_1d(val)

    @upper.deleter
    def upper(self):
        self._upper = np.inf * np.ones((self.ndim, ), dtype=float)

    @property
    def for_multivariate(self):
        """
        True if this method can be used in a Multivariate KDE estimation.

        To be available, the method must be separable, such that it is possible
        to compute the values for one axis using this method, and on the other
        axis using any other of the :py:func:`for_multivariate` method.

        The multivariate KDE needs two extra methods:

        - `pdf_contribution(self, points, out)` computes the contribution of
          each of the exogenous points to the value of the pdf at the given
          points, placing the output in `out`. If `points` is an array of `M`
          values and there are `N` exogenous points, `out` must be of shape
          `(N,M)`
        - `from_binned(self, mesh, binned, dim)` computes the KDE using the
          method on the already binned data (whose mesh is given), on the
          dimension `d`.
        """
        return hasattr(self, "pdf_contribution") and \
            hasattr(self, "from_binned")


def generate_grid(kde, N=None, cut=None):
    r"""
    Helper method returning a regular grid on the domain of the KDE.

    Parameters
    ----------
    kde: KDEnDMethod
        Fitted KDE object
    N: int or list of int
        Number of points in the grid
    cut: float
        For unbounded domains, how far past the maximum should
        the grid extend to, in term of KDE bandwidth

    Returns
    -------
    A vector of N regularly spaced points
    """
    N = np.asarray(kde.grid_size(N), dtype=int)
    if N.ndim == 0:
        N = N * np.ones(kde.ndim, dtype=int)
    elif N.ndim != 1 or N.shape[0] != kde.ndim:
        raise ValueError("N must be a single integer, or a 1D array with "
                         "as many element as dimensions in the KDE")
    if cut is None:
        cut = kde.kernel.cut
    if kde.bandwidth.ndim == 0:
        cut = kde.bandwidth * cut * np.ones(kde.ndim, dtype=float)
    elif kde.bandwidth.ndim == 1:
        cut = kde.bandwidth * cut
    else:
        cut = np.dot(kde.bandwidth, cut * np.ones(kde.ndim, dtype=float))
    lower = np.array(kde.lower)
    upper = np.array(kde.upper)
    ndim = kde.ndim
    axes = [None] * ndim
    for i in range(ndim):
        if lower[i] == -np.inf:
            lower[i] = np.min(kde.exog[:, i]) - cut[i]
        if upper[i] == np.inf:
            upper[i] = np.max(kde.exog[:, i]) + cut[i]
        axes[i] = np.linspace(lower[i], upper[i], N[i])
    return Grid(axes)
