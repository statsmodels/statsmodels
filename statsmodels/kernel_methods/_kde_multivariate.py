"""
:Author: Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>

This module contains the multi-variate KDE meta-method.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import copy as shallow_copy

import numpy as np

from ..compat.python import range
from . import kernels
from . import _kdenc_methods, _kde1d_reflection
from .kde_utils import numpy_trans_method, AxesType, namedtuple
from ._kde_methods import KDEMethod, filter_exog
from .bandwidths import KDE1DAdaptor
from .fast_linbin import fast_linbin_nd as fast_bin_nd

AxesMethods = namedtuple("AxesMethods", ["methods", "kernels"])


def _compute_bandwidth(kde, default):
    """
    Compute the bandwidth and covariance for the estimated model, based of its
    exog attribute
    """
    n = kde.ndim
    if kde.bandwidth is not None:
        bw = kde.bandwidth
    else:
        bw = default
    if callable(bw):
        bw = bw(kde)
    else:
        adapt = KDE1DAdaptor(kde)
        for i in range(n):
            local_bw = bw[i]
            if callable(local_bw):
                adapt.axis = i
                local_bw = float(local_bw(adapt))
            else:
                local_bw = float(local_bw)
            bw[i] = local_bw
    bw = np.asarray(bw, dtype=float)
    if bw.shape != (n,):
        raise ValueError("Error, there must be one bandwidth per variable")
    return bw


class Multivariate(KDEMethod):
    """
    This class works as an adaptor for various 1D methods to work together.

    Parameters
    ----------
    **kwords: dict
        Can be used to set at construction time any attribute
    """
    def __init__(self, **kwords):
        KDEMethod.__init__(self)
        self._bin_data = None
        self.base_p2 = 16
        self._methods = {}
        self._kernels = {}
        self._kernels_type = dict(c=kernels.normal1d(),
                                  o=kernels.WangRyzin(),
                                  u=kernels.AitchisonAitken())
        self._methods_type = dict(c=_kde1d_reflection.Reflection1D(),
                                  o=_kdenc_methods.Ordered(),
                                  u=_kdenc_methods.Unordered())
        for k in kwords:
            if hasattr(self, k):
                setattr(self, k, kwords[k])
            else:
                raise ValueError("Error, unknown attribute '{}'".format(k))

    def copy(self):
        """
        Creates a shallow copy of the object
        """
        return shallow_copy(self)

    @property
    def kernels(self):
        """
        Kernels for earch dimension.

        Before fitting, this should be a dictionary, associating for some
        dimensions the kernel you want. Any dimension non-present in this
        dictionary will be given a default kernel depending on its axis type.
        """
        return self._kernels

    @property
    def continuous_method(self):
        """
        Default method for continuous axes
        """
        return self._methods_type['c']

    @continuous_method.setter
    def continuous_method(self, m):
        self._methods_type['c'] = m

    @property
    def ordered_method(self):
        """
        Default method for ordered axes
        """
        return self._methods_type['o']

    @ordered_method.setter
    def ordered_method(self, m):
        self._methods_type['o'] = m

    @property
    def unordered_method(self):
        """
        Default method for unordered axes
        """
        return self._methods_type['u']

    @unordered_method.setter
    def unordered_method(self, m):
        self._methods_type['u'] = m

    @property
    def adjust(self):
        """
        Bandwidth adjustment values
        """
        return self._adjust

    @adjust.setter
    def adjust(self, val):
        try:
            self._adjust = np.asarray(float(val))
        except TypeError:
            val = np.atleast_1d(val).astype(float)
            if val.shape != (self.npts,):
                raise ValueError("Error, adjust must be a single value or a "
                                 "value per point")
            self._adjust = val
        if self._methods:
            for m in self._methods:
                m.adjust = self._adjust

    @adjust.deleter
    def adjust(self):
        self.adjust = np.asarray(1.)

    @property
    def npts(self):
        """
        Number of points in the dataset
        """
        return self._exog.shape[0]

    @property
    def ndim(self):
        """
        Number of dimensions of the data
        """
        return self._exog.shape[1]

    @property
    def lower(self):
        """
        Lower bounds of the domain
        """
        return self._lower

    @property
    def upper(self):
        """
        Upper bound of the domain
        """
        return self._upper

    def get_methods(self, axis_types):
        """
        Get the list of methods and kernels for each axis (after fitting only)

        Returns
        -------
        methods: list of methods
            Methods per axis
        kernels: list of kernels
            Kernel per axis
        """
        m = [None] * len(axis_types)
        k = [None] * len(axis_types)
        for i, t in enumerate(axis_types):
            try:
                m[i] = self._methods[i]
            except (IndexError, KeyError):
                m[i] = self._methods_type[t].copy()
            try:
                k[i] = self._kernels[i]
            except (IndexError, KeyError):
                k[i] = self._kernels_type[t].for_ndim(1)
        return AxesMethods(m, k)

    @property
    def methods(self):
        """
        Methods for each axes.

        Before fitting, this should be a dictionnary specifying the methods for
        the axes that won't use the default ones.

        After fitting, this is a list of methods, one per axis.
        """
        return self._methods

    def fit(self, kde):
        if len(kde.axis_type) == 1:
            axis_type = AxesType(kde.axis_type[0] * kde.ndim)
        else:
            axis_type = AxesType(kde.axis_type)
        if len(axis_type) != kde.ndim:
            raise ValueError("You must specify exacltly one axis type, or as "
                             "many as there are axis")
        methods, kernels = self.get_methods(axis_type)
        ndim = kde.ndim
        if ndim == 1:
            if kde.bandwidth is None:
                kde = kde.copy()
                if methods[0].axis_type == 'o':
                    kde.bandwidth = self.bandwidth.ordered
                elif methods[0].axis_type == 'u':
                    kde.bandwidth = self.bandwidth.unordered
                else:
                    kde.bandwidth = self.bandwidth.continuous
            return methods[0].fit(kde)
        bin_type = ''.join(m.bin_type for m in methods)
        self._bin_type = bin_type
        kde = filter_exog(kde, bin_type)

        bw = _compute_bandwidth(kde, self.bandwidth)

        new_kde = kde.copy()
        new_kde.bandwidth = bw
        adapt = KDE1DAdaptor(new_kde)
        bin_data = None
        for d, m in enumerate(methods):
            adapt.axis = d
            f = m.fit(adapt)
            methods[d] = f
            if f.to_bin is not None:
                bin_data = True

        fitted = self.copy()
        if hasattr(fitted, 'bandwidth'):
            del fitted._bandwidth
        fitted._axis_type = axis_type
        fitted._kernels = kernels
        fitted._exog = kde.exog
        fitted._methods = methods
        fitted._lower = np.array([m.lower for m in fitted.methods])
        fitted._upper = np.array([m.upper for m in fitted.methods])
        fitted._bin_data = bin_data
        fitted._weights = kde.weights
        fitted._adjust = kde.adjust
        fitted._total_weights = kde.total_weights
        fitted._fitted = True
        return fitted

    @property
    def bandwidth(self):
        if self._fitted:
            result = np.empty((self.ndim,), dtype=float)
            for d in range(self.ndim):
                result[d] = self.methods[d].bandwidth
            return result
        return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, value):
        if self._fitted:
            value = np.asarray(value)
            if value.shape != (self.ndim,):
                raise ValueError("The shape of the bandwidth must be (D,)")
            for d in range(self.ndim):
                self.methods[d].bandwidth = value[d]
        else:
            self._bandwidth = value

    @property
    def bin_type(self):
        """
        Type of bin for each dimension (see :py:class:`.kde_utils.Grid`)
        """
        return self._bin_type

    @numpy_trans_method('ndim', 1)
    def pdf(self, points, out):
        """
        Compute the PDF of the distribution

        Parameters
        ----------
        points: ndarray of shape (M,D)
            Points on which the PDF should be evaluated
        out: ndarray of shape (M,)
            Array in which the result is stored

        Returns
        -------
        out: ndarray of shape (M,)
            For each input point, the value of the PDF on that point
        """
        full_out = np.empty_like(points)
        for i in range(self.ndim):
            self._methods[i].pdf(points[:, i], out=full_out[:, i])
        np.prod(full_out, axis=1, out=out)
        return out

    def __call__(self, points, out=None):
        """
        Alias for :py:meth:`pdf`
        """
        return self.pdf(points, out)

    @property
    def to_bin(self):
        """
        Property holding the data to be binned. It is different from
        :py:attr:`exog` if any method provide this.
        """
        if self._bin_data is not None:
            if self._bin_data is True:
                self._bin_data = self._exog.copy()
                for d, m in enumerate(self.methods):
                    if m.to_bin is not None:
                        self._bin_data[:, d] = m.to_bin
            return self._bin_data
        return self._exog

    def update_inputs(self, exog, weights=1., adjust=1.):
        """
        Update the inputs from a consistent set of data, weights and
        adjustments
        """
        exog = np.atleast_2d(exog)
        if exog.ndim > 2 or exog.shape[1] != self.ndim:
            raise ValueError("Error, wrong number of dimensions for exog, "
                             "this cannot be changed after fitting!")
        npts, ndim = exog.shape
        weights = np.asarray(weights, dtype=float).squeeze()
        if weights.ndim > 1:
            raise ValueError("Error, weights must be at most a 1D array")
        if weights.ndim == 1 and weights.shape != (npts,):
            raise ValueError("Error, weights must be a single value or have "
                             "as many values as points in exog")
        adjust = np.asarray(adjust, dtype=float).squeeze()
        if adjust.ndim > 1:
            raise ValueError("Error, adjust must be at most a 1D array")
        if adjust.ndim == 1 and adjust.shape != (npts,):
            raise ValueError("Error, adjust must be a single value or have as "
                             "many values as points in exog")
        self._exog = exog
        self._weights = weights
        self._adjust = adjust
        bin_data = None
        for d, m in enumerate(self.methods):
            m.update_inputs(exog[:, d], weights, adjust)
            if m.to_bin is not None:
                bin_data = True
        self._bin_data = bin_data

    def grid_size(self, N=None):
        """
        Compute an acceptable size for a grid
        """
        if N is None:
            p2 = self.base_p2 // self.ndim
            if self.base_p2 % self.ndim > 0:
                p2 += 1
            return 2 ** p2
        return N

    def grid(self, N=None, cut=None):
        """
        Compute the PDF on a grid

        Parameters
        ----------
        N: int or tuple of int
            Number of bins on each dimension. If a single number is used, this
            is valid for each dimension.
        cut: float or tuple of float
            If defined, override the cutting value for each dimension. If a
            tuple is defined, each non-None value override a specific
            dimension.

        Returns
        -------
        mesh: `statsmodels.kernel_methods.kde_utils.Grid`
            Grid on which the PDF has been evaluated
        values: ndarray
            Values of the PDF on the mesh.
        """
        to_bin = self.to_bin
        bin_type = ''.join(m.bin_type for m in self.methods)
        bounds = np.c_[self.lower, self.upper]

        if cut is None:
            cut = [getattr(m.kernel, 'cut', None) for m in self.methods]
        elif np.isscalar(cut):
            cut = [cut] * self.ndim

        for d in range(self.ndim):
            m = self.methods[d]
            if m.transform_axis is not None:
                lower, upper = m.transform_axis(bounds[d])
            else:
                lower, upper = bounds[d]
            if lower == -np.inf:
                bounds[d, 0] = to_bin[:, d].min() - cut[d] * m.bandwidth
            if upper == np.inf:
                bounds[d, 1] = to_bin[:, d].max() + cut[d] * m.bandwidth

        N = self.grid_size(N)
        mesh, binned = fast_bin_nd(to_bin, bounds, N, self.weights, bin_type)
        binned /= self._total_weights

        for d, m in enumerate(self.methods):
            binned = m.from_binned(mesh, binned, dim=d)

        for d, m in enumerate(self.methods):
            if m.transform_bins is not None:
                binned = m.transform_bins(mesh, binned, axis=d)
        mesh.transform([m.restore_axis for m in self.methods])

        return mesh, binned
