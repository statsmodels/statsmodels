r"""
:Author: Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>

This module contains a set of methods to compute multivariates KDEs.
"""

import numpy as np
from scipy import linalg, fftpack
from ..compat.python import range
from .kde_utils import numpy_trans_method, atleast_2df, Grid
from . import kernels
from copy import copy as shallow_copy
from .fast_linbin import fast_linbin_nd as fast_bin_nd
from ._kde_methods import KDEMethod, _array_arg, filter_exog
from ._kde1d_reflection import Reflection1D
from ._kde1d_cyclic import Cyclic1D


def generate_grid(kde, N=None, cut=None):
    r"""
    Helper method returning a regular grid on the domain of the KDE.

    Parameters
    ----------
    kde: KDE1DMethod
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
        raise ValueError("N must be a single integer, or a 1D array with as many element as dimensions in the KDE")
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


def _compute_bandwidth(kde, default):
    """
    Compute the bandwidth and covariance for the estimated model, based of its exog attribute
    """
    if kde.bandwidth is not None:
        bw = kde.bandwidth
    else:
        bw = default
    if callable(bw):
        bw = bw(kde)
    return bw


def fftdensity(exog, kernel_rfft, bw_inv, lower, upper, N, weights, total_weights):
    """
    Compute the density estimate using a FFT approximation.

    Parameters
    ----------
    exog: ndarray
        2D array with the data to fit
    kernel_rfft: function
        Function computing the rFFT for the kernel
    lower: float
        Lower bound on which to compute the density
    upper: float
        Upper bound on which to compute the density
    N: int or list of int
        Number of buckets to compute, for each dimension
    weights: ndarray or None
        Weights of the data, or None if they all have the same weight.
    total_weights: float
        Sum of the weights, or len(exog) if weights is None

    Returns
    -------
    mesh: ndarray
        Points on which the density had been evaluated
    density: ndarray
        Density evaluated on the mesh

    Notes
    -----
    No checks are made to ensure the consistency of the input!
    """
    mesh, DataHist = fast_bin_nd(exog, np.c_[lower, upper], N, weights=weights, bin_type='c')
    DataHist /= total_weights * mesh.start_volume
    FFTData = np.fft.rfftn(DataHist)

    dx = mesh.start_interval.copy()
    if bw_inv.ndim == 2:
        dx = np.dot(np.diag(dx), bw_inv)
    else:
        dx *= bw_inv

    smth = kernel_rfft(DataHist.shape, dx)

    SmoothFFTData = FFTData * smth
    density = np.fft.irfftn(SmoothFFTData, DataHist.shape)
    return mesh, density


def dctdensity(exog, kernel_dct, bw_inv, lower, upper, N, weights, total_weights):
    """
    Compute the density estimate using a FFT approximation.

    Parameters
    ----------
    exog: ndarray
        2D array with the data to fit
    kernel_dct: function
        Function computing the DCT for the kernel
    lower: float
        Lower bound on which to compute the density
    upper: float
        Upper bound on which to compute the density
    N: int or list of int
        Number of buckets to compute, for each dimension
    weights: ndarray or None
        Weights of the data, or None if they all have the same weight.
    total_weights: float
        Sum of the weights, or len(exog) if weights is None

    Returns
    -------
    mesh: ndarray
        Points on which the density had been evaluated
    density: ndarray
        Density evaluated on the mesh

    Notes
    -----
    No checks are made to ensure the consistency of the input!
    """
    mesh, DataHist = fast_bin_nd(exog, np.c_[lower, upper], N, weights=weights, bin_type='r')
    DataHist /= total_weights * mesh.start_volume
    FFTData = fftpack.dct(DataHist, axis=0)
    for a in range(1, DataHist.ndim):
        FFTData[:] = fftpack.dct(FFTData, axis=a)

    dx = mesh.start_interval.copy()
    dx *= bw_inv

    smth = kernel_dct(DataHist.shape, dx)

    SmoothFFTData = FFTData * smth
    density = fftpack.idct(SmoothFFTData, DataHist.shape, axis=0)
    for a in range(1, DataHist.ndim):
        density[:] = fftpack.idct(density, DataHist.shape, axis=a)
    return mesh, density


class KDEnDMethod(KDEMethod):
    """
    Base class providing a default grid method and a default method for unbounded evaluation of the PDF and CDF. It also
    provides default methods for the other metrics, based on PDF and CDF calculations.

    The default class can only deal with open, continuous, multivariate data.

    :Note:
        - It is expected that all grid methods will return the same grid if used with the same arguments.
        - It is fair to assume all array-like arguments will be at least 2D arrays, with the first dimension denoting
        the dimension.


    Attributes
    ----------
    base_p2: int
        Log2 of the number of points wanted in a grid.
    """

    name = 'unbounded'

    def __init__(self):
        KDEMethod.__init__(self)
        self._inv_bw = None
        self._det_inv_bw = None
        self.base_p2 = 10
        self._kernel = kernels.normal()

    def fit(self, kde, compute_bandwidth=True):
        """
        Extract the parameters required for the computation and returns a stand-alone estimator capable of performing
        most computations.

        Parameters
        ----------
        kde: pyqt_fit.kde.KDE
            KDE object being fitted
        compute_bandwidth: bool
            If true (default), the bandwidth is computed

        Returns
        -------
        An estimator object that doesn't depend on the KDE object.

        Notes
        -----
        By default, most values can be adjusted after estimation. However, it is not allowed to change the number of
        exogenous variables or the dimension of the problem.
        """
        ndim = kde.ndim
        if ndim == 1 and type(self) == KDEnDMethod:
            method = Reflection1D()
            return method.fit(kde, compute_bandwidth)

        kde = filter_exog(kde, self.bin_type)
        fitted = self.copy()
        fitted._fitted = True
        fitted._exog = kde.exog
        fitted._upper = _array_arg(kde.upper, 'upper', ndim)
        fitted._lower = _array_arg(kde.lower, 'lower', ndim)
        if np.any(kde.axis_type != 'c') or np.any(fitted.axis_type != kde.axis_type):
            raise ValueError("Error, all axis must be continuous")
        if kde.kernel is not None:
            fitted._kernel = kde.kernel.for_ndim(ndim)
        else:
            fitted._kernel = self.kernel.for_ndim(ndim)
        fitted._weights = kde.weights
        fitted._adjust = kde.adjust
        fitted._total_weights = kde.total_weights
        if compute_bandwidth:
            bw = _compute_bandwidth(kde, self.bandwidth)
            if bw is not None:
                fitted.bandwidth = bw
            else:
                raise ValueError("Error, no bandwidth has been specified")
        return fitted

    def copy(self):
        return shallow_copy(self)

    @property
    def axis_type(self):
        if len(self._axis_type) != self.ndim:
            self._axis_type.set('c' * self.ndim)
        return self._axis_type

    @property
    def bin_type(self):
        return 'b'*self.ndim

    @property
    def bandwidth(self):
        """
        Selected bandwidth.

        Unlike the bandwidth for the KDE, this must be an actual value and not a method.
        """
        return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, bw):
        if self._fitted:
            bw = np.asarray(bw).squeeze()
            if bw.ndim == 0:
                inv_bw = 1 / bw
                det_inv_bw = inv_bw ** self.ndim
            elif bw.ndim == 1:
                assert bw.shape[0] == self.ndim
                inv_bw = 1 / bw
                det_inv_bw = np.product(inv_bw)
            elif bw.ndim == 2:
                assert bw.shape == (self.ndim, self.ndim)
                inv_bw = linalg.inv(bw)
                det_inv_bw = linalg.det(inv_bw)
            else:
                raise ValueError("Error, specified bandiwdth has more than 2 dimension")
            self._bandwidth = bw
            self._inv_bw = inv_bw
            self._det_inv_bw = det_inv_bw
        else:
            self._bandwidth = bw

    @property
    def inv_bandwidth(self):
        """
        Inverse of the selected bandwidth
        """
        return self._inv_bw

    @property
    def det_inv_bandwidth(self):
        """
        Inverse of the selected bandwidth
        """
        return self._det_inv_bw

    def update_inputs(self, exog, weights=1., adjust=1.):
        exog = atleast_2df(exog)
        if exog.ndim != 2:
            raise ValueError("Error, exog must be at most a 2D array")
        weights = np.asarray(weights)
        adjust = np.asarray(adjust)
        if weights.ndim != 0 and weights.shape != (exog.shape[0],):
            raise ValueError("Error, weights must be either a single number, "
                             "or a 1D array with the same length as exog")
        if adjust.ndim != 0 and adjust.shape != (exog.shape[0],):
            raise ValueError("Error, adjust must be either a single number, "
                             "or a 1D array with the same length as exog")
        self._exog = exog
        self._weights = weights
        self._adjust = adjust
        if weights.ndim > 0:
            self._total_weights = weights.sum()
        else:
            self._total_weights = self.npts

    def closed(self, dim=None):
        """
        Returns true if the density domain is closed (i.e. lower and upper
        are both finite)

        Parameters
        ----------
        dim: int
            Dimension to test. If None, test of all dimensions are closed
        """
        if dim is None:
            return all(self.closed(i) for i in range(self.ndim))
        return self.lower[dim] > -np.inf and self.upper[dim] < np.inf

    def bounded(self, dim=None):
        """
        Returns true if the density domain is actually bounded

        Parameters
        ----------
        dim: int
            Dimension to test. If None, test of all dimensions are bounded
        """
        if dim is None:
            return all(self.bounded(i) for i in range(self.ndim))
        return self.lower[dim] > -np.inf or self.upper[dim] < np.inf

    @numpy_trans_method('ndim', 1)
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
        Returns the ``out`` variable, updated with the PDF.

        :Default: Direct implementation of the formula for unbounded pdf
            computation.
        """
        exog = self.exog

        m, d = points.shape
        assert d == self.ndim

        kernel = self.kernel
        inv_bw = self.inv_bandwidth

        def scalar_inv_bw(pts):
            return (pts * inv_bw)

        def matrix_inv_bw(pts):
            return np.dot(pts, inv_bw)
        if inv_bw.ndim == 2:
            inv_bw_fct = matrix_inv_bw
        else:
            inv_bw_fct = scalar_inv_bw

        det_inv_bw = self.det_inv_bandwidth
        weights = self.weights
        adjust = self.adjust

        if self.npts > m:
            factor = weights * det_inv_bw / adjust
            # There are fewer points that data: loop over points
            energy = np.empty((exog.shape[0],), dtype=out.dtype)
            for idx in range(m):
                diff = inv_bw_fct(points[idx] - exog)
                kernel.pdf(diff, out=energy)
                energy *= factor
                out[idx] = np.sum(energy)
        else:
            weights = np.atleast_1d(weights)
            adjust = np.atleast_1d(adjust)
            out[...] = 0

            # There are fewer data that points: loop over data
            dw = 1 if weights.shape[0] > 1 else 0
            da = 1 if adjust.shape[0] > 1 else 0
            na = 0
            nw = 0
            n = self.npts
            energy = np.empty((points.shape[0],), dtype=out.dtype)
            for idx in range(n):
                diff = inv_bw_fct(points - exog[idx])
                kernel.pdf(diff, out=energy)
                energy *= weights[nw] / adjust[na]
                out += energy
                # Iteration for weights and adjust
                na += da
                nw += dw
            out *= det_inv_bw

        out /= self.total_weights
        return out

    def __call__(self, points, out=None):
        """
        Call the :py:meth:`pdf` method.
        """
        return self.pdf(points, out)

    def grid(self, N=None, cut=None):
        """
        Compute the PDF on a grid.

        Parameters
        ----------
        N: int or list of int
            Number of elements on the grid, per dimension
        """
        gr = generate_grid(self, N, cut)
        data = self.pdf(gr.linear()).reshape(gr.shape)
        return gr, data

    @numpy_trans_method('ndim', 1)
    def cdf(self, points, out):
        # bw = self.bandwidth
        # if bw.ndim < 2:  # We have a diagonal matrix
        #     exog = self.exog
        # else:
        #     pass
        raise NotImplementedError()

    def grid_size(self, N=None):
        if N is None:
            p2 = self.base_p2 // self.ndim
            if self.base_p2 % self.ndim > 0:
                p2 += 1
            return 2 ** p2
        return N


class Cyclic(KDEnDMethod):

    name = "cyclic"

    def fit(self, kde, compute_bandwidth=True):
        if kde.ndim == 1:
            cyc = Cyclic1D()
            return cyc.fit(kde, compute_bandwidth)
        return super(Cyclic, self).fit(kde, compute_bandwidth)

    @property
    def bin_type(self):
        return 'c'*self.ndim

    @numpy_trans_method('ndim', 1)
    def pdf(self, points, out):
        for i in range(self.ndim):
            if self.bounded(i) and not self.closed(i):
                raise ValueError("Error, cyclic method requires all dimensions to be closed or not bounded")
        if not self.bounded():
            return super(Cyclic, self).pdf(points, out)
        exog = self.exog

        m, d = points.shape
        assert d == self.ndim

        kernel = self.kernel
        inv_bw = self.inv_bandwidth

        def scalar_inv_bw(pts):
            return (pts * inv_bw)

        def matrix_inv_bw(pts):
            return np.dot(pts, inv_bw)

        if inv_bw.ndim == 2:
            inv_bw_fct = matrix_inv_bw
        else:
            inv_bw_fct = scalar_inv_bw

        # if inv_bw.ndim == 2:
            # raise ValueError("Error, this method cannot handle non-diagonal bandwidth matrix.")
        det_inv_bw = self.det_inv_bandwidth
        weights = self.weights
        adjust = self.adjust

        span = inv_bw_fct(self.upper - self.lower)

        if self.npts > m:
            factor = weights * det_inv_bw / adjust
            # There are fewer points that data: loop over points
            energy = np.empty((exog.shape[0],), dtype=out.dtype)
            # print("iterate on points")
            for idx in range(m):
                diff = inv_bw_fct(points[idx] - exog)
                kernel.pdf(diff, out=energy)
                for d in range(self.ndim):
                    if np.isfinite(span[d]):
                        energy += kernel.pdf(diff - span)
                        energy += kernel.pdf(diff + span)
                energy *= factor
                out[idx] = np.sum(energy)
        else:
            weights = np.atleast_1d(weights)
            adjust = np.atleast_1d(adjust)
            out[...] = 0

            # There are fewer data that points: loop over data
            dw = 1 if weights.shape[0] > 1 else 0
            da = 1 if adjust.shape[0] > 1 else 0
            na = 0
            nw = 0
            n = self.npts
            energy = np.empty((points.shape[0],), dtype=out.dtype)
            # print("iterate on exog")
            for idx in range(n):
                diff = inv_bw_fct(points - exog[idx]) / adjust[na]
                kernel.pdf(diff, out=energy)
                for d in range(self.ndim):
                    if np.isfinite(span[d]):
                        energy += kernel.pdf(diff - span)
                        energy += kernel.pdf(diff + span)
                energy *= weights[nw] / adjust[na]
                out += energy
                # Iteration for weights and adjust
                na += da
                nw += dw
            out *= det_inv_bw

        out /= self.total_weights
        return out

    def grid(self, N=None, cut=None):
        if self.adjust.shape:
            return KDEnDMethod.grid(self, N, cut)
        for i in range(self.ndim):
            if self.bounded(i) and not self.closed(i):
                raise ValueError("Error, cyclic method requires all dimensions to be closed or not bounded")
        bw_inv = self.inv_bandwidth / self.adjust
        exog = self.exog
        N = self.grid_size(N)

        lower = self.lower.copy()
        upper = self.upper.copy()

        if cut is None:
            cut = self.kernel.cut
        if self.bandwidth.ndim == 0:
            cut = self.bandwidth * cut * np.ones(self.ndim, dtype=float)
        elif self.bandwidth.ndim == 1:
            cut = self.bandwidth * cut
        else:
            cut = np.dot(self.bandwidth, cut * np.ones(self.ndim, dtype=float))

        for d in range(self.ndim):
            if upper[d] == np.inf:
                lower[d] = np.min(exog[:, d]) - cut[d]
                upper[d] = np.max(exog[:, d]) + cut[d]

        weights = self.weights

        return fftdensity(exog, self.kernel.rfft, bw_inv, lower, upper, N, weights, self.total_weights)


class Reflection(KDEnDMethod):

    name = "reflection"

    def fit(self, kde, compute_bandwidth=True):
        if kde.ndim == 1:
            cyc = Reflection1D()
            return cyc.fit(kde, compute_bandwidth)
        return super(Reflection, self).fit(kde, compute_bandwidth)

    @property
    def bin_type(self):
        return 'c'*self.ndim

    @numpy_trans_method('ndim', 1)
    def pdf(self, points, out):
        if not self.bounded():
            return super(Reflection, self).pdf(points, out)
        exog = self.exog

        m, d = points.shape
        assert d == self.ndim

        kernel = self.kernel
        inv_bw = self.inv_bandwidth

        def scalar_inv_bw(pts):
            return (pts * inv_bw)

        def matrix_inv_bw(pts):
            return np.dot(pts, inv_bw)

        if inv_bw.ndim == 2:
            inv_bw_fct = matrix_inv_bw
        else:
            inv_bw_fct = scalar_inv_bw

        # if inv_bw.ndim == 2:
            # raise ValueError("Error, this method cannot handle non-diagonal bandwidth matrix.")
        det_inv_bw = self.det_inv_bandwidth
        weights = self.weights
        adjust = self.adjust

        span = inv_bw_fct(self.upper - self.lower)

        if self.npts > m:
            factor = weights * det_inv_bw / adjust
            # There are fewer points that data: loop over points
            energy = np.empty((exog.shape[0],), dtype=out.dtype)
            # print("iterate on points")
            for idx in range(m):
                diff = inv_bw_fct(points[idx] - exog)
                kernel.pdf(diff, out=energy)
                for d in range(self.ndim):
                    if np.isfinite(span[d]):
                        energy += kernel.pdf(diff - span)
                        energy += kernel.pdf(diff + span)
                energy *= factor
                out[idx] = np.sum(energy)
        else:
            weights = np.atleast_1d(weights)
            adjust = np.atleast_1d(adjust)
            out[...] = 0

            # There are fewer data that points: loop over data
            dw = 1 if weights.shape[0] > 1 else 0
            da = 1 if adjust.shape[0] > 1 else 0
            na = 0
            nw = 0
            n = self.npts
            energy = np.empty((points.shape[0],), dtype=out.dtype)
            # print("iterate on exog")
            for idx in range(n):
                diff = inv_bw_fct(points - exog[idx]) / adjust[na]
                kernel.pdf(diff, out=energy)
                for d in range(self.ndim):
                    if np.isfinite(span[d]):
                        energy += kernel.pdf(diff - span)
                        energy += kernel.pdf(diff + span)
                energy *= weights[nw] / adjust[na]
                out += energy
                # Iteration for weights and adjust
                na += da
                nw += dw
            out *= det_inv_bw

        out /= self.total_weights
        return out

    def grid(self, N=None, cut=None):
        if self.adjust.shape:
            return KDEnDMethod.grid(self, N, cut)
        if self.det_inv_bandwidth.ndim == 2:
            return super(Reflection, self).grid(N, cut)
        bw_inv = self.inv_bandwidth / self.adjust
        exog = self.exog
        N = self.grid_size(N)

        lower = self.lower.copy()
        upper = self.upper.copy()

        if cut is None:
            cut = self.kernel.cut
        if self.bandwidth.ndim == 0:
            cut = self.bandwidth * cut * np.ones(self.ndim, dtype=float)
        elif self.bandwidth.ndim == 1:
            cut = self.bandwidth * cut

        for d in range(self.ndim):
            if upper[d] == np.inf:
                lower[d] = np.min(exog[:, d]) - cut[d]
                upper[d] = np.max(exog[:, d]) + cut[d]

        weights = self.weights

        return dctdensity(exog, self.kernel.rfft, bw_inv, lower, upper, N, weights, self.total_weights)
