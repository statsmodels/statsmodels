r"""
This module contains a set of methods to compute univariate KDEs.

:Author: Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>

These methods provide various variations on :math:`\hat{K}(x;X,h,L,U)`, the
modified kernel evaluated on the point :math:`x` based on the estimation points
:math:`X`, a bandwidth :math:`h` and on the domain :math:`[L,U]`.

The definitions of the methods rely on the following definitions:

.. math::

   \begin{array}{rcl}
     a_0(l,u) &=& \int_l^u K(z) dz\\
     a_1(l,u) &=& \int_l^u zK(z) dz\\
     a_2(l,u) &=& \int_l^u z^2K(z) dz
   \end{array}

These definitions correspond to:

- :math:`a_0(l,u)` -- The partial cumulative distribution function
- :math:`a_1(l,u)` -- The partial first moment of the distribution. In
  particular, :math:`a_1(-\infty, \infty)` is the mean of the kernel (i.e. and
  should be 0).
- :math:`a_2(l,u)` -- The partial second moment of the distribution. In
  particular, :math:`a_2(-\infty, \infty)` is the variance of the kernel (i.e.
  which should be close to 1, unless using higher order kernel).

References:
```````````
.. [1] Jones, M. C. 1993. Simple boundary correction for kernel density
    estimation. Statistics and Computing 3: 135--146.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import copy as shallow_copy

import numpy as np
from scipy import fftpack, integrate

from .kde_utils import make_ufunc, numpy_trans1d_method, finite, AxesType, Grid
from .fast_linbin import fast_linbin as fast_bin
from ._kde_methods import KDEMethod, filter_exog, invert_cdf
from . import kernels


def generate_grid1d(kde, N=None, cut=None, span=None):
    r"""
    Helper method returning a regular grid on the domain of the KDE.

    Parameters
    ----------
    kde: KDE1DMethod
        Fitted KDE object
    N: int
        Number of points in the grid
    cut: float
        For unbounded domains, how far past the maximum should
        the grid extend to, in term of KDE bandwidth

    Returns
    -------
    A vector of N regularly spaced points
    """
    N = kde.grid_size(N)
    if span is None:
        lower = kde.lower
        upper = kde.upper
    else:
        lower, upper = span
    if cut is None:
        cut = kde.kernel.cut
    if not finite(lower):
        lower = np.min(kde.exog) - cut * kde.bandwidth
    if not finite(upper):
        upper = np.max(kde.exog) + cut * kde.bandwidth
    mesh, step = np.linspace(lower, upper, N, endpoint=False, retstep=True)
    mesh += step / 2
    return Grid(mesh, bounds=[lower, upper])


def _compute_bandwidth(kde, default):
    """
    Compute the bandwidth for the estimated model, based of its exog attribute
    """
    if kde.bandwidth is None:
        bw = default
    else:
        bw = kde.bandwidth
    if callable(bw):
        bw = float(bw(kde))
    else:
        bw = float(bw)
    return bw


def convolve(exog, point, fct, out=None, scaling=1., weights=1., factor=1.,
             dim=-1):
    """
    Convolve a set of weighted point with a function

    Parameters
    ----------
    exog: ndarray
        Points to convolve. Must be a 1D array.
    point: float or ndarray
        Points where the convolution is evaluated.
    fct: fun
        Function used for the convolution
    out: ndarray
        Array of same size as `point`, in which the result will be stored.
    scaling: float or ndarray
        Scaling of the convolution function. It may be an array the same size
        as exog.
    weights: float or ndarray
        Weights for the exog points.
    factor: float
        Normalization factor. The final result will be divided by that value.

    Returns
    -------
    ndarray
        Convolution of the exog points by the scaled function evaluation on the
        point

    Notes
    -----

    The basic idea is to evaluate the convolution of of a function on the exog
    on a point. Anything can be an array if you are careful to choose your
    dimensions. Just remember than the list of exog values will be the last
    dimension.
    """
    z = (point - exog) / scaling

    terms = fct(z)

    terms = (terms * weights) / scaling

    if out is None or np.isscalar(out):
        out = terms.sum(axis=dim)
    else:
        terms.sum(axis=dim, out=out)
    out /= factor

    return out


class KDE1DMethod(KDEMethod):
    """
    Base class providing a default grid method and a default method for
    unbounded evaluation of the PDF and CDF. It also provides default methods
    for the other metrics, based on PDF and CDF calculations.

    :Note:
        - It is expected that all grid methods will return the same grid if
          used with the same arguments.
        - It is fair to assume all array-like arguments will be at least 1D
          arrays.
    """

    #: Name of the method, for presentation purposes
    name = 'unbounded1d'

    def __init__(self):
        KDEMethod.__init__(self)
        self._kernel = kernels.normal1d()

    @property
    def axis_type(self):
        """
        Instance of AxesType describing the axis (e.g. always 'c')
        """
        return AxesType('c')

    @axis_type.setter
    def axis_type(self, value):
        if value != 'c':
            raise ValueError("Error, this method can only be used for 1D "
                             "continuous axis")

    @property
    def bin_type(self):
        """
        Type of the bins adapted for the method (default: 'b')
        """
        return 'b'

    def fit(self, kde, compute_bandwidth=True):
        """
        Extract the parameters required for the computation and returns
        a stand-alone estimator capable of performing most computations.

        Parameters
        ----------
        compute_bandwidth: bool
            If true (default), the bandwidth is computed

        Returns
        -------
        An estimator object that doesn't depend on the KDE object.


        Notes
        -----
        By default, most values can be adjusted after estimation. However, it
        is not allowed to change the number of exogenous variables or the
        dimension of the problem.
        """
        if kde.ndim != 1:
            raise ValueError("Error, this is a 1D method, expecting a 1D "
                             "problem")
        if np.any(kde.axis_type != self.axis_type):
            raise ValueError("Error, incompatible method for the type of axis")

        kde = filter_exog(kde, self.bin_type)

        if compute_bandwidth:
            k = kde.copy()
            bw = _compute_bandwidth(k, self._bandwidth)
        else:
            bw = kde.bandwidth

        fitted = self.copy()
        fitted._bandwidth = bw
        fitted._fitted = True
        fitted._upper = float(kde.upper)
        fitted._lower = float(kde.lower)
        fitted._exog = kde.exog.reshape((kde.npts,))
        if kde.kernel is not None:
            fitted._kernel = kde.kernel.for_ndim(1)
        elif hasattr(self, '_kernel') and self._kernel is not None:
            fitted._kernel = self._kernel.for_ndim(1)
        else:
            raise ValueError("No kernel specified and this method doesn't "
                             "have a default kernel.")
        fitted._weights = kde.weights
        fitted._adjust = kde.adjust
        fitted._total_weights = kde.total_weights
        return fitted

    @KDEMethod.exog.setter
    def exog(self, value):
        value = np.atleast_1d(value).astype(float)
        if value.shape != (self.npts,):
            raise ValueError("Bad shape: to change the number of points use "
                             "update_inputs")
        self._exog = value

    def copy(self):
        """
        Create a shallow copy of the estimated object
        """
        return shallow_copy(self)

    @property
    def ndim(self):
        """
        Dimension of the problem
        """
        return 1

    @KDEMethod.bandwidth.setter
    def bandwidth(self, val):
        val = float(val)
        assert val > 0, "The bandwidth must be strictly positive"
        self._bandwidth = val

    def update_inputs(self, exog, weights=1., adjust=1.):
        """
        Update all the variable lengths inputs at once to ensure consistency
        """
        exog = np.atleast_1d(exog)
        if exog.ndim != 1:
            raise ValueError(("Error, exog must be a 1D array (nb dimensions: "
                              "{})").format(exog.ndim))
        weights = np.asarray(weights)
        adjust = np.asarray(adjust)
        if weights.ndim != 0 and weights.shape != exog.shape:
            raise ValueError("Error, weights must be either a single number, "
                             "or an array the same shape as exog")
        if adjust.ndim != 0 and adjust.shape != exog.shape:
            raise ValueError("Error, adjust must be either a single number, "
                             "or an array the same shape as exog")
        self._exog = exog
        self._weights = weights
        self._adjust = adjust
        if weights.ndim > 0:
            self._total_weights = weights.sum()
        else:
            self._total_weights = self.npts

    @property
    def to_bin(self):
        """
        Property holding the data to be binned. This is useful when the PDF is
        not evaluated on the real dataset, but on a transformed one.
        """
        return None

    #: Function used to transform an axis, or None for no transformation
    transform_axis = None
    #: Inverse function of transform_axis
    restore_axis = None
    #: Function used to adapt the bin values when restoring an axis
    transform_bins = None

    @KDEMethod.lower.setter
    def lower(self, val):
        val = float(val)
        self._lower = val

    @KDEMethod.upper.setter
    def upper(self, val):
        val = float(val)
        self._upper = val

    @property
    def closed(self):
        """
        Returns true if the density domain is closed (lower and upper
        are both finite).
        """
        return self.lower > -np.inf and self.upper < np.inf

    @property
    def bounded(self):
        """
        Returns true if the density domain is actually bounded.
        """
        return self.lower > -np.inf or self.upper < np.inf

    @numpy_trans1d_method(in_dtype=float)
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
        return convolve(self.exog, points[..., None], self.kernel.pdf, out,
                        self.bandwidth * self.adjust, self.weights,
                        self.total_weights)

    def __call__(self, points, out=None):
        """
        Call the :py:meth:`pdf` method.
        """
        return self.pdf(points, out)

    @numpy_trans1d_method(in_dtype=float)
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
            The default implementation uses the formula for unbounded CDF
            computation.
        """
        exog = self.exog
        points = points[..., np.newaxis]
        bw = self.bandwidth * self.adjust

        z = (points - exog) / bw

        kernel = self.kernel

        terms = kernel.cdf(z)
        terms *= self.weights

        terms.sum(axis=-1, out=out)
        out /= self.total_weights

        out[points[..., 0] >= self.upper] = 1
        out[points[..., 0] <= self.lower] = 0

        return out

    @numpy_trans1d_method(in_dtype=float)
    def icdf(self, points, out):
        r"""
        Compute the inverse cumulative distribution (quantile) function,
        defined as:

        .. math::

            icdf(p) = \inf\left\{x\in\mathbb{R} : cdf(x) \geq p\right\}

        Parameters
        ----------
        points: ndarray
            Points to evaluate the iCDF on
        out: ndarray
            Result object. If must have the same shapes as ``points``

        Returns
        -------
        ndarray
            Returns the ``out`` variable, updated with the iCDF.

        Notes
        -----
        This method first approximates the result using linear interpolation
        on the CDF and refine the result numerically using the Newton method.
        """
        return invert_cdf(points, out, self.pdf, self.cdf, self.cdf_grid(),
                          self.lower, self.upper)

    @numpy_trans1d_method(in_dtype=float)
    def sf(self, points, out):
        r"""
        Compute the survival function, defined as:

        .. math::

            sf(x) = P(X \geq x) = \int_x^u p(t) dt = 1 - cdf(x)

        Parameters
        ----------
        points: ndarray
            Points to evaluate the survival function on
        out: ndarray
            Result object. If must have the same shapes as ``points``

        Returns
        -------
        ndarray
            Returns the ``out`` variable, updated with the survival function.

        Notes
        -----
        Compute explicitly :math:`1 - cdf(x)`
        """
        self.cdf(points, out)
        out -= 1
        out *= -1
        return out

    @numpy_trans1d_method(in_dtype=float)
    def isf(self, points, out):
        r"""
        Compute the inverse survival function, defined as:

        .. math::

            isf(p) = \sup\left\{x\in\mathbb{R} : sf(x) \leq p\right\}

        Parameters
        ----------
        points: ndarray
            Points to evaluate the iSF on
        out: ndarray
            Result object. If must have the same shapes as ``points``

        Returns
        -------
        ndarray
            Returns the ``out`` variable, updated with the inverse survival
            function.

        Notes
        -----
        Compute :math:`icdf(1-p)`
        """
        return self.icdf(1 - points, out)

    @numpy_trans1d_method(in_dtype=float)
    def hazard(self, points, out):
        r"""
        Compute the hazard function evaluated on the points.

        Parameters
        ----------
        points: ndarray
            Points to evaluate the hazard function on
        out: ndarray
            Result object. If must have the same shapes as ``points``

        Returns
        -------
        ndarray
            Returns the ``out`` variable, updated with the hazard function

        Notes
        -----
        The hazard function is defined as:

        .. math::

            h(x) = \frac{p(x)}{sf(x)}

        where :math:`p(x)` is the probability density function and
        :math:`sf(x)` is the survival function.
        """
        self.pdf(points, out=out)
        sf = np.empty(out.shape, dtype=out.dtype)
        self.sf(points, sf)
        sf[sf < 0] = 0  # Some methods can produce negative sf
        out /= sf
        return out

    @numpy_trans1d_method(in_dtype=float)
    def cumhazard(self, points, out):
        r"""
        Compute the cumulative hazard function evaluated on the points.

        Parameters
        ----------
        points: ndarray
            Points to evaluate the cumuladavid gutive hazard function on
        out: ndarray
            Result object. If must have the same shapes as ``points``

        Returns
        -------
        ndarray
            Returns the ``out`` variable, updated with the cumulative hazard
            function

        Notes
        -----
        The hazard function is defined as:

        .. math::

            ch(x) = \int_l^x h(t) dt = -\ln sf(x)

        where :math:`l` is the lower bound of the domain, :math:`h` the hazard
        function and :math:`sf` the survival function.
        """
        self.sf(points, out)
        out[out < 0] = 0  # Some methods can produce negative sf
        np.log(out, out=out)
        out *= -1
        return out

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

        Notes
        -----
        By default, this method evaluates :math:`pdf(x)` on a grid generated
        using :py:func:`generate_grid1d`
        """
        N = self.grid_size(N)
        g = generate_grid1d(self, N, cut, span)
        out = np.empty(g.shape, dtype=float)
        return g, self.pdf(g.full(), out)

    def from_binned(self, mesh, bins, normed=False, dim=-1):
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
        result = np.empty_like(bins)
        if dim < 0:
            dim = mesh.ndim + dim
        pdf = self.kernel.pdf
        if mesh.ndim == 1:
            pts = mesh.grid[dim]
            convolve(pts, pts[..., None], pdf, result,
                     scaling=self.bandwidth * self.adjust,
                     weights=bins)
        else:
            left = np.index_exp[:] * dim
            right = np.index_exp[:] * (mesh.ndim - dim - 1)
            eval_pts = mesh.grid[dim]
            pts = eval_pts.view()
            pts.shape = (1,) * dim + (len(pts),) + (1,) * (mesh.ndim - dim - 1)
            for i, p in enumerate(mesh.grid[dim]):
                access = left + (i,) + right
                convolve(pts, p, pdf, result[access],
                         scaling=self.bandwidth * self.adjust,
                         weights=bins,
                         dim=dim)
        if normed:
            result /= self.total_weights
        return result

    def cdf_grid(self, N=None, cut=None, span=None):
        """
        Evaluate the CDF of the distribution on a regular grid with at least
        ``N`` elements.

        Parameters
        ----------
        N: int
            minimum number of element in the returned grid. Most
            methods will want to round it to the next power of 2.
        cut: float
            for unbounded domains, how far from the last data
            point should the grid go, as a fraction of the bandwidth.

        Returns
        -------
        mesh : :py:class:`Grid`
            Grid on which the CDF has bin evaluated
        values : ndarray
            Values of the CDF for each position of the grid.

        Notes
        -----
        By defaults, thie method evaluate :math:`cdf(x)` on a grid generated
        using :py:func:`generate_grid1d`
        """
        N = self.grid_size(N)
        if N <= 2 ** 11:
            g = generate_grid1d(self, N, cut)
            out = np.empty(g.shape, dtype=float)
            return g, self.cdf(g.full(), out)
        return self.numeric_cdf_grid(N, cut)

    def icdf_grid(self, N=None, cut=None):
        """
        Compute the inverse cumulative distribution (quantile) function on
        a grid.

        Parameters
        ----------
        N: int
            minimum number of element in the returned grid. Most
            methods will want to round it to the next power of 2.
        cut: float
            for unbounded domains, how far from the last data
            point should the grid go, as a fraction of the bandwidth.

        Returns
        -------
        mesh : :py:class:`Grid`
            Grid on which the inverse CDF has bin evaluated
        values : ndarray
            Values of the inverse CDF for each position of the grid.

        Notes
        -----
        The default implementation is not as good an approximation as the
        plain icdf default method: it performs a linear interpolation of
        the inverse CDF on a grid
        """
        xs, ys = self.cdf_grid(N, cut)
        xs = xs.linear()
        N = len(xs)
        points = np.linspace(0, 1, N)
        icdf = np.interp(points, ys, xs, self.lower, self.upper)
        return Grid(points, bounds=[0, 1]), icdf

    def sf_grid(self, N=None, cut=None):
        r"""
        Compute the survival function on a grid.

        Parameters
        ----------
        N: int
            minimum number of element in the returned grid. Most
            methods will want to round it to the next power of 2.
        cut: float
            for unbounded domains, how far from the last data
            point should the grid go, as a fraction of the bandwidth.

        Returns
        -------
        mesh : :py:class:`Grid`
            Grid on which the survival function has bin evaluated
        values : ndarray
            Values of the inverse survival function for each position of the
            grid.

        Notes
        -----
        Compute explicitly :math:`1 - cdf(x)`
        """
        points, out = self.cdf_grid(N, cut)
        out -= 1
        out *= -1
        return points, out

    def isf_grid(self, N=None, cut=None):
        """
        Compute the inverse survival function on a grid.

        Parameters
        ----------
        N: int
            minimum number of element in the returned grid. Most
            methods will want to round it to the next power of 2.
        cut: float
            for unbounded domains, how far from the last data
            point should the grid go, as a fraction of the bandwidth.

        Returns
        -------
        (ndarray, ndarray)
            The array of positions the CDF has been estimated on, and the
            estimations.

        Notes
        -----
        The default implementation is not as good an approximation as the
        plain isf default method: it performs a linear interpolation of the
        inverse survival function on a grid.
        """
        xs, ys = self.sf_grid(N, cut)
        xs = xs.full()
        N = len(xs)
        points = np.linspace(0, 1, N)
        isf = np.interp(points, ys[::-1], xs[::-1], self.upper, self.lower)
        return Grid(points, bounds=[0, 1]), isf

    def hazard_grid(self, N=None, cut=None):
        r"""
        Compute the hazard function on a grid.

        Parameters
        ----------
        N: int
            minimum number of element in the returned grid. Most
            methods will want to round it to the next power of 2.
        cut: float
            for unbounded domains, how far from the last data
            point should the grid go, as a fraction of the bandwidth.

        Returns
        -------
        (ndarray, ndarray)
            The array of positions the hazard function has been
            estimated on, and the estimations.

        Notes
        -----
        Compute explicitly :math:`pdf(x) / sf(x)`
        """
        points, out = self.grid(N, cut)
        _, sf = self.sf_grid(N, cut)
        sf[sf < 0] = 0  # Some methods can produce negative sf
        out /= sf
        return points, out

    def cumhazard_grid(self, N=None, cut=None):
        r"""
        Compute the hazard function on a grid.

        Parameters
        ----------
        N: int
            minimum number of element in the returned grid. Most
            methods will want to round it to the next power of 2.
        cut: float
            for unbounded domains, how far from the last data
            point should the grid go, as a fraction of the bandwidth.

        Returns
        -------
        (ndarray, ndarray)
            The array of positions the hazard function has been
            estimated on, and the estimations.

        Notes
        -----
        Compute explicitly :math:`-\ln sf(x)`
        """
        points, out = self.sf_grid(N, cut)
        out[out < 0] = 0  # Some methods can produce negative sf
        np.log(out, out=out)
        out *= -1
        return points, out

    def __str__(self):
        """
        Return the name of the method
        """
        return self.name

    @numpy_trans1d_method(in_dtype=float)
    def numeric_cdf(self, points, out):
        """
        Provide a numeric approximation of the CDF based on integrating the pdf
        using :py:func:`scipy.integrate.quad`.
        """
        pts = points.ravel()

        pts[pts < self.lower] = self.lower
        pts[pts > self.upper] = self.upper

        ix = pts.argsort()

        sp = pts[ix]

        pdf_out = np.empty((1,), dtype=float)

        def pdf(x):
            return self.pdf(np.array([x]), pdf_out)

        @make_ufunc()
        def comp_cdf(i):
            low = self.lower if i == 0 else sp[i - 1]
            if sp[i] == -np.inf:
                return 0
            elif sp[i] == np.inf:
                return 1
            return integrate.quad(pdf, low, sp[i])[0]

        parts = np.empty(sp.shape, dtype=float)
        comp_cdf(np.arange(len(sp)), out=parts)

        ints = parts.cumsum()
        ints[ints > 1] = 1

        out.put(ix, ints)
        return out

    def numeric_cdf_grid(self, N=None, cut=None):
        """
        Compute the CDF on a grid using a trivial, but fast, numeric
        integration of the pdf.
        """
        pts, pdf = self.grid(N, cut)
        return pts, pts.cum_integrate(pdf)

    def grid_size(self, N=None):
        """
        Returns a valid grid size.
        """
        if N is None:
            return 2 ** 10
        return N


def fftdensity_from_binned(mesh, bins, kernel_rfft, bw, normed=False,
                           total_weights=None, dim=-1):
    """
    Parameters
    ----------
    mesh: Grid
        Grid object representing the position of the bins
    bins: ndarray
        2D array of same shape as the grid.
    kernel_rfft: function
        Function computing the rFFT for the kernel
    bw: float
        Bandwidth of the kernel
    normed: bool
        If true, the bins will be normalized at the end.
    dim: int
        Axis on which to perform the FFT

    Returns
    -------
    ndarray
        An array of same size as the bins, convoluted by the kernel
    """
    if dim < 0:
        dim += mesh.ndim
    FFTData = np.fft.rfft(bins, axis=dim)

    smth = kernel_rfft(bins.shape[dim], mesh.start_interval[dim] / bw)
    if mesh.ndim > 1:
        smth.shape = (1,) * dim + (len(smth),) + (1,) * (mesh.ndim - dim - 1)

    SmoothFFTData = FFTData * smth
    density = np.fft.irfft(SmoothFFTData, bins.shape[dim], axis=dim)
    density /= mesh.start_interval[dim]
    if normed:
        if total_weights is None:
            total_weights = bins.sum()
        density /= total_weights * mesh.start_interval[dim]
    return density


def fftdensity(exog, kernel_rfft, bw, lower, upper, N, weights, total_weights):
    """
    Compute the density estimate using a FFT approximation.

    Parameters
    ----------
    exog: ndarray
        1D array with the data to fit
    kernel_rfft: function
        Function computing the rFFT for the kernel
    bw: float
        Bandwidth of the kernel
    lower: float
        Lower bound on which to compute the density
    upper: float
        Upper bound on which to compute the density
    N: int
        Number of buckets to compute
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
    mesh, DataHist = fast_bin(exog, [lower, upper], N, weights=weights,
                              bin_type='c')
    DataHist /= total_weights
    return mesh, fftdensity_from_binned(mesh, DataHist, kernel_rfft, bw)


def dctdensity_from_binned(mesh, bins, kernel_dct, bw, normed=False,
                           total_weights=None, dim=-1):
    """
    Parameters
    ----------
    mesh: Grid
        Grid object representing the position of the bins
    bins: ndarray
        2D array of same shape as the grid.
    kernel_rfft: function
        Function computing the rFFT for the kernel
    bw: float
        Bandwidth of the kernel
    normed: bool
        If true, the bins will be normalized at the end.
    dim: int
        Axis on which to perform the FFT

    Returns
    -------
    ndarray
        An array of same size as the bins, convoluted by the kernel
    """
    if dim < 0:
        dim += mesh.ndim
    DCTData = fftpack.dct(bins, axis=dim)

    smth = kernel_dct(bins.shape[dim], mesh.start_interval[dim] / bw)
    if mesh.ndim > 1:
        smth.shape = (1,) * dim + (len(smth),) + (1,) * (mesh.ndim - dim - 1)

    # Smooth the DCTransformed data using t_star
    SmDCTData = DCTData * smth
    # Inverse DCT to get density
    R = mesh.bounds[dim][1] - mesh.bounds[dim][0]
    density = fftpack.idct(SmDCTData, axis=dim) / (2 * R)
    if normed:
        if total_weights is None:
            total_weights = bins.sum()
        density /= total_weights
    return density


def dctdensity(exog, kernel_dct, bw, lower, upper, N, weights, total_weights):
    """
    Compute the density estimate using a DCT approximation.

    Parameters
    ----------
    exog: ndarray
        1D array with the data to fit
    kernel_dct: function
        Function computing the DCT of the kernel
    lower: float
        Lower bound on which to compute the density
    upper: float
        Upper bound on which to compute the density
    N: int
        Number of buckets to compute
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
    # Histogram the data to get a crude first approximation of the density
    mesh, DataHist = fast_bin(exog, [lower, upper], N, weights=weights,
                              bin_type='r')

    DataHist /= total_weights
    return mesh, dctdensity_from_binned(mesh, DataHist, kernel_dct, bw)
