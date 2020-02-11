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

import numpy as np
from scipy import fftpack, integrate, optimize
from .kde_utils import (make_ufunc, numpy_trans1d_method, numpy_trans1d,
                        finite, AxesType, Grid)
from .fast_linbin import fast_linbin as fast_bin
from copy import copy as shallow_copy
from .kernels import Kernel1D
from ._kde_methods import KDEMethod, filter_exog
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


def convolve(exog,
             point,
             fct,
             out=None,
             scaling=1.,
             weights=1.,
             factor=1.,
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
        self._kernel = kernels.Gaussian1D()

    @property
    def axis_type(self):
        """
        Instance of AxesType describing the axis (e.g. always 'C')
        """
        return AxesType('C')

    @axis_type.setter
    def axis_type(self, value):
        if value != 'C':
            raise ValueError('Error, this method can only be used for 1D '
                             'continuous axis')

    @property
    def bin_type(self):
        """
        Type of the bins adapted for the method (default: 'B')
        """
        return 'B'

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
            raise ValueError(
                "Error, this is a 1D method, expecting a 1D problem")
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
        fitted._exog = kde.exog.reshape((kde.npts, ))
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
        if value.shape != (self.npts, ):
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
            raise ValueError(
                "Error, exog must be a 1D array (nb dimensions: {})".format(
                    exog.ndim))
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
        return convolve(self.exog, points[..., None], self.kernel.pdf, out,
                        self.bandwidth * self.adjust, self.weights,
                        self.total_weights)

    def __call__(self, points, out=None):
        """
        Call the :py:meth:`pdf` method.
        """
        return self.pdf(points, out)

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

    @numpy_trans1d_method()
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
        This method first approximates the result using linear interpolation on
        the CDF and refine the result numerically using the Newton method.
        """
        xs, ys = self.cdf_grid()
        xs = xs.linear()
        coarse_result = np.interp(points, ys, xs, self.lower, self.upper)
        lower = self.lower
        upper = self.upper
        cdf = self.cdf
        pdf_out = np.empty(1, dtype=float)

        def pdf(x):
            if x <= lower:
                return 0
            if x >= upper:
                return 0
            return self.pdf(np.atleast_1d(x), pdf_out)

        @make_ufunc()
        def find_inverse(p, approx):
            if p > 1 - 1e-10:
                return upper
            if p < 1e-10:
                return lower
            if approx >= xs[-1] or approx <= xs[0]:
                return approx
            cdf_out = np.empty(1, dtype=float)

            def f(x):
                if x <= lower:
                    return -p
                elif x >= upper:
                    return 1 - p
                return cdf(np.atleast_1d(x), cdf_out) - p

            return optimize.newton(f, approx, fprime=pdf, tol=1e-6)

        return find_inverse(points, coarse_result, out=out)

    @numpy_trans1d_method()
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

    @numpy_trans1d_method()
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

    @numpy_trans1d_method()
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

    @numpy_trans1d_method()
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
            raise ValueError(
                "Error, cannot use binned data with non-constant adjustment.")
        result = np.empty_like(bins)
        if dim < 0:
            dim = mesh.ndim + dim
        pdf = self.kernel.pdf
        if mesh.ndim == 1:
            pts = mesh.grid[dim]
            convolve(pts,
                     pts[..., None],
                     pdf,
                     result,
                     scaling=self.bandwidth * self.adjust,
                     weights=bins)
        else:
            left = np.index_exp[:] * dim
            right = np.index_exp[:] * (mesh.ndim - dim - 1)
            eval_pts = mesh.grid[dim]
            pts = eval_pts.view()
            pts.shape = (1, ) * dim + (
                len(pts), ) + (1, ) * (mesh.ndim - dim - 1)
            for i, p in enumerate(mesh.grid[dim]):
                access = left + (i, ) + right
                convolve(pts,
                         p,
                         pdf,
                         result[access],
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
        if N <= 2**11:
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

    @numpy_trans1d_method()
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

        pdf_out = np.empty((1, ), dtype=float)

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
            return 2**10
        return N


def fftdensity_from_binned(mesh,
                           bins,
                           kernel_rfft,
                           bw,
                           normed=False,
                           total_weights=None,
                           dim=-1):
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
        smth.shape = (1, ) * dim + (
            len(smth), ) + (1, ) * (mesh.ndim - dim - 1)

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
    mesh, DataHist = fast_bin(exog, [lower, upper],
                              N,
                              weights=weights,
                              bin_type='C')
    DataHist /= total_weights
    return mesh, fftdensity_from_binned(mesh, DataHist, kernel_rfft, bw)


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
        return 'C'

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
        if not self.closed:
            raise ValueError(
                "Cyclic boundary conditions can only be used with "
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
            raise ValueError(
                "Cyclic boundary conditions can only be used with "
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
        terms -= kernel.cdf(
            (L - exog) / bw)  # Remove the parts left of the lower bound

        terms += kernel.cdf(z + span)  # Repeat on the left
        terms -= kernel.cdf(
            (L - exog) / bw + span)  # Remove parts left of lower bounds

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

        return fftdensity(exog, self.kernel.rfft, bw, lower, upper, N,
                          self.weights, self.total_weights)

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
            raise ValueError(
                "Error, cannot use binned data with non-constant adjustment.")
        return fftdensity_from_binned(mesh, bins, self.kernel.rfft,
                                      self.adjust * self.bandwidth, normed,
                                      self.total_weights, dim)

    def grid_size(self, N=None):
        """
        Returns a valid grid size.
        """
        if N is None:
            if self.adjust.shape:
                return 2**10
            return 2**16
        return N  # 2 ** int(np.ceil(np.log2(N)))


def dctdensity_from_binned(mesh,
                           bins,
                           kernel_dct,
                           bw,
                           normed=False,
                           total_weights=None,
                           dim=-1):
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
        smth.shape = (1, ) * dim + (
            len(smth), ) + (1, ) * (mesh.ndim - dim - 1)

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
    mesh, DataHist = fast_bin(exog, [lower, upper],
                              N,
                              weights=weights,
                              bin_type='R')

    DataHist /= total_weights
    return mesh, dctdensity_from_binned(mesh, DataHist, kernel_dct, bw)


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
        return 'R'

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
            terms -= kernel.cdf(
                (L - exog) / bw)  # Remove the truncated part on the left
            terms += kernel.cdf(z1 - (2 * L / bw))  # Add the reflected part
            terms -= kernel.cdf(
                (exog - L) /
                bw)  # Remove the truncated part from the reflection

        if U < np.inf:
            terms += kernel.cdf(z1 - (2 * U / bw))  # Add the reflected part

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
            raise ValueError(
                "Error, cannot use binned data with non-constant adjustment.")
        return dctdensity_from_binned(mesh,
                                      bins,
                                      self.kernel.dct,
                                      self.bandwidth * self.adjust,
                                      normed,
                                      self.total_weights,
                                      dim=dim)

    def grid_size(self, N=None):
        """
        Returns a valid grid size.
        """
        if N is None:
            if self.adjust.shape:
                return 2**10
            return 2**16
        return N  # 2 ** int(np.ceil(np.log2(N)))


class Renormalization(Cyclic1D):
    r"""
    This method consists in using the Gaussian kernel method, but renormalize
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
            for unbounded pdf computation using the :py:func:`convolve`
            function.
        """
        if not self.bounded:
            return Cyclic1D.pdf(self, points, out)

        exog = self.exog
        points = points[..., np.newaxis]

        bw = self.bandwidth * self.adjust

        local_lower = (points - self.lower) / bw
        local_upper = (points - self.upper) / bw
        z = (points - exog) / bw

        kernel = self.kernel

        a1 = (kernel.cdf(local_lower) - kernel.cdf(local_upper))

        terms = kernel(z) * ((self.weights / bw) / a1)

        terms.sum(axis=-1, out=out)
        out /= self.total_weights

        return out

    @property
    def bin_type(self):
        """
        Type of the bins adapted for the method
        """
        return 'B'

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
        # here we assume that bw << est_R / 8 otherwise the FFT approximation
        # is bad anyway
        shift_N = N // 8
        comp_N = N + N // 4
        comp_lower = lower - R / 8
        comp_upper = upper + R / 8

        weights = self.weights

        mesh, density = fftdensity(exog, kernel.rfft, bw, comp_lower,
                                   comp_upper, comp_N, weights,
                                   self.total_weights)

        mesh = mesh.full()
        mesh = mesh[shift_N:shift_N + N]
        density = density[shift_N:shift_N + N]

        # Apply renormalization
        local_lower = (mesh - lower) / bw
        local_upper = (mesh - upper) / bw
        a1 = (kernel.cdf(local_lower) - kernel.cdf(local_upper))

        density /= a1

        return Grid(mesh, bounds=[lower, upper]), density


class _LinearCombinationKernel(Kernel1D):
    def __init__(self, ker):
        self._kernel = ker

    def pdf(self, points, out=None):
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
        out = self._kernel(points, out)
        out *= points
        return out

    __call__ = pdf


class LinearCombination(Cyclic1D):
    r"""
    This method uses the linear combination correction published in [KM1]_.

    The estimation is done with a modified kernel given by:

    .. math::

        \hat{K}(x;X,h,L,U) \triangleq \frac{a_2(l,u)
        - a_1(-u, -l) z}{a_2(l,u)a_0(l,u)
        - a_1(-u,-l)^2} K(z)

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

        local_lower = (self.lower - points) / bw
        local_upper = (self.upper - points) / bw
        z = (points - exog) / bw

        kernel = self.kernel

        a0 = kernel.cdf(local_upper) - kernel.cdf(local_lower)
        a1 = kernel.pm1(-local_lower) - kernel.pm1(-local_upper)
        a2 = kernel.pm2(local_upper) - kernel.pm2(local_lower)

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
        return 'B'

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
        # here we assume that bw << est_R / 8 otherwise the FFT approximation
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
        local_lower = (lower - grid) / bw
        local_upper = (upper - grid) / bw
        a0 = kernel.cdf(local_upper) - kernel.cdf(local_lower)
        a1 = kernel.pm1(-local_lower) - kernel.pm1(-local_upper)
        a2 = kernel.pm2(local_upper) - kernel.pm2(local_lower)

        density *= a2
        density -= a1 * z_density
        density /= a2 * a0 - a1 * a1

        return Grid(grid), density


# Note: We cannot use namedtuple, because we are defining __call__.
class Transform(tuple):
    """Named tuple storing the three function needed to transform an axis."""
    __slots__ = ()

    def __new__(cls, call, inv, Dinv):
        return tuple.__new__(cls, [call, inv, Dinv])

    @property
    def __call__(self):
        """Map coordinates from the original axis to the transformed axis."""
        return self[0]

    @property
    def inv(self):
        """
        Map coordinates from the transformed axis back to the original one.
        """
        return self[1]

    @property
    def Dinv(self):
        """Derivative of the inverse transform function."""
        return self[2]


def _inverse(x, out=None):
    return np.divide(1, x, out)


#: Transform object for a log-transform mapping [0, +oo] to [-oo, +oo]
LogTransform = Transform(np.log, np.exp, np.exp)
#: Transform object for an exp-transform mapping [-oo, +oo] to [o, +oo]
ExpTransform = Transform(np.exp, np.log, _inverse)


def transform_distribution(xs, ys, Dinv, out):
    r"""
    Transform a distribution into another one by a change a variable.

    Parameters
    ----------
    xs: ndarray
        Evaluation points of the distribution
    ys: ndarray
        Distribution value on the points xs
    Dinv: func
        Function evaluating the derivative of the inverse transformation
        function
    out: ndarray
        Array in which to store the result

    Returns
    -------
    ndarray
        The variable ``out``, updated wih the transformed distribution

    Notes
    -----
    Given a random variable :math:`X` of distribution :math:`f_X`, the random
    variable :math:`Y = g(X)` has a distribution :math:`f_Y` given by:

    .. math::

        f_Y(y) = \left| \frac{1}{g'(g^{-1}(y))} \right| \cdot f_X(g^{-1}(y))

    """
    di = np.asarray(Dinv(xs))
    np.abs(di, out=di)
    _inverse(di, out=di)
    np.multiply(di, ys, out=out)
    return out


def create_transform(obj, inv=None, Dinv=None):
    """
    Create a transform object.

    Parameters
    ----------
    obj: fun
        This can be either simple a function, or a function-object with an
        'inv' and/or 'Dinv' attributes containing the inverse function and its
        derivative (respectively)
    inv: fun
        If provided, inverse of the main function
    Dinv: fun
        If provided, derivative of the inverse function

    Returns
    -------
    transform : :py:class:`Transform`
        A transform object with function, inverse and derivative of the inverse

    Notes
    -----
    The inverse function must be provided, either as argument or as attribute
    to the object. The derivative of the inverse will be estimated numerically
    if not provided.

    All the functions should accept an ``out`` argument to store the result.
    """
    if isinstance(obj, Transform):
        return obj
    fct = obj.__call__
    if inv is None:
        if not hasattr(obj, 'inv'):
            raise AttributeError(
                "Error, transform object must have a 'inv' "
                "attribute or you must specify the 'inv' argument")
        inv = obj.inv
    if Dinv is None:
        if hasattr(obj, 'Dinv'):
            Dinv = obj.Dinv
        else:

            @numpy_trans1d()
            def Dinv(x, out):
                dx = x * 1e-9
                dx[x == 0] = np.min(dx[x != 0])
                np.divide(inv(x + dx) - inv(x - dx), 2 * dx, out=out)
                return out

    return Transform(fct, inv, Dinv)


class _transKDE(object):
    def __init__(self, method):
        self.method = method

    def copy(self):
        res = _transKDE(self.method)
        res.__dict__.update(self.__dict__)
        return res

    def fit(self):
        return self.method.fit(self)


class Transform1D(KDE1DMethod):
    r"""
    Compute the Kernel Density Estimate of a dataset, transforming it first to
    a domain where distances are "more meaningful".

    Often, KDE is best estimated in a different domain. This object takes a KDE
    object (or one compatible), and
    a transformation function.

    Given a random variable :math:`X` of distribution :math:`f_X`, the random
    variable :math:`Y = g(X)` has a distribution :math:`f_Y` given by:

    .. math::

        f_Y(y) = \left| \frac{1}{g'(g^{-1}(y))} \right| \cdot f_X(g^{-1}(y))

    In our term, :math:`Y` is the random variable the user is interested in,
    and :math:`X` the random variable we can estimate using the KDE. In this
    case, :math:`g` is the transform from :math:`Y` to :math:`X`.

    So to estimate the distribution on a set of points given in :math:`x`, we
    need a total of three functions:

        - Direct function: transform from the original space to the one in
          which the KDE will be perform (i.e. :math:`g^{-1}: y \mapsto x`)
        - Invert function: transform from the KDE space to the original one
          (i.e. :math:`g: x \mapsto y`)
        - Derivative of the invert function

    If the derivative is not provided, it will be estimated numerically.
    """
    def __init__(self, trans, method=None, inv=None, Dinv=None):
        """
        Parameters
        ----------
        trans:
            Either a simple function, or a function object with
            attributes `inv` and `Dinv` to use in case they are not provided as
            arguments. The helper :py:func:`create_transform` will provide
            numeric approximation of the derivative if required.
        method:
            instance of KDE1DMethod used in the transformed domain.
            Default is :py:class:`Reflection`
        inv:
            Invert of the function. If not provided, `trans` must have
            it as attribute.
        Dinv:
            Derivative of the invert function.

        Notes
        -----
        all given functions should accept an optional ``out`` argument to get
        a pre-allocated array to store its result.
        Also the ``out`` parameter may be one of the input argument.
        """
        super(Transform1D, self).__init__()
        self.trans = create_transform(trans, inv, Dinv)
        if method is None:
            method = Reflection1D()
        self._method = method
        self._clean_attrs()

    #: Name of the method, for presentation purposes
    name = 'transformkde1d'

    _to_clean = [
        '_bandwidth', '_adjust', '_weights', '_kernel', '_total_weights'
    ]

    def _clean_attrs(self):
        """
        Remove attributes not needed for this class
        """
        for attr in Transform1D._to_clean:
            if hasattr(self, attr):
                delattr(self, attr)

    @property
    def method(self):
        """
        Method used in the transformed space.

        Notes
        -----
        The method can only be changed before fitting!
        """
        return self._method

    def _trans_kde(self, kde):
        trans_kde = _transKDE(self.method)
        trans_kde.lower = self.trans(kde.lower)
        trans_kde.upper = self.trans(kde.upper)
        trans_kde.exog = self.trans(kde.exog)

        copy_attrs = [
            'weights', 'adjust', 'kernel', 'bandwidth', 'total_weights',
            'ndim', 'npts', 'axis_type'
        ]

        for attr in copy_attrs:
            setattr(trans_kde, attr, getattr(kde, attr))
        return trans_kde

    @method.setter
    def method(self, m):
        if self._fitted:
            self._method = m.fit(self._trans_kde(self))
        else:
            self._method = m

    def update_inputs(self, exog, weights=1., adjust=1.):
        """
        Update all the variable lengths inputs at once to ensure consistency
        """
        exog = np.atleast_1d(exog)
        if exog.ndim != 1:
            raise ValueError(
                "Error, exog must be a 1D array (nb dimensions: {})".format(
                    exog.ndim))
        weights = np.asarray(weights).squeeze()
        adjust = np.asarray(adjust).squeeze()
        if weights.ndim != 0 and weights.shape != exog.shape:
            raise ValueError("Error, weights must be either a single number, "
                             "or an array the same shape as exog")
        if adjust.ndim != 0 and adjust.shape != exog.shape:
            raise ValueError(
                "Error, adjust must be either a single number, or an array "
                "the same shape as exog")
        self._exog = exog
        self.method.update_inputs(self.trans(exog), weights, adjust)

    @property
    def to_bin(self):
        """
        Return the exog data, transformed into the domain in which they should
        be binned.
        """
        return self.method.exog

    @KDEMethod.exog.setter
    def exog(self, val):
        val = np.atleast_1d(val).reshape(self._exog.shape)
        self.method.exog = self.trans(val)
        self._exog = val

    @KDEMethod.lower.setter
    def lower(self, val):
        val = float(val)
        trans_val = self.trans(val)
        self.method.lower = trans_val
        self._lower = val

    @KDEMethod.upper.setter
    def upper(self, val):
        val = float(val)
        trans_val = self.trans(val)
        self.method.upper = trans_val
        self._upper = val

    # List of attributes to forward to the method object
    _fwd_attrs = [
        'weights', 'adjust', 'kernel', 'bandwidth', 'total_weights',
        'axis_type'
    ]

    def fit(self, kde):
        """
        Method called by the KDE object right after fitting to allow for
        one-time calculation.

        This method copy, and transform, the various attributes of the KDE.
        """
        kde = filter_exog(kde, self._method.bin_type)
        self._kernel = self._method._kernel
        fitted = super(Transform1D, self).fit(kde, False)
        fitted._clean_attrs()

        trans_method = self.method.fit(fitted._trans_kde(kde))
        fitted._method = trans_method
        fitted._fitted = True

        return fitted

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
        trans = self.trans
        pts = trans(points)
        pdf = np.empty(points.shape, points.dtype)
        self.method(pts, out=pdf)
        return transform_distribution(pts, pdf, trans.Dinv, out=out)

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
        if span is not None:
            span = self.trans(span[0]), self.trans(span[1])
        xs, ys = self.method.grid(N, cut, span)
        trans = self.trans
        out = np.empty(ys.shape, ys.dtype)
        transform_distribution(xs.full(), ys, trans.Dinv, out=out)
        xs.transform(self.trans.inv)
        return xs, out

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
        return self.method.cdf(self.trans(points), out)

    def cdf_grid(self, N=None, cut=None):
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
        """
        xs, ys = self.method.cdf_grid(N, cut)
        xs.transform(self.trans.inv)
        return xs, ys

    def sf(self, points, out=None):
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
        """
        return self.method.sf(self.trans(points), out)

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
        """
        xs, ys = self.method.sf_grid(N, cut)
        xs.transform(self.trans.inv)
        return xs, ys

    def icdf(self, points, out=None):
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
        This method computes the icdf in the transformed axis, and transform
        the result back.
        """
        out = self.method.icdf(points, out)
        self.trans.inv(out, out=out)
        return out

    def icdf_grid(self, N=None, cut=None):
        r"""
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
        This method computes the icdf in the transformed axis, and transform
        the result back.
        """
        xs, ys = self.method.icdf_grid(N, cut)
        self.trans.inv(ys, out=ys)
        return xs, ys

    def isf(self, points, out=None):
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
        This method computes the isf in the transformed axis, and transform the
        result back.
        """
        out = self.method.isf(points, out)
        self.trans.inv(out, out=out)
        return out

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
        This method computes the isf in the transformed axis, and transform the
        result back.
        """
        xs, ys = self.method.isf_grid(N, cut)
        self.trans.inv(ys, out=ys)
        return xs, ys

    def transform_axis(self, values):
        '''
        Function used to transform an axis, or None for no transformation
        '''
        return self.trans(values)

    def restore_axis(self, transformed_values):
        '''
        Inverse function of transform_axis
        '''
        return self.trans.inv(transformed_values)

    def transform_bins(self, mesh, bins, axis=-1):
        '''
        Function used to adapt the bin values when restoring an axis
        '''
        out = np.empty_like(bins)
        xs = mesh.sparse()[axis]
        return transform_distribution(xs, bins, self.trans.Dinv, out=out)


def _add_fwd_attr(cls, to_fwd, attr):
    try:
        fwd_obj = getattr(cls, to_fwd)
        doc = getattr(fwd_obj, '__doc__')
    except AttributeError:
        doc = 'Attribute forwarded to {}'.format(to_fwd)

    def getter(self):
        return getattr(getattr(self, to_fwd), attr)

    def setter(self, val):
        setattr(getattr(self, to_fwd), attr, val)

    def deleter(self):
        delattr(getattr(self, to_fwd), attr)

    setattr(cls, attr, property(getter, setter, deleter, doc=doc))


for attr in Transform1D._fwd_attrs:
    _add_fwd_attr(Transform1D, 'method', attr)

#: Alias toward the default, unbounded, method
Unbounded1D = Reflection1D
