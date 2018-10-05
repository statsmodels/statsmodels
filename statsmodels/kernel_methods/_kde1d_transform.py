"""
This module implements the Transform1D KDE estimation method, which transform the bounded domain
into an unbounded one and back.
"""
from __future__ import division, absolute_import, print_function

import numpy as np
from .kde_utils import numpy_trans1d_method, namedtuple, numpy_trans1d
from ._kde1d_methods import KDE1DMethod
from ._kde_methods import KDEMethod, filter_exog
from ._kde1d_reflection import Reflection1D

_Transform_doc = "Named tuple storing the three function needed to transform an axis"
_Transform_field_docs = ["Map coordinates from the original axis to the transformed axis.",
                         "Map coordinates from the transformed axis back to the original one.",
                         "Derivative of the inverse transform function."]
Transform = namedtuple('Transform', ['__call__', 'inv', 'Dinv'],
                       doc=_Transform_doc,
                       field_docs=_Transform_field_docs)


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
        This can be either simple a function, or a function-object with an 'inv' and/or 'Dinv' attributes
        containing the inverse function and its derivative (respectively)
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
    The inverse function must be provided, either as argument or as attribute to the object. The derivative of the
    inverse will be estimated numerically if not provided.

    All the functions should accept an ``out`` argument to store the result.
    """
    if isinstance(obj, Transform):
        return obj
    fct = obj.__call__
    if inv is None:
        if not hasattr(obj, 'inv'):
            raise AttributeError("Error, transform object must have a 'inv' "
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
            arguments. The helper :py:func:`create_transform` will provide numeric
            approximation of the derivative if required.
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

    _to_clean = ['_bandwidth', '_adjust',
                 '_weights', '_kernel', '_total_weights']

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

        copy_attrs = ['weights', 'adjust', 'kernel', 'bandwidth',
                      'total_weights', 'ndim', 'npts', 'axis_type']

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
            raise ValueError("Error, exog must be a 1D array (nb dimensions: {})".format(exog.ndim))
        weights = np.asarray(weights).squeeze()
        adjust = np.asarray(adjust).squeeze()
        if weights.ndim != 0 and weights.shape != exog.shape:
            raise ValueError("Error, weights must be either a single number, or an array the same shape as exog")
        if adjust.ndim != 0 and adjust.shape != exog.shape:
            raise ValueError("Error, adjust must be either a single number, or an array the same shape as exog")
        self._exog = exog
        self.method.update_inputs(self.trans(exog), weights, adjust)

    @property
    def to_bin(self):
        """
        Return the exog data, transformed into the domain in which they should be binned.
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
    _fwd_attrs = ['weights', 'adjust', 'kernel', 'bandwidth',
                  'total_weights', 'axis_type']

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
            for unbounded pdf computation using the :py:func:`convolve` function.
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
            Values of the inverse survival function for each position of the grid.
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
        This method computes the icdf in the transformed axis, and transform the result back.
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
        This method computes the icdf in the transformed axis, and transform the result back.
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
            Returns the ``out`` variable, updated with the inverse survival function.

        Notes
        -----
        This method computes the isf in the transformed axis, and transform the result back.
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
        This method computes the isf in the transformed axis, and transform the result back.
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
