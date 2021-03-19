r"""
Module implementing kernel-based estimation of density of probability.

Given a kernel :math:`K`, the density function is estimated from a sampling
:math:`X = \{X_i \in \mathbb{R}^n\}_{i\in\{1,\ldots,m\}}` as:

.. math::

    f(\mathbf{z}) \triangleq \frac{1}{hW} \sum_{i=1}^m \frac{w_i}{\lambda_i}
    K\left(\frac{X_i-\mathbf{z}}{h\lambda_i}\right)

    W = \sum_{i=1}^m w_i

where :math:`h` is the bandwidth of the kernel, :math:`w_i` are the weights of
the data points and :math:`\lambda_i` are the adaptation factor of the kernel
width.

The kernel is a function of :math:`\mathbb{R}^n` such that:

.. math::

    \begin{array}{rclcl}
       \idotsint_{\mathbb{R}^n} f(\mathbf{z}) d\mathbf{z}
       & = & 1 & \Longleftrightarrow & \text{$f$ is a probability}\\
       \idotsint_{\mathbb{R}^n} \mathbf{z}f(\mathbf{z}) d\mathbf{z} &=&
       \mathbf{0} & \Longleftrightarrow & \text{$f$ is
       centered}\\
       \forall \mathbf{u}\in\mathbb{R}^n, \|\mathbf{u}\|
       = 1\qquad\int_{\mathbb{R}} t^2f(t \mathbf{u}) dt &\approx&
       1 & \Longleftrightarrow & \text{The co-variance matrix of $f$ is close
       to be the identity.}
    \end{array}

The constraint on the covariance is only required to provide a uniform meaning
for the bandwidth of the kernel.

If the domain of the density estimation is bounded to the interval
:math:`[L,U]`, the density is then estimated with:

.. math::

    f(x) \triangleq \frac{1}{hW} \sum_{i=1}^n \frac{w_i}{\lambda_i}
    \hat{K}(x;X,\lambda_i h,L,U)

where :math:`\hat{K}` is a modified kernel that depends on the exact method
used. Currently, only 1D KDE supports bounded domains.
"""

import numpy as np
from .kde_utils import atleast_2df, AxesType
from .kde_multivariate import Multivariate
from copy import copy as shallow_copy

# default_method = kde1d_methods.Reflection
# default_method = kdend_methods.KDEnDMethod
default_method = Multivariate


class KDE(object):
    r"""
    Prepare a nD kernel density estimation, possible on a bounded domain.

    Parameters
    ----------
    exog: ndarray
        2D array DxN with the N input points in D dimension.
    kwords: dict
        setting attributes at construction time. Any named argument will be
        equivalent to setting the property after the fact. For example::

            >>> xs = [1,2,3]
            >>> k = KDE1D(xs, lower=0)

        will be equivalent to::

            >>> k = KDE1D(xs)
            >>> k.lower = 0

    Notes
    -----

    This is the object from which you define and prepare your model. Once
    prepared, the model needs to be fitted, which returns an estimator object.

    The model knows about a set number of parameters that all methods must
    account for. However, check on the method's documentation to make sure if
    there aren't other parameters.
    """
    def __init__(self, exog, **kwords):
        self._exog = None
        self._lower = -np.inf
        self._upper = np.inf
        self._method = None
        self._axis_type = AxesType()
        self._weights = np.array(1.)
        self._adjust = np.array(1.)
        self._bandwidth = None
        self._kernel = None

        self.exog = exog

        for n in kwords:
            if hasattr(self, n):
                setattr(self, n, kwords[n])
            else:
                raise AttributeError(
                    "Error, unknown attribute: '{}'".format(n))

        if self._method is None:
            self._method = default_method()

    def copy(self):
        """
        Shallow copy of the KDE object
        """
        res = shallow_copy(self)
        res._method = self._method.copy()
        res._axis_type = shallow_copy(self._axis_type)
        res._lower = shallow_copy(self._lower)
        res._upper = shallow_copy(self._upper)
        return res

    @property
    def lower(self):
        """
        List with the lower bound for the domain on each dimension. None for
        automatic computation of the bound.
        """
        return self._lower

    @lower.setter
    def lower(self, val):
        self._lower = val

    @lower.deleter
    def lower(self):
        self._lower = -np.inf * np.ones(self.ndim, dtype=float)

    @property
    def upper(self):
        """
        List with the upper bound for the domain on each dimension. None for
        automatic computation of the bound.
        """
        return self._upper

    @upper.setter
    def upper(self, val):
        self._upper = val

    @upper.deleter
    def upper(self):
        self._upper = np.inf * np.ones(self.ndim, dtype=float)

    @property
    def exog(self):
        """
        2D array with exogenous data. The array has shape NxD for N points in D
        dimension.
        """
        return self._exog

    @exog.setter
    def exog(self, value):
        value = atleast_2df(value).astype(float)
        ndim = value.shape[1]
        if ndim != self.ndim:
            self._axis_type = AxesType('C' * ndim)
            if ndim == 1:
                self._lower = -np.inf
                self._upper = np.inf
            else:
                self._lower = [-np.inf] * ndim
                self._upper = [np.inf] * ndim
        self._exog = value

    @property
    def ndim(self):
        """
        Number of dimensions of the problem.
        """
        if self._exog is None:
            return 0
        return self._exog.shape[1]

    @property
    def npts(self):
        """
        Number of points in the problem.
        """
        return self._exog.shape[0]

    @property
    def axis_type(self):
        """
        Set the types of Axis. Defined in :py:class:`AxesTypes`
        """
        return self._axis_type

    @axis_type.setter
    def axis_type(self, value):
        self._axis_type = value

    @axis_type.deleter
    def axis_type(self):
        self._axis_type = AxesType()

    @property
    def method(self):
        """
        Method used to estimate the KDE.
        """
        return self._method

    @method.setter
    def method(self, m):
        if isinstance(m, type):
            self._method = m()
        else:
            self._method = m

    @property
    def weights(self):
        """
        1D array containing, for each point, its weight.
        """
        return self._weights

    @weights.setter
    def weights(self, value):
        value = np.asarray(value, dtype=float)
        if value.ndim > 1:
            raise ValueError(
                "Error, the weights must be a scalar or a 1D array")
        if value.ndim == 0:
            del self.weights
        else:
            self._weights = value

    @weights.deleter
    def weights(self):
        self._weights = np.array(1.)

    @property
    def adjust(self):
        """
        Multiplicating factor to apply to the bandwidth: it can be a single
        value or a 1D array.
        """
        return self._adjust

    @adjust.setter
    def adjust(self, value):
        value = np.asarray(value, dtype=float)
        if value.ndim > 1:
            raise ValueError("Error, adjust must be a 1D array")
        self._adjust = value

    @adjust.deleter
    def adjust(self):
        self._adjust = np.array(1.)

    @property
    def bandwidth(self):
        """
        Method to estimate the bandwidth, or bandwidth to use.
        """
        return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, value):
        self._bandwidth = value

    @property
    def kernel(self):
        """
        Kernel to use for the bandwidth estimation.
        """
        return self._kernel

    @kernel.setter
    def kernel(self, value):
        self._kernel = value

    def fit(self):
        """
        Compute the bandwidths and find the proper KDE method for the current
        dataset.

        Returns
        -------
        estimator : kde_methods.KDEMethod
            An object fully instantiated with pre-computed parameters for
            efficient estimation of the KDE

        Notes
        -----
        The returned object doesn't maintain any reference to the
        :py:class:`KDE` object, which can therefore be modified freely.
        However, the data are not (always) copied either, so modifying the data
        in place may modify the returned object.
        """
        return self.method.fit(self)

    @property
    def total_weights(self):
        """
        Total weights for the current dataset.
        """
        if self._weights.shape:
            return self._weights.sum()
        return self.npts
