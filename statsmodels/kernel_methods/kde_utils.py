"""
:Author: Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>

Module contained a variety of small useful functions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
try:
    from inspect import getfullargspec as getargspec
except ImportError:
    from inspect import getargspec

import numpy as np

from ..compat.python import string_types, range
from ._grid import Grid  # noqa
from ._grid_interpolation import GridInterpolator  # noqa
from .namedtuple import namedtuple  # NOQA need to be defined here


# Find the largest float available for this numpy
if hasattr(np, 'float128'):
    large_float = np.float128
elif hasattr(np, 'float96'):
    large_float = np.float96
else:
    large_float = np.float64


def finite(val):
    return val is not None and np.isfinite(val)


def atleast_2df(*arys):
    """
    Return at least a 2D array, fortran style (e.g. adding dimensions at the end)
    """
    res = []
    for ary in arys:
        ary = np.asanyarray(ary)
        if ary.ndim == 0:
            ary = ary.reshape(1, 1)
        elif ary.ndim == 1:
            ary = ary[:, np.newaxis]
        res.append(ary)
    if len(res) == 1:
        return res[0]
    return res


def make_ufunc(nin=None, nout=1):
    """
    Decorator used to create a ufunc using `np.frompyfunc`. Note that the
    returns array will always be of dtype 'object'. You should use the `out` if
    you know the wanted type for the output.

    :param int nin: Number of input. Default is found by using
        ``inspect.getfullargspec``
    :param int nout: Number of output. Default is 1.
    """
    def f(fct):
        if nin is None:
            Nin = len(getargspec(fct).args)
        else:
            Nin = nin
        return np.frompyfunc(fct, Nin, nout)
    return f


def _process_trans_args(z, out, input_dim, output_dim, in_dtype, out_dtype):
    """
    This function is the heart of the numpy_trans* functions.
    """
    z = np.asarray(z)
    if in_dtype is not None:
        z = z.astype(in_dtype)
    if z.ndim > 2:
        raise ValueError('Error, the input array must be at most 2D')
    z = atleast_2df(z)
    input_shape = z.shape
    need_transpose = False
    # Compute data shape (i.e. input without the dimension)
    z_empty = False
    if input_dim <= 0:
        npts = input_shape[0]
        input_dim = input_shape[-1]
    else:
        if input_shape[-1] == input_dim:
            npts = input_shape[0]
        elif input_shape[0] == input_dim:
            npts = input_shape[1]
            need_transpose = True
        else:
            raise ValueError("Error, the input array is of dimension {0} "
                             "(expected: {1})".format(input_shape[-1], input_dim))
    # Allocate the output
    if out is None:
        # Compute the output shape
        if output_dim > 1:
            if need_transpose:
                output_shape = (output_dim, npts)
            else:
                output_shape = (npts, output_dim)
        else:
            output_shape = (npts,)
        if out_dtype is None:
            out_dtype = z.dtype
            if issubclass(out_dtype.type, np.integer):
                out_dtype = np.float64
        out = np.empty(output_shape, dtype=out_dtype)
        write_out = out.view()
        if z_empty and output_dim == 1:
            out.shape = ()
    else:
        write_out = out.view()
    # Transpose if needed
    if need_transpose:
        write_out = write_out.T
        z = z.T
    return z, write_out, out


def numpy_trans(input_dim, output_dim, out_dtype=None, in_dtype=float):
    """
    Decorator to create a function taking a single array-like argument and return a numpy array with the same number of
    points.

    The function will always get an input and output with the last index corresponding to the dimension of the problem.

    Parameters
    ----------
    input_dim: int
        Number of dimensions of the input. The behavior depends on the value:
            > 0 : There is a dimension, and its size is known. The dimension should be the first or last index. If it is
                  on the first, the arrays are transposed before being sent to the function.
            else: The last index is the dimension, but it may be any number. A 1D array will be considered n points in
                  1D.

    output_dim: int
        Dimension of the output. If more than 1, the last index of the output array is the dimension. It cannot be 0 or
        less.

    out_dtype: dtype or None
        Expected types of the output array.
        If the output array is created by this function, dtype specifies its type. If dtype is None, the output array is
        given the same as the input array, unless it is an integer, in which case the output will be a float64.

    in_dtype: dtype or None
        If not None, the input array will be converted to this type before being passed on.

    Notes
    -----
    If input_dim is not 0, the function will always receive a 2D array with the second index for the dimension.
    """
    if out_dtype is not None:
        out_dtype = np.dtype(out_dtype)
    if in_dtype is not None:
        in_dtype = np.dtype(in_dtype)
    if output_dim <= 0:
        raise ValueError("Error, the number of output dimension must be strictly more than 0.")

    def decorator(fct):
        @functools.wraps(fct)
        def f(z, out=None):
            z, write_out, out = _process_trans_args(z, out, input_dim, output_dim,
                                                    in_dtype, out_dtype)
            fct(z, out=write_out)
            return out
        return f
    return decorator


def _process_trans1d_args(z, out, in_dtype, out_dtype):
    z = np.asarray(z)
    if in_dtype is not None:
        z = z.astype(in_dtype)
    npts = np.prod(z.shape)
    if npts == 0:
        npts = 1
    if out is None:
        if out_dtype is None:
            dtype = z.dtype
        else:
            dtype = out_dtype
        if issubclass(dtype.type, np.integer):
            dtype = np.float64
        out = np.empty(z.shape, dtype=dtype)
    return z, out, out


def numpy_trans1d(out_dtype=None, in_dtype=None):
    """
    This decorator helps provide a uniform interface to 1D numpy transformation functions.

    The returned function takes any array-like argument and transform it as a 1D ndarray sent to the decorated function.
    If the `out` argument is not provided, it will be allocated with the same size and shape as the first argument. And
    as with the first argument, it will be reshaped as a 1D ndarray before being sent to the function.

    Examples
    --------

    The following example illustrate how a 2D array will be passed as 1D, and the output allocated as the input
    argument:

    >>> @numpy_trans1d()
    ... def broadsum(z, out):
    ...   out[:] = np.sum(z, axis=0)
    >>> broadsum([[1,2],[3,4]])
    array([[ 10.,  10.], [ 10.,  10.]])

    """
    if out_dtype is not None:
        out_dtype = np.dtype(out_dtype)
    if in_dtype is not None:
        in_dtype = np.dtype(in_dtype)

    def decorator(fct):
        @functools.wraps(fct)
        def f(z, out=None):
            z, out, write_out = _process_trans1d_args(z, out, in_dtype, out_dtype)
            fct(z, write_out)
            return out
        return f
    return decorator


def numpy_trans_method(input_dim, output_dim, out_dtype=None, in_dtype=float):
    """
    Decorator to create a method taking a single array-like argument and return a numpy array with the same number of
    points.

    The function will always get an input and output with the last index corresponding to the dimension of the problem.

    Parameters
    ----------
    input_dim: int or str
        Number of dimensions of the input. The behavior depends on the value:
            > 0 : There is a dimension, and its size is known. The dimension should be the first or last index. If it is
                  on the first, the arrays are transposed before being sent to the function.
            else: The last index is the dimension, but it may be any number. A 1D array will be considered n points in
                  1D.
        If a string, it should be the name of an attribute containing the input dimension.

    output_dim: int or str
        Dimension of the output. If more than 1, the last index of the output array is the dimension. If cannot be 0 or
        less.
        If a string, it should be the name of an attribute containing the output dimension

    out_dtype: dtype or None
        Expected types of the output array.
        If the output array is created by this function, dtype specifies its type. If dtype is None, the output array is
        given the same as the input array, unless it is an integer, in which case the output will be a float64.

    in_dtype: dtype or None
        If not None, the input array will be converted to this type before being passed on.

    Notes
    -----
    If input_dim is not 0, the function will always receive a 2D array with the second index for the dimension.
    """
    if output_dim <= 0:
        raise ValueError("Error, the number of output dimension must be strictly more than 0.")
    # Resolve how to get input dimension
    if isinstance(input_dim, string_types):
        def get_input_dim(self):
            return getattr(self, input_dim)
    else:
        def get_input_dim(self):
            return input_dim
    # Resolve how to get output dimension
    if isinstance(output_dim, string_types):
        def get_output_dim(self):
            return getattr(self, output_dim)
    else:
        def get_output_dim(self):
            return output_dim
    if out_dtype is not None:
        out_dtype = np.dtype(out_dtype)
    if in_dtype is not None:
        in_dtype = np.dtype(in_dtype)

    # Decorator itself
    def decorator(fct):
        @functools.wraps(fct)
        def f(self, z, out=None):
            z, write_out, out = _process_trans_args(z, out, get_input_dim(self), get_output_dim(self),
                                                    in_dtype, out_dtype)
            fct(self, z, out=write_out)
            return out
        return f
    return decorator


def numpy_trans1d_method(out_dtype=None, in_dtype=None):
    '''
    This is the method equivalent to :py:func:`numpy_trans1d`
    '''
    if out_dtype is not None:
        out_dtype = np.dtype(out_dtype)
    if in_dtype is not None:
        in_dtype = np.dtype(in_dtype)

    def decorator(fct):
        @functools.wraps(fct)
        def f(self, z, out=None):
            z, real_out, write_out = _process_trans1d_args(z, out, in_dtype, out_dtype)
            fct(self, z, out=write_out)
            return real_out
        return f
    return decorator


class AxesType(object):
    """
    Class defining the type of each axis.

    The type of each axis is defined as a single letter. The basic types are:

        'c'
            Continuous axis
        'u'
            Discrete, un-ordered, axis
        'o'
            Discrete, ordered, axis
    """
    _dtype = np.dtype(np.str_).char + '1'

    def __init__(self, value='c'):
        self._types = np.empty((), dtype=self._dtype)
        self.set(value)

    def set(self, value):
        value = np.array(list(value), dtype=self._types.dtype)
        self._types = value

    def copy(self):
        return AxesType(self)

    def __len__(self):
        return len(self._types)

    def __repr__(self):
        return "AxesType('{}')".format(''.join(self._types))

    def __str__(self):
        return ''.join(self._types)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return ''.join(self._types[idx])
        return self._types[idx]

    def __iter__(self):
        return iter(self._types)

    def __setitem__(self, idx, value):
        if isinstance(idx, slice):
            value = list(value)
            self._types[idx] = value
        else:
            self._types[idx] = value

    def __delitem__(self, idx):
        del self._types[idx]

    def resize(self, nl, default='c'):
        cur_l = len(self)
        if nl < cur_l:
            self._types = self._types[nl:]
        elif nl > cur_l:
            self._types = np.resize(self._types, nl)
            self._types[cur_l:] = default

    def __eq__(self, other):
        if isinstance(other, AxesType):
            return self._types == other._types
        return self._types == other

    def __ne__(self, other):
        if isinstance(other, AxesType):
            return self._types != other._types
        return self._types != other

#
from scipy import sqrt
from numpy import finfo, asarray, asfarray, zeros

_epsilon = sqrt(finfo(float).eps)


def approx_jacobian(x, func, epsilon, *args):
    """
    Approximate the Jacobian matrix of callable function func

    :param ndarray x: The state vector at which the Jacobian matrix is desired
    :param callable func: A vector-valued function of the form f(x,*args)
    :param ndarray epsilon: The peturbation used to determine the partial derivatives
    :param tuple args: Additional arguments passed to func

    :returns: An array of dimensions (lenf, lenx) where lenf is the length
         of the outputs of func, and lenx is the number of

    .. note::

         The approximation is done using forward differences

    """
    x0 = asarray(x)
    x0 = asfarray(x0, dtype=x0.dtype)
    epsilon = x0.dtype.type(epsilon)
    f0 = func(*((x0,) + args))
    jac = zeros([len(x0), len(f0)], dtype=x0.dtype)
    dx = zeros(len(x0), dtype=x0.dtype)
    for i in range(len(x0)):
        dx[i] = epsilon
        jac[i] = (func(*((x0 + dx,) + args)) - f0) / epsilon
        dx[i] = 0.0
    return jac.transpose()
