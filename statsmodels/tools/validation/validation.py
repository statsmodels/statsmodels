import numpy as np
import pandas as pd


def _right_squeeze(arr, stop_dim=0):
    """
    Remove trailing singleton dimensions

    Parameters
    ----------
    arr : ndarray
        Input array
    stop_dim : int
        Dimension where checking should stop so that shape[i] is not checked
        for i < stop_dim

    Returns
    -------
    squeezed : ndarray
        Array with all trailing singleton dimensions (0 or 1) removed.
        Singleton dimensions for dimension < stop_dim are retained.
    """
    last = arr.ndim
    for s in reversed(arr.shape):
        if s > 1:
            break
        last -= 1
    last = max(last, stop_dim)

    return arr.reshape(arr.shape[:last])


def array_like(obj, name, dtype=np.double, ndim=1, maxdim=None,
               shape=None, order='C', contiguous=False):
    """
    Convert array-like to an array and check conditions

    Parameters
    ----------
    obj : array_like
         An array, any object exposing the array interface, an object whose
        __array__ method returns an array, or any (nested) sequence.
    name : str
        Name of the variable to use in exceptions
    dtype : {None, numpy.dtype}
        Required dtype. Default is double. If None, does not change the dtype
        of obj (if present) or uses NumPy to automatically detect the dtype
    ndim : {int, None}
        Required number of dimensions of obj. If None, no check is performed.
        If the numebr of dimensions of obj is less than ndim, additional axes
        are inserted on the right. See examples.
    maxdim : {int, None}
        Maximum allowed dimension.  Use ``maxdim`` instead of ``ndim`` when
        inputs are allowed to have ndim 1, 2, ..., or maxdim.
    shape : {tuple[int], None}
        Required shape obj.  If None, no check is performed. Partially
        restricted shapes can be checked using None. See examples.
    order : {'C', 'F'}
        Order of the array
    contiguous : bool
        Ensure that the array's data is contiguous with order ``order``

    Examples
    --------
    Convert a list or pandas series to an array
    >>> import pandas as pd
    >>> x = [0, 1, 2, 3]
    >>> a = array_like(x, 'x', ndim=1)
    >>> a.shape
    (4,)

    >>> a = array_like(pd.Series(x), 'x', ndim=1)
    >>> a.shape
    (4,)
    >>> type(a.orig)
    pandas.core.series.Series

    Squeezes singleton dimensions when required
    >>> x = np.array(x).reshape((4, 1))
    >>> a = array_like(x, 'x', ndim=1)
    >>> a.shape
    (4,)

    Right-appends when required size is larger than actual
    >>> x = [0, 1, 2, 3]
    >>> a = array_like(x, 'x', ndim=2)
    >>> a.shape
    (4, 1)

    Check only the first and last dimension of the input
    >>> x = np.arange(4*10*4).reshape((4, 10, 4))
    >>> y = array_like(x, 'x', ndim=3, shape=(4, None, 4))

    Check only the first two dimensions
    >>> z = array_like(x, 'x', ndim=3, shape=(4, 10))

    Raises ValueError if constraints are not satisfied
    >>> z = array_like(x, 'x', ndim=2)
    Traceback (most recent call last):
     ...
    ValueError: x is required to have ndim 2 but has ndim 3

    >>> z = array_like(x, 'x', shape=(10, 4, 4))
    Traceback (most recent call last):
     ...
    ValueError: x is required to have shape (10, 4, 4) but has shape (4, 10, 4)

    >>> z = array_like(x, 'x', shape=(None, 4, 4))
    Traceback (most recent call last):
     ...
    ValueError: x is required to have shape (*, 4, 4) but has shape (4, 10, 4)
    """
    arr = np.asarray(obj, dtype=dtype, order=order)
    if maxdim is not None:
        if arr.ndim > maxdim:
            msg = '{0} must have ndim <= {1}'.format(name, maxdim)
            raise ValueError(msg)
    elif ndim is not None:
        if arr.ndim > ndim:
            arr = _right_squeeze(arr, stop_dim=ndim)
        elif arr.ndim < ndim:
            arr = np.reshape(arr, arr.shape + (1,) * (ndim - arr.ndim))
        if arr.ndim != ndim:
            msg = '{0} is required to have ndim {1} but has ndim {2}'
            raise ValueError(msg.format(name, ndim, arr.ndim))
    if shape is not None:
        for actual, req in zip(arr.shape, shape):
            if req is not None and actual != req:
                req_shape = str(shape).replace('None, ', '*, ')
                msg = '{0} is required to have shape {1} but has shape {2}'
                raise ValueError(msg.format(name, req_shape, arr.shape))
    if contiguous:
        arr = np.ascontiguousarray(arr, dtype=dtype)
    return arr


class PandasWrapper(object):
    """
    Wrap array_like using the index from the original input, if pandas

    Parameters
    ----------
    pandas_obj : {Series, DataFrame}
        Object to extract the index from for wrapping

    Notes
    -----
    Raises if ``orig`` is a pandas type but obj and and ``orig`` have
    different numbers of elements in axis 0. Also raises if the ndim of obj
    is larger than 2.
    """

    def __init__(self, pandas_obj):
        self._pandas_obj = pandas_obj
        self._is_pandas = isinstance(pandas_obj, (pd.Series, pd.DataFrame))

    def wrap(self, obj, columns=None, append=None, trim_start=0, trim_end=0):
        """
        Parameters
        ----------
        :param obj:
        :param columns:
        :param append:
        :param trim_start:
        :param trim_end:
        :return:

        Returns
        -------
        wrapper : callable
        Callable that has one required input and one optional:

        * `obj`: array_like to wrap
        * `columns`: (optional) Column names or series name, if obj is 1d
        * `trim_start`: (optional, default 0) number of observations to drop
          from the start of the index, so that the index applied is
          index[trim_start:]
        * `trim_start`: (optional, default 0) number of observations to drop
          from the end of the index , so that the index applied is
          index[:nobs - trim_end]
        """
        obj = np.asarray(obj)
        if not self._is_pandas:
            return obj

        if obj.shape[0] + trim_start + trim_end != self._pandas_obj.shape[0]:
            raise ValueError('obj must have the same number of elements in '
                             'axis 0 as orig')
        index = self._pandas_obj.index
        index = index[trim_start:index.shape[0] - trim_end]
        if obj.ndim == 1:
            if columns is None:
                name = getattr(self._pandas_obj, 'name', None)
            elif isinstance(columns, str):
                name = columns
            else:
                name = columns[0]
            if append is not None:
                name = append if name is None else name + '_' + append

            return pd.Series(obj, name=name, index=index)
        elif obj.ndim == 2:
            if columns is None:
                columns = getattr(self._pandas_obj, 'columns', None)
            if append is not None:
                new = []
                for c in columns:
                    new.append(append if c is None else str(c) + '_' + append)
                columns = new
            return pd.DataFrame(obj, columns=columns, index=index)
        else:
            raise ValueError('Can only wrap 1 or 2-d array_like')
