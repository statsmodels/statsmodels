'''
Utility functions models code
'''
from statsmodels.compat.python import reduce, lzip, lmap, asstr2, range, long
import numpy as np
import numpy.lib.recfunctions as nprf
import numpy.linalg as L
from scipy.linalg import svdvals
import pandas as pd

from statsmodels.datasets import webuse
from statsmodels.tools.data import _is_using_pandas, _is_recarray
from statsmodels.compat.numpy import np_matrix_rank


def _make_dictnames(tmp_arr, offset=0):
    """
    Helper function to create a dictionary mapping a column number
    to the name in tmp_arr.
    """
    col_map = {}
    for i, col_name in enumerate(tmp_arr):
        col_map.update({i+offset : col_name})
    return col_map


def drop_missing(Y, X=None, axis=1):
    """
    Returns views on the arrays Y and X where missing observations are dropped.

    Y : array-like
    X : array-like, optional
    axis : int
        Axis along which to look for missing observations.  Default is 1, ie.,
        observations in rows.

    Returns
    -------
    Y : array
        All Y where the
    X : array

    Notes
    -----
    If either Y or X is 1d, it is reshaped to be 2d.
    """
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y[:, None]
    if X is not None:
        X = np.array(X)
        if X.ndim == 1:
            X = X[:, None]
        keepidx = np.logical_and(~np.isnan(Y).any(axis),
                                 ~np.isnan(X).any(axis))
        return Y[keepidx], X[keepidx]
    else:
        keepidx = ~np.isnan(Y).any(axis)
        return Y[keepidx]


# TODO: needs to better preserve dtype and be more flexible
# ie., if you still have a string variable in your array you don't
# want to cast it to float
# TODO: add name validator (ie., bad names for datasets.grunfeld)
def categorical(data, col=None, dictnames=False, drop=False, ):
    '''
    Returns a dummy matrix given an array of categorical variables.

    Parameters
    ----------
    data : array
        A structured array, recarray, or array.  This can be either
        a 1d vector of the categorical variable or a 2d array with
        the column specifying the categorical variable specified by the col
        argument.
    col : 'string', int, or None
        If data is a structured array or a recarray, `col` can be a string
        that is the name of the column that contains the variable.  For all
        arrays `col` can be an int that is the (zero-based) column index
        number.  `col` can only be None for a 1d array.  The default is None.
    dictnames : bool, optional
        If True, a dictionary mapping the column number to the categorical
        name is returned.  Used to have information about plain arrays.
    drop : bool
        Whether or not keep the categorical variable in the returned matrix.

    Returns
    --------
    dummy_matrix, [dictnames, optional]
        A matrix of dummy (indicator/binary) float variables for the
        categorical data.  If dictnames is True, then the dictionary
        is returned as well.

    Notes
    -----
    This returns a dummy variable for EVERY distinct variable.  If a
    a structured or recarray is provided, the names for the new variable is the
    old variable name - underscore - category name.  So if the a variable
    'vote' had answers as 'yes' or 'no' then the returned array would have to
    new variables-- 'vote_yes' and 'vote_no'.  There is currently
    no name checking.

    Examples
    --------
    >>> import numpy as np
    >>> import statsmodels.api as sm

    Univariate examples

    >>> import string
    >>> string_var = [string.ascii_lowercase[0:5], \
                      string.ascii_lowercase[5:10], \
                      string.ascii_lowercase[10:15], \
                      string.ascii_lowercase[15:20],   \
                      string.ascii_lowercase[20:25]]
    >>> string_var *= 5
    >>> string_var = np.asarray(sorted(string_var))
    >>> design = sm.tools.categorical(string_var, drop=True)

    Or for a numerical categorical variable

    >>> instr = np.floor(np.arange(10,60, step=2)/10)
    >>> design = sm.tools.categorical(instr, drop=True)

    With a structured array

    >>> num = np.random.randn(25,2)
    >>> struct_ar = np.zeros((25,1), dtype=[('var1', 'f4'),('var2', 'f4'),  \
                    ('instrument','f4'),('str_instr','a5')])
    >>> struct_ar['var1'] = num[:,0][:,None]
    >>> struct_ar['var2'] = num[:,1][:,None]
    >>> struct_ar['instrument'] = instr[:,None]
    >>> struct_ar['str_instr'] = string_var[:,None]
    >>> design = sm.tools.categorical(struct_ar, col='instrument', drop=True)

    Or

    >>> design2 = sm.tools.categorical(struct_ar, col='str_instr', drop=True)
    '''
    if isinstance(col, (list, tuple)):
        try:
            assert len(col) == 1
            col = col[0]
        except:
            raise ValueError("Can only convert one column at a time")

    # TODO: add a NameValidator function
    # catch recarrays and structured arrays
    if data.dtype.names or data.__class__ is np.recarray:
        if not col and np.squeeze(data).ndim > 1:
            raise IndexError("col is None and the input array is not 1d")
        if isinstance(col, (int, long)):
            col = data.dtype.names[col]
        if col is None and data.dtype.names and len(data.dtype.names) == 1:
            col = data.dtype.names[0]

        tmp_arr = np.unique(data[col])

        # if the cols are shape (#,) vs (#,1) need to add an axis and flip
        _swap = True
        if data[col].ndim == 1:
            tmp_arr = tmp_arr[:, None]
            _swap = False
        tmp_dummy = (tmp_arr == data[col]).astype(float)
        if _swap:
            tmp_dummy = np.squeeze(tmp_dummy).swapaxes(1, 0)

        if not tmp_arr.dtype.names:  # how do we get to this code path?
            tmp_arr = [asstr2(item) for item in np.squeeze(tmp_arr)]
        elif tmp_arr.dtype.names:
            tmp_arr = [asstr2(item) for item in np.squeeze(tmp_arr.tolist())]

        # prepend the varname and underscore, if col is numeric attribute
        # lookup is lost for recarrays...
        if col is None:
            try:
                col = data.dtype.names[0]
            except:
                col = 'var'
        # TODO: the above needs to be made robust because there could be many
        # var_yes, var_no varaibles for instance.
        tmp_arr = [col + '_' + item for item in tmp_arr]
        # TODO: test this for rec and structured arrays!!!

        if drop is True:
            if len(data.dtype) <= 1:
                if tmp_dummy.shape[0] < tmp_dummy.shape[1]:
                    tmp_dummy = np.squeeze(tmp_dummy).swapaxes(1, 0)
                dt = lzip(tmp_arr, [tmp_dummy.dtype.str]*len(tmp_arr))
                # preserve array type
                return np.array(lmap(tuple, tmp_dummy.tolist()),
                                dtype=dt).view(type(data))

            data = nprf.drop_fields(data, col, usemask=False,
                                    asrecarray=type(data) is np.recarray)
        data = nprf.append_fields(data, tmp_arr, data=tmp_dummy,
                                  usemask=False,
                                  asrecarray=type(data) is np.recarray)
        return data

    # handle ndarrays and catch array-like for an error
    elif data.__class__ is np.ndarray or not isinstance(data, np.ndarray):
        if not isinstance(data, np.ndarray):
            raise NotImplementedError("Array-like objects are not supported")

        if isinstance(col, (int, long)):
            offset = data.shape[1]          # need error catching here?
            tmp_arr = np.unique(data[:, col])
            tmp_dummy = (tmp_arr[:, np.newaxis] == data[:, col]).astype(float)
            tmp_dummy = tmp_dummy.swapaxes(1, 0)
            if drop is True:
                offset -= 1
                data = np.delete(data, col, axis=1).astype(float)
            data = np.column_stack((data, tmp_dummy))
            if dictnames is True:
                col_map = _make_dictnames(tmp_arr, offset)
                return data, col_map
            return data
        elif col is None and np.squeeze(data).ndim == 1:
            tmp_arr = np.unique(data)
            tmp_dummy = (tmp_arr[:, None] == data).astype(float)
            tmp_dummy = tmp_dummy.swapaxes(1, 0)
            if drop is True:
                if dictnames is True:
                    col_map = _make_dictnames(tmp_arr)
                    return tmp_dummy, col_map
                return tmp_dummy
            else:
                data = np.column_stack((data, tmp_dummy))
                if dictnames is True:
                    col_map = _make_dictnames(tmp_arr, offset=1)
                    return data, col_map
                return data
        else:
            raise IndexError("The index %s is not understood" % col)


# TODO: add an axis argument to this for sysreg
def add_constant(data, prepend=True, has_constant='skip'):
    """
    Adds a column of ones to an array

    Parameters
    ----------
    data : array-like
        ``data`` is the column-ordered design matrix
    prepend : bool
        If true, the constant is in the first column.  Else the constant is
        appended (last column).
    has_constant : str {'raise', 'add', 'skip'}
        Behavior if ``data`` already has a constant. The default will return
        data without adding another constant. If 'raise', will raise an
        error if a constant is present. Using 'add' will duplicate the
        constant, if one is present.

    Returns
    -------
    data : array, recarray or DataFrame
        The original values with a constant (column of ones) as the first or
        last column. Returned value depends on input type.

    Notes
    -----
    When the input is recarray or a pandas Series or DataFrame, the added
    column's name is 'const'.
    """
    if _is_using_pandas(data, None) or _is_recarray(data):
        from statsmodels.tsa.tsatools import add_trend
        return add_trend(data, trend='c', prepend=prepend, has_constant=has_constant)

    # Special case for NumPy
    x = np.asanyarray(data)
    if x.ndim == 1:
        x = x[:,None]
    elif x.ndim > 2:
        raise ValueError('Only implementd 2-dimensional arrays')

    is_nonzero_const = np.ptp(x, axis=0) == 0
    is_nonzero_const &= np.all(x != 0.0, axis=0)
    if is_nonzero_const.any():
        if has_constant == 'skip':
            return x
        elif has_constant == 'raise':
            raise ValueError("data already contains a constant")

    x = [np.ones(x.shape[0]), x]
    x = x if prepend else x[::-1]
    return np.column_stack(x)


def isestimable(C, D):
    """ True if (Q, P) contrast `C` is estimable for (N, P) design `D`

    From an Q x P contrast matrix `C` and an N x P design matrix `D`, checks if
    the contrast `C` is estimable by looking at the rank of ``vstack([C,D])``
    and verifying it is the same as the rank of `D`.

    Parameters
    ----------
    C : (Q, P) array-like
        contrast matrix. If `C` has is 1 dimensional assume shape (1, P)
    D: (N, P) array-like
        design matrix

    Returns
    -------
    tf : bool
        True if the contrast `C` is estimable on design `D`

    Examples
    --------
    >>> D = np.array([[1, 1, 1, 0, 0, 0],
    ...               [0, 0, 0, 1, 1, 1],
    ...               [1, 1, 1, 1, 1, 1]]).T
    >>> isestimable([1, 0, 0], D)
    False
    >>> isestimable([1, -1, 0], D)
    True
    """
    C = np.asarray(C)
    D = np.asarray(D)
    if C.ndim == 1:
        C = C[None, :]
    if C.shape[1] != D.shape[1]:
        raise ValueError('Contrast should have %d columns' % D.shape[1])
    new = np.vstack([C, D])
    if np_matrix_rank(new) != np_matrix_rank(D):
        return False
    return True


def pinv_extended(X, rcond=1e-15):
    """
    Return the pinv of an array X as well as the singular values
    used in computation.

    Code adapted from numpy.
    """
    X = np.asarray(X)
    X = X.conjugate()
    u, s, vt = np.linalg.svd(X, 0)
    s_orig = np.copy(s)
    m = u.shape[0]
    n = vt.shape[1]
    cutoff = rcond * np.maximum.reduce(s)
    for i in range(min(n, m)):
        if s[i] > cutoff:
            s[i] = 1./s[i]
        else:
            s[i] = 0.
    res = np.dot(np.transpose(vt), np.multiply(s[:, np.core.newaxis],
                                               np.transpose(u)))
    return res, s_orig


def recipr(x):
    """
    Return the reciprocal of an array, setting all entries less than or
    equal to 0 to 0. Therefore, it presumes that X should be positive in
    general.
    """

    x = np.asarray(x)
    out = np.zeros_like(x, dtype=np.float64)
    nans = np.isnan(x.flat)
    pos = ~nans
    pos[pos] = pos[pos] & (x.flat[pos] > 0)
    out.flat[pos] = 1.0 / x.flat[pos]
    out.flat[nans] = np.nan
    return out


def recipr0(x):
    """
    Return the reciprocal of an array, setting all entries equal to 0
    as 0. It does not assume that X should be positive in
    general.
    """
    x = np.asarray(x)
    out = np.zeros_like(x, dtype=np.float64)
    nans = np.isnan(x.flat)
    non_zero = ~nans
    non_zero[non_zero] = non_zero[non_zero] & (x.flat[non_zero] != 0)
    out.flat[non_zero] = 1.0 / x.flat[non_zero]
    out.flat[nans] = np.nan
    return out


def clean0(matrix):
    """
    Erase columns of zeros: can save some time in pseudoinverse.
    """
    colsum = np.add.reduce(matrix**2, 0)
    val = [matrix[:, i] for i in np.flatnonzero(colsum)]
    return np.array(np.transpose(val))


def fullrank(X, r=None):
    """
    Return a matrix whose column span is the same as X.

    If the rank of X is known it can be specified as r -- no check
    is made to ensure that this really is the rank of X.

    """

    if r is None:
        r = np_matrix_rank(X)

    V, D, U = L.svd(X, full_matrices=0)
    order = np.argsort(D)
    order = order[::-1]
    value = []
    for i in range(r):
        value.append(V[:, order[i]])
    return np.asarray(np.transpose(value)).astype(np.float64)


def unsqueeze(data, axis, oldshape):
    """
    Unsqueeze a collapsed array

    >>> from numpy import mean
    >>> from numpy.random import standard_normal
    >>> x = standard_normal((3,4,5))
    >>> m = mean(x, axis=1)
    >>> m.shape
    (3, 5)
    >>> m = unsqueeze(m, 1, x.shape)
    >>> m.shape
    (3, 1, 5)
    >>>
    """
    newshape = list(oldshape)
    newshape[axis] = 1
    return data.reshape(newshape)


def chain_dot(*arrs):
    """
    Returns the dot product of the given matrices.

    Parameters
    ----------
    arrs: argument list of ndarray

    Returns
    -------
    Dot product of all arguments.

    Examples
    --------
    >>> import numpy as np
    >>> from statsmodels.tools import chain_dot
    >>> A = np.arange(1,13).reshape(3,4)
    >>> B = np.arange(3,15).reshape(4,3)
    >>> C = np.arange(5,8).reshape(3,1)
    >>> chain_dot(A,B,C)
    array([[1820],
       [4300],
       [6780]])
    """
    return reduce(lambda x, y: np.dot(y, x), arrs[::-1])


def nan_dot(A, B):
    """
    Returns np.dot(left_matrix, right_matrix) with the convention that
    nan * 0 = 0 and nan * x = nan if x != 0.

    Parameters
    ----------
    A, B : np.ndarrays
    """
    # Find out who should be nan due to nan * nonzero
    should_be_nan_1 = np.dot(np.isnan(A), (B != 0))
    should_be_nan_2 = np.dot((A != 0), np.isnan(B))
    should_be_nan = should_be_nan_1 + should_be_nan_2

    # Multiply after setting all nan to 0
    # This is what happens if there were no nan * nonzero conflicts
    C = np.dot(np.nan_to_num(A), np.nan_to_num(B))

    C[should_be_nan] = np.nan

    return C


def maybe_unwrap_results(results):
    """
    Gets raw results back from wrapped results.

    Can be used in plotting functions or other post-estimation type
    routines.
    """
    return getattr(results, '_results', results)

class Bunch(dict):
    """
    Returns a dict-like object with keys accessible via attribute lookup.
    """
    def __init__(self, *args, **kwargs):
        super(Bunch, self).__init__(*args, **kwargs)
        self.__dict__ = self


def _ensure_2d(x, ndarray=False):
    """

    Parameters
    ----------
    x : array, Series, DataFrame or None
        Input to verify dimensions, and to transform as necesary
    ndarray : bool
        Flag indicating whether to always return a NumPy array. Setting False
        will return an pandas DataFrame when the input is a Series or a
        DataFrame.

    Returns
    -------
    out : array, DataFrame or None
        array or DataFrame with 2 dimensiona.  One dimensional arrays are
        returned as nobs by 1. None is returned if x is None.
    names : list of str or None
        list containing variables names when the input is a pandas datatype.
        Returns None if the input is an ndarray.

    Notes
    -----
    Accepts None for simplicity
    """
    if x is None:
        return x
    is_pandas = _is_using_pandas(x, None)
    if x.ndim == 2:
        if is_pandas:
            return x, x.columns
        else:
            return x, None
    elif x.ndim > 2:
        raise ValueError('x mst be 1 or 2-dimensional.')

    name = x.name if is_pandas else None
    if ndarray:
        return np.asarray(x)[:, None], name
    else:
        return pd.DataFrame(x), name
