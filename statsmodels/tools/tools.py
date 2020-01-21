"""
Utility functions models code
"""
import numpy as np
import numpy.lib.recfunctions as nprf
import pandas as pd

from statsmodels.compat.python import lzip, lmap

from statsmodels.tools.data import _is_using_pandas, _is_recarray
from statsmodels.tools.validation import array_like


def asstr2(s):
    if isinstance(s, str):
        return s
    elif isinstance(s, bytes):
        return s.decode('latin1')
    else:
        return str(s)


def _make_dictnames(tmp_arr, offset=0):
    """
    Helper function to create a dictionary mapping a column number
    to the name in tmp_arr.
    """
    col_map = {}
    for i, col_name in enumerate(tmp_arr):
        col_map[i + offset] = col_name
    return col_map


def drop_missing(Y, X=None, axis=1):
    """
    Returns views on the arrays Y and X where missing observations are dropped.

    Y : array_like
    X : array_like, optional
    axis : int
        Axis along which to look for missing observations.  Default is 1, ie.,
        observations in rows.

    Returns
    -------
    Y : ndarray
        All Y where the
    X : ndarray

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
# ie., if you still have a string variable in your array you do not
# want to cast it to float
# TODO: add name validator (ie., bad names for datasets.grunfeld)
def categorical(data, col=None, dictnames=False, drop=False):
    """
    Construct a dummy matrix from categorical variables

    Parameters
    ----------
    data : array_like
        A structured array, recarray, array, Series or DataFrame.  This can be
        either a 1d vector of the categorical variable or a 2d array with
        the column specifying the categorical variable specified by the col
        argument.
    col : {str, int, None}
        If data is a DataFrame col must in a column of data. If data is a
        Series, col must be either the name of the Series or None. If data is a
        structured array or a recarray, `col` can be a string that is the name
        of the column that contains the variable.  For all other
        arrays `col` can be an int that is the (zero-based) column index
        number.  `col` can only be None for a 1d array.  The default is None.
    dictnames : bool, optional
        If True, a dictionary mapping the column number to the categorical
        name is returned.  Used to have information about plain arrays.
    drop : bool
        Whether or not keep the categorical variable in the returned matrix.

    Returns
    -------
    dummy_matrix : array_like
        A matrix of dummy (indicator/binary) float variables for the
        categorical data.
    dictnames :  dict[int, str], optional
        Mapping between column numbers and categorical names.

    Notes
    -----
    This returns a dummy variable for *each* distinct variable.  If a
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
    >>> string_var = [string.ascii_lowercase[0:5],
    ...               string.ascii_lowercase[5:10],
    ...               string.ascii_lowercase[10:15],
    ...               string.ascii_lowercase[15:20],
    ...               string.ascii_lowercase[20:25]]
    >>> string_var *= 5
    >>> string_var = np.asarray(sorted(string_var))
    >>> design = sm.tools.categorical(string_var, drop=True)

    Or for a numerical categorical variable

    >>> instr = np.floor(np.arange(10,60, step=2)/10)
    >>> design = sm.tools.categorical(instr, drop=True)

    With a structured array

    >>> num = np.random.randn(25,2)
    >>> struct_ar = np.zeros((25,1),
    ...                      dtype=[('var1', 'f4'),('var2', 'f4'),
    ...                             ('instrument','f4'),('str_instr','a5')])
    >>> struct_ar['var1'] = num[:,0][:,None]
    >>> struct_ar['var2'] = num[:,1][:,None]
    >>> struct_ar['instrument'] = instr[:,None]
    >>> struct_ar['str_instr'] = string_var[:,None]
    >>> design = sm.tools.categorical(struct_ar, col='instrument', drop=True)

    Or

    >>> design2 = sm.tools.categorical(struct_ar, col='str_instr', drop=True)
    """
    # TODO: add a NameValidator function
    if isinstance(col, (list, tuple)):
        if len(col) == 1:
            col = col[0]
        else:
            raise ValueError("Can only convert one column at a time")
    if (not isinstance(data, (pd.DataFrame, pd.Series)) and
            not isinstance(col, (str, int)) and
            col is not None):
        raise TypeError('col must be a str, int or None')

    # Pull out a Series from a DataFrame if provided
    if isinstance(data, pd.DataFrame):
        if col is None:
            raise TypeError('col must be a str or int when using a DataFrame')
        elif col not in data:
            raise ValueError('Column \'{0}\' not found in data'.format(col))
        data = data[col]
        # Set col to None since we not have a Series
        col = None

    if isinstance(data, pd.Series):
        if col is not None and data.name != col:
            raise ValueError('data.name does not match col '
                             '\'{0}\''.format(col))
        data_cat = pd.Categorical(data)
        dummies = pd.get_dummies(data_cat)
        col_map = {i: cat for i, cat in enumerate(data_cat.categories) if
                   cat in dummies}
        if not drop:
            dummies.columns = list(dummies.columns)
            dummies = pd.concat([dummies, data], 1)
        if dictnames:
            return dummies, col_map
        return dummies
    # catch recarrays and structured arrays
    elif data.dtype.names or data.__class__ is np.recarray:
        # deprecated: remove path after 0.12
        import warnings
        from statsmodels.tools.sm_exceptions import recarray_warning
        warnings.warn(recarray_warning, FutureWarning)
        if not col and np.squeeze(data).ndim > 1:
            raise IndexError("col is None and the input array is not 1d")
        if isinstance(col, int):
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

    # Catch array_like for an error
    elif not isinstance(data, np.ndarray):
        raise NotImplementedError("array_like objects are not supported")
    else:
        if isinstance(col, int):
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
    Add a column of ones to an array.

    Parameters
    ----------
    data : array_like
        A column-ordered design matrix.
    prepend : bool
        If true, the constant is in the first column.  Else the constant is
        appended (last column).
    has_constant : str {'raise', 'add', 'skip'}
        Behavior if ``data`` already has a constant. The default will return
        data without adding another constant. If 'raise', will raise an
        error if any column has a constant value. Using 'add' will add a
        column of 1s if a constant column is present.

    Returns
    -------
    array_like
        The original values with a constant (column of ones) as the first or
        last column. Returned value type depends on input type.

    Notes
    -----
    When the input is recarray or a pandas Series or DataFrame, the added
    column's name is 'const'.
    """
    if _is_using_pandas(data, None) or _is_recarray(data):
        if _is_recarray(data):
            # deprecated: remove recarray support after 0.12
            import warnings
            from statsmodels.tools.sm_exceptions import recarray_warning
            warnings.warn(recarray_warning, FutureWarning)
        from statsmodels.tsa.tsatools import add_trend
        return add_trend(data, trend='c', prepend=prepend, has_constant=has_constant)

    # Special case for NumPy
    x = np.asanyarray(data)
    if x.ndim == 1:
        x = x[:, None]
    elif x.ndim > 2:
        raise ValueError('Only implemented for 2-dimensional arrays')

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


def isestimable(c, d):
    """
    True if (Q, P) contrast `c` is estimable for (N, P) design `d`.

    From an Q x P contrast matrix `C` and an N x P design matrix `D`, checks if
    the contrast `C` is estimable by looking at the rank of ``vstack([C,D])``
    and verifying it is the same as the rank of `D`.

    Parameters
    ----------
    c : array_like
        A contrast matrix with shape (Q, P). If 1 dimensional assume shape is
        (1, P).
    d : array_like
        The design matrix, (N, P).

    Returns
    -------
    bool
        True if the contrast `c` is estimable on design `d`.

    Examples
    --------
    >>> d = np.array([[1, 1, 1, 0, 0, 0],
    ...               [0, 0, 0, 1, 1, 1],
    ...               [1, 1, 1, 1, 1, 1]]).T
    >>> isestimable([1, 0, 0], d)
    False
    >>> isestimable([1, -1, 0], d)
    True
    """
    c = array_like(c, 'c', maxdim=2)
    d = array_like(d, 'd', ndim=2)
    c = c[None, :] if c.ndim == 1 else c
    if c.shape[1] != d.shape[1]:
        raise ValueError('Contrast should have %d columns' % d.shape[1])
    new = np.vstack([c, d])
    if np.linalg.matrix_rank(new) != np.linalg.matrix_rank(d):
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
    Reciprocal of an array with entries less than or equal to 0 set to 0.

    Parameters
    ----------
    x : array_like
        The input array.

    Returns
    -------
    ndarray
        The array with 0-filled reciprocals.
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
    Reciprocal of an array with entries less than 0 set to 0.

    Parameters
    ----------
    x : array_like
        The input array.

    Returns
    -------
    ndarray
        The array with 0-filled reciprocals.
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

    Parameters
    ----------
    matrix : ndarray
        The array to clean.

    Returns
    -------
    ndarray
        The cleaned array.
    """
    colsum = np.add.reduce(matrix**2, 0)
    val = [matrix[:, i] for i in np.flatnonzero(colsum)]
    return np.array(np.transpose(val))


def fullrank(x, r=None):
    """
    Return an array whose column span is the same as x.

    Parameters
    ----------
    x : ndarray
        The array to adjust, 2d.
    r : int, optional
        The rank of x. If not provided, determined by `np.linalg.matrix_rank`.

    Returns
    -------
    ndarray
        The array adjusted to have full rank.

    Notes
    -----
    If the rank of x is known it can be specified as r -- no check
    is made to ensure that this really is the rank of x.
    """
    if r is None:
        r = np.linalg.matrix_rank(x)

    v, d, u = np.linalg.svd(x, full_matrices=False)
    order = np.argsort(d)
    order = order[::-1]
    value = []
    for i in range(r):
        value.append(v[:, order[i]])
    return np.asarray(np.transpose(value)).astype(np.float64)


def unsqueeze(data, axis, oldshape):
    """
    Unsqueeze a collapsed array.

    Parameters
    ----------
    data : ndarray
        The data to unsqueeze.
    axis : int
        The axis to unsqueeze.
    oldshape : tuple[int]
        The original shape before the squeeze or reduce operation.

    Returns
    -------
    ndarray
        The unsqueezed array.

    Examples
    --------
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


def nan_dot(A, B):
    """
    Returns np.dot(left_matrix, right_matrix) with the convention that
    nan * 0 = 0 and nan * x = nan if x != 0.

    Parameters
    ----------
    A, B : ndarray
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

    Parameters
    ----------
    *args
        Arguments passed to dict constructor, tuples (key, value).
    **kwargs
        Keyword agument passed to dict constructor, key=value.
    """
    def __init__(self, *args, **kwargs):
        super(Bunch, self).__init__(*args, **kwargs)
        self.__dict__ = self


def _ensure_2d(x, ndarray=False):
    """

    Parameters
    ----------
    x : ndarray, Series, DataFrame or None
        Input to verify dimensions, and to transform as necesary
    ndarray : bool
        Flag indicating whether to always return a NumPy array. Setting False
        will return an pandas DataFrame when the input is a Series or a
        DataFrame.

    Returns
    -------
    out : ndarray, DataFrame or None
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
