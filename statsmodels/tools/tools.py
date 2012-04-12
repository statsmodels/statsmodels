'''
Utility functions models code
'''

import numpy as np
import numpy.lib.recfunctions as nprf
import numpy.linalg as L
from scipy.interpolate import interp1d
from scipy.linalg import svdvals
from statsmodels.distributions import (ECDF, monotone_fn_inverter,
                                               StepFunction)
from statsmodels.tools.data import _is_using_pandas
from pandas import DataFrame

def _make_dictnames(tmp_arr, offset=0):
    """
    Helper function to create a dictionary mapping a column number
    to the name in tmp_arr.
    """
    col_map = {}
    for i,col_name in enumerate(tmp_arr):
        col_map.update({i+offset : col_name})
    return col_map

def drop_missing(Y,X=None, axis=1):
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
        Y = Y[:,None]
    if X is not None:
        X = np.array(X)
        if X.ndim == 1:
            X = X[:,None]
        keepidx = np.logical_and(~np.isnan(Y).any(axis),~np.isnan(X).any(axis))
        return Y[keepidx], X[keepidx]
    else:
        keepidx = ~np.isnan(Y).any(axis)
        return Y[keepidx]

#TODO: needs to better preserve dtype and be more flexible
# ie., if you still have a string variable in your array you don't
# want to cast it to float
#TODO: add name validator (ie., bad names for datasets.grunfeld)
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
    >>> string_var = [string.lowercase[0:5], string.lowercase[5:10],   \
                string.lowercase[10:15], string.lowercase[15:20],   \
                string.lowercase[20:25]]
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

#TODO: add a NameValidator function
    # catch recarrays and structured arrays
    if data.dtype.names or data.__class__ is np.recarray:
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
            tmp_arr = tmp_arr[:,None]
            _swap = False
        tmp_dummy = (tmp_arr==data[col]).astype(float)
        if _swap:
            tmp_dummy = np.squeeze(tmp_dummy).swapaxes(1,0)

        if not tmp_arr.dtype.names:
            tmp_arr = np.squeeze(tmp_arr).astype('str').tolist()
        elif tmp_arr.dtype.names:
            tmp_arr = np.squeeze(tmp_arr.tolist()).astype('str').tolist()

# prepend the varname and underscore, if col is numeric attribute lookup
# is lost for recarrays...
        if col is None:
            try:
                col = data.dtype.names[0]
            except:
                col = 'var'
#TODO: the above needs to be made robust because there could be many
# var_yes, var_no varaibles for instance.
        tmp_arr = [col + '_'+ item for item in tmp_arr]
#TODO: test this for rec and structured arrays!!!

        if drop is True:
            # if len(data.dtype) is 1 then we have a 1 column array
#            if len(data.dtype) == 1:
            if len(data.dtype) <= 1:
                if tmp_dummy.shape[0] < tmp_dummy.shape[1]:
                    tmp_dummy = np.squeeze(tmp_dummy).swapaxes(1,0)
                dt = zip(tmp_arr, [tmp_dummy.dtype.str]*len(tmp_arr))
                # preserve array type
                return np.array(map(tuple, tmp_dummy.tolist()),
                        dtype=dt).view(type(data))

            data=nprf.drop_fields(data, col, usemask=False,
                            asrecarray=type(data) is np.recarray)
        data=nprf.append_fields(data, tmp_arr, data=tmp_dummy,
            usemask=False, asrecarray=type(data) is np.recarray)
        return data

    # handle ndarrays and catch array-like for an error
    elif data.__class__ is np.ndarray or not isinstance(data,np.ndarray):
        if not isinstance(data, np.ndarray):
            raise NotImplementedError("Array-like objects are not supported")

        if isinstance(col, int):
            offset = data.shape[1]          # need error catching here?
            tmp_arr = np.unique(data[:,col])
            tmp_dummy = (tmp_arr[:,np.newaxis]==data[:,col]).astype(float)
            tmp_dummy = tmp_dummy.swapaxes(1,0)
            if drop is True:
                offset -= 1
                data = np.delete(data, col, axis=1).astype(float)
            data = np.column_stack((data,tmp_dummy))
            if dictnames is True:
                col_map = _make_dictnames(tmp_arr, offset)
                return data, col_map
            return data
        elif col is None and np.squeeze(data).ndim == 1:
            tmp_arr = np.unique(data)
            tmp_dummy = (tmp_arr[:,None]==data).astype(float)
            tmp_dummy = tmp_dummy.swapaxes(1,0)
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

def _series_add_constant(data, prepend):
    const = np.ones_like(data)
    const.name = 'const'
    if not prepend:
        results = DataFrame([data, const]).T
        results.columns = [data.name, 'const']
    else:
        results = DataFrame([const, data]).T
        results.columns = ['const', data.name]
    return results

def _dataframe_add_constant(data, prepend):
    # check for const.
    if np.any(data.var(0) == 1):
        return data
    if prepend:
        data.insert(0, 'const', 1)
    else:
        data['const'] = 1
    return data

def _pandas_add_constant(data, prepend):
    from pandas import Series
    if isinstance(data, Series):
        return _series_add_constant(data, prepend)
    else:
        return _dataframe_add_constant(data, prepend)


#TODO: add an axis argument to this for sysreg
def add_constant(data, prepend=False):
    '''
    This appends a column of ones to an array if prepend==False.

    For ndarrays and pandas.DataFrames, checks to make sure a constant is not
    already included. If there is at least one column of ones then the
    original object is returned.  Does not check for a constant if a structured
    or recarray is
    given.

    Parameters
    ----------
    data : array-like
        `data` is the column-ordered design matrix
    prepend : bool
        True and the constant is prepended rather than appended.

    Returns
    -------
    data : array
        The original array with a constant (column of ones) as the first or
        last column.

    Notes
    -----

    .. WARNING::
       The default of prepend will be changed to True in the next release of
       statsmodels. We recommend to use an explicit prepend in any permanent
       code.
    '''
    if not prepend:
        import inspect
        frame = inspect.currentframe().f_back
        info = inspect.getframeinfo(frame)
        try:
            to_warn = 'prepend' not in '\n'.join(info.code_context)
        except: # python 2.5 compatibility
            to_warn = 'prepend' not in '\n'.join(info[3])
        if to_warn:
            import warnings
            warnings.warn("The default of `prepend` will be changed to True "
                          "in 0.5.0, use explicit prepend",
                          FutureWarning)

    if _is_using_pandas(data, None):
        # work on a copy
        return _pandas_add_constant(data.copy(), prepend)
    else:
        data = np.asarray(data)
    if not data.dtype.names:
        var0 = data.var(0) == 0
        if np.any(var0):
            return data
        data = np.column_stack((data, np.ones((data.shape[0], 1))))
        if prepend:
            return np.roll(data, 1, 1)
    else:
        return_rec = data.__class__ is np.recarray
        if prepend:
            ones = np.ones((data.shape[0], 1), dtype=[('const', float)])
            data = nprf.append_fields(ones, data.dtype.names, [data[i] for
                i in data.dtype.names], usemask=False, asrecarray=return_rec)
        else:
            data = nprf.append_fields(data, 'const', np.ones(data.shape[0]),
                    usemask=False, asrecarray = return_rec)
    return data

def isestimable(C, D):
    """
    From an q x p contrast matrix C and an n x p design matrix D, checks
    if the contrast C is estimable by looking at the rank of vstack([C,D]) and
    verifying it is the same as the rank of D.
    """
    if C.ndim == 1:
        C.shape = (C.shape[0], 1)
    new = np.vstack([C, D])
    if rank(new) != rank(D):
        return False
    return True

def recipr(X):
    """
    Return the reciprocal of an array, setting all entries less than or
    equal to 0 to 0. Therefore, it presumes that X should be positive in
    general.
    """
    x = np.maximum(np.asarray(X).astype(np.float64), 0)
    return np.greater(x, 0.) / (x + np.less_equal(x, 0.))

def recipr0(X):
    """
    Return the reciprocal of an array, setting all entries equal to 0
    as 0. It does not assume that X should be positive in
    general.
    """
    test = np.equal(np.asarray(X), 0)
    return np.where(test, 0, 1. / X)

def clean0(matrix):
    """
    Erase columns of zeros: can save some time in pseudoinverse.
    """
    colsum = np.add.reduce(matrix**2, 0)
    val = [matrix[:,i] for i in np.flatnonzero(colsum)]
    return np.array(np.transpose(val))

def rank(X, cond=1.0e-12):
    """
    Return the rank of a matrix X based on its generalized inverse,
    not the SVD.
    """
    X = np.asarray(X)
    if len(X.shape) == 2:
        D = svdvals(X)
        return int(np.add.reduce(np.greater(D / D.max(), cond).astype(np.int32)))
    else:
        return int(not np.alltrue(np.equal(X, 0.)))

def fullrank(X, r=None):
    """
    Return a matrix whose column span is the same as X.

    If the rank of X is known it can be specified as r -- no check
    is made to ensure that this really is the rank of X.

    """

    if r is None:
        r = rank(X)

    V, D, U = L.svd(X, full_matrices=0)
    order = np.argsort(D)
    order = order[::-1]
    value = []
    for i in range(r):
        value.append(V[:,order[i]])
    return np.asarray(np.transpose(value)).astype(np.float64)

StepFunction = np.deprecate(StepFunction,
                old_name = 'statsmodels.tools.tools.StepFunction',
                new_name = 'statsmodels.distributions.StepFunction')
monotone_fn_inverter = np.deprecate(monotone_fn_inverter,
                old_name = 'statsmodels.tools.tools.monotone_fn_inverter',
                new_name = 'statsmodels.distributions.monotone_fn_inverter')
ECDF = np.deprecate(ECDF,
                old_name = 'statsmodels.tools.tools.ECDF',
                new_name = 'statsmodels.distributions.ECDF')


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

    Example
    -------
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

def webuse(data, baseurl='http://www.stata-press.com/data/r11/'):
    """
    Parameters
    ----------
    data : str
        Name of dataset to fetch.

    Returns
    -------
    dta : Record Array
        A record array containing the Stata dataset.

    Examples
    --------
    >>> dta = webuse('auto')

    Notes
    -----
    Make sure baseurl has trailing forward slash. Doesn't do any
    error checking in response URLs.
    """
    # lazy imports
    import pandas
    from statsmodels.iolib import genfromdta
    from urllib2 import urlopen
    from urlparse import urljoin
    from StringIO import StringIO

    url = urljoin(baseurl, data+'.dta')
    dta = urlopen(url)
    dta = StringIO(dta.read()) # make it truly file-like
    return genfromdta(dta)
