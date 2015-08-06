import numpy as np
import numpy.linalg as L
import scipy.interpolate
import scipy.linalg

__docformat__ = 'restructuredtext'

def recipr(X):
    """
    Return the reciprocal of an array, setting all entries less than or
    equal to 0 to 0. Therefore, it presumes that X should be positive in
    general.
    """
    x = np.maximum(np.asarray(X).astype(np.float64), 0)
    return np.greater(x, 0.) / (x + np.less_equal(x, 0.))

def mad(a, c=0.6745, axis=0):
    """
    Median Absolute Deviation:

    median(abs(a - median(a))) / c

    """

    _shape = a.shape
    a.shape = np.product(a.shape,axis=0)
    m = np.median(np.fabs(a - np.median(a))) / c
    a.shape = _shape
    return m

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
        D = scipy.linalg.svdvals(X)
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

class StepFunction(object):
    """
    A basic step function: values at the ends are handled in the simplest
    way possible: everything to the left of x[0] is set to ival; everything
    to the right of x[-1] is set to y[-1].

    Examples
    --------
    >>> from numpy import arange
    >>> from statsmodels.sandbox.utils_old import StepFunction
    >>>
    >>> x = arange(20)
    >>> y = arange(20)
    >>> f = StepFunction(x, y)
    >>>
    >>> print f(3.2)
    3.0
    >>> print f([[3.2,4.5],[24,-3.1]])
    [[  3.   4.]
     [ 19.   0.]]
    """

    def __init__(self, x, y, ival=0., sorted=False):

        _x = np.asarray(x)
        _y = np.asarray(y)

        if _x.shape != _y.shape:
            raise ValueError('in StepFunction: x and y do not have the same shape')
        if len(_x.shape) != 1:
            raise ValueError('in StepFunction: x and y must be 1-dimensional')

        self.x = np.hstack([[-np.inf], _x])
        self.y = np.hstack([[ival], _y])

        if not sorted:
            asort = np.argsort(self.x)
            self.x = np.take(self.x, asort, 0)
            self.y = np.take(self.y, asort, 0)
        self.n = self.x.shape[0]

    def __call__(self, time):

        tind = np.searchsorted(self.x, time) - 1
        _shape = tind.shape
        return self.y[tind]

def ECDF(values):
    """
    Return the ECDF of an array as a step function.
    """
    x = np.array(values, copy=True)
    x.sort()
    x.shape = np.product(x.shape,axis=0)
    n = x.shape[0]
    y = (np.arange(n) + 1.) / n
    return StepFunction(x, y)

def monotone_fn_inverter(fn, x, vectorized=True, **keywords):
    """
    Given a monotone function fn (no checking is done to verify monotonicity)
    and a set of x values, return an linearly interpolated approximation
    to its inverse from its values on x.
    """

    if vectorized:
        y = fn(x, **keywords)
    else:
        y = []
        for _x in x:
            y.append(fn(_x, **keywords))
        y = np.array(y)

    a = np.argsort(y)

    return scipy.interpolate.interp1d(y[a], x[a])
