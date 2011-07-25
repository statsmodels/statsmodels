"""
Empirical CDF Functions
"""
import numpy as np

class StepFunction:
    """
    A basic step function.

    Values at the ends are handled in the simplest way possible:
    everything to the left of x[0] is set to ival; everything
    to the right of x[-1] is set to y[-1].

    Parameters
    ----------
    x : array-like
    y : array-like
    ival : float
        ival is the value given to the values to the left of x[0]. Default
        is 0.
    sorted : bool
        Default is False.

    Examples
    --------
    >>> import numpy as np
    >>> from scikits.statsmodels.tools import StepFunction
    >>>
    >>> x = np.arange(20)
    >>> y = np.arange(20)
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
            msg = "x and y do not have the same shape"
            raise ValueError(msg)
        if len(_x.shape) != 1:
            msg = 'x and y must be 1-dimensional'
            raise ValueError(msg)

        self.x = np.r_[-np.inf, _x]
        self.y = np.r_[ival, _y]

        if not sorted:
            asort = np.argsort(self.x)
            self.x = np.take(self.x, asort, 0)
            self.y = np.take(self.y, asort, 0)
        self.n = self.x.shape[0]

    def __call__(self, time):

        tind = np.searchsorted(self.x, time) - 1
        _shape = tind.shape
        return self.y[tind]

def ecdf(x):
    """
    Return the Empirical CDF of an array as a step function.

    Parameters
    ----------
    x : array-like
        Observations

    Returns
    -------
    Empirical CDF as a step function.
    """
    x = np.array(x, copy=True)
    x.sort()
    nobs = len(x)
    y = np.linspace(1./nobs,1,nobs)
    return StepFunction(x, y)

def monotone_fn_inverter(fn, x, vectorized=True, **keywords):
    """
    Given a monotone function x (no checking is done to verify monotonicity)
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

    return interp1d(y[a], x[a])

