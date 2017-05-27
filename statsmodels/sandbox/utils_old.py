import numpy as np
import numpy.linalg as L
import scipy.interpolate
import scipy.linalg

from statsmodels.tools.tools import recipr, recipr0, clean0, fullrank
from statsmodels.distributions.empirical_distribution import StepFunction, monotone_fn_inverter

__docformat__ = 'restructuredtext'



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



# Note 2017-05-26: This behaves slightly differently than
# statsmodels.distributions.empirical_distribution.ECDF
# For reference, the example in the docstring from that class is roughly:
# >>> ecdf = empirical_distribution.ECDF([3, 3, 1, 4])
# >>> ecdf([3, 55, 0.5, 1.5])
# array([ 0.75,  1.  ,  0.  ,  0.25])
#
# The same values here produce:
# >>> ecdf = utils_old.ECDF([3, 3, 1, 4])
# >>> ecdf([3, 55, 0.5, 1.5])
# array([ 0.25,  1.  ,  0.  ,  0.25])
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

