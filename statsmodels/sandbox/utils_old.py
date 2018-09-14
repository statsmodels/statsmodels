import numpy as np
import scipy.linalg

from statsmodels.tools.tools import recipr, recipr0, clean0  # noqa:F401
from statsmodels.robust.scale import mad  # noqa:F401
from statsmodels.distributions.empirical_distribution import (  # noqa:F401
    StepFunction, monotone_fn_inverter)

__docformat__ = 'restructuredtext'


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

    V, D, U = np.linalg.svd(X, full_matrices=0)
    order = np.argsort(D)
    order = order[::-1]
    value = []
    for i in range(r):
        value.append(V[:,order[i]])
    return np.asarray(np.transpose(value)).astype(np.float64)


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
