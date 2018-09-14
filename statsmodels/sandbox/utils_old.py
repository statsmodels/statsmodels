import numpy as np
import scipy.linalg

from statsmodels.tools.tools import (  # noqa:F401
    fullrank, recipr, recipr0, clean0)
from statsmodels.distributions.empirical_distribution import (  # noqa:F401
    StepFunction, monotone_fn_inverter, ECDF)
from statsmodels.robust.scale import mad  # noqa:F401

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
