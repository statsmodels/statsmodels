import numpy as np
import pandas as pd
from statsmodels.regression.dimred import (SlicedInverseReg, SAVE, PHD)
from numpy.testing import (assert_equal)


def test_poisson():

    np.random.seed(43242)

    # Generate a non-orthogonal design matrix
    xmat = np.random.normal(size=(500, 5))
    xmat[:, 1] = 0.5*xmat[:, 0] + np.sqrt(1 - 0.5**2) * xmat[:, 1]
    xmat[:, 3] = 0.5*xmat[:, 2] + np.sqrt(1 - 0.5**2) * xmat[:, 3]

    b = np.r_[0, 1, -1, 0, 0.5]
    lpr = np.dot(xmat, b)
    ev = np.exp(lpr)
    y = np.random.poisson(ev)

    for method in range(6):

        if method == 0:
            model = SlicedInverseReg(y, xmat)
            rslt = model.fit()
        elif method == 1:
            model = SAVE(y, xmat)
            rslt = model.fit(slice_n=100)
        elif method == 2:
            model = SAVE(y, xmat, bc=True)
            rslt = model.fit(slice_n=100)
        elif method == 3:
            df = pd.DataFrame({"y": y,
                               "x0": xmat[:, 0],
                               "x1": xmat[:, 1],
                               "x2": xmat[:, 2],
                               "x3": xmat[:, 3],
                               "x4": xmat[:, 4]})
            model = SlicedInverseReg.from_formula(
                        "y ~ 0 + x0 + x1 + x2 + x3 + x4", data=df)
            rslt = model.fit()
        elif method == 4:
            model = PHD(y, xmat)
            rslt = model.fit()
        elif method == 5:
            model = PHD(y, xmat)
            rslt = model.fit(resid=True)

        # Check for concentration in one direction (this is
        # a single index model)
        assert_equal(np.abs(rslt.eigs[0] / rslt.eigs[1]) > 5, True)

        # Check that the estimated direction aligns with the true
        # direction
        params = np.asarray(rslt.params)
        q = np.dot(params[:, 0], b)
        q /= np.sqrt(np.sum(params[:, 0]**2))
        q /= np.sqrt(np.sum(b**2))
        assert_equal(np.abs(q) > 0.95, True)
