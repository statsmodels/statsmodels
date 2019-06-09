import numpy as np
import pandas as pd
from statsmodels.regression.dimred import (SlicedInverseReg,
     SAVE, PHD, CovReduce)
from numpy.testing import (assert_equal, assert_allclose)
from statsmodels.tools.numdiff import approx_fprime


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


def test_covreduce():

    np.random.seed(34324)

    p = 4
    endog = []
    exog = []
    for k in range(3):
        c = np.eye(p)
        x = np.random.normal(size=(2, 2))
        # The differences between the covariance matrices
        # are all in the first 2 rows/columns.
        c[0:2, 0:2] = np.dot(x.T, x)

        cr = np.linalg.cholesky(c)
        m = 1000*k + 50*k
        x = np.random.normal(size=(m, p))
        x = np.dot(x, cr.T)
        exog.append(x)
        endog.append(k * np.ones(m))

    endog = np.concatenate(endog)
    exog = np.concatenate(exog, axis=0)

    for dim in 1, 2, 3:

        cr = CovReduce(endog, exog, dim)

        pt = np.random.normal(size=(p, dim))
        pt, _, _ = np.linalg.svd(pt, 0)
        gn = approx_fprime(pt.ravel(), cr.loglike, 1e-7)
        g = cr.score(pt.ravel())

        assert_allclose(g, gn, 1e-5, 1e-5)

        rslt = cr.fit()
        proj = rslt.params
        assert_equal(proj.shape[0], p)
        assert_equal(proj.shape[1], dim)
        assert_allclose(np.dot(proj.T, proj), np.eye(dim), 1e-8, 1e-8)

        if dim == 2:
            # Here we know the approximate truth
            projt = np.zeros((p, 2))
            projt[0:2, 0:2] = np.eye(2)
            assert_allclose(np.trace(np.dot(proj.T, projt)), 2,
                            rtol=1e-3, atol=1e-3)
