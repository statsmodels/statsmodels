# -*- coding: utf-8 -*-
"""Tests for finding a positive semi-definite correlation or covariance matrix

Created on Mon May 27 12:07:02 2013

Author: Josef Perktold
"""

import numpy as np
import scipy.sparse as sparse
from numpy.testing import assert_almost_equal, assert_allclose
from statsmodels.stats.correlation_tools import (
    corr_nearest, corr_clipped, cov_nearest,
    _project_correlation_factors, corr_nearest_factor, _spg_optim,
    corr_thresholded, cov_nearest_factor_homog, FactoredPSDMatrix)
import warnings


def norm_f(x, y):
    '''Frobenious norm (squared sum) of difference between two arrays
    '''
    d = ((x - y)**2).sum()
    return np.sqrt(d)

class Holder(object):
    pass

# R library Matrix results
cov1_r = Holder()
#> nc  <- nearPD(pr, conv.tol = 1e-7, keepDiag = TRUE, doDykstra =FALSE, corr=TRUE)
#> cat_items(nc, prefix="cov1_r.")
cov1_r.mat = '''<S4 object of class structure("dpoMatrix", package = "Matrix")>'''
cov1_r.eigenvalues = np.array([
     4.197315628646795, 0.7540460243978023, 0.5077608149667492,
     0.3801267599652769, 0.1607508970775889, 4.197315628646795e-08
    ])
cov1_r.corr = '''TRUE'''
cov1_r.normF = 0.0743805226512533
cov1_r.iterations = 11
cov1_r.rel_tol = 8.288594638441735e-08
cov1_r.converged = '''TRUE'''
#> mkarray2(as.matrix(nc$mat), name="cov1_r.mat")
cov1_r.mat = np.array([
     1, 0.487968018215892, 0.642651880010906, 0.4906386709070835,
     0.6440990530811909, 0.8087111845493985, 0.487968018215892, 1,
     0.5141147294352735, 0.2506688108312097, 0.672351311297074,
     0.725832055882795, 0.642651880010906, 0.5141147294352735, 1,
     0.596827778712154, 0.5821917790519067, 0.7449631633814129,
     0.4906386709070835, 0.2506688108312097, 0.596827778712154, 1,
     0.729882058012399, 0.772150225146826, 0.6440990530811909,
     0.672351311297074, 0.5821917790519067, 0.729882058012399, 1,
     0.813191720191944, 0.8087111845493985, 0.725832055882795,
     0.7449631633814129, 0.772150225146826, 0.813191720191944, 1
    ]).reshape(6,6, order='F')


cov_r = Holder()
#nc  <- nearPD(pr+0.01*diag(6), conv.tol = 1e-7, keepDiag = TRUE, doDykstra =FALSE, corr=FALSE)
#> cat_items(nc, prefix="cov_r.")
#cov_r.mat = '''<S4 object of class structure("dpoMatrix", package = "Matrix")>'''
cov_r.eigenvalues = np.array([
     4.209897516692652, 0.7668341923072066, 0.518956980021938,
     0.390838551407132, 0.1734728460460068, 4.209897516692652e-08
    ])
cov_r.corr = '''FALSE'''
cov_r.normF = 0.0623948693159157
cov_r.iterations = 11
cov_r.rel_tol = 5.83987595937896e-08
cov_r.converged = '''TRUE'''

#> mkarray2(as.matrix(nc$mat), name="cov_r.mat")
cov_r.mat = np.array([
     1.01, 0.486207476951913, 0.6428524769306785, 0.4886092840296514,
     0.645175579158233, 0.811533860074678, 0.486207476951913, 1.01,
     0.514394615153752, 0.2478398278204047, 0.673852495852274,
     0.7297661648968664, 0.6428524769306785, 0.514394615153752, 1.01,
     0.5971503271420517, 0.582018469844712, 0.7445177382760834,
     0.4886092840296514, 0.2478398278204047, 0.5971503271420517, 1.01,
     0.73161232298669, 0.7766852947049376, 0.645175579158233,
     0.673852495852274, 0.582018469844712, 0.73161232298669, 1.01,
     0.8107916469252828, 0.811533860074678, 0.7297661648968664,
     0.7445177382760834, 0.7766852947049376, 0.8107916469252828, 1.01
    ]).reshape(6,6, order='F')

def test_corr_psd():
    # test positive definite matrix is unchanged
    x = np.array([[1, -0.2, -0.9], [-0.2, 1, -0.2], [-0.9, -0.2, 1]])

    y = corr_nearest(x, n_fact=100)
    #print np.max(np.abs(x - y))
    assert_almost_equal(x, y, decimal=14)

    y = corr_clipped(x)
    assert_almost_equal(x, y, decimal=14)

    y = cov_nearest(x, n_fact=100)
    assert_almost_equal(x, y, decimal=14)

    x2 = x + 0.001 * np.eye(3)
    y = cov_nearest(x2, n_fact=100)
    assert_almost_equal(x2, y, decimal=14)


class CheckCorrPSDMixin(object):

    def test_nearest(self):
        x = self.x
        res_r = self.res
        y = corr_nearest(x, threshold=1e-7, n_fact=100)
        #print np.max(np.abs(x - y))
        assert_almost_equal(y, res_r.mat, decimal=3)
        d = norm_f(x, y)
        assert_allclose(d, res_r.normF, rtol=0.0015)
        evals = np.linalg.eigvalsh(y)
        #print 'evals', evals / res_r.eigenvalues[::-1] - 1
        assert_allclose(evals, res_r.eigenvalues[::-1], rtol=0.003, atol=1e-7)
        #print evals[0] / 1e-7 - 1
        assert_allclose(evals[0], 1e-7, rtol=1e-6)


    def test_clipped(self):
        x = self.x
        res_r = self.res
        y = corr_clipped(x, threshold=1e-7)
        #print np.max(np.abs(x - y)), np.max(np.abs((x - y) / y))
        assert_almost_equal(y, res_r.mat, decimal=1)
        d = norm_f(x, y)
        assert_allclose(d, res_r.normF, rtol=0.15)

        evals = np.linalg.eigvalsh(y)
        assert_allclose(evals, res_r.eigenvalues[::-1], rtol=0.1, atol=1e-7)
        assert_allclose(evals[0], 1e-7, rtol=0.02)

    def test_cov_nearest(self):
        x = self.x
        res_r = self.res
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y = cov_nearest(x, method='nearest', threshold=1e-7)
        #print np.max(np.abs(x - y))
        assert_almost_equal(y, res_r.mat, decimal=2)
        d = norm_f(x, y)
        assert_allclose(d, res_r.normF, rtol=0.0015)


class TestCovPSD(object):

    @classmethod
    def setup_class(cls):
        x = np.array([ 1,     0.477, 0.644, 0.478, 0.651, 0.826,
                       0.477, 1,     0.516, 0.233, 0.682, 0.75,
                       0.644, 0.516, 1,     0.599, 0.581, 0.742,
                       0.478, 0.233, 0.599, 1,     0.741, 0.8,
                       0.651, 0.682, 0.581, 0.741, 1,     0.798,
                       0.826, 0.75,  0.742, 0.8,   0.798, 1]).reshape(6,6)
        cls.x = x + 0.01 * np.eye(6)
        cls.res = cov_r

    def test_cov_nearest(self):
        x = self.x
        res_r = self.res
        y = cov_nearest(x, method='nearest')
        #print np.max(np.abs(x - y))
        assert_almost_equal(y, res_r.mat, decimal=3)
        d = norm_f(x, y)
        assert_allclose(d, res_r.normF, rtol=0.001)

        y = cov_nearest(x, method='clipped')
        #print np.max(np.abs(x - y))
        assert_almost_equal(y, res_r.mat, decimal=2)
        d = norm_f(x, y)
        assert_allclose(d, res_r.normF, rtol=0.15)


class TestCorrPSD1(CheckCorrPSDMixin):

    @classmethod
    def setup_class(cls):
        x = np.array([ 1,     0.477, 0.644, 0.478, 0.651, 0.826,
                       0.477, 1,     0.516, 0.233, 0.682, 0.75,
                       0.644, 0.516, 1,     0.599, 0.581, 0.742,
                       0.478, 0.233, 0.599, 1,     0.741, 0.8,
                       0.651, 0.682, 0.581, 0.741, 1,     0.798,
                       0.826, 0.75,  0.742, 0.8,   0.798, 1]).reshape(6,6)
        cls.x = x
        cls.res = cov1_r

def test_corrpsd_threshold():
    x = np.array([[1, -0.9, -0.9], [-0.9, 1, -0.9], [-0.9, -0.9, 1]])

    #print np.linalg.eigvalsh(x)
    for threshold in [0, 1e-15, 1e-10, 1e-6]:

        y = corr_nearest(x, n_fact=100, threshold=threshold)
        evals = np.linalg.eigvalsh(y)
        #print 'evals', evals, threshold
        assert_allclose(evals[0], threshold, rtol=1e-6, atol=1e-15)

        y = corr_clipped(x, threshold=threshold)
        evals = np.linalg.eigvalsh(y)
        #print 'evals', evals, threshold
        assert_allclose(evals[0], threshold, rtol=0.25, atol=1e-15)

        y = cov_nearest(x, method='nearest', n_fact=100, threshold=threshold)
        evals = np.linalg.eigvalsh(y)
        #print 'evals', evals, threshold
        #print evals[0] / threshold - 1
        assert_allclose(evals[0], threshold, rtol=1e-6, atol=1e-15)

        y = cov_nearest(x, n_fact=100, threshold=threshold)
        evals = np.linalg.eigvalsh(y)
        #print 'evals', evals, threshold
        #print evals[0] / threshold - 1
        assert_allclose(evals[0], threshold, rtol=0.25, atol=1e-15)

class Test_Factor(object):

    def test_corr_nearest_factor(self):

        d = 100

        for dm in 1,2:

            # Construct a test matrix with exact factor structure
            X = np.zeros((d,dm), dtype=np.float64)
            x = np.linspace(0, 2*np.pi, d)
            for j in range(dm):
                X[:,j] = np.sin(x*(j+1))
            _project_correlation_factors(X)
            X *= 0.7
            mat = np.dot(X, X.T)
            np.fill_diagonal(mat, 1.)

            # Try to recover the structure
            rslt = corr_nearest_factor(mat, dm)
            C = rslt.corr
            mat1 = C.to_matrix()

            assert(np.abs(mat - mat1).max() < 1e-3)


    # Test that we get the same result if the input is dense or sparse
    def test_corr_nearest_factor_sparse(self):

        d = 100

        for dm in 1,2:

            # Generate a test matrix of factors
            X = np.zeros((d,dm), dtype=np.float64)
            x = np.linspace(0, 2*np.pi, d)
            for j in range(dm):
                X[:,j] = np.sin(x*(j+1))

            # Get the correlation matrix
            _project_correlation_factors(X)
            X *= 0.7
            mat = np.dot(X, X.T)
            np.fill_diagonal(mat, 1)

            # Threshold it
            mat *= (np.abs(mat) >= 0.4)
            smat = sparse.csr_matrix(mat)

            fac_dense = corr_nearest_factor(smat, dm).corr
            mat_dense = fac_dense.to_matrix()

            fac_sparse = corr_nearest_factor(smat, dm).corr
            mat_sparse = fac_sparse.to_matrix()

            assert_allclose(mat_dense, mat_sparse, rtol=0.25,
                            atol=1e-3)


    # Test on a quadratic function.
    def test_spg_optim(self):

        dm = 100

        ind = np.arange(dm)
        indmat = np.abs(ind[:,None] - ind[None,:])
        M = 0.8**indmat

        def obj(x):
            return np.dot(x, np.dot(M, x))

        def grad(x):
            return 2*np.dot(M, x)

        def project(x):
            return x

        x = np.random.normal(size=dm)
        rslt = _spg_optim(obj, grad, x, project)
        xnew = rslt.params
        assert(obj(xnew) < 1e-4)

    def test_decorrelate(self):

        d = 30
        dg = np.linspace(1, 2, d)
        root = np.random.normal(size=(d, 4))
        fac = FactoredPSDMatrix(dg, root)
        mat = fac.to_matrix()
        rmat = np.linalg.cholesky(mat)
        dcr = fac.decorrelate(rmat)
        idm = np.dot(dcr, dcr.T)
        assert_almost_equal(idm, np.eye(d))

        rhs = np.random.normal(size=(d, 5))
        mat2 = np.dot(rhs.T, np.linalg.solve(mat, rhs))
        mat3 = fac.decorrelate(rhs)
        mat3 = np.dot(mat3.T, mat3)
        assert_almost_equal(mat2, mat3)

    def test_logdet(self):

        d = 30
        dg = np.linspace(1, 2, d)
        root = np.random.normal(size=(d, 4))
        fac = FactoredPSDMatrix(dg, root)
        mat = fac.to_matrix()

        _, ld = np.linalg.slogdet(mat)
        ld2 = fac.logdet()

        assert_almost_equal(ld, ld2)

    def test_solve(self):

        d = 30
        dg = np.linspace(1, 2, d)
        root = np.random.normal(size=(d, 2))
        fac = FactoredPSDMatrix(dg, root)
        rhs = np.random.normal(size=(d, 5))
        sr1 = fac.solve(rhs)
        mat = fac.to_matrix()
        sr2 = np.linalg.solve(mat, rhs)
        assert_almost_equal(sr1, sr2)

    def test_cov_nearest_factor_homog(self):

        d = 100

        for dm in 1,2:

            # Construct a test matrix with exact factor structure
            X = np.zeros((d,dm), dtype=np.float64)
            x = np.linspace(0, 2*np.pi, d)
            for j in range(dm):
                X[:,j] = np.sin(x*(j+1))
            mat = np.dot(X, X.T)
            np.fill_diagonal(mat, np.diag(mat) + 3.1)

            # Try to recover the structure
            rslt = cov_nearest_factor_homog(mat, dm)
            mat1 = rslt.to_matrix()

            assert(np.abs(mat - mat1).max() < 1e-4)


    # Check that dense and sparse inputs give the same result
    def test_cov_nearest_factor_homog_sparse(self):

        d = 100

        for dm in 1,2:

            # Construct a test matrix with exact factor structure
            X = np.zeros((d,dm), dtype=np.float64)
            x = np.linspace(0, 2*np.pi, d)
            for j in range(dm):
                X[:,j] = np.sin(x*(j+1))
            mat = np.dot(X, X.T)
            np.fill_diagonal(mat, np.diag(mat) + 3.1)

            # Fit to dense
            rslt = cov_nearest_factor_homog(mat, dm)
            mat1 = rslt.to_matrix()

            # Fit to sparse
            smat = sparse.csr_matrix(mat)
            rslt = cov_nearest_factor_homog(smat, dm)
            mat2 = rslt.to_matrix()

            assert_allclose(mat1, mat2, rtol=0.25, atol=1e-3)

    def test_corr_thresholded(self):

        import datetime

        t1 = datetime.datetime.now()
        X = np.random.normal(size=(2000,10))
        tcor = corr_thresholded(X, 0.2, max_elt=4e6)
        t2 = datetime.datetime.now()
        ss = (t2-t1).seconds

        fcor = np.corrcoef(X)
        fcor *= (np.abs(fcor) >= 0.2)

        assert_allclose(tcor.todense(), fcor, rtol=0.25, atol=1e-3)

