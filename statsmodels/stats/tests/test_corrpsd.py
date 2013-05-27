# -*- coding: utf-8 -*-
"""Tests for findind a positive semi-definite correlation of covariance matrix

Created on Mon May 27 12:07:02 2013

Author: Josef Perktold
"""

import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
from statsmodels.stats.correlation_tools import (
                 corr_nearest, corr_clipped, cov_nearest)

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
