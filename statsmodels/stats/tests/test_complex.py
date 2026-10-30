"""
Created on Nov. 28, 2023 11:58:08 a.m.

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
from numpy.testing import assert_allclose

from statsmodels.stats._complex import (
    RandVarComplex,
    cov_rvec,
    cov_from_rvec,
    cov_ext,
    )

class CheckComplex():

    def test_basic(self):
        z = self.zrvs
        zstat = self.zstat

        cov = zstat.cov(mean=self.mean)
        cov2 = np.cov(z, rowvar=False, ddof=0)
        assert_allclose(cov, cov2, rtol=1e-13)

        pcov = zstat.pcov()
        pcov2 = z.T @ z / z.shape[0]
        assert_allclose(pcov, pcov2, rtol=1e-13)

    def test_conversion(self):
        zstat = self.zstat

        covz = zstat.cov()
        pcovz = zstat.pcov()
        covr = zstat.cov_rvec()
        cove = zstat.cov_ext()

        covr2 = cov_rvec(covz, pcovz)
        covz2, pcovz2 = cov_from_rvec(covr)
        cove2 = cov_ext(covz, pcovz)
        assert_allclose(covz, covz2, rtol=1e-13)
        assert_allclose(pcovz, pcovz2, rtol=1e-13)
        assert_allclose(covr, covr2, rtol=1e-13)
        assert_allclose(cove, cove2, rtol=1e-13)

        covrc = zstat.cov_circular()
        covzc, pcovzc = cov_from_rvec(covrc)
        assert_allclose(covzc, covz, rtol=1e-13)
        assert_allclose(pcovzc, 0, atol=1e-13)


class TestComplex1(CheckComplex):

    @classmethod
    def setup_class(cls):
        np.random.seed(974284)
        nobs = 500
        k = 2
        z = np.random.randn(nobs, k) + np.random.randn(nobs, k) * 1j
        cls.zrvs = z - z.mean(0)
        cls.mean = 0
        cls.zstat = RandVarComplex(cls.zrvs, demean=False)


class TestComplex2(CheckComplex):

    @classmethod
    def setup_class(cls):
        np.random.seed(974284)
        nobs = 500
        k = 2
        # variance of real and complex parts differ
        z = np.random.randn(nobs, k) + 0.5 * np.random.randn(nobs, k) * 1j
        cls.zrvs = z - z.mean(0)
        cls.mean = 0
        cls.zstat = RandVarComplex(cls.zrvs, demean=False)


class TestComplex3(CheckComplex):

    @classmethod
    def setup_class(cls):
        np.random.seed(974284)
        nobs = 500
        k = 2
        # variance of real and complex parts differ
        z = np.random.randn(nobs, k) + 0.5 * np.random.randn(nobs, k) * 1j
        cls.zrvs = z - z.mean(0)
        cls.mean = 0
        cls.zstat = RandVarComplex(cls.zrvs, demean=False)


class TestComplex4(CheckComplex):
    # real and complex parts: variance differs and parts are correlated

    @classmethod
    def setup_class(cls):
        np.random.seed(974284)
        nobs = 500
        k = 2
        k2 = 2 * k

        # cov = np.eye(k2) + np.ones((k2, k2)) / 2  # with correlation
        cov = np.diag([1, 1, 0.5, 0.5]) + np.ones((k2, k2)) / 2
        r = np.random.multivariate_normal(np.zeros(k2), cov, size=nobs)
        z = r[:, :k] + 1j * r[:, k:]
        cls.zrvs = z - z.mean(0)
        cls.mean = 0
        cls.zstat = RandVarComplex(cls.zrvs, demean=False)
