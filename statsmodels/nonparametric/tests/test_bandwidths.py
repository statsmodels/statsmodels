# -*- coding: utf-8 -*-
"""

Tests for bandwidth selection and calculation.

Author: Padarn Wilson
"""

import numpy as np
from statsmodels.sandbox.nonparametric import kernels

from numpy.testing import assert_allclose


class CheckNormalReferenceConstant(object):

    def test_calculate_normal_reference_constant(self):
        const = self.constant
        kern = self.kern
        assert_allclose(const, kern.normal_reference_constant, 1e-2)


class TestEpanechnikov(CheckNormalReferenceConstant):

    kern = kernels.Epanechnikov()
    constant = 2.34


class TestGaussian(CheckNormalReferenceConstant):

    kern = kernels.Gaussian()
    constant = 1.06


class TestBiweight(CheckNormalReferenceConstant):

    kern = kernels.Biweight()
    constant = 2.78


class TestTriweight(CheckNormalReferenceConstant):

    kern = kernels.Triweight()
    constant = 3.15

if __name__ == "__main__":
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb'], exit=False)
