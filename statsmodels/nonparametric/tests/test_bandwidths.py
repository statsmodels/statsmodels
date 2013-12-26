# -*- coding: utf-8 -*-
"""

Tests for bandwidth selection and calculation.

Author: Padarn Wilson
"""

import numpy as np
from statsmodels.sandbox.nonparametric import kernels

from numpy.testing import assert_allclose


class CheckSilvermanConstant(object):

    def test_calculate_silverman_constant(self):
        const = self.constant
        kern = self.kern
        assert_allclose(const, kern.silverman_constant, 1e-2)


class TestEpanechnikov(CheckSilvermanConstant):

    kern = kernels.Epanechnikov()
    constant = 2.34


class TestGaussian(CheckSilvermanConstant):

    kern = kernels.Gaussian()
    constant = 1.06


class TestBiweight(CheckSilvermanConstant):

    kern = kernels.Biweight()
    constant = 2.78


class TestTriweight(CheckSilvermanConstant):

    kern = kernels.Triweight()
    constant = 3.15

if __name__ == "__main__":
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb'], exit=False)
