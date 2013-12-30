# -*- coding: utf-8 -*-
"""

Tests for bandwidth selection and calculation.

Author: Padarn Wilson
"""

import numpy as np
from scipy import stats

from statsmodels.sandbox.nonparametric import kernels
from statsmodels.distributions.mixture_rvs import mixture_rvs
from statsmodels.nonparametric.kde import KDEUnivariate as KDE
from statsmodels.nonparametric.bandwidths import select_bandwidth



from numpy.testing import assert_allclose

# setup test data

np.random.seed(12345)
Xi = mixture_rvs([.25,.75], size=200, dist=[stats.norm, stats.norm],
                kwargs = (dict(loc=-1,scale=.5),dict(loc=1,scale=.5)))

class TestBandwidthCalculation(object):

    def test_calculate_bandwidth_gaussian(self):

        bw_expected = [0.29774853596742024,
                       0.25304408155871411,
                       0.29781147113698891]

        kern = kernels.Gaussian()
        
        bw_calc = [0, 0, 0]
        for ii, bw in enumerate(['scott','silverman','normal_reference']):
            bw_calc[ii] = select_bandwidth(Xi, bw, kern)

        assert_allclose(bw_expected, bw_calc)


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
