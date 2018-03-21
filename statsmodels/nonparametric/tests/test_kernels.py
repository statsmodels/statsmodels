# -*- coding: utf-8 -*-
"""

Created on Sat Dec 14 17:23:25 2013

Author: Josef Perktold
"""
from __future__ import print_function
import os
import numpy as np
from statsmodels.sandbox.nonparametric import kernels

from numpy.testing import assert_allclose, assert_array_less

DEBUG = 0

curdir = os.path.dirname(os.path.abspath(__file__))
fname = 'results/results_kernel_regression.csv'
results = np.recfromcsv(os.path.join(curdir, fname))

y = results['accident']
x = results['service']
positive = x >= 0
x = np.log(x[positive])
y = y[positive]
xg = np.linspace(x.min(), x.max(), 40) # grid points default in Stata

#kern_name = 'gau'
#kern = kernels.Gaussian()
#kern_name = 'epan2'
#kern = kernels.Epanechnikov()
#kern_name = 'rec'
#kern = kernels.Uniform()  # ours looks awful
#kern_name = 'tri'
#kern = kernels.Triangular()
#kern_name = 'cos'
#kern = kernels.Cosine()  #doesn't match up, nan in Stata results ?
#kern_name = 'bi'
#kern = kernels.Biweight()

class CheckKernelMixin(object):

    se_rtol = 0.7
    upp_rtol = 0.1
    low_rtol = 0.2
    low_atol = 0.3

    def test_smoothconf(self):
        kern_name = self.kern_name
        kern = self.kern
        #fittedg = np.array([kernels.Epanechnikov().smoothconf(x, y, xi) for xi in xg])
        fittedg = np.array([kern.smoothconf(x, y, xi) for xi in xg])
        # attach for inspection from outside of test run
        self.fittedg = fittedg

        res_fitted = results['s_' + kern_name]
        res_se = results['se_' + kern_name]
        crit = 1.9599639845400545  # norm.isf(0.05 / 2)
        # implied standard deviation from conf_int
        se = (fittedg[:, 2] - fittedg[:, 1]) / crit
        fitted = fittedg[:, 1]

        # check both rtol & atol
        assert_allclose(fitted, res_fitted, rtol=5e-7, atol=1e-20)
        assert_allclose(fitted, res_fitted, rtol=0, atol=1e-6)

        # TODO: check we are using a different algorithm for se
        # The following are very rough checks

        self.se = se
        self.res_se = res_se
        se_valid = np.isfinite(res_se)
        # if np.any(~se_valid):
        #    print('nan in stata result', self.__class__.__name__)
        assert_allclose(se[se_valid], res_se[se_valid], rtol=self.se_rtol, atol=0.2)
        # check that most values are closer
        mask = np.abs(se - res_se) > (0.2 + 0.2 * res_se)
        if not hasattr(self, 'se_n_diff'):
            se_n_diff = 40 * 0.125
        else:
            se_n_diff = self.se_n_diff
        assert_array_less(mask.sum(), se_n_diff + 1)  # at most 5 large diffs

        if DEBUG:
            # raises: RuntimeWarning: invalid value encountered in divide
            print(fitted / res_fitted - 1)
            print(se / res_se - 1)
        # Stata only displays ci, doesn't save it
        res_upp = res_fitted + crit * res_se
        res_low = res_fitted - crit * res_se
        self.res_fittedg = np.column_stack((res_low, res_fitted, res_upp))
        if DEBUG:
            print(fittedg[:, 2] / res_upp - 1)
            print(fittedg[:, 2] - res_upp)
            print(fittedg[:, 0] - res_low)
            print(np.max(np.abs(fittedg[:, 2] / res_upp - 1)))
        assert_allclose(fittedg[se_valid, 2], res_upp[se_valid],
                        rtol=self.upp_rtol, atol=0.2)
        assert_allclose(fittedg[se_valid, 0], res_low[se_valid],
                        rtol=self.low_rtol, atol=self.low_atol)

        #assert_allclose(fitted, res_fitted, rtol=0, atol=1e-6)

    def t_est_smoothconf_data(self):
        kern = self.kern
        crit = 1.9599639845400545  # norm.isf(0.05 / 2)
        # no reference results saved to csv yet
        fitted_x = np.array([kern.smoothconf(x, y, xi) for xi in x])
        if DEBUG:
            print(fitted_x[:, 2] - fitted_x[:, 1]) / crit

class TestEpan(CheckKernelMixin):
    kern_name = 'epan2'
    kern = kernels.Epanechnikov()

class TestGau(CheckKernelMixin):
    kern_name = 'gau'
    kern = kernels.Gaussian()

class TestUniform(CheckKernelMixin):
    kern_name = 'rec'
    kern = kernels.Uniform()
    se_rtol = 0.8
    se_n_diff = 8
    upp_rtol = 0.4
    low_rtol = 0.2
    low_atol = 0.8

class TestTriangular(CheckKernelMixin):
    kern_name = 'tri'
    kern = kernels.Triangular()
    se_n_diff = 10
    upp_rtol = 0.15
    low_rtol = 0.3

class T_estCosine(CheckKernelMixin):
    # Stata results for Cosine look strange, has nans
    kern_name = 'cos'
    kern = kernels.Cosine2()

class TestBiweight(CheckKernelMixin):
    kern_name = 'bi'
    kern = kernels.Biweight()
    se_n_diff = 9
    low_rtol = 0.3
