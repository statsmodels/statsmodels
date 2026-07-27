"""

Created on Sat Dec 14 17:23:25 2013

Author: Josef Perktold
"""

from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import pandas as pd
import pytest

from statsmodels.sandbox.nonparametric import kernels

DEBUG = 0
curdir = Path(__file__).resolve().parent
fname = "results/results_kernel_regression.csv"
results = pd.read_csv(Path(curdir).joinpath(fname))
y = results["accident"].to_numpy(copy=True)
x = results["service"].to_numpy(copy=True)
positive = x >= 0
x = np.log(x[positive])
y = y[positive]
xg = np.linspace(x.min(), x.max(), 40)


class CheckKernelMixin:
    se_rtol = 0.7
    upp_rtol = 0.1
    low_rtol = 0.2
    low_atol = 0.3

    def test_smoothconf(self):
        kern_name = self.kern_name
        kern = self.kern
        fittedg = np.array([kern.smoothconf(x, y, xi) for xi in xg])
        self.fittedg = fittedg
        res_fitted = results["s_" + kern_name]
        res_se = results["se_" + kern_name]
        crit = 1.9599639845400545
        se = (fittedg[:, 2] - fittedg[:, 1]) / crit
        fitted = fittedg[:, 1]
        assert_allclose(fitted, res_fitted, rtol=5e-07, atol=1e-20)
        assert_allclose(fitted, res_fitted, rtol=0, atol=1e-06)
        self.se = se
        self.res_se = res_se
        se_valid = np.isfinite(res_se)
        assert_allclose(se[se_valid], res_se[se_valid], rtol=self.se_rtol, atol=0.2)
        mask = np.abs(se - res_se) > 0.2 + 0.2 * res_se
        if not hasattr(self, "se_n_diff"):
            se_n_diff = 40 * 0.125
        else:
            se_n_diff = self.se_n_diff
        assert_array_less(mask.sum(), se_n_diff + 1)
        res_upp = res_fitted + crit * res_se
        res_low = res_fitted - crit * res_se
        self.res_fittedg = np.column_stack((res_low, res_fitted, res_upp))
        assert_allclose(
            fittedg[se_valid, 2], res_upp[se_valid], rtol=self.upp_rtol, atol=0.2
        )
        assert_allclose(
            fittedg[se_valid, 0],
            res_low[se_valid],
            rtol=self.low_rtol,
            atol=self.low_atol,
        )

    @pytest.mark.slow
    @pytest.mark.smoke
    def test_smoothconf_data(self):
        kern = self.kern
        np.array([kern.smoothconf(x, y, xi) for xi in x])


class TestEpan(CheckKernelMixin):
    kern_name = "epan2"
    kern = kernels.Epanechnikov()


class TestGau(CheckKernelMixin):
    kern_name = "gau"
    kern = kernels.Gaussian()


class TestUniform(CheckKernelMixin):
    kern_name = "rec"
    kern = kernels.Uniform()
    se_rtol = 0.8
    se_n_diff = 8
    upp_rtol = 0.4
    low_rtol = 0.2
    low_atol = 0.8


class TestTriangular(CheckKernelMixin):
    kern_name = "tri"
    kern = kernels.Triangular()
    se_n_diff = 10
    upp_rtol = 0.15
    low_rtol = 0.3


class TestCosine(CheckKernelMixin):
    kern_name = "cos"
    kern = kernels.Cosine2()

    @pytest.mark.xfail(reason="NaN mismatch", raises=AssertionError, strict=True)
    def test_smoothconf(self):
        super().test_smoothconf()


class TestBiweight(CheckKernelMixin):
    kern_name = "bi"
    kern = kernels.Biweight()
    se_n_diff = 9
    low_rtol = 0.3


def test_tricube():
    res_kx = [
        0.0,
        0.1669853116259163,
        0.5789448302469136,
        0.8243179321289062,
        0.8641975308641975,
        0.8243179321289062,
        0.5789448302469136,
        0.1669853116259163,
        0.0,
    ]
    xx = np.linspace(-1, 1, 9)
    kx = kernels.Tricube()(xx)
    assert_allclose(kx, res_kx, rtol=1e-10)
