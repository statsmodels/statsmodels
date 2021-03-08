# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 16:18:21 2021

Author: Josef Perktold
License: BSD-3

"""


import numpy as np
from numpy.testing import assert_array_less
from scipy import stats
import pytest

import statsmodels.nonparametric.kernels_asymmetric as kern


kernels_rplus = [("gamma", 0.1),
                 ("gamma2", 0.1),
                 ("invgamma", 0.02),
                 ("invgauss", 0.01),
                 ("recipinvgauss", 0.1),
                 ("bs", 0.1),
                 ("lognorm", 0.01),
                 ("weibull", 0.1),
                 ]

kernels_unit = [("beta", 0.005),
                ("beta2", 0.005),
                ]


class TestKernelsRplus(object):

    @classmethod
    def setup_class(cls):
        b = 2
        scale = 1.5
        np.random.seed(1)
        nobs = 1000
        distr0 = stats.gamma(b, scale=scale)
        rvs = distr0.rvs(size=nobs)
        x_plot = np.linspace(0.5, 16, 51) + 1e-13

        cls.rvs = rvs
        cls.x_plot = x_plot
        cls.pdf_dgp = distr0.pdf(x_plot)
        cls.cdf_dgp = distr0.cdf(x_plot)
        cls.amse_pdf = 1e-4  # tol for average mean squared error
        cls.amse_cdf = 5e-4

    @pytest.mark.parametrize('case', kernels_rplus)
    def test_kernels(self, case):
        name, bw = case

        rvs = self.rvs
        x_plot = self.x_plot

        kde = []
        kce = []
        func_pdf = getattr(kern, "kernel_pdf_" + name)
        func_cdf = getattr(kern, "kernel_cdf_" + name)
        for xi in x_plot:
            kde.append(func_pdf(xi, rvs, bw))
            kce.append(func_cdf(xi, rvs, bw))

        kde = np.asarray(kde)
        kce = np.asarray(kce)

        # average mean squared error
        amse = ((kde - self.pdf_dgp)**2).mean()
        assert_array_less(amse, self.amse_pdf)
        amse = ((kce - self.cdf_dgp)**2).mean()
        assert_array_less(amse, self.amse_cdf)


class TestKernelsUnit(TestKernelsRplus):

    @classmethod
    def setup_class(cls):
        np.random.seed(987456)
        nobs = 1000
        distr0 = stats.beta(2, 3)
        rvs = distr0.rvs(size=nobs)
        x_plot = np.linspace(0, 1, 51)

        cls.rvs = rvs
        cls.x_plot = x_plot
        cls.pdf_dgp = distr0.pdf(x_plot)
        cls.cdf_dgp = distr0.cdf(x_plot)
        cls.amse_pdf = 0.01
        cls.amse_cdf = 5e-3

    @pytest.mark.parametrize('case', kernels_unit)
    def test_kernels(self, case):
        super(TestKernelsUnit, self).test_kernels(case)