# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 19:19:45 2021

Author: Josef Perktold
License: BSD-3

"""
import numpy as np
from scipy import stats

from statsmodels.distributions.copula.copulas import Copula


class GaussianCopula(Copula):
    r"""Gaussian copula.

    It is constructed from a multivariate normal distribution over
    :math:`\mathbb{R}^d` by using the probability integral transform.

    For a given correlation matrix :math:`R \in[-1, 1]^{d \times d}`,
    the Gaussian copula with parameter matrix :math:`R` can be written
    as:

    .. math::

        C_R^{\text{Gauss}}(u) = \Phi_R\left(\Phi^{-1}(u_1),\dots,
        \Phi^{-1}(u_d) \right),

    where :math:`\Phi^{-1}` is the inverse cumulative distribution function
    of a standard normal and :math:`\Phi_R` is the joint cumulative
    distribution function of a multivariate normal distribution with mean
    vector zero and covariance matrix equal to the correlation
    matrix :math:`R`.

    """

    def __init__(self, cov=None):
        super().__init__(d=np.inf)
        if cov is None:
            cov = [[1., 0.], [0., 1.]]
        self.density = stats.norm()
        self.mv_density = stats.multivariate_normal(cov=cov)

    def random(self, n=1, random_state=None):
        x = self.mv_density.rvs(size=n, random_state=random_state)
        return self.density.cdf(x)

    def pdf(self, u):
        ppf = self.density.ppf(u)
        mv_pdf_ppf = self.mv_density.pdf(ppf)

        return mv_pdf_ppf / np.prod(self.density.pdf(ppf), axis=1)

    def cdf(self, u):
        ppf = self.density.ppf(u)
        return self.mv_density.cdf(ppf)


class StudentCopula(Copula):
    """Student copula."""

    def __init__(self, df=1, cov=None):
        super().__init__(d=np.inf)
        if cov is None:
            cov = [[1., 0.], [0., 1.]]
        self.density = stats.t(df=df)
        self.mv_density = stats.multivariate_t(shape=cov, df=df)

    def random(self, n=1, random_state=None):
        x = self.mv_density.rvs(size=n, random_state=random_state)
        return self.density.cdf(x)

    def pdf(self, u):
        ppf = self.density.ppf(u)
        mv_pdf_ppf = self.mv_density.pdf(ppf)

        return mv_pdf_ppf / np.prod(self.density.pdf(ppf), axis=1)

    def cdf(self, u):
        raise NotImplementedError("CDF not available in closed form.")
        # ppf = self.density.ppf(u)
        # mvt = MVT([0, 0], self.cov, self.df)
        # return mvt.cdf(ppf)
