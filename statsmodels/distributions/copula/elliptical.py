# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 19:19:45 2021

Author: Josef Perktold
Author: Pamphile Roy
License: BSD-3

"""
import numpy as np
from scipy import stats
# scipy compat:
from statsmodels.compat.scipy import multivariate_t

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

    def __init__(self, corr=None):
        super().__init__(d=np.inf)
        if corr is None:
            corr = [[1., 0.], [0., 1.]]
        self.corr = np.asarray(corr)
        self.density = stats.norm()
        self.mv_density = stats.multivariate_normal(cov=corr)

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

    def tau(self, pearson_corr=None):

        if pearson_corr is None:
            if self.corr.shape == (2, 2):
                corr = self.corr[0, 1]
            else:
                corr = self.corr
        else:
            corr = pearson_corr
        tau = 2 * np.arcsin(corr) / np.pi
        return tau

    def dependence_tail(self, pearson_corr=None):

        return 0, 0

    def corr_from_tau(self, tau):
        """pearson correlation from kendall's tau

        Joe (2014) p. 164
        """
        corr = np.sin(tau * np.pi / 2)
        return corr


class StudentTCopula(Copula):
    """Student copula."""

    def __init__(self, df=1, corr=None):
        super().__init__(d=np.inf)
        if corr is None:
            corr = [[1., 0.], [0., 1.]]

        self.df = df
        self.corr = np.asarray(corr)
        self.density = stats.t(df=df)
        self.mv_density = multivariate_t(shape=corr, df=df)

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
        # mvt = MVT([0, 0], self.corr, self.df)
        # return mvt.cdf(ppf)

    def tau(self, pearson_corr=None):
        """bivariate kendall's tau

        Joe (2014) p. 182
        """
        if pearson_corr is None:
            if self.corr.shape == (2, 2):
                corr = self.corr[0, 1]
            else:
                corr = self.corr
        else:
            corr = pearson_corr
        rho = 2 * np.arcsin(corr) / np.pi
        return rho

    def spearmans_rho(self, pearson_corr=None):
        """bivariate spearman's rho

        Joe (2014) p. 182
        """
        if pearson_corr is None:
            if self.corr.shape == (2, 2):
                corr = self.corr[0, 1]
            else:
                corr = self.corr
        else:
            corr = pearson_corr
        tau = 6 * np.arcsin(corr / 2) / np.pi
        return tau

    def dependence_tail(self, pearson_corr=None):
        """bivariate tail dependence parameter

        Joe (2014) p. 182
        """
        if pearson_corr is None:
            if self.corr.shape == (2, 2):
                corr = self.corr[0, 1]
            else:
                corr = self.corr
        else:
            corr = pearson_corr

        df = self.df
        t = - np.sqrt((df + 1) * (1 - corr) / 1 + corr)
        # Note self.density is frozen, df cannot change, use stats.t instead
        lam = 2 * stats.t.cdf(t, df + 1)
        return lam, lam

    def corr_from_tau(self, tau):
        """pearson correlation from kendall's tau

        Joe (2014) p. 164
        """
        corr = np.sin(tau * np.pi / 2)
        return corr
