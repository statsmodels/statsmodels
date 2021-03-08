# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 11:12:24 2021

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from scipy import stats, special


def kernel_pdf_gamma(x, sample, bw):
    """gamma kernel for pdf

    Reference
    Chen 2000
    Bouezmarni and Scaillet 2205
    """
    return stats.gamma.pdf(sample, x / bw + 1, scale=bw).mean(-1)


def kernel_cdf_gamma(x, sample, bw):
    """gamma kernel for cdf

    Reference
    Chen 2000
    Bouezmarni and Scaillet 2205
    """
    # it uses the survival function, but I don't know why.
    return stats.gamma.sf(sample, x / bw + 1, scale=bw).mean(-1)


def _kernel_pdf_gamma(x, sample, bw):
    """gamma kernel for pdf, without boundary corrected part

    drops `+ 1` in shape parameter

    It should be possible to use this if probability in
    neighborhood of zero boundary is small.

    """
    return stats.gamma.pdf(sample, x / bw, scale=bw).mean()


def _kernel_cdf_gamma(x, sample, bw):
    """gamma kernel for cdf, without boundary corrected part

    drops `+ 1` in shape parameter

    It should be possible to use this if probability in
    neighborhood of zero boundary is small.

    """
    return stats.gamma.sf(sample, x / bw, scale=bw).mean()


def kernel_pdf_gamma2(x, sample, bw):
    """gamma kernel for pdf with boundary correction

    """
    # without vectorizing:
    if np.size(x) == 1:
        if x < 2 * bw:
            a = (x / bw)**2 + 1
        else:
            a = x / bw
    else:
        a = x / bw
        a[x < 2 * bw] = a**2 + 1
    pdf = stats.gamma.pdf(sample, a, scale=bw).mean()

    return pdf


def kernel_cdf_gamma2(x, sample, bw):
    """gamma kernel for pdf with boundary correction

    """
    # without vectorizing:
    if np.size(x) == 1:
        if x < 2 * bw:
            a = (x / bw)**2 + 1
        else:
            a = x / bw
    else:
        a = x / bw
        a[x < 2 * bw] = a**2 + 1
    pdf = stats.gamma.sf(sample, a, scale=bw).mean()

    return pdf


def kernel_pdf_invgamma(x, sample, bw):
    """inverse gamma kernel for pdf

    de Micheaux, Ouimet (arxiv Nov 2020) for cdf kernel
    """
    return stats.invgamma.pdf(sample, 1 / bw + 1, scale=x / bw).mean()


def kernel_cdf_invgamma(x, sample, bw):
    """inverse gamma kernel for pdf

    de Micheaux, Ouimet (arxiv Nov 2020) for cdf kernel
    """
    return stats.invgamma.sf(sample, 1 / bw + 1, scale=x / bw).mean()


def kernel_pdf_beta(x, sample, bw):
    return stats.beta.pdf(sample, x / bw + 1, (1 - x) / bw + 1).mean()


def kernel_cdf_beta(x, sample, bw):
    return stats.beta.sf(sample, x / bw + 1, (1 - x) / bw + 1).mean()


def kernel_pdf_beta2(x, sample, bw):
    """beta kernel for pdf with boundary correction

    not vectorized in x

    Chen 1999
    """
    # a = 2 * bw**2 + 2.5 -
    #     np.sqrt(4 * bw**4 + 6 * bw**2 + 2.25 - x**2 - x / bw)
    # terms a1 and a2 are independent of x
    a1 = 2 * bw**2 + 2.5
    a2 = 4 * bw**4 + 6 * bw**2 + 2.25
    if x < 2 * bw:
        a = a1 - np.sqrt(a2 - x**2 - x / bw)
        pdf = stats.beta.pdf(sample, a, (1 - x) / bw).mean()
    elif x > (1 - 2 * bw):
        x_ = 1 - x
        a = a1 - np.sqrt(a2 - x_**2 - x_ / bw)
        pdf = stats.beta.pdf(sample, x / bw, a).mean()
    else:
        pdf = stats.beta.pdf(sample, x / bw, (1 - x) / bw).mean()

    return pdf


def kernel_cdf_beta2(x, sample, bw):
    """beta kernel for pdf with boundary correction

    not vectorized in x

    Chen 1999
    """
    # a = 2 * bw**2 + 2.5 -
    #     np.sqrt(4 * bw**4 + 6 * bw**2 + 2.25 - x**2 - x / bw)
    # terms a1 and a2 are independent of x
    a1 = 2 * bw**2 + 2.5
    a2 = 4 * bw**4 + 6 * bw**2 + 2.25
    if x < 2 * bw:
        a = a1 - np.sqrt(a2 - x**2 - x / bw)
        pdf = stats.beta.sf(sample, a, (1 - x) / bw).mean()
    elif x > (1 - 2 * bw):
        x_ = 1 - x
        a = a1 - np.sqrt(a2 - x_**2 - x_ / bw)
        pdf = stats.beta.sf(sample, x / bw, a).mean()
    else:
        pdf = stats.beta.sf(sample, x / bw, (1 - x) / bw).mean()

    return pdf


def kernel_pdf_invgauss(x, sample, bw):
    """inverse gaussian kernel density

    Scaillet 2004
    """
    m = x
    lam = 1 / bw
    return stats.invgauss.pdf(sample, m / lam, scale=lam).mean()


def kernel_pdf_invgauss_(x, sample, bw):
    """inverse gaussian kernel density, explicit formula

    Scaillet 2004
    """
    pdf = (1 / np.sqrt(2 * np.pi * bw * sample**3) *
           np.exp(- 1 / (2 * bw * x) * (sample / x - 2 + x / sample)))
    return pdf.mean()


def kernel_cdf_invgauss(x, sample, bw):
    """inverse gaussian kernel for cdf

    Scaillet 2004
    """
    m = x
    lam = 1 / bw
    return stats.invgauss.sf(sample, m / lam, scale=lam).mean()


def kernel_pdf_recipinvgauss(x, sample, bw):
    """reciprocal inverse gaussian kernel density

    Scaillet 2004
    """
    # need shape-scale parameterization for scipy
    # references use m, lambda parameterization
    m = 1 / (x - bw)
    lam = 1 / bw
    return stats.recipinvgauss.pdf(sample, m / lam, scale=1 / lam).mean()


def kernel_pdf_recipinvgauss_(x, sample, bw):
    """reciprocal inverse gaussian kernel density, explicit formula

    Scaillet 2004
    """

    pdf = (1 / np.sqrt(2 * np.pi * bw * sample) *
           np.exp(- (x - bw) / (2 * bw) * sample / (x - bw) - 2 +
                  (x - bw) / sample))
    return pdf.mean()


def kernel_cdf_recipinvgauss(x, sample, bw):
    """reciprocal inverse gaussian kernel for cdf

    Scaillet 2004
    """
    # need shape-scale parameterization for scipy
    # references use m, lambda parameterization
    m = 1 / (x - bw)
    lam = 1 / bw
    return stats.recipinvgauss.sf(sample, m / lam, scale=1 / lam).mean()


def kernel_pdf_bs(x, sample, bw):
    """birnbaum saunders (normal distribution) kernel density

    Jin, Kawczak 2003
    """
    # need shape-scale parameterization for scipy
    return stats.fatiguelife.pdf(sample, bw, scale=x).mean()


def kernel_cdf_bs(x, sample, bw):
    """birnbaum saunders (normal distribution) kernel cdf

    Jin, Kawczak 2003
    """
    # need shape-scale parameterization for scipy
    return stats.fatiguelife.sf(sample, bw, scale=x).mean()


def kernel_pdf_lognorm(x, sample, bw):
    """log-normal kernel density

    Jin, Kawczak 2003
    """

    # need shape-scale parameterization for scipy
    # not sure why JK picked this normalization, makes required bw small
    # maybe we should skip this transformation and just use bw
    # Funke and Kawka 2015 (table 1) use bw (or bw**2) corresponding to
    #    variance of normal pdf
    # bw = np.exp(bw_**2 / 4) - 1  # this is inverse transformation
    bw_ = np.sqrt(4*np.log(1+bw))
    return stats.lognorm.pdf(sample, bw_, scale=x).mean()


def kernel_cdf_lognorm(x, sample, bw):
    """log-normal kernel cdf

    Jin, Kawczak 2003
    """

    # need shape-scale parameterization for scipy
    # not sure why JK picked this normalization, makes required bw small
    # maybe we should skip this transformation and just use bw
    # Funke and Kawka 2015 (table 1) use bw (or bw**2) corresponding to
    #    variance of normal pdf
    # bw = np.exp(bw_**2 / 4) - 1  # this is inverse transformation
    bw_ = np.sqrt(4*np.log(1+bw))
    return stats.lognorm.sf(sample, bw_, scale=x).mean()


def kernel_pdf_lognorm_(x, sample, bw):
    """log-normal kernel density

    Jin, Kawczak 2003
    """
    term = 8 * np.log(1 + bw)  # this is 2 * variance in normal pdf
    pdf = (1 / np.sqrt(term * np.pi) / sample *
           np.exp(- (np.log(x) - np.log(sample))**2 / term))
    return pdf.mean()


def kernel_pdf_weibull(x, sample, bw):
    """weibull kernel density

    Mombeni et al. for distribution, i.e. cdf, kernel
    """
    # need shape-scale parameterization for scipy
    # references use m, lambda parameterization
    return stats.weibull_min.pdf(sample, 1 / bw,
                                 scale=x / special.gamma(1 + bw)).mean()


def kernel_cdf_weibull(x, sample, bw):
    """weibull kernel density

    Mombeni et al. for distribution, i.e. cdf, kernel
    """
    # need shape-scale parameterization for scipy
    # references use m, lambda parameterization
    return stats.weibull_min.sf(sample, 1 / bw,
                                scale=x / special.gamma(1 + bw)).mean()
