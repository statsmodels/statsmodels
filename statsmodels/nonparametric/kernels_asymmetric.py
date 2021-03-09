# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 11:12:24 2021

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from scipy import stats, special


def pdf_kernel_asym(x, sample, bw, kernel_type, weights=None):

    kfunc = kernel_dict_pdf[kernel_type]
    pdfi = kfunc(x, sample, bw, return_comp=True)

    if weights is None:
        return pdfi.mean(-1)
    else:
        return pdfi @ weights


def cdf_kernel_asym(x, sample, bw, kernel_type, weights=None):

    kfunc = kernel_dict_cdf[kernel_type]
    cdfi = kfunc(x, sample, bw, return_comp=True)

    if weights is None:
        return cdfi.mean(-1)
    else:
        return cdfi @ weights


def kernel_pdf_gamma(x, sample, bw, return_comp=False):
    """gamma kernel for pdf

    Reference
    Chen 2000
    Bouezmarni and Scaillet 2205
    """

    pdfi = stats.gamma.pdf(sample, x / bw + 1, scale=bw)

    if return_comp:
        return pdfi
    else:
        return pdfi.mean(-1)


def kernel_cdf_gamma(x, sample, bw, return_comp=False):
    """gamma kernel for cdf

    Reference
    Chen 2000
    Bouezmarni and Scaillet 2205
    """
    # it uses the survival function, but I don't know why.
    cdfi = stats.gamma.sf(sample, x / bw + 1, scale=bw)

    if return_comp:
        return cdfi
    else:
        return cdfi.mean(-1)


def _kernel_pdf_gamma(x, sample, bw):
    """gamma kernel for pdf, without boundary corrected part

    drops `+ 1` in shape parameter

    It should be possible to use this if probability in
    neighborhood of zero boundary is small.

    """
    return stats.gamma.pdf(sample, x / bw, scale=bw).mean(-1)


def _kernel_cdf_gamma(x, sample, bw):
    """gamma kernel for cdf, without boundary corrected part

    drops `+ 1` in shape parameter

    It should be possible to use this if probability in
    neighborhood of zero boundary is small.

    """
    return stats.gamma.sf(sample, x / bw, scale=bw).mean(-1)


def kernel_pdf_gamma2(x, sample, bw):
    """gamma kernel for pdf with boundary correction

    """

    if np.size(x) == 1:
        # without vectorizing, easier to read
        if x < 2 * bw:
            a = (x / bw)**2 + 1
        else:
            a = x / bw
    else:
        a = x / bw
        mask = x < 2 * bw
        a[mask] = a[mask]**2 + 1
    pdf = stats.gamma.pdf(sample, a, scale=bw).mean(-1)

    return pdf


def kernel_cdf_gamma2(x, sample, bw):
    """gamma kernel for pdf with boundary correction

    """

    if np.size(x) == 1:
        # without vectorizing
        if x < 2 * bw:
            a = (x / bw)**2 + 1
        else:
            a = x / bw
    else:
        a = x / bw
        mask = x < 2 * bw
        a[mask] = a[mask]**2 + 1
    pdf = stats.gamma.sf(sample, a, scale=bw).mean(-1)

    return pdf


def kernel_pdf_invgamma(x, sample, bw):
    """inverse gamma kernel for pdf

    de Micheaux, Ouimet (arxiv Nov 2020) for cdf kernel
    """
    return stats.invgamma.pdf(sample, 1 / bw + 1, scale=x / bw).mean(-1)


def kernel_cdf_invgamma(x, sample, bw):
    """inverse gamma kernel for pdf

    de Micheaux, Ouimet (arxiv Nov 2020) for cdf kernel
    """
    return stats.invgamma.sf(sample, 1 / bw + 1, scale=x / bw).mean(-1)


def kernel_pdf_beta(x, sample, bw):
    return stats.beta.pdf(sample, x / bw + 1, (1 - x) / bw + 1).mean(-1)


def kernel_cdf_beta(x, sample, bw):
    return stats.beta.sf(sample, x / bw + 1, (1 - x) / bw + 1).mean(-1)


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

    if np.size(x) == 1:
        # without vectorizing:
        if x < 2 * bw:
            a = a1 - np.sqrt(a2 - x**2 - x / bw)
            pdf = stats.beta.pdf(sample, a, (1 - x) / bw).mean(-1)
        elif x > (1 - 2 * bw):
            x_ = 1 - x
            a = a1 - np.sqrt(a2 - x_**2 - x_ / bw)
            pdf = stats.beta.pdf(sample, x / bw, a).mean(-1)
        else:
            pdf = stats.beta.pdf(sample, x / bw, (1 - x) / bw).mean(-1)
    else:
        alpha = x / bw
        beta = (1 - x) / bw

        mask_low = x < 2 * bw
        x_ = x[mask_low]
        alpha[mask_low] = a1 - np.sqrt(a2 - x_**2 - x_ / bw)

        mask_upp = x > (1 - 2 * bw)
        x_ = 1 - x[mask_upp]
        beta[mask_upp] = a1 - np.sqrt(a2 - x_**2 - x_ / bw)

        pdf = stats.beta.pdf(sample, alpha, beta).mean(-1)

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

    if np.size(x) == 1:
        # without vectorizing:
        if x < 2 * bw:
            a = a1 - np.sqrt(a2 - x**2 - x / bw)
            pdf = stats.beta.sf(sample, a, (1 - x) / bw).mean(-1)
        elif x > (1 - 2 * bw):
            x_ = 1 - x
            a = a1 - np.sqrt(a2 - x_**2 - x_ / bw)
            pdf = stats.beta.sf(sample, x / bw, a).mean(-1)
        else:
            pdf = stats.beta.sf(sample, x / bw, (1 - x) / bw).mean(-1)
    else:
        alpha = x / bw
        beta = (1 - x) / bw
        mask_low = x < 2 * bw

        x_ = x[mask_low]
        alpha[mask_low] = a1 - np.sqrt(a2 - x_**2 - x_ / bw)

        mask_upp = x > (1 - 2 * bw)
        x_ = 1 - x[mask_upp]
        beta[mask_upp] = a1 - np.sqrt(a2 - x_**2 - x_ / bw)

        pdf = stats.beta.sf(sample, alpha, beta).mean(-1)

    return pdf


def kernel_pdf_invgauss(x, sample, bw):
    """inverse gaussian kernel density

    Scaillet 2004
    """
    m = x
    lam = 1 / bw
    return stats.invgauss.pdf(sample, m / lam, scale=lam).mean(-1)


def kernel_pdf_invgauss_(x, sample, bw):
    """inverse gaussian kernel density, explicit formula

    Scaillet 2004
    """
    pdf = (1 / np.sqrt(2 * np.pi * bw * sample**3) *
           np.exp(- 1 / (2 * bw * x) * (sample / x - 2 + x / sample)))
    return pdf.mean(-1)


def kernel_cdf_invgauss(x, sample, bw):
    """inverse gaussian kernel for cdf

    Scaillet 2004
    """
    m = x
    lam = 1 / bw
    return stats.invgauss.sf(sample, m / lam, scale=lam).mean(-1)


def kernel_pdf_recipinvgauss(x, sample, bw):
    """reciprocal inverse gaussian kernel density

    Scaillet 2004
    """
    # need shape-scale parameterization for scipy
    # references use m, lambda parameterization
    m = 1 / (x - bw)
    lam = 1 / bw
    return stats.recipinvgauss.pdf(sample, m / lam, scale=1 / lam).mean(-1)


def kernel_pdf_recipinvgauss_(x, sample, bw):
    """reciprocal inverse gaussian kernel density, explicit formula

    Scaillet 2004
    """

    pdf = (1 / np.sqrt(2 * np.pi * bw * sample) *
           np.exp(- (x - bw) / (2 * bw) * sample / (x - bw) - 2 +
                  (x - bw) / sample))
    return pdf.mean(-1)


def kernel_cdf_recipinvgauss(x, sample, bw):
    """reciprocal inverse gaussian kernel for cdf

    Scaillet 2004
    """
    # need shape-scale parameterization for scipy
    # references use m, lambda parameterization
    m = 1 / (x - bw)
    lam = 1 / bw
    return stats.recipinvgauss.sf(sample, m / lam, scale=1 / lam).mean(-1)


def kernel_pdf_bs(x, sample, bw):
    """birnbaum saunders (normal distribution) kernel density

    Jin, Kawczak 2003
    """
    # need shape-scale parameterization for scipy
    return stats.fatiguelife.pdf(sample, bw, scale=x).mean(-1)


def kernel_cdf_bs(x, sample, bw):
    """birnbaum saunders (normal distribution) kernel cdf

    Jin, Kawczak 2003
    """
    # need shape-scale parameterization for scipy
    return stats.fatiguelife.sf(sample, bw, scale=x).mean(-1)


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
    return stats.lognorm.pdf(sample, bw_, scale=x).mean(-1)


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
    return stats.lognorm.sf(sample, bw_, scale=x).mean(-1)


def kernel_pdf_lognorm_(x, sample, bw):
    """log-normal kernel density

    Jin, Kawczak 2003
    """
    term = 8 * np.log(1 + bw)  # this is 2 * variance in normal pdf
    pdf = (1 / np.sqrt(term * np.pi) / sample *
           np.exp(- (np.log(x) - np.log(sample))**2 / term))
    return pdf.mean(-1)


def kernel_pdf_weibull(x, sample, bw):
    """weibull kernel density

    Mombeni et al. for distribution, i.e. cdf, kernel
    """
    # need shape-scale parameterization for scipy
    # references use m, lambda parameterization
    return stats.weibull_min.pdf(sample, 1 / bw,
                                 scale=x / special.gamma(1 + bw)).mean(-1)


def kernel_cdf_weibull(x, sample, bw):
    """weibull kernel density

    Mombeni et al. for distribution, i.e. cdf, kernel
    """
    # need shape-scale parameterization for scipy
    # references use m, lambda parameterization
    return stats.weibull_min.sf(sample, 1 / bw,
                                scale=x / special.gamma(1 + bw)).mean(-1)


# produced wth
# print("\n".join(['"%s": %s,' % (i.split("_")[-1], i) for i in dir(kern)
#                  if "kernel" in i and not i.endswith("_")]))
kernel_dict_cdf = {
    "beta": kernel_cdf_beta,
    "beta2": kernel_cdf_beta2,
    "bs": kernel_cdf_bs,
    "gamma": kernel_cdf_gamma,
    "gamma2": kernel_cdf_gamma2,
    "invgamma": kernel_cdf_invgamma,
    "invgauss": kernel_cdf_invgauss,
    "lognorm": kernel_cdf_lognorm,
    "recipinvgauss": kernel_cdf_recipinvgauss,
    "weibull": kernel_cdf_weibull,
    }

kernel_dict_pdf = {
    "beta": kernel_pdf_beta,
    "beta2": kernel_pdf_beta2,
    "bs": kernel_pdf_bs,
    "gamma": kernel_pdf_gamma,
    "gamma2": kernel_pdf_gamma2,
    "invgamma": kernel_pdf_invgamma,
    "invgauss": kernel_pdf_invgauss,
    "lognorm": kernel_pdf_lognorm,
    "recipinvgauss": kernel_pdf_recipinvgauss,
    "weibull": kernel_pdf_weibull,
    }
