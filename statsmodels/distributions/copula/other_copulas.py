# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 19:19:45 2021

Author: Josef Perktold
License: BSD-3

"""
import numpy as np
from scipy import stats
from scipy._lib._util import check_random_state  # noqa

from statsmodels.distributions.copula.copulas import Copula


class IndependentCopula(Copula):
    """Independent copula.

    .. math::

        C_\theta(u,v) = uv

    """
    def __init__(self, d=2):
        self.d = d
        super().__init__(d=self.d)

    def random(self, n=1, random_state=None):
        rng = check_random_state(random_state)
        x = rng.random((n, self.d))
        return x

    def pdf(self, u):
        return np.ones((len(u), 1))

    def cdf(self, u):
        return np.prod(u, axis=1)

    def plot_pdf(self, *args):
        raise NotImplementedError("PDF is constant over the domain.")


def rvs_kernel(sample, size, bw=1, k_func=None, return_extras=False):
    """Random sampling from empirical copula using Beta distribution

    Parameters
    ----------
    sample : ndarray
        Sample of multivariate observations in (o, 1) interval.
    size : int
        Number of observations to simulate.
    bw : float
        Bandwidth for Beta sampling. The beta copula corresponds to a kernel
        estimate of the distribution. bw=1 corresponds to the empirical beta
        copula. A small bandwidth like bw=0.001 corresponds to small noise
        added to the empirical distribution. Larger bw, e.g. bw=10 corresponds
        to kernel estimate with more smoothing.
    k_func : None or callable
        The default kernel function is currently a beta function with 1 added
        to the first beta parameter.
    return_extras : bool
        If this is False, then only the random sample will be returned.
        If true, then extra information is returned that is mainly of interest
        for verification.

    Returns
    -------
    rvs : ndarray
        Multivariate sample with ``size`` observations drawn from the Beta
        Copula.

    Notes
    -----
    Status: experimental, API will change.
    """
    # vectorized for observations
    n = sample.shape[0]
    if k_func is None:
        kfunc = _kernel_rvs_beta1
    idx = np.random.randint(0, n, size=size)
    xi = sample[idx]
    krvs = np.column_stack([kfunc(xii, bw) for xii in xi.T])

    if return_extras:
        return krvs, idx, xi
    else:
        return krvs


def _kernel_rvs_beta(x, bw):
    # Beta kernel for density, pdf, estimation
    return stats.beta.rvs(x / bw + 1, (1 - x) / bw + 1, size=x.shape)


def _kernel_rvs_beta1(x, bw):
    # Beta kernel for density, pdf, estimation
    # Kiriliouk, Segers, Tsukuhara 2020 arxiv, using bandwith 1/nobs sample
    return stats.beta.rvs(x / bw, (1 - x) / bw + 1)
