"""
Created on Fri Jan 29 19:19:45 2021

Author: Josef Perktold
License: BSD-3

"""

from statsmodels.compat.pandas import deprecate_kwarg

import numpy as np
from scipy import stats

from statsmodels.distributions.copula.copulas import Copula
from statsmodels.tools.rng_qrng import check_random_state


class IndependenceCopula(Copula):
    r"""Independence copula.

    Copula with independent random variables.

    .. math::

        C_\theta(u,v) = uv

    Parameters
    ----------
    k_dim : int, optional
        Dimension, number of components in the multivariate random variable.

    Notes
    -----
    IndependenceCopula does not have copula parameters.
    If non-empty ``args`` are provided in methods, then a ValueError is raised.
    The ``args`` keyword is provided for a consistent interface across
    copulas.

    """

    def __init__(self, k_dim=2):
        super().__init__(k_dim=k_dim)

    def _handle_args(self, args):
        if args != () and args is not None:
            msg = "Independence copula does not use copula parameters."
            raise ValueError(msg)
        else:
            return args

    @deprecate_kwarg("random_state", "rng")
    def rvs(self, nobs=1, args=(), rng=None):
        """Generate random variates from the copula.

        Parameters
        ----------
        nobs : int, optional
            Number of samples to generate from the copula. Default is 1.
        args : tuple, optional
            Arguments for copula parameters. Not used by ``IndependenceCopula``.
        rng : int, array_like of int, numpy.random.Generator, or numpy.random.RandomState, optional
            If `rng` is None, a new ``Generator`` is created using fresh
            entropy from the operating system. If `rng` is an int or array
            of ints, a new ``Generator`` is created, seeded with `rng`. If
            `rng` is already a ``Generator`` or ``RandomState`` instance,
            that instance is used.
        rng : int, array_like of int, numpy.random.Generator, or numpy.random.RandomState, optional
            .. deprecated:: 0.15

               random_state has been deprecated. In-line with SPEC-007, use
               rng for passing a random number generator or seed.

        Returns
        -------
        sample : array_like (nobs, k_dim)
            Sample from the copula.
        """
        self._handle_args(args)
        rng = check_random_state(rng)
        x = rng.random((nobs, self.k_dim))
        return x

    def pdf(self, u, args=()):
        """Probability density function of the independence copula.

        Parameters
        ----------
        u : array_like, 2-D
            Points of random variables in unit hypercube at which method is
            evaluated.
            The second (or last) dimension should be the same as the
            dimension of the random variable, e.g., 2 for bivariate copula.
        args : tuple, optional
            Not used by ``IndependenceCopula``.

        Returns
        -------
        ndarray
            Copula pdf evaluated at points ``u``. Constant equal to 1.
        """
        u = np.asarray(u)
        return np.ones(u.shape[:-1])

    def cdf(self, u, args=()):
        """Cumulative distribution function of the independence copula.

        Parameters
        ----------
        u : array_like, 2-D
            Points of random variables in unit hypercube at which method is
            evaluated.
            The second (or last) dimension should be the same as the
            dimension of the random variable, e.g., 2 for bivariate copula.
        args : tuple, optional
            Not used by ``IndependenceCopula``.

        Returns
        -------
        ndarray
            Copula cdf evaluated at points ``u``, i.e., the product of the
            components of ``u``.
        """
        return np.prod(u, axis=-1)

    def tau(self):
        """Kendall's tau of the independence copula.

        Returns
        -------
        float
            Kendall's tau, which is always 0 for the independence copula.
        """
        return 0

    def plot_pdf(self, *args):
        """Not implemented.

        Raises
        ------
        NotImplementedError
            The independence copula's pdf is constant over the domain and
            is not plotted.
        """
        raise NotImplementedError("PDF is constant over the domain.")


def rvs_kernel(sample, size, bw=1, k_func=None, return_extras=False, rng=None):
    """Random sampling from empirical copula using Beta distribution

    Parameters
    ----------
    sample : ndarray
        Sample of multivariate observations in (0, 1) interval.
    size : int
        Number of observations to simulate.
    bw : float, optional
        Bandwidth for Beta sampling. The beta copula corresponds to a kernel
        estimate of the distribution. bw=1 corresponds to the empirical beta
        copula. A small bandwidth like bw=0.001 corresponds to small noise
        added to the empirical distribution. Larger bw, e.g., bw=10 corresponds
        to kernel estimate with more smoothing.
    k_func : callable, optional
        The default kernel function is currently a beta function with 1 added
        to the first beta parameter.
    return_extras : bool, optional
        If this is False, then only the random sample will be returned.
        If true, then extra information is returned that is mainly of interest
        for verification.
    rng : int, array_like of int, numpy.random.Generator, or numpy.random.RandomState, optional
        If `rng` is None, a new ``Generator`` is created using fresh
        entropy from the operating system. If `rng` is an int or array
        of ints, a new ``Generator`` is created, seeded with `rng`. If
        `rng` is already a ``Generator`` or ``RandomState`` instance,
        that instance is used.

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
    rng = check_random_state(rng)
    if k_func is None:

        def kfunc(x, bw):
            return _kernel_rvs_beta1(x, bw, rng=rng)

    else:
        kfunc = k_func
    if isinstance(rng, np.random.RandomState):
        idx = rng.randint(0, n, size=size)
    else:
        idx = rng.integers(0, n, size=size)
    xi = sample[idx]
    krvs = np.column_stack([kfunc(xii, bw) for xii in xi.T])

    if return_extras:
        return krvs, idx, xi
    else:
        return krvs


def _kernel_rvs_beta(x, bw, rng=None):
    # Beta kernel for density, pdf, estimation
    return stats.beta.rvs(x / bw + 1, (1 - x) / bw + 1, size=x.shape, random_state=rng)


def _kernel_rvs_beta1(x, bw, rng=None):
    # Beta kernel for density, pdf, estimation
    # Kiriliouk, Segers, Tsukuhara 2020 arxiv, using bandwith 1/nobs sample
    return stats.beta.rvs(x / bw, (1 - x) / bw + 1, random_state=rng)
