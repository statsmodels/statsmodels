'''

Which Archimedean is Best?
Extreme Value copulas formulas are based on Genest 2009

References
----------

Genest, C., 2009. Rank-based inference for bivariate extreme-value
copulas. The Annals of Statistics, 37(5), pp.2990-3022.

'''
from abc import ABC, abstractmethod

import numpy as np
from scipy import stats
from scipy.special import expm1
from scipy._lib._util import check_random_state

from statsmodels.graphics import utils


def copula_bv_indep(u, v):
    '''independent bivariate copula
    '''
    return u*v


def copula_bv_min(u, v):
    '''comonotonic bivariate copula
    '''
    return np.minimum(u, v)


def copula_bv_max(u, v):
    '''countermonotonic bivariate copula
    '''
    return np.maximum(u + v - 1, 0)


def copula_bv_clayton(u, theta):
    '''Clayton or Cook, Johnson bivariate copula
    '''
    u, v = u
    if isinstance(theta, tuple) and len(theta) == 1:
        theta = theta[0]

    if not theta > 0:
        raise ValueError('theta needs to be strictly positive')

    return np.power(np.power(u, -theta) + np.power(v, -theta) - 1, -theta)


def copula_bv_frank(u, v, theta):
    '''Cook, Johnson bivariate copula
    '''
    if not theta > 0:
        raise ValueError('theta needs to be strictly positive')
    cdfv = -np.log(1 + expm1(-theta*u) * expm1(-theta*v) / expm1(-theta))/theta
    cdfv = np.minimum(cdfv, 1)  # necessary for example if theta=100
    return cdfv


def copula_bv_gauss(u, v, rho):
    raise NotImplementedError


def copula_bv_t(u, v, rho, df):
    raise NotImplementedError


# Archimedean Copulas through generator functions
# ===============================================


def copula_bv_archimedean(u, v, transform, args=()):
    '''
    '''
    phi = transform.evaluate
    phi_inv = transform.inverse
    cdfv = phi_inv(phi(u, *args) + phi(v, *args), *args)
    return cdfv


def copula_mv_archimedean(u, transform, args=(), axis=-1):
    '''generic multivariate Archimedean copula
    '''
    phi = transform.evaluate
    phi_inv = transform.inverse
    cdfv = phi_inv(phi(u, *args).sum(axis), *args)
    return cdfv


def copula_power_mv_archimedean(u, transform, alpha, beta, args=(), axis=-1):
    '''generic multivariate Archimedean copula with additional power transforms

    Nelson p.144, equ. 4.5.2
    '''

    def phi(u, alpha, beta, args=()):
        return np.power(transform.evaluate(np.power(u, alpha), *args), beta)

    def phi_inv(t, alpha, beta, args=()):
        return np.power(transform.evaluate(np.power(t, 1. / beta), *args),
                        1. / alpha)

    cdfv = phi_inv(phi(u, *args).sum(axis), *args)
    return cdfv


# ==========================================================================

# define dictionary of copulas by names and aliases
copulanamesbv = {'indep': copula_bv_indep,
                 'i': copula_bv_indep,
                 'min': copula_bv_min,
                 'max': copula_bv_max,
                 'clayton': copula_bv_clayton,
                 'cookjohnson': copula_bv_clayton,
                 'cj': copula_bv_clayton,
                 'frank': copula_bv_frank,
                 'gauss': copula_bv_gauss,
                 'normal': copula_bv_gauss,
                 't': copula_bv_t}


class CopulaDistributionBivariate:
    """Bivariate copula.

    might be obsolete,
    see CopulaDistribution for multivariate distribution class

    Instantiation needs the arguments, cop_args, that are required for copula
    """
    def __init__(self, marginalcdfs, copula, copargs=()):
        if copula in copulanamesbv:
            self.copula = copulanamesbv[copula]
        else:
            self.copula = copula

        # no checking done on marginals
        self.marginalcdfs = marginalcdfs
        self.copargs = copargs

    def cdf(self, xy, args=None):
        '''xx needs to be iterable, instead of x,y for extension to multivariate
        '''
        x, y = xy
        if args is None:
            args = self.copargs
        return self.copula(self.marginalcdfs[0](x), self.marginalcdfs[1](y),
                           *args)


class CopulaDistribution:
    """Multivariate copula.

    Instantiation needs the arguments, cop_args, that are required for copula

    Parameters
    ----------
    marginals : list of distribution instances
        Marginal distributions.
    copula : instance of copula class
    copargs : tuple
        Parameters for copula

    Notes
    -----
    experimental, argument handling not yet finalized

    """
    def __init__(self, marginals, copula, copargs=()):
        if copula in copulanamesbv:
            self.copula = copulanamesbv[copula]
        else:
            # assume it's an appropriate copula class
            self.copula = copula

        # no checking done on marginals
        self.marginals = marginals
        self.copargs = copargs
        self.k_vars = len(marginals)

    def random(self, nobs=1, random_state=None):
        """Draw `n` in the half-open interval ``[0, 1)``.

        Sample the joint distribution.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space.
            Default is 1.
        random_state : {None, int, `numpy.random.Generator`}, optional
            If `seed` is None the `numpy.random.Generator` singleton is used.
            If `seed` is an int, a new ``Generator`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` instance then that instance is
            used.

        Returns
        -------
        sample : array_like (n, d)
            Sample from the joint distribution.

        """
        rng = check_random_state(random_state)
        if self.copula is None:
            # this means marginals are independents
            sample = rng.random((nobs, len(self.marginals)))
        else:
            sample = self.copula.random(nobs, random_state=random_state)

        for i, dist in enumerate(self.marginals):
            sample[:, i] = dist.ppf(0.5 + (1 - 1e-10) * (sample[:, i] - 0.5))
        return sample

    def cdf(self, y, args=None):
        """CDF of copula distribution.

        Parameters
        ----------
        y : array_like
            Values of random variable at which to evaluate cdf.
            If 2-dimensional, then components of multivariate random variable
            need to be in columns
        args : tuple
            Copula parameters.
            Warning: interface for parameters will still change.

        Returns
        -------
        cdf values

        """
        y = np.asarray(y)
        if args is None:
            args = self.copargs

        cdf_marg = []
        for i in range(self.k_vars):
            cdf_marg.append(self.marginals[i].cdf(y[..., i]))

        u = np.column_stack(cdf_marg)
        if y.ndim == 1:
            u = u.squeeze()
        return self.copula.cdf(u, args)

    def pdf(self, y, args=None):
        """PDF of copula distribution.

        Parameters
        ----------
        y : array_like
            Values of random variable at which to evaluate cdf.
            If 2-dimensional, then components of multivariate random variable
            need to be in columns
        args : tuple
            Copula parameters.
            Warning: interface for parameters will still change.

        Returns
        -------
        pdf values
        """
        return np.exp(self.logpdf(y, args=args))

    def logpdf(self, y, args=None):
        """Log-pdf of copula distribution.

        Parameters
        ----------
        y : array_like
            Values of random variable at which to evaluate cdf.
            If 2-dimensional, then components of multivariate random variable
            need to be in columns
        args : tuple
            Copula parameters.
            Warning: interface for parameters will still change.

        Returns
        -------
        log-pdf values

        """
        y = np.asarray(y)
        if args is None:
            args = self.copargs

        lpdf = 0.0
        cdf_marg = []
        for i in range(self.k_vars):
            lpdf += self.marginals[i].logpdf(y[..., i])
            cdf_marg.append(self.marginals[i].cdf(y[..., i]))

        u = np.column_stack(cdf_marg)
        if y.ndim == 1:
            u = u.squeeze()

        lpdf += self.copula.logpdf(u, args)
        return lpdf


class Copula(ABC):
    r"""A generic Copula class meant for subclassing.

    Notes
    -----
    A function :math:`\phi` on :math:`[0, \infty]` is the Laplace-Stieltjes
    transform of a distribution function if and only if :math:`\phi` is
    completely monotone and :math:`\phi(0) = 1` [2]_.

    The following algorithm for sampling a ``d``-dimensional exchangeable
    Archimedean copula with generator :math:`\phi` is due to Marshall, Olkin
    (1988) [1]_, where :math:`LS^{−1}(\phi)` denotes the inverse
    Laplace-Stieltjes transform of :math:`\phi`.

    From a mixture representation with respect to :math:`F`, the following
    algorithm may be derived for sampling Archimedean copulas, see [1]_.

    1. Sample :math:`V \sim F = LS^{−1}(\phi)`.
    2. Sample i.i.d. :math:`X_i \sim U[0,1], i \in \{1,...,d\}`.
    3. Return:math:`(U_1,..., U_d)`, where :math:`U_i = \phi(−\log(X_i)/V), i
       \in \{1, ...,d\}`.

    Detailed properties of each copula can be found in [3]_.

    Instances of the class can access the attributes: ``rng`` for the random
    number generator (used for the ``seed``).

    **Subclassing**

    When subclassing `Copula` to create a new copula, ``__init__`` and
    ``random`` must be redefined.

    * ``__init__(theta)``: If the copula
      does not take advantage of a ``theta``, this parameter can be omitted.
    * ``random(n, random_state)``: draw ``n`` from the copula.
    * ``pdf(x)``: PDF from the copula.
    * ``cdf(x)``: CDF from the copula.

    References
    ----------
    .. [1] Marshall AW, Olkin I. “Families of Multivariate Distributions”,
      Journal of the American Statistical Association, 83, 834–841, 1988.
    .. [2] Marius Hofert. "Sampling Archimedean copulas",
      Universität Ulm, 2008.
    .. [3] Harry Joe. "Dependence Modeling with Copulas", Monographs on
      Statistics and Applied Probability 134, 2015.

    """

    def __init__(self, d):
        self.d = d

    def random(self, nobs=1, random_state=None):
        """Draw `n` in the half-open interval ``[0, 1)``.

        Marginals are uniformly distributed.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate from the copula. Default is 1.
        random_state : {None, int, `numpy.random.Generator`}, optional
            If `seed` is None the `numpy.random.Generator` singleton is used.
            If `seed` is an int, a new ``Generator`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` instance then that instance is
            used.

        Returns
        -------
        sample : array_like (nobs, d)
            Sample from the copula.

        """
        raise NotImplementedError

    @abstractmethod
    def pdf(self, u):
        """Probability density function."""

    def logpdf(self, u):
        """Log of the PDF."""
        return np.log(self.pdf(u))

    @abstractmethod
    def cdf(self, u):
        """Cumulative density function."""

    def plot_scatter(self, sample=None, nobs=None, random_state=None, ax=None):
        """Sample the copula and plot.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate from the copula. Default is 1.
        random_state : {None, int, `numpy.random.Generator`}, optional
            If `seed` is None the `numpy.random.Generator` singleton is used.
            If `seed` is an int, a new ``Generator`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` instance then that instance is
            used.
        ax : AxesSubplot, optional
            If given, this subplot is used to plot in instead of a new figure
            being created.

        Returns
        -------
        fig : Figure
            If `ax` is None, the created figure.  Otherwise the figure to which
            `ax` is connected.
        sample : array_like (n, d)
            Sample from the copula.

        """
        if self.d != 2:
            raise ValueError("Can only plot 2-dimensional Copula.")

        if sample is None:
            sample = self.random(nobs=nobs, random_state=random_state)

        fig, ax = utils.create_mpl_ax(ax)
        ax.scatter(sample[:, 0], sample[:, 1])
        ax.set_xlabel('u')
        ax.set_ylabel('v')

        return fig, sample

    def plot_pdf(self, ticks_nbr=10, ax=None):
        """Plot the PDF.

        Parameters
        ----------
        ticks_nbr : int, optional
            Number of color isolines for the PDF. Default is 10.
        ax : AxesSubplot, optional
            If given, this subplot is used to plot in instead of a new figure
            being created.

        Returns
        -------
        fig : Figure
            If `ax` is None, the created figure.  Otherwise the figure to which
            `ax` is connected.

        """
        from matplotlib import pyplot as plt
        if self.d != 2:
            import warnings
            warnings.warn("Plotting 2-dimensional Copula.")

        n_samples = 100

        eps = 1e-4
        uu, vv = np.meshgrid(np.linspace(eps, 1 - eps, n_samples),
                             np.linspace(eps, 1 - eps, n_samples))
        points = np.vstack([uu.ravel(), vv.ravel()]).T

        data = self.pdf(points).T.reshape(uu.shape)
        min_ = np.nanpercentile(data, 5)
        max_ = np.nanpercentile(data, 95)

        fig, ax = utils.create_mpl_ax(ax)

        vticks = np.linspace(min_, max_, num=ticks_nbr)
        range_cbar = [min_, max_]
        cs = ax.contourf(uu, vv, data, vticks,
                         antialiased=True, vmin=range_cbar[0],
                         vmax=range_cbar[1])

        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        cbar = plt.colorbar(cs, ticks=vticks)
        cbar.set_label('p')
        fig.tight_layout()

        return fig

    def tau_simulated(self, nobs=1024, random_state=None):
        """Empirical Kendall's tau.

        Returns
        -------
        tau : float
            Kendall's tau.

        """
        x = self.random(nobs, random_state=random_state)
        return stats.kendalltau(x[:, 0], x[:, 1])[0]

    def fit_theta(self, x):
        """Compute ``theta`` using empirical Kendall's tau on sample data.

        Parameters
        ----------
        x : array_like
            Sample data used to fit `theta` using Kendall's tau.

        Returns
        -------
        theta : float
            Theta.

        """
        tau = stats.kendalltau(x[:, 0], x[:, 1])[0]
        self.theta = self._theta_from_tau(tau)
        return self.theta

    def _theta_from_tau(self, tau):
        """Compute ``theta`` from tau.

        Parameters
        ----------
        tau : float
            Kendall's tau.

        Returns
        -------
        theta : float
            Theta.

        """
        raise NotImplementedError
