"""Joint Distribution using Copula.

PDF, CDF and theta formulas looked, and re-wrote, from `sdv-dev/Copulas
<https://github.com/sdv-dev/Copulas>`_ by
P.T. Roy, licensed under
    `MIT <https://github.com/sdv-dev/Copulas/blob/master/LICENSE>`_.

"""
import sys
from abc import ABC, abstractmethod

import numpy as np
from scipy import stats
from scipy._lib._util import check_random_state
import scipy.integrate as integrate
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

from statsmodels.graphics import utils


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

    @abstractmethod
    def __init__(self, theta):
        self.theta = theta

    @abstractmethod
    def random(self, n=1, random_state=None):
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
        sample : array_like (n, d)
            Sample from the copula.

        """

    @abstractmethod
    def pdf(self, x):
        """Probability density function."""

    def logpdf(self, x):
        """Log of the PDF."""
        return np.log(self.pdf(x))

    @abstractmethod
    def cdf(self, x):
        """Cumulative density function."""

    def plot(self, n, random_state=None, ax=None):
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
        sample = self.random(n=n, random_state=random_state)

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
        n_samples = 100

        xx, yy = np.meshgrid(np.linspace(0.0001, 1, n_samples),
                             np.linspace(0.0001, 1, n_samples))
        points = np.vstack([xx.ravel(), yy.ravel()]).T

        data = self.pdf(points).T.reshape(xx.shape)
        min_ = np.percentile(data, 5)
        max_ = np.percentile(data, 95)

        fig, ax = utils.create_mpl_ax(ax)

        vticks = np.linspace(min_, max_, num=ticks_nbr)
        range_cbar = [min_, max_]
        cs = ax.contourf(xx, yy, data, vticks,
                         antialiased=True, vmin=range_cbar[0],
                         vmax=range_cbar[1])

        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        cbar = plt.colorbar(cs, ticks=vticks)
        cbar.set_label('p')

        return fig

    @property
    def tau(self):
        """Empirical Kendall's tau.

        Returns
        -------
        tau : float
            Kendall's tau.

        """
        x = self.random(1024, random_state=0)
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


class IndependentCopula(Copula):
    """Independent copula.

    .. math::

        C_\theta(u,v) = uv

    """
    def __init__(self, d=2):
        self.d = d

    def random(self, n=1, random_state=None):
        rng = check_random_state(random_state)
        x = rng.random((n, self.d))
        return x

    def pdf(self, x):
        return np.ones((len(x), 1))

    def cdf(self, x):
        return np.prod(x, axis=1)

    def plot_pdf(self, *args):
        raise NotImplementedError("PDF is constant over the domain.")


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
        if cov is None:
            cov = [[1., 0.], [0., 1.]]
        self.density = stats.norm()
        self.mv_density = stats.multivariate_normal(cov=cov)

    def random(self, n=1, random_state=None):
        x = self.mv_density.rvs(size=n, random_state=random_state)
        return self.density.cdf(x)

    def pdf(self, x):
        return self.mv_density.pdf(x)

    def cdf(self, x):
        return self.mv_density.cdf(x)


class StudentCopula(Copula):
    """Student copula."""

    def __init__(self, df=1, cov=None):
        if cov is None:
            cov = [[1., 0.], [0., 1.]]
        self.density = stats.t(df=df)
        self.mv_density = stats.multivariate_t(shape=cov, df=df)

    def random(self, n=1, random_state=None):
        x = self.mv_density.rvs(size=n, random_state=random_state)
        return self.density.cdf(x)

    def pdf(self, x):
        return self.mv_density.pdf(x)

    def cdf(self, x):
        return self.mv_density.cdf(x)


class ClaytonCopula(Copula):
    r"""Clayton copula.

    Dependence is greater in the negative tail than in the positive.

    .. math::

        C_\theta(u,v) = \left[ \max\left\{ u^{-\theta} + v^{-\theta} -1 ;
        0 \right\} \right]^{-1/\theta}

    with :math:`\theta\in[-1,\infty)\backslash\{0\}`.

    """

    def __init__(self, theta=1):
        if theta <= -1 or theta == 0:
            raise ValueError('Theta must be > -1 and !=0')
        self.theta = theta

    def random(self, n=1, random_state=None):
        rng = check_random_state(random_state)
        x = rng.random((n, 2))
        v = stats.gamma(1. / self.theta).rvs(size=(n, 1), random_state=rng)
        return (1 - np.log(x) / v) ** (-1. / self.theta)

    def pdf(self, x):
        a = (self.theta + 1) * np.prod(x, axis=1) ** -(self.theta + 1)
        b = np.sum(x ** -self.theta, axis=1) - 1
        c = -(2 * self.theta + 1) / self.theta
        return a * b ** c

    def cdf(self, x):
        return (np.sum(x ** -self.theta, axis=1) - 1) ** -1.0 / self.theta

    def _theta_from_tau(self, tau):
        return 2 * tau / (1 - tau)


class FrankCopula(Copula):
    r"""Frank copula.

    Dependence is symmetric.

    .. math::

        C_\theta(u,v) = -\frac{1}{\theta} \log\!\left[ 1+
        \frac{(\exp(-\theta u)-1)(\exp(-\theta v)-1)}{\exp(-\theta)-1} \right]

    with :math:`\theta\in \mathbb{R}\backslash\{0\}`.

    """

    def __init__(self, theta=2):
        if theta == 0:
            raise ValueError('Theta must be !=0')
        self.theta = theta

    def random(self, n=1, random_state=None):
        rng = check_random_state(random_state)
        x = rng.random((n, 2))
        v = stats.logser.rvs(1. - np.exp(-self.theta),
                             size=(n, 1), random_state=rng)

        return -1. / self.theta * np.log(1.
                                         + np.exp(-(-np.log(x) / v))
                                         * (np.exp(-self.theta) - 1.))

    def pdf(self, x):
        g_ = np.exp(-self.theta * np.sum(x, axis=1)) - 1
        g1 = np.exp(-self.theta) - 1

        num = -self.theta * g1 * (1 + g_)
        aux = np.prod(np.exp(-self.theta * x) - 1, axis=1) + g1
        den = aux ** 2
        return num / den

    def cdf(self, x):
        num = np.prod(np.exp(np.prod(-self.theta * x, axis=1)) - 1, axis=1)
        den = np.exp(-self.theta) - 1

        return -1.0 / self.theta * np.log(1 + num / den)

    def _theta_from_tau(self, tau):
        MIN_FLOAT_LOG = np.log(sys.float_info.min)
        MAX_FLOAT_LOG = np.log(sys.float_info.max)
        EPSILON = np.finfo(np.float32).eps

        def _theta_from_tau(alpha):
            def debye(t):
                return t / (np.exp(t) - 1)

            debye_value = integrate.quad(debye, EPSILON, alpha)[0] / alpha
            return 4 * (debye_value - 1) / alpha + 1 - tau

        result = least_squares(_theta_from_tau, 1, bounds=(MIN_FLOAT_LOG,
                                                           MAX_FLOAT_LOG))
        self.theta = result.x[0]
        return self.theta


class GumbelCopula(Copula):
    r"""Gumbel copula.

    Dependence is greater in the positive tail than in the negative.

    .. math::

        C_\theta(u,v) = \exp\!\left[ -\left( (-\log(u))^\theta +
        (-\log(v))^\theta \right)^{1/\theta} \right]

    with :math:`\theta\in[1,\infty)`.

    """

    def __init__(self, theta=2):
        if theta <= 1:
            raise ValueError('Theta must be > 1')
        self.theta = theta

    def random(self, n=1, random_state=None):
        rng = check_random_state(random_state)
        x = rng.random((n, 2))
        v = stats.levy_stable.rvs(
            1. / self.theta, 1., 0,
            np.cos(np.pi / (2 * self.theta)) ** self.theta,
            size=(n, 1), random_state=rng
        )
        return np.exp(-(-np.log(x) / v) ** (1. / self.theta))

    def pdf(self, x):
        prod_ = np.prod(x, axis=1)
        a = prod_ ** -1
        tmp = np.sum((-np.log(x)) ** self.theta, axis=1)
        b = tmp ** (-2 + 2.0 / self.theta)
        c = prod_ ** (self.theta - 1)
        d = 1 + (self.theta - 1) * tmp ** (-1.0 / self.theta)

        return self.cdf(x) * a * b * c * d

    def cdf(self, x):
        h = - np.sum(-np.log(x) ** self.theta, axis=1)
        cdf = np.exp(h ** (1.0 / self.theta))
        return cdf

    def _theta_from_tau(self, tau):
        return 1 / (1 - tau)


class JointDistribution:
    """Construct a joint distribution.

    Parameters
    ----------
    dists : list(distributions) (d,)
        List of univariate distribution. With ``d`` the
        number of variables. Distributions must implement
        the inverse CDF function as ``ppf``.
    copula : Copula, optional
        A copula. It must implement a ``random`` function to sample from
        the copula/distribution.

    """

    def __init__(self, dists, copula=None):
        self.dists = dists
        self.copula = copula
        self.d = len(self.dists)

    def random(self, n=1, random_state=None):
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
            sample = rng.random((n, self.d))
        else:
            sample = self.copula.random(n, random_state=random_state)

        for i, dist in enumerate(self.dists):
            sample[:, i] = dist.ppf(0.5 + (1 - 1e-10) * (sample[:, i] - 0.5))
        return sample
