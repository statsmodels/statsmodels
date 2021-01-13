import numpy as np
from scipy import stats
from abc import ABC, abstractmethod


class Copula(ABC):
    r"""A generic Copula class meant for subclassing.

    Parameters
    ----------
    seed : {None, int, `numpy.random.Generator`}, optional
        If `seed` is None the `numpy.random.Generator` singleton is used.
        If `seed` is an int, a new ``Generator`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` instance then that instance is
        used.

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

    * ``__init__(seed=None)``: If the sampler
      does not take advantage of a ``seed``, this parameter can be omitted.
    * ``random(n)``: draw ``n`` from the copula.

    References
    ----------
    .. [1] Marshall AW, Olkin I. “Families of Multivariate Distributions”,
      Journal of the American Statistical Association, 83, 834–841, 1988.
    .. [2] Marius Hofert. "Sampling Archimedean copulas", Universität Ulm, 2008.
    .. [3] Harry Joe. "Dependence Modeling with Copulas", Monographs on
      Statistics and Applied Probability 134, 2015.

    """

    @abstractmethod
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def random(self, n):
        """Draw `n` in the half-open interval ``[0, 1)``.

        Marginals are uniformly distributed.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space.
            Default is 1.

        Returns
        -------
        sample : array_like (n, d)
            Sample from the copula.

        """


class IndependentCopula(Copula):
    """Independent copula."""

    def __init__(self, seed=None):
        super().__init__(seed=seed)

    def random(self, n):
        x = self.rng.random((n, 2))
        # v = ...
        return x  # np.exp(- (-np.log(x) / v))



class GaussianCopula(Copula):
    """Gaussian copula."""

    def __init__(self, cov=1, seed=None):
        super().__init__(seed=seed)
        self.density = stats.norm()
        self.mv_density = stats.multivariate_normal(cov=cov)

    def random(self, n):
        x = self.mv_density.rvs(size=n, random_state=self.rng)
        return self.density.cdf(x)


class StudentCopula(Copula):
    """Student copula."""

    def __init__(self, df=1, cov=1, seed=None):
        super().__init__(seed=seed)
        self.density = stats.t(df=df)
        self.mv_density = stats.multivariate_t(shape=cov, df=df)

    def random(self, n):
        x = self.mv_density.rvs(size=n, random_state=self.rng)
        return self.density.cdf(x)


class ClaytonCopula(Copula):
    """Clayton copula.

    Dependence is greater in the negative tail than in the positive.

    """
    def __init__(self, theta=1, seed=None):
        super().__init__(seed=seed)
        if theta <= -1 or theta == 0:
            raise ValueError('Theta must be > -1 and !=0')
        self.theta = theta

    def random(self, n):
        x = self.rng.random((n, 2))
        v = stats.gamma(1. / self.theta).rvs(size=(n, 1),
                                             random_state=self.rng)
        return (1 - np.log(x) / v)**(-1. / self.theta)


class FrankCopula(Copula):
    """Frank copula.

    Dependence is symmetric.

    """
    def __init__(self, theta=2, seed=None):
        super().__init__(seed=seed)
        if theta == 0:
            raise ValueError('Theta must be !=0')
        self.theta = theta

    def random(self, n):
        x = self.rng.random((n, 2))
        v = stats.logser.rvs(1. - np.exp(-self.theta), size=(n, 1),
                             random_state=self.rng)

        return -1. / self.theta * np.log(1.
                                         + np.exp(-(-np.log(x) / v))
                                         * (np.exp(-self.theta) - 1.))


class GumbelCopula(Copula):
    """Gumbel copula.

    Dependence is greater in the positive tail than in the negative.

    """
    def __init__(self, theta=2, seed=None):
        super().__init__(seed=seed)
        if theta <= 1:
            raise ValueError('Theta must be > 1')
        self.theta = theta

    def random(self, n):
        x = self.rng.random((n, 2))
        v = stats.levy_stable.rvs(1./self.theta, 1., 0,
                                  np.cos(np.pi / (2 * self.theta))**self.theta,
                                  size=(n, 1), random_state=self.rng)
        return np.exp(-(-np.log(x) / v)**(1. / self.theta))


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
    seed : {None, int, `numpy.random.Generator`}, optional
        If `seed` is None the `numpy.random.Generator` singleton is used.
        If `seed` is an int, a new ``Generator`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` instance then that instance is
        used.

    """
    def __init__(self, dists, copula=None, seed=None):
        self.rng = np.random.default_rng(seed)
        self.dists = dists
        self.copula = copula
        self.d = len(self.dists)

    def random(self, n):
        """Draw `n` in the half-open interval ``[0, 1)``.

        Sample the joint distribution.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space.
            Default is 1.

        Returns
        -------
        sample : array_like (n, d)
            Sample from the joint distribution.

        """
        if self.copula is None:
            # this means marginals are independents
            sample = self.rng.random((n, self.d))
        else:
            sample = self.copula.random(n)

        for i, dist in enumerate(self.dists):
            sample[:, i] = dist.ppf(0.5 + (1 - 1e-10) * (sample[:, i] - 0.5))
        return sample
