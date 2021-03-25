"""Joint Distribution using Copula.

PDF, CDF and theta formulas looked, and re-wrote, from `sdv-dev/Copulas
<https://github.com/sdv-dev/Copulas>`_ by
P.T. Roy, licensed under
    `MIT <https://github.com/sdv-dev/Copulas/blob/master/LICENSE>`_.

"""
from scipy._lib._util import check_random_state


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
