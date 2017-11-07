"""Latin Hypercube Sampling methods."""
from __future__ import division
import numpy as np
from scipy.optimize import brute, fmin
try:
    from scipy.optimize import basinhopping
    have_basinhopping = True
except ImportError:
    have_basinhopping = False
from .sequences import discrepancy


def olhs(dim, n_sample, bounds=None):
    """Orthogonal array-based Latin hypercube sampling (OA-LHS).

    On top of the constrain from the Latin Hypercube, an orthogonal array of
    size n_sample is defined and only one point is allowed per subspace.

    Parameters
    ----------
    dim : int
        Dimension of the parameter space.
    n_sample : int
        Number of samples to generate in the parametr space.
    bounds : tuple or array_like ([min, k_vars], [max, k_vars])
        Desired range of transformed data. The transformation apply the bounds
        on the sample and not the theoretical space, unit cube. Thus min and
        max values of the sample will coincide with the bounds.

    Returns
    -------
    sample : array_like (n_samples, k_vars)
        Latin hypercube Sampling.

    References
    ----------
    [1] Art B. Owen, "Orthogonal arrays for computer experiments, integration
    and visualization", Statistica Sinica, 1992.

    """
    sample = []
    step = 1.0 / n_sample

    for i in range(dim):
        # Enforce a unique point per grid
        temp = [np.random.uniform(low=j * step, high=(j + 1) * step)
                for j in range(n_sample)]
        np.random.shuffle(temp)

        sample.append(temp)

    sample = np.array(sample).T

    # Sample scaling from unit hypercube to feature range
    if bounds is not None:
        min_ = bounds.min(axis=0)
        max_ = bounds.max(axis=0)
        sample = sample * (max_ - min_) + min_

    return sample


def lhs(dim, n_sample, bounds=None, centered=False):
    """Latin hypercube sampling (LHS).

    The parameter space is subdivided into an orthogonal grid of n_sample per
    dimension. Within this multi-dimensional grid, n_sample are selected by
    ensuring there is only one sample per row and column.

    Parameters
    ----------
    dim : int
        Dimension of the parameter space.
    n_sample : int
        Number of samples to generate in the parametr space.
    bounds : tuple or array_like ([min, k_vars], [max, k_vars])
        Desired range of transformed data. The transformation apply the bounds
        on the sample and not the theoretical space, unit cube. Thus min and
        max values of the sample will coincide with the bounds.

    Returns
    -------
    sample : array_like (n_samples, k_vars)
        Latin hypercube Sampling.

    References
    ----------
    [1] Mckay et al., "A Comparison of Three Methods for Selecting Values of
    Input Variables in the Analysis of Output from a Computer Code",
    Technometrics, 1979.

    """
    if centered:
        r = 0.5
    else:
        r = np.random.random_sample((n_sample, dim))

    q = np.random.random_integers(low=1, high=n_sample, size=(n_sample, dim))

    sample = 1. / n_sample * (q - r)

    # Sample scaling from unit hypercube to feature range
    if bounds is not None:
        min_ = bounds.min(axis=0)
        max_ = bounds.max(axis=0)
        sample = sample * (max_ - min_) + min_

    return sample


def optimal_design(dim, n_sample, bounds=None, doe=None, niter=1, force=False):
    """Optimal design.

    Optimize the design by doing random permutations to lower the centered
    discrepancy.

    Centered discrepancy based design show better space filling robustness
    toward 2D and 3D subprojections. Distance based design better space filling
    but less robust to subprojections.

    Parameters
    ----------
    dim : int
        Dimension of the parameter space.
    n_sample : int
        Number of samples to generate in the parametr space.
    bounds : tuple or array_like ([min, k_vars], [max, k_vars])
        Desired range of transformed data. The transformation apply the bounds
        on the sample and not the theoretical space, unit cube. Thus min and
        max values of the sample will coincide with the bounds.
    doe : array_like (n_samples, k_vars)
        Initial design of experiment to optimize.

    Returns
    -------
    sample : array_like (n_samples, k_vars)
        Optimal Latin hypercube Sampling.

    References
    ----------
    [1] Damblin et al., "Numerical studies of space filling designs:
    optimization of Latin Hypercube Samples and subprojection properties",
    Journal of Simulation, 2013.

    """
    if doe is None:
        doe = olhs(dim, n_sample, bounds)

    def _disturb_doe(x, sample, bounds):
        """Disturbe the Design of Experiment.

        Parameters
        ----------
        x : list
            It is a list of:
                idx : int
                    Index value of the components to compute
        sample : array_like (n_samples, k_vars)
            Sample to perturbe.
        bounds : tuple or array_like ([min, k_vars], [max, k_vars])
            Desired range of transformed data. The transformation apply the
            bounds on the sample and not the theoretical space, unit cube. Thus
            min and max values of the sample will coincide with the bounds.

        Returns
        -------
        discrepancy : float
            Centered discrepancy.

        """
        col, row_1, row_2 = map(int, x)
        doe[row_1, col], doe[row_2, col] = doe[row_2, col], doe[row_1, col]

        return discrepancy(sample, bounds)

    # Total number of possible design
    complexity = dim * n_sample ** 2

    if have_basinhopping and ((complexity > 1e6) or force):
        bounds_optim = ([0, dim - 1], [0, n_sample - 1], [0, n_sample - 1])
    else:
        bounds_optim = (slice(0, dim - 1, 1), slice(0, n_sample - 1, 1),
                        slice(0, n_sample - 1, 1))

    for n in range(niter):
        if have_basinhopping and ((complexity > 1e6) or force):
            minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds_optim,
                                "args": (doe, bounds)}
            optimum = basinhopping(_disturb_doe, [0, 0, 0], niter=100,
                                   minimizer_kwargs=minimizer_kwargs).x
        else:
            optimum = brute(_disturb_doe, ranges=bounds_optim,
                            finish=fmin, args=(doe, bounds))

        col, row_1, row_2 = map(int, optimum)
        doe[row_1, col], doe[row_2, col] = doe[row_2, col], doe[row_1, col]

    return doe
