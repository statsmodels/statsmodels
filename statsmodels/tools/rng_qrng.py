"""Random number generator helpers"""

import warnings

import numpy as np
from scipy import stats

from statsmodels.tools.validation import is_int_like


def check_random_state(seed=None, deprecated=False, warn=True):
    """
    Turn a seed into a random number generator

    Parameters
    ----------
    seed : int, array_like of int, numpy.random.Generator, numpy.random.RandomState, scipy.stats.qmc.QMCEngine, optional
        If `seed` is None fresh, unpredictable entropy will be pulled from the
        OS and `numpy.random.Generator` is used.
        If `seed` is an int or ``array_like[ints]``, a new ``Generator`` instance
        is used, seeded with `seed`.
        If `seed` is already a ``Generator``, ``RandomState`` or
        `scipy.stats.qmc.QMCEngine` instance then that instance is used.
    deprecated : bool, optional
        If False, returns default_rng(seed). If True, returns RandomState(seed)
        when seed is an int or array-like of ints.
    warn : bool, optional
        Whether to issue a warning that the future behavior for integer or
        array-like seed will switch to calling default_rng(seed).

    Returns
    -------
    rng : numpy.random.Generator, numpy.random.RandomState, or scipy.stats.qmc.QMCEngine
        Random number generator.

    Notes
    -----
    `scipy.stats.qmc.QMCEngine` requires SciPy >=1.7. It also means that the
    generator only has the method ``random``.

    .. versionchanged:: 0.16.0

       An integer seed is checked with
       :func:`~statsmodels.tools.validation.is_int_like` (``strict=True``).
       Any other scalar that ``int`` converts, such as a float (``2.5`` was
       the seed 2), a string such as ``"3"`` or a bool, is deprecated: it
       gives the same generator as before with a ``FutureWarning`` and will
       raise a ``TypeError`` in statsmodels 0.17.
    """
    if hasattr(stats, "qmc") and isinstance(seed, stats.qmc.QMCEngine):
        return seed
    elif isinstance(seed, (np.random.RandomState, np.random.Generator)):
        return seed
    elif seed is not None:
        msg = (
            "When creating a random number generator from a value, the "
            "seed must either be an integer or array-like of ints"
        )
        if is_int_like(seed, strict=True):
            seed = int(seed)
        elif np.ndim(seed) == 0:
            # a scalar that is not an integer: int() used to convert it
            try:
                legacy = int(seed)
            except (TypeError, ValueError):
                raise TypeError(msg) from None
            warnings.warn(
                f"Passing seed of type {type(seed).__name__} is deprecated and "
                f"will raise a TypeError in statsmodels 0.17. Pass seed={legacy} "
                "as an int instead.",
                FutureWarning,
                stacklevel=2,
            )
            seed = legacy
        else:
            seed = np.asarray(seed)
            if not np.issubdtype(seed.dtype, np.integer):
                raise TypeError(msg)
        if deprecated:
            if warn:
                warnings.warn(
                    "After statsmodels 0.15 is released, passing an integer when "
                    "creating a random number generator will pass the value to "
                    "np.random.default_rng, rather than the current behavior of passing "
                    "it to np.random.RandomState. To continue using RandomState, directly "
                    "pass a RandomState instance.",
                    FutureWarning,
                    stacklevel=2,
                )
            return np.random.RandomState(seed)
        else:
            return np.random.default_rng(seed)
    else:
        return np.random.default_rng()
