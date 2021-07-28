import numbers

import numpy as np
import scipy.stats as stats


def check_random_state(seed=None):
    """Turn `seed` into a random number generator.

    Parameters
    ----------
    seed : {None, int, `numpy.random.Generator`,
            `numpy.random.RandomState`, `scipy.stats.qmc.QMCEngine`}, optional

        If `seed` is None fresh, unpredictable entropy will be pulled
        from the OS and `numpy.random.Generator` is used.
        If `seed` is an int, a new ``Generator`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator``, ``RandomState`` or
        `scipy.stats.qmc.QMCEngine` instance then
        that instance is used.

        `scipy.stats.qmc.QMCEngine` requires SciPy >=1.7. It also means
        that the generator only have the method ``random``.

    Returns
    -------
    seed : {`numpy.random.Generator`, `numpy.random.RandomState`}
        Random number generator.

    """
    if seed is None or isinstance(seed, (numbers.Integral, np.integer)):
        if not hasattr(np.random, 'Generator'):
            # This can be removed once numpy 1.16 is dropped
            msg = ("NumPy 1.16 doesn't have Generator, use either "
                   "NumPy >= 1.17 or `seed=np.random.RandomState(seed)`")
            raise ValueError(msg)
        return np.random.default_rng(seed)
    elif isinstance(seed, np.random.RandomState):
        return seed
    elif isinstance(seed, np.random.Generator):
        # The two checks can be merged once numpy 1.16 is dropped
        return seed
    elif isinstance(seed, np.random.Generator):
        return seed
    elif hasattr(stats.qmc, 'QMCEngine') and isinstance(seed, stats.qmc.QMCEngine):
        return seed
    else:
        raise ValueError('%r cannot be used to seed a numpy.random.Generator'
                         ' instance' % seed)
