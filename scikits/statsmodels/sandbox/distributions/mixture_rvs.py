import numpy as np

def _make_index(prob,size):
    """
    Returns a boolean index for given probabilities.

    Notes
    ---------
    prob = [.75,.25] means that there is a 75% chance of the first column
    being True and a 25% chance of the second column being True. The
    columns are mutually exclusive.
    """
    rv = np.random.uniform(size=(size,1))
    cumprob = np.cumsum(prob)
    return np.logical_and(np.r_[0,cumprob[:-1]] <= rv, rv < cumprob)

def mixture_rvs(prob, size, dist, kwargs=None):
    """
    Sample from a mixture of distributions.

    Parameters
    ----------
    prob : array-like
        Probability of sampling from each distribution in dist
    size : int
        The length of the returned sample.
    dist : array-like
        An iterable of distributions objects from scipy.stats.
    kwargs : tuple of dicts, optional
        A tuple of dicts.  Each dict in kwargs can have keys loc, scale, and
        args to be passed to the respective distribution in dist.  If not
        provided, the distribution defaults are used.

    Examples
    --------
    Say we want 5000 random variables from mixture of normals with two
    distributions norm(-1,.5) and norm(1,.5) and we want to sample from the
    first with probability .75 and the second with probability .25.

    >>> from scipy import stats
    >>> prob = [.75,.25]
    >>> Y = mixture(prob, 5000, dist=[stats.norm, stats.norm], kwargs =
                (dict(loc=-1,scale=.5),dict(loc=1,scale=.5)))
    """
    if len(prob) != len(dist):
        raise ValueError("You must provide as many probabilities as distributions")
    if not np.allclose(np.sum(prob), 1):
        raise ValueError("prob does not sum to 1")

    if kwargs is None:
        kwargs = ({},)*len(prob)

    idx = _make_index(prob,size)
    sample = np.empty(size)
    for i in range(len(prob)):
        sample_idx = idx[...,i]
        sample_size = sample_idx.sum()
        loc = kwargs[i].get('loc',0)
        scale = kwargs[i].get('scale',1)
        args = kwargs[i].get('args',())
        sample[sample_idx] = dist[i].rvs(loc=loc,scale=scale,args=args,
            size=sample_size)
    return sample
