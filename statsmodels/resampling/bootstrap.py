"""
Bootstrap resampler

Example use: generating confidence intervals for the median of a dataset:

>>> import statsmodels.api as sm
>>> import numpy as np
>>> data = sm.datasets.engel.load_pandas().data
>>> income = data['income'].values
>>> (lower, upper) = bootstrap_confidence_interval(income,
                                                   np.median,
                                                   1000,
                                                   alpha=0.05)
>>> assert lower < np.median(income) < upper
"""


import numpy as np


def bootstrap_sampler(original_dataset):
    """
    A generator for simple independent bootstrap samples, with replacement,
    from the original dataset, with the same number of observations
    as in the original dataset.
    """
    n = len(original_dataset)
    while True:
        indices = np.random.randint(n, size=n)
        yield original_dataset[indices]


def bootstrap_confidence_interval(data,
                                  statistic,
                                  num_samples=1000,
                                  alpha=0.05):
    """
    Estimates a 100.0*(1-alpha)% confidence interval for the given
    statistic (a function) of the data.

    Returns a tuple
        (lower, upper)
    representing the confidence interval.

    Uses the percentile Bootstrap method [Efron and Tibshirani (1993, equ 13.5
    p. 171)] with num_samples independent samples from the given data.
    """
    assert 0 < alpha < 1
    sampler = bootstrap_sampler(data)
    stat = np.sort([statistic(next(sampler))
                    for _ in range(num_samples)])
    return (stat[int((alpha/2.0)*num_samples)],
            stat[int((1-alpha/2.0)*num_samples)])

