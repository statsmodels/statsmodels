"""
Cross-validation iterators for GAM

Author: Luca Puggini

"""

from statsmodels.compat.python import with_metaclass

from abc import ABCMeta, abstractmethod

import numpy as np

from statsmodels.tools.rng_qrng import check_random_state


class BaseCrossValidator(with_metaclass(ABCMeta)):
    """
    Base class for cross-validation iterators

    Subclasses split the data into train and test sets, for example
    ``KFold`` or ``LeavePOut``.
    """

    def __init__(self):
        pass

    @abstractmethod
    def split(self):
        pass


class KFold(BaseCrossValidator):
    """
    K-Folds cross validation iterator

    Provides train/test indexes to split data in train and test sets.

    Parameters
    ----------
    k_folds : int
        Number of folds.
    shuffle : bool
        If true, then the index is shuffled before splitting into train and
        test indices.
    rng : {None, int, array_like[int], numpy.random.Generator, numpy.random.RandomState}, optional
        If `rng` is None, a new ``Generator`` is created using fresh
        entropy from the operating system. If `rng` is an int or array
        of ints, a new ``Generator`` is created, seeded with `rng`. If
        `rng` is already a ``Generator`` or ``RandomState`` instance,
        that instance is used.

    Notes
    -----
    All folds except for last fold have size trunc(n/k), the last fold has
    the remainder.
    """

    def __init__(self, k_folds, shuffle=False, rng=None):
        self.nobs = None
        self.k_folds = k_folds
        self.shuffle = shuffle
        self.rng = check_random_state(rng)

    def split(self, X, y=None, label=None):
        """
        Yield index splits into train and test sets

        Parameters
        ----------
        X : array_like
            Data used only to determine the number of observations.
        y : array_like, optional
            Unused, present for signature compatibility.
        label : array_like, optional
            Unused, present for signature compatibility.

        Yields
        ------
        train_index : ndarray
            Boolean index array selecting the training observations.
        test_index : ndarray
            Boolean index array selecting the test observations.
        """
        # TODO: X and y are redundant, we only need nobs

        nobs = X.shape[0]
        index = np.array(range(nobs))

        if self.shuffle:
            self.rng.shuffle(index)

        folds = np.array_split(index, self.k_folds)
        for fold in folds:
            test_index = np.zeros(nobs, dtype=bool)
            test_index[fold] = True
            train_index = np.logical_not(test_index)
            yield train_index, test_index
