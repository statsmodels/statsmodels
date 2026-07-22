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
    The BaseCrossValidator class is a base class for all the iterators that
    split the data in train and test as for example KFolds or LeavePOut
    """

    def __init__(self):
        pass

    @abstractmethod
    def split(self):
        pass


class KFold(BaseCrossValidator):
    """
    K-Folds cross validation iterator:
    Provides train/test indexes to split data in train test sets

    Parameters
    ----------
    k: int
        number of folds
    shuffle : bool
        If true, then the index is shuffled before splitting into train and
        test indices.
    rng : int, np.random.RandomState or np.random.Generator, optional
        The rng to use during KFold cross-validation if shuffling. If None, uses
        the singleton RandomState provided by NumPy. If an int, uses the
        ``default_rng``. If a RandomState instance or a Generator instance,
        uses this instance.

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
        """yield index split into train and test sets"""
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
