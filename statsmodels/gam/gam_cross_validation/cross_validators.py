# -*- coding: utf-8 -*-
"""
Cross-validation iterators for GAM

Author: Luca Puggini

"""

from __future__ import division

from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import numpy as np


class BaseCrossValidator(with_metaclass(ABCMeta)):
    """
    The BaseCrossValidator class is a base class for all the iterators that
    split the data in train and test as for example KFolds or LeavePOut
    """
    def __init__(self):

        return

    @abstractmethod
    def split(self):

        return


class KFold(BaseCrossValidator):
    """
    K-Folds cross validation iterator:
    Provides train/test indexes to split data in train test sets
    """

    def __init__(self, k_folds, shuffle=False):
        """
        K-Folds cross validation iterator:
        Provides train/test indexes to split data in train test sets

        Parameters
        ----------
        k: int
            number of folds

        Examples
        --------

        Notes
        -----
        All the folds have size trunc(n/k), the last one has the complementary
        """

        self.n_samples = None
        self.k_folds = k_folds
        self.shuffle = shuffle
        return

    def split(self, X, y=None, label=None):

        n_samples = X.shape[0]
        index = np.array(range(n_samples))

        if self.shuffle:
            np.random.shuffle(index)

        folds = np.array_split(index, self.k_folds)
        for fold in folds:
            test_index = np.array([False]*n_samples)
            test_index[fold] = True
            train_index = np.logical_not(test_index)
            yield train_index, test_index
