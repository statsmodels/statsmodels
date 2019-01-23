# -*- coding: utf-8 -*-
"""
Cross-validation classes for GAM

Author: Luca Puggini

"""

from __future__ import division
from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import itertools
import numpy as np
from statsmodels.gam.smooth_basis import (GenericSmoothers,
                                          UnivariateGenericSmoother)

try:
    import matplotlib.pyplot as plt
    have_matplotlib = True
except ImportError:
    have_matplotlib = False


class BaseCV(with_metaclass(ABCMeta)):
    """
    BaseCV class. It computes the cross validation error of a given model.
    All the cross validation classes can be derived by this one
    (e.g. GamCV, LassoCV,...)
    """

    def __init__(self, cv, x, y):
        self.cv = cv
        self.x = x
        self.y = y
        self.train_test_cv_indices = self.cv.split(self.x, self.y, label=None)

    def fit(self, **kwargs):
        # kwargs are the input values for the fit method of the
        # cross-validated object

        cv_err = []

        for train_index, test_index in self.train_test_cv_indices:
            cv_err.append(self._error(train_index, test_index, **kwargs))

        return np.array(cv_err)

    @abstractmethod
    def _error(self, train_index, test_index, **kwargs):
        # train the model on the train set
        #   and returns the error on the test set
        pass


def _split_train_test_smoothers(x, smoothers, train_index, test_index):
    train_smoothers = []
    test_smoothers = []
    for i, smoother in enumerate(smoothers.smoothers):
        train_basis = smoother.basis[train_index]
        train_der_basis = smoother.der_basis[train_index]
        train_der2_basis = smoother.der2_basis[train_index]
        train_cov_der2 = smoother.cov_der2
        # TODO: Double check this part. cov_der2 is calculated with all data
        train_x = smoother.x[train_index]

        train_smoothers.append(UnivariateGenericSmoother(train_x, train_basis,
            train_der_basis, train_der2_basis, train_cov_der2,
            smoother.variable_name + ' train'))

        test_basis = smoother.basis[test_index]
        test_der_basis = smoother.der_basis[test_index]
        test_der2_basis = smoother.der2_basis[test_index]
        test_cov_der2 = smoother.cov_der2
        # TODO: Double check this part. cov_der2 is calculated with all data
        test_x = smoother.x[test_index]

        test_smoothers.append(UnivariateGenericSmoother(test_x, test_basis,
            test_der_basis, train_der2_basis, test_cov_der2,
            smoother.variable_name + ' test'))

    train_multivariate_smoothers = GenericSmoothers(x[train_index],
                                                    train_smoothers)
    test_multivariate_smoothers = GenericSmoothers(x[test_index],
                                                   test_smoothers)

    return train_multivariate_smoothers, test_multivariate_smoothers


class MultivariateGAMCV(BaseCV):
    def __init__(self, smoothers, alphas, gam, cost, y, cv):
        self.cost = cost
        self.gam = gam
        self.smoothers = smoothers
        self.alphas = alphas
        self.cv = cv
        super(MultivariateGAMCV, self).__init__(cv, self.smoothers.basis, y)

    def _error(self, train_index, test_index, **kwargs):
        full_basis_train = self.smoothers.basis[train_index]
        train_smoothers, test_smoothers = _split_train_test_smoothers(
            self.smoothers.x, self.smoothers, train_index, test_index)

        y_train = self.y[train_index]
        y_test = self.y[test_index]

        gam = self.gam(y_train, smoother=train_smoothers, alpha=self.alphas)
        gam_res = gam.fit(**kwargs)
        y_est = gam_res.predict(test_smoothers.basis, transform=False)

        return self.cost(y_test, y_est)


class BasePenaltiesPathCV(with_metaclass(ABCMeta)):
    """
    Base class for cross validation over a grid of parameters.

    The best parameter is saved in alpha_cv

    This class is currently not used
    """

    def __init__(self, alphas):
        self.alphas = alphas
        self.alpha_cv = None
        self.cv_error = None
        self.cv_std = None

    def plot_path(self):
        if have_matplotlib:
            plt.plot(self.alphas, self.cv_error, c='black')
            plt.plot(self.alphas, self.cv_error + 1.96 * self.cv_std,
                     c='blue')
            plt.plot(self.alphas, self.cv_error - 1.96 * self.cv_std,
                     c='blue')

            plt.plot(self.alphas, self.cv_error, 'o', c='black')
            plt.plot(self.alphas, self.cv_error + 1.96 * self.cv_std, 'o',
                     c='blue')
            plt.plot(self.alphas, self.cv_error - 1.96 * self.cv_std, 'o',
                     c='blue')

            return
            # TODO add return


class MultivariateGAMCVPath(object):
    """k-fold cross-validation for GAM

    Warning: The API of this class is preliminary and will change.

    """

    def __init__(self, smoothers, alphas, gam, cost, y, cv):
        self.cost = cost
        self.smoothers = smoothers
        self.gam = gam
        self.alphas = alphas
        self.alphas_grid = list(itertools.product(*self.alphas))
        self.y = y
        self.cv = cv
        self.cv_error = np.zeros(shape=(len(self.alphas_grid, )))
        self.cv_std = np.zeros(shape=(len(self.alphas_grid, )))
        self.alpha_cv = None

    def fit(self, **kwargs):
        for i, alphas_i in enumerate(self.alphas_grid):
            gam_cv = MultivariateGAMCV(smoothers=self.smoothers,
                                       alphas=alphas_i,
                                       gam=self.gam,
                                       cost=self.cost,
                                       y=self.y,
                                       cv=self.cv)
            cv_err = gam_cv.fit(**kwargs)
            self.cv_error[i] = cv_err.mean()
            self.cv_std[i] = cv_err.std()

        self.alpha_cv = self.alphas_grid[np.argmin(self.cv_error)]
        return self
