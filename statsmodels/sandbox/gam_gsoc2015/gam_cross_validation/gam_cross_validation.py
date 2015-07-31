
from __future__ import division

__author__ = 'Luca Puggini'

from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.sandbox.gam_gsoc2015.gam import UnivariateGamPenalty

class BaseCV(with_metaclass(ABCMeta)):
    """
    BaseCV class. It computes the cross validation error of a given model.
    All the cross validation classes can be derived by this one (e.g. GamCV, LassoCV,...)
    """
    def __init__(self):

        self.train_test_cv_indices = None
        return

    def fit(self, **kwargs):
        # kwargs are the input values for the fit method of the cross-validated object

        cv_err = []
        for train_index, test_index in self.train_test_cv_indices:
            cv_err.append(self._error(train_index, test_index, **kwargs))

        return np.array(cv_err)

    @abstractmethod
    def _error(self, train_index, test_index, **kwargs):
        # train the model on the train set and returns the error on the test set
        return


class UnivariateGamCV(BaseCV):

    def __init__(self, gam, alpha, cost, univariate_smoother, y, cv):
        # the gam class has already an instance
        self.cost = cost
        self.gam = gam
        self.univariate_smoother = univariate_smoother
        self.alpha = alpha
        self.y = y
        self.cv = cv
        self.train_test_cv_indices = self.cv.split(self.univariate_smoother.der2_basis_, self.y, label=None)
        return

    def _error(self, train_index, test_index, **kwargs):

        der2_train = self.univariate_smoother.der2_basis_[train_index]
        basis_train = self.univariate_smoother.basis_[train_index]
        basis_test = self.univariate_smoother.basis_[test_index]
        y_train = self.y[train_index]
        y_test = self.y[test_index]

        gp = UnivariateGamPenalty(self.univariate_smoother, self.alpha)
        gam = self.gam(y_train, basis_train, penal=gp).fit(**kwargs)
        y_est = gam.predict(basis_test)

        return self.cost(y_test, y_est)


class BasePenaltiesPathCV(with_metaclass(ABCMeta)):
    """
    Base class for cross validation over a grid of parameters.
    The best parameter is saved in alpha_cv_
    """
    def __init__(self, alphas):

        self.alphas = alphas
        self.alpha_cv_ = None
        self.cv_error_ = None
        self.cv_std_ = None
        return

    def plot_path(self):

        plt.plot(self.alphas, self.cv_error_, c='black')
        plt.plot(self.alphas, self.cv_error_ + self.cv_std_, c='blue')
        plt.plot(self.alphas, self.cv_error_ - self.cv_std_, c='blue')

        plt.plot(self.alphas, self.cv_error_, 'o', c='black')
        plt.plot(self.alphas, self.cv_error_ + self.cv_std_, 'o', c='blue')
        plt.plot(self.alphas, self.cv_error_ - self.cv_std_, 'o', c='blue')

        return


class UnivariateGamCVPath(BasePenaltiesPathCV):

    def __init__(self, univariate_smoother, alphas, gam, cost, y, cv):

        self.cost = cost
        self.univariate_smoother = univariate_smoother
        self.gam = gam
        self.alphas = alphas
        self.y = y
        self.cv = cv

        return

    def fit(self, **kwargs):

        self.cv_error_ = np.zeros(shape=(len(self.alphas,)))
        self.cv_std_ = np.zeros(shape=(len(self.alphas, )))
        for i, alpha in enumerate(self.alphas):
            gam_cv = UnivariateGamCV(self.gam, alpha, self.cost, univariate_smoother=self.univariate_smoother, y=self.y,
                                     cv=self.cv)
            cv_err = gam_cv.fit(**kwargs)
            self.cv_error_[i] = cv_err.mean()
            self.cv_std_[i] = cv_err.std()

        self.alpha_cv_ = self.alphas[np.argmin(self.cv_error_)]
        return self

