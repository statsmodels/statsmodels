from __future__ import division

__author__ = 'Luca Puggini: <lucapuggio@gmail.com>'

from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import itertools
import numpy as np
from statsmodels.gam.smooth_basis import GenericSmoothers, UnivariateGenericSmoother

try:
    import matplotlib.pyplot as plt
    have_matplotlib = True
except:
    have_matplotlib = False


class BaseCV(with_metaclass(ABCMeta)):
    """
    BaseCV class. It computes the cross validation error of a given model.
    All the cross validation classes can be derived by this one (e.g. GamCV, LassoCV,...)
    """

    def __init__(self, cv, x, y):
        self.cv = cv
        self.x = x
        self.y = y
        self.train_test_cv_indices = self.cv.split(self.x, self.y, label=None)

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


def _split_train_test_smoothers(x, smoothers, train_index, test_index):
    train_smoothers = []
    test_smoothers = []
    for i, smoother in enumerate(smoothers.smoothers_):
        train_basis = smoother.basis_[train_index]
        train_der_basis = smoother.der_basis_[train_index]
        train_der2_basis = smoother.der2_basis_[train_index]
        train_cov_der2 = smoother.cov_der2_  # TODO: Double check this part. cov_der2 is calculated with all the data
        train_x = smoother.x[train_index]

        train_smoothers.append(UnivariateGenericSmoother(train_x, train_basis, train_der_basis, train_der2_basis,
                                                         train_cov_der2, smoother.variable_name + ' train'))

        test_basis = smoother.basis_[test_index]
        test_der_basis = smoother.der_basis_[test_index]
        test_der2_basis = smoother.der2_basis_[test_index]
        test_cov_der2 = smoother.cov_der2_  # TODO: Double check this part. cov_der2 is calculated with all the data
        test_x = smoother.x[test_index]

        test_smoothers.append(UnivariateGenericSmoother(test_x, test_basis, test_der_basis, train_der2_basis,
                                                        test_cov_der2, smoother.variable_name + ' test'))

    train_multivariate_smoothers = GenericSmoothers(x[train_index], train_smoothers)
    test_multivariate_smoothers = GenericSmoothers(x[test_index], test_smoothers)

    return train_multivariate_smoothers, test_multivariate_smoothers


class MultivariateGAMCV(BaseCV):
    def __init__(self, smoothers, alphas, gam, cost, y, cv):
        # the gam class has already an instance
        self.cost = cost
        self.gam = gam
        self.smoothers = smoothers
        self.alphas = alphas
        self.cv = cv
        super(MultivariateGAMCV, self).__init__(cv, self.smoothers.basis_, y)

    def _error(self, train_index, test_index, **kwargs):
        full_basis_train = self.smoothers.basis_[train_index]
        train_smoothers, test_smoothers = _split_train_test_smoothers(self.smoothers.x, self.smoothers, train_index,
                                                                      test_index)

        y_train = self.y[train_index]
        y_test = self.y[test_index]

        gam = self.gam(y_train, train_smoothers, alpha=self.alphas)
        gam_res = gam.fit(**kwargs)
        y_est = gam_res.predict(test_smoothers.basis_)

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
        if not have_matplotlib:
            return
        else:
            plt.plot(self.alphas, self.cv_error_, c='black')
            plt.plot(self.alphas, self.cv_error_ + 1.96 * self.cv_std_, c='blue')
            plt.plot(self.alphas, self.cv_error_ - 1.96 * self.cv_std_, c='blue')

            plt.plot(self.alphas, self.cv_error_, 'o', c='black')
            plt.plot(self.alphas, self.cv_error_ + 1.96 * self.cv_std_, 'o', c='blue')
            plt.plot(self.alphas, self.cv_error_ - 1.96 * self.cv_std_, 'o', c='blue')

            return


class MultivariateGAMCVPath:
    def __init__(self, smoothers, alphas, gam, cost, y, cv):
        self.cost = cost
        self.smoothers = smoothers
        self.gam = gam
        self.alphas = alphas
        self.alphas_grid = list(itertools.product(*self.alphas))
        self.y = y
        self.cv = cv
        self.cv_error_ = np.zeros(shape=(len(self.alphas_grid, )))
        self.cv_std_ = np.zeros(shape=(len(self.alphas_grid, )))
        self.alpha_cv_ = None

        return

    def fit(self, **kwargs):
        for i, alphas_i in enumerate(self.alphas_grid):
            gam_cv = MultivariateGAMCV(smoothers=self.smoothers, alphas=alphas_i,
                                       gam=self.gam, cost=self.cost, y=self.y, cv=self.cv)
            cv_err = gam_cv.fit(**kwargs)
            self.cv_error_[i] = cv_err.mean()
            self.cv_std_[i] = cv_err.std()

        self.alpha_cv_ = self.alphas_grid[np.argmin(self.cv_error_)]
        return self
