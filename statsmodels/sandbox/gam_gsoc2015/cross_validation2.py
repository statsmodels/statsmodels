__author__ = 'Luca Puggini: <lucapuggio@gmail.com>'
__date__ = '08/07/15'

from statsmodels.sandbox.gam_gsoc2015.smooth_basis import make_poly_basis, make_bsplines_basis
from statsmodels.sandbox.gam_gsoc2015.gam import GamPenalty, GLMGam, MultivariateGamPenalty, LogitGam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from statsmodels.sandbox.gam_gsoc2015.gam import GamPenalty, GLMGam, MultivariateGamPenalty, LogitGam, Penalty
from statsmodels.sandbox.tools.cross_val import KFold

from abc import ABCMeta, abstractmethod


######################################################################################


class BaseCrossValidator(metaclass=ABCMeta):
    """
    The BaseCrossValidator class is a base class for all the iterators that split the data in train and test as for
    example KFolds or LeavePOut
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


##############################################################################################

class BaseCV(metaclass=ABCMeta):
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


class GamCV(BaseCV):

    def __init__(self, gam, alpha, cost, basis, der_basis, der2, cov_der2, y, cv):
        # the gam class has already an instance
        self.cost = cost
        self.gam = gam
        self.basis = basis
        self.der_basis = der_basis
        self.der2 = der2
        self.alpha = alpha
        self.cov_der2 = cov_der2 #TODO: Maybe cov_der2 has to be recomputed every time?
        self.y = y
        self.cv = cv
        self.train_test_cv_indices = self.cv.split(self.basis)
        return

    def _error(self, train_index, test_index, **kwargs):

        der2_train = self.der2[train_index]
        basis_train = self.basis[train_index]
        basis_test = self.basis[test_index]
        y_train = self.y[train_index]
        y_test = self.y[test_index]

        gp = GamPenalty(1, self.alpha, self.cov_der2, der2_train)
        gam = self.gam(y_train, basis_train, penal=gp).fit(**kwargs)
        y_est = gam.predict(basis_test)

        return self.cost(y_test, y_est)

#######################################################################################################

class BasePenaltiesPathCV(metaclass=ABCMeta):
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


class GamCVPath(BasePenaltiesPathCV):

    def __init__(self, alphas, gam, cost, basis, der_basis, der2, cov_der2, y, cv):

        self.cost = cost
        self.gam = gam
        self.basis = basis
        self.der_basis = der_basis
        self.der2 = der2
        self.alphas = alphas
        self.cov_der2 = cov_der2 #TODO: Maybe cov_der2 has to be recomputed every time?
        self.y = y
        self.cv = cv

        return

    def fit(self, **kwargs):

        self.cv_error_ = np.zeros(shape=(len(self.alphas,)))
        self.cv_std_ = np.zeros(shape=(len(self.alphas, )))
        for i, alpha in enumerate(alphas):
            gam_cv = GamCV(self.gam, alpha, self.cost, self.basis, self.der_basis, self.der2,
                           self.cov_der2, self.y, self.cv)
            cv_err = gam_cv.fit(**kwargs)
            self.cv_error_[i] = cv_err.mean()
            self.cv_std_[i] = cv_err.std()

        self.alpha_cv_ = self.alphas[np.argmin(self.cv_error_)]
        return self


##################################################################################################################
## EXAMPLE ##

def sample_metric(y1, y2):
    return np.linalg.norm(y1 - y2)/len(y1)


n = 2000
x = np.linspace(-1, 1, n)
y = x*x*x - x*x + np.random.normal(0, .1, n)

df = 10
degree = 6
basis, der_basis, der2 = make_bsplines_basis(x, df=df, degree=degree)
cov_der2 = np.dot(der2.T, der2)

gam = GLMGam
kfolds = KFold(10, shuffle=True)

#gam_cv_error = GamCV(gam=gam, alpha=0, cost=sample_metric, basis=basis, der_basis=der_basis, der2=der2,
#                     cov_der2=cov_der2, y=y, cv=kfolds).fit(method='nm', max_start_irls=0, disp=1, maxiter=5000, maxfun=5000)
#print(gam_cv_error)

alphas = np.linspace(.0, .5, 10)
gam_cv_path = GamCVPath(gam=gam, alphas=alphas, cost=sample_metric, basis=basis, der_basis=der_basis, der2=der2,
                        cov_der2=cov_der2, y=y, cv=kfolds)

gam_cv_path.fit(method='nm', max_start_irls=0, disp=1, maxiter=5000, maxfun=5000)

best_alpha = gam_cv_path.alpha_cv_


gp = GamPenalty(alpha=gam_cv_path.alpha_cv_, cov_der2=cov_der2, der2=der2)
gam = GLMGam(y, basis, penal=gp)
res = gam.fit(method='nm', max_start_irls=0, disp=0, maxiter=5000, maxfun=5000)
y_est = res.predict()



plt.plot(y_est)
plt.plot(y, '.')
plt.show()

gam_cv_path.plot_path()
plt.show()