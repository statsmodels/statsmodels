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

class GamCV:

    def __init__(self, gam, alpha, cost, basis, der_basis, der2, cov_der2, y):
        # the gam class has already an instance
        self.cost = cost
        self.gam = gam
        self.basis = basis
        self.der_basis = der_basis
        self.der2 = der2
        self.alpha = alpha
        self.cov_der2 = cov_der2 #TODO: Maybe cov_der2 has to be recomputed every time?
        self.y = y
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

    def fit(self, **kwargs): #TODO: add fit parameters (method='nm', max_start_irls=0, disp=1, maxiter=5000, maxfun=5000

        cv_err = []
        for train_index, test_index in self.cv_index:
            cv_err.append(self._error(train_index, test_index, **kwargs))

        return np.array(cv_err)

    def fit_alphas_path(self, alphas, **kwargs): #TODO: add fit parameters (method='nm', max_start_irls=0, disp=1, maxiter=5000, maxfun=5000

        err_m = []
        err_std = []
        for alpha in alphas:
            self.alpha = alpha
            err = self.fit(**kwargs)
            err_m.append(err.mean())
            err_std.append(err.std())

        return np.array(err_m), np.array(err_std)


class GamKFoldsCV(GamCV):

    def __init__(self, gam, alpha, cost, basis, der_basis, der2, cov_der2, y, k):

        self.n_obs = basis.shape[0]
        GamCV.__init__(self, gam, alpha, cost, basis, der_basis, der2, cov_der2, y)
        self.cv_index = KFold(self.n_obs, k)
        return


#########################################################################################################

from abc import ABCMeta, abstractmethod

class CVIterator(metaclass=ABCMeta):

    def __init__(self):

        self.n = None # the value of n is obtained in the fit method
        return

    def fit(self, n):

        self.n = n
        return self

    @abstractmethod
    def __iter__(self):

        return


class KFold_new_version(CVIterator): # new version of KFold
    """
    K-Folds cross validation iterator:
    Provides train/test indexes to split data in train test sets
    """

    def __init__(self, k):
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
        assert k > 0, ValueError('cannot have k below 1')
        self.k = k
        return

    def __iter__(self):
        n = self.n
        k = self.k
        j = int(np.ceil(n/k))

        for i in range(k):
            test_index  = np.zeros(n, dtype=np.bool)
            if i<k-1:
                test_index[i*j:(i+1)*j] = True
            else:
                test_index[i*j:] = True
            train_index = np.logical_not(test_index)
            yield train_index, test_index

    def __repr__(self):
        return '%s.%s(n=%i, k=%i)' % (
                                self.__class__.__module__,
                                self.__class__.__name__,
                                self.n,
                                self.k,
                                )



class GamCV_new_version(GamCV):

    def __init__(self, gam, alpha, cost, basis, der_basis, y, der2, cov_der2, cv):

        self.n_obs = basis.shape[0]
        GamCV.__init__(self, gam, alpha, cost, basis, der_basis, der2, cov_der2, y)
        self.cv_index = cv.fit(basis.shape[0])

        return


'''
def sample_metric(y1, y2):

        return np.linalg.norm(y1 - y2)/len(y1)


n = 200
x = np.linspace(-1, 1, n)
y = x*x*x - x*x + np.random.normal(0, .1, n)

df = 10
degree = 6
basis, der_basis, der2 = make_bsplines_basis(x, df=df, degree=degree)
cov_der2 = np.dot(der2.T, der2)

gam = GLMGam
alphas = np.linspace(0, .1, 5)
k = 5
cv = KFold_new_version(k)
gam_cv = GamCV_new_version(gam=gam, alpha=0, cost=sample_metric, basis=basis, der_basis=der_basis, der2=der2, cov_der2=cov_der2, y=y, cv=cv)

cv_path_m, cv_path_std = gam_cv.fit_alphas_path(alphas, method='nm', max_start_irls=0, disp=1, maxiter=5000, maxfun=5000)
best_alpha = alphas[np.argmin(cv_path_m)]
gp = GamPenalty(alpha=best_alpha, cov_der2=cov_der2, der2=der2)
model = GLMGam(y, basis, penal=gp)
res = model.fit(method='nm', max_start_irls=0, disp=0, maxiter=5000, maxfun=5000)
y_est = res.predict()

plt.plot(y_est)
plt.plot(y, '.')
plt.show()


'''