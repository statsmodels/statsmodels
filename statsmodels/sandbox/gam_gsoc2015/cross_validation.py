# __author__ = 'Luca Puggini: <lucapuggio@gmail.com>'
# __date__ = '08/07/15'
#
# from statsmodels.sandbox.gam_gsoc2015.smooth_basis import make_poly_basis, make_bsplines_basis
# from statsmodels.sandbox.gam_gsoc2015.gam import GamPenalty, GLMGam, MultivariateGamPenalty, LogitGam
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# from statsmodels.sandbox.gam_gsoc2015.gam import GamPenalty, GLMGam, MultivariateGamPenalty, LogitGam, Penalty
# from statsmodels.sandbox.tools.cross_val import KFold
#
# from abc import ABCMeta, abstractmethod
#
#
#
# class BaseCV(metaclass=ABCMeta):
#     """
#     BaseCV class. All the cross validation classes can be derived by this one (e.g. GamCV, LassoCV,...)
#     """
#     def __init__(self):
#
#         self.alpha = None
#         self.alphas = None
#         self.cv_err_ = None
#         self.cv_std_ = None
#         self.alpha_cv_ = None # is the optimal alpha chosen by CV
#
#         return
#
#     def fit(self, **kwargs):
#         # kwargs are the input values for the fit method of the cross-validated object
#
#         cv_err = []
#         for train_index, test_index in self.cv_index:
#             cv_err.append(self._error(train_index, test_index, **kwargs))
#
#         return np.array(cv_err)
#
#     def fit_alphas_path(self, alphas, **kwargs):
#         # kwargs are the arguments for the fit method of the gam function
#
#         self.cv_err_ = np.zeros(shape=(len(alphas), ))
#         self.cv_std_ = np.zeros(shape=(len(alphas), ))
#         self.alphas = alphas
#
#         for i, alpha in enumerate(alphas):
#             self.alpha = alpha
#             err = self.fit(**kwargs)
#             self.cv_err_[i] = err.mean()
#             self.cv_std_[i] = err.std()
#
#         self.alpha_cv_ = self.alphas[np.argmin(self.cv_err_)]
#
#         return self
#
#     @abstractmethod
#     def _error(self, train_index, test_index, **kwargs):
#         # train the model on the train set and returns the error on the test set
#         return
#
#
#
# class GamCV(BaseCV):
#
#     def __init__(self, gam, alpha, cost, basis, der_basis, der2, cov_der2, y):
#         # the gam class has already an instance
#         self.cost = cost
#         self.gam = gam
#         self.basis = basis
#         self.der_basis = der_basis
#         self.der2 = der2
#         self.alpha = alpha
#         self.cov_der2 = cov_der2 #TODO: Maybe cov_der2 has to be recomputed every time?
#         self.y = y
#         return
#
#     def _error(self, train_index, test_index, **kwargs):
#
#         der2_train = self.der2[train_index]
#         basis_train = self.basis[train_index]
#         basis_test = self.basis[test_index]
#         y_train = self.y[train_index]
#         y_test = self.y[test_index]
#
#         gp = GamPenalty(1, self.alpha, self.cov_der2, der2_train)
#         gam = self.gam(y_train, basis_train, penal=gp).fit(**kwargs)
#         y_est = gam.predict(basis_test)
#
#         return self.cost(y_test, y_est)
#
#
#
# class GamKFoldsCV(GamCV):
#
#     def __init__(self, gam, alpha, cost, basis, der_basis, der2, cov_der2, y, k):
#
#         self.n_obs = basis.shape[0]
#         GamCV.__init__(self, gam, alpha, cost, basis, der_basis, der2, cov_der2, y)
#         self.cv_index = KFold(self.n_obs, k)
#         return
#
#
# #########################################################################################################
#
#
# class BaseCrossValidator(metaclass=ABCMeta):
#
#     def __init__(self):
#
#         self.n_samples = None
#         return
#
#     def split(self, X, y=None, labels=None):
#         # by default the cross validation uses only the number of samples.
#         # this method may be overwritten
#
#         self.n_samples = X.shape[0]
#         return self
#
#     @abstractmethod
#     def __iter__(self):
#
#         return
#
#
# class KFold_new_version(BaseCrossValidator): # new version of KFold
#     """
#     K-Folds cross validation iterator:
#     Provides train/test indexes to split data in train test sets
#     """
#
#     def __init__(self, kfolds, shuffle=False):
#         """
#         K-Folds cross validation iterator:
#         Provides train/test indexes to split data in train test sets
#
#         Parameters
#         ----------
#         k: int
#             number of folds
#
#         Examples
#         --------
#
#         Notes
#         -----
#         All the folds have size trunc(n/k), the last one has the complementary
#         """
#         assert k > 0, ValueError('cannot have k below 1')
#         self.kfolds = kfolds
#         self.shuffle = shuffle
#         return
#
#     def __iter__(self):
#
#         index = np.array(range(self.n_samples))
#         if self.shuffle:
#
#
#         folds = np.array_split(index, self.kfolds)
#
#         for i in range(self.kfolds):
#
#
#
#
#
#     def __repr__(self):
#         return '%s.%s(n=%i, k=%i)' % (
#                                 self.__class__.__module__,
#                                 self.__class__.__name__,
#                                 self.n,
#                                 self.k,
#                                 )
#
# class GamCV_new_version(GamCV):
#
#     def __init__(self, gam, alpha, cost, basis, der_basis, y, der2, cov_der2, cv):
#
#         self.n_obs = basis.shape[0]
#         GamCV.__init__(self, gam, alpha, cost, basis, der_basis, der2, cov_der2, y)
#         self.cv_index = cv.split(basis)
#
#         return
#
#
#
# def sample_metric(y1, y2):
#
#         return np.linalg.norm(y1 - y2)/len(y1)
#
#
# n = 200
# x = np.linspace(-1, 1, n)
# y = x*x*x - x*x + np.random.normal(0, .1, n)
#
# df = 10
# degree = 6
# basis, der_basis, der2 = make_bsplines_basis(x, df=df, degree=degree)
# cov_der2 = np.dot(der2.T, der2)
#
# gam = GLMGam
# alphas = np.linspace(.1, .2, 3)
# k = 5
# cv = KFold_new_version(k)
# gam_cv = GamCV_new_version(gam=gam, alpha=0, cost=sample_metric, basis=basis, der_basis=der_basis, der2=der2, cov_der2=cov_der2, y=y, cv=cv)
#
# gam_cv = gam_cv.fit_alphas_path(alphas, method='nm', max_start_irls=0, disp=1, maxiter=5000, maxfun=5000)
#
# gp = GamPenalty(alpha=gam_cv.alpha_cv_, cov_der2=cov_der2, der2=der2)
# model = GLMGam(y, basis, penal=gp)
# res = model.fit(method='nm', max_start_irls=0, disp=0, maxiter=5000, maxfun=5000)
# y_est = res.predict()
#
# plt.plot(y_est)
# plt.plot(y, '.')
# plt.show()
