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






