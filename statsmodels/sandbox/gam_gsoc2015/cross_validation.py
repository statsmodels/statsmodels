__author__ = 'Luca Puggini'
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

    def _error(self, train_index, test_index):

        der2_train = self.der2[train_index]
        basis_train = self.basis[train_index]
        basis_test = self.basis[test_index]
        y_train = self.y[train_index]
        y_test = self.y[test_index]

        gp = GamPenalty(1, self.alpha, self.cov_der2, der2_train)
        gam = self.gam(y_train, basis_train, penal=gp).fit()
        y_est = gam.predict(basis_test)

        return self.cost(y_test, y_est)

    def fit(self):

        cv_err = []
        for train_index, test_index in self.cv_index:
            cv_err.append(self._error(train_index, test_index))

        return np.array(cv_err)

    def fit_alphas_path(self, alphas):

        err_m = []
        err_std = []
        for alpha in alphas:
            self.alpha = alpha
            err = self.fit()
            err_m.append(err.mean())
            err_std.append(err.std())

        return np.array(err_m), np.array(err_std)


class GamKfoldsCV(GamCV):

    def __init__(self, gam, alpha, cost, basis, der_basis, der2, cov_der2, y, k):

        self.n_obs = basis.shape[0]
        GamCV.__init__(self, gam, alpha, cost, basis, der_basis, der2, cov_der2, y)
        self.cv_index = KFold(self.n_obs, k)
        return





def sample_metric(y1, y2):

    return np.linalg.norm(y1 - y2)/len(y1)



n = 1000
np.random.seed(1)
index = np.array(range(n))
np.random.shuffle(index)

X = np.linspace(-10, 10, n)
Y = X**2 - X + np.random.normal(0, 10, n)

X = X[index]
Y = Y[index]

alphas = np.linspace(0, .1, 20)
degree = 8 # required for the basis generation
k_folds = 5

basis, der_basis, der2 = make_poly_basis(X, degree=degree)
cov_der2 = np.dot(der2.T, der2)

gam_cv = GamKfoldsCV(GLMGam, 0, sample_metric, basis, der_basis, der2, cov_der2, Y, k_folds)
cv_err = gam_cv.fit()

cv_path_m, cv_path_std = gam_cv.fit_alphas_path(alphas)
print('mean cv err =', cv_path_m, ' std=', cv_path_std)

best_alpha = alphas[np.argmin(cv_path_m)]

gp = GamPenalty(alpha=best_alpha, cov_der2=cov_der2, der2=der2)
model = GLMGam(Y, basis, penal=gp)
res = model.fit(maxiter=10000)
y_est = res.predict()

plt.subplot(3, 1, 1)
plt.plot(X, Y, '.')
plt.plot(X, y_est, '.')

plt.subplot(3, 1, 2)
plt.plot(alphas, cv_path_m)
plt.plot(alphas, cv_path_m, 'o')
plt.plot(alphas, cv_path_m + cv_path_std)
plt.plot(alphas, cv_path_m - cv_path_std)

plt.subplot(3, 1, 3)
plt.plot(alphas, cv_path_m, 'o')
plt.plot(alphas, cv_path_m)

plt.show()

