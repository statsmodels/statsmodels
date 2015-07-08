__author__ = 'Luca Puggini'
__date__ = '08/07/15'

from statsmodels.sandbox.gam_gsoc2015.smooth_basis import make_poly_basis, make_bsplines_basis
from statsmodels.sandbox.gam_gsoc2015.gam import GamPenalty, GLMGam, MultivariateGamPenalty, LogitGam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from statsmodels.sandbox.gam_gsoc2015.gam import GamPenalty, GLMGam, MultivariateGamPenalty, LogitGam, Penalty




def error(y1, y2, metric):

    return metric(y1 - y2)/len(y1)


def kfolds_cv(x, y, k_folds, metric, alpha, model=None, shuffle=True):
    # TODO: We pass the data (X,y) and the number of folds k. We may want to pass a generic model

    n = len(y)
    index = np.array(range(n))

    if shuffle:
        np.random.shuffle(index)

    fold_index = np.array_split(index, k_folds)
    cv = []
    for k in range(k_folds):
        # the data is divided into train and test
        train_index = np.concatenate([fold_index[i] for i in range(k_folds) if i != k])
        test_index = fold_index[k]
        x_train = X[train_index]
        y_train = y[train_index]
        x_test = X[test_index]
        y_test = y[test_index]

        # TODO: The following lines are GAM specific. They are required but the code is not generalizable
        # TODO: for gam it is required to transform the data. We should decide how to do this
        basis_train, der_basis_train, der2_train = make_poly_basis(x_train, degree=degree)
        cov_der2_train = np.dot(der2_train.T, der2_train)
        basis_test, _, _ = make_poly_basis(x_test, degree=degree)

        # # TODO: The penalty must be redefined everytime. We should decide how to do this.
        gp = GamPenalty(alpha=alpha, cov_der2=cov_der2_train, der2=der2_train)
        model = GLMGam(y_train, basis_train, penal=gp)

        # the model is trained on k-1 folds
        res = model.fit(maxiter=10000)
        y_est = res.predict(basis_test)

        # the prediction error is computed on the remaining fold
        err = error(y_est, y_test, metric)

        cv.append(err)
    return np.array(cv)



def best_alpha(metric, alphas = (0, 1, 2), kfolds=3):
    # find the alpha that returns the smaller error after the grid search
    cv_err = []
    for alpha in alphas:
        cv = kfolds_cv(X, Y, kfolds, sample_metric, alpha=alpha).mean()
        cv_err.append(cv.mean())

    alpha_best = alphas[np.argmin(cv_err)]

    print('CV error=', cv_err)
    return alpha_best



n = 2000
X = np.linspace(-10, 10, n)
Y = X**2 - X + np.random.normal(0, 10, n)

sample_metric = np.linalg.norm
alphas = [0, 1.e-5, 1.e-4, 1.e-3, 1.e-2, .1, 1]
degree = 6 # required for the basis generation
kfolds = 3

alpha_best = best_alpha(sample_metric, alphas, kfolds=kfolds)

basis, der_basis, der2 = make_poly_basis(X, degree=degree)
cov_der2 = np.dot(der2.T, der2)
gp = GamPenalty(alpha=alpha_best, cov_der2=cov_der2, der2=der2)
model = GLMGam(Y, basis, penal=gp)
res = model.fit(maxiter=10000)

y_est = res.predict()

plt.plot(X, Y, '.')
plt.plot(X, y_est)
plt.show()

print('best alpha=', alpha_best, 'params=', res.params)