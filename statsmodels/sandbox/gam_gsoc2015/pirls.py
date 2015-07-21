__author__ = 'Luca Puggini: <lucapuggio@gmail.com>'
__date__ = '16/07/15'

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
import pandas as pd
from numpy.testing import assert_allclose
from statsmodels.genmod.generalized_linear_model import GLM, GLMResults, GLMResultsWrapper
import statsmodels.regression.linear_model as lm
import scipy as sp

# def equally_spaced_knots(x, n_knots):
#     x_min = x.min()
#     x_max = x.max()
#     knots = np.linspace(x_min, x_max, n_knots)
#     return knots
#
# def rk(x, z):
#     p1 = ((z - 1/2)**2 - 1/12) * ((x - 1/2)**2 - 1/12) / 4
#     p2 = ((np.abs(z - x) - 1/2)**4 - 1/2 * (np.abs(z - x) - 1/2)**2 + 7/240) / 24.
#     return p1 - p2
#
#
# def splines_x(x, xk):
#     n_columns = len(xk) + 2
#     n_samples = x.shape[0]
#     xs = np.ones(shape=(n_samples, n_columns))
#     xs[:, 1] = x
#     # for loop equivalent to outer(x, xk, fun=rk)
#     for i, xi in enumerate(x):
#         for j, xkj in enumerate(xk):
#             s_ij = rk(xi, xkj)
#             xs[i, j+2] = s_ij
#     return xs
#
#
# def splines_s(xk):
#     q = len(xk) + 2
#     s = np.zeros(shape=(q, q))
#     for i, x1 in enumerate(xk):
#         for j, x2 in enumerate(xk):
#             s[i+2, j+2] = rk(x1, x2)
#     return s
#
#
# def get_sqrt(x):
#     """
#     :param x:
#     :return: b the sqrt of the matrix x. np.dot(b.T, b) = x
#     """
#     u, s, v = np.linalg.svd(x)
#     sqrt_s = np.sqrt(s)
#     b = np.dot(u, np.dot(np.diag(sqrt_s), v))
#     return b
#
# class GamPirls(GLM):
#
#     def _fit_pirls(self, y, spl_x, spl_s, alpha, start_params=None, maxiter=100, tol=1e-8,
#                    scale=None, cov_type='nonrobust', cov_kwds=None, use_t=None, weights=None):
#
#         n_samples, n_columns = spl_x.shape
#
#         if isinstance(spl_s, list):
#             alpha_spl_s = [spl_s[i] * alpha[i] for i in range(len(spl_s))]
#             alpha_spl_s = sp.linalg.block_diag(*alpha_spl_s)
#         else:
#             alpha_spl_s = alpha * spl_s
#
#         rs = get_sqrt(alpha_spl_s)
#
#         x1 = np.vstack([spl_x, rs])  # augmented x
#         n_samp1es_x1 = x1.shape[0]
#         y1 = np.array([0] * n_samp1es_x1)  # augmented y
#         y1[:n_samples] = y
#
#         if weights is None:
#             self.weights = np.array([1./n_samp1es_x1] * n_samp1es_x1)  # TODO: should the weight be of size n_samples_x1?
#                                                                        # Probably we have to use equation 4.22 from the Wood's
#                                                                        # book and replace WLS with OLS
#
#         if start_params is None:
#             params = np.zeros(shape=(n_columns,))
#             params[0] = 1
#         else:
#             params = start_params
#
#         lin_pred = np.dot(x1, params)[:n_samples]
#         self.mu = self.family.fitted(lin_pred)
#         dev = self.family.deviance(self.endog, self.mu)
#
#         history = dict(params=[None, params], deviance=[np.inf, dev])
#         converged = False
#         criterion = history['deviance']
#
#         new_norm = 0
#         old_norm = 1
#         iteration = 0
#         for iteration in range(maxiter):
#             if np.abs(new_norm - old_norm) < tol:
#                 break
#
#             lin_pred = np.dot(x1, params)[:n_samples]
#             self.mu = self.family.fitted(lin_pred)
#             z = np.zeros(shape=(n_samp1es_x1,))
#             z[:n_samples] = (y - self.mu) / self.mu + lin_pred # TODO: review this and the following line
#             wls_results = lm.WLS(z, x1, self.weights).fit() # TODO: should weights be used?
#             lin_pred = np.dot(spl_x, wls_results.params)
#
#             new_mu = self.family.fitted(lin_pred)
#
#             old_norm = new_norm
#             new_norm = np.linalg.norm((z[:n_samples] - new_mu))
#
#         self.scale = 0 # TODO: check the right value scale
#         self.data_weights = 0 # TODO: add support for weights
#         glm_results = GLMResults(self, wls_results.params,
#                                  wls_results.normalized_cov_params,
#                                  self.scale,
#                                  cov_type=cov_type, cov_kwds=cov_kwds,
#                                  use_t=use_t)
#
#         glm_results.method = "PIRLS"
#         history['iteration'] = iteration + 1
#         glm_results.fit_history = history
#         glm_results.converged = converged
#         return GLMResultsWrapper(glm_results)

"""
import os
cur_dir = os.path.dirname(os.path.abspath('__file__'))
file_path = os.path.join(cur_dir, "tests/results", "gam_PIRLS_results.csv")
data = pd.read_csv(file_path)

print('Univariate GAM ')
X = data['x'].as_matrix()
Y = data['y'].as_matrix()

XK = np.array([0.2, .4, .6, .8])

spl_x_R = data[['spl_x.1', 'spl_x.2', 'spl_x.3', 'spl_x.4', 'spl_x.5', 'spl_x.6']].as_matrix()
SPL_X = splines_x(X, XK)

assert_allclose(spl_x_R, SPL_X)

spl_s_R = [[0,    0,  0.000000000,  0.000000000,  0.000000000,  0.000000000],
           [0,    0,  0.000000000,  0.000000000,  0.000000000,  0.000000000],
           [0,    0,  0.001400000,  0.000200000, -0.001133333, -0.001000000],
           [0,    0,  0.000200000,  0.002733333,  0.001666667, -0.001133333],
           [0,    0, -0.001133333,  0.001666667,  0.002733333,  0.000200000],
           [0,    0, -0.001000000, -0.001133333,  0.000200000,  0.001400000]]
SPL_S = splines_s(xk=XK)
assert_allclose(spl_s_R, SPL_S, atol=4.e-10)

b = get_sqrt(SPL_S)

assert_allclose(np.dot(b.T, b), SPL_S)

for i, alpha in enumerate([0, .1, 10, 200]):

    gam = GamPirls(Y, SPL_X)
    gam_results = gam._fit_pirls(Y, SPL_X, SPL_S, alpha)
    Y_EST = np.dot(SPL_X, gam_results.params)
    plt.subplot(2, 2, i+1)
    plt.title('Alpha=' + str(alpha))
    plt.plot(X, Y, '.')
    plt.plot(X, Y_EST, '.')
plt.show()



print('multivariate gam')
X2 = np.random.uniform(0, 1, len(Y))
XK2 = [0, 0.5, 1]
SPL_X2 = splines_x(X2, XK2)
SPL_S2 = splines_s(xk=XK2)

SPL_X_FULL = np.hstack([SPL_X, SPL_X2])
SPL_S_FULL = [SPL_S, SPL_S2]

Y = X2*X2 + X*X*X + np.random.normal(0, .2, len(Y))
gam = GamPirls(Y, SPL_X_FULL)
alpha = 0
gam_results = gam._fit_pirls(Y, SPL_X_FULL, SPL_S_FULL, alpha=[alpha]*2)

n_var1 = SPL_X.shape[1]
n_var2 = SPL_X2.shape[1]

plt.subplot(3, 1, 1)
plt.plot(X, np.dot(SPL_X, gam_results.params[:n_var1]), '.', label='Estimated')
plt.plot(X, X*X*X, '.', label='Real')
plt.legend(loc='best')
plt.subplot(3, 1, 2)
plt.plot(X2, np.dot(SPL_X2, gam_results.params[n_var1:]), '.', label='Estimated')
plt.plot(X2, X2*X2, '.', label='Real')
plt.legend(loc='best')
plt.subplot(3, 1, 3)
plt.plot(Y, '.')
plt.plot(np.dot(SPL_X_FULL, gam_results.params), '.')
plt.show()


"""