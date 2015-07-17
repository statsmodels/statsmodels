__author__ = 'Luca Puggini: <lucapuggio@gmail.com>'
__date__ = '16/07/15'

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
import pandas as pd
from numpy.testing import assert_allclose
from statsmodels.genmod.generalized_linear_model import GLM
import statsmodels.regression.linear_model as lm

def equally_spaced_knots(x, n_knots):
    x_min = x.min()
    x_max = x.max()
    knots = np.linspace(x_min, x_max, n_knots)
    return knots

def rk(x, z):
    p1 = ((z - 1/2)**2 - 1/12) * ((x - 1/2)**2 - 1/12) / 4
    p2 = ((np.abs(z - x) - 1/2)**4 - 1/2 * (np.abs(z - x) - 1/2)**2 + 7/240) / 24.
    return p1 - p2


def splines_x(x, xk):
    n_columns = len(xk) + 2
    n_samples = x.shape[0]
    xs = np.ones(shape=(n_samples, n_columns))
    xs[:, 1] = x
    # for loop equivalent to outer(x, xk, fun=rk)
    for i, xi in enumerate(x):
        for j, xkj in enumerate(xk):
            s_ij = rk(xi, xkj)
            xs[i, j+2] = s_ij
    return xs


def splines_s(xk):
    q = len(xk) + 2
    s = np.zeros(shape=(q, q))
    for i, x1 in enumerate(xk):
        for j, x2 in enumerate(xk):
            s[i+2, j+2] = rk(x1, x2)
    return s


def get_sqrt(x):
    '''

    :param x:
    :return: b the sqrt of the matrix x. np.dot(b.T, b) = x
    '''
    u, s, v = np.linalg.svd(x)
    sqrt_s = np.sqrt(s)
    b = np.dot(u, np.dot(np.diag(sqrt_s), v))
    return b

class GamPirls(GLM):

    def _fit_pirls(self, y, spl_x, spl_s, alpha, start_params=None, maxiter=100, tol=1e-8,
                  scale=None, cov_type='nonrobust', cov_kwds=None,
                  use_t=None,):

        rs = get_sqrt(alpha * spl_s) # TODO: this should handle also multivariate analysis i.e. spl_s = [s1, s2, s3...]

        n_samples, n_columns = spl_x.shape
        x1 = np.vstack([spl_x, rs]) # augmented x
        n_samp1es_x1 = x1.shape[0]
        y1 = np.array([0] * n_samp1es_x1) # augmented y
        y1[:n_samples] = y

        params = np.zeros(shape=(n_columns,))
        params[0] = 1
        new_norm = 0
        old_norm = 1
        for i in range(maxiter):
            print('iteration=', i, 'old_norm=', old_norm, 'new_norm=', new_norm)
            if np.abs(new_norm - old_norm) < tol:
                break
            lin_pred = np.dot(x1, params)[:n_samples]
            mu = self.family.fitted(lin_pred)
            z = np.zeros(shape=(n_samp1es_x1,))
            z[:n_samples] = (y - mu) / mu + lin_pred # TODO: review this and the following line
            lm_results = lm.OLS(z, x1).fit()
            lin_pred = np.dot(spl_x, lm_results.params)

            new_mu = self.family.fitted(lin_pred)

            old_norm = new_norm
            new_norm = np.linalg.norm((z[:n_samples] - new_mu))
        return lm_results.params


data = pd.read_csv('/home/donbeo/Documents/statsmodels/statsmodels/sandbox/gam_gsoc2015/tests/results/gam_PIRLS_results.csv')


X = data['x'].as_matrix()
Y = data['y'].as_matrix()

XK = np.array([0.2, .4, .6, .8])

spl_x_R = data[['spl_x.1', 'spl_x.2', 'spl_x.3', 'spl_x.4', 'spl_x.5', 'spl_x.6']].as_matrix()
SPL_X = splines_x(X, XK)

assert_allclose(spl_x_R, SPL_X)

spl_s_R =[[0, 0, 0.000000000,  0.000000000,  0.000000000,  0.000000000],
          [0,    0,  0.000000000,  0.000000000,  0.000000000,  0.000000000],
          [0,    0,  0.001400000,  0.000200000, -0.001133333, -0.001000000],
          [0,    0,  0.000200000,  0.002733333,  0.001666667, -0.001133333],
          [0,    0, -0.001133333,  0.001666667,  0.002733333,  0.000200000],
          [0,    0, -0.001000000, -0.001133333,  0.000200000,  0.001400000]]
SPL_S = splines_s(xk=XK)
assert_allclose(spl_s_R, SPL_S, atol=4.e-10)



b = get_sqrt(SPL_S)

assert_allclose(np.dot(b.T, b), SPL_S)

link = lambda x:x


gam = GamPirls(Y, SPL_X)
params = gam._fit_pirls(Y, SPL_X, SPL_S, 1)
Y_EST = np.dot(SPL_X, params)

plt.plot(X, Y, '.')
plt.plot(X, Y_EST, '.')
plt.show()
