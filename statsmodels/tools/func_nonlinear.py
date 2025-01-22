"""
examples and special cases of non-linear functions

Created on Jun. 25, 2024 5:33:47 a.m.

Author: Josef Perktold
License: BSD-3
"""

import numpy as np

class Predictor():
    def __init__(self, exog=None):
        self.exog = exog

    def predict_deriv(self, params, exog=None):
        if exog is None:
            exog = self.exog
        from statsmodels.tools.numdiff import approx_fprime
        return approx_fprime(params[:-1], self.predict)


class Linear(Predictor):
    """Exponential mean function (inverse link)
    """

    def predict(self, params, exog=None, linear=False):
        if exog is None:
            exog = self.exog
        xb = np.dot(exog, params[:self.exog.shape[1]])
        return xb


    def predict_deriv(self, params, exog=None, linear=False):
        if exog is None:
            exog = self.exog
        return exog


def func_menten(params, x):
    a, b = params
    return a * x / (np.exp(b) + x)


class MentenNL(Predictor):
    """Michaelis–Menten function.
    """

    def param_names(self):
        return ["a", "b"]

    def predict(self, params, exog=None):
        if exog is None:
            exog = self.exog
        return np.squeeze(func_menten(params[:2], exog))

    def predict_deriv(self, params, exog=None):
        if exog is None:
            exog = self.exog
        from statsmodels.tools.numdiff import approx_fprime
        return approx_fprime(params[:2], self.predict)


class ExpNL(Predictor):
    """Exponential mean function (inverse link)
    """

    def predict(self, params, exog=None, linear=False):
        if exog is None:
            exog = self.exog
        xb = np.dot(exog, params[:self.exog.shape[1]])
        if linear:
            return xb
        else:
            return np.exp(xb)

    def predict_deriv(self, params, exog=None, linear=False):
        if exog is None:
            exog = self.exog
        if linear:
            return exog
        else:
            xb = np.dot(exog, params[:self.exog.shape[1]])
            return np.exp(xb)[:, None] * exog


def sigmoid(params, x):
    x0, y0, c, k = params
    y = c / (1. + np.exp(-k * (x - x0))) + y0
    return y


def sigmoid_deriv(params, x):
    x0, y0, c, k = params  # noqa
    term = np.exp(-k * (x - x0))
    denom = 1. / (1 + term)
    denom2 = denom**2
    dx0 =  - c * denom2 * term * k
    dy0 = np.ones(x.shape[0])
    dc = denom
    dk =  c * denom2 * term * (x - x0)

    return np.column_stack([dx0, dy0, dc, dk])

def sig_start(y, x):
    #return x.min(), y.min(), y.max(), np.corrcoef(x, y)[0, 1]
    return np.median(x), np.median(y), y.max(), np.corrcoef(x, y)[0, 1]


class SigmoidNL(Predictor):

    def predict(self, params, exog=None, linear=False):
        if exog is None:
            exog = self.exog
        xb = np.dot(exog, params[:self.exog.shape[1]])
        if linear:
            return xb
        else:
            return np.squeeze(sigmoid(params[:4], exog))

    def predict_deriv(self, params, exog=None, linear=False):
        if exog is None:
            exog = self.exog
        if linear:
            # doesn't make sense in this case
            return exog
        else:
            #xb = np.dot(exog, params[:self.exog.shape[1]])
            return sigmoid_deriv(params[:4], exog)

    def get_start_params(self, endog, exog=None):
        if exog is None:
            exog = self.exog
        return sig_start(endog, exog)
