import patsy
from patsy import dmatrices, dmatrix, demo_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tools.numdiff import approx_fprime
from scipy.interpolate import splev
import scipy as sp
from numpy.linalg import norm



class Penalty(object):
    """
    A class for representing a scalar-value penalty.
    Parameters
    wts : array-like
        A vector of weights that determines the weight of the penalty
        for each parameter.
    Notes
    -----
    The class has a member called `alpha` that scales the weights.
    """

    def __init__(self, wts):
        self.wts = wts
        self.alpha = 1.

    def func(self, params):
        """
        A penalty function on a vector of parameters.
        Parameters
        ----------
        params : array-like
            A vector of parameters.
        Returns
        -------
        A scalar penaty value; greater values imply greater
        penalization.
        """
        raise NotImplementedError

    def grad(self, params):
        """
        The gradient of a penalty function.
        Parameters
        ----------
        params : array-like
            A vector of parameters
        Returns
        -------
        The gradient of the penalty with respect to each element in
        `params`.
        """
        raise NotImplementedError


class L2(Penalty):
    """
    The L2 (ridge) penalty.
    """

    def __init__(self, wts=None):
        if wts is None:
            self.wts = 1.
        else:
            self.wts = wts
        self.alpha = 1.

    def func(self, params):
        return np.sum(self.wts * self.alpha * params**2)

    def grad(self, params):
        return 2 * self.wts * self.alpha * params


def _R_compat_quantile(x, probs):
    #return np.percentile(x, 100 * np.asarray(probs))
    probs = np.asarray(probs)
    quantiles = np.asarray([np.percentile(x, 100 * prob)
                            for prob in probs.ravel(order="C")])
    return quantiles.reshape(probs.shape, order="C")


## from patsy splines.py
def _eval_bspline_basis(x, knots, degree):
    try:
        from scipy.interpolate import splev
    except ImportError: # pragma: no cover
        raise ImportError("spline functionality requires scipy")
    # 'knots' are assumed to be already pre-processed. E.g. usually you
    # want to include duplicate copies of boundary knots; you should do
    # that *before* calling this constructor.
    knots = np.atleast_1d(np.asarray(knots, dtype=float))
    assert knots.ndim == 1
    knots.sort()
    degree = int(degree)
    x = np.atleast_1d(x)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    assert x.ndim == 1
    # XX FIXME: when points fall outside of the boundaries, splev and R seem
    # to handle them differently. I don't know why yet. So until we understand
    # this and decide what to do with it, I'm going to play it safe and
    # disallow such points.
    if np.min(x) < np.min(knots) or np.max(x) > np.max(knots):
        raise NotImplementedError("some data points fall outside the "
                                  "outermost knots, and I'm not sure how "
                                  "to handle them. (Patches accepted!)")
    # Thanks to Charles Harris for explaining splev. It's not well
    # documented, but basically it computes an arbitrary b-spline basis
    # given knots and degree on some specificed points (or derivatives
    # thereof, but we don't use that functionality), and then returns some
    # linear combination of these basis functions. To get out the basis
    # functions themselves, we use linear combinations like [1, 0, 0], [0,
    # 1, 0], [0, 0, 1].
    # NB: This probably makes it rather inefficient (though I haven't checked
    # to be sure -- maybe the fortran code actually skips computing the basis
    # function for coefficients that are zero).
    # Note: the order of a spline is the same as its degree + 1.
    # Note: there are (len(knots) - order) basis functions.
    n_bases = len(knots) - (degree + 1)
    basis = np.empty((x.shape[0], n_bases), dtype=float)
    der1_basis = np.empty((x.shape[0], n_bases), dtype=float)
    der2_basis = np.empty((x.shape[0], n_bases), dtype=float)

    for i in range(n_bases):
        coefs = np.zeros((n_bases,))
        coefs[i] = 1
        basis[:, i] = splev(x, (knots, coefs, degree))
        der1_basis[:, i] = splev(x, (knots, coefs, degree), der=1)
        der2_basis[:, i] = splev(x, (knots, coefs, degree), der=2)


    return basis, der1_basis, der2_basis


class GamPenalty(Penalty):

    def __init__(self, wts=1, alpha=1):

        self.wts = wts #should we keep wts????
        self.alpha = alpha

    def func(self, params, der2):
        '''
        1) params are the coefficients in the regression model
        2) der2  is the second derivative of the splines basis
        '''

        # The second derivative of the estimated regression function
        f = np.dot(der2, params)

        return self.alpha * np.sum(f**2)

    def grad(self, params, der2, cov_der2):
        '''
        1) params are the coefficients in the regression model
        2) der2  is the second derivative of the splines basis
        3) cov_der2 is obtained as np.dot(der2.T, der2)
        '''

        return 2 * self.alpha * np.dot(cov_der2, params)


gp = GamPenalty()



class MSE():

    def __init__(self):
        return

    def func(self, X, y, params):

        return norm(y - np.dot(X, params))

    def grad(self, X, y, params):

        return 2 * np.sum(X.T * (y - np.dot(X, params)), axis=1)


def cost(params, alpha, basis, der2, y):
    GB = GamPenalty(alpha=alpha)
    penalty = GB.func(params, der2)
    mse = norm(y - np.dot(basis, params))
    return mse + penalty

def grad_cost(params, alpha, basis, der2, y):
    GB = GamPenalty(alpha=alpha)
    cov_der2 = np.dot(der2.T, der2)

    grad_penalty = GB.grad(params, der2, cov_der2)
    mse = MSE()
    grad_mse = mse.grad(basis, y, params)
    return grad_penalty + grad_mse


n = 200
data = pd.DataFrame()
x = np.linspace(-1, 1, n)
y = x*x - x
d = {"x": x}
dm = dmatrix("bs(x, df=5, degree=2, include_intercept=True)", d)




df = 10
degree = 5
order = degree + 1
n_inner_knots = df - order
lower_bound = np.min(x)
upper_bound = np.max(x)
knot_quantiles = np.linspace(0, 1, n_inner_knots + 2)[1:-1]
inner_knots = _R_compat_quantile(x, knot_quantiles)
all_knots = np.concatenate(([lower_bound, upper_bound] * order, inner_knots))

basis, der_basis, der2_basis = _eval_bspline_basis(x, all_knots, degree)
params = np.random.normal(0, 1, df)


alpha = 0.0001

opt = sp.optimize.minimize(cost, params, args=(alpha, basis, der2_basis, y))

## opt1 is not working properly
opt1 = sp.optimize.minimize(cost, params, args=(alpha, basis, der2_basis, y),
                            jac=grad_cost, method='Newton-CG')

beta = opt.x

plt.subplot(2, 1, 1)
plt.title('alpha = ' + str(alpha))
plt.plot(x, np.dot(basis, beta), label='GAM')
plt.plot(x, y, label='True')
plt.legend()
plt.subplot(2, 1, 2)
plt.title('alpha = ' + str(alpha))
plt.plot(x, np.dot(basis, opt1.x), label='GAM')
plt.plot(x, y, label='True')
plt.legend()
plt.show()
