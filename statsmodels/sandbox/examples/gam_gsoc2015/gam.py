import patsy
from patsy import dmatrices, dmatrix, demo_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.api import GLM
from scipy.interpolate import splev
import scipy as sp
from numpy.linalg import norm
from statsmodels.discrete.discrete_model import Poisson, Logit, Probit
from statsmodels.base.model import LikelihoodModel
from statsmodels.robust.robust_linear_model import RLM       
import sklearn as sk
from sklearn.linear_model import LogisticRegression
       
### Obtain b splines from patsy ###

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



class PenalizedMixin(object):
    """Mixin class for Maximum Penalized Likelihood
    TODO: missing **kwds or explicit keywords
    TODO: do we really need `pen_weight` keyword in likelihood methods?
    """

    def __init__(self, *args, **kwds):
        super(PenalizedMixin, self).__init__(*args, **kwds)

        penal = kwds.pop('penal', None)
        # I keep the following instead of adding default in pop for future changes
        if penal is None:
            # TODO: switch to unpenalized by default
            self.penal = SCADSmoothed(0.1, c0=0.0001)
        else:
            self.penal = penal

        # TODO: define pen_weight as average pen_weight? i.e. per observation
        # I would have prefered len(self.endog) * kwds.get('pen_weight', 1)
        # or use pen_weight_factor in signature
        self.pen_weight =  kwds.get('pen_weight', len(self.endog))

        self._init_keys.extend(['penal', 'pen_weight'])



    def loglike(self, params, pen_weight=None):
        if pen_weight is None:
            pen_weight = self.pen_weight

        llf = super(PenalizedMixin, self).loglike(params)
        if pen_weight != 0:
            llf -= pen_weight * self.penal.func(params)

        return llf


    def loglikeobs(self, params, pen_weight=None):
        if pen_weight is None:
            pen_weight = self.pen_weight

        llf = super(PenalizedMixin, self).loglikeobs(params)
        nobs_llf = float(llf.shape[0])

        if pen_weight != 0:
            llf -= pen_weight / nobs_llf * self.penal.func(params)

        return llf


    def score(self, params, pen_weight=None):
        if pen_weight is None:
            pen_weight = self.pen_weight

        sc = super(PenalizedMixin, self).score(params)
        if pen_weight != 0:
            sc -= pen_weight * self.penal.grad(params)

        return sc


    def scoreobs(self, params, pen_weight=None):
        if pen_weight is None:
            pen_weight = self.pen_weight

        sc = super(PenalizedMixin, self).scoreobs(params)
        nobs_sc = float(sc.shape[0])
        if pen_weight != 0:
            sc -= pen_weight / nobs_sc  * self.penal.grad(params)

        return sc


    def hessian_(self, params, pen_weight=None):
        if pen_weight is None:
            pen_weight = self.pen_weight
            loglike = self.loglike
        else:
            loglike = lambda p: self.loglike(p, pen_weight=pen_weight)

        from statsmodels.tools.numdiff import approx_hess
        return approx_hess(params, loglike)


    def hessian(self, params, pen_weight=None):
        if pen_weight is None:
            pen_weight = self.pen_weight

        hess = super(PenalizedMixin, self).hessian(params)
        if pen_weight != 0:
            h = self.penal.deriv2(params)
            if h.ndim == 1:
                hess -= np.diag(pen_weight * h)
            else:
                hess -= pen_weight * h

        return hess


    def fit(self, method=None, trim=None, **kwds):
        # If method is None, then we choose a default method ourselves

        # TODO: temporary hack, need extra fit kwds
        # we need to rule out fit methods in a model that will not work with
        # penalization
        if hasattr(self, 'family'):  # assume this identifies GLM
            kwds.update({'max_start_irls' : 0})

        # currently we use `bfgs` by default
        if method is None:
            method = 'bfgs'

        if trim is None:
            trim = False  # see below infinite recursion in `fit_constrained

        res = super(PenalizedMixin, self).fit(method=method, **kwds)

        if trim is False:
            # note boolean check for "is False" not evaluates to False
            return res
        else:
            # TODO: make it penal function dependent
            # temporary standin, only works for Poisson and GLM,
            # and is computationally inefficient
            drop_index = np.nonzero(np.abs(res.params) < 1e-4) [0]
            keep_index = np.nonzero(np.abs(res.params) > 1e-4) [0]
            rmat = np.eye(len(res.params))[drop_index]

            # calling fit_constrained raise
            # "RuntimeError: maximum recursion depth exceeded in __instancecheck__"
            # fit_constrained is calling fit, recursive endless loop
            if drop_index.any():
                # todo : trim kwyword doesn't work, why not?
                #res_aux = self.fit_constrained(rmat, trim=False)
                res_aux = self._fit_zeros(keep_index, **kwds)
                return res_aux
            else:
                return res


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


class GamPenalty(Penalty):
    
    def __init__(self, wts=1, alpha=1, cov_der2=None, der2=None):
        
        self.wts = wts #should we keep wts????
        self.alpha = alpha
        self.cov_der2 = cov_der2
        self.der2 = der2
    
    def func(self, params):
        ''' 
        1) params are the coefficients in the regression model
        2) der2  is the second derivative of the splines basis
        '''
        
        # The second derivative of the estimated regression function
        f = np.dot(self.der2, params)

        return self.alpha * np.sum(f**2)

    def grad(self, params):
        ''' 
        1) params are the coefficients in the regression model
        2) der2  is the second derivative of the splines basis
        3) cov_der2 is obtained as np.dot(der2.T, der2)
        '''
                
        return 2 * self.alpha * np.dot(self.cov_der2, params) 
    
    def deriv2(self, params):

        return 2 * self.alpha * self.cov_der2


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



class LogitGam(PenalizedMixin, Logit):
    pass



class GLMGam(PenalizedMixin, GLM):
    pass

n = 200
data = pd.DataFrame()
x = np.linspace(-10, 10, n)
poly = x*x
yc = 1/(1 + np.exp(-poly)) + np.random.normal(0, 0.1, n)
y = yc.copy()
y[yc>yc.mean()] = 1
y[yc<=yc.mean()] = 0 
d = {"x": x}
dm = dmatrix("bs(x, df=5, degree=2, include_intercept=True)", d)


   

df = 10
degree = 7
order = degree + 1
n_inner_knots = df - order
lower_bound = np.min(x)
upper_bound = np.max(x)
knot_quantiles = np.linspace(0, 1, n_inner_knots + 2)[1:-1]
inner_knots = _R_compat_quantile(x, knot_quantiles)
all_knots = np.concatenate(([lower_bound, upper_bound] * order, inner_knots))

basis, der_basis, der2_basis = _eval_bspline_basis(x, all_knots, degree)

params0 = np.random.normal(0, 1, df)


# add a column of ones to simulate the intercept
one = np.ones(shape=(n,1))
basis = np.hstack([basis, one])
der2_basis = np.hstack([der2_basis, one])


cov_der2 = np.dot(der2_basis.T, der2_basis)
g = LogitGam(y, basis, 
             penal=GamPenalty(wts=1, alpha=0.0001, cov_der2=cov_der2, 
                              der2=der2_basis))
res_g = g.fit()
plt.subplot(3, 1, 1)
plt.plot(x, np.dot(basis, res_g.params))
plt.plot(x, poly, 'o')
plt.subplot(3, 1, 2)
plt.plot(x, 1/(1+np.exp(-np.dot(basis, res_g.params))))
plt.plot(x, yc, 'o')
plt.subplot(3, 1, 3)
lr = LogisticRegression().fit(basis, y)
plt.plot(x, lr.predict_proba(basis)[:, 1])
plt.plot(x, yc, 'o')
plt.show()



## observe that alpha is set to inf ## 
y2 = x*x + np.random.normal(0, 10, n)
glm_gam = GLMGam(y2, basis, 
                 penal=GamPenalty(wts=1, alpha=np.inf, cov_der2=cov_der2, 
                                  der2=der2_basis))
res_glm_gam = glm_gam.fit()
plt.plot(x, np.dot(basis, res_glm_gam.params))
plt.plot(x, y2, 'o')
plt.show()


