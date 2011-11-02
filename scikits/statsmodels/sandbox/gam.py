"""
Generalized additive models

"""

# JP:
# changes: use PolySmoother instead of crashing bspline
# TODO: check/catalogue required interface of a smoother
# TODO: replace default smoother by corresponding function to initialize
#       other smoothers
# TODO: fix iteration, don't define class with iterator methods, use looping;
#       add maximum iteration and other optional stop criteria
# fixed some of the dimension problems in PolySmoother,
#       now graph for example looks good
# NOTE: example script is now in examples folder

import numpy as np

from scikits.statsmodels.genmod.families import family
from scikits.statsmodels.sandbox.nonparametric.smoothers import PolySmoother as SmoothingSpline   # relative import
from scikits.statsmodels.genmod.generalized_linear_model import GLM

def default_smoother(x):
    _x = x.copy()
    _x.sort()
    n = x.shape[0]
    # taken form smooth.spline in R

    if n < 50:
        nknots = n
    else:
        a1 = np.log(50) / np.log(2)
        a2 = np.log(100) / np.log(2)
        a3 = np.log(140) / np.log(2)
        a4 = np.log(200) / np.log(2)
        if n < 200:
            nknots = 2**(a1 + (a2 - a1) * (n - 50)/150.)
        elif n < 800:
            nknots = 2**(a2 + (a3 - a2) * (n - 200)/600.)
        elif n < 3200:
            nknots = 2**(a3 + (a4 - a3) * (n - 800)/2400.)
        else:
            nknots = 200 + (n - 3200.)**0.2
        knots = _x[np.linspace(0, n-1, nknots).astype(np.int32)]

    s = SmoothingSpline(knots, x=x.copy())
    s.gram(d=2)
    s.target_df = 5
    return s

class Offset(object):

    def __init__(self, fn, offset):
        self.fn = fn
        self.offset = offset

    def __call__(self, *args, **kw):
        return self.fn(*args, **kw) + self.offset

class Results(object):

    def __init__(self, Y, alpha, design, smoothers, family, offset):
        self.Y = Y
        self.alpha = alpha
        self.smoothers = smoothers
        self.offset = offset
        self.family = family
        self.design = design
        self.offset = offset
        self.mu = self(design)

    def __call__(self, design):
        return self.family.link.inverse(self.predict(design))

    def predict(self, design):
        return np.sum(self.smoothed(design), axis=0) + self.alpha

    def smoothed(self, design):
        return np.array([self.smoothers[i]() + self.offset[i] for i in range(design.shape[1])])

class AdditiveModel(object):

    def __init__(self, design, smoothers=None, weights=None):
        self.design = design
        if not weights is None:
            self.weights = weights
        else:
            self.weights = np.ones(self.design.shape[0])

        self.smoothers = smoothers or [default_smoother(design[:,i]) for i in range(design.shape[1])]
        for i in range(design.shape[1]):
            self.smoothers[i].df = 10
        self.family = family.Gaussian()

    def __iter__(self):
        self.iter = 0
        self.dev = np.inf
        return self

    def next(self):
        _results = self.results; Y = self.results.Y
        mu = _results.predict(self.design)
        offset = np.zeros(self.design.shape[1], np.float64)
        alpha = (Y * self.weights).sum() / self.weights.sum()
        for i in range(self.design.shape[1]):
            tmp = self.smoothers[i]()
            self.smoothers[i].smooth(Y - alpha - mu + tmp,
                                     weights=self.weights)
            tmp2 = self.smoothers[i]()
            offset[i] = -(tmp2*self.weights).sum() / self.weights.sum()
            mu += tmp2 - tmp

        #print self.iter
        #self.iter += 1 #missing incrementing of iter counter NOT
        return Results(Y, alpha, self.design, self.smoothers, self.family, offset)

    def cont(self, tol=1.0e-04):
        self.iter += 1 #moved here to always count, not necessary
        print self.iter,
        curdev = (((self.results.Y - self.results.predict(self.design))**2) * self.weights).sum()

        if self.iter > 30: #kill it, no max iterationoption
            return False
        if np.fabs((self.dev - curdev) / curdev) < tol:
            self.dev = curdev
            return False

        #self.iter += 1
        self.dev = curdev
        return True

    def df_resid(self):
        return self.results.Y.shape[0] - np.array([self.smoothers[i].df_fit() for i in range(self.design.shape[1])]).sum()

    def estimate_scale(self):
        return ((self.results.Y - self.results(self.design))**2).sum() / self.df_resid()

    def fit(self, Y):
        iter(self)  # what does this do? anything?
        mu = 0
        alpha = (Y * self.weights).sum() / self.weights.sum()

        offset = np.zeros(self.design.shape[1], np.float64)

        for i in range(self.design.shape[1]):
            self.smoothers[i].smooth(Y - alpha - mu,
                                     weights=self.weights)
            tmp = self.smoothers[i]()
            offset[i] = (tmp * self.weights).sum() / self.weights.sum()
            tmp -= tmp.sum()
            mu += tmp

        self.results = Results(Y, alpha, self.design, self.smoothers, self.family, offset)

        while self.cont():
            self.results = self.next()

        return self.results

class Model(GLM, AdditiveModel):

    niter = 2

#    def __init__(self, design, smoothers=None, family=family.Gaussian()):
#        GLM.__init__(self, design, family=family)
#        AdditiveModel.__init__(self, design, smoothers=smoothers)
#        self.family = family
    def __init__(self, endog, exog, smoothers=None, family=family.Gaussian()):
        GLM.__init__(self, endog, exog, family=family)
        AdditiveModel.__init__(self, exog, smoothers=smoothers)

    def next(self):
        _results = self.results; Y = _results.Y
        if np.isnan(self.weights).all(): print "nanweights1"
        _results.mu = self.family.link.inverse(_results.predict(self.design))
        weights = self.family.weights(_results.mu)
        if np.isnan(weights).all():
            self.weights = weights
            print "nanweights2"
        Z = _results.predict(self.design) + self.family.link.deriv(_results.mu) * (Y - _results.mu)
        m = AdditiveModel(self.design, smoothers=self.smoothers, weights=self.weights)
        _results = m.fit(Z)
        _results.Y = Y
        _results.mu = self.family.link.inverse(_results.predict(self.design))
        self.iter += 1
        self.results = _results

        return _results

    def estimate_scale(self, Y=None):
        """
        Return Pearson\'s X^2 estimate of scale.
        """

        if Y is None:
            Y = self.Y
        resid = Y - self.results.mu
        return (np.power(resid, 2) / self.family.variance(self.results.mu)).sum() / AdditiveModel.df_resid(self)

    def fit(self, Y):
        self.Y = np.asarray(Y, np.float64)

        iter(self)
        alpha = self.Y.mean()
        Z = self.family.link(alpha) + self.family.link.deriv(alpha) * (Y - alpha)
        m = AdditiveModel(self.design, smoothers=self.smoothers)
        self.results = m.fit(Z)
        self.results.mu = self.family.link.inverse(self.results.predict(self.design))
        self.results.Y = Y

        while self.cont():
            self.results = self.next()
            self.scale = self.results.scale = self.estimate_scale()


        return self.results
