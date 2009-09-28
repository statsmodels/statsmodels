"""
Mixed effects models


Notes
------
This still depends on nipy
"""

import numpy as np
import numpy.linalg as L
#import nipy

from scikits.statsmodels.sandbox.formula import Formula, I

class Unit(object):
    """
    Individual experimental unit for
    EM implementation of (repeated measures)
    mixed effects model.

    \'Maximum Likelihood Computations with Repeated Measures:
    Application of the EM Algorithm\'

    Nan Laird; Nicholas Lange; Daniel Stram

    Journal of the American Statistical Association,
    Vol. 82, No. 397. (Mar., 1987), pp. 97-105.
    """

    def __getitem__(self, item):
        return self.dict[item]

    def __setitem__(self, item, value):
        self.dict[item] = value

    def __init__(self, dict_):
        self.dict = dict_ # don't use build in names

    def __call__(self, formula, **extra):
        """
        Return the corresponding design matrix from formula,
        perform a check whether formula just has an intercept in it, in
        which case the number of rows must be computed.
        """
        if hasattr(self, 'n') and 'nrow' not in extra:
            extra['nrow'] = self.n
        return formula(namespace=self.dict, **extra)

    def design(self, formula, **extra):
        v = np.transpose(self(formula, **extra))
        self.n = v.shape[0]
        return v

    def _compute_S(self, D, sigma):
        """
        Display (3.3) from Laird, Lange, Stram (see help(Unit))
        """
        self.S = (np.identity(self.n) * sigma**2 +
                  np.dot(self.Z, np.dot(D, self.Z.T)))

    def _compute_W(self):
        """
        Display (3.2) from Laird, Lange, Stram (see help(Unit))
        """
        self.W = L.inv(self.S)

    def compute_P(self, Sinv):
        """
        Display (3.10) from Laird, Lange, Stram (see help(Unit))
        """
        t = np.dot(self.W, self.X)
        self.P = self.W - np.dot(np.dot(t, Sinv), t.T)

    def _compute_r(self, alpha):
        """
        Display (3.5) from Laird, Lange, Stram (see help(Unit))
        """
        self.r = self.Y - np.dot(self.X, alpha)

    def _compute_b(self, D):
        """
        Display (3.4) from Laird, Lange, Stram (see help(Unit))
        """
        self.b = np.dot(D, np.dot(np.dot(self.Z.T, self.W), self.r))

    def fit(self, a, D, sigma):
        """
        Compute unit specific parameters in
        Laird, Lange, Stram (see help(Unit)).

        Displays (3.2)-(3.5).
        """

        self._compute_S(D, sigma)
        self._compute_W()
        self._compute_r(a)
        self._compute_b(D)

    def compute_xtwy(self):
        """
        Utility function to compute X^tWY for Unit instance.
        """
        return np.dot(np.dot(self.W, self.Y), self.X)

    def compute_xtwx(self):
        """
        Utility function to compute X^tWX for Unit instance.
        """
        return np.dot(np.dot(self.X.T, self.W), self.X)

    def cov_random(self, D, Sinv=None):
        """
        Approximate covariance of estimates of random effects. Just after
        Display (3.10) in Laird, Lange, Stram (see help(Unit)).
        """
        if Sinv is not None:
            self.compute_P(Sinv)
        t = np.dot(self.Z, D)
        return D - np.dot(np.dot(t.T, self.P), t)

    def logL(self, a, ML=False):
        """
        Individual contributions to the log-likelihood, tries to return REML
        contribution by default though this requires estimated
        fixed effect a to be passed as an argument.
        """

        if ML:
            return (np.log(L.det(self.W)) - (self.r * np.dot(self.W, self.r)).sum()) / 2.
        else:
            if a is None:
                raise ValueError, 'need fixed effect a for REML contribution to log-likelihood'
            r = self.Y - np.dot(self.X, a)
            return (np.log(L.det(self.W)) - (r * np.dot(self.W, r)).sum()) / 2.

    def deviance(self, ML=False):
        return - 2 * self.logL(ML=ML)

class Mixed(object):

    """
    Model for
    EM implementation of (repeated measures)
    mixed effects model.

    \'Maximum Likelihood Computations with Repeated Measures:
    Application of the EM Algorithm\'

    Nan Laird; Nicholas Lange; Daniel Stram

    Journal of the American Statistical Association,
    Vol. 82, No. 397. (Mar., 1987), pp. 97-105.
    """

    def __init__(self, units, response, fixed=I, random=I):
        self.units = units
        self.m = len(self.units)

        self.fixed = Formula(fixed)
        self.random = Formula(random)
        self.response = Formula(response)

        self.N = 0
        for unit in self.units:
            unit.Y = np.squeeze(unit.design(self.response)) # respnose is just 'y'
            unit.X = unit.design(self.fixed)
            unit.Z = unit.design(self.random)
            self.N += unit.X.shape[0]

        # Determine size of fixed effects

        d = self.units[0].design(self.fixed)
        self.p = d.shape[1]  # d.shape = p
        self.a = np.zeros(self.p, np.float64)

        # Determine size of D, and sensible initial estimates
        # of sigma and D
        d = self.units[0].design(self.random)
        self.q = d.shape[1]  # d.shape = q

        self.D = np.zeros((self.q,)*2, np.float64)
        self.sigma = 1.

        self.dev = np.inf

    def _compute_a(self):
        """
        Display (3.1) of
        Laird, Lange, Stram (see help(Mixed)).

        """

        for unit in self.units:
            unit.fit(self.a, self.D, self.sigma)

        S = sum([unit.compute_xtwx() for unit in self.units])
        Y = sum([unit.compute_xtwy() for unit in self.units])

        self.Sinv = L.pinv(S)
        self.a = np.dot(self.Sinv, Y)

    def _compute_sigma(self, ML=False):
        """
        Estimate sigma. If ML is True, return the ML estimate of sigma,
        else return the REML estimate.

        If ML, this is (3.6) in Laird, Lange, Stram (see help(Mixed)),
        otherwise it corresponds to (3.8).

        """
        sigmasq = 0.
        for unit in self.units:
            if ML:
                W = unit.W
            else:
                unit.compute_P(self.Sinv)
                W = unit.P
            t = unit.r - np.dot(unit.Z, unit.b)
            sigmasq += np.power(t, 2).sum()
            sigmasq += self.sigma**2 * np.trace(np.identity(unit.n) -
                                               self.sigma**2 * W)
        self.sigma = np.sqrt(sigmasq / self.N)

    def _compute_D(self, ML=False):
        """
        Estimate random effects covariance D.
        If ML is True, return the ML estimate of sigma,
        else return the REML estimate.

        If ML, this is (3.7) in Laird, Lange, Stram (see help(Mixed)),
        otherwise it corresponds to (3.9).

        """
        D = 0.
        for unit in self.units:
            if ML:
                W = unit.W
            else:
                unit.compute_P(self.Sinv)
                W = unit.P
            D += np.multiply.outer(unit.b, unit.b)
            t = np.dot(unit.Z, self.D)
            D += self.D - np.dot(np.dot(t.T, W), t)

        self.D = D / self.m

    def cov_fixed(self):
        """
        Approximate covariance of estimates of fixed effects. Just after
        Display (3.10) in Laird, Lange, Stram (see help(Mixed)).
        """
        return self.Sinv

    def deviance(self, ML=False):
        return -2 * self.logL(ML=ML)

    def logL(self, ML=False):
        """
        Return log-likelihood, REML by default.
        """
        logL = 0.

        for unit in self.units:
            logL += unit.logL(a=self.a, ML=ML)
        if not ML:
            logL += np.log(L.det(self.Sinv)) / 2
        return logL

    def initialize(self):
        S = sum([np.dot(unit.X.T, unit.X) for unit in self.units])
        Y = sum([np.dot(unit.X.T, unit.Y) for unit in self.units])
        self.a = L.lstsq(S, Y)[0]

        D = 0
        t = 0
        sigmasq = 0
        for unit in self.units:
            unit.r = unit.Y - np.dot(unit.X, self.a)
            if self.q > 1:
                unit.b = L.lstsq(unit.Z, unit.r)[0]
            else:
                Z = unit.Z.reshape((unit.Z.shape[0], 1))
                unit.b = L.lstsq(Z, unit.r)[0]

            sigmasq += (np.power(unit.Y, 2).sum() -
                        (self.a * np.dot(unit.X.T, unit.Y)).sum() -
                        (unit.b * np.dot(unit.Z.T, unit.r)).sum())
            D += np.multiply.outer(unit.b, unit.b)
            t += L.pinv(np.dot(unit.Z.T, unit.Z))

        sigmasq /= (self.N - (self.m - 1) * self.q - self.p)
        self.sigma = np.sqrt(sigmasq)
        self.D = (D - sigmasq * t) / self.m

    def cont(self, ML=False, tol=1.0e-05):

        self.dev, old = self.deviance(ML=ML), self.dev
        if np.fabs((self.dev - old)) * self.dev < tol:
            return False
        return True

    def fit(self, niter=100, ML=False):

        for i in range(niter):
            self._compute_a()
            self._compute_sigma(ML=ML)
            self._compute_D(ML=ML)
            if not self.cont(ML=ML):
                break


if __name__ == '__main__':
    import numpy.random as R
    R.seed(54321)
    nsubj = 400
    units  = []

    n = 3

    from scikits.statsmodels.sandbox.formula import Term
    fixed = Term('f')
    random = Term('r')
    response = Term('y')

    nx = 4
    beta = np.ones(nx)
    for i in range(nsubj):
        d = R.standard_normal()
        X = R.standard_normal((nx,n))
        Z = X[0:2]
        #Y = R.standard_normal((n,)) + d * 4
        Y = np.dot(X.T,beta) + d * 4
        units.append(Unit({'f':X, 'r':Z, 'y':Y}))

    #m = Mixed(units, response)#, fixed, random)
    m = Mixed(units, response, fixed, random)
    #m = Mixed(units, response, fixed + random, random)
    m.initialize()
    m.fit()
    #print dir(m)
    #print vars(m)
    print 'estimates for fixed effects'
    print m.a
    bfixed_cov = m.cov_fixed()
    print 'beta fixed standard errors'
    print np.sqrt(np.diag(bfixed_cov))




    a = Unit({})
    a['x'] = np.array([2,3])
    a['y'] = np.array([3,4])

    x = Term('x')
    y = Term('y')

    fixed = x + y + x * y
    random = Formula(x)

    a.X = a.design(fixed)
    a.Z = a.design(random)

    print help(a._compute_S)
