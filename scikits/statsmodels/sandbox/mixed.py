"""
Mixed effects models

Author: Jonathan Taylor
Author: Josef Perktold
License: BSD-3


Notes
------

It's pretty slow if the model is misspecified, in my first example convergence
in loglike is not reached within 2000 iterations. Added stop criteria based
on convergence of parameters instead.

With correctly specified model, convergence is fast, in 6 iterations in
example.

"""

import numpy as np
import numpy.linalg as L

from scikits.statsmodels.base.model import LikelihoodModelResults


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


    Parameters
    ----------
    endog : ndarray, (nobs,)
        response, endogenous variable
    exog_fe : ndarray, (nobs, k_vars_fe)
        explanatory variables as regressors or fixed effects,
        should include exog_re to correct mean of random
        coefficients, see Notes
    exog_re : ndarray, (nobs, k_vars_re)
        explanatory variables or random effects or coefficients

    Notes
    -----
    If the exog_re variables are not included in exog_fe, then the
    mean of the random constants or coefficients are not centered.
    The covariance matrix of the random parameter estimates are not
    centered in this case. (That's how it looks to me. JP)

    """


    def __init__(self, endog, exog_fe, exog_re):

        self.Y = endog
        self.X = exog_fe
        self.Z = exog_re
        self.n = endog.shape[0]

    def _compute_S(self, D, sigma):
        """covariance of observations (nobs_i, nobs_i)  (JP check)
        Display (3.3) from Laird, Lange, Stram (see help(Unit))
        """
        self.S = (np.identity(self.n) * sigma**2 +
                  np.dot(self.Z, np.dot(D, self.Z.T)))

    def _compute_W(self):
        """inverse covariance of observations (nobs_i, nobs_i)  (JP check)
        Display (3.2) from Laird, Lange, Stram (see help(Unit))
        """
        self.W = L.inv(self.S)

    def compute_P(self, Sinv):
        """projection matrix (nobs_i, nobs_i) (M in regression ?)  (JP check, guessing)
        Display (3.10) from Laird, Lange, Stram (see help(Unit))

        W - W X Sinv X' W'
        """
        t = np.dot(self.W, self.X)
        self.P = self.W - np.dot(np.dot(t, Sinv), t.T)

    def _compute_r(self, alpha):
        """residual after removing fixed effects

        Display (3.5) from Laird, Lange, Stram (see help(Unit))
        """
        self.r = self.Y - np.dot(self.X, alpha)

    def _compute_b(self, D):
        """coefficients for random effects/coefficients
        Display (3.4) from Laird, Lange, Stram (see help(Unit))

        D Z' W r
        """
        self.b = np.dot(D, np.dot(np.dot(self.Z.T, self.W), self.r))

    def fit(self, a, D, sigma):
        """
        Compute unit specific parameters in
        Laird, Lange, Stram (see help(Unit)).

        Displays (3.2)-(3.5).
        """

        self._compute_S(D, sigma)    #random effect plus error covariance
        self._compute_W()            #inv(S)
        self._compute_r(a)           #residual after removing fixed effects/exogs
        self._compute_b(D)           #?  coefficients on random exog, Z ?

    def compute_xtwy(self):
        """
        Utility function to compute X^tWY (transposed ?) for Unit instance.
        """
        return np.dot(np.dot(self.W, self.Y), self.X) #is this transposed ?

    def compute_xtwx(self):
        """
        Utility function to compute X^tWX for Unit instance.
        """
        return np.dot(np.dot(self.X.T, self.W), self.X)

    def cov_random(self, D, Sinv=None):
        """
        Approximate covariance of estimates of random effects. Just after
        Display (3.10) in Laird, Lange, Stram (see help(Unit)).

        D - D' Z' P Z D

        Notes
        -----
        In example where the mean of the random coefficient is not zero, this
        is not a covariance but a non-centered moment. (proof by example)

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

        no constant with pi included

        a is not used if ML=true  (should be a=None in signature)
        If ML is false, then the residuals are calculated for the given fixed
        effects parameters a.
        """

        if ML:
            return (np.log(L.det(self.W)) - (self.r * np.dot(self.W, self.r)).sum()) / 2.
        else:
            if a is None:
                raise ValueError('need fixed effect a for REML contribution to log-likelihood')
            r = self.Y - np.dot(self.X, a)
            return (np.log(L.det(self.W)) - (r * np.dot(self.W, r)).sum()) / 2.

    def deviance(self, ML=False):
        '''deviance defined as 2 times the negative loglikelihood

        '''
        return - 2 * self.logL(ML=ML)


class OneWayMixed(object):

    """
    Model for
    EM implementation of (repeated measures)
    mixed effects model.

    \'Maximum Likelihood Computations with Repeated Measures:
    Application of the EM Algorithm\'

    Nan Laird; Nicholas Lange; Daniel Stram

    Journal of the American Statistical Association,
    Vol. 82, No. 397. (Mar., 1987), pp. 97-105.


    Parameters
    ----------
    units : list of units
       the data for the individual units should be attached to the units
    response, fixed and random : formula expression, called as argument to Formula


    *available results and alias*

    (subject to renaming, and coversion to cached attributes)

    params() -> self.a : coefficient for fixed effects or exog
    cov_params() -> self.Sinv : covariance estimate of fixed effects/exog
    bse() : standard deviation of params

    cov_random -> self.D : estimate of random effects covariance
    params_random_units -> [self.units[...].b] : random coefficient for each unit


    *attributes*

    (others)

    self.m : number of units
    self.p : k_vars_fixed
    self.q : k_vars_random
    self.N : nobs (total)


    Notes
    -----
    Fit returns a result instance, but not all results that use the inherited
    methods have been checked.

    Parameters need to change: drop formula and we require a naming convention for
    the units (currently Y,X,Z). - endog, exog_fe, endog_re ?

    logL does not include constant, e.g. sqrt(pi)


    convergence criteria for iteration
    Currently convergence in the iterative solver is reached if either the loglikelihood
    *or* the fixed effects parameter don't change above tolerance.

    In some examples, the fixed effects parameters converged to 1e-5 within 150 iterations
    while the log likelihood did not converge within 2000 iterations. This might be
    the case if the fixed effects parameters are well estimated, but there are still
    changes in the random effects. If params_rtol and params_atol are set at a higher
    level, then the random effects might not be estimated to a very high precision.

    The above was with a misspecified model, without a constant. With a
    correctly specified model convergence is fast, within a few iterations
    (6 in example).

    """

    def __init__(self, units):
        self.units = units
        self.m = len(self.units)

        self.N = sum(unit.X.shape[0] for unit in self.units)
        self.n_units = self.N     #alias for now

        # Determine size of fixed effects
        d = self.units[0].X
        self.p = d.shape[1]  # d.shape = p
        self.k_exog_fe = self.p   #alias for now
        self.a = np.zeros(self.p, np.float64)

        # Determine size of D, and sensible initial estimates
        # of sigma and D
        d = self.units[0].Z
        self.q = d.shape[1]  # Z.shape = q
        self.k_exog_re = self.q   #alias for now
        self.D = np.zeros((self.q,)*2, np.float64)
        self.sigma = 1.

        self.dev = np.inf   #initialize for iterations, move it?

    def _compute_a(self):
        """fixed effects parameters

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

        sigma is the standard deviation of the noise (residual)

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
        Approximate covariance of estimates of fixed effects.

        Just after Display (3.10) in Laird, Lange, Stram (see help(Mixed)).
        """
        return self.Sinv

    #----------- alias (JP)   move to results class ?

    def cov_random(self):
        """
        Estimate random effects covariance D.

        If ML is True, return the ML estimate of sigma, else return the REML estimate.

        see _compute_D, alias for self.D
        """
        return self.D

    @property
    def params(self):
        '''
        estimated coefficients for exogeneous variables or fixed effects

        see _compute_a, alias for self.a
        '''
        return self.a

    @property
    def params_random_units(self):
        '''random coefficients for each unit

        JP: I think, there is no restriction on the kvars in Z, the random effects
        design matrix, for each unit to be the same, maybe for D? yes there is.
        I let it raise an exception for now if it cannot be converted to array.

        '''
        return np.array([unit.b for unit in self.units])

    def cov_params(self):
        '''
        estimated covariance for coefficients for exogeneous variables or fixed effects

        see cov_fixed, and Sinv in _compute_a
        '''
        return self.cov_fixed()


    @property
    def bse(self):
        '''
        standard errors of estimated coefficients for exogeneous variables (fixed)

        '''
        np.sqrt(np.diag(self.cov_params()))

    #----------- end alias

    def deviance(self, ML=False):
        '''deviance defined as 2 times the negative loglikelihood

        '''
        return -2 * self.logL(ML=ML)

    def logL(self, ML=False):
        """
        Return log-likelihood, REML by default.

        """
        #I don't know what the difference between REML and ML is here.
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

        #TODO: JP added df_resid check
        self.df_resid = (self.N - (self.m - 1) * self.q - self.p)
        sigmasq /= (self.N - (self.m - 1) * self.q - self.p)
        self.sigma = np.sqrt(sigmasq)
        self.D = (D - sigmasq * t) / self.m

    def cont(self, ML=False, rtol=1.0e-05, params_rtol=1e-5, params_atol=1e-4):
        '''convergence check for iterative estimation

        '''

        self.dev, old = self.deviance(ML=ML), self.dev

        #self.history.append(np.hstack((self.dev, self.a)))
        self.history['llf'].append(self.dev)
        self.history['params'].append(self.a.copy())
        self.history['D'].append(self.D.copy())

        if np.fabs((self.dev - old) / self.dev) < rtol:   #why is there times `*`?
            #print np.fabs((self.dev - old)), self.dev, old
            self.termination = 'llf'
            return False

        #break if parameters converged
        #TODO: check termination conditions, OR or AND
        if np.all(np.abs(self.a - self._a_old) < (params_rtol * self.a + params_atol)):
            self.termination = 'params'
            return False

        self._a_old =  self.a.copy()
        return True

    def fit(self, maxiter=100, ML=False, rtol=1.0e-05, params_rtol=1e-6, params_atol=1e-6):

        #initialize for convergence criteria
        self._a_old = np.inf * self.a
        self.history = {'llf':[], 'params':[], 'D':[]}

        for i in range(maxiter):
            self._compute_a()              #a, Sinv :  params, cov_params of fixed exog
            self._compute_sigma(ML=ML)     #sigma   MLE or REML of sigma ?
            self._compute_D(ML=ML)         #D :  covariance of random effects, MLE or REML
            if not self.cont(ML=ML, rtol=rtol, params_rtol=params_rtol,
                                             params_atol=params_atol):
                break
        else: #if end of loop is reached without break
            self.termination = 'maxiter'
            print 'Warning: maximum number of iterations reached'

        self.iterations = i

        results = OneWayMixedResults(self)
        #compatibility functions for fixed effects/exog
        results.scale = 1
        results.normalized_cov_params = self.cov_params()
        return results


class OneWayMixedResults(LikelihoodModelResults):
    '''Results class for OneWayMixed models

    '''
    def __init__(self, model):
        self.model = model
        self.params = model.params

    @property
    def params_random_units(self):
        return self.model.params_random_units

    def cov_random(self):
        return self.model.cov_random()

    def mean_random(self, idx='lastexog'):
        if idx == 'lastexog':
            meanr = self.params[self.model.k_exog_re:]
        elif type(idx) == list:
            if not len(idx) == self.model.k_exog_re:
                raise ValueError
            else:
                meanr = self.params[idx]
        else:
            meanr = np.zeros(self.model.k_exog_re)

        return meanr

    def std_random(self):
        return np.sqrt(np.diag(self.cov_random()))

    def plot_random_univariate(self, bins=None):
        #outsource this
        import matplotlib.pyplot as plt
        from scipy.stats import norm as normal
        fig = plt.figure()
        k = self.model.k_exog_re
        if k > 3:
            rows, cols = int(np.ceil(5 * 0.5)), 2
        else:
            rows, cols = k, 1
        if bins is None:
            #bins = self.model.n_units // 20    #TODO: just roughly, check
            bins = np.sqrt(self.model.n_units)

        for ii in range(k):
            ax = fig.add_subplot(rows, cols, ii)
            freq, bins, _ = ax.hist(self.params_random_units[:,ii], bins=bins, normed=True)
            points = np.linspace(bins[0], bins[-1], 200)
            loc = self.mean_random()[ii]
            scale = self.std_random()[ii]
            #ax.plot(points, normal.pdf(points, loc=loc, scale=scale))
            #loc of sample is approx. zero, with Z appended to X
            #alternative, add fixed  to mean
            ax.plot(points, normal.pdf(points, scale=scale))


        #next is only temporarily here
        fig2 = plt.figure()
        ax = fig2.add_subplot(1,1,1)
        re1, re2 = self.params_random_units.T
        ax.plot(re1, re2, 'o', alpha=0.75)
        return fig, fig2


if __name__ == '__main__':
    #see examples/ex_mixed_lls_1.py
    pass
