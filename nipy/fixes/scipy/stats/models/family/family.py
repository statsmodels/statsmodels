'''
Family provides the distributions used by GLM.

Usage example (specifies defaults)
-------------
import models
from models.family import links

models.family.Family
    general base class

    Methods
    --------
    weights
    deviance
    devresid

    fitted :
        fitted values based on linear predictor eta

        Parameters
        ---------
        eta : array-like
            XBeta in a classic linear model

        outputs
        -------
        mu : array-like
            mean parameter based on linear predictor eta
            link.inverse(eta) where the link is either the default link
            or specified

    predict : array-like
        Returns the linear predictors based on given mu values.

        Parameters
        -----------
            mu : array-like

        Outputs
        -------
            eta -- link(mu)

models.family.Binomial(link = links.logit)
    available links are logit, probit, log, cauchy, cloglog
models.family.Gamma(link = links.inverse)
    available links are log, identity, inverse
models.family.Gaussian(link = links.identity)
    available links are log, identity, inverse
models.family.InverseGaussian(link = links.inverse_squared)
    available links are inverse_squared, inverse, identity, log
models.family.Poisson(link = links.logit)
    available links are log, identity, sqrt

'''
#TODO: quasi, quasibinomial, quasipoisson
#see http://www.biostat.jhsph.edu/~qli/biostatistics_r_doc/library/stats/html/family.html
# for comparison to R, and McCullagh and Nelder

import numpy as np
from models.family import links as L
from models.family import varfuncs as V

class Family(object):

    """
    A class to model one-parameter exponential
    families.

    INPUTS:
       link      -- a Link instance
       variance  -- a variance function (models means as a function
                    of mean)

    """

    valid = [-np.inf, np.inf]

    tol = 1.0e-05
    links = []
    fixedscale = None

    def _setlink(self, link):
        self._link = link
        if hasattr(self, "links"):
            if link not in self.links:
                raise ValueError, 'invalid link for family, should be in %s' % `self.links`

    def _getlink(self):
        return self._link

    link = property(_getlink, _setlink)

    def __init__(self, link, variance):
        self.link = link()
        self.variance = variance

    def weights(self, mu):

        """
        Weights for IRLS step.

        w = 1 / (link'(mu)**2 * variance(mu))

        INPUTS:
           mu  -- mean parameter in exponential family

        OUTPUTS:
           w   -- weights used in WLS step of GLM/GAM fit

        """
        return 1. / (self.link.deriv(mu)**2 * self.variance(mu))

    def deviance(self, Y, mu, scale=1.):
        """
        Deviance of (Y,mu) pair. Deviance is usually defined
        as the difference

        DEV = (SUM_i -2 log Likelihood(Y_i,mu_i) + 2 log Likelihood(mu_i,mu_i)) / scale

        INPUTS:
           Y     -- response variable
           mu    -- mean parameter
           scale -- optional scale in denominator of deviance

        OUTPUTS: dev
           dev   -- DEV, as described aboce

        """

        return np.power(self.devresid(Y, mu), 2).sum() / scale

    def devresid(self, Y, mu):
        """
        The deviance residuals, defined as the residuals
        in the deviance.

        Without knowing the link, they default to Pearson residuals

        resid_P = (Y - mu) * sqrt(weight(mu))

        INPUTS:
           Y     -- response variable
           mu    -- mean parameter

        OUTPUTS: resid
           resid -- deviance residuals
        """

        return (Y - mu) * np.sqrt(self.weights(mu))

    def fitted(self, eta):
        """
        Fitted values based on linear predictors eta.

        INPUTS:
           eta  -- values of linear predictors, say,
                   X beta in a generalized linear model.

        OUTPUTS: mu
           mu   -- link.inverse(eta), mean parameter based on eta

        """
        return self.link.inverse(eta)

    def predict(self, mu):
        """
        Linear predictors based on given mu values.

        INPUTS:
           mu   -- mean parameter of one-parameter exponential family

        OUTPUTS: eta
           eta  -- link(mu), linear predictors, based on
                   mean parameters mu

        """
        return self.link(mu)

class Poisson(Family):

    """
    Poisson exponential family.

    INPUTS:
       link      -- a Link instance

    """

    links = [L.log, L.identity, L.sqrt]
    variance = V.mu
    valid = [0, np.inf]
    fixedscale = np.array(1.)

    def __init__(self, link=L.log):
        self.variance = Poisson.variance
        self.link = link

    def devresid(self, Y, mu):
        """
        Poisson deviance residual

        INPUTS:
           Y     -- response variable
           mu    -- mean parameter

        OUTPUTS: resid
           resid -- deviance residuals

        """
        return np.sign(Y - mu) * np.sqrt(2 * Y * np.log(Y / mu) - 2 * (Y - mu))
# shouldn't it be
#        return np.sign(Y - mu) * np.sqrt(2 *np.sum(Y*np.log(Y/mu) - Y + mu))
# Gill p 58

    def deviance(self, Y, mu, scale=1.):
        '''
        Poisson deviance

        If a constant term is included it is

        2 * sum_i{y_i*log*y_i/mu_i)}
        '''
        return 2*np.sum(Y*np.log(Y/mu))

class Gaussian(Family):

    """
    Gaussian exponential family.

    INPUTS:
       link      -- a Link instance

    """

    links = [L.log, L.identity, L.inverse]
    variance = V.constant

    def __init__(self, link=L.identity):
        self.variance = Gaussian.variance
        self.link = link

    def devresid(self, Y, mu, scale=1.):
        """
        Gaussian deviance residual

        INPUTS:
           Y     -- response variable
           mu    -- mean parameter
           scale -- optional scale in denominator (after taking sqrt)

        OUTPUTS: resid
           resid -- deviance residuals
        """

        return (Y - mu) / np.sqrt(self.variance(mu) * scale)

class Gamma(Family):

    """
    Gamma exponential family.

    INPUTS:
       link      -- a Link instance

    BUGS:
       no deviance residuals?

    """

    links = [L.log, L.identity, L.inverse]
    variance = V.mu_squared

    def __init__(self, link=L.inverse):
        self.variance = Gamma.variance
        self.link = link


class Binomial(Family):

    """
    Binomial exponential family.

    INPUTS:
       link      -- a Link instance
       n         -- number of trials for Binomial
    """

    links = [L.logit, L.probit, L.cauchy, L.log, L.cloglog]
    variance = V.binary
    fixedscale = np.array(1.) # actually should be 1/k

    def __init__(self, link=L.logit, n=1):
        self.n = n
        self.variance = V.Binomial(n=self.n)
        self.link = link

    def devresid(self, Y, mu):
        """
        Binomial deviance residual

        INPUTS:
           Y     -- response variable
           mu    -- mean parameter

        OUTPUTS: resid
           resid -- deviance residuals

        """

        mu = self.link.clean(mu)
        return np.sign(Y - mu) * np.sqrt(-2 * (Y * np.log(mu / self.n) + (self.n - Y) * np.log(1 - mu / self.n)))

class InverseGaussian(Family):

    """
    InverseGaussian exponential family.

    INPUTS:
       link      -- a Link instance
       n         -- number of trials for Binomial

    """

    links = [L.inverse_squared, L.inverse, L.identity, L.log]
    variance = V.mu_cubed

    def __init__(self, link=L.inverse_squared):
        self.n = n
        self.variance = InverseGaussian.variance
        self.link = link
