'''
Defines the link functions to be used with GLM families.
'''

import numpy as np
import scipy.stats

#TODO: are the instance actually "aliases"
# I used this terminology in varfuncs as well -ss

class Link(object):

    """
    A generic link function for one-parameter exponential family.

    `Link` does nothing, but lays out the methods expected of any subclass.
    """

    def __call__(self, p):
        """
        Return the value of the link function.  This is just a placeholder.

        Parameters
        ----------
        p : array-like
            Probabilities

        Returns
        -------
        The value of the link function g(p) = z
        """
        return NotImplementedError

    def inverse(self, z):
        """
        Inverse of the link function.  Just a placeholder.

        Parameters
        ----------
        z : array-like
            `z` is usually the linear predictor of the transformed variable
            in the IRLS algorithm for GLM.

        Returns
        -------
        The value of the inverse of the link function g^(-1)(z) = p


        """
        return NotImplementedError

    def deriv(self, p):
        """
        Derivative of the link function g'(p).  Just a placeholder.

        Parameters
        ----------
        p : array-like

        Returns
        -------
        The value of the derivative of the link function g'(p)
        """
        return NotImplementedError

class Logit(Link):
    """
    The logit transform

    Notes
    -----
    call and derivative use a private method _clean to make trim p by
    1e-10 so that p is in (0,1)

    Alias of Logit:
    logit = Logit()
    """

    tol = 1.0e-10

    def _clean(self, p):
        """
        Clip logistic values to range (tol, 1-tol)

        Parameters
        -----------
        p : array-like
            Probabilities

        Returns
        --------
        pclip : array
            Clipped probabilities
        """
        return np.clip(p, Logit.tol, 1. - Logit.tol)

    def __call__(self, p):
        """
        The logit transform

        Parameters
        ----------
        p : array-like
            Probabilities

        Returns
        -------
        z : array
            Logit transform of `p`

        Notes
        -----
        g(p) = log(p / (1 - p))
        """
        p = self._clean(p)
        return np.log(p / (1. - p))

    def inverse(self, z):
        """
        Inverse of the logit transform

        Parameters
        ----------
        z : array-like
            The value of the logit transform at `p`

        Returns
        -------
        p : array
            Probabilities

        Notes
        -----
        g^(-1)(z) = exp(z)/(1+exp(z))
        """
        t = np.exp(z)
        return t / (1. + t)

    def deriv(self, p):

        """
        Derivative of the logit transform

        Parameters
        ----------
        p: array-like
            Probabilities

        Returns
        -------
        g'(p) : array
           Value of the derivative of logit transform at `p`

        Notes
        -----
        g'(p) = 1 / (p * (1 - p))

        Alias for `Logit`:
        logit = Logit()
        """
        p = self._clean(p)
        return 1. / (p * (1 - p))

#logit = Logit()
class logit(Logit):
    pass

class Power(Link):
    """
    The power transform

    Parameters
    ----------
    power : float
        The exponent of the power transform

    Notes
    -----
    Aliases of Power:
    inverse = Power(power=-1)
    sqrt = Power(power=.5)
    inverse_squared = Power(power=-2.)
    identity = Power(power=1.)
    """

    def __init__(self, power=1.):
        self.power = power

    def __call__(self, p):
        """
        Power transform link function

        Parameters
        ----------
        p : array-like
            Mean parameters

        Returns
        -------
        z : array-like
            Power transform of x

        Notes
        -----
        g(p) = x**self.power
        """

        return np.power(p, self.power)

    def inverse(self, z):
        """
        Inverse of the power transform link function


        Parameters
        ----------
        `z` : array-like
            Value of the transformed mean parameters at `p`

        Returns
        -------
        `p` : array
            Mean parameters

        Notes
        -----
        g^(-1)(z`) = `z`**(1/`power`)
        """
        return np.power(z, 1. / self.power)

    def deriv(self, p):
        """
        Derivative of the power transform

        Parameters
        ----------
        p : array-like
            Mean parameters

        Returns
        --------
        g'(p) : array
            Derivative of power transform of `p`

        Notes
        -----
        g'(`p`) = `power` * `p`**(`power` - 1)
        """
        return self.power * np.power(p, self.power - 1)

#inverse = Power(power=-1.)
class inverse_power(Power):
    """
    The inverse transform

    Notes
    -----
    g(p) = 1/p

    Alias of statsmodels.family.links.Power(power=-1.)
    """
    def __init__(self):
        super(inverse_power, self).__init__(power=-1.)

#sqrt = Power(power=0.5)
class sqrt(Power):
    """
    The square-root transform

    Notes
    -----
    g(`p`) = sqrt(`p`)

    Alias of statsmodels.family.links.Power(power=.5)
    """
    def __init__(self):
        super(sqrt, self).__init__(power=.5)

class inverse_squared(Power):
#inverse_squared = Power(power=-2.)
    """
    The inverse squared transform

    Notes
    -----
    g(`p`) = 1/(`p`\ \*\*2)

    Alias of statsmodels.family.links.Power(power=2.)
    """
    def __init__(self):
        super(inverse_squared, self).__init__(power=-2.)

class identity(Power):
    """
    The identity transform

    Notes
    -----
    g(`p`) = `p`

    Alias of statsmodels.family.links.Power(power=1.)
    """
    def __init__(self):
        super(identity, self).__init__(power=1.)

class Log(Link):
    """
    The log transform

    Notes
    -----
    call and derivative call a private method _clean to trim the data by
    1e-10 so that p is in (0,1). log is an alias of Log.
    """

    tol = 1.0e-10

    def _clean(self, x):
        return np.clip(x, Logit.tol, np.inf)

    def __call__(self, p, **extra):
        """
        Log transform link function

        Parameters
        ----------
        x : array-like
            Mean parameters

        Returns
        -------
        z : array
            log(x)

        Notes
        -----
        g(p) = log(p)
        """
        x = self._clean(p)
        return np.log(p)

    def inverse(self, z):
        """
        Inverse of log transform link function

        Parameters
        ----------
        z : array
            The inverse of the link function at `p`

        Returns
        -------
        p : array
            The mean probabilities given the value of the inverse `z`

        Notes
        -----
        g^{-1}(z) = exp(z)
        """
        return np.exp(z)

    def deriv(self, p):
        """
        Derivative of log transform link function

        Parameters
        ----------
        p : array-like
            Mean parameters

        Returns
        -------
        g'(p) : array
            derivative of log transform of x

        Notes
        -----
        g(x) = 1/x
        """
        p = self._clean(p)
        return 1. / p

class log(Log):
    """
    The log transform

    Notes
    -----
    log is a an alias of Log.
    """
    pass

#TODO: the CDFLink is untested
class CDFLink(Logit):
    """
    The use the CDF of a scipy.stats distribution

    CDFLink is a subclass of logit in order to use its _clean method
    for the link and its derivative.

    Parameters
    ----------
    dbn : scipy.stats distribution
        Default is dbn=scipy.stats.norm

    Notes
    -----
    The CDF link is untested.
    """

    def __init__(self, dbn=scipy.stats.norm):
        self.dbn = dbn

    def __call__(self, p):
        """
        CDF link function

        Parameters
        ----------
        p : array-like
            Mean parameters

        Returns
        -------
        z : array
           (ppf) inverse of CDF transform of p

        Notes
        -----
        g(`p`) = `dbn`.ppf(`p`)
        """
        p = self._clean(p)
        return self.dbn.ppf(p)

    def inverse(self, z):
        """
        The inverse of the CDF link

        Parameters
        ----------
        z : array-like
            The value of the inverse of the link function at `p`

        Returns
        -------
        p : array
            Mean probabilities.  The value of the inverse of CDF link of `z`

        Notes
        -----
        g^(-1)(`z`) = `dbn`.cdf(`z`)
        """
        return self.dbn.cdf(z)

    def deriv(self, p):
        """
        Derivative of CDF link

        Parameters
        ----------
        p : array-like
            mean parameters

        Returns
        -------
        g'(p) : array
         The derivative of CDF transform at `p`

        Notes
        -----
        g'(`p`) = 1./ `dbn`.pdf(`p`)
        """
# Or is it
#        g'(`p`) = 1/`dbn`.pdf(`dbn`.ppf(`p`))
#TODO: make sure this is correct.
#can we just have a numerical approximation?
        p = self._clean(p)
        return 1. / self.dbn.pdf(p)

#probit = CDFLink()
class probit(CDFLink):
    """
    The probit (standard normal CDF) transform

    Notes
    --------
    g(p) = scipy.stats.norm.ppf(p)

    probit is an alias of CDFLink.
    """
    pass

class cauchy(CDFLink):
    """
    The Cauchy (standard Cauchy CDF) transform

    Notes
    -----
    g(p) = scipy.stats.cauchy.ppf(p)

    cauchy is an alias of CDFLink with dbn=scipy.stats.cauchy
    """
    def __init__(self):
        super(cauchy, self).__init__(dbn=scipy.stats.cauchy)

#TODO: CLogLog is untested
class CLogLog(Logit):
    """
    The complementary log-log transform

    CLogLog inherits from Logit in order to have access to its _clean method
    for the link and its derivative.

    Notes
    -----
    CLogLog is untested.
    """
    def __call__(self, p):
        """
        C-Log-Log transform link function

        Parameters
        ----------
        p : array
            Mean parameters

        Returns
        -------
        z : array
            The CLogLog transform of `p`

        Notes
        -----
        g(p) = log(-log(1-p))
        """
        p = self._clean(p)
        return np.log(-np.log(1-p))

    def inverse(self, z):
        """
        Inverse of C-Log-Log transform link function


        Parameters
        ----------
        z : array-like
            The value of the inverse of the CLogLog link function at `p`

        Returns
        -------
        p : array
           Mean parameters

        Notes
        -----
        g^(-1)(`z`) = 1-exp(-exp(`z`))
        """
        return 1-np.exp(-np.exp(z))

    def deriv(self, p):
        """
        Derivatve of C-Log-Log transform link function

        Parameters
        ----------
        p : array-like
            Mean parameters

        Returns
        -------
        g'(p) : array
           The derivative of the CLogLog transform link function

        Notes
        -----
        g'(p) = - 1 / (log(p) * p)
        """
        p = self._clean(p)
        return 1. / ((p-1)*(np.log(1-p)))

class cloglog(CLogLog):
    """
    The CLogLog transform link function.

    Notes
    -----
    g(`p`) = log(-log(1-`p`))

    cloglog is an alias for CLogLog
    cloglog = CLogLog()
    """
    pass

class NegativeBinomial(object):
    '''
    The negative binomial link function

    Parameters
    ----------
    alpha : float, optional
        Alpha is the ancillary parameter of the Negative Binomial link function.
        It is assumed to be nonstochastic.  The default value is 1. Permissible
        values are usually assumed to be in (.01,2).
    '''

    tol = 1.0e-10

    def __init__(self, alpha=1.):
        self.alpha = alpha

    def _clean(self, x):
        return np.clip(x, NegativeBinomial.tol, np.inf)

    def __call__(self, x):
        '''
        Negative Binomial transform link function

        Parameters
        ----------
        p : array-like
            Mean parameters

        Returns
        -------
        z : array
            The negative binomial transform of `p`

        Notes
        -----
        g(p) = log(p/(p + 1/alpha))
        '''
        p = self._clean(p)
        return np.log(p/(p+1/self.alpha))

    def inverse(self, z):
        '''
        Inverse of the negative binomial transform

        Parameters
        -----------
        z : array-like
            The value of the inverse of the negative binomial link at `p`.
        Returns
        -------
        p : array
            Mean parameters

        Notes
        -----
        g^(-1)(z) = exp(z)/(alpha*(1-exp(z)))
        '''
        return np.exp(z)/(self.alpha*(1-np.exp(z)))

    def deriv(self,p):
        '''
        Derivative of the negative binomial transform

        Parameters
        ----------
        p : array-like
            Mean parameters

        Returns
        -------
        g'(p) : array
            The derivative of the negative binomial transform link function

        Notes
        -----
        g'(x) = 1/(x+alpha*x^2)
        '''
        return 1/(p+self.alpha*p**2)

class nbinom(NegativeBinomial):
    """
    The negative binomial link function.

    Notes
    -----
    g(p) = log(p/(p + 1/alpha))

    nbinom is an alias of NegativeBinomial.
    nbinom = NegativeBinomial(alpha=1.)
    """
    pass
