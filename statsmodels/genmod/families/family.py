'''
The one parameter exponential family distributions used by GLM.
'''
# TODO: quasi, quasibinomial, quasipoisson
# see http://www.biostat.jhsph.edu/~qli/biostatistics_r_doc/library/stats/html/family.html
# for comparison to R, and McCullagh and Nelder

import numpy as np
from scipy import special
from . import links as L
from . import varfuncs as V
FLOAT_EPS = np.finfo(float).eps


class Family(object):
    """
    The parent class for one-parameter exponential families.

    Parameters
    ----------
    link : a link function instance
        Link is the linear transformation function.
        See the individual families for available links.
    variance : a variance function
        Measures the variance as a function of the mean probabilities.
        See the individual families for the default variance function.

    See Also
    --------
    :ref:`links`

    """
    # TODO: change these class attributes, use valid somewhere...
    valid = [-np.inf, np.inf]

    links = []

    def _setlink(self, link):
        """
        Helper method to set the link for a family.

        Raises a ValueError exception if the link is not available.  Note that
        the error message might not be that informative because it tells you
        that the link should be in the base class for the link function.

        See glm.GLM for a list of appropriate links for each family but note
        that not all of these are currently available.
        """
        # TODO: change the links class attribute in the families to hold
        # meaningful information instead of a list of links instances such as
        # [<statsmodels.family.links.Log object at 0x9a4240c>,
        #  <statsmodels.family.links.Power object at 0x9a423ec>,
        #  <statsmodels.family.links.Power object at 0x9a4236c>]
        # for Poisson...
        self._link = link
        if not isinstance(link, L.Link):
            raise TypeError("The input should be a valid Link object.")
        if hasattr(self, "links"):
            validlink = link in self.links
            validlink = max([isinstance(link, _) for _ in self.links])
            if not validlink:
                errmsg = "Invalid link for family, should be in %s. (got %s)"
                raise ValueError(errmsg % (repr(self.links), link))

    def _getlink(self):
        """
        Helper method to get the link for a family.
        """
        return self._link

    # link property for each family is a pointer to link instance
    link = property(_getlink, _setlink, doc="Link function for family")

    def __init__(self, link, variance):
        self.link = link()
        self.variance = variance

    def starting_mu(self, y):
        r"""
        Starting value for mu in the IRLS algorithm.

        Parameters
        ----------
        y : array
            The untransformed response variable.

        Returns
        -------
        mu_0 : array
            The first guess on the transformed response variable.

        Notes
        -----
        .. math::

           \mu_0 = (Y + \overline{Y})/2

        Notes
        -----
        Only the Binomial family takes a different initial value.
        """
        return (y + y.mean())/2.

    def weights(self, mu):
        r"""
        Weights for IRLS steps

        Parameters
        ----------
        mu : array-like
            The transformed mean response variable in the exponential family

        Returns
        -------
        w : array
            The weights for the IRLS steps

        Notes
        -----
        .. math::

           w = 1 / (g'(\mu)^2  * Var(\mu))
        """
        return 1. / (self.link.deriv(mu)**2 * self.variance(mu))

    def deviance(self, endog, mu, freq_weights=1., scale=1.):
        r"""
        The deviance function evaluated at (endog,mu,freq_weights,mu).

        Deviance is usually defined as twice the loglikelihood ratio.

        Parameters
        ----------
        endog : array-like
            The endogenous response variable
        mu : array-like
            The inverse of the link function at the linear predicted values.
        freq_weights : array-like
            1d array of frequency weights. The default is 1.
        scale : float, optional
            An optional scale argument. The default is 1.

        Returns
        -------
        Deviance : array
            The value of deviance function defined below.

        Notes
        -----
        Deviance is defined

        .. math::

           D = \sum_i (2 * freq\_weights_i * llf(Y_i, Y_i) - 2 *
               llf(Y_i, \mu_i)) / scale

        where y is the endogenous variable. The deviance functions are
        analytically defined for each family.
        """
        raise NotImplementedError

    def resid_dev(self, endog, mu, freq_weights=1., scale=1.):
        """
        The deviance residuals

        Parameters
        ----------
        endog : array
            The endogenous response variable
        mu : array
            The inverse of the link function at the linear predicted values.
        freq_weights : array-like
            1d array of frequency weights. The default is 1.
        scale : float, optional
            An optional argument to divide the residuals by scale. The default
            is 1.

        Returns
        -------
        Deviance residuals.

        Notes
        -----
        The deviance residuals are defined for each family.
        """
        raise NotImplementedError

    def fitted(self, lin_pred):
        """
        Fitted values based on linear predictors lin_pred.

        Parameters
        -----------
        lin_pred : array
            Values of the linear predictor of the model.
            dot(X,beta) in a classical linear model.

        Returns
        --------
        mu : array
            The mean response variables given by the inverse of the link
            function.
        """
        fits = self.link.inverse(lin_pred)
        return fits

    def predict(self, mu):
        """
        Linear predictors based on given mu values.

        Parameters
        ----------
        mu : array
            The mean response variables

        Returns
        -------
        lin_pred : array
            Linear predictors based on the mean response variables.  The value
            of the link function at the given mu.
        """
        return self.link(mu)

    def loglike(self, endog, mu, freq_weights=1., scale=1.):
        """
        The log-likelihood function in terms of the fitted mean response.

        Parameters
        ----------
        endog : array
            Usually the endogenous response variable.
        mu : array
            Usually but not always the fitted mean response variable.
        freq_weights : array-like
            1d array of frequency weights. The default is 1.
        scale : float
            The scale parameter. The default is 1.

        Returns
        -------
        llf : float
            The value of the loglikelihood evaluated at
            (endog,mu,freq_weights,scale) as defined below.

        Notes
        -----
        This is defined for each family.  endog and mu are not restricted to
        `endog` and `mu` respectively.  For instance, the deviance function
        calls both loglike(endog,endog) and loglike(endog,mu) to get the
        likelihood ratio.
        """
        raise NotImplementedError

    def resid_anscombe(self, endog, mu):
        """
        The Anscombe residuals

        See Also
        --------
        statsmodels.genmod.families.family.Family : `resid_anscombe` for the
          individual families for more information
        """
        raise NotImplementedError


class Poisson(Family):
    """
    Poisson exponential family.

    Parameters
    ----------
    link : a link instance, optional
        The default link for the Poisson family is the log link. Available
        links are log, identity, and sqrt. See statsmodels.family.links for
        more information.

    Attributes
    ----------
    Poisson.link : a link instance
        The link function of the Poisson instance.
    Poisson.variance : varfuncs instance
        `variance` is an instance of
        statsmodels.genmod.families.family.varfuncs.mu

    See also
    --------
    statsmodels.genmod.families.family.Family
    :ref:`links`

    """

    links = [L.log, L.identity, L.sqrt]
    variance = V.mu
    valid = [0, np.inf]
    safe_links = [L.Log, ]

    def __init__(self, link=L.log):
        self.variance = Poisson.variance
        self.link = link()

    def _clean(self, x):
        """
        Helper function to trim the data so that is in (0,inf)

        Notes
        -----
        The need for this function was discovered through usage and its
        possible that other families might need a check for validity of the
        domain.
        """
        return np.clip(x, FLOAT_EPS, np.inf)

    def resid_dev(self, endog, mu, scale=1.):
        r"""Poisson deviance residual

        Parameters
        ----------
        endog : array-like
            Endogenous response variable
        mu : array-like
            Fitted mean response variable
        scale : float, optional
            An optional argument to divide the residuals by scale. The default
            is 1.

        Returns
        -------
        resid_dev : array
            Deviance residuals as defined below

        Notes
        -----
        .. math::

           resid\_dev_i = sign(Y_i - \mu_i) * \sqrt{2 *
                          (Y_i * \log(Y_i / \mu_i) - (Y_i - \mu_i))} / scale
        """
        endog_mu = self._clean(endog / mu)
        return (np.sign(endog - mu) *
                np.sqrt(2 * (endog * np.log(endog_mu) - (endog - mu))) / scale)

    def deviance(self, endog, mu, freq_weights=1., scale=1.):
        r'''
        Poisson deviance function

        Parameters
        ----------
        endog : array-like
            Endogenous response variable
        mu : array-like
            Fitted mean response variable
        freq_weights : array-like
            1d array of frequency weights. The default is 1.
        scale : float, optional
            An optional scale argument. The default is 1.

        Returns
        -------
        deviance : float
            The deviance function at (endog,mu,freq_weights,scale) as defined
            below.

        Notes
        -----
        If a constant term is included it is defined as

        .. math::

           D = 2 * \sum_i (freq\_weights_i * Y_i * \log(Y_i / \mu_i))/ scale
        '''
        endog_mu = self._clean(endog / mu)
        return 2 * np.sum(freq_weights * (endog * np.log(endog_mu) -
                                          (endog - mu))) / scale


    def loglike(self, endog, mu, freq_weights=1., scale=1.):
        r"""
        The log-likelihood function in terms of the fitted mean response.

        Parameters
        ----------
        endog : array-like
            Endogenous response variable
        mu : array-like
            Fitted mean response variable
        freq_weights : array-like
            1d array of frequency weights. The default is 1.
        scale : float, optional
            The scale parameter, defaults to 1.

        Returns
        -------
        llf : float
            The value of the loglikelihood function evaluated at
            (endog,mu,freq_weights,scale) as defined below.

        Notes
        -----
        .. math::

           llf = scale * \sum_i freq\_weights_i * (Y_i * \log(\mu_i) - \mu_i -
                 \ln \Gamma(Y_i + 1))
        """
        loglike = np.sum(freq_weights * (endog * np.log(mu) - mu -
                         special.gammaln(endog + 1)))
        return scale * loglike

    def resid_anscombe(self, endog, mu):
        r"""
        Anscombe residuals for the Poisson exponential family distribution

        Parameters
        ----------
        endog : array-like
            Endogenous response variable
        mu : array-like
            Fitted mean response variable

        Returns
        -------
        resid_anscombe : array
            The Anscome residuals for the Poisson family defined below

        Notes
        -----
        .. math::

           resid\_anscombe_i = (3/2) * (Y_i^{2/3} - \mu_i^{2/3}) / \mu_i^{1/6}
        """
        return (3 / 2.) * (endog**(2/3.) - mu**(2 / 3.)) / mu**(1 / 6.)


class Gaussian(Family):
    """
    Gaussian exponential family distribution.

    Parameters
    ----------
    link : a link instance, optional
        The default link for the Gaussian family is the identity link.
        Available links are log, identity, and inverse.
        See statsmodels.family.links for more information.

    Attributes
    ----------
    Gaussian.link : a link instance
        The link function of the Gaussian instance
    Gaussian.variance : varfunc instance
        `variance` is an instance of statsmodels.family.varfuncs.constant

    See also
    --------
    statsmodels.genmod.families.family.Family
    :ref:`links`

    """

    links = [L.log, L.identity, L.inverse_power]
    variance = V.constant
    safe_links = links

    def __init__(self, link=L.identity):
        self.variance = Gaussian.variance
        self.link = link()

    def resid_dev(self, endog, mu, scale=1.):
        r"""
        Gaussian deviance residuals

        Parameters
        -----------
        endog : array-like
            Endogenous response variable
        mu : array-like
            Fitted mean response variable
        scale : float, optional
            An optional argument to divide the residuals by scale. The default
            is 1.

        Returns
        -------
        resid_dev : array
            Deviance residuals as defined below

        Notes
        --------
        .. math::

           resid\_dev_i = (Y_i - \mu_i) / \sqrt{Var(\mu_i)} / scale
        """

        return (endog - mu) / np.sqrt(self.variance(mu)) / scale

    def deviance(self, endog, mu, freq_weights=1., scale=1.):
        r"""
        Gaussian deviance function

        Parameters
        ----------
        endog : array-like
            Endogenous response variable
        mu : array-like
            Fitted mean response variable
        freq_weights : array-like
            1d array of frequency weights. The default is 1.
        scale : float, optional
            An optional scale argument. The default is 1.

        Returns
        -------
        deviance : float
            The deviance function at (endog,mu,freq_weights,scale)
            as defined below.

        Notes
        --------
        .. math::

           D = \sum_i freq\_weights_i * (Y_i - \mu_i)^2 / scale
        """
        return np.sum((freq_weights * (endog - mu)**2)) / scale

    def loglike(self, endog, mu, freq_weights=1., scale=1.):
        r"""
        The log-likelihood in terms of the fitted mean response.

        Parameters
        ----------
        endog : array-like
            Endogenous response variable
        mu : array-like
            Fitted mean response variable
        freq_weights : array-like
            1d array of frequency weights. The default is 1.
        scale : float, optional
            Scales the loglikelihood function. The default is 1.

        Returns
        -------
        llf : float
            The value of the loglikelihood function evaluated at
            (endog,mu,freq_weights,scale) as defined below.

        Notes
        -----
        If the link is the identity link function then the
        loglikelihood function is the same as the classical OLS model.

        .. math::

           llf = -nobs / 2 * (\log(SSR) + (1 + \log(2 \pi / nobs)))

        where

        .. math::
           SSR = \sum_i (Y_i - g^{-1}(\mu_i))^2

        If the links is not the identity link then the loglikelihood
        function is defined as

        .. math::

           llf = \sum_i freq\_weights_i * ((Y_i * \mu_i - \mu_i^2 / 2) / scale-
                 Y^2 / (2 * scale) - (1/2) * \log(2 * \pi * scale))
        """
        if isinstance(self.link, L.Power) and self.link.power == 1:
            # This is just the loglikelihood for classical OLS
            nobs2 = np.sum(freq_weights, axis=0) / 2.
            SSR = np.sum((endog-self.fitted(mu))**2, axis=0)
            llf = -np.log(SSR) * nobs2
            llf -= (1+np.log(np.pi/nobs2))*nobs2
            return llf
        else:
            return np.sum(freq_weights * ((endog * mu - mu**2/2)/scale -
                          endog**2/(2 * scale) - .5*np.log(2 * np.pi * scale)))

    def resid_anscombe(self, endog, mu):
        r"""
        The Anscombe residuals for the Gaussian exponential family distribution

        Parameters
        ----------
        endog : array
            Endogenous response variable
        mu : array
            Fitted mean response variable

        Returns
        -------
        resid_anscombe : array
            The Anscombe residuals for the Gaussian family defined below

        Notes
        --------
        .. math::

           resid\_anscombe_i = Y_i - \mu_i
        """
        return endog - mu


class Gamma(Family):
    """
    Gamma exponential family distribution.

    Parameters
    ----------
    link : a link instance, optional
        The default link for the Gamma family is the inverse link.
        Available links are log, identity, and inverse.
        See statsmodels.family.links for more information.

    Attributes
    ----------
    Gamma.link : a link instance
        The link function of the Gamma instance
    Gamma.variance : varfunc instance
        `variance` is an instance of statsmodels.family.varfuncs.mu_squared

    See also
    --------
    statsmodels.genmod.families.family.Family
    :ref:`links`

    """

    links = [L.log, L.identity, L.inverse_power]
    variance = V.mu_squared
    safe_links = [L.Log, ]

    def __init__(self, link=L.inverse_power):
        self.variance = Gamma.variance
        self.link = link()

    def _clean(self, x):
        """
        Helper function to trim the data so that is in (0,inf)

        Notes
        -----
        The need for this function was discovered through usage and its
        possible that other families might need a check for validity of the
        domain.
        """
        return np.clip(x, FLOAT_EPS, np.inf)

    def deviance(self, endog, mu, freq_weights=1., scale=1.):
        r"""
        Gamma deviance function

        Parameters
        -----------
        endog : array-like
            Endogenous response variable
        mu : array-like
            Fitted mean response variable
        freq_weights : array-like
            1d array of frequency weights. The default is 1.
        scale : float, optional
            An optional scale argument. The default is 1.

        Returns
        -------
        deviance : float
            Deviance function as defined below

        Notes
        -----
        .. math::

           D = 2 * \sum_i freq\_weights_i * ((Y_i - \mu_i)/\mu_i - \log(Y_i /
               \mu_i))
        """
        endog_mu = self._clean(endog/mu)
        return 2*np.sum(freq_weights*((endog-mu)/mu-np.log(endog_mu)))

    def resid_dev(self, endog, mu, scale=1.):
        r"""
        Gamma deviance residuals

        Parameters
        -----------
        endog : array-like
            Endogenous response variable
        mu : array-like
            Fitted mean response variable
        scale : float, optional
            An optional argument to divide the residuals by scale. The default
            is 1.

        Returns
        -------
        resid_dev : array
            Deviance residuals as defined below

        Notes
        -----
        .. math::

           resid\_dev_i = sign(Y_i - \mu_i) \sqrt{-2 *
                          (-(Y_i - \mu_i) / \mu_i + \log(Y_i / \mu_i))}
        """
        endog_mu = self._clean(endog / mu)
        return np.sign(endog - mu) * np.sqrt(-2 * (-(endog - mu)/mu +
                                                   np.log(endog_mu)))

    def loglike(self, endog, mu, freq_weights=1., scale=1.):
        r"""
        The log-likelihood function in terms of the fitted mean response.

        Parameters
        ----------
        endog : array-like
            Endogenous response variable
        mu : array-like
            Fitted mean response variable
        freq_weights : array-like
            1d array of frequency weights. The default is 1.
        scale : float, optional
            The default is 1.

        Returns
        -------
        llf : float
            The value of the loglikelihood function evaluated at
            (endog,mu,freq_weights,scale) as defined below.

        Notes
        --------
        .. math::

           llf = -1 / scale * \sum_i *(Y_i / \mu_i+ \log(\mu_i)+
                 (scale -1) * \log(Y) + \log(scale) + scale *
                 \ln \Gamma(1 / scale))
        """
        endog_mu = self._clean(endog / mu)
        return - np.sum((endog_mu - np.log(endog_mu) + scale *
                         np.log(endog) + np.log(scale) + scale *
                         special.gammaln(1./scale)) * freq_weights) / scale

        # in Stata scale is set to equal 1 for reporting llf
        # in R it's the dispersion, though there is a loss of precision vs.
        # our results due to an assumed difference in implementation

    def resid_anscombe(self, endog, mu):
        r"""
        The Anscombe residuals for Gamma exponential family distribution

        Parameters
        ----------
        endog : array
            Endogenous response variable
        mu : array
            Fitted mean response variable

        Returns
        -------
        resid_anscombe : array
            The Anscombe residuals for the Gamma family defined below

        Notes
        -----
        .. math::

           resid\_anscombe_i = 3 * (Y_i^{1/3} - \mu_i^{1/3}) / \mu_i^{1/3}
        """
        return 3 * (endog**(1/3.) - mu**(1/3.)) / mu**(1/3.)


class Binomial(Family):
    """
    Binomial exponential family distribution.

    Parameters
    ----------
    link : a link instance, optional
        The default link for the Binomial family is the logit link.
        Available links are logit, probit, cauchy, log, and cloglog.
        See statsmodels.family.links for more information.

    Attributes
    ----------
    Binomial.link : a link instance
        The link function of the Binomial instance
    Binomial.variance : varfunc instance
        `variance` is an instance of statsmodels.family.varfuncs.binary

    See also
    --------
    statsmodels.genmod.families.family.Family
    :ref:`links`

    Notes
    -----
    endog for Binomial can be specified in one of three ways.

    """

    links = [L.logit, L.probit, L.cauchy, L.log, L.cloglog, L.identity]
    variance = V.binary  # this is not used below in an effort to include n

    # Other safe links, e.g. cloglog and probit are subclasses
    safe_links = [L.Logit, L.CDFLink]

    def __init__(self, link=L.logit):  # , n=1.):
        # TODO: it *should* work for a constant n>1 actually, if freq_weights
        # is equal to n
        self.n = 1
        # overwritten by initialize if needed but always used to initialize
        # variance since endog is assumed/forced to be (0,1)
        self.variance = V.Binomial(n=self.n)
        self.link = link()

    def starting_mu(self, y):
        """
        The starting values for the IRLS algorithm for the Binomial family.
        A good choice for the binomial family is :math:`\mu_0 = (Y_i + 0.5)/2`
        """
        return (y + .5)/2

    def initialize(self, endog, freq_weights):
        '''
        Initialize the response variable.

        Parameters
        ----------
        endog : array
            Endogenous response variable

        Returns
        --------
        If `endog` is binary, returns `endog`

        If `endog` is a 2d array, then the input is assumed to be in the format
        (successes, failures) and
        successes/(success + failures) is returned.  And n is set to
        successes + failures.
        '''
        # if not np.all(np.asarray(freq_weights) == 1):
        #     self.variance = V.Binomial(n=freq_weights)
        if (endog.ndim > 1 and endog.shape[1] > 1):
            y = endog[:, 0]
            # overwrite self.freq_weights for deviance below
            self.n = endog.sum(1)
            return y*1./self.n, self.n
        else:
            return endog, np.ones(endog.shape[0])

    def deviance(self, endog, mu, freq_weights=1, scale=1.):
        r'''
        Deviance function for either Bernoulli or Binomial data.

        Parameters
        ----------
        endog : array-like
            Endogenous response variable (already transformed to a probability
            if appropriate).
        mu : array
            Fitted mean response variable
        freq_weights : array-like
            1d array of frequency weights. The default is 1.
        scale : float, optional
            An optional scale argument. The default is 1.

        Returns
        --------
        deviance : float
            The deviance function as defined below

        Notes
        -----
        If the endogenous variable is binary:

        .. math::

           D = -2 * \sum_i freq\_weights * (I_{1,i} * \log(\mu_i) + I_{0,i} *
               \log(1 - \mu_i))

        where :math:`I_{1,i}` is an indicator function that evalueates to 1 if
        :math:`Y_i = 1`. and :math:`I_{0,i}` is an indicator function that
        evaluates to 1 if :math:`Y_i = 0`.

        If the model is ninomial:

        .. math::

           D = 2 * \sum_i freq\_weights * (\log(Y_i / \mu_i) + (n_i - Y_i) *
               \log((n_i - Y_i) / n_i - \mu_i))

        where :math:`Y_i` and :math:`n` are as defined in Binomial.initialize.
        '''
        if np.shape(self.n) == () and self.n == 1:
            one = np.equal(endog, 1)
            return -2 * np.sum((one * np.log(mu + 1e-200) + (1-one) *
                               np.log(1 - mu + 1e-200)) * freq_weights)

        else:
            return 2 * np.sum(self.n * freq_weights *
                              (endog * np.log(endog/mu + 1e-200) +
                               (1 - endog) * np.log((1 - endog) /
                               (1 - mu) + 1e-200)))

    def resid_dev(self, endog, mu, scale=1.):
        r"""
        Binomial deviance residuals

        Parameters
        -----------
        endog : array-like
            Endogenous response variable
        mu : array-like
            Fitted mean response variable
        scale : float, optional
            An optional argument to divide the residuals by scale. The default
            is 1.

        Returns
        -------
        resid_dev : array
            Deviance residuals as defined below

        Notes
        -----
        If the endogenous variable is binary:

        .. math::

           resid\_dev_i = sign(Y_i - \mu_i) * \sqrt{-2 *
                          \log(I_{1,i} * \mu_i + I_{0,i} * (1 - \mu_i))}

        where :math:`I_{1,i}` is an indicator function that evalueates to 1 if
        :math:`Y_i = 1`. and :math:`I_{0,i}` is an indicator function that
        evaluates to 1 if :math:`Y_i = 0`.

        If the endogenous variable is binomial:

        .. math::

           resid\_dev_i = sign(Y_i - \mu_i) \sqrt{2 * n_i *
                          (Y_i * \log(Y_i / \mu_i) + (1 - Y_i) *
                          \log(1 - Y_i)/(1 - \mu_i))}

        where :math:`Y_i` and :math:`n` are as defined in Binomial.initialize.
        """

        mu = self.link._clean(mu)
        if np.shape(self.n) == () and self.n == 1:
            one = np.equal(endog, 1)
            return np.sign(endog-mu)*np.sqrt(-2 *
                                             np.log(one * mu + (1 - one) *
                                                    (1 - mu)))/scale
        else:
            return (np.sign(endog - mu) *
                    np.sqrt(2 * self.n *
                            (endog * np.log(endog/mu + 1e-200) +
                             (1 - endog) * np.log((1 - endog)/(1 - mu) +
                                                  1e-200)))/scale)

    def loglike(self, endog, mu, freq_weights=1, scale=1.):
        r"""
        The log-likelihood function in terms of the fitted mean response.

        Parameters
        ----------
        endog : array-like
            Endogenous response variable
        mu : array-like
            Fitted mean response variable
        freq_weights : array-like
            1d array of frequency weights. The default is 1.
        scale : float, optional
            Not used for the Binomial GLM.

        Returns
        -------
        llf : float
            The value of the loglikelihood function evaluated at
            (endog,mu,freq_weights,scale) as defined below.

        Notes
        --------
        If the endogenous variable is binary:

        .. math::

         llf = scale * \sum_i (y_i * \log(\mu_i/(1-\mu_i)) + \log(1-\mu_i)) *
               freq\_weights_i

        If the endogenous variable is binomial:

        .. math::

           llf = scale * \sum_i freq\_weights_i * (\ln \Gamma(n+1) -
                 \ln \Gamma(y_i + 1) - \ln \Gamma(n_i - y_i +1) + y_i *
                 \log(\mu_i / (1 - \mu_i)) + n * \log(1 - \mu_i))

        where :math:`y_i = Y_i * n_i` with :math:`Y_i` and :math:`n_i` as
        defined in Binomial initialize.  This simply makes :math:`y_i` the
        original number of successes.
        """

        if np.shape(self.n) == () and self.n == 1:
            return scale * np.sum((endog * np.log(mu/(1 - mu) + 1e-200) +
                                   np.log(1 - mu)) * freq_weights)
        else:
            y = endog * self.n  # convert back to successes
            return scale * np.sum((special.gammaln(self.n + 1) -
                                   special.gammaln(y + 1) -
                                   special.gammaln(self.n - y + 1) + y *
                                   np.log(mu/(1 - mu)) + self.n *
                                   np.log(1 - mu)) * freq_weights)

    def resid_anscombe(self, endog, mu):
        '''
        The Anscombe residuals

        Parameters
        ----------
        endog : array-like
            Endogenous response variable
        mu : array-like
            Fitted mean response variable

        Returns
        -------
        resid_anscombe : array
            The Anscombe residuals as defined below.

        Notes
        -----
        sqrt(n)*(cox_snell(endog)-cox_snell(mu))/(mu**(1/6.)*(1-mu)**(1/6.))

        where cox_snell is defined as
        cox_snell(x) = betainc(2/3., 2/3., x)*betainc(2/3.,2/3.)
        where betainc is the incomplete beta function

        The name 'cox_snell' is idiosyncratic and is simply used for
        convenience following the approach suggested in Cox and Snell (1968).
        Further note that
        cox_snell(x) = x**(2/3.)/(2/3.)*hyp2f1(2/3.,1/3.,5/3.,x)
        where hyp2f1 is the hypergeometric 2f1 function.  The Anscombe
        residuals are sometimes defined in the literature using the
        hyp2f1 formulation.  Both betainc and hyp2f1 can be found in scipy.

        References
        ----------
        Anscombe, FJ. (1953) "Contribution to the discussion of H. Hotelling's
            paper." Journal of the Royal Statistical Society B. 15, 229-30.

        Cox, DR and Snell, EJ. (1968) "A General Definition of Residuals."
            Journal of the Royal Statistical Society B. 30, 248-75.

        '''
        cox_snell = lambda x: (special.betainc(2/3., 2/3., x)
                               * special.beta(2/3., 2/3.))
        return np.sqrt(self.n) * ((cox_snell(endog) - cox_snell(mu)) /
                                  (mu**(1/6.) * (1 - mu)**(1/6.)))


class InverseGaussian(Family):
    """
    InverseGaussian exponential family.

    Parameters
    ----------
    link : a link instance, optional
        The default link for the inverse Gaussian family is the
        inverse squared link.
        Available links are inverse_squared, inverse, log, and identity.
        See statsmodels.family.links for more information.

    Attributes
    ----------
    InverseGaussian.link : a link instance
        The link function of the inverse Gaussian instance
    InverseGaussian.variance : varfunc instance
        `variance` is an instance of statsmodels.family.varfuncs.mu_cubed

    See also
    --------
    statsmodels.genmod.families.family.Family
    :ref:`links`

    Notes
    -----
    The inverse Guassian distribution is sometimes referred to in the
    literature as the Wald distribution.

    """

    links = [L.inverse_squared, L.inverse_power, L.identity, L.log]
    variance = V.mu_cubed
    safe_links = [L.inverse_squared, L.Log, ]

    def __init__(self, link=L.inverse_squared):
        self.variance = InverseGaussian.variance
        self.link = link()

    def resid_dev(self, endog, mu, scale=1.):
        r"""
        Returns the deviance residuals for the inverse Gaussian family.

        Parameters
        -----------
        endog : array-like
            Endogenous response variable
        mu : array-like
            Fitted mean response variable
        freq_weights : array-like
            1d array of frequency weights. The default is 1.
        scale : float, optional
            An optional argument to divide the residuals by scale. The default
            is 1.

        Returns
        -------
        resid_dev : array
            Deviance residuals as defined below

        Notes
        -----
        .. math::

           resid\_dev_i = sign(Y_i - \mu_i) *
                          \sqrt {(Y_i - \mu_i)^2 / (Y_i * \mu_i^2)} / scale
        """
        return np.sign(endog-mu) * np.sqrt((endog-mu)**2/(endog*mu**2))/scale

    def deviance(self, endog, mu, freq_weights=1., scale=1.):
        r"""
        Inverse Gaussian deviance function

        Parameters
        -----------
        endog : array-like
            Endogenous response variable
        mu : array-like
            Fitted mean response variable
        freq_weights : array-like
            1d array of frequency weights. The default is 1.
        scale : float, optional
            An optional scale argument. The default is 1.

        Returns
        -------
        deviance : float
            Deviance function as defined below

        Notes
        -----
        .. math::

           D = \sum_i freq\_weights_i * ((Y_i - \mu_i)^2 / (Y_i *\mu_i^2)) /
               scale
        """
        return np.sum(freq_weights*(endog-mu)**2/(endog*mu**2))/scale

    def loglike(self, endog, mu, freq_weights=1., scale=1.):
        r"""
        The log-likelihood function in terms of the fitted mean response.

        Parameters
        ----------
        endog : array-like
            Endogenous response variable
        mu : array-like
            Fitted mean response variable
        freq_weights : array-like
            1d array of frequency weights. The default is 1.
        scale : float, optional
            The default is 1.

        Returns
        -------
        llf : float
            The value of the loglikelihood function evaluated at
            (endog,mu,freq_weights,scale) as defined below.

        Notes
        -----
        .. math::

           llf = -1/2 * \sum_i freq\_weights_i * ((Y_i - \mu_i)^2 / (Y_i *
                 \mu_i * scale) + \log(scale * Y_i^3) + \log(2 * \pi))
        """
        return -.5 * np.sum(((endog - mu)**2/(endog * mu**2 * scale) +
                             np.log(scale * endog**3) + np.log(2 * np.pi)) *
                            freq_weights)

    def resid_anscombe(self, endog, mu):
        r"""
        The Anscombe residuals for the inverse Gaussian distribution

        Parameters
        ----------
        endog : array
            Endogenous response variable
        mu : array
            Fitted mean response variable

        Returns
        -------
        resid_anscombe : array
            The Anscombe residuals for the inverse Gaussian distribution  as
            defined below

        Notes
        -----
        .. math::

           resid\_anscombe_i = \log(Y_i / \mu_i) / \sqrt{\mu_i}
        """
        return np.log(endog / mu) / np.sqrt(mu)


class NegativeBinomial(Family):
    r"""
    Negative Binomial exponential family.

    Parameters
    ----------
    link : a link instance, optional
        The default link for the negative binomial family is the log link.
        Available links are log, cloglog, identity, nbinom and power.
        See statsmodels.family.links for more information.
    alpha : float, optional
        The ancillary parameter for the negative binomial distribution.
        For now `alpha` is assumed to be nonstochastic.  The default value
        is 1.  Permissible values are usually assumed to be between .01 and 2.


    Attributes
    ----------
    NegativeBinomial.link : a link instance
        The link function of the negative binomial instance
    NegativeBinomial.variance : varfunc instance
        `variance` is an instance of statsmodels.family.varfuncs.nbinom

    See also
    --------
    statsmodels.genmod.families.family.Family
    :ref:`links`

    Notes
    -----
    Power link functions are not yet supported.

    Parameterization for :math:`y=0,1,2,\ldots` is

     :math:`f(y) = \frac{\Gamma(y+\frac{1}{\alpha})}{y!\Gamma(\frac{1}{\alpha})}
     \left(\frac{1}{1+\alpha\mu}\right)^{\frac{1}{\alpha}}
     \left(\frac{\alpha\mu}{1+\alpha\mu}\right)^y`

    with :math:`E[Y]=\mu\,` and :math:`Var[Y]=\mu+\alpha\mu^2`.


    """
    links = [L.log, L.cloglog, L.identity, L.nbinom, L.Power]
    # TODO: add the ability to use the power links with an if test
    # similar to below
    variance = V.nbinom
    safe_links = [L.Log, ]

    def __init__(self, link=L.log, alpha=1.):
        self.alpha = 1. * alpha  # make it at least float
        self.variance = V.NegativeBinomial(alpha=self.alpha)
        if isinstance(link, L.NegativeBinomial):
            self.link = link(alpha=self.alpha)
        else:
            self.link = link()

    def _clean(self, x):
        """
        Helper function to trim the data so that is in (0,inf)

        Notes
        -----
        The need for this function was discovered through usage and its
        possible that other families might need a check for validity of the
        domain.
        """
        return np.clip(x, FLOAT_EPS, np.inf)

    def deviance(self, endog, mu, freq_weights=1., scale=1.):
        r"""
        Returns the value of the deviance function.

        Parameters
        -----------
        endog : array-like
            Endogenous response variable
        mu : array-like
            Fitted mean response variable
        freq_weights : array-like
            1d array of frequency weights. The default is 1.
        scale : float, optional
            An optional scale argument. The default is 1.

        Returns
        -------
        deviance : float
            Deviance function as defined below

        Notes
        -----
        :math:`D = \sum_i piecewise_i` where :math:`piecewise_i` is defined as:

        If :math:`Y_{i} = 0`:

        :math:`piecewise_i = 2* \log(1 + \alpha * \mu_i) / \alpha`

        If :math:`Y_{i} > 0`:

        .. math:

           piecewise_i = 2 * Y_i * \log(Y_i / \mu_i) - (2 / \alpha) *
            (1 + \alpha * Y_i) * \ln(1 + \alpha * Y_i) / (1 + \alpha * \mu_i)

        """
        iszero = np.equal(endog, 0)
        notzero = 1 - iszero
        endog_mu = self._clean(endog/mu)
        tmp = iszero * 2 * np.log(1 + self.alpha * mu) / self.alpha
        tmp += notzero * (2 * endog * np.log(endog_mu) - 2 / self.alpha *
                          (1 + self.alpha * endog) *
                          np.log((1 + self.alpha * endog) /
                                 (1 + self.alpha * mu)))
        return np.sum(freq_weights * tmp) / scale

    def resid_dev(self, endog, mu, scale=1.):
        r"""
        Negative Binomial Deviance Residual

        Parameters
        ----------
        endog : array-like
            `endog` is the response variable
        mu : array-like
            `mu` is the fitted value of the model
        scale : float, optional
            An optional argument to divide the residuals by scale. The default
            is 1.

        Returns
        --------
        resid_dev : array
            The array of deviance residuals

        Notes
        -----
        :math:`resid\_dev_i = sign(Y_i-\mu_i) * \sqrt{piecewise_i}`

        where :math:`piecewise_i` is defined as

        If :math:`Y_i = 0`:

        :math:`piecewise_i = 2 * \log(1 + \alpha * \mu_i)/ \alpha`

        If :math:`Y_i > 0`:

        :math:`piecewise_i = 2 * Y_i * \log(Y_i / \mu_i) - (2 / \alpha) *
        (1 + \alpha * Y_i) * \log((1 + \alpha * Y_i) / (1 + \alpha * \mu_i))`
        """
        iszero = np.equal(endog, 0)
        notzero = 1 - iszero
        endog_mu = self._clean(endog / mu)
        tmp = iszero * 2 * np.log(1 + self.alpha * mu) / self.alpha
        tmp += notzero * (2 * endog * np.log(endog_mu) - 2 / self.alpha *
                          (1 + self.alpha * endog) *
                          np.log((1 + self.alpha * endog) /
                                 (1 + self.alpha * mu)))
        return np.sign(endog - mu) * np.sqrt(tmp) / scale

    def loglike(self, endog, mu, freq_weights=1., scale=1.):
        r"""
        The log-likelihood function in terms of the fitted mean response.

        Parameters
        ----------
        endog : array-like
            Endogenous response variable
        mu : array-like
            The fitted mean response values
        freq_weights : array-like
            1d array of frequency weights. The default is 1.
        scale : float
            The scale parameter. The default is 1.

        Returns
        -------
        llf : float
            The value of the loglikelihood function evaluated at
            (endog,mu,freq_weights,scale) as defined below.

        Notes
        -----
        Defined as:

        .. math::

           llf = \sum_i freq\_weights_i * (Y_i * \log{(\alpha * e^{\eta_i} /
                 (1 + \alpha * e^{\eta_i}))} - \log{(1 + \alpha * e^{\eta_i})}/
                 \alpha + Constant)

        where :math:`Constant` is defined as:

        .. math::

           Constant = \ln \Gamma{(Y_i + 1/ \alpha )} - \ln \Gamma(Y_i + 1) -
                      \ln \Gamma{(1/ \alpha )}
        """
        lin_pred = self._link(mu)
        constant = (special.gammaln(endog + 1 / self.alpha) -
                    special.gammaln(endog+1)-special.gammaln(1/self.alpha))
        exp_lin_pred = np.exp(lin_pred)
        return np.sum((endog * np.log(self.alpha * exp_lin_pred /
                                      (1 + self.alpha * exp_lin_pred)) -
                      np.log(1 + self.alpha * exp_lin_pred) /
                      self.alpha + constant) * freq_weights)

    def resid_anscombe(self, endog, mu):
        """
        The Anscombe residuals for the negative binomial family

        Parameters
        ----------
        endog : array-like
            Endogenous response variable
        mu : array-like
            Fitted mean response variable

        Returns
        -------
        resid_anscombe : array
            The Anscombe residuals as defined below.

        Notes
        -----
        `resid_anscombe` = (hyp2f1(-alpha*endog)-hyp2f1(-alpha*mu)+\
                1.5*(endog**(2/3.)-mu**(2/3.)))/(mu+alpha*mu**2)**(1/6.)

        where hyp2f1 is the hypergeometric 2f1 function parameterized as
        hyp2f1(x) = hyp2f1(2/3.,1/3.,5/3.,x)
        """

        hyp2f1 = lambda x: special.hyp2f1(2 / 3., 1 / 3., 5 / 3., x)
        return ((hyp2f1(-self.alpha * endog) - hyp2f1(-self.alpha * mu) +
                 1.5 * (endog**(2 / 3.) - mu**(2 / 3.))) /
                (mu + self.alpha * mu**2)**(1 / 6.))


class Tweedie(Family):
    """
    Tweedie family.

    Parameters
    ----------
    link : a link instance, optional
        The default link for the Tweedie family is the log link when the
        link_power is 0. Otherwise, the power link is default.
        Available links are log and Power.
    var_power : float, optional
        The variance power.
    link_power : float, optional
        The link power.

    Attributes
    ----------
    Tweedie.link : a link instance
        The link function of the Tweedie instance
    Tweedie.variance : varfunc instance
        `variance` is an instance of statsmodels.family.varfuncs.Power
    Tweedie.link_power : float
        The power of the link function, or 0 if its a log link.
    Tweedie.var_power : float
        The power of the variance function.

    See also
    --------
    statsmodels.genmod.families.family.Family
    :ref:`links`

    Notes
    -----
    Logliklihood function not implemented because of the complexity of
    calculating an infinite series of summations. The variance power can be
    estimated using the `estimate_tweedie_power` function that is part of the
    `GLM` class.
    """
    links = [L.log, L.Power]
    variance = V.Power
    safe_links = [L.log, L.Power]

    def __init__(self, link=None, var_power=1., link_power=0):
        self.var_power = var_power
        self.link_power = link_power
        self.variance = V.Power(power=var_power * 1.)
        if link_power != 0 and not ((link is L.Power) or (link is None)):
            msg = 'link_power of {} not supported specified link'
            msg = msg.format(link_power)
            raise ValueError(msg)
        if (link_power == 0) and ((link is None) or (link is L.Log)):
            self.link = L.log()
        elif link_power != 0:
            self.link = L.Power(power=link_power * 1.)
        else:
            self.link = link()

    def _clean(self, x):
        """
        Helper function to trim the data so that is in (0,inf)

        Notes
        -----
        The need for this function was discovered through usage and its
        possible that other families might need a check for validity of the
        domain.
        """
        return np.clip(x, 0, np.inf)

    def deviance(self, endog, mu, freq_weights=1., scale=1.):
        r"""
        Returns the value of the deviance function.

        Parameters
        -----------
        endog : array-like
            Endogenous response variable
        mu : array-like
            Fitted mean response variable
        freq_weights : array-like
            1d array of frequency weights. The default is 1.
        scale : float, optional
            An optional scale argument. The default is 1.

        Returns
        -------
        deviance : float
            Deviance function as defined below

        Notes
        -----
        When :math:`p = 1`,

        .. math::

            resid\_dev_i = \mu

        when :math:`endog = 0` and

        .. math::

            resid\_dev_i = endog * \log(endog / \mu) + (\mu - endog)

        otherwise.

        When :math:`p = 2`,

        .. math::

            resid\_dev_i =  (endog - \mu) / \mu - \log(endog / \mu)

        For all other p,

        .. math::

            resid\_dev_i = endog ^{2 - p} / ((1 - p) * (2 - p)) -
                           endog * \mu ^{1 - p} / (1 - p) + \mu ^{2 - p} /
                           (2 - p)

        Once :math:`resid\_dev_i` is calculated, then calculate deviance as

        .. math::

            D = \sum{2 * freq\_weights * resid\_dev_i}
        """
        p = self.var_power
        if p == 1:
            dev = np.where(endog == 0,
                           mu,
                           endog * np.log(endog / mu) + (mu - endog))
        elif p == 2:
            endog1 = np.clip(endog, FLOAT_EPS, np.inf)
            dev = ((endog - mu) / mu) - np.log(endog1 / mu)
        else:
            dev = (endog ** (2 - p) / ((1 - p) * (2 - p)) -
                   endog * mu ** (1-p) / (1 - p) + mu ** (2 - p) / (2 - p))
        return np.sum(2 * freq_weights * dev)

    def resid_dev(self, endog, mu, scale=1.):
        r"""
        Tweedie Deviance Residual

        Parameters
        ----------
        endog : array-like
            `endog` is the response variable
        mu : array-like
            `mu` is the fitted value of the model
        scale : float, optional
            An optional argument to divide the residuals by scale. The default
            is 1.

        Returns
        --------
        resid_dev : array
            The array of deviance residuals

        Notes
        -----
        When :math:`p = 1`,

        .. math::

            resid\_dev_i = \mu

        when :math:`endog = 0` and

        .. math::

            resid\_dev_i = endog * \log(endog / \mu) + (\mu - endog)

        otherwise.

        When :math:`p = 2`,

        .. math::

            resid\_dev_i =  (endog - \mu) / \mu - \log(endog / \mu)

        For all other p,

        .. math::

            resid\_dev_i = endog ^{2 - p} / ((1 - p) * (2 - p)) -
                           endog * \mu ^{1 - p} / (1 - p) + \mu ^{2 - p} /
                           (2 - p)
        """
        p = self.var_power
        if p == 1:
            dev = np.where(endog == 0,
                           mu,
                           endog * np.log(endog / mu) + (mu - endog))
        elif p == 2:
            endog1 = np.clip(endog, FLOAT_EPS, np.inf)
            dev = ((endog - mu) / mu) - np.log(endog1 / mu)
        else:
            dev = (endog ** (2 - p) / ((1 - p) * (2 - p)) -
                   endog * mu ** (1-p) / (1 - p) + mu ** (2 - p) / (2 - p))
        return np.sign(endog - mu) * np.sqrt(2 * dev)

    def loglike(self, endog, mu, freq_weights=1., scale=1.):
        r"""
        The log-likelihood function in terms of the fitted mean response.

        Parameters
        ----------
        endog : array-like
            Endogenous response variable
        mu : array-like
            The fitted mean response values
        freq_weights : array-like
            1d array of frequency weights. The default is 1.
        scale : float
            The scale parameter. The default is 1.

        Returns
        -------
        llf : float
            The value of the loglikelihood function evaluated at
            (endog,mu,freq_weights,scale) as defined below.

        Notes
        -----
        This is not implemented because of the complexity of calculating an
        infinite series of sums.
        """
        return np.nan

    def resid_anscombe(self, endog, mu):
        """
        The Anscombe residuals for the Tweedie family

        Parameters
        ----------
        endog : array-like
            Endogenous response variable
        mu : array-like
            Fitted mean response variable

        Returns
        -------
        resid_anscombe : array
            The Anscombe residuals as defined below.

        Notes
        -----
        When :math:`p = 3`, then

        .. math::

            resid\_anscombe_i = (\log(endog) - \log(\mu)) / \sqrt{mu}

        Otherwise,

        .. math::

            c = (3 - p) / 3

        .. math::

            resid\_anscombe_i = (1 / c) * (endog ^ c - \mu ^ c) / \mu ^{p / 6}
        """
        if self.var_power == 3:
            return (np.log(endog) - np.log(mu)) / np.sqrt(mu)
        else:
            c = (3. - self.var_power) / 3.
            return ((1. / c) * (endog ** c - mu ** c) /
                    mu ** (self.var_power / 6.))
