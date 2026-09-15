# -*- coding: utf-8 -*-
"""
Conditional logit

Sources: sandbox-statsmodels:runmnl.py

General References
--------------------

Greene, W. `Econometric Analysis`. Prentice Hall, 5th. edition. 2003.
Train, K. `Discrete Choice Methods with Simulation`.
    Cambridge University Press. 2003
--------------------
"""

import time
from statsmodels.compat.collections import OrderedDict

import numpy as np
import pandas as pd

from statsmodels.tools.decorators import cache_readonly
from .discrete_model import Logit
from .dcm_base import (DiscreteChoiceModel, DiscreteChoiceModelResults)


### Public Model Classes ####

class CLogit(DiscreteChoiceModel):
    __doc__ = """
    Conditional Logit

    Parameters
    ----------
    endog_data : array
        dummy encoding of realized choices.
    exog_data : array (nobs, k*)
        array with explanatory variables.Variables for the model are select
        by V, so, k* can be >= than k. An intercept is not included by
        default and should be added by the user.
    V: dict
        a dictionary with the names of the explanatory variables for the
        utility function for each alternative.
        Alternative specific variables (common coefficients) have to be first.
        For specific variables (various coefficients), choose an alternative
        and drop all specific variables on it.
    ncommon : int
        number of explanatory variables with common coefficients.
    ref_level : str
        Name of the key for the alternative of reference.
    name_intercept : str
        name of the column with the intercept. 'None' if an intercept is not
        included.

    Attributes
    ----------
    endog : array (nobs*J, )
        the endogenous response variable
    endog_bychoices: array (nobs,J)
        the endogenous response variable by choices
    exog_matrix: array   (nobs*J,K)
        the enxogenous response variables
    exog_bychoices: list of arrays J * (nobs,K)
        the enxogenous response variables by choices. one array of exog
        for each choice.
    nobs : float
        number of observations.
    J  : float
        The number of choices for the endogenous variable. Note that this
        is zero-indexed.
    K : float
        The actual number of parameters for the exogenous design.  Includes
        the constant if the design has one and excludes the constant of one
        choice which should be dropped for identification.
    loglikeobs
    params
    score
    jac
    hessian
    information
    predict
    residuals
    resid_misclassified
    pred_table
    summary : Summary instance
        summarize the results inside CLogitResults class.

    Notes
    -----
    Utility for choice j is given by

        $V_j = X_j * beta + Z * gamma_j$

    where X_j contains generic variables (terminology Hess) that have the same
    coefficient across choices, and Z are variables, like individual-specific
    variables that have different coefficients across variables.

    If there are choice specific constants, then they should be contained in Z.
    For identification, the constant of one choice should be dropped.

    """

    def cdf(self, item):
        """
        Conditional Logit cumulative distribution function.

        Parameters
        ----------
        X : array (nobs,K)
            the linear predictor of the model.

        Returns
        --------
        cdf : ndarray
            the cdf evaluated at `X`.

        Notes
        -----
        .. math:: \\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right){\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}
        """
        eXB = np.exp(item)
        return  eXB / eXB.sum(1)[:, None]

    def pdf(self, item):
        """
        Conditional Logit probability density function.

        """
        raise NotImplementedError

    def loglike(self, params):

        """
        Log-likelihood of the conditional logit model.

        Parameters
        ----------
        params : array
            the parameters of the conditional logit model.

        Returns
        -------
        loglike : float
            the log-likelihood function of the model evaluated at `params`.

        Notes
        ------
        .. math:: \\ln L=\\sum_{i=1}^{n}\\sum_{j=0}^{J}d_{ij}\\ln\\left(\\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\right)

        where :math:`d_{ij}=1` if individual `i` chose alternative `j` and 0
        if not.

        """

        xb = self.xbetas(params)
        loglike = (self.endog_bychoices * np.log(self.cdf(xb))).sum(1)
        return loglike.sum()

    def loglikeobs(self, params):
        """
        Log-likelihood for each observation.

        Parameters
        ----------
        params : array-like
            The parameters of the model.

        Returns
        -------
        loglike : ndarray (nobs,K)
            The log likelihood for each observation of the model evaluated
            at `params`. See Notes

        Notes
        ------
        .. math:: \\ln L_{i}=\\sum_{j=0}^{J}d_{ij}\\ln\\left(\\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\right)

        for observations :math:`i=1,...,n`

        where :math:`d_{ij}=1` if individual `i` chose alternative `j` and 0
        if not.
        """

        xb = self.xbetas(params)
        return  (self.endog_bychoices * np.log(self.cdf(xb))).sum(1)

    def score(self, params):
        """
        Score/gradient matrix for conditional logit model log-likelihood

        Parameters
        ----------
        params : array
            the parameters of the conditional logit model.

        Returns
        --------
        score : ndarray 1d (K)
            the score vector of the model evaluated at `params`.

        Notes
        -----
        It is the first derivative of the loglikelihood function of the
        conditional logit model evaluated at `params`.

        .. math:: \\frac{\\partial\\ln L}{\\partial\\beta_{j}}=\\sum_{i}\\left(d_{ij}-\\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\right)x_{i}

        for :math:`j=1,...,J`
        """

        firstterm = (self.endog_bychoices - self.cdf(self.xbetas(params)))\
                    .reshape(-1, 1)
        return np.dot(firstterm.T, self.exog).flatten()

    def jac(self, params):
        """
        Jacobian matrix for conditional logit model log-likelihood.

        Parameters
        ----------
        params : array
            the parameters of the conditional logit model.

        Returns
        --------
        jac : ndarray, (nobs, K)
            the jacobian for each observation.

        Notes
        -----
        It is the first derivative of the loglikelihood function of the
        conditional logit model for each observation evaluated at `params`.

        .. math:: \\frac{\\partial\\ln L_{i}}{\\partial\\beta_{j}}=\\left(d_{ij}-\\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\right)x_{i}

        for :math:`j=1,...,J`, for observations :math:`i=1,...,n`

        """

        firsterm = (self.endog_bychoices - self.cdf(self.xbetas(params)))\
                    .reshape(-1, 1)
        return (firsterm * self.exog)

    def hessian(self, params):
        """
        Conditional logit Hessian matrix of the log-likelihood

        Parameters
        -----------
        params : array-like
            The parameters of the model

        Returns
        -------
        hess : ndarray, (K, K)
            The Hessian
        Notes
        -----
        It is the second derivative with respect to the flattened parameters
        of the loglikelihood function of the conditional logit model
        evaluated at `params`.

        .. math:: \\frac{\\partial^{2}\\ln L}{\\partial\\beta_{j}\\partial\\beta_{l}}=-\\sum_{i=1}^{n}\\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\left[\\boldsymbol{1}\\left(j=l\\right)-\\frac{\\exp\\left(\\beta_{l}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\right]x_{i}x_{l}^{\\prime}

        where
        :math:`\\boldsymbol{1}\\left(j=l\\right)` equals 1 if `j` = `l` and 0
        otherwise.
        """

        # TODO: analytical derivatives
        from statsmodels.tools.numdiff import approx_hess
        # need options for hess (epsilon)
        return approx_hess(params, self.loglike)

    def information(self, params):
        """
        Fisher information matrix of model

        Returns -Hessian of loglike evaluated at params.
        """
        raise NotImplementedError

    def fit(self, start_params=None, maxiter=10000, maxfun=5000,
            method="newton", full_output=1, disp=None, callback=None, **kwds):
        """
        Fits CLogit() model using maximum likelihood.
        In a model linear the log-likelihood function of the sample, is
        global concave for β parameters, which facilitates its numerical
        maximization (McFadden, 1973).
        Fixed Method = Newton, because it'll find the maximum in a few
        iterations. Newton method require a likelihood function, a gradient,
        and a Hessian. Since analytical solutions are known, we give it.
        Initial parameters estimates from the standard logit
        Returns
        -------
        Fit object for likelihood based models
        See: GenericLikelihoodModelResults

        """

        if start_params is None:

            Logit_res = Logit(self.endog, self.exog_matrix).fit(disp=0)
            start_params = Logit_res.params.values

        else:
            start_params = np.asarray(start_params)

        start_time = time.time()
        model_fit = super(CLogit, self).fit(disp = disp,
                                            start_params = start_params,
                                            method=method, maxiter=maxiter,
                                            maxfun=maxfun, **kwds)

        self.params = model_fit.params
        end_time = time.time()
        self.elapsed_time = end_time - start_time

        return model_fit

    def predict(self, params, linear=False):
        """
        Predict response variable of a model given exogenous variables.

        Parameters
        ----------
        params : array-like
            Fitted parameters of the model.
        linear : bool, optional
            If True, returns the linear predictor dot(exog_bychoices,params).
            Else, returns the value of the cdf at the linear predictor.

        Returns
        -------
        array (nobs,K)
            Fitted values at exog.
        """
        if not linear:
            return self.cdf(self.xbetas(params))
        else:
            return self.xbetas(params)

    def summary(self):
        # TODO add __doc__ = CLogitResults.__doc__
        return CLogitResults(self).summary()

    @cache_readonly
    def residuals(self):
        """
        Residuals

        Returns
        -------
        array
            Residuals.

        Notes
        -----
        The residuals for the Conditional Logit model are defined as

        .. math:: y_i - p_i

        where : math:`y_i` is the endogenous variable and math:`p_i`
        predicted probabilities for each category value.
        """

        return self.endog_bychoices - self.predict(self.params)

    @cache_readonly
    def resid_misclassified(self):
        """
        Residuals indicating which observations are misclassified.

        Returns
        -------
        array (nobs,K)
            Residuals.

        Notes
        -----
        The residuals for the Conditional Logit model are defined

        .. math:: argmax(y_i) \\neq argmax(p_i)

        where :math:`argmax(y_i)` is the index of the category for the
        endogenous variable and :math:`argmax(p_i)` is the index of the
        predicted probabilities for each category. That is, the residual
        is a binary indicator that is 0 if the category with the highest
        predicted probability is the same as that of the observed variable
        and 1 otherwise.
        """
        # it's 0 or 1 - 0 for correct prediction and 1 for a missed one

        return (self.endog_bychoices.argmax(1) !=
                self.predict(self.params).argmax(1)).astype(float)

    def pred_table(self, params = None):
        """
        Returns the J x J prediction table.

        Parameters
        ----------
        params : array-like, optional
            If None, the parameters of the fitted model are used.

        Notes
        -----
        pred_table[i,j] refers to the number of times "i" was observed and
        the model predicted "j". Correct predictions are along the diagonal.
        """
        # TODO. doesn't match the results of green.
        # greene = np.array([[ 32.,   8.,   5.,  13.],
        #                   [  7.,  37.,   5.,  14.],
        #                   [  3.,   5.,  15.,   6.],
        #                   [ 16.,  13.,   6.,  25.]])
        # clogit_mod.endog_bychoices.sum(0) #OK: 58.,  63.,  30.,  59.
        # clogit_mod.predict(clogit_res.params).sum(0) #OK: 58.,  63.,  30.,  59.
        # clogit_mod.pred_table().sum(0) # 56.,  64.,  23.,  67.
        # clogit_mod.pred_table().sum(1) # OK: 58.,  63.,  30.,  59.

        if params == None:
            params = self.params

        # these are the real choices
        idx = self.endog_bychoices.argmax(1)
        # these are the predicted choices
        idy = self.predict(params).argmax(1)

        return np.histogram2d(idx, idy, bins = self.J)[0]


### Results Class ###
class CLogitResults(DiscreteChoiceModelResults):

    def summary(self, title = None, alpha = .05):
        """Summarize the Clogit Results

        Parameters
        -----------
        title : string, optional
            Title for the top table. If not None, then this replaces the
            default title
        alpha : float
            significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : class to hold summary
            results

        """

        top_left = [('Dep. Variable:', None),
                    ('Model:', [self.model.__class__.__name__]),
                    ('Method:', [self.mle_settings['optimizer']]),
                    ('Date:', None),
                    ('Time:', None),
                    ('Converged:', ["%s" % self.mle_retvals['converged']]),
                    ('Iterations:', ["%s" % self.mle_retvals['iterations']]),
                    ('Elapsed time (seg.):',
                                    ["%10.4f" % self.model.elapsed_time]),
                    ('Num. alternatives:', [self.model.J])
                      ]

        top_right = [
                     ('No. Cases:', [self.nobs]),
                     ('No. Observations:', [self.nobs_bychoice]),
                     ('Df Residuals:', [self.model.df_resid]),
                     ('Df Model:', [self.model.df_model]),
                     ('Log-Likelihood:', None),
                     ('LL-Null:', ["%#8.5g" % self.llnull]),
                     ('Pseudo R-squ.:', ["%#6.4g" % self.prsquared]),
                     ('LLR p-value:', ["%#6.4g" % self.llr_pvalue]),
                     ('Likelihood ratio test:', ["%#8.5g" %self.llrt]),
                     ('AIC:', ["%#8.5g" %self.aic])

                                     ]
        if title is None:
            title = self.model.__class__.__name__ + ' ' + \
            "results"

        #boiler plate
        from statsmodels.iolib.summary import Summary, SimpleTable

        smry = Summary()
        # for top of table
        smry.add_table_2cols(self, gleft=top_left, gright=top_right,
                             title=title)

        # Frequencies of alternatives
        mydata = [self.freq_alt, self.perc_alt]
        myheaders = self.alt
        mytitle = ("")
        mystubs = ["Frequencies of alternatives: ", "Percentage:"]
        tbl = SimpleTable(mydata, myheaders, mystubs, title = mytitle,
                          data_fmts = ["%5.2f"])
        smry.tables.append(tbl)

        # for parameters
        smry.add_table_params(self, alpha=alpha, use_t=False)

        return smry


# Marginal Effects
# TODO: move to dcm_base

class ClogitMargeff(CLogit):

    __doc__ = """
    Conditional Logit Marginal effects
        Derivatives of the probabilitis with respect to the explanatory variables.

    Notes
    -----
    * for alternative specific coefficients are:

        P[j] * (params_asc[j] - sum_k P[k]*params_asc[k])

        where params_asc are params for the individual-specific variables

        See _derivative_exog in class MultinomialModel

    * for generic coefficient are:

        params_gen[j] * P[j] (1 - P[j])

        where params_gen are params for the alternative-specific variables.

    """

    def __init__(self, model):

        self.model = model
        params = model.params
        self.ncommon = model.ncommon
        K = model.K
        self.J = model.J
        exog = model.exog_matrix

        self.K_asc = self.ncommon
        self.K_gen = K - self.ncommon

        self.exog_asc = exog[np.arange(self.ncommon)]
        self.exog_gen = exog[np.arange(self.ncommon, K)]

        self.params_asc = params[np.arange(self.ncommon)]
        self.params_gen = params[np.arange(self.ncommon, K)]

    def _derivative_exog_asc(self):
        return NotImplementedError

    def _derivative_exog_gen(self):
        return NotImplementedError

    def get_margeff(self):
        return NotImplementedError

if __name__ == "__main__":

    DEBUG = 0
    print('Example:')
    import statsmodels.api as sm
    # Loading data as pandas object
    data = sm.datasets.modechoice.load_pandas()
    data.endog[:5]
    data.exog[:5]
    data.exog['Intercept'] = 1  # include an intercept
    y, X = data.endog, data.exog

    # Set up model

    # Names of the variables for the utility function for each alternative
    # variables with common coefficients have to be first in each array
    V = OrderedDict((
        ('air',   ['gc', 'ttme', 'Intercept', 'hinc']),
        ('train', ['gc', 'ttme', 'Intercept']),
        ('bus',   ['gc', 'ttme', 'Intercept']),
        ('car',   ['gc', 'ttme']))
        )
    # Number of common coefficients
    ncommon = 2

    # Describe model
    clogit_mod = CLogit(y, X,  V, ncommon,
                        ref_level = 'car', name_intercept = 'Intercept')
    # Fit model
    clogit_res = clogit_mod.fit(disp=1)

    # Summarize model
    print(clogit_mod.summary())

    # Marginal Effects
    # clogit_margeff = ClogitMargeff(clogit_mod)
    # print clogit_margeff._derivative_exog_asc()
