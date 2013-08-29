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

import numpy as np
import pandas as pd
from statsmodels.base.model import (GenericLikelihoodModel,
                                    GenericLikelihoodModelResults)
import statsmodels.api as sm
import time


class CLogit(GenericLikelihoodModel):
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
        For specific variables (various coefficients), choose an alternative and
        drop all specific variables on it.
    ncommon : int
        number of explanatory variables with common coefficients.

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

    def __init__(self, endog_data, exog_data, V, ncommon, **kwds):

        self.exog_data = exog_data
        self.V = V
        self.ncommon = ncommon

        self.J = len(self.V)
        self.nobs = endog_data.shape[0] / self.J

        # Endog_bychoices
        self.endog_bychoices = endog_data.values.reshape(-1, self.J)
        # Exog_bychoices
        exog_bychoices = []
        exog_bychoices_names = []
        self.choice_index = np.arange(self.J * self.nobs) % self.J

        for ii, key in enumerate(iter(self.V)):
            (exog_bychoices.append(self.exog_data[self.V[key]]
                                    [self.choice_index == ii]
                                    .values.reshape(self.nobs, -1)))

        for key in self.V:
            exog_bychoices_names.append(self.V[key])

        self.exog_bychoices = exog_bychoices

        # Betas
        beta_not_common = ([len(exog_bychoices_names[ii]) - self.ncommon
                            for ii in range(self.J)])
        zi = np.r_[[self.ncommon], self.ncommon + np.array(beta_not_common)\
                    .cumsum()]
        z = np.arange(max(zi))
        beta_ind = [np.r_[np.arange(self.ncommon), z[zi[ii]:zi[ii + 1]]]
                               for ii in range(len(zi) - 1)]  # index of betas
        beta_ind = beta_ind
        self.beta_ind = beta_ind
        beta_ind_str = ([map(str, (np.r_[np.arange(self.ncommon),
                                        z[zi[ii]:zi[ii + 1]]]).tolist())
                               for ii in range(len(zi) - 1)])  # str index of betas

        betas = {}

        for sublist in range(self.J):
            aa = []
            for ii in range(len(exog_bychoices_names[sublist])):
                aa.append(beta_ind_str[sublist][ii]
                          + "_" + exog_bychoices_names[sublist][ii])
            betas[sublist] = aa

        for key in betas:
            print '{0} => {1:10}'.format(V.keys()[key], betas[key])

        # Exog
        pieces = []
        Vkeys = []
        for ii in range(self.J):
            pieces.append(pd.DataFrame(exog_bychoices[ii], columns=betas[ii]))
            Vkeys.append(ii + 1)

        self.exog_matrix_all = (pd.concat(pieces, axis = 0, keys = Vkeys,
                                     names =['choice', 'nobs'])
                           .fillna(value = 0).sortlevel(1).reset_index())

        self.exog_matrix = self.exog_matrix_all.iloc[:, 2:]

        print 'Parameters to estimate: '
        print self.exog_matrix.columns.tolist()

        super(CLogit, self).__init__(endog = endog_data, exog = self.exog_matrix,
                                        **kwds)

        self.K = len(self.exog_matrix.columns)
        self.df_model = self.K
        self.df_resid = int(self.nobs - self.K)

    def xbetas(self, params):
        '''the Utilities V_i

        '''
        res = np.empty((self.nobs, self.J))
        for choiceind in range(self.J):
            res[:, choiceind] = np.dot(self.exog_bychoices[choiceind],
                                      params[self.beta_ind[choiceind]])
        return res

    def cdf(self, X):
        """
        Conditional Logit cumulative distribution function.

        Parameters
        ----------
        X : array
            the linear predictor of the model.

        Returns
        --------
        cdf : ndarray
            the cdf evaluated at `X`.

        Notes
        -----
        .. math:: \\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right){\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}
        """
        eXB = np.exp(X)
        return  eXB / eXB.sum(1)[:, None]

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
        loglike : ndarray (nobs,)
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
        return  np.dot(firstterm.T, self.exog).flatten()

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

    def hessianX(self, params):
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
        return NotImplement

    def fit(self, start_params=None, maxiter=10000, maxfun=5000,
            method="newton", full_output=1, disp=1, callback=None, **kwds):
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
            print 'Estimating initial parameters..'
            logit_mod = sm.Logit(self.endog, self.exog_matrix)
            logit_res = logit_mod.fit()
            start_params = logit_res.params.values
            print start_params
        else:
            start_params = np.asarray(start_params)

        start_time = time.time()
        print 'Estimating model..'
        model_fit = super(CLogit, self).fit(start_params=start_params,
                        method=method, maxiter=maxiter, maxfun=maxfun, **kwds)
        end_time = time.time()
        self.elapsed_time = end_time - start_time

        print("the elapsed time was %g seconds" % (self.elapsed_time))
        return model_fit

### Results Class ###


class CLogitResults (GenericLikelihoodModelResults):

    # TODO on summary: McFadden R^2, Likelihood ratio test

    def __init__(self, model, mlefit, **kwds):

        self.model = model
        self.mlefit = mlefit
        self.nobs_bychoice = model.nobs
        self.__dict__.update(mlefit.__dict__)
        self.alt = model.V.keys()
        self.freq_alt = (model.endog_bychoices[:, ].sum(0) / model.nobs)\
                        .tolist()

    def __getstate__(self):
        try:
            #remove unpicklable callback
            self.mle_settings['callback'] = None
        except (AttributeError, KeyError):
            pass
        return self.__dict__


    def summary(self, title= None, alpha=.05):
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
            this holds the summary tables and text, which caclon be printed or
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

        top_right = [('No. Cases:', [self.nobs]),
                     ('No. Observations:', [self.nobs_bychoice]),
                     ('Df Residuals:', None),
                     ('Df Model:', None),
                     ('Log-Likelihood:', None)
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
        mydata = [self.freq_alt]
        myheaders = self.alt
        mytitle = ("Frequencies of alternatives: ")

        tbl = SimpleTable(mydata, myheaders, title = mytitle,
                          data_fmts = ["%3.3f"])
        smry.tables.append(tbl)

        # for parameters
        # TODO rename parameters

        smry.add_table_params(self, alpha=alpha, use_t=False)

        return smry

if __name__ == "__main__":

    from patsy import dmatrices

    url = "http://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/ModeChoice.csv"
    file_ = "ModeChoice.csv"
    import os
    if not os.path.exists(file_):
        import urllib
        urllib.urlretrieve(url, "ModeChoice.csv")
    df = pd.read_csv(file_)
    df.describe()

    f = 'mode  ~ ttme+invc+invt+gc+hinc+psize'
    y, X = dmatrices(f, df, return_type='dataframe')

    # Names of the variables for the utility function for each alternative
    # variables with common coefficients have to be first in each array
    V = {
        "air": ['gc', 'ttme', 'Intercept', 'hinc'],
        "train": ['gc', 'ttme', 'Intercept'],
        "bus ": ['gc', 'ttme', 'Intercept'],
        "car": ['gc', 'ttme']
        }

    # Number of common coefficients
    ncommon = 2

    # Model
    print 'Example:'
    start_time = time.time()

    clogit_mod = CLogit(y, X,  V, ncommon)
    clogit_res = clogit_mod.fit()

    end_time = time.time()
    print("the whole elapsed time was %g seconds.") % (end_time - start_time)

    print clogit_res.params
    print CLogitResults(clogit_mod, clogit_res).summary()
