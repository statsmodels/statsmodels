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

# TODO:
    adapt it to the structure of others discrete models
        (source:discrete_model.py)
    add dataset Mode choice
        (http://statsmodels.sourceforge.net/devel/datasets/
            dataset_proposal.html#dataset-proposal)
    add example
        (http://statsmodels.sourceforge.net/devel/dev/examples.html)
    add test
    send patsy proposal for data handle (see Issue #941)

"""
import numpy as np
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.discrete.discrete_model import DiscreteResults
import time

class CLogit(GenericLikelihoodModel):
    __doc__ = """
    Conditional Logit

    Parameters
    ----------
    endog : array (nobs,nchoices)
        dummy encoding of realized choices
    exog_bychoices : list of arrays
        explanatory variables, one array of exog for each choice. Variables
        with common coefficients have to be first in each array
    ncommon : int
        number of explanatory variables with common coefficients

    Attributes
    ----------
    endog : array
        A reference to the endogenous response variable
    exog : array
        A reference to the exogenous design.
    J / nchoices : float
        The number of choices for the endogenous variable. Note that this
        is zero-indexed.
    K : float
        The actual number of parameters for the exogenous design.  Includes
        the constant if the design has one.

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

    def __init__(self, endog, exog_bychoices, ncommon, **kwds):

        super(CLogit, self).__init__(endog, **kwds)
        self.endog = endog
        self.exog_bychoices = exog_bychoices
        self.ncommon = ncommon
        self.nobs, self.nchoices = endog.shape

        paramsind = [exog_bychoices[ii].shape[1]-ncommon for ii in range(self.nchoices)]
        zi = np.r_[[ncommon], ncommon + np.array(paramsind).cumsum()]
        self.zi = zi
        z = np.arange(max(zi))

        params_indices = [np.r_[np.arange(ncommon), z[zi[ii]:zi[ii+1]]]
                       for ii in range(len(zi)-1)]

        self.params_indices = params_indices

        params_num = []                                 # params to estimate
        for sublist in params_indices:
            for item in sublist:
                if item not in params_num:
                    params_num.append(item)

        self.params_num = params_num

        self.df_model = len(self.params_num)
        self.df_resid = int(self.nobs - len(self.params_num))


        # TODO cleanup J/nchoice K / numparams
        self.J = self.nchoices
        self.K = len(self.params_num)

    def _build_exog(self):
        """
        Build the exogenous matrix
        """
        # TODO exog_names. See at the end

        return NotImplementedError

    def xbetas(self, params):
        '''these are the V_i
        '''
        res = np.empty((self.nobs, self.nchoices))
        for choiceind in range(self.nchoices):
            res[:, choiceind] = np.dot(self.exog_bychoices[choiceind],
                                      params[self.params_indices[choiceind]])
        return res

    def cdf(self, X):
        """
        Conditional Logit cumulative distribution function.

        Parameters
        ----------
        X : array
            The linear predictor of the model XB.

        Returns
        --------
        cdf : ndarray
            The cdf evaluated at `X`.

        Notes
        -----
        The cdf is the same as in the multinomial logit model.
        .. math:: \\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}
        """
        eXB = np.exp(X)
        return eXB/eXB.sum(1)[:, None]


    def loglike(self, params):

        """
        Log-likelihood of the conditional logit model.

        Parameters
        ----------
        params : array-like
            The parameters of the conditional logit model.

        Returns
        -------
        loglike : float
            The log-likelihood function of the model evaluated at `params`.
            See notes.

        Notes
        ------
        .. math:: \\ln L=\\sum_{i=1}^{n}\\sum_{j=0}^{J}d_{ij}\\ln\\left(\\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\right)

        where :math:`d_{ij}=1` if individual `i` chose alternative `j` and 0
        if not.

        The loglike is the same as for the multinomial logit model.
        """

        xb = self.xbetas(params)
        loglike = (self.endog * np.log(self.cdf(xb))).sum(1)

        return loglike.sum()


    def scoreX(self, params):
        """
        Score/gradient matrix for conditional logit model log-likelihood

        Parameters
        ----------
        params : array
            The parameters of the conditional logit model.

        Returns
        --------
        score : ndarray, (K * (J-1),)
            The 2-d score vector, i.e. the first derivative of the
            loglikelihood function, of the conditional logit model evaluated at
            `params`.

        Notes
        -----
        .. math:: \\frac{\\partial\\ln L}{\\partial\\beta_{j}}=\\sum_{i}\\left(d_{ij}-\\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\right)x_{i}

        for :math:`j=1,...,J`

        In the multinomial model the score matrix is K x J-1 but is returned
        as a flattened array to work with the solvers.
        """

        #firstterm = self.endog[:,1:] - self.cdf(np.dot(self.xbetas(params)))[:,1:]
        #return np.dot(firstterm.T, self.exog).flatten()

        raise NotImplementedError

    def jacX(self, params):
        """
        Jacobian matrix for conditional logit model log-likelihood

        Parameters
        ----------
        params : array
            The parameters of the conditional logit model.

        Returns
        --------
        jac : ndarray, (nobs, k_vars*(J-1))
            The derivative of the loglikelihood for each observation evaluated
            at `params` .

        Notes
        -----
        .. math:: \\frac{\\partial\\ln L_{i}}{\\partial\\beta_{j}}=\\left(d_{ij}-\\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\right)x_{i}

        for :math:`j=1,...,J`, for observations :math:`i=1,...,n`

        In the multinomial model the score vector is K x (J-1) but is returned
        as a flattened array. The Jacobian has the observations in rows and
        the flatteded array of derivatives in columns.
        """

       # firstterm = self.endog[:,1:] - self.cdf(self.xbetas(params))[:,1:]
       # return (firstterm[:,:,None] * self.exog[:,None,:]).reshape(self.exog.shape[0], -1)
        return NotImplementedError

    def hessianX(self, params):
        """
        Conditional logit Hessian matrix of the log-likelihood

        Parameters
        -----------
        params : array-like
            The parameters of the model

        Returns
        -------
        hess : ndarray, (J*K, J*K)
            The Hessian, second derivative of loglikelihood function with
            respect to the flattened parameters, evaluated at `params`

        Notes
        -----
        .. math:: \\frac{\\partial^{2}\\ln L}{\\partial\\beta_{j}\\partial\\beta_{l}}=-\\sum_{i=1}^{n}\\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\left[\\boldsymbol{1}\\left(j=l\\right)-\\frac{\\exp\\left(\\beta_{l}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\right]x_{i}x_{l}^{\\prime}

        where
        :math:`\\boldsymbol{1}\\left(j=l\\right)` equals 1 if `j` = `l` and 0
        otherwise.

        In the multinomial model the actual Hessian matrix has J**2 * K x K elements. The Hessian
        is reshaped to be square (J*K, J*K) so that the solvers can use it. This implementation does not take advantage of the symmetry of
        the Hessian and could probably be refactored for speed.
        """

        return NotImplementedError


    def fit(self, start_params=None, maxiter=10000, maxfun=5000, method="newton",
            full_output=1, disp=1, callback=None,**kwds):
        """
        Fits CLogit() model using maximum likelihood.
        In a model linear the log-likelihood function of the sample, is
        global concave for β parameters, which facilitates its numerical
        maximization (McFadden, 1973).
        Fixed Method = Newton, because it'll find the maximum in a few iterations.
        Newton method require a likelihood function, a score/gradient,
        and a Hessian. Since analytical solutions are known, we give it.

        Returns
        -------
        Fit object for likelihood based models
        See: GenericLikelihoodModelResults

        """
        #ORRO (2006) documentación np.ones??
        if start_params is None:
            start_params = np.zeros(len(self.params_num))
        else:
            start_params = np.asarray(start_params)

        # TODO: check number of  iterations. Seems too high.
        start_time = time.time()

        model_fit =  super(CLogit, self).fit(start_params=start_params, method=method,
                                    maxiter=maxiter, maxfun=maxfun,**kwds)

        end_time = time.time()
        print("the elapsed time was %g seconds" % (end_time - start_time))
        return model_fit

### Results Class ###

class CLogitResults (DiscreteResults):

# TODO on summary: frequencies of alternatives, McFadden R^2, Likelihood
#   ratio test, method, iterations.

    def __init__(self, model, mlefit):
        #super(DiscreteResults, self).__init__(model, params,
        #        np.linalg.inv(-hessian), scale=1.)
        self.model = model
        self.df_model = model.df_model
        self.df_resid = model.df_resid
        self.nobs = model.endog.shape[0]
        self.__dict__.update(mlefit.__dict__)

    def __getstate__(self):
        try:
            #remove unpicklable callback
            self.mle_settings['callback'] = None
        except (AttributeError, KeyError):
            pass
        return self.__dict__

    def _get_endog_name(self, yname, yname_list):
        if yname is None:
            yname = self.model.endog_names
        if yname_list is None:
            yname_list = self.model.endog_names
        return yname, yname_list

    def summary(self, yname=None, xname=None, title=None, alpha=.05,
                yname_list=None):

        """Summarize the Clogit Results

        Parameters
        -----------
        yname : string, optional
            Default is `y`
        xname : list of strings, optional
            Default is `var_##` for ## in p the number of regressors
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
                     ('Method:', ['NotImplemented'] ),
                     ('Date:', None),
                     ('Time:', None),
                     ('Converged:', ["%s" % self.mle_retvals['converged']]),
                     ('Elapsed time:' , ['NotImplemented'] )
                      ]

        top_right = [('No. Observations:', None),
                     ('Df Residuals:', None),
                     ('Df Model:', None),
                     ('Log-Likelihood:', None),
                     ]

        if title is None:
            title = self.model.__class__.__name__ + ' ' + \
            "new class of results - in process of implementing"

        #boiler plate
        from statsmodels.iolib.summary import Summary
        smry = Summary()
        # for top of table
        smry.add_table_2cols(self, gleft=top_left, gright=top_right,
                             title=title)
        # for parameters, etc
        smry.add_table_params(self, alpha=alpha, use_t=False)

        return smry

if __name__ == "__main__":

    # examples
    import dcm_clogit_examples
