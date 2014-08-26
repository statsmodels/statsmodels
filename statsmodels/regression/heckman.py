"""
Heckman correction for sample selection bias.

Created August 19, 2014 by B.I.
Last modified August 25, 2014 by B.I.

NO warranty is provided for this software.
"""

import numpy as np
import statsmodels.api as sm
import statsmodels.base.model as base
from statsmodels.iolib import summary
from scipy.stats import norm
from scipy.stats import t


class Heckman(base.LikelihoodModel):
    """
    Class for Heckman correction for sample selection bias model.

    Attributes
    ----------
    endog : 1darray
        Data for the dependent variable. Should be set to np.nan for
        censored observations.
    exog : 2darray
        Data for the regression (response) equation
    exog_select : 2darray
        Data for the selection equation
    """

    def __init__(self, endog, exog, exog_select):
        #TODO: add code to take care of missing data in X and Z
        #TODO: add type checking and shape checking
        #TODO: make sure that selection equation contains at least one more unique var than in reg eqn

        # store data
        self.exog_select = exog_select
        self.treated = np.asarray(~np.isnan(endog))
        super(Heckman, self).__init__(endog, exog)

        # store observation counts
        self.nobs_total = endog.size
        self.nobs_uncensored = self.nobs = np.sum(self.treated)
        self.nobs_censored = self.nobs_total - self.nobs_uncensored

        # store variable names if data came in as Pandas objects
        try:
            self.yname = endog.name
        except AttributeError:
            self.yname = None

        try:
            self.xname = [v for v in exog.columns]
        except AttributeError:
            try:
                self.xname = exog.name
            except AttributeError:
                self.xname = None

        try:
            self.zname = [v for v in exog_select.columns]
        except AttributeError:
            try:
                self.zname = exog_select.name
            except AttributeError:
                self.zname = None

    def initialize(self):
        self.wendog = self.endog
        self.wexog = self.exog
        self.wexog_select = self.exog_select

    def whiten(self, data):
        """
        Model whitener for Heckman correction model does nothing.
        """
        return data

    def fit(self, method='2step'):
        """
        Fit the Heckman selection model.

        Parameters
        ----------
        method : str
            Can only be "2step", which uses Heckman's two-step method.

        Returns
        -------
        A HeckmanResults class instance.

        See Also
        ---------
        HeckmanResults


        """

        ## prep data
        Y = np.asarray(self.endog)
        Y = Y[self.treated]

        X = np.asarray(self.exog)
        X = X[self.treated,:]

        Z = np.asarray(self.exog_select)

        ## fit
        if method=='twostep':
            results = self._fit_twostep(Y, X, Z)
        elif method=='mle':
            results = self._fit_mle(Y, X, Z)
        else:
            raise ValueError("Invalid choice for estimation method.")


        ## return fitted Heckman model
        return results


    def _fit_twostep(self, Y, X, Z):
        ########################################################################
        # PRIVATE METHOD
        # Fits using Heckman two-step from Heckman (1979).
        ########################################################################

        ## Step 1
        step1model = sm.Probit(self.treated, Z)
        step1res = step1model.fit(disp=False)
        step1_fitted = np.atleast_2d(step1res.fittedvalues).T
        step1_varcov = step1res.cov_params()

        inverse_mills = norm.pdf(step1_fitted)/norm.cdf(step1_fitted)

        ## Step 2
        W = np.hstack((X, inverse_mills[self.treated] ) )
        step2model = sm.OLS(Y, W)
        step2res = step2model.fit()

        params = step2res.params[:-1]
        betaHat_inverse_mills = step2res.params[-1]


        ## Compute standard errors
        # Compute estimated error variance of censored regression
        delta = np.multiply(inverse_mills, inverse_mills + step1_fitted)[self.treated]

        sigma2Hat = step2res.resid.dot(step2res.resid) / self.nobs_uncensored + \
            (betaHat_inverse_mills**2 * sum(delta)) / self.nobs_uncensored
        sigma2Hat = sigma2Hat[0]
        sigmaHat = np.sqrt(sigma2Hat)
        rhoHat = betaHat_inverse_mills / sigmaHat

        # compute standard errors of beta estimates of censored regression
        R = np.zeros([self.nobs_uncensored, self.nobs_uncensored])
        for i in range(self.nobs_uncensored):
            R[i,i] = 1 - rhoHat**2 * delta[i]

        D = np.zeros([self.nobs_uncensored, self.nobs_uncensored])
        for i in range(self.nobs_uncensored):
            D[i,i] = delta[i]


        Q = rhoHat**2 * (W.T.dot(D).dot(Z[self.treated])).dot(step1_varcov).dot(Z[self.treated].T.dot(D).dot(W))

        normalized_varcov_all = np.linalg.inv(W.T.dot(W)).dot(W.T.dot(R).dot(W)+Q).dot(np.linalg.inv(W.T.dot(W)))
        normalized_varcov = normalized_varcov_all[:-1,:-1]

        varcov_all = sigma2Hat * normalized_varcov_all
        varcov = varcov_all[:-1,:-1]

        stderr_all = np.sqrt(np.diag(varcov_all))
        stderr = stderr_all[:-1]
        stderr_betaHat_inverse_mills = stderr_all[-1]


        ## store results
        results = HeckmanResults(self, params, normalized_varcov, sigma2Hat,
            select_res=step1res,
            param_inverse_mills=betaHat_inverse_mills, stderr_inverse_mills=stderr_betaHat_inverse_mills,
            var_reg_error=sigma2Hat, corr_eqnerrors=rhoHat)

        return results


    def _fit_mle(self, Y, X, Z):
        raise ValueError("Invalid choice for estimation method."
            " MLE estimation may be implemented at a later time.")
        return None


class HeckmanResults(base.LikelihoodModelResults):
    """
    Class to represent results/fits for Heckman model.

    Attributes
    ----------
    select_res : ProbitResult object
        The ProbitResult object created when estimating the selection equation.
    param_inverse_mills : scalar
        Parameter estimate of the coef on the inverse Mills term in the second step.
    stderr_inverse_mills : scalar
        Standard error of the parameter estimate of the coef on the inverse Mills
        term in the second step.
    var_reg_error : scalar
        Estimate of the "sigma" term, i.e. the error variance estimate of the
        regression (response) equation
    corr_eqnerrors : scalar
        Estimate of the "rho" term, i.e. the correlation estimate of the errors between the
        regression (response) equation and the selection equation
    """

    #TODO: better to inherit from RegressionResults?

    def __init__(self, model, params, normalized_cov_params=None, scale=1.,
        select_res=None,
        param_inverse_mills=None, stderr_inverse_mills=None,
        var_reg_error=None, corr_eqnerrors=None):

        super(HeckmanResults, self).__init__(model, params,
                                                normalized_cov_params,
                                                scale)

        self.select_res = select_res
        self.param_inverse_mills = param_inverse_mills
        self.stderr_inverse_mills = stderr_inverse_mills
        self.var_reg_error = var_reg_error
        self.corr_eqnerrors = corr_eqnerrors

        if not hasattr(self, 'use_t'):
            self.use_t = False

        if not hasattr(self.select_res, 'use_t'):
            self.select_res.use_t = False


    def summary(self, disp=True, yname=None, xname=None, zname=None, title=None, alpha=.05):
        """Summarize the Heckman model Results

        Parameters
        -----------
        disp  : bool, optional
            Default is True. If True, then results will be printed.
        yname : string, optional
            Default is `y`
        xname : list of strings, optional
            Default is `x_##` for ## in p the number of regressors
            in the regression (response) equation.
        zname : list of strings, optional
            Default is `z_##` for ## in p the number of regressors
            in the selection equation.
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

        ## Put in y,x,zname detected from data if none supplied
        if yname is None:
            yname=self.model.yname

        if xname is None:
            xname=self.model.xname

        if zname is None:
            zname=self.model.zname


        ## create summary object
        # instantiate the object
        smry = summary.Summary()

        # add top info
        top_left = [('Dep. Variable:', None),
                    ('Model:', None),
                    ('Method:', ['Heckman Two-Step']),
                    ('Date:', None),
                    ('Time:', None),
                    ('No. Total Obs.:', ["%#i" % self.model.nobs_total]),
                    ('No. Censored Obs.:', ["%#i" % self.model.nobs_censored]),
                    ('No. Uncensored Obs.:', ["%#i" % self.model.nobs_uncensored]),
                    ]

        if hasattr(self, 'cov_type'):
            top_left.append(('Covariance Type:', [self.cov_type]))

        top_right = [
                     ]

        if title is None:
            title = self.model.__class__.__name__ + ' ' + "Regression Results"

        smry.add_table_2cols(self, gleft=top_left, gright=top_right,
                          yname=yname, xname=xname, title=title)

        # add the Heckman-corrected regression table
        smry.add_table_params(self, yname=yname, xname=xname, alpha=alpha,
                             use_t=self.use_t)

        # add the selection equation estimates table
        smry.add_table_params(self.select_res, yname=yname, xname=zname, alpha=alpha,
                             use_t=self.select_res.use_t)

        # add the estimate to the inverse Mills estimate
        smry.add_table_params(
            base.LikelihoodModelResults(None, np.atleast_1d(self.param_inverse_mills),
            normalized_cov_params=np.atleast_1d(self.stderr_inverse_mills**2), scale=1.),
            yname=None, xname=['IMR (Lambda)'], alpha=alpha,
            use_t=False)  #TODO: return t-score instead of z-score for IMR

        # add point estimates for rho and sigma
        diagn_left = [('rho:', ["%#6.3f" % self.corr_eqnerrors]),
                      ('sigma:', ["%#6.3f" % np.sqrt(self.var_reg_error)]),
                      ]

        diagn_right = [
                       ]

        smry.add_table_2cols(self, gleft=diagn_left, gright=diagn_right,
                          yname=yname, xname=xname,
                          title="")

        # add text at end
        smry.add_extra_txt(['First table are the estimates for the regression (response) equation.',
            'Second table are the estimates for the selection equation.',
            'Third table is the estimate for the coef of the inverse Mills ratio (Heckman\'s Lambda).'])

        ## Print summary if option set to do so
        if(disp):
            print(smry)

        return smry

