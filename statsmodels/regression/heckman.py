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
    """

    def __init__(self, endog, exog, select_exog):
        #TODO: add code to take care of missing data in X and Z
        #TODO: add type checking and shape checking
        #TODO: make sure that selection equation contains more at least one more unique var than in reg eqn

        self.select_exog = select_exog
        self.treated = np.asarray(~np.isnan(endog))
        super(Heckman, self).__init__(endog, exog)

        self.nobs_total = endog.size
        self.nobs_uncensored = self.nobs = np.sum(self.treated)
        self.nobs_censored = self.nobs_total - self.nobs_uncensored

    def initialize(self):
        self.wendog = self.endog
        self.wexog = self.exog
        self.wselect_exog = self.select_exog

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

        Z = np.asarray(self.select_exog)

        ## fit
        if method=='2step':
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
            fitted = HeckmanResults(self, params, normalized_varcov, sigma2Hat,
                param_inverse_mills=betaHat_inverse_mills, params_select=step1res.params,
                var_reg_error=sigma2Hat, corr_eqnerrors=rhoHat,
                stderr_params=stderr, stderr_inverse_mills=stderr_betaHat_inverse_mills, stderr_params_select=np.sqrt(np.diag(step1_varcov)))

        elif method=='mle':
            raise ValueError("Invalid choice for estimation method. MLE estimation may be implemented at a later time.")
        else:
            raise ValueError("Invalid choice for estimation method.")


        ## return fitted Heckman model
        return fitted




class HeckmanResults(base.LikelihoodModelResults):
    """
    Class to represent results/fits for Heckman model.
    """

    #TODO: better to inherit from RegressionResults?

    def __init__(self, model, params, normalized_cov_params=None, scale=1.,
        param_inverse_mills=None, params_select=None,
        var_reg_error=None, corr_eqnerrors=None,
        stderr_params=None, stderr_inverse_mills=None, stderr_params_select=None):

        super(HeckmanResults, self).__init__(model, params,
                                                normalized_cov_params,
                                                scale)

        self.param_inverse_mills = param_inverse_mills
        self.params_select = params_select
        self.var_reg_error = var_reg_error
        self.corr_eqnerrors = corr_eqnerrors
        self.stderr_params = stderr_params
        self.stderr_inverse_mills = stderr_inverse_mills
        self.stderr_params_select = stderr_params_select


    def summary(self, yname=None, xname=None, zname=None, title=None, alpha=.05):
        """Summarize the Heckman model Results

        Parameters
        -----------
        yname : string, optional
            Default is `y`
        xname : list of strings, optional
            Default is `x_##` for ## in p the number of regressors
            in the regression equation.
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

        ## create summary object
        # instantiate the object
        smry = summary.Summary()

        # add top info
        top_left = [('Dep. Variable:', None),
                    ('Model:', None),
                    ('Method:', ['Heckman 2 Step']),
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
        #TODO

        # add the estimate to the inverse Mills estimate
        #TODO

        # add point estimates for rho and sigma
        diagn_left = [('rho:', ["%#6.3f" % self.corr_eqnerrors]),
                      ('sigma:', ["%#6.3f" % np.sqrt(self.var_reg_error)]),
                      ]

        diagn_right = [
                       ]

        smry.add_table_2cols(self, gleft=diagn_left, gright=diagn_right,
                          yname=yname, xname=xname,
                          title="")

        return smry
