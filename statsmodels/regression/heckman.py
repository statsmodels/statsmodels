"""
Heckman correction for sample selection bias (the Heckit procedure).

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
import pdb

class Heckman(base.LikelihoodModel):
    """
    Class for Heckman correction for sample selection bias model.

    Attributes
    ----------
    endog : 1darray
        Data for the dependent variable. Should be set to np.nan for
        censored observations.
    exog : 2darray
        Data for the regression (response) equation. If a constant
        term is desired, the user should directly add the constant
        column to the data before using it as an argument here.
    exog_select : 2darray
        Data for the selection equation. If a constant
        term is desired, the user should directly add the constant
        column to the data before using it as an argument here.
    **kwargs:
        missing=

    See Also
    --------
    The selection equation should contain at least one variable that
    is not in the regression (response) equation, i.e. the selection
    equation should contain at least one instrument. However, if the
    user chooses not to do this, this module will still go ahead and
    estimate the Heckman correction.
    """

    def __init__(self, endog, exog, exog_select, **kwargs):
        #TODO: make sure that selection equation contains at least one more unique var than in reg eqn

        # check that Z has same index as X (and consequently Y through super().__init__)
        try:
            if not all(endog.index==exog_select.index):
                raise ValueError("Z indices need to be the same as X and Y indices")
        except:
            pass

        # shape checks
        if (len(endog) == len(exog)) and (len(endog) == len(exog_select)):
            pass
        else:
            raise ValueError("Y, X, and Z data shapes do not conform with each other.")

        try:
            if (endog.ndim == 1) and (exog.ndim <= 2) and (exog_select.ndim <= 2):
                pass
            else:
                raise ValueError("Y, X, and Z data shapes do not conform with each other.")
        except:
            #TODO: implement dimension check if user inputs data as lists
            pass

        # give missing (treated) values in endog variable finite values so that super().__init__
        # does not strip them out -- they will be put back after the call to super().__init__
        treated = np.asarray(~np.isnan(endog))

        try:
            endog_nomissing = endog.copy()
            endog_nomissing[~treated] = -99999
        except:
            endog_nomissing = [endog[i] if treated[i] else -99999 for i in range(len(treated))]

        # create 1-D array that will be np.nan for every row of exog_select that has any missing
        # values and a finite value otherwise for the call to super().__init__ so that it can
        # strip out rows where exog_select has missing data if missing option is set

        exog_select_1dnan_placeholder = \
            [np.nan if any(np.isnan(row)) else 1 for row in np.asarray(exog_select)]

        try:
            exog_select_1dnan_placeholder = pd.Series(exog_select_1dnan_placeholder, index=endog.index)
        except:
            pass

        # create array of sequential row positions so that rows of exog_select that have missing
        # data can be identified after call to super().__init__
        obsno = list(range(len(endog)))

        # call super().__init__
        super(Heckman, self).__init__(
            endog_nomissing, exog=exog,
            exog_select_1dnan_placeholder=exog_select_1dnan_placeholder, obsno=obsno,
            treated=treated,
            **kwargs)

        # put np.nan back into endog for treated rows
        self.endog = self.data.endog = \
            [self.endog[i] if self.treated[i] else np.nan for i in range(len(self.treated))]

        # strip out rows stripped out by call to super().__init__ in Z variable
        self.exog_select = np.asarray([np.asarray(exog_select)[obs] for obs in self.obsno])

        # store variable names of exog_select
        try:
            self.exog_select_names = exog_select.columns.tolist()
        except:
            self.exog_select_names = None

        # delete attributes created by the call to super().__init__ that are no longer needed
        del self.exog_select_1dnan_placeholder
        del self.obsno


        # store observation counts
        self.nobs_total = len(endog)
        self.nobs_uncensored = self.nobs = np.sum(self.treated)
        self.nobs_censored = self.nobs_total - self.nobs_uncensored


    def initialize(self):
        self.wendog = self.endog
        self.wexog = self.exog


    def whiten(self, data):
        """
        Model whitener for Heckman correction model does nothing.
        """
        return data

    def get_datamats(self):
        Y = np.asarray(self.endog)
        Y = Y[self.treated]

        X = np.asarray(self.exog)
        X = X[self.treated,:]

        Z = np.asarray(self.exog_select)

        return Y, X, Z


    def fit(self, method='twostep'):
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
        Y, X, Z = self.get_datamats()

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
            var_reg_error=sigma2Hat, corr_eqnerrors=rhoHat,
            method='twostep')

        return results


    def _fit_mle(self, Y, X, Z):
        #TODO: implement MLE fitting
        raise ValueError("Invalid choice for estimation method."
            " MLE estimation may be implemented at a later time.")
        return None


    def loglike(self, params_all):
        """
        Log-likelihood of model.

        Parameters
        ----------
        params_all : array-like
            Parameter estimates, with the parameters for the regression
            equation coming first, then the parameters for the
            selection equation, then rho, then sigma2.

        Returns
        -------
        loglike : float
            The value of the log-likelihood function for a Heckman correction model.

        """

        # set up data and parameters needed to compute log likelihood
        Y, X, Z = self.get_datamats()
        D = self.treated

        num_xvars = X.shape[1]
        num_zvars = Z.shape[1]

        xbeta = np.asarray(params_all[:num_xvars])  # reg eqn coefs
        zbeta = np.asarray(params_all[num_xvars:num_xvars+num_zvars])  # selection eqn coefs
        rho = params_all[-2]
        sigma2 = params_all[-1]

        # line the data vectors up
        Z_zbeta_aligned = Z.dot(zbeta)

        X_xbeta = X.dot(xbeta)
        X_xbeta_aligned = np.empty(self.nobs_total)
        X_xbeta_aligned[:] = np.nan
        X_xbeta_aligned[D] = X_xbeta
        del X_xbeta

        Y_aligned = np.empty(self.nobs_total)
        Y_aligned[:] = np.nan
        Y_aligned[D] = Y

        # create an array where each row is the log likelihood contribution of
        # the regression model for the corresponding observation
        ll_contrib_regmod = np.multiply(D,
            np.log(
                (2*np.pi*sigma2)**(-1/2) * \
                    np.exp(
                        -(Y_aligned - X_xbeta_aligned - rho * norm.pdf(Z_zbeta_aligned)/norm.cdf(Z_zbeta_aligned))**2 / (2*sigma2)
                    )
                )
            )
        ll_contrib_regmod[~D] = 0

        # create an array where each row is the log likelihood contribution of
        # the selection model for the corresponding observation
        ll_contrib_selectmod = np.multiply(D, np.log(norm.cdf(Z_zbeta_aligned))) + \
            np.multiply(1-D, np.log(1-norm.cdf(Z_zbeta_aligned)))

        # compute the log likelihood given the data and inputted parameters
        ll = np.sum(ll_contrib_regmod) + np.sum(ll_contrib_selectmod)

        return ll


    def predict(self, params, exog=None):
        """
        Return linear predicted values from a design matrix.

        Parameters
        ----------
        exog : array-like
            Design / exogenous data
        params : array-like, optional after fit has been called
            Parameters of a linear model

        Returns
        -------
        An array of fitted values

        Notes
        -----
        If the model has not yet been fit, params is not optional.
        """
        if exog is None:
            exog = self.exog
        return np.dot(exog, params)

        if self._results is None and params is None:
            raise ValueError("If the model has not been fit, then you must specify the params argument.")
        if self._results is not None:
            return np.dot(exog, self._results.params)
        else:
            return np.dot(exog, params)


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
    method : string
        The method used to produce the estimates, i.e. 'twostep', 'mle'
    """

    #TODO: better to inherit from RegressionResults?

    def __init__(self, model, params, normalized_cov_params=None, scale=1.,
        select_res=None,
        param_inverse_mills=None, stderr_inverse_mills=None,
        var_reg_error=None, corr_eqnerrors=None,
        method=None):

        super(HeckmanResults, self).__init__(model, params,
                                                normalized_cov_params,
                                                scale)

        self.select_res = select_res
        self.param_inverse_mills = param_inverse_mills
        self.stderr_inverse_mills = stderr_inverse_mills
        self.var_reg_error = var_reg_error
        self.corr_eqnerrors = corr_eqnerrors
        self.method = method

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

        ## Put in Z name detected from data if none supplied, unless that too could not be
        ## inferred from data, then put in generic names
        if zname is None and self.model.exog_select_names is not None:
            zname=self.model.exog_select_names
        elif zname is None and self.model.exog_select_names is None:
            try:
                zname = ['z' + str(i) for i in range(len(self.model.exog_select[0]))]
                zname[0]  = 'z0_or_zconst'
            except:
                zname = 'z0_or_zconst'


        ## create summary object
        # instantiate the object
        smry = summary.Summary()

        # add top info
        if self.method=='twostep':
            methodstr = 'Heckman Two-Step'
        elif self.method=='mle':
            methodstr = 'Heckman MLE'
        else:
            raise ValueError("Invalid method set")

        top_left = [('Dep. Variable:', None),
                    ('Model:', None),
                    ('Method:', [methodstr]),
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

