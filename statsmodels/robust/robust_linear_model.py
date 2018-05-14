"""
Robust linear models with support for the M-estimators  listed under
:ref:`norms <norms>`.

References
----------
PJ Huber.  'Robust Statistics' John Wiley and Sons, Inc., New York.  1981.

PJ Huber.  1973,  'The 1972 Wald Memorial Lectures: Robust Regression:
    Asymptotics, Conjectures, and Monte Carlo.'  The Annals of Statistics,
    1.5, 799-821.

R Venables, B Ripley. 'Modern Applied Statistics in S'  Springer, New York,
    2002.
"""
from statsmodels.compat.python import string_types
import numpy as np
import scipy.stats as stats

from statsmodels.tools.decorators import (cache_readonly,
                                                  resettable_cache)
import statsmodels.regression.linear_model as lm
import statsmodels.regression._tools as reg_tools
import statsmodels.robust.norms as norms
import statsmodels.robust.scale as scale
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.compat.numpy import np_matrix_rank

__all__ = ['RLM']

def _check_convergence(criterion, iteration, tol, maxiter):
    return not (np.any(np.fabs(criterion[iteration] -
                criterion[iteration-1]) > tol) and iteration < maxiter)

class RLM(base.LikelihoodModel):
    __doc__ = """
    Robust Linear Models

    Estimate a robust linear model via iteratively reweighted least squares
    given a robust criterion estimator.

    %(params)s
    M : statsmodels.robust.norms.RobustNorm, optional
        The robust criterion function for downweighting outliers.
        The current options are LeastSquares, HuberT, RamsayE, AndrewWave,
        TrimmedMean, Hampel, and TukeyBiweight.  The default is HuberT().
        See statsmodels.robust.norms for more information.
    %(extra_params)s

    Notes
    -----

    **Attributes**

    df_model : float
        The degrees of freedom of the model.  The number of regressors p less
        one for the intercept.  Note that the reported model degrees
        of freedom does not count the intercept as a regressor, though
        the model is assumed to have an intercept.
    df_resid : float
        The residual degrees of freedom.  The number of observations n
        less the number of regressors p.  Note that here p does include
        the intercept as using a degree of freedom.
    endog : array
        See above.  Note that endog is a reference to the data so that if
        data is already an array and it is changed, then `endog` changes
        as well.
    exog : array
        See above.  Note that endog is a reference to the data so that if
        data is already an array and it is changed, then `endog` changes
        as well.
    M : statsmodels.robust.norms.RobustNorm
         See above.  Robust estimator instance instantiated.
    nobs : float
        The number of observations n
    pinv_wexog : array
        The pseudoinverse of the design / exogenous data array.  Note that
        RLM has no whiten method, so this is just the pseudo inverse of the
        design.
    normalized_cov_params : array
        The p x p normalized covariance of the design / exogenous data.
        This is approximately equal to (X.T X)^(-1)


    Examples
    ---------
    >>> import statsmodels.api as sm
    >>> data = sm.datasets.stackloss.load()
    >>> data.exog = sm.add_constant(data.exog)
    >>> rlm_model = sm.RLM(data.endog, data.exog, \
                           M=sm.robust.norms.HuberT())

    >>> rlm_results = rlm_model.fit()
    >>> rlm_results.params
    array([  0.82938433,   0.92606597,  -0.12784672, -41.02649835])
    >>> rlm_results.bse
    array([ 0.11100521,  0.30293016,  0.12864961,  9.79189854])
    >>> rlm_results_HC2 = rlm_model.fit(cov="H2")
    >>> rlm_results_HC2.params
    array([  0.82938433,   0.92606597,  -0.12784672, -41.02649835])
    >>> rlm_results_HC2.bse
    array([ 0.11945975,  0.32235497,  0.11796313,  9.08950419])
    >>> mod = sm.RLM(data.endog, data.exog, M=sm.robust.norms.Hampel())
    >>> rlm_hamp_hub = mod.fit(scale_est=sm.robust.scale.HuberScale())
    >>> rlm_hamp_hub.params
    array([  0.73175452,   1.25082038,  -0.14794399, -40.27122257])
    """ % {'params' : base._model_params_doc,
            'extra_params' : base._missing_param_doc}

    def __init__(self, endog, exog, M=norms.HuberT(), missing='none',
                 **kwargs):
        self.M = M
        super(base.LikelihoodModel, self).__init__(endog, exog,
                missing=missing, **kwargs)
        self._initialize()
        #things to remove_data
        self._data_attr.extend(['weights', 'pinv_wexog'])

    def _initialize(self):
        """
        Initializes the model for the IRLS fit.

        Resets the history and number of iterations.
        """
        self.pinv_wexog = np.linalg.pinv(self.exog)
        self.normalized_cov_params = np.dot(self.pinv_wexog,
                                        np.transpose(self.pinv_wexog))
        self.df_resid = (np.float(self.exog.shape[0] -
                         np_matrix_rank(self.exog)))
        self.df_model = np.float(np_matrix_rank(self.exog)-1)
        self.nobs = float(self.endog.shape[0])

    def score(self, params):
        raise NotImplementedError

    def information(self, params):
        raise NotImplementedError

    def predict(self, params, exog=None):
        """
        Return linear predicted values from a design matrix.

        Parameters
        ----------
        params : array-like, optional after fit has been called
            Parameters of a linear model
        exog : array-like, optional.
            Design / exogenous data. Model exog is used if None.

        Returns
        -------
        An array of fitted values

        Notes
        -----
        If the model as not yet been fit, params is not optional.
        """
        #copied from linear_model
        if exog is None:
            exog = self.exog
        return np.dot(exog, params)

    def loglike(self, params):
        raise NotImplementedError

    def deviance(self, tmp_results):
        """
        Returns the (unnormalized) log-likelihood from the M estimator.
        """
        return self.M((self.endog - tmp_results.fittedvalues) /
                          tmp_results.scale).sum()

    def _update_history(self, tmp_results, history, conv):
        history['params'].append(tmp_results.params)
        history['scale'].append(tmp_results.scale)
        if conv == 'dev':
            history['deviance'].append(self.deviance(tmp_results))
        elif conv == 'sresid':
            history['sresid'].append(tmp_results.resid/tmp_results.scale)
        elif conv == 'weights':
            history['weights'].append(tmp_results.model.weights)
        return history

    def _estimate_scale(self, resid):
        """
        Estimates the scale based on the option provided to the fit method.
        """
        if isinstance(self.scale_est, str):
            if self.scale_est.lower() == 'mad':
                return scale.mad(resid, center=0)
            else:
                raise ValueError("Option %s for scale_est not understood" %
                                 self.scale_est)
        elif isinstance(self.scale_est, scale.HuberScale):
            return self.scale_est(self.df_resid, self.nobs, resid)
        else:
            return scale.scale_est(self, resid)**2

    def fit(self, maxiter=50, tol=1e-8, scale_est='mad', init=None, cov='H1',
            update_scale=True, conv='dev'):
        """
        Fits the model using iteratively reweighted least squares.

        The IRLS routine runs until the specified objective converges to `tol`
        or `maxiter` has been reached.

        Parameters
        ----------
        conv : string
            Indicates the convergence criteria.
            Available options are "coefs" (the coefficients), "weights" (the
            weights in the iteration), "sresid" (the standardized residuals),
            and "dev" (the un-normalized log-likelihood for the M
            estimator).  The default is "dev".
        cov : string, optional
            'H1', 'H2', or 'H3'
            Indicates how the covariance matrix is estimated.  Default is 'H1'.
            See rlm.RLMResults for more information.
        init : string
            Specifies method for the initial estimates of the parameters.
            Default is None, which means that the least squares estimate
            is used.  Currently it is the only available choice.
        maxiter : int
            The maximum number of iterations to try. Default is 50.
        scale_est : string or HuberScale()
            'mad' or HuberScale()
            Indicates the estimate to use for scaling the weights in the IRLS.
            The default is 'mad' (median absolute deviation.  Other options are
            'HuberScale' for Huber's proposal 2. Huber's proposal 2 has
            optional keyword arguments d, tol, and maxiter for specifying the
            tuning constant, the convergence tolerance, and the maximum number
            of iterations. See statsmodels.robust.scale for more information.
        tol : float
            The convergence tolerance of the estimate.  Default is 1e-8.
        update_scale : Bool
            If `update_scale` is False then the scale estimate for the
            weights is held constant over the iteration.  Otherwise, it
            is updated for each fit in the iteration.  Default is True.

        Returns
        -------
        results : object
            statsmodels.rlm.RLMresults
        """
        if not cov.upper() in ["H1","H2","H3"]:
            raise ValueError("Covariance matrix %s not understood" % cov)
        else:
            self.cov = cov.upper()
        conv = conv.lower()
        if not conv in ["weights","coefs","dev","sresid"]:
            raise ValueError("Convergence argument %s not understood" \
                % conv)
        self.scale_est = scale_est

        wls_results = lm.WLS(self.endog, self.exog).fit()
        if not init:
            self.scale = self._estimate_scale(wls_results.resid)

        history = dict(params = [np.inf], scale = [])
        if conv == 'coefs':
            criterion = history['params']
        elif conv == 'dev':
            history.update(dict(deviance = [np.inf]))
            criterion = history['deviance']
        elif conv == 'sresid':
            history.update(dict(sresid = [np.inf]))
            criterion = history['sresid']
        elif conv == 'weights':
            history.update(dict(weights = [np.inf]))
            criterion = history['weights']

        # done one iteration so update
        history = self._update_history(wls_results, history, conv)
        iteration = 1
        converged = 0
        while not converged:
            self.weights = self.M.weights(wls_results.resid/self.scale)
            wls_results = reg_tools._MinimalWLS(self.endog, self.exog,
                                                weights=self.weights).fit()
            if update_scale is True:
                self.scale = self._estimate_scale(wls_results.resid)
            history = self._update_history(wls_results, history, conv)
            iteration += 1
            converged = _check_convergence(criterion, iteration, tol, maxiter)
        results = RLMResults(self, wls_results.params,
                            self.normalized_cov_params, self.scale)

        history['iteration'] = iteration
        results.fit_history = history
        results.fit_options = dict(cov=cov.upper(), scale_est=scale_est,
                                   norm=self.M.__class__.__name__, conv=conv)
        #norm is not changed in fit, no old state

        #doing the next causes exception
        #self.cov = self.scale_est = None #reset for additional fits
        #iteration and history could contain wrong state with repeated fit
        return RLMResultsWrapper(results)

class RLMResults(base.LikelihoodModelResults):
    """
    Class to contain RLM results

    Returns
    -------
    **Attributes**

    bcov_scaled : array
        p x p scaled covariance matrix specified in the model fit method.
        The default is H1. H1 is defined as
        ``k**2 * (1/df_resid*sum(M.psi(sresid)**2)*scale**2)/
        ((1/nobs*sum(M.psi_deriv(sresid)))**2) * (X.T X)^(-1)``

        where ``k = 1 + (df_model +1)/nobs * var_psiprime/m**2``
        where ``m = mean(M.psi_deriv(sresid))`` and
        ``var_psiprime = var(M.psi_deriv(sresid))``

        H2 is defined as
        ``k * (1/df_resid) * sum(M.psi(sresid)**2) *scale**2/
        ((1/nobs)*sum(M.psi_deriv(sresid)))*W_inv``

        H3 is defined as
        ``1/k * (1/df_resid * sum(M.psi(sresid)**2)*scale**2 *
        (W_inv X.T X W_inv))``

        where `k` is defined as above and
        ``W_inv = (M.psi_deriv(sresid) exog.T exog)^(-1)``

        See the technical documentation for cleaner formulae.
    bcov_unscaled : array
        The usual p x p covariance matrix with scale set equal to 1.  It
        is then just equivalent to normalized_cov_params.
    bse : array
        An array of the standard errors of the parameters.  The standard
        errors are taken from the robust covariance matrix specified in the
        argument to fit.
    chisq : array
        An array of the chi-squared values of the paramter estimates.
    df_model
        See RLM.df_model
    df_resid
        See RLM.df_resid
    fit_history : dict
        Contains information about the iterations. Its keys are `deviance`,
        `params`, `iteration` and the convergence criteria specified in
        `RLM.fit`, if different from `deviance` or `params`.
    fit_options : dict
        Contains the options given to fit.
    fittedvalues : array
        The linear predicted values.  dot(exog, params)
    model : statsmodels.rlm.RLM
        A reference to the model instance
    nobs : float
        The number of observations n
    normalized_cov_params : array
        See RLM.normalized_cov_params
    params : array
        The coefficients of the fitted model
    pinv_wexog : array
        See RLM.pinv_wexog
    pvalues : array
        The p values associated with `tvalues`. Note that `tvalues` are assumed to be distributed
        standard normal rather than Student's t.
    resid : array
        The residuals of the fitted model.  endog - fittedvalues
    scale : float
        The type of scale is determined in the arguments to the fit method in
        RLM.  The reported scale is taken from the residuals of the weighted
        least squares in the last IRLS iteration if update_scale is True.  If
        update_scale is False, then it is the scale given by the first OLS
        fit before the IRLS iterations.
    sresid : array
        The scaled residuals.
    tvalues : array
        The "t-statistics" of params. These are defined as params/bse where bse are taken
        from the robust covariance matrix specified in the argument to fit.
    weights : array
        The reported weights are determined by passing the scaled residuals
        from the last weighted least squares fit in the IRLS algortihm.

    See also
    --------
    statsmodels.base.model.LikelihoodModelResults
    """


    def __init__(self, model, params, normalized_cov_params, scale):
        super(RLMResults, self).__init__(model, params,
                normalized_cov_params, scale)
        self.model = model
        self.df_model = model.df_model
        self.df_resid = model.df_resid
        self.nobs = model.nobs
        self._cache = resettable_cache()
        #for remove_data
        self.data_in_cache = ['sresid']

        self.cov_params_default = self.bcov_scaled
        #TODO: "pvals" should come from chisq on bse?

    @cache_readonly
    def fittedvalues(self):
        return np.dot(self.model.exog, self.params)

    @cache_readonly
    def resid(self):
        return self.model.endog - self.fittedvalues   # before bcov

    @cache_readonly
    def sresid(self):
        return self.resid/self.scale

    @cache_readonly
    def bcov_unscaled(self):
        return self.normalized_cov_params

    @cache_readonly
    def weights(self):
        return self.model.weights

    @cache_readonly
    def bcov_scaled(self):
        model = self.model
        m = np.mean(model.M.psi_deriv(self.sresid))
        var_psiprime = np.var(model.M.psi_deriv(self.sresid))
        k = 1 + (self.df_model+1)/self.nobs * var_psiprime/m**2

        if model.cov == "H1":
            return k**2 * (1/self.df_resid*\
                np.sum(model.M.psi(self.sresid)**2)*self.scale**2)\
                /((1/self.nobs*np.sum(model.M.psi_deriv(self.sresid)))**2)\
                *model.normalized_cov_params
        else:
            W = np.dot(model.M.psi_deriv(self.sresid)*model.exog.T,
                    model.exog)
            W_inv = np.linalg.inv(W)
            # [W_jk]^-1 = [SUM(psi_deriv(Sr_i)*x_ij*x_jk)]^-1
            # where Sr are the standardized residuals
            if model.cov == "H2":
            # These are correct, based on Huber (1973) 8.13
                return k*(1/self.df_resid)*np.sum(\
                    model.M.psi(self.sresid)**2)*self.scale**2\
                    /((1/self.nobs)*np.sum(\
                    model.M.psi_deriv(self.sresid)))*W_inv
            elif model.cov == "H3":
                return k**-1*1/self.df_resid*np.sum(\
                    model.M.psi(self.sresid)**2)*self.scale**2\
                    *np.dot(np.dot(W_inv, np.dot(model.exog.T,model.exog)),\
                    W_inv)

    @cache_readonly
    def pvalues(self):
        return stats.norm.sf(np.abs(self.tvalues))*2

    @cache_readonly
    def bse(self):
        return np.sqrt(np.diag(self.bcov_scaled))

    @cache_readonly
    def chisq(self):
        return (self.params/self.bse)**2

    def remove_data(self):
        super(self.__class__, self).remove_data()
        #self.model.history['sresid'] = None
        #self.model.history['weights'] = None

    remove_data.__doc__ = base.LikelihoodModelResults.remove_data.__doc__

    def summary(self, yname=None, xname=None, title=0, alpha=.05,
                return_fmt='text'):
        """
        This is for testing the new summary setup
        """
        from statsmodels.iolib.summary import (summary_top,
                                            summary_params, summary_return)

##        left = [(i, None) for i in (
##                        'Dependent Variable:',
##                        'Model type:',
##                        'Method:',
##			'Date:',
##                        'Time:',
##                        'Number of Obs:',
##                        'df resid',
##		        'df model',
##                         )]
        top_left = [('Dep. Variable:', None),
                    ('Model:', None),
                    ('Method:', ['IRLS']),
                    ('Norm:', [self.fit_options['norm']]),
                    ('Scale Est.:', [self.fit_options['scale_est']]),
                    ('Cov Type:', [self.fit_options['cov']]),
                    ('Date:', None),
                    ('Time:', None),
                    ('No. Iterations:', ["%d" % self.fit_history['iteration']])
                    ]
        top_right = [('No. Observations:', None),
                     ('Df Residuals:', None),
                     ('Df Model:', None)
                     ]

        if not title is None:
            title = "Robust linear Model Regression Results"

        #boiler plate
        from statsmodels.iolib.summary import Summary
        smry = Summary()
        smry.add_table_2cols(self, gleft=top_left, gright=top_right, #[],
                          yname=yname, xname=xname, title=title)
        smry.add_table_params(self, yname=yname, xname=xname, alpha=alpha,
                             use_t=self.use_t)

        #diagnostic table is not used yet
#        smry.add_table_2cols(self, gleft=diagn_left, gright=diagn_right,
#                          yname=yname, xname=xname,
#                          title="")

#add warnings/notes, added to text format only
        etext =[]
        wstr = \
'''If the model instance has been used for another fit with different fit
parameters, then the fit options might not be the correct ones anymore .'''
        etext.append(wstr)

        if etext:
            smry.add_extra_txt(etext)

        return smry


    def summary2(self, xname=None, yname=None, title=None, alpha=.05,
                float_format="%.4f"):
        """Experimental summary function for regression results

        Parameters
        -----------
        xname : List of strings of length equal to the number of parameters
            Names of the independent variables (optional)
        yname : string
            Name of the dependent variable (optional)
        title : string, optional
            Title for the top table. If not None, then this replaces the
            default title
        alpha : float
            significance level for the confidence intervals
        float_format: string
            print format for floats in parameters summary

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
        # Summary
        from statsmodels.iolib import summary2
        smry = summary2.Summary()
        smry.add_base(results=self, alpha=alpha, float_format=float_format,
                xname=xname, yname=yname, title=title)

        return smry


class RLMResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(RLMResultsWrapper, RLMResults)

if __name__=="__main__":
#NOTE: This is to be removed
#Delivery Time Data is taken from Montgomery and Peck
    import statsmodels.api as sm

#delivery time(minutes)
    endog = np.array([16.68, 11.50, 12.03, 14.88, 13.75, 18.11, 8.00, 17.83,
    79.24, 21.50, 40.33, 21.00, 13.50, 19.75, 24.00, 29.00, 15.35, 19.00,
    9.50, 35.10, 17.90, 52.32, 18.75, 19.83, 10.75])

#number of cases, distance (Feet)
    exog = np.array([[7, 3, 3, 4, 6, 7, 2, 7, 30, 5, 16, 10, 4, 6, 9, 10, 6,
    7, 3, 17, 10, 26, 9, 8, 4], [560, 220, 340, 80, 150, 330, 110, 210, 1460,
    605, 688, 215, 255, 462, 448, 776, 200, 132, 36, 770, 140, 810, 450, 635,
    150]])
    exog = exog.T
    exog = sm.add_constant(exog)

#    model_ols = models.regression.OLS(endog, exog)
#    results_ols = model_ols.fit()

#    model_ramsaysE = RLM(endog, exog, M=norms.RamsayE())
#    results_ramsaysE = model_ramsaysE.fit(update_scale=False)

#    model_andrewWave = RLM(endog, exog, M=norms.AndrewWave())
#    results_andrewWave = model_andrewWave.fit(update_scale=False)

#    model_hampel = RLM(endog, exog, M=norms.Hampel(a=1.7,b=3.4,c=8.5)) # convergence problems with scale changed, not with 2,4,8 though?
#    results_hampel = model_hampel.fit(update_scale=False)

#######################
### Stack Loss Data ###
#######################
    from statsmodels.datasets.stackloss import load
    data = load()
    data.exog = sm.add_constant(data.exog)
#############
### Huber ###
#############
#    m1_Huber = RLM(data.endog, data.exog, M=norms.HuberT())
#    results_Huber1 = m1_Huber.fit()
#    m2_Huber = RLM(data.endog, data.exog, M=norms.HuberT())
#    results_Huber2 = m2_Huber.fit(cov="H2")
#    m3_Huber = RLM(data.endog, data.exog, M=norms.HuberT())
#    results_Huber3 = m3_Huber.fit(cov="H3")
##############
### Hampel ###
##############
#    m1_Hampel = RLM(data.endog, data.exog, M=norms.Hampel())
#    results_Hampel1 = m1_Hampel.fit()
#    m2_Hampel = RLM(data.endog, data.exog, M=norms.Hampel())
#    results_Hampel2 = m2_Hampel.fit(cov="H2")
#    m3_Hampel = RLM(data.endog, data.exog, M=norms.Hampel())
#    results_Hampel3 = m3_Hampel.fit(cov="H3")
################
### Bisquare ###
################
#    m1_Bisquare = RLM(data.endog, data.exog, M=norms.TukeyBiweight())
#    results_Bisquare1 = m1_Bisquare.fit()
#    m2_Bisquare = RLM(data.endog, data.exog, M=norms.TukeyBiweight())
#    results_Bisquare2 = m2_Bisquare.fit(cov="H2")
#    m3_Bisquare = RLM(data.endog, data.exog, M=norms.TukeyBiweight())
#    results_Bisquare3 = m3_Bisquare.fit(cov="H3")


##############################################
# Huber's Proposal 2 scaling                 #
##############################################

################
### Huber'sT ###
################
    m1_Huber_H = RLM(data.endog, data.exog, M=norms.HuberT())
    results_Huber1_H = m1_Huber_H.fit(scale_est=scale.HuberScale())
#    m2_Huber_H
#    m3_Huber_H
#    m4 = RLM(data.endog, data.exog, M=norms.HuberT())
#    results4 = m1.fit(scale_est="Huber")
#    m5 = RLM(data.endog, data.exog, M=norms.Hampel())
#    results5 = m2.fit(scale_est="Huber")
#    m6 = RLM(data.endog, data.exog, M=norms.TukeyBiweight())
#    results6 = m3.fit(scale_est="Huber")




#    print """Least squares fit
#%s
#Huber Params, t = 2.
#%s
#Ramsay's E Params
#%s
#Andrew's Wave Params
#%s
#Hampel's 17A Function
#%s
#""" % (results_ols.params, results_huber.params, results_ramsaysE.params,
#            results_andrewWave.params, results_hampel.params)

