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
import numpy as np
from scikits.statsmodels.tools.tools import rank
from scikits.statsmodels.regression.linear_model import WLS, GLS
import norms
import scale
from scikits.statsmodels.base.model import (LikelihoodModel,
        LikelihoodModelResults)
from scikits.statsmodels.tools.decorators import (cache_readonly,
        resettable_cache)
from scipy.stats import norm

__all__ = ['RLM']

class RLM(LikelihoodModel):
    """
    Robust Linear Models

    Estimate a robust linear model via iteratively reweighted least squares
    given a robust criterion estimator.

    Parameters
    ----------
    endog : array-like
        1d endogenous response variable
    exog : array-like
        n x p exogenous design matrix
    M : scikits.statsmodels.robust.norms.RobustNorm, optional
        The robust criterion function for downweighting outliers.
        The current options are LeastSquares, HuberT, RamsayE, AndrewWave,
        TrimmedMean, Hampel, and TukeyBiweight.  The default is HuberT().
        See scikits.statsmodels.robust.norms for more information.

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
    history : dict
        Contains information about the iterations. Its keys are `fittedvalues`,
        `deviance`, and `params`.
    M : scikits.statsmodels.robust.norms.RobustNorm
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
    >>> import scikits.statsmodels.api as sm
    >>> data = sm.datasets.stackloss.load()
    >>> data.exog = sm.add_constant(data.exog)
    >>> rlm_model = sm.RLM(data.endog, data.exog,
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
    >>>
    >>> rlm_hamp_hub = sm.RLM(data.endog, data.exog,
                          M=sm.robust.norms.Hampel()).fit(
                          sm.robust.scale.HuberScale())

    >>> rlm_hamp_hub.params
    array([  0.73175452,   1.25082038,  -0.14794399, -40.27122257])

    """

    def __init__(self, endog, exog, M=norms.HuberT()):
        self.M = M
        self.endog = np.asarray(endog)
        self.exog = np.asarray(exog)
        self._initialize()

    def _initialize(self):
        """
        Initializes the model for the IRLS fit.

        Resets the history and number of iterations.
        """
        self.history = {'deviance' : [np.inf], 'params' : [np.inf],
            'weights' : [np.inf], 'sresid' : [np.inf], 'scale' : []}
        self.iteration = 0
        self.pinv_wexog = np.linalg.pinv(self.exog)
        self.normalized_cov_params = np.dot(self.pinv_wexog,
                                        np.transpose(self.pinv_wexog))
        self.df_resid = np.float(self.exog.shape[0] - rank(self.exog))
        self.df_model = np.float(rank(self.exog)-1)
        self.nobs = float(self.endog.shape[0])

    def score(self, params):
        raise NotImplementedError

    def information(self, params):
        raise NotImplementedError

    def loglike(self, params):
        raise NotImplementedError

    def deviance(self, tmp_results):
        """
        Returns the (unnormalized) log-likelihood from the M estimator.
        """
        return self.M((self.endog - tmp_results.fittedvalues)/\
                    tmp_results.scale).sum()

    def _update_history(self, tmp_results):
        self.history['deviance'].append(self.deviance(tmp_results))
        self.history['params'].append(tmp_results.params)
        self.history['scale'].append(tmp_results.scale)
        self.history['sresid'].append(tmp_results.resid/tmp_results.scale)
        self.history['weights'].append(tmp_results.model.weights)

    def _estimate_scale(self, resid):
        """
        Estimates the scale based on the option provided to the fit method.
        """
        if isinstance(self.scale_est, str):
            if self.scale_est.lower() == 'mad':
                return scale.mad(resid)
            if self.scale_est.lower() == 'stand_mad':
                return scale.stand_mad(resid)
        elif isinstance(self.scale_est, scale.HuberScale):
            return scale.hubers_scale(self.df_resid, self.nobs, resid)
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
            weights in the iteration), "resids" (the standardized residuals),
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
            'mad', 'stand_mad', or HuberScale()
            Indicates the estimate to use for scaling the weights in the IRLS.
            The default is 'mad' (median absolute deviation.  Other options are
            use 'stand_mad' for the median absolute deviation standardized
            around the median and 'HuberScale' for Huber's proposal 2.
            Huber's proposal 2 has optional keyword arguments d, tol, and
            maxiter for specifying the tuning constant, the convergence
            tolerance, and the maximum number of iterations.
            See models.robust.scale for more information.
        tol : float
            The convergence tolerance of the estimate.  Default is 1e-8.
        update_scale : Bool
            If `update_scale` is False then the scale estimate for the
            weights is held constant over the iteration.  Otherwise, it
            is updated for each fit in the iteration.  Default is True.

        Returns
        -------
        results : object
            scikits.statsmodels.rlm.RLMresults
        """
        if not cov.upper() in ["H1","H2","H3"]:
            raise ValueError("Covariance matrix %s not understood" % cov)
        else:
            self.cov = cov.upper()
        conv = conv.lower()
        if not conv in ["weights","coefs","dev","resid"]:
            raise ValueError("Convergence argument %s not understood" \
                % conv)
        self.scale_est = scale_est
        wls_results = WLS(self.endog, self.exog).fit()
        if not init:
            self.scale = self._estimate_scale(wls_results.resid)
        self._update_history(wls_results)
        self.iteration = 1
        if conv == 'coefs':
            criterion = self.history['params']
        elif conv == 'dev':
            criterion = self.history['deviance']
        elif conv == 'resid':
            criterion = self.history['sresid']
        elif conv == 'weights':
            criterion = self.history['weights']
        while (np.all(np.fabs(criterion[self.iteration]-\
                criterion[self.iteration-1]) > tol) and \
                self.iteration < maxiter):
#            self.weights = self.M.weights((self.endog - \
#                    wls_results.fittedvalues)/self.scale)
            self.weights = self.M.weights(wls_results.resid/self.scale)
            wls_results = WLS(self.endog, self.exog,
                                    weights=self.weights).fit()
            if update_scale is True:
                self.scale = self._estimate_scale(wls_results.resid)
            self._update_history(wls_results)
            self.iteration += 1
        results = RLMResults(self, wls_results.params,
                            self.normalized_cov_params, self.scale)
        return results

class RLMResults(LikelihoodModelResults):
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
    fittedvalues : array
        The linear predicted values.  dot(exog, params)
    model : scikits.statsmodels.rlm.RLM
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
    scikits.statsmodels.model.LikelihoodModelResults
    """


    def __init__(self, model, params, normalized_cov_params, scale):
        super(RLMResults, self).__init__(model, params,
                normalized_cov_params, scale)
        self.model = model
        self.df_model = model.df_model
        self.df_resid = model.df_resid
        self.nobs = model.nobs
        self._cache = resettable_cache()

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
        return self.cov_params(scale=1.)

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

    def t(self):
        """
        Deprecated method to return t-values. Use tvalues attribute instead.
        """
        import warnings
        warnings.warn("t will be removed in the next release. Use attribute "
                "tvalues instead", FutureWarning)
        return self.tvalues

    @cache_readonly
    def pvalues(self):
        return norm.sf(np.abs(self.tvalues))*2

    @cache_readonly
    def bse(self):
        return np.sqrt(np.diag(self.bcov_scaled))

    @cache_readonly
    def chisq(self):
        return (self.params/self.bse)**2

    def summary(self, yname=None, xnames=None, title=0, alpha=.05,
                returns='print'):
        """
        This is for testing the new summary setup
        """
        from scikits.statsmodels.iolib.summary import summary as smry
        return smry(self, yname=yname, xname=xnames, title=0, alpha=.05, returns='print')


if __name__=="__main__":
#NOTE: This is to be removed
#Delivery Time Data is taken from Montgomery and Peck
    import scikits.statsmodels.api as sm

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

#    model_huber = RLM(endog, exog, M=norms.HuberT(t=2.))
#    results_huber = model_huber.fit(scale_est="stand_mad", update_scale=False)

#    model_ramsaysE = RLM(endog, exog, M=norms.RamsayE())
#    results_ramsaysE = model_ramsaysE.fit(update_scale=False)

#    model_andrewWave = RLM(endog, exog, M=norms.AndrewWave())
#    results_andrewWave = model_andrewWave.fit(update_scale=False)

#    model_hampel = RLM(endog, exog, M=norms.Hampel(a=1.7,b=3.4,c=8.5)) # convergence problems with scale changed, not with 2,4,8 though?
#    results_hampel = model_hampel.fit(update_scale=False)

#######################
### Stack Loss Data ###
#######################
    from scikits.statsmodels.datasets.stackloss import load
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

