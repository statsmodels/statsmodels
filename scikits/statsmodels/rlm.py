"""
Robust linear models

References
----------
PJ Huber.  'Robust Statistics' John Wiley and Sons, Inc., New York.  1981.

PJ Huber.  1973,  'The 1972 Wald Memorial Lectures: Robust Regression:
    Asymptotics, Conjectures, and Monte Carlo.'  The Annals of Statistics,
    1.5, 799-821.
"""
import numpy as np
import tools
from regression import WLS, GLS
from robust import norms, scale
from model import LikelihoodModel, LikelihoodModelResults

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

    Attributes
    ----------
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
         See above.  Robust esstimator instance instantiated.
    nobs : float
        The number of observations n
    pinv_wexog : array
        The pseudoinverse of the design / exogenous data array.  Note that
        RLM has no whiten method, so this is just the pseudo inverse of the
        design.
    normalized_cov_params : array
        The p x p normalized covariance of the design / exogenous data.
        This is approximately equal to (X.T X)^(-1)

    Methods
    -------
    deviance
        Returns the (unnormalized) log-likelihood of the model
    fit
        Fits the model.  Returns an RLMResults class.
    information
        Not yet implemented.
    newton
        Not yet implemented.
    results
        A property that returns an RLMResults class.  Equivalent to calling
        fit with the default arguments.
    score
        Not yet implemented.

    Examples
    ---------
    >>> import scikits.statsmodels as models
    >>> from scikits.statsmodels.datasets.stackloss.data import load
    >>> data = load()
    >>> data.exog = models.tools.add_constant(data.exog)
    >>> rlm_model = models.RLM(data.endog, data.exog, \
    ...     M=models.robust.norms.HuberT())
    >>> rlm_results = rlm_model.fit()
    >>> rlm_results.params
    array([  0.82938433,   0.92606597,  -0.12784672, -41.02649835])
    >>> rlm_results.bse
    array([ 0.11100521,  0.30293016,  0.12864961,  9.79189854])
    >>> rlm_results_HC2 = rlm_model.fit(cov="H2")
    >>>rlm_results_HC2.params
    array([  0.82938433,   0.92606597,  -0.12784672, -41.02649835])
    >>> rlm_results_HC2.bse
    array([ 0.11945975,  0.32235497,  0.11796313,  9.08950419])
    >>>
    >>> rlm_hamp_hub = models.RLM(data.endog, data.exog, \
    ...     M=models.robust.norms.Hampel()).fit(scale_est=\
            models.robust.scale.Hubers_scale())
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
        self.df_resid = np.float(self.exog.shape[0] - tools.rank(self.exog))
        self.df_model = np.float(tools.rank(self.exog)-1)
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
                return scale.MAD(resid)
            if self.scale_est.lower() == 'stand_mad':
                return scale.stand_MAD(resid)
        elif isinstance(self.scale_est, scale.Hubers_scale):
            return scale.hubers_scale(self.df_resid, self.nobs, resid)
        else:
            return scale.scale_est(self, resid)**2

    def fit(self, maxiter=50, tol=1e-8, scale_est='MAD', init=None, cov='H1',
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
        scale_est : string or Hubers_scale()
            'MAD', 'stand_MAD', or Hubers_scale()
            Indicates the estimate to use for scaling the weights in the IRLS.
            The default is 'MAD' (median absolute deviation.  Other options are
            use 'stand_MAD' for the median absolute deviation standardized
            around the median and 'Hubers_scale' for Huber's proposal 2.
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
            raise AttributeError, "Covariance matrix %s not understood" % cov
        else:
            self.cov = cov.upper()
        conv = conv.lower()
        if not conv in ["weights","coefs","dev","resid"]:
            raise AttributeError, "Convergence argument %s not understood" \
                % conv
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
            self.weights = self.M.weights((self.endog - \
                    wls_results.fittedvalues)/self.scale)
            wls_results = WLS(self.endog, self.exog,
                                    weights=self.weights).fit()
            if update_scale is True:
                self.scale = self._estimate_scale(wls_results.resid)
            self._update_history(wls_results)
            self.iteration += 1
        self.results = RLMResults(self, wls_results.params,
                            self.normalized_cov_params, self.scale)
        return self.results

class RLMResults(LikelihoodModelResults):
    """
    Class to contain RLM results

    Attributes
    ----------
    bcov_scaled
    bcov_unscaled
    bse
    df_model
    df_resid
    fittedvalues
    model
    nobs
    normalized_cov_params
    params
    pinv_wexog
    resid
    scale
    sresid
    weights

    Methods
    -------
    conf_int
    cov_params
    f_test
    t
    t_test

    Examples

    Notes
    -----
    The three options for robust covariance matrices were proposed by
    Huber (1973, 1981; see References in scikits.statsmodels.rlm)

    If cov is "H1" then the covariance matrix (and the standard errors
    associated with the parameter estimates) is computed as


    """
    def __init__(self, model, params, normalized_cov_params, scale):
        super(RLMResults, self).__init__(model, params,
                normalized_cov_params, scale)
        self._get_results(model)

    def _get_results(self, model):
        #TODO: "pvals" should come from chisq on bse?
        self.df_model = model.df_model
        self.df_resid = model.df_resid
        self.fitted_values = np.dot(model.exog, self.params)
        self.resid = model.endog - self.fitted_values   # before bcov
        self.sresid = self.resid/self.scale
        self.pinv_wexog = model.pinv_wexog    # for bcov,
                                                # this is getting sloppy
        self.bcov_unscaled = self.cov_params(scale=1)
        self.nobs = model.nobs
        self.weights = model.weights
        m = np.mean(model.M.psi_deriv(self.resid/self.scale))
        var_psiprime = np.var(model.M.psi_deriv(self.resid/self.scale))
        k = 1 + (self.df_model+1)/self.nobs * var_psiprime/m**2
#make these properties like for regression?
        if model.cov == "H1":
            self.bcov_scaled = k**2 * (1/self.df_resid*\
                np.sum(model.M.psi(self.sresid)**2)*self.scale**2)\
                /((1/self.nobs*np.sum(model.M.psi_deriv(self.sresid)))**2)\
                *model.normalized_cov_params
        else:
#FIXME: could be optimized to not take the inverse?  Document for now.
            W = np.dot(model.M.psi_deriv(self.sresid)*model.exog.T,model.exog)
            W_inv = np.linalg.inv(W)
# [W_jk]^-1 = [SUM(psi_deriv(Sr_i)*x_ij*x_jk)]^-1
# where Sr are the standardized residuals
            if model.cov == "H2":
# These are correct, based on Huber (1973) 8.13
                self.bcov_scaled = k*(1/self.df_resid)*np.sum(\
                        model.M.psi(self.sresid)**2)*self.scale**2\
                        /((1/self.nobs)*np.sum(\
                        model.M.psi_deriv(self.sresid)))*W_inv
            elif model.cov == "H3":
                self.bcov_scaled = k**-1*1/self.df_resid*np.sum(\
                    model.M.psi(self.sresid)**2)*self.scale**2\
                    *np.dot(np.dot(W_inv, np.dot(model.exog.T,model.exog)),\
                    W_inv)
        self.bse = np.sqrt(np.diag(self.bcov_scaled))

if __name__=="__main__":
#NOTE: This is to be removed
#Delivery Time Data is taken from Montgomery and Peck
    import models

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
    exog = models.tools.add_constant(exog)

#    model_ols = models.regression.OLS(endog, exog)
#    results_ols = model_ols.fit()

#    model_huber = RLM(endog, exog, M=norms.HuberT(t=2.))
#    results_huber = model_huber.fit(scale_est="stand_MAD", update_scale=False)

#    model_ramsaysE = RLM(endog, exog, M=norms.RamsayE())
#    results_ramsaysE = model_ramsaysE.fit(update_scale=False)

#    model_andrewWave = RLM(endog, exog, M=norms.AndrewWave())
#    results_andrewWave = model_andrewWave.fit(update_scale=False)

#    model_hampel = RLM(endog, exog, M=norms.Hampel(a=1.7,b=3.4,c=8.5)) # convergence problems with scale changed, not with 2,4,8 though?
#    results_hampel = model_hampel.fit(update_scale=False)

#######################
### Stack Loss Data ###
#######################
    from models.datasets.stackloss.data import load
    data = load()
    data.exog = models.tools.add_constant(data.exog)
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
    results_Huber1_H = m1_Huber_H.fit(scale_est=scale.Hubers_scale())
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

