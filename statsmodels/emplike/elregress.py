"""
Empirical Likelihood Linear Regression Inference

Included in this script are functions to conduct hypothesis test of linear
regression parameters as well as restrictions.


General References
-----------------

Owen, A.B.(2001). Empirical Likelihood. Chapman and Hall

"""
import numpy as np
from scipy.stats import chi2
from scipy import optimize
from descriptive2 import _OptFuncts
# When descriptive merged, this will be changed
from statsmodels.tools.tools import add_constant

class _ELRegOpts(_OptFuncts):
    """

    A class that holds functions to be optimized over when conducting
    hypothesis tests and calculating confidence intervals.

    Parameters
    ----------

    OLSResults : Results instance
        A fitted OLS result

    """
    def __init__(self):
        pass

    def _opt_nuis_regress(self, nuisance_params, param_nums=None,
                          endog=None, exog=None,
                          nobs=None, nvar=None, params=None, b0_vals=None,
                          stochastic_exog=None):
        """
        A function that is optimized over nuisance parameters to conduct a
        hypothesis test for the parameters of interest

        Parameters
        ----------
        nuisance_params: 1darray
            Parameters to be optimized over

        Returns
        -------
        llr : float
            -2 x the log-likelihood of the nuisance parameters and the
            hypothesized value of the parameter(s) of interest.
        """
        params[param_nums] = b0_vals
        nuis_param_index = np.int_(np.delete(np.arange(nvar),
                                             param_nums))
        params[nuis_param_index] = nuisance_params
        new_params = params.reshape(nvar, 1)
        est_vect = exog * \
          (endog - np.squeeze(np.dot(exog, new_params))).reshape(nobs, 1)
        if not stochastic_exog:
            exog_means = np.mean(exog, axis=0)[1:]
            exog_mom2 = (np.sum(exog * exog, axis=0))[1:]\
                          / nobs
            mean_est_vect = exog[:, 1:] - exog_means
            mom2_est_vect = (exog * exog)[:, 1:] - exog_mom2
            regressor_est_vect = np.concatenate((mean_est_vect, mom2_est_vect),
                                                axis=1)
            est_vect = np.concatenate((est_vect, regressor_est_vect),
                                           axis=1)

        wts = np.ones(nobs) * (1. / nobs)
        x0 = np.zeros(est_vect.shape[1]).reshape(-1, 1)
        try:
            eta_star = self._modif_newton(x0, est_vect, wts)
            denom = 1. + np.dot(eta_star, est_vect.T)
            self.new_weights = 1. / nobs * 1. / denom
            llr = np.sum(np.log(nobs * self.new_weights))
            return -2 * llr
        except np.linalg.linalg.LinAlgError:
            return np.inf


    def _ci_limits_beta_origin(self, beta):
        """
        Returns the likelihood for a parameter vector give a parameter
        of interest

        Parameters
        ----------
        beta: float
            parameter of interest

        Returns
        ------

        llr: float
            log likelihood ratio
        """
        return self.test_beta_origin([beta], [self.param_nums],
                             method=self.method,
                             start_int_params=self.start_eta, only_h0=1,
                             stochastic_exog=self._stochastic_exog) - \
                             self.llr - self.r0

# class ElOriginRegress(_ElOriginRegresssSetup):
#     """

#     Empirical Likelihood inference AND estimation for linear regression
#     through the origin.

#     Parameters
#     ---------

#     endog: nx1 array
#         Array of response variables

#     exog: nxk array
#         Array of exogenous variables.  Assumes no array of ones.

#     Methods
#     -------

#     fit:
#         Fits the model and adds fitted attributes

#     test_beta_origin:
#         Conducts a hypothesis test for a regression parameter when the
#         regression is forced through the origin.

#     ci_beta_origin:
#         Computes the confidence interval for a regression parameter when
#         the regression is forced through the origin.


#     """
#     def __init__(self, endog, exog):
#         super(ElOriginRegress, self).__init__(endog, exog)

#     def fit(self):
#         """

#         Fits the model and provides regression results.  Results become
#         attributes of ElOriginRegress instance.

#         Attributes Added
#         ----------------
#         fittedvalues
#         mse_model
#         mse_resid
#         mse_total
#         rsquared
#         rsquared_adj

#         See Also
#         --------

#         RegressionResults

#         Example
#         -------

#         data = sm.datasets.stackloss.load()
#         el_model = ElOriginRegress(data.endog, data.exog)
#         el_model.fit()
#             Optimization terminated successfully.
#              Current function value: 16.055419
#              Iterations: 23
#              Function evaluations: 955
#         el_model.params
#         >>>array([[ 0.        ],
#                   [ 0.50122897],
#                   [ 1.90456428],
#                   [-0.60036974]])
#         el_model.rsquared
#         >>>0.81907913157749168


#         Notes
#         -----
#         Since EL estimation does not drop the intercept parameter but instead
#         estimates the slope parameters conditional on the slope parameter being
#         0, the first element for fitted.params will be the intercept
#         parameter (0).


#         """
#         fit_results = self._orig_params()
#         params = fit_results[3]
#         llr = fit_results[0]
#         ols_model = OLS(self.endog, self.new_exog)
#         results = RegressionResults(ols_model,  # Params not same as OLS
#                                     params.reshape(self.nvar + 1,))
#         fitted = _OriginResults(self.endog, self.exog)
#         fitted.llr = llr
#         fitted.params = params
#         fitted.fittedvalues = results.fittedvalues
#         fitted.mse_model = results.mse_model
#         fitted.mse_resid = results.mse_resid
#         fitted.mse_total = results.mse_total
#         fitted.rsquared = results.rsquared
#         fitted.rsquared_adj = results.rsquared_adj
#         fitted._new_fit = self._new_fit
#         return fitted

#     def predict(self, params, exog=None):
#         if exog is None:
#             exog = self.exog
#         return np.dot(add_constant(exog, prepend=1), params)


# class _OriginResults(ElOriginRegress):
#     def __init__(self, endog, exog):
#         super(_OriginResults, self).__init__(endog, exog)

#     def test_beta_origin(self, value, param_num, method='powell',
#                             only_h0=False,
#                             stochastic_exog=1,
#                             start_int_params=None):
#         """

#         Returns the pvalue and the llr for a hypothesized parameter value
#         for a regression that goes through the origin.

#         Note, must call orig_params first.

#         Parameters
#         ----------

#         value: float
#             The hypothesized value to be tested

#         param_num: float
#             Which parameters to test.  Note this uses python
#             indexing but the '0' parameter refers to the intercept term,
#             which is assumed 0.  Therefore, param_num should be > 0.

#         Returns
#         -------

#         res: tuple
#             pvalue and likelihood ratio
#         """

#         value.insert(0, 0)
#         param_num.insert(0, 0)
#         llr_beta0 = self._new_fit.test_beta(value, param_num,
#                                               method=method,
#                                          stochastic_exog=stochastic_exog)[0]

#         self.new_weights = self._new_fit.new_weights
#         if only_h0:
#             return llr_beta0    # Used for confidence intervals
#         else:
#             llr_beta_hat = self._new_fit.test_beta(
#                                 list(self.params[param_num]),
#                                 [param_num], method=method,
#                                 stochastic_exog=stochastic_exog)[0]
#             llr = llr_beta0 - llr_beta_hat
#             pval = 1 - chi2.cdf(llr, len(param_num) - 1)
#             return llr, pval

#     def ci_beta_origin(self, param_num, upper_bound,
#                        lower_bound, sig=.05, method='powell',
#                        stochastic_exog=1,
#                        start_int_params=None):
#         """

#         Returns the confidence interval for a regression parameter when the
#         regression is forced through the origin.

#         Parameters
#         ----------

#         param_num: int
#             The parameter number to be tested.  Note this uses python
#             indexing but the '0' parameter refers to the intercept term.

#         upper_bound: float
#             The maximum value the upper confidence limit can be.  The
#             closer this is to the confidence limit, the quicker the
#             computation.

#         lower_bound: float
#             The minimum value the lower confidence limit can be.

#         sig: float, optional
#             The significance level.  Default .05

#         method: str, optional
#             Algorithm to optimize of nuisance params.  Can be 'nm' or
#             'powell'.  Default is 'powell'.

#         start_int_params: n x k array, optional
#             starting array of parameters that optimize the log star equation.
#             See also, ElLinReg.ci_beta.

#         Returns
#         -------

#         CI: tuple
#             The confidence interval for the parameter 'param_num'

#         See Also
#         -------

#         ElLinReg.ci_beta for tips and examples of successful
#         optimization

#         """
#         self.start_eta = start_int_params
#         self.method = method
#         self.r0 = chi2.ppf(1 - sig, 1)
#         self.param_nums = param_num
#         self._stochastic_exog = stochastic_exog
#         beta_high = upper_bound
#         beta_low = lower_bound
#         ll = optimize.brentq(self._ci_limits_beta_origin, beta_low,
#                              self.params[self.param_nums])
#         ul = optimize.brentq(self._ci_limits_beta_origin,
#                              self.params[self.param_nums], beta_high)
#         return (ll, ul)


# class _ANOVAOpt(_OptFuncts):
#     """

#     Class containing functions that are optimized over when
#     conducting ANOVA

#     """
#     def _opt_common_mu(self, mu):
#         empt_array = np.zeros((self.nobs, self.num_groups))
#         obs_num = 0
#         for arr_num in range(len(self.data)):
#             new_obs_num = obs_num + len(self.data[arr_num])
#             empt_array[obs_num: new_obs_num, arr_num] = self.data[arr_num] - \
#               mu
#             obs_num = new_obs_num
#         est_vect = empt_array
#         wts = np.ones(est_vect.shape[0]) * (1. / (est_vect.shape[0]))
#         eta_star = self._modif_newton(np.zeros(self.num_groups), est_vect, wts)
#         self.eta_star = eta_star
#         denom = 1. + np.dot(eta_star, est_vect.T)
#         self.new_weights = 1. / self.nobs * 1. / denom
#         llr = np.sum(np.log(self.nobs * self.new_weights))
#         return -2 * llr


# class ANOVA(_ANOVAOpt):
#     """

#     A class for ANOVA and comparing means.

#     Parameters
#     ---------

#     data: list of arrays
#         data should be a list containing 1 dimensional arrays.  Each array
#         is the data collected from a certain group.


#     """

#     def __init__(self, data):
#         self.data = data
#         self.num_groups = len(self.data)
#         self.nobs = 0
#         for i in self.data:
#             self.nobs = self.nobs + len(i)

#     def compute_ANOVA(self, mu=None, mu_start=0, print_weights=0):

#         """
#         Returns -2 log likelihood, the pvalue and the maximum likelihood
#         estimate for a common mean.

#         Parameters
#         ----------

#         mu: float, optional
#             If a mu is specified, ANOVA is conducted with mu as the
#             common mean.  Otherwise, the common mean is the maximum
#             empirical likelihood estimate of the common mean.
#             Default is None.

#         mu_start: float, optional
#             Starting value for commean mean if specific mu is not specified.
#             Default = 0

#         print_weights: bool, optional
#             if TRUE, returns the weights on observations that maximize the
#             likelihood.  Default is FALSE

#         Returns
#         -------

#         res: tuple
#             The p-vale, log-likelihood and estimate for the common mean.

#         """

#         if mu is not None:
#             llr = self._opt_common_mu(mu)
#             pval = 1 - chi2.cdf(llr, self.num_groups - 1)
#             if print_weights:
#                 return llr, pval, mu, self.new_weights
#             else:
#                 return llr, pval, mu
#         else:
#             res = optimize.fmin_powell(self._opt_common_mu, mu_start,
#                                        full_output=1)
#             llr = res[1]
#             mu_common = res[0]
#             pval = 1 - chi2.cdf(llr, self.num_groups - 1)
#             if print_weights:
#                 return llr, pval, mu_common, self.new_weights
#             else:
#                 return llr, pval, mu_common


