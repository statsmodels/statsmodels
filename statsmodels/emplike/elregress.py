"""
Empirical Likelihood Linear Regression Inference

Included in this script are functions to conduct hypothesis test of linear
regression parameters as well as restrictions.

Start Date: 22 June
Last Updated: 28 June

General References
-----------------

Owen, A.B.(2001). Empirical Likelihood. Chapman and Hall

"""
import numpy as np
from scipy.stats import chi2
from statsmodels.regression.linear_model import OLS
from scipy import optimize
from descriptive import OptFuncts
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import RegressionResults


class _ElRegSetup(OptFuncts):
    """

    Empirical Likelihood is a method of inference, not estimation.  Therefore,
    all of the estimates for EL regression are the same as OLS since both
    methods set the estimating equation X(Y-XB) = 0.  to obtain parameter
    estimates.

    This class fits the data using OLS and initializes the OLS results that are
    the same when using Empirical Likelihood.  Note that estimates are
    different when the regression is forced through the origin.

    Attributes
    ----------
    exog
    nobs
    nvar
    endog
    params
    fittedvalues
    mse_model
    mse_resid
    mse_total
    resid
    rsquared
    rsquared_adj

    See Also
    --------
    OLS documentation
    El_Origin_Regress

    """
    def __init__(self, endog, exog):
        self.exog = exog
        self.nobs = float(self.exog.shape[0])
        self.nvar = float(self.exog.shape[1])
        self.endog = endog.reshape(self.nobs, 1)
        self.ols_fit = OLS(self.endog, self.exog).fit()
        self.params = self.ols_fit.params
        self.fittedvalues = self.ols_fit.fittedvalues
        self.mse_model = self.ols_fit.mse_model
        self.mse_resid = self.ols_fit.mse_resid
        self.mse_total = self.ols_fit.mse_total
        self.resid = self.ols_fit.resid
        self.rsquared = self.ols_fit.rsquared
        self.rsquared_adj = self.ols_fit.rsquared_adj
        #All of the above are the same when using EL or OLS


class _ElRegOpts(_ElRegSetup):
    """

    A class that holds functions to be optimized over when conducting
    hypothesis tests and calculating confidence intervals.

    """
    def __init__(self, endog, exog):
            super(_ElRegOpts, self).__init__(endog, exog)

    def _opt_nuis_regress(self, nuisance_params):
        """
        A function that is optimized over nuisance parameters to conduct a
        hypothesis test for the parameters of interest

        Parameters
        ----------

        params: 1d array
            The regression coefficients of the model.  This includes the
            nuisance and parameters of interests.

        Returns
        -------

        llr: float
            -2 times the log likelihood of the nuisance parameters and the
            hypothesized value of the parameter(s) of interest.

        """
        params = np.copy(self.params)
        params[self.param_nums] = self.b0_vals
        nuis_param_index = np.int_(np.delete(np.arange(self.nvar),
                                             self.param_nums))
        params[nuis_param_index] = nuisance_params
        self.new_params = params.reshape(self.nvar, 1)
        self.est_vect = self.exog * (self.endog - np.dot(self.exog,
                                                         self.new_params))
        if not self._stochastic_exog:
            exog_means = np.mean(self.exog, axis=0)[1:]
            exog_mom2 = (np.sum(self.exog * self.exog, axis=0))[1:]\
                          / self.nobs
            mean_est_vect = self.exog[:, 1:] - exog_means
            mom2_est_vect = (self.exog * self.exog)[:, 1:] - exog_mom2
            regressor_est_vect = np.concatenate((mean_est_vect, mom2_est_vect),
                                                axis=1)
            self.est_vect = np.concatenate((self.est_vect, regressor_est_vect),
                                           axis=1)

        eta_star = self._modif_newton(self.start_lbda)
        self.eta_star = eta_star
        denom = 1. + np.dot(eta_star, self.est_vect.T)
        self.new_weights = 1. / self.nobs * 1. / denom
        llr = np.sum(np.log(self.nobs * self.new_weights))
        return -2 * llr

    def _ci_limits_beta(self, beta):
        self.b0_vals = beta
        return self.hy_test_beta([beta], [self.param_nums],
                                 method=self.method,
                                 start_int_params=self.start_eta,
                                 stochastic_exog=self._stochastic_exog)[1]\
                                  - self.r0

    def _ci_limits_beta_origin(self, beta):
        return self.hy_test_beta_origin(beta, self.param_nums,
                             method=self.method,
                             start_int_params=self.start_eta, only_h0=1,
                             stochastic_exog=self._stochastic_exog) - \
                             self.llr - self.r0


class ElLinReg(_ElRegOpts):

    """
    Class that conducts hyptheses tests and calculates confidence intervals
    for parameters.

    Parameters
    ----------

    endog: nx1 array
        Dependent variable

    exog: nxk array
        X matrix of independent variables.  El_Lin_Reg assumes that there is a
        constant included in X.  For regression through the origin, see
        El_Origin_Regress

    Methods
    -------

    hy_test_beta:
        Conducts hypothesis tests for regression parameters

    ci_veta:
        Finds confidence intervals for regression parameters

    """
    def __init__(self, endog, exog):
        super(ElLinReg, self).__init__(endog, exog)

    def hy_test_beta(self, b0_vals, param_nums, print_weights=0,
                     ret_params=0, method='powell', start_int_params=None,
                     stochastic_exog=1):
        """
        Tests single or joint hypotheses of the regression parameters

        Parameters
        ----------

        b0_vals: list
            The hypthesized value of the parameter to be tested

        param_nums: list
            The parameter number to be tested

        print_weights: bool, optional
            If true, returns the weights that optimize the likelihood
            ratio at b0_vals.  Default is False

        ret_params: bool, optional
            If true, returns the parameter vector that maximizes the likelihood
            ratio at b0_vals.  Also returns the weights.  Default is False

        method: string, optional
            Can either be 'nm' for Nelder-Mead or 'powell' for Powell.  The
            optimization method that optimizes over nuisance parameters.
            Default is 'nm'

        start_int_params: 1d array, optional
            The starting values for the interior minimization.  The starting
            values of lambda as in Owen pg. 63. Default is an array of 0.

        stochastic_exog: bool, optional
            When TRUE, the exogenous variables are assumed to be stochastic.
            When the regressors are nonstochastic, moment conditions are
            placed on the exogenous variables.  Confidence intervals for
            stochastic regressors are at least as large as non-stochastic
            regressors.  Default = TRUE

        Returns
        -------

        res: tuple
            The p-value and -2 times the log likelihood ratio for the
            hypothesized values.

        Examples
        -------_-
        data = sm.datasets.longley.load()
        data.exog = sm.add_constant(data.exog[:,:-1], prepend=1)
        el_analysis = El_Lin_Reg(data.endog, data.exog)
        # Test the hypothesis that the intercept is -6000.
        el_analysis.hy_test_beta([0], [-6000])
        >>> (0.78472652375012586, 0.074619017259285519)
        # Test the hypothesis that the coefficient on the
        #   parameter after the incercept is 0
        el_analysis.hy_test_beta([1], [0])
        >>> (0.2224473814889133, 1.4885126021160364)
        # Test the hypothesis that the second parameter after the intercept
        # is 12000 and the third regression parameter is 50
        el_analysis.hy_test_beta([2, 3], [12000,50])
        >>> (0.0, 105.64623449375982)
        """

        if start_int_params is not None:
            self.start_lbda = start_int_params
        elif stochastic_exog:
            self.start_lbda = np.zeros(self.nvar)
        else:
            self.start_lbda = np.zeros(self.nvar +
                                       2 * (self.exog.shape[1] - 1))
        self._stochastic_exog = stochastic_exog
        self.param_nums = np.asarray(param_nums)
        self.b0_vals = np.asarray(b0_vals)
        if len(param_nums) == len(self.params):
            llr = self._opt_nuis_regress(self.b0_vals)
            pval = 1 - chi2.cdf(llr, len(param_nums))
            return (pval, llr)
        x0 = np.delete(self.params, self.param_nums)
        if method == 'nm':
            llr = optimize.fmin(self._opt_nuis_regress, x0, maxfun=10000,
                                 maxiter=10000, full_output=1)[1]
        if method == 'powell':
            llr = optimize.fmin_powell(self._opt_nuis_regress, x0,
                                 full_output=1)[1]
        pval = 1 - chi2.cdf(llr, len(param_nums))
        if ret_params:   # Used only for origin regress
            return pval, llr, self.new_weights, self.new_params
        elif print_weights:
            return pval, llr, self.new_weights
        else:
            return pval, llr

    def ci_beta(self, param_num, sig=.05, upper_bound=None, lower_bound=None,
                method='powell', start_int_params=None, stochastic_exog=1):
        """

        Computes the confidence interval for the parameter given by param_num

        Parameters
        ---------

        param_num: float
            The parameter thats confidence interval is desired

        sig: float, optional
            The significance level.  Default is .05

        upper_bound: float, optional
            Tha mximum value the upper limit can be.  Default is the
            99.9% confidence value under OLS assumptions.

        lower_bound: float
            The minimum value the lower limit can be.  Default is the 99.9%
            confidence value under OLS assumptions.


        method: string, optional
            Can either be 'nm' for Nelder-Mead or 'powell' for Powell.  The
            optimization method that optimizes over nuisance parameters.
            Default is 'nm'

        start_int_params: 1d array, optional
            The starting values for the interior minimization.  The starting
            values of lambda as in Owen pg. 63. Default is an array of 0.

        Returns
        -------

        ci: tuple
            The confidence interval

        See Also
        --------
        hy_test_beta

        Notes
        -----

        This function uses brentq to find the value of beta where
        hy_test_beta([beta], param_num)[1] is equal to the critical
        value.

        The function returns the results of each iteration of brentq at
        each value of beta.

        The current function value of the last printed optimization
        should be the critical value at the desired significance level.
        For alpha=.05, the value is 3.841459.

        To ensure optimization terminated successfully, it is suggested to
        do hy_test_beta([lower_limit], [param_num])

        If the optimization does not terminate successfully, consider switching
        optimization algorithms.

        If optimization is still not successful, try changing the values of
        start_int_params.  If the current function value repeatedly jumps
        from a number beteween 0 and the critical value and a very large number
        (>50), the starting parameters of the interior minimization need
        to be changed.

        Example
        ------
        data=sm.datasets.stackloss.load()
        data.exog= sm.add_constant(data.exog, prepend=1)
        el_regression = ElLinReg(data.endog, data.exog)
        ci_intercept = el_regression.ci_beta(0)
        ci_intercept
        >>>(-52.771288377249604, -25.21626358895916)
        el_regression.hy_test_beta([ci_intercept[0]], [0])
        >>> (0.049999999999999378, 3.8414588206941378)
        # p-value approx .05 so lower limit is accurate
        el_regression.hy_test_beta([ci_intercept[1]], [0])
        Optimization terminated successfully.
            Current function value: 3.408024
            Iterations: 113
            Function evaluations: 200
        >>>(0.064880108063052777, 3.4080236322591979)
        # P-value is not .05, indicating that the inner optimization
        # got stuck on the boundary of the convex hull. Note function value
        # of 3.40
        ci_intercept_powell = el_regression.ci_beta(0, method='powell')
        ...
        ...
        # The last optimization result:
        Optimization terminated successfully.
            Current function value: 3.841459
            Iterations: 12
            Function evaluations: 484

        # Note the function value of 3.84
        ci_intercept_powell
        >>>(-52.77091422379838, -24.116074241618467)
        # Lower limit is the same with relative error<10**-5
        el_regression.hy_test_beta([ci_intercept_powell[1]], [0],
                                   method='powell')
        >>> (0.04999999999999738, 3.8414588206942133)
        # Note method='powell' in hy_test_beta

        """
        self.start_eta = start_int_params
        self.method = method
        self.r0 = chi2.ppf(1 - sig, 1)
        self.param_nums = param_num
        self._stochastic_exog = stochastic_exog
        if upper_bound is not None:
            beta_high = upper_bound
        else:
            beta_high = self.ols_fit.conf_int(.001)[self.param_nums][1]
        if lower_bound is not None:
            beta_low = lower_bound
        else:
            beta_low = self.ols_fit.conf_int(.001)[self.param_nums][0]
        print 'Finding Lower Limit'
        ll = optimize.brentq(self._ci_limits_beta, beta_low,
                             self.params[self.param_nums])
        print 'Finding Upper Limit'
        ul = optimize.brentq(self._ci_limits_beta,
                             self.params[self.param_nums], beta_high)
        return (ll, ul)


class _ElOriginRegresssSetup(ElLinReg):
    """
    This class is used to find the parameters of a linear regression model
    with a 0 intercept term

    Parameters
    ----------
    endog: nx1 array
        Response Variable

    exog: nxk array
        Exogenous variables.  Assumed to NOT have an array of 1's

    Methods
    -------
    _orig_params:
        Estimates the parameters for regression through the origin
    """
    def __init__(self, endog, exog):
        self.nobs = float(endog.shape[0])
        self.nvar = float(exog.shape[1])
        self.endog = endog
        self.exog = exog

    def _orig_params(self):
        """

        Returns the Empirical Likelihood parameters of a regression model
        that is forced through the origin.

        In regular OLS, the errors do not sum to 0.  However, in EL the maximum
        empirical likelihood estimate of the parameters solves the estimating
        equation:
        X'_1 (y-X*beta) = 0 where X'_1 is a vector of ones.

        See Owen, page 82.

        """

        self.new_exog = add_constant(self.exog, prepend=1)
        self.new_fit = ElLinReg(self.endog, self.new_exog)
        # new fit is what all of the inference is drawn on.  it includes
        # a vector of 1's but the coefficient of the ones is restricted
        # to be 0. self.endog does not contain 1's but self.new_fit.endog does.
        # The user should not be concerned with self.new_fit
        results = self.new_fit.hy_test_beta([0], [0], print_weights=1,
                                      ret_params=1)
        return results


class ElOriginRegress(_ElOriginRegresssSetup):
    """

    Empirical Likelihood inference AND estimation for linear regression
    through the origin.

    Parameters
    ---------

    endog: nx1 array
        Array of response variables

    exog: nxk array
        Array of exogenous variables.  Assumes no array of ones.

    Methods
    -------

    fit:
        Fits the model and adds fitted attributes

    hy_test_beta_origin:
        Conducts a hypothesis test for a regression parameter when the
        regression is forced through the origin.

    ci_beta_origin:
        Computes the confidence interval for a regression parameter when
        the regression is forced through the origin.


    """
    def __init__(self, endog, exog):
        super(ElOriginRegress, self).__init__(endog, exog)

    def fit(self):
        """

        Fits the model and provides regression results.  Results become
        attributes of ElOriginRegress instance.

        Attributes Added
        ----------------
        fittedvalues
        mse_model
        mse_resid
        mse_total
        rsquared
        rsquared_adj

        See Also
        --------

        RegressionResults

        Example
        -------

        data = sm.datasets.stackloss.load()
        el_model = ElOriginRegress(data.endog, data.exog)
        el_model.fit()
            Optimization terminated successfully.
             Current function value: 16.055419
             Iterations: 23
             Function evaluations: 955
        el_model.params
        >>>array([[ 0.        ],
                  [ 0.50630858],
                  [ 1.89804086],
                  [-0.60254578]])
        el_model.rsquared
        >>>0.81907913157749168


        Notes
        -----
        Since EL estimation does not drop the intercept parameter but instead
        estimates the slope parameters conditional on the slope parameter being
        0, the first element for fitted.params will be the intercept
        parameter (0).


        """
        fit_results = self._orig_params()
        self.params = fit_results[3]
        self.llr = fit_results[1]
        ols_model = OLS(self.endog, self.new_exog)
        results = RegressionResults(ols_model,
                                    self.params.reshape(self.nvar + 1,))
        self.fittedvalues = results.fittedvalues
        self.mse_model = results.mse_model
        self.mse_resid = results.mse_resid
        self.mse_total = results.mse_total
        self.rsquared = results.rsquared
        self.rsquared_adj = results.rsquared_adj

    def hy_test_beta_origin(self, value, param_num, method='powell',
                            only_h0=False,
                            stochastic_exog=1,
                            start_int_params=None):
        """

        Returns the pvalue and the llr for a hypothesized parameter value
        for a regression that goes through the origin.

        Note, must call orig_params first.

        Parameters
        ----------

        value: float
            The hypothesized value to be tested

        param_num: float
            Which parameter to test.  Note this uses python
            indexing but the '0' parameter refers to the intercept term.

        Returns
        -------

        res: tuple
            pvalue and likelihood ratio


        """

        llr_beta0 = self.new_fit.hy_test_beta([0, value], [0, param_num],
                                              method=method,
                                         stochastic_exog=stochastic_exog)[1]

        if only_h0:  # Used for confidence intervals
            return llr_beta0
        else:
            llr_beta_hat = self.new_fit.hy_test_beta([0,
                                float(self.params[param_num])],
                                [0, param_num], method=method,
                                start_int_params=start_int_params,
                                stochastic_exog=stochastic_exog)[1]
            llr = llr_beta0 - llr_beta_hat
            pval = 1 - chi2.cdf(llr, 1)
            return pval, llr

    def ci_beta_origin(self, param_num, upper_bound,
                       lower_bound, sig=.05, method='powell',
                       stochastic_exog=1,
                       start_int_params=None):
        """

        Returns the confidence interval for a regression parameter when the
        regression is forced through the origin.

        Parameters
        ----------

        param_num: int
            The parameter number to be tested.  Note this uses python
            indexing but the '0' parameter refers to the intercept term.

        upper_bound: float
            The maximum value the upper confidence limit can be.  The
            closer this is to the confidence limit, the quicker the
            computation.

        lower_bound: float
            The minimum value the lower confidence limit can be.

        sig: float, optional
            The significance level.  Default .05

        method: str, optional
            Algorithm to optimize of nuisance params.  Can be 'nm' or
            'powell'.  Default is 'powell'.

        start_int_params: n x k array, optional
            starting array of parameters that optimize the log star equation.
            See also, ElLinReg.ci_beta.

        Returns
        -------

        CI: tuple
            The confidence interval for the parameter 'param_num'

        See Also
        -------

        ElLinReg.ci_beta for tips and examples of successful
        optimization

        """
        self.start_eta = start_int_params
        self.method = method
        self.r0 = chi2.ppf(1 - sig, 1)
        self.param_nums = param_num
        self._stochastic_exog = stochastic_exog
        beta_high = upper_bound
        beta_low = lower_bound
        print 'Finding Lower Limit'
        ll = optimize.brentq(self._ci_limits_beta_origin, beta_low,
        self.params[self.param_nums])
        print 'Finding Upper Limit'
        ul = optimize.brentq(self._ci_limits_beta_origin,
                             self.params[self.param_nums], beta_high)
        return (ll, ul)
