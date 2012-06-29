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


class El_Reg_Setup(OptFuncts):
    """

    Empirical Likelihood is a method of inference, not estimation.  Therefore,
    all of the estimates for EL regression are the same as OLS since both
    methods set the estimating equation X(Y-XB) = 0.  to obtain parameter
    estimates.

    This class fits the data using OLS and initializes the OLS results that are
    the same when using EMpirical Likelihood.  Note that estimates are
    different when the regression is forced through the origin.

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


class El_Reg_Opts(El_Reg_Setup):
    """

    A class that holds functions to be optimized over when conducting
    hypothesis tests and calculating confidence intervals.

    """
    def __init__(self, endog, exog):
            super(El_Reg_Opts, self).__init__(endog, exog)

    def _opt_nuis_regress(self, params):
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
        params[self.param_nums] = self.b0_vals
        params = params.reshape(self.nvar, 1)
        self.est_vect = self.exog * (self.endog - np.dot(self.exog, params))
        eta_star = self._modif_newton(np.ones(self.nvar)* 1. / self.nobs)
        denom = 1. + np.dot(eta_star, self.est_vect.T)
        self.new_weights = 1. / self.nobs * 1. / denom
        llr = np.sum(np.log(self.nobs * self.new_weights))
        return -2 * llr


class El_Lin_Reg(El_Reg_Opts):

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

    """
    def __init__(self, endog, exog):
        super(El_Lin_Reg, self).__init__(endog, exog)

    def hy_test_beta(self, param_nums, b0_vals):
        """
        Tests single or joint hypotheses of the regression parameters

        Parameters
        ----------

        param_nums: list
            The parameter number to be tested

        b0_vals: list
            The hypthesized value of the parameter to be tested

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

        self.param_nums = param_nums
        self.b0_vals = b0_vals
        llr = optimize.fmin_powell(self._opt_nuis_regress, self.params,
                                 full_output=1)[1]
        pval = 1 - chi2.cdf(llr, len(param_nums))
        return pval, llr
