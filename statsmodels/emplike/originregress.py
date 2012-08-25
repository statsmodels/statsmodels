import numpy as np
from scipy.stats import chi2
from scipy import optimize
from statsmodels.emplike.descriptive2 import _OptFuncts
# When descriptive merged, this will be changed
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS, RegressionResults

class ElOriginRegress(object):
    """

    Empirical Likelihood inference and estimation for linear regression
    through the origin.

    Parameters
    ----------

    endog: nx1 array
        Array of response variables

    exog: nxk array
        Array of exogenous variables.  Assumes no array of ones.

    """
    def __init__(self, endog, exog):
        self.endog = endog
        self.exog = exog
        self.nobs = self.exog.shape[0]
        try:
            self.nvar = exog.shape[1]
        except IndexError:
            self.nvar = 1

    def fit(self):
        """
        Fits the model and provides regression results.

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
                  [ 0.50122897],
                  [ 1.90456428],
                  [-0.60036974]])
        el_model.rsquared
        >>>0.81907913157749168


        Notes
        -----
        Since EL estimation does not drop the intercept parameter but instead
        estimates the slope parameters conditional on the slope parameter being
        0, the first element for fitted.params will be the intercept
        parameter (0).


        """
        exog_with = add_constant(self.exog, prepend =1)
        unrestricted_fit = OLS(self.endog, self.exog).fit()
        restricted_model = OLS(self.endog, exog_with)
        restricted_fit = restricted_model.fit()
        restricted_el = restricted_fit.el_test(
        np.array([0]), np.array([0]), ret_params=1)
        params  = np.squeeze(restricted_el[3])
        beta_hat_llr = restricted_el[0]
        ls_params = np.hstack((0, unrestricted_fit.params))
        ls_llr =  restricted_fit.el_test(ls_params, np.arange(self.nvar+1))[0]
        return OriginResults(restricted_model, params, beta_hat_llr, ls_llr)


    def predict(self, params, exog=None):
        if exog is None:
            exog = self.exog
        return np.dot(add_constant(exog, prepend=1), params)


class OriginResults():
    def __init__(self, model, params, est_llr, ls_llr):
        self.model = model
        self.params = np.squeeze(params)
        self.ls_llr = ls_llr - est_llr
        self.llr = est_llr



    def test_params(self, b0_vals, param_nums, method='nm',
                            stochastic_exog=1, return_weights=0):
        """

        Returns the pvalue and the llr for a hypothesized parameter value
        for a regression that goes through the origin.

        Note, must call orig_params first.

        Parameters
        ----------

        value: float
            The hypothesized value to be tested

        param_num: float
            Which parameters to test.  Note this uses python
            indexing but the '0' parameter refers to the intercept term,
            which is assumed 0.  Therefore, param_num should be > 0.

        Returns
        -------

        res: tuple
            pvalue and likelihood ratio
        """
        b0_vals = np.hstack((0, b0_vals))
        param_nums = np.hstack((0, param_nums))
        test_res = self.model.fit().el_test(b0_vals, param_nums, method=method,
                                  stochastic_exog=stochastic_exog,
                                  return_weights=return_weights)
        llr_test = test_res[0]
        llr_res = llr_test - self.llr
        pval = chi2.sf(llr_res, self.model.exog.shape[1] - 1)
        if return_weights:
            return llr_res, pval, test_res[2]
        else:
            return llr_res, pval


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
        ll = optimize.brentq(self._ci_limits_beta_origin, beta_low,
                             self.params[self.param_nums])
        ul = optimize.brentq(self._ci_limits_beta_origin,
                             self.params[self.param_nums], beta_high)
        return (ll, ul)



