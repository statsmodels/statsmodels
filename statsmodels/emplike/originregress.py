"""
This module implements empirical likelihood regression that is forced through
the origin

This is different than regression not forced through the origin because the
maximum empirical likelihood estimate is calculated with a vector of ones in
the exogenous matrix but restricts the intercept parameter to be 0.  This
results in significantly more narrow confidence intervals and different
parameter estimates.

For notes on regression not forced through the origin, see empirical likelihood
methods in the OLSResults class.

References
----------
Owen, A.B. (2001). Empirical Likelihood.  Chapman and Hall. p. 82.

"""

import warnings

import numpy as np
from scipy import optimize
from scipy.stats import chi2

from statsmodels.emplike.descriptive import EmpLikeTestResult
from statsmodels.regression.linear_model import OLS, RegressionResults

# When descriptive merged, this will be changed
from statsmodels.tools.tools import add_constant
from statsmodels.tools.validation import bool_like


class ELOriginRegress:
    """
    Empirical Likelihood inference and estimation for linear regression
    through the origin

    Parameters
    ----------
    endog : array_like
        Array of response variables.
    exog : array_like
        Array of exogenous variables.  Assumes no array of ones

    Attributes
    ----------
    endog : array_like
        Array of response variables
    exog : array_like
        Array of exogenous variables.  Assumes no array of ones.
    nobs : int
        Number of observations.
    nvar : float
        Number of exogenous regressors.
    """
    def __init__(self, endog, exog):
        self.endog = endog
        self.exog = exog
        self.nobs = self.exog.shape[0]
        try:
            self.nvar = float(exog.shape[1])
        except IndexError:
            self.nvar = 1.

    def fit(self):
        """
        Fits the model and provides regression results

        Returns
        -------
        OriginResults
            Empirical likelihood regression results class.
        """
        exog_with = add_constant(self.exog, prepend=True)
        restricted_model = OLS(self.endog, exog_with)
        restricted_fit = restricted_model.fit()
        restricted_el = restricted_fit.el_test(
            np.array([0]), np.array([0]), ret_params=1, result_object=False
        )
        params = np.squeeze(restricted_el[3])
        beta_hat_llr = restricted_el[0]
        llf = np.sum(np.log(restricted_el[2]))
        return OriginResults(restricted_model, params, beta_hat_llr, llf)

    def predict(self, params, exog=None):
        """
        Return fitted values for a regression through the origin

        Parameters
        ----------
        params : ndarray
            Parameters, including the (fixed at 0) intercept term,
            as returned by `fit`.
        exog : array_like, optional
            Exogenous variables to use for the prediction.  If None,
            the exog attached to the model is used.

        Returns
        -------
        ndarray
            The predicted values, exog @ params.
        """
        if exog is None:
            exog = self.exog
        return np.dot(add_constant(exog, prepend=True), params)


class OriginResults(RegressionResults):
    """
    A Results class for empirical likelihood regression through the origin

    Parameters
    ----------
    model : OLS
        An OLS model with an intercept.
    params : 1darray
        Fitted parameters.
    est_llr : float
        The log likelihood ratio of the model with the intercept restricted to
        0 at the maximum likelihood estimates of the parameters.
        llr_restricted/llr_unrestricted
    llf_el : float
        The log likelihood of the fitted model with the intercept restricted to 0.

    Attributes
    ----------
    model : OLS
        An OLS model with an intercept.
    params : 1darray
        Fitted parameter.
    llr : float
        The log likelihood ratio of the maximum empirical likelihood estimate.
    llf_el : float
        The log likelihood of the fitted model with the intercept restricted to 0.

    Notes
    -----
    IMPORTANT.  Since EL estimation does not drop the intercept parameter but
    instead estimates the slope parameters conditional on the slope parameter
    being 0, the first element for params will be the intercept, which is
    restricted to 0.

    IMPORTANT.  This class inherits from RegressionResults but inference is
    conducted via empirical likelihood.  Therefore, any methods that
    require an estimate of the covariance matrix will not function.  Instead
    use el_test and conf_int_el to conduct inference.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> data = sm.datasets.bc.load()
    >>> model = sm.emplike.ELOriginRegress(data.endog, data.exog)
    >>> fitted = model.fit()
    >>> fitted.params #  0 is the intercept term.
    array([ 0.        ,  0.00351813])

    >>> fitted.el_test(np.array([.0034]), np.array([1]))
    (3.6696503297979302, 0.055411808127497755)
    >>> fitted.conf_int_el(1)
    (0.0033971871114706867, 0.0036373150174892847)

    # No covariance matrix so normal inference is not valid
    >>> fitted.conf_int()
    Traceback (most recent call last):
     ...
    TypeError: unsupported operand type(s) for *: 'instancemethod' and 'float'
    """
    def __init__(self, model, params, est_llr, llf_el):
        self.model = model
        self.params = np.squeeze(params)
        self.llr = est_llr
        self.llf_el = llf_el

    def el_test(
        self,
        b0_vals,
        param_nums,
        method="nm",
        stochastic_exog=1,
        return_weights=0,
        *,
        result_object=None,
    ):
        """
        Returns the llr and p-value for a hypothesized parameter value
        for a regression that goes through the origin

        Parameters
        ----------
        b0_vals : 1darray
            The hypothesized value to be tested.
        param_nums : 1darray
            Which parameters to test.  Note this uses python
            indexing but the '0' parameter refers to the intercept term,
            which is assumed 0.  Therefore, param_num should be > 0.
        method : str, optional
            Can either be 'nm' for Nelder-Mead or 'powell' for Powell.  The
            optimization method that optimizes over nuisance parameters.
            Default is 'nm'.
        stochastic_exog : bool, optional
            When True, the exogenous variables are assumed to be stochastic.
            When the regressors are nonstochastic, moment conditions are
            placed on the exogenous variables.  Confidence intervals for
            stochastic regressors are at least as large as non-stochastic
            regressors.  Default is True.
        return_weights : bool, optional
            If true, returns the weights that optimize the likelihood
            ratio at b0_vals.  Default is False.
        result_object : bool, optional
            Flag indicating whether to return the results as an
            ``EmpLikeTestResult`` NamedTuple instead of a plain tuple. When
            ``return_weights=True`` the NamedTuple holds the same three
            elements as the legacy tuple, so it unpacks identically and is
            always returned, with no warning. When ``return_weights=False``
            the legacy two-element tuple is returned by default and a
            ``FutureWarning`` is issued.

            .. deprecated:: 0.15.0

                In release 0.16.0 or after July 2027, whichever is later, the
                default will change to always return an
                ``EmpLikeTestResult``. Set ``result_object=True`` to opt in
                now, or ``result_object=False`` to silence the warning and
                keep the current return type.

        Returns
        -------
        EmpLikeTestResult or tuple
            If ``result_object=True`` or ``return_weights=True``, a
            NamedTuple with fields:

            llr : float
                The log likelihood ratio for the hypothesized values.
            pvalue : float
                The p-value corresponding to ``llr``.
            weights : ndarray or None
                The observation weights that optimize the likelihood ratio.
                ``None`` when ``return_weights`` is False, since they are
                not computed in that case.

            See :class:`~statsmodels.emplike.descriptive.EmpLikeTestResult`.

            Otherwise (the deprecated default), the plain ``(llr, pvalue)``
            tuple.
        """
        result_object = bool_like(result_object, "result_object", optional=True)
        b0_vals = np.hstack((0, b0_vals))
        param_nums = np.hstack((0, param_nums))
        test_res = self.model.fit().el_test(
            b0_vals,
            param_nums,
            method=method,
            stochastic_exog=stochastic_exog,
            return_weights=return_weights,
            result_object=False,
        )
        llr_test = test_res[0]
        llr_res = llr_test - self.llr
        pval = chi2.sf(llr_res, self.model.exog.shape[1] - 1)
        # The weights come from the underlying OLSResults.el_test, which only
        # computes them when return_weights is True, so they stay None here
        # otherwise.
        weights = test_res[2] if return_weights else None
        if result_object is None and not return_weights:
            warnings.warn(
                "OriginResults.el_test currently returns a plain tuple whose "
                "length depends on the return_weights argument. In release "
                "0.16.0 or after July 2027, whichever is later, the default "
                "behavior will switch to always returning an "
                "EmpLikeTestResult NamedTuple. Set result_object=True to "
                "switch now, or result_object=False to keep the current "
                "behavior and silence this warning.",
                FutureWarning,
                stacklevel=2,
            )
        if result_object or return_weights:
            return EmpLikeTestResult(llr_res, pval, weights)
        return llr_res, pval

    def conf_int_el(
        self,
        param_num,
        upper_bound=None,
        lower_bound=None,
        sig=0.05,
        method="nm",
        stochastic_exog=True,
    ):
        """
        Returns the confidence interval for a regression parameter when the
        regression is forced through the origin

        Parameters
        ----------
        param_num : int
            The parameter number to be tested.  Note this uses python
            indexing but the '0' parameter refers to the intercept term.
        upper_bound : float, optional
            The maximum value the upper confidence limit can be.  The
            closer this is to the confidence limit, the quicker the
            computation.  Default is the .00001 confidence limit under
            normality.
        lower_bound : float, optional
            The minimum value the lower confidence limit can be.
            Default is the .00001 confidence limit under normality.
        sig : float, optional
            The significance level.  Default .05.
        method : str, optional
            Algorithm to optimize over nuisance params.  Can be 'nm' or
            'powell'.  Default is 'nm'.
        stochastic_exog : bool, optional
            Default is True.

        Returns
        -------
        lowerl : float
            The lower confidence limit.
        upperl : float
            The upper confidence limit.
        """
        r0 = chi2.ppf(1 - sig, 1)
        param_num = np.array([param_num])
        if upper_bound is None:
            ci = np.asarray(self.model.fit().conf_int(.0001))
            upper_bound = (np.squeeze(ci[param_num])[1])
        if lower_bound is None:
            ci = np.asarray(self.model.fit().conf_int(.0001))
            lower_bound = (np.squeeze(ci[param_num])[0])

        def f(b0):
            b0 = np.array([b0])
            val = self.el_test(
                b0,
                param_num,
                method=method,
                stochastic_exog=stochastic_exog,
                result_object=True,
            )
            return val.llr - r0

        _param = np.squeeze(self.params[param_num])
        lowerl = optimize.brentq(f, np.squeeze(lower_bound), _param)
        upperl = optimize.brentq(f, _param, np.squeeze(upper_bound))
        return (lowerl, upperl)
