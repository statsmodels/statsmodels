"""
Markov switching regression models

Author: Chad Fulton
License: BSD
"""

from __future__ import division, absolute_import, print_function

import numpy as np
import statsmodels.base.wrapper as wrap

from statsmodels.tsa.regime_switching import markov_switching


class MarkovRegression(markov_switching.MarkovSwitching):
    """
    Markov switching regression model

    Parameters
    ----------
    endog : array_like
        The endogenous variable.
    k_regimes : integer
        The number of regimes.
    trend : {'nc', 'c', 't', 'ct'}
        Whether or not to include a trend. To include an intercept, time trend,
        or both, set `trend='c'`, `trend='t'`, or `trend='ct'`. For no trend,
        set `trend='nc'`. Default is an intercept.
    exog : array_like, optional
        Array of exogenous regressors, shaped nobs x k.
    exog_tvtp : array_like, optional
        Array of exogenous or lagged variables to use in calculating
        time-varying transition probabilities (TVTP). TVTP is only used if this
        variable is provided. If an intercept is desired, a column of ones must
        be explicitly included in this array.
    switching_trend : boolean or iterable, optional
        If a boolean, sets whether or not all trend coefficients are
        switching across regimes. If an iterable, should be of length equal
        to the number of trend variables, where each element is
        a boolean describing whether the corresponding coefficient is
        switching. Default is True.
    switching_exog : boolean or iterable, optional
        If a boolean, sets whether or not all regression coefficients are
        switching across regimes. If an iterable, should be of length equal
        to the number of exogenous variables, where each element is
        a boolean describing whether the corresponding coefficient is
        switching. Default is True.
    switching_variance : boolean, optional
        Whether or not there is regime-specific heteroskedasticity, i.e.
        whether or not the error term has a switching variance. Default is
        False.

    Notes
    -----
    The `trend` is accomodated by prepending columns to the `exog` array. Thus
    if `trend='c'`, the passed `exog` array should not already have a column of
    ones.

    References
    ----------
    Kim, Chang-Jin, and Charles R. Nelson. 1999.
    "State-Space Models with Regime Switching:
    Classical and Gibbs-Sampling Approaches with Applications".
    MIT Press Books. The MIT Press.

    """

    def __init__(self, endog, k_regimes, trend='c', exog=None, exog_tvtp=None,
                 switching_trend=True, switching_exog=True,
                 switching_variance=False, dates=None, freq=None,
                 missing='none'):

        # Properties
        self.trend = trend
        self.switching_trend = switching_trend
        self.switching_exog = switching_exog
        self.switching_variance = switching_variance

        # Exogenous data
        self.k_exog, exog = markov_switching._prepare_exog(exog)

        # Trend
        nobs = len(endog)
        self.k_trend = 0
        self._k_exog = self.k_exog
        trend_exog = None
        if trend == 'c':
            trend_exog = np.ones((nobs, 1))
            self.k_trend = 1
        elif trend == 't':
            trend_exog = (np.arange(nobs) + 1)[:, np.newaxis]
            self.k_trend = 1
        elif trend == 'ct':
            trend_exog = np.c_[np.ones((nobs, 1)),
                               (np.arange(nobs) + 1)[:, np.newaxis]]
            self.k_trend = 2
        if trend_exog is not None:
            exog = trend_exog if exog is None else np.c_[trend_exog, exog]
            self._k_exog += self.k_trend

        # Initialize the base model
        super(MarkovRegression, self).__init__(
            endog, k_regimes, exog_tvtp=exog_tvtp, exog=exog, dates=dates,
            freq=freq, missing=missing)

        # Switching options
        if self.switching_trend is True or self.switching_trend is False:
            self.switching_trend = [self.switching_trend] * self.k_trend
        elif not len(self.switching_trend) == self.k_trend:
            raise ValueError('Invalid iterable passed to `switching_trend`.')
        if self.switching_exog is True or self.switching_exog is False:
            self.switching_exog = [self.switching_exog] * self.k_exog
        elif not len(self.switching_exog) == self.k_exog:
            raise ValueError('Invalid iterable passed to `switching_exog`.')

        self.switching_coeffs = (
            np.r_[self.switching_trend,
                  self.switching_exog].astype(bool).tolist())

        # Parameters
        self.parameters['exog'] = self.switching_coeffs
        self.parameters['variance'] = [1] if self.switching_variance else [0]

    def _resid(self, params):
        """
        Compute residuals

        Notes
        -----
        This function should be overridden in subclassing models.

        In the base model (Markov switching regression), the parameters only
        depend on the regime in the current time period, so
        :math:`f(y \mid S_t, S_{t-1}) = f(y \mid S_t)`

        As in the transition matrix, the previous regime S_{t-1} is represented
        as a column and the next regime S_t is represented as a row. Thus the
        values should be the same across columns (i.e. for each value in a
        given row).
        """
        params = np.array(params, ndmin=1)

        # Since in the base model the values are the same across columns, we
        # only compute a single column, and then expand it below.
        resid = np.zeros((self.k_regimes, 1, self.nobs), dtype=params.dtype)

        for i in range(self.k_regimes):
            # Predict
            if self._k_exog > 0:
                coeffs = params[self.parameters[i, 'exog']]
                resid[i, 0] = np.dot(self.exog, coeffs)

            # Residual
            resid[i] = self.endog - resid[i]

        # Repeat across columns
        return np.repeat(resid, self.k_regimes, axis=1)

    def _conditional_likelihoods(self, params):
        # Get the residuals
        resid = self._resid(params)

        # Compute the conditional likelihoods
        variance = params[self.parameters['variance']].squeeze()
        if self.switching_variance:
            variance = np.reshape(variance, (self.k_regimes, 1, 1))

        conditional_likelihoods = (
            np.exp(-0.5 * resid**2 / variance) / np.sqrt(2 * np.pi * variance))

        return conditional_likelihoods

    def filter(self, *args, **kwargs):
        kwargs.setdefault('results_class', MarkovRegressionResults)
        kwargs.setdefault('results_wrapper_class',
                          MarkovRegressionResultsWrapper)
        return super(MarkovRegression, self).filter(*args, **kwargs)

    def smooth(self, *args, **kwargs):
        kwargs.setdefault('results_class', MarkovRegressionResults)
        kwargs.setdefault('results_wrapper_class',
                          MarkovRegressionResultsWrapper)
        return super(MarkovRegression, self).smooth(*args, **kwargs)

    def _em_iteration(self, params0):
        # Inherited parameters
        result, params1 = super(MarkovRegression, self)._em_iteration(params0)

        # Regression coefficients
        if self._k_exog > 0:
            # First, get coefficients as if all were switching
            tmp = np.sqrt(result.smoothed_marginal_probabilities)
            coeffs = np.zeros((self.k_regimes, self._k_exog))
            for i in range(self.k_regimes):
                _endog = tmp[i] * self.endog
                _exog = tmp[i][:, np.newaxis] * self.exog
                coeffs[i] = np.dot(np.linalg.pinv(_exog), _endog)

            # Next, collapse the non-switching coefficients
            for j in range(self._k_exog):
                if not self.parameters.switching['exog'][j]:
                    _coeff = 0
                    for i in range(self.k_regimes):
                        _coeff += np.sum(
                            coeffs[i, j] *
                            result.smoothed_marginal_probabilities[i])
                    coeffs[:, j] = _coeff / self.nobs

            for i in range(self.k_regimes):
                params1[self.parameters[i, 'exog']] = coeffs[i]

        # Variances

        # First, get variances as if it was switching
        variances = np.zeros(self.k_regimes, dtype=params0.dtype)
        for i in range(self.k_regimes):
            if self._k_exog > 0:
                beta = params1[self.parameters[i, 'exog']]
                resid = self.endog - np.dot(self.exog, beta)
            else:
                resid = self.endog
            variances[i] = (
                np.sum(resid**2 * result.smoothed_marginal_probabilities[i]) /
                np.sum(result.smoothed_marginal_probabilities[i]))

        # Next, collapse if it is non-switching
        if not self.switching_variance:
            variance = 0
            for i in range(self.k_regimes):
                variance += np.sum(variances[i] *
                                   result.smoothed_marginal_probabilities[i])
            variances = variance / self.nobs
        params1[self.parameters['variance']] = variances

        return result, params1

    @property
    def start_params(self):
        """
        Notes
        -----
        These are not very sophisticated and / or good. We set equal transition
        probabilities and interpolate regression coefficients between zero and
        the OLS estimates, where the interpolation is based on the regime
        number. We rely heavily on the EM algorithm to quickly find much better
        starting parameters, which are then used by the typical scoring
        approach.
        """
        # Inherited parameters
        params = markov_switching.MarkovSwitching.start_params.fget(self)

        # Regression coefficients
        if self._k_exog > 0:
            beta = np.dot(np.linalg.pinv(self.exog), self.endog)
            variance = np.var(self.endog - np.dot(self.exog, beta))

            if np.any(self.switching_coeffs):
                for i in range(self.k_regimes):
                    params[self.parameters[i, 'exog']] = (
                        beta * (i / self.k_regimes))
            else:
                params[self.parameters['exog']] = beta
        else:
            variance = np.var(self.endog)

        # Variances
        if self.switching_variance:
            params[self.parameters['variance']] = (
                np.linspace(variance / 10., variance, num=self.k_regimes))
        else:
            params[self.parameters['variance']] = variance

        return params

    @property
    def param_names(self):
        # Inherited parameters
        param_names = np.array(
            markov_switching.MarkovSwitching.param_names.fget(self),
            dtype=object)

        # Regression coefficients
        if np.any(self.switching_coeffs):
            for i in range(self.k_regimes):
                param_names[self.parameters[i, 'exog']] = [
                    '%s[%d]' % (exog_name, i) for exog_name in self.exog_names]
        else:
            param_names[self.parameters['exog']] = self.exog_names

        # Variances
        if self.switching_variance:
            for i in range(self.k_regimes):
                param_names[self.parameters[i, 'variance']] = 'sigma2[%d]' % i
        else:
            param_names[self.parameters['variance']] = 'sigma2'

        return param_names.tolist()

    def transform_params(self, unconstrained):
        # Inherited parameters
        constrained = super(MarkovRegression, self).transform_params(
            unconstrained)

        # Nothing to do for regression coefficients
        constrained[self.parameters['exog']] = (
            unconstrained[self.parameters['exog']])

        # Force variances to be positive
        constrained[self.parameters['variance']] = (
            unconstrained[self.parameters['variance']]**2)

        return constrained

    def untransform_params(self, constrained):
        # Inherited parameters
        unconstrained = super(MarkovRegression, self).untransform_params(
            constrained)

        # Nothing to do for regression coefficients
        unconstrained[self.parameters['exog']] = (
            constrained[self.parameters['exog']])

        # Force variances to be positive
        unconstrained[self.parameters['variance']] = (
            constrained[self.parameters['variance']]**0.5)

        return unconstrained


class MarkovRegressionResults(markov_switching.MarkovSwitchingResults):
    pass


class MarkovRegressionResultsWrapper(
        markov_switching.MarkovSwitchingResultsWrapper):
    pass
wrap.populate_wrapper(MarkovRegressionResultsWrapper, MarkovRegressionResults)
