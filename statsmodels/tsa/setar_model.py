"""
Self-Exciting Threshold Autoregression

Author: Chad Fulton
License: BSD

References
----------

Hansen, Bruce. 1999.
"Testing for Linearity."
Journal of Economic Surveys 13 (5): 551-576.

Hansen, Bruce E. 1997.
"Inference in TAR Models."
Studies in Nonlinear Dynamics & Econometrics 2 (1) (January 1).

Lin, Jin-Lung, and C. W. J. Granger. 1994.
"Forecasting from Non-linear Models in Practice."
Journal of Forecasting 13 (1) (January): 1-9.


Notes
-----

- Assumes homoskedasticity in terms of constructing CIs for threshold
  parameters (see Hansen 1997)
  TODO implement the heteroskedastic correct CIs
- TODO add ability to have different AR orders for each regime
- TODO add ability to remove specific AR orders (e.g. what Potter 1995 does)
       i.e. by manipulating the self.exog matrix
- TODO implement finite sample standard errors (Hansen 1997)
- TODO add interval forecasts (Hyndman 1995)
- TODO add NFE forecasting method (De Gooijer and De Bruin 1998)
- TODO generalize to TAR

"""

from __future__ import division
import numpy as np
import pandas as pd
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.base import data
from statsmodels.tsa.tsatools import add_constant, lagmat
from statsmodels.regression.linear_model import OLS, OLSResults
from statsmodels.tools.decorators import (cache_readonly, cache_writable,
                                          resettable_cache)
import statsmodels.base.wrapper as wrap


class InvalidRegimeError(ValueError):
    pass


class SETAR(OLS, tsbase.TimeSeriesModel):
    """
    Self-Exciting Threshold Autoregressive Model

    Parameters
    ----------
    endog : array-like
        The endogenous variable.
    order : integer
        The order of the SETAR model, indication the number of regimes.
    ar_order : integer
        The order of the autoregressive parameters.
    delay : integer, optional
        The delay for the self-exciting threshold variable.
    thresholds : iterable, optional
        The threshold values separating the data into regimes.
    trend : str {'c','nc'}
        Whether to include a constant or not
        'c' includes constant
        'nc' no constant
    min_regime_frac : scalar, optional
        The minumum fraction of observations in each regime.
    max_delay : integer, optional
        The maximum delay parameter to consider if a grid search is used. If
        left blank, it is set to be the ar_order.
    threshold_grid_size : integer, optional
        The approximate number of elements in the threshold grid if a grid
        search is used.


    Notes
    -----
    threshold_grid_size is only approximate because it uses values from the
    threshold variable itself, approximately evenly spaced, and there may be a
    few more elements in the grid search than requested


    References
    ----------
    See Hansen (1997) Table 1 for threshold critical values.

    """

    threshold_crits = {
        0.8:   4.50,    0.85: 5.10,     0.9: 5.94,
        0.925: 6.53,    0.95: 7.35,     0.975: 8.75,
        0.99:  10.59
    }

    def __init__(self, endog, order, ar_order,
                 delay=None, thresholds=None, trend='c',
                 min_regime_frac=0.1, max_delay=None, threshold_grid_size=100,
                 dates=None, freq=None, missing='none'):

        if delay is not None and delay < 1 or delay > ar_order:
            raise ValueError('Delay parameter must be greater than zero'
                             ' and less than ar_order. Got %d.' % delay)

        # Unsure of statistical properties if length of sample changes when
        # estimating hyperparameters, which happens if delay can be greater
        # than ar_order, so that the number of initial observations changes
        if delay is None and max_delay > ar_order:
            raise ValueError('Maximum delay for grid search must not be '
                             ' greater than the autoregressive order.')

        if delay is None and thresholds is not None:
            raise ValueError('Thresholds cannot be specified without delay'
                             ' parameter.')

        if thresholds is not None and not len(thresholds) + 1 == order:
            raise ValueError('Number of thresholds must match'
                             ' the order of the SETAR model')

        # "Immutable" properties
        self.nobs_initial = ar_order
        self.nobs = endog.shape[0] - ar_order

        self.order = order
        self.ar_order = ar_order
        self.k_trend = int(trend == 'c')
        self.min_regime_frac = min_regime_frac
        self.min_regime_num = np.ceil(min_regime_frac * self.nobs)
        self.max_delay = max_delay if max_delay is not None else ar_order
        self.threshold_grid_size = threshold_grid_size

        # "Flexible" properties
        self.delay = delay
        self.thresholds = thresholds
        if self.thresholds:
            self.thresholds = np.sort(self.thresholds)
        self.regime_indicators = None

        # Estimation properties
        self.nobs_regimes = None
        self.objectives = {}
        self.ar1_resids = None

        # Make a copy of original datasets
        orig_endog = endog
        orig_exog = lagmat(orig_endog, ar_order)

        # Trends
        if self.k_trend:
            orig_exog = add_constant(orig_exog)

        # Create datasets / complete initialization
        endog = orig_endog[self.nobs_initial:]
        exog = orig_exog[self.nobs_initial:]
        super(SETAR, self).__init__(endog, exog,
                                    hasconst=self.k_trend, missing=missing)

        # Overwrite originals
        self.data.orig_endog = orig_endog
        self.data.orig_exog = orig_exog

    def initialize(self):
        """
        Initialize datasets

        Since we manipulate exog and endog as the delay and thresholds are
        changed / selected, this function (and its parent) are called to keep
        all variables up-to-date (mostly making sure shapes are the same)
        """
        self.data.endog = self.endog
        self.data.exog = self.exog
        self.weights = np.repeat(1., self.endog.shape[0])

        super(SETAR, self).initialize()

    def build_exog(self, delay, thresholds, check_nobs=True):
        """
        Build the exogenous matrix for SETAR(m) estimation.

        Parameters
        ----------
        delay : integer
            The delay for the self-exciting threshold variable.
        thresholds : iterable
            The threshold values separating the data into regimes.
        check_nobs : bool, optional
            Whether or not to checks that there are enough observations in each
            regime.

        Returns
        -------
        exog : array-like
            A matrix of lags (up to
            ar_order, plus a constant term) horizontally duplicated once each
            for the number of regimes. Each duplication has the rows for which
            the model dicatates another regime set to zero.
        regime_indicators : array
            Array of which (zero-indexed) regime each observation falls into
        nobs_regimes : iterable
            Number of observations in each regime
        """
        exog = self.exog
        order = len(thresholds) + 1

        exog_transpose = exog.T
        threshold_var = exog[:, delay - (1 - self.k_trend)]
        regime_indicators = np.searchsorted(thresholds, threshold_var)

        k = self.ar_order + self.k_trend
        exog_list = []
        nobs_regimes = ()
        for i in range(order):
            in_regime = (regime_indicators == i)
            nobs_regime = in_regime.sum()

            if check_nobs and nobs_regime < self.min_regime_num:
                raise InvalidRegimeError('Regime %d has too few observations:'
                                         ' threshold values may need to be'
                                         ' adjusted' % i)

            exog_list.append(np.multiply(exog_transpose, in_regime).T)
            nobs_regimes += (nobs_regime,)

        exog = np.concatenate(exog_list, 1)

        return exog, regime_indicators, nobs_regimes

    def fit(self):
        """
        Fits SETAR() model using arranged autoregression.

        Returns
        -------
        statsmodels.tsa.arima_model.SETARResults class

        See also
        --------
        statsmodels.regression.linear_model.OLS : this estimates each regime
        SETARResults : results class returned by fit

        """

        if self.delay is None or self.thresholds is None:
            self.delay, self.thresholds = self.select_hyperparameters()

        self.exog, self.regime_indicators, self.nobs_regimes = self.build_exog(
            self.delay, self.thresholds
        )
        self.initialize()

        beta = self._fit()
        lfit = SETARResults(
            self, beta, normalized_cov_params=self.normalized_cov_params
        )

        return lfit

    def _get_predict_start(self, start, dynamic):
        """
        Returns the index of the given start date.
        """
        if start is None:
            start = 0

        dates = self.data.dates
        if isinstance(start, str):
            if dates is None:
                raise ValueError("Got a string for start and dates is None")
            dtstart = self._str_to_date(start)
            self.data.predict_start = dtstart

            if dynamic and dtstart < self.data.dates[0]:
                raise ValueError('Cannot start dynamic prediction earlier than'
                                 ' record %d (the AR order) due to conditional'
                                 ' least squares estimation. Got %s.' %
                                 (self.ar_order, repr(start)))

            try:
                start = self._get_dates_loc(dates, dtstart)
            except KeyError:
                raise ValueError("Start must be in dates. Got %s | %s" %
                                 (str(start), str(dtstart)))

        self._set_predict_start_date(start)
        return start

    def predict(self, delay, thresholds, params, start=None, end=None,
                dynamic=False, method='mc'):
        """
        In-sample predictions and/or out-of-sample forecasts

        Parameters
        ----------
        delay : integer
            The delay for the self-exciting threshold variable.
        thresholds : iterable
            The threshold values separating the data into regimes.
        params : array-like, optional after fit has been called
            Parameters of a linear model
        start : int, str, or datetime, optional
            Zero-indexed observation number at which to start forecasting, ie.,
            the first forecast is start. Can also be a date string to
            parse or a datetime type.
        end : int, str, or datetime, optional
            Zero-indexed observation number at which to end forecasting, ie.,
            the first forecast is start. Can also be a date string to
            parse or a datetime type. However, if the dates index does not
            have a fixed frequency, end must be an integer index if you
            want out of sample prediction.
        dynamic : bool, optional
            The `dynamic` keyword affects in-sample prediction. If dynamic
            is False, then the in-sample lagged values are used for
            prediction. If `dynamic` is True, then in-sample forecasts are
            used in place of lagged dependent variables. The first forecasted
            value is `start`.
        method : str {'n','nfe','mc','bs'}, optional
            Method to use for dynamic prediction and out-of-sample forecasting
            'n' naive method, assumes errors are at means (zero)
            'nfe' Normal Forecast Error method (Not yet implemented)
            'mc' Monte Carlo method (Gaussian errors)
            'bs' Bootstrap method (errors drawn randomly with replacement from
                 residuals)

        Returns
        -------
        predicted : array
            Array of predicted and/or forecasted values

        Notes
        -----
        1. Static in-sample prediction is just like that for OLS because the
           regime is known and static implies using actual valeus, so we can
           just dot the full set of parameters with the self.exog dataset
           (recall that this is not simply a lagmat(endog, ...) - it was
           created in self.build_exog).

        2. Dynamic in-sample prediction takes actual data up to start as the
           initial values for forecasting.

        3. Out-of-sample forecasting is performed if end is after the last
           sample observation.
        """

        start = self._get_predict_start(start, dynamic)
        end, out_of_sample = self._get_predict_end(end)

        # In-sample prediction
        prediction = []

        # Static: for all y_t, use *actual* y_{t-1}, ..., y_{t-p}
        if not dynamic:
            exog = self.exog[start:end + 1, ]
            prediction = np.dot(exog, params)
        # Dynamic: use y_{start-1-ar_order}, ..., y_{start-1} as initial
        # datapoints, forecast everything else
        else:
            orig_start = start + self.nobs_initial
            initial = self.data.orig_endog[
                orig_start - self.ar_order:orig_start
            ].squeeze()
            prediction = self.forecast(delay, thresholds, params,
                                       end - (start - 1), initial,
                                       method=method)

        # Out-of-sample forecasting
        if out_of_sample:
            # Get our initial data
            initial = prediction
            required_obs = self.ar_order - len(initial)
            if required_obs > 0:
                initial = np.r_[
                    self.data.orig_endog[start-required_obs:start].squeeze(),
                    initial
                ]

            # Add the forecast
            forecast = self.forecast(delay, thresholds, params, out_of_sample,
                                     initial, method=method)
            prediction = np.r_[prediction, forecast]

        # Date handling if Pandas endog
        predict_dates = getattr(self.data, 'predict_dates', None)
        if (predict_dates is not None and
                isinstance(self.data, data.PandasData)):
            prediction = pd.TimeSeries(prediction,
                                       index=self.data.predict_dates)

        return prediction

    def _forecast_nfe(self, delay, thresholds, params, steps, initial):
        raise NotImplementedError

    def _forecast_monte_carlo(self, delay, thresholds, params, steps, initial,
                              errors, scale):
        """
        Generic method for generating forecasts using a Monte Carlo approach
        """
        forecast = []

        # Generate forecasts
        for step in range(steps):
            # Forecast the next value
            lags = initial[:-self.ar_order-1:-1]
            regime = np.searchsorted(thresholds, lags[delay-1])
            k = self.ar_order + self.k_trend
            exog = np.r_[1, lags] if self.k_trend else lags

            if step == 0:
                # First forecast has no error
                error = 0
            elif isinstance(errors, tuple):
                error = errors[regime][step-1]
            else:
                error = errors[step-1]
            forecast.append(
                np.mean(error*scale[step] + np.dot(
                    exog,
                    params[k*regime:k*(regime + 1)])
                )
            )
            initial = np.r_[initial[1:], forecast[-1]]

        return np.array(forecast)

    def forecast(self, delay, thresholds, params, steps=1, initial=None,
                 method='mc', reps=100, resids=None, scale=None):
        """
        Out-of-sample forecasts

        Parameters
        ----------
        delay : integer
            The delay for the self-exciting threshold variable.
        thresholds : iterable
            The threshold values separating the data into regimes.
        params : array-like, optional after fit has been called
            Parameters of a linear model
        steps : int, optional
            The number of out of sample forecasts from the end of the
            sample.
        initial : array, optional
            The predetermined endogenous values on which to begin forecasting.
        method : str {'n','nfe','mc','bs'}, optional
            Method to use in out-of-sample forecasting
            'n' naive method, assumes errors are at means (zero)
            'nfe' Normal Forecast Error method (Not yet implemented)
            'mc' Monte Carlo method (Gaussian errors)
            'bs' Bootstrap method (errors drawn randomly with replacement from
                 residuals)
        reps : int, optional
            The number of repetitions for draws in the monte carlo and
            bootstrap methods.
        resids : array, tuple optional
            The residuals from fit(), from which to draw the errors if the
            bootstrap method is used. Optionally can provide different pools of
            errors per regime, in which case a tuple of length equal to the
            order of the model (i.e. the number of regimes) must be provided.
        scale : array, optional
            Optional array of scale parameters to allow for heteroskedasticity
            in monte carlo sample generation. Array must have length `steps`.

        Returns
        -------
        forecast : array
            Array of out-of-sample forecasts
        """

        if initial is None:
            initial = self.endog

        if initial.shape[0] < self.ar_order:
            raise ValueError('Cannot forecast with less than %d (the AR order)'
                             ' initial datapoints.' % self.ar_order)

        if isinstance(resids, tuple) and not len(resids) == len(thresholds)+1:
            raise ValueError('If regime-specific residuals are provided as a'
                             ' tuple, it must be of length equal to the'
                             ' number of regimes (here %d). Got %d.' %
                             (len(thresholds)+1, len(resids)))

        if scale is None:
            scale = np.ones((steps, 1))
        elif not len(scale) == steps:
            raise ValueError('If the scale array is provided, it must be of'
                             ' length `steps` (here %d) to provide a scale for'
                             ' the error term of each observation. Got %d.' %
                             (steps, len(scale)))

        if method == 'n':
            errors = np.zeros((steps, 1))
            forecast = self._forecast_monte_carlo(
                delay, thresholds, params, steps, initial, errors=errors
            )
        elif method == 'mc':
            errors = np.random.normal(size=(steps-1, reps))
            forecast = self._forecast_monte_carlo(
                delay, thresholds, params, steps, initial,
                errors=errors, scale=scale
            )
        elif method == 'bs':
            if resids is None:
                # Note: must cache predict_dates; it gets overwritten
                #       in the self.predict() call
                dates = self.data.predict_dates
                resids = self.endog - self.predict(params)
                self.data.predict_dates = dates

            # If we have regime-specific residual pools, fore each regime we
            # make enough draws with replacement from each pool to fill the
            # entire forecast, and provide an array for errors for each regime
            if isinstance(resids, tuple):
                errors = ()
                for regime_resids in resids:
                    errors += (regime_resids[
                        np.random.random_integers(
                            0, len(regime_resids)-1, (steps-1, reps)
                        )
                    ],)
            # Otherwise, we just provide a single array of residuals
            else:
                errors = resids[
                    np.random.random_integers(
                        0, len(resids)-1, (steps-1, reps)
                    )
                ]
            forecast = self._forecast_monte_carlo(
                delay, thresholds, params, steps, initial,
                errors=errors, scale=scale
            )
        elif method == 'nfe':
            forecast = self._forecast_nfe(params, steps, initial)
        else:
            raise ValueError('Invalid forecasting method. Valid methods are'
                             ' "n", "nfe", "mc", and "bs". Got %s.' % method)

        return forecast

    def _grid_search_objective(self, delay, thresholds, XX, resids):
        """
        Objective function to maximize in SETAR(2) hyperparameter grid search

        Corresponds to f_2(\gamma, d) in Hansen (1999), but extended to any
        number of thresholds.

        Parameters
        ----------
        delay : integer
            The delay for the self-exciting threshold variable.
        thresholds : iterable
            The threshold values separating the data into regimes.
        XX : array-like
            (X'X)^{-1} from a SETAR(1) specification (i.e. AR(1))
        resids : array-like
            The residuals from a SETAR(1) specification (i.e. AR(1))

        Returns
        -------
        objective : float
            The value of the objective function
        """
        exog, _, _ = self.build_exog(delay, thresholds)

        # Intermediate calculations
        k = self.ar_order + self.k_trend
        X1 = exog[:, :-k]
        X = self.exog
        X1X1 = X1.T.dot(X1)
        XX1 = X.T.dot(X1)
        Mn = np.linalg.inv(
            X1X1 - XX1.T.dot(XX).dot(XX1)
        )

        # Return objective
        return resids.T.dot(X1).dot(Mn).dot(X1.T).dot(resids)

    def _select_hyperparameters_grid(self, thresholds, threshold_grid_size,
                                     XX, resids, delay_grid=None):
        """
        Maximizes objective function, given already selected thresholds and,
        optionally, an already selected delay.

        Parameters
        ----------
        thresholds : iterable
            Already-selected threshold values (can be empty).
        threshold_grid_size : integer
            The approximate number of elements in the threshold grid if a grid
            search is used.
        XX : array-like
            (X'X)^{-1} from a SETAR(1) specification (i.e. AR(1))
        resids : array-like
            The residuals from a SETAR(1) specification (i.e. AR(1))
        delay_grid : iterable, optional
            The grid of delay parameters to check.

        Returns
        -------
        delay : int
            Selected delay parameter
        thresholds : iterable
            Selected threshold parameter(s)
        """

        if delay_grid is None:
            delay_grid = range(1, self.max_delay + 1)

        max_obj = 0
        params = (None, None)
        # Iterate over possible delay values
        for delay in delay_grid:

            # Build the appropriate threshold grid given delay
            threshold_var = np.unique(np.sort(self.endog[:-delay]))
            nobs = len(threshold_var)
            indices = np.arange(self.min_regime_num,
                                nobs - self.min_regime_num,
                                max(np.floor(nobs / threshold_grid_size), 1),
                                dtype=int)
            threshold_grid = threshold_var[indices]

            # Iterate over possible threshold values
            for threshold in threshold_grid:
                if threshold in thresholds:
                    continue
                try:
                    iteration_thresholds = np.sort([threshold] + thresholds)
                    key = (delay,)+tuple(iteration_thresholds)
                    if key not in self.objectives:
                        obj = self._grid_search_objective(
                            delay, iteration_thresholds,
                            XX, resids
                        )
                        self.objectives[key] = obj
                    if self.objectives[key] > max_obj:
                        max_obj = self.objectives[key]
                        params = (delay, threshold)
                # Some threshold values don't allow enough values in each
                # regime; we just need to not select those thresholds
                except InvalidRegimeError:
                    pass

        return params

    def select_hyperparameters(self, threshold_grid_size=None, maxiter=100):
        """
        Select delay and threshold hyperparameters via grid search.

        Selected parameters minimize the sum of squared errors over the grid.

        Parameters
        ----------
        threshold_grid_size : integer, optional
            The approximate number of elements in the threshold grid if a grid
            search is used.
        maxiter : integer, optional
            Maximum iterations in iterative threshold (re)estimation.

        Returns
        -------
        delay : int
            Selected delay parameter
        thresholds : iterable
            Selected threshold parameter(s)
        """

        # SETAR(1) is a special case
        if self.order == 1:
            return 0, ()

        # Cache calculations
        XX = np.linalg.inv(self.exog.T.dot(self.exog))    # (X'X)^{-1}
        self.ar1_resids = resids = self.endog - np.dot(   # SETAR(1) residuals
            self.exog,
            XX.dot(self.exog.T.dot(self.endog))
        )

        # Get default threshold grid size, if necessary
        if threshold_grid_size is None:
            threshold_grid_size = self.threshold_grid_size

        # Set delay grid if delay is specified
        delay_grid = [self.delay] if self.delay is not None else None

        # Estimate the delay and an initial value for the dominant threshold
        thresholds = []
        delay, threshold = self._select_hyperparameters_grid(
            thresholds, threshold_grid_size, XX, resids, delay_grid=delay_grid
        )
        thresholds.append(threshold)

        # Get remaining thresholds
        for i in range(2, self.order):

            # Get initial estimate of next threshold
            _, threshold = self._select_hyperparameters_grid(
                thresholds, threshold_grid_size, XX, resids,
                delay_grid=[delay]
            )
            thresholds.append(threshold)

            # Iterate threshold selection to convergence
            proposed = thresholds[:]
            iteration = 0
            while True:
                iteration += 1

                # Recalculate each threshold individually, holding the others
                # constant, starting at the first threshold
                for j in range(i):
                    _, threshold = self._select_hyperparameters_grid(
                        thresholds[:j] + thresholds[j + 1:],
                        threshold_grid_size, XX, resids,
                        delay_grid=[delay]
                    )
                    proposed[j] = threshold

                # If the recalculation produced no change, we've converged
                if proposed == thresholds:
                    break
                # If convergence is not happening fast enough
                if iteration > maxiter:
                    print ('Warning: Maximum number of iterations has been '
                           'exceeded.')
                    break

                thresholds = proposed[:]

        return delay, np.sort(thresholds)


class SETARResults(OLSResults, tsbase.TimeSeriesModelResults):
    """
    Class to hold results from fitting a SETAR model.

    Parameters
    ----------
    model : ARMA instance
        The fitted model instance
    params : array
        Fitted parameters
    normalized_cov_params : array, optional
        The normalized variance covariance matrix
    scale : float, optional
        Optional argument to scale the variance covariance matrix.
    """

    _cache = {}

    def __init__(self, model, params, normalized_cov_params=None, scale=1.):
        super(SETARResults, self).__init__(model, params,
                                           normalized_cov_params, scale)
        self._cache_alternatives = {}

    @cache_readonly
    def _AR(self):
        return self._get_model(1, self.model.ar_order)

    def _get_model(self, order, ar_order):
        if order not in self._cache_alternatives:
            mod = SETAR(
                self.model.data.orig_endog,
                order=order,
                ar_order=ar_order,
                threshold_grid_size=self.model.threshold_grid_size
            )
            # We can supply some already-calculated objective values to the
            # alternative model, since it's on the same dataset
            mod.objectives = self.model.objectives
            self._cache_alternatives[order] = mod.fit()
        return self._cache_alternatives[order]

    def f_stat(self, null=None):
        """
        F statistic for order selection

        Parameters
        ----------
        null : SETARResults, optional
            The null hypothesis to test against. If not provided, calculated
            with SETAR(1) as the null.

        Returns
        -------
        f_stat : float
            The value of the F-statistic
        """
        if null is None:
            null = self._AR
        elif isinstance(null, int):
            null = self._get_model(null, self.model.ar_order)
        return self.model.nobs * (null.ssr - self.ssr) / self.ssr

    def order_test(self, null=1, reps=100, heteroskedasticity='n'):
        """
        SETAR Order Selection Test

        Parameters
        ----------
        null : SETARResults, int, optional
            The model under the null hypothesis. Can also be an integer
            indicating the order of the SETAR model under the null hypothesis.
            The order must be less than the order of the alternate hypothesis
            (i.e. the fitted model).
        reps : int, optional
            the number of bootstrap replications to perform
        heteroskedasticity : str {'n','r','g'}, optional
            Assumption on type of heteroskedasticity in the error term
            'n' No heteroskedasticity (homoskedasticity)
            'r' Between-regime heteroskedasticity only, so that there is
                within-regime homoskedasticity.
                Only applicable if the null and alternative hypotheses are both
                of order greater than one.
            'g' Heteroskedasticity of general form

        Returns
        -------
        f_stat : float
            The value of the max-F statistic
        pvalue : float
            The bootstrapped p-value for the F test

        Notes
        -----
        Between-regime heteroskedastic assumes homoskedasticity within regimes.
        Thus to do the bootstrap, essentially each regime has a different
        "pool" of errors available to be drawn from, corresponding to the
        actual residuals from that regime. This is equivalent to assuming that
        the errors are scaled differently in between-regimes, but scaled the
        same within-regimes.

        Heteroskedasticity of general form allows each observation to have its
        own scale. Thus in addition to a common pool of errors (which are
        rescaled residuals from the entire sample), we pass an array of scales
        to reverse the rescaling when generating the bootstrap observation.

        Each repetition involves running select_hyperparameters() on a model.
        This will be slow even in models of reasonable sizes and orders.
        """

        if isinstance(null, int):
            null = self._get_model(null, self.model.ar_order)

        if null.model.order >= self.model.order:
            raise ValueError('Model under the null hypothesis must have'
                             ' order less than %d (the order of the currently'
                             ' fitted model). Got %d.' %
                             (self.model.order, null_order))

        if heteroskedasticity == 'r' and null.model.order == 1:
            raise ValueError('The regime heteroskedastic test is only'
                             ' applicable when testing between two'
                             ' higher-order SETAR models.')

        exog = self.model.data.orig_exog[self.model.nobs_initial:]
        # This is a bit of a cludge, to deal with pandas datasets
        dta = data.handle_data(
            self.model.data.orig_endog[:self.model.nobs_initial], None
        )
        initial = dta.endog

        scale = None
        if heteroskedasticity == 'n':
            errors = null.resid
        elif heteroskedasticity == 'r':
            # Utilizes the tuple option for different bootstrapping errors
            # in different regimes
            errors = tuple([
                null.resid[null.regime_indicators == regime]
                for regime in range(null.order)
            ])
        elif heteroskedasticity == 'g':
            # Utilizes the scale option for differently scaled errors in
            # different periods
            res = OLS(null.resid**2, exog**2).fit()
            scale = res.fittedvalues
            # Temporarily replace negative scales with infinity so that the
            # division makes the scaled error zero
            scale[scale < 0] = np.Inf
            errors = null.resid / scale**0.5
            # Now, set those scales (which were negative) back to zero
            scale[scale == np.Inf] = 0
        else:
            raise ValueError('Invalid type of heteroskedasticity. Valid types'
                             ' are "n", "r", and "g". Got %s.' %
                             heteroskedasticity)

        f_stats = []
        for rep in range(reps):
            # Create a sample from these parameters with these errors
            # (this amounts to doing a bootstrap forecast)
            sample = self.model.forecast(
                null.delay, null.thresholds, null.params, int(null.nobs),
                initial=initial, method='bs', reps=1,
                resids=errors, scale=scale
            )
            # Estimate a SETAR model on the simulated sample
            simul_res = SETAR(
                np.r_[initial, sample],
                order=self.model.order,
                ar_order=self.model.ar_order,
                threshold_grid_size=self.model.threshold_grid_size
            ).fit()
            # Estimate the null SETAR model on the simulated sample
            simul_null = simul_res._get_model(null.order, null.ar_order)
            f_stats.append(simul_res.f_stat(simul_null))

        f_stat = self.f_stat(null)
        pvalue = np.mean(f_stats > f_stat)

        return f_stat, pvalue, f_stats

    @cache_readonly
    def ar_order(self):
        return self.model.ar_order

    @cache_readonly
    def delay(self):
        return self.model.delay

    @cache_readonly
    def order(self):
        return self.model.order

    @cache_readonly
    def thresholds(self):
        return self.model.thresholds

    @cache_readonly
    def regime_indicators(self):
        return self.model.regime_indicators

    @cache_readonly
    def bse(self):
        # Get White's corrected standard errors here. Report them, because
        # otherwise we don't allow heteroskedasticity between regimes
        return self.HC0_se

    @cache_readonly
    def fittedvalues(self):
        return self.model.predict(self.model.delay,
                                  self.model.thresholds,
                                  self.params)

    @cache_readonly
    def wresid(self):
        return self.resid

    @cache_readonly
    def resid(self):
        return self.model.endog - self.model.predict(self.model.delay,
                                                     self.model.thresholds,
                                                     self.params)

    @cache_readonly
    def threshold_SSR(self):
        """
        Sum of squared errors for each threshold, calculated for each
        possible alternative threshold value.
        """

        # Compute the sum of squared residuals for each model
        return {
            key: (self._AR.ssr - obj)
            for key, obj in self.model.objectives.items()
            if len(key) == len(self.model.thresholds) + 1
        }

    @cache_readonly
    def threshold_LR(self):
        """
        Likelihood ratio statistics for each threshold, calculated for each
        possible alternative threshold value.
        """

        # Local copies
        delay = self.model.delay
        thresholds = self.model.thresholds.tolist()

        # Compute the likelihood ratio statistics for each model, relative to
        # the thresholds, the confidence sets, and the conservative confidence
        # intervals
        threshold_LR = []
        for threshold_idx in range(len(thresholds)):
            threshold = thresholds[threshold_idx]
            held_thresholds = set(
                thresholds[:threshold_idx] + thresholds[threshold_idx+1:]
            )

            LR_set = {}
            for (key, SSR) in self.threshold_SSR.items():
                alt_thresholds = set(key[1:])
                # Only need to test the specific alternative of this specific
                # threshold being replaced
                if (not key[0] == delay) or (not
                   alt_thresholds.issuperset(held_thresholds)):
                    continue
                alt_threshold = alt_thresholds.difference(
                    held_thresholds
                ).pop()

                # Likelihood ratio statistic
                LR_set[alt_threshold] = (
                    self.model.nobs * (SSR - self.ssr) / self.ssr
                )
            threshold_LR.append(LR_set)

        return threshold_LR

    def conf_set_thresholds(self, alpha=0.05):
        """
        Compute confidence sets by inverting the LR statistic

        Parameters
        ----------
        alpha : float
            significance level for the confidence sets

        Returns
        -------
        conf_sets : iterable
            A list of confidence sets, one for each threshold.
            Each confidence set is a list of thresholds in the confidence set
            at the specified level.
        """
        if 1-alpha not in self.model.threshold_crits.keys():
            raise ValueError('Threshold confidence intervals can only be'
                             ' calculated at levels [0.01, 0.025, 0.05, 0.075,'
                             ' 0.1, 0.15, 0.2]. God %f.' % alpha)
        crit = self.model.threshold_crits[1 - alpha]

        thresholds = self.model.thresholds.tolist()

        # Compute the confidence sets
        conf_sets = []
        for threshold_idx in range(len(thresholds)):
            threshold = thresholds[threshold_idx]
            held_thresholds = set(
                thresholds[:threshold_idx] + thresholds[threshold_idx+1:]
            )

            conf_sets.append(np.sort([
                alt_threshold for (alt_threshold, LR)
                in self.threshold_LR[threshold_idx].items()
                if LR < crit
            ]))

        return conf_sets

    def conf_int_thresholds(self, alpha=0.05):
        """
        Compute conservative confidence intervals by inverting the LR statistic

        Parameters
        ----------
        alpha : float
            significance level for the confidence intervals

        Returns
        -------
        conf_ints : iterable
            A list of confidence intervales, one for each threshold.
            Each confidence interval is a tuple (lower value, upper value).
        """
        conf_ints = [
            (conf_set[0], conf_set[-1])
            for conf_set in self.conf_set_thresholds(alpha=alpha)
        ]

        return conf_ints

    def plot_threshold_ci(self, threshold_idx, ax=None, **kwargs):
        """
        Plot the likelihood ratio sequence and confidence interval cutoffs
        for alternate values of a given threshold.

        Plots threshold on the horizontal and the likelihood ratio sequence on
        vertical axis.

        Parameters
        ----------
        threshold_idx : array_like
            The index of the threshold
        ax : Matplotlib AxesSubplot instance, optional
            If given, this subplot is used to plot in instead of a new figure
            being created.
        **kwargs : kwargs, optional
            Optional keyword arguments that are directly passed on to the
            Matplotlib ``plot`` and ``axhline`` functions.

        Returns
        -------
        fig : Matplotlib figure instance
            If `ax` is None, the created figure. Otherwise the figure to which
            `ax` is connected.
        """
        from statsmodels.graphics import utils
        fig, ax = utils.create_mpl_ax(ax)
        fig.set(**kwargs)

        # Make sure we have the LR statistics
        self.conf_int_thresholds()

        # Plot the LRs
        LR_set = self.threshold_LR[threshold_idx]
        LR_keys = np.sort(LR_set.keys())
        LR, = ax.step(LR_keys, [LR_set[key] for key in LR_keys], 'k-')

        # Plot the critical values
        xlim = ax.get_xlim()
        crits = self.model.threshold_crits
        l90 = ax.hlines(crits[0.90], xlim[0], xlim[1], linestyle='--')
        l95 = ax.hlines(crits[0.95], xlim[0], xlim[1], linestyle='-.')
        l99 = ax.hlines(crits[0.99], xlim[0], xlim[1], linestyle='dotted')

        # Add a legend
        labels = [
            '$LR_n(\gamma_%d)$' % (threshold_idx+1),
            '90% Critical', '95% Critical', '99% Critical'
        ]
        ax.legend([LR, l90, l95, l99], labels, 'lower right')

        # Add titles
        ax.set(
            title=('Confidence Interval Construction for Threshold'
                   ' $\gamma_%d$' % (threshold_idx + 1)),
            xlabel='Threshold Variable: $Y_{t-%d}$' % self.model.delay,
            ylabel=('Likelihood Ratio Sequence in $\gamma_%d$' %
                    (threshold_idx+1))
        )

        return fig

    def predict(self, start=None, end=None, dynamic=False, method='mc'):
        """
        In-sample predictions and/or out-of-sample forecasts

        Parameters
        ----------
        start : int, str, or datetime, optional
            Zero-indexed observation number at which to start forecasting, ie.,
            the first forecast is start. Can also be a date string to
            parse or a datetime type.
        end : int, str, or datetime, optional
            Zero-indexed observation number at which to end forecasting, ie.,
            the first forecast is start. Can also be a date string to
            parse or a datetime type. However, if the dates index does not
            have a fixed frequency, end must be an integer index if you
            want out of sample prediction.
        dynamic : bool, optional
            The `dynamic` keyword affects in-sample prediction. If dynamic
            is False, then the in-sample lagged values are used for
            prediction. If `dynamic` is True, then in-sample forecasts are
            used in place of lagged dependent variables. The first forecasted
            value is `start`.
        method : str {'n','nfe','mc','bs'}, optional
            Method to use for dynamic prediction and out-of-sample forecasting
            'n' naive method, assumes errors are at means (zero)
            'nfe' Normal Forecast Error method (Not yet implemented)
            'mc' Monte Carlo method (Gaussian errors)
            'bs' Bootstrap method (errors drawn randomly with replacement from
                 residuals)

        Returns
        -------
        predicted : array
            Array of predicted and/or forecasted values

        See Also
        --------
        statsmodels.tsa.setar_model.SETAR.predict : prediction implementation
        """
        return self.model.predict(self.model.delay, self.model.thresholds,
                                  self.params, start, end,
                                  dynamic=dynamic, method=method)

    def forecast(self, steps=1, method='mc', reps=100):
        """
        Out-of-sample forecasts

        Parameters
        ----------
        steps : int, optional
            The number of out of sample forecasts from the end of the
            sample.
        method : str {'n','nfe','mc','bs'}, optional
            Method to use in out-of-sample forecasting
            'n' naive method, assumes errors are at means (zero)
            'nfe' Normal Forecast Error method (Not yet implemented)
            'mc' Monte Carlo method (Gaussian errors)
            'bs' Bootstrap method (errors drawn randomly with replacement from
                 residuals)
        reps : int, optional
            The number of repetitions for draws in the monte carlo and
            bootstrap methods.

        Returns
        -------
        forecast : array
            Array of out-of-sample forecasts

        See Also
        --------
        statsmodels.tsa.setar_model.SETAR.forecast : forecasting implementation
        """
        return self.model.forecast(self.model.delay, self.model.thresholds,
                                   self.params, steps, method=method,
                                   reps=reps, resids=self.resid)

    def _make_exog_names(self):
        exog_names = []
        for regime in range(self.model.order + 1):
            exog_names += ['Const.']
            exog_names += [
                'y_{t-%d}^{(%d)}' % (i, regime)
                for i in range(1, self.model.ar_order + 1)
            ]

        return exog_names

    def _make_regime_descriptions(self):
        titles = []
        length = 0
        if self.model.order == 0:
            titles.append('\gamma_1 \lt \infty')
        else:
            delay = self.model.delay
            thresholds = self.model.thresholds
            for regime in range(self.model.order):
                if regime == 0:
                    titles.append(
                        'y_{t-%d} in (-Inf, %.2f]' %
                        (delay, thresholds[0])
                    )
                elif regime == self.model.order - 1:
                    titles.append(
                        'y_{t-%d} in (%.2f, Inf)' %
                        (delay, thresholds[-1])
                    )
                else:
                    titles.append(
                        'y_{t-%d} in (%.2f, %.2f]' %
                        (delay, thresholds[regime - 1], thresholds[regime])
                    )
                if len(titles[-1]) > length:
                    length = len(titles[-1])

        return titles

    def summary(self, yname=None, title=None, alpha=.05):
        """
        Summarize the SETAR Results

        Parameters
        ----------
        yname : string, optional
            Default is `y`
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

        xname = self._make_exog_names()

        model = (
            self.model.__class__.__name__ + '('
            + repr(self.model.order) + ';'
            + ','.join([repr(self.model.ar_order), repr(self.model.delay)])
            + ')'
        )

        try:
            dates = self.data.dates
            sample = [('Sample:', [dates[0].strftime('%m-%d-%Y')])]
            sample += [('', [' - ' + dates[-1].strftime('%m-%d-%Y')])]
        except:
            start = self.model.nobs_initial + 1
            end = repr(self.model.data.orig_endog.shape[0])
            sample = [('Sample:', [repr(start) + ' - ' + end])]

        top_left = [('Dep. Variable:', None),
                    ('Model:', [model]),
                    ('Method:', ['Least Squares']),
                    ('Date:', None),
                    ('Time:', None)
                    ] + sample

        top_right = [('No. Observations:', None),
                     ('Df Residuals:', None),
                     ('Df Model:', None),
                     ('Log-Likelihood:', None),
                     ('AIC:', ["%#8.4g" % self.aic]),
                     ('BIC:', ["%#8.4g" % self.bic])
                     ]

        if title is None:
            title = self.model.__class__.__name__ + ' ' + "Regression Results"

        # Create summary table instance
        from statsmodels.iolib.summary import Summary, summary_params, forg
        from statsmodels.iolib.table import SimpleTable
        from statsmodels.iolib.tableformatting import fmt_params
        smry = Summary()
        warnings = []

        # Add model information
        smry.add_table_2cols(self, gleft=top_left, gright=top_right,
                             yname=yname, xname=xname, title=title)

        # Add hyperparameters summary table
        if (1 - alpha) not in self.model.threshold_crits:
            warnings.append("Critical value for threshold estimates is"
                            " unavailable at the %d%% level. Using 95%%"
                            " instead." % ((1-alpha)*100))
            alpha = 0.05
        alp = str((1-alpha)*100)+'%'
        conf_int = self.conf_int_thresholds(alpha)

        # (see summary_params())
        confint = [
            "%s %s" % tuple(map(forg, conf_int[i]))
            for i in range(len(conf_int))
        ]
        confint.insert(0, '')
        len_ci = map(len, confint)
        max_ci = max(len_ci)
        min_ci = min(len_ci)

        if min_ci < max_ci:
            confint = [ci.center(max_ci) for ci in confint]

        thresholds = list(self.model.thresholds)
        param_header = ['coef', '[' + alp + ' Conf. Int.]']
        param_stubs = ['Delay'] + ['\gamma_%d' % (threshold_idx + 1)
                                   for threshold_idx in range(len(thresholds))]
        param_data = zip([self.model.delay] + map(forg, thresholds), confint)

        parameter_table = SimpleTable(param_data,
                                      param_header,
                                      param_stubs,
                                      title=None,
                                      txt_fmt=fmt_params)
        smry.tables.append(parameter_table)

        # Add parameter tables for each regime
        results = np.c_[
            self.params, self.bse, self.tvalues, self.pvalues,
        ].T
        conf = self.conf_int(alpha)
        k = self.model.ar_order + self.model.k_trend
        regime_desc = self._make_regime_descriptions()
        max_len = max(map(len, regime_desc))
        for regime in range(1, self.model.order + 1):
            res = (self,)
            res += tuple(results[:, k*(regime - 1):k*regime])
            res += (conf[k*(regime - 1):k*regime],)
            table = summary_params(res, yname=yname,
                                   xname=xname[k*regime:k*(regime+1)],
                                   alpha=alpha, use_t=True)

            # Add regime descriptives, if multiple regimes
            if self.model.order > 1:
                # Replace the header row
                header = ["\n" + str(cell) for cell in table.pop(0)]
                title = ("Regime %d" % regime).center(max_len)
                desc = regime_desc[regime - 1].center(max_len)
                header[0] = "%s \n %s" % (title, desc)
                table.insert_header_row(0, header)
                # Add diagnostic information
                nobs = [
                    'nobs_%d' % regime, self.model.nobs_regimes[regime - 1],
                    '', '', '', ''
                ]
                table.insert(len(table), nobs, 'header')

            smry.tables.append(table)

        # Add notes / warnings, added to text format only
        warnings.append("Reported parameter standard errors are White's (1980)"
                        " heteroskedasticity robust standard errors.")
        warnings.append("Threshold confidence intervals calculated as"
                        " Hansen's (1997) conservative (non-disjoint)"
                        " intervals")

        if self.model.exog.shape[0] < self.model.exog.shape[1]:
            wstr = "The input rank is higher than the number of observations."
            warnings.append(wstr)

        if warnings:
            etext = [
                "[{0}] {1}".format(i + 1, text)
                for i, text in enumerate(warnings)
            ]
            etext.insert(0, "Notes / Warnings:")
            smry.add_extra_txt(etext)

        return smry


class SETARResultsWrapper(wrap.ResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(tsbase.TimeSeriesResultsWrapper._wrap_attrs,
                                   _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(
        tsbase.TimeSeriesResultsWrapper._wrap_methods,
        _methods
    )
wrap.populate_wrapper(SETARResultsWrapper, SETARResults)
