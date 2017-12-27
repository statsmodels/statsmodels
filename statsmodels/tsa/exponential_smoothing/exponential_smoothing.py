"""
Exponential smoothing models

Author: Chad Fulton, Terence L van Zyl
License: BSD-3
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import pandas as pd
import warnings

from scipy.optimize import minimize

import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tools.decorators import cache_readonly, resettable_cache
import statsmodels.base.wrapper as wrap
from statsmodels.base.data import PandasData
from statsmodels.base.optimizer import Optimizer
from statsmodels.tools.numdiff import (_get_epsilon, approx_hess_cs,
                                       approx_fprime_cs, approx_fprime)

from statsmodels.tools.tools import Bunch
from statsmodels.tsa.statespace.tools import find_best_blas_type, prepare_exog
from statsmodels.tsa.exponential_smoothing._exponential_smoother import (
    ses_filter, des_filter, ces_filter, zes_filter)

prefix_es_filter_map = {
    's': ses_filter, 'd': des_filter,
    'c': ces_filter, 'z': zes_filter
}


def py_es_filter(nobs, nobs_init, nobs_output, trend, seasonal, damped_trend,
                 seasonal_periods, smoothing_level, smoothing_slope,
                 smoothing_seasonal, damping_slope, endog, missing,
                 filtered_level, filtered_slope, filtered_seasonal, forecasts):

    # Iterations
    for t in range(nobs_init, nobs_output):
        # Endog component
        endog_t = endog[t - nobs_init]
        if seasonal == 'add':
            endog_adj = endog_t - filtered_seasonal[t - seasonal_periods]
        elif seasonal == 'mul':
            endog_adj = endog_t / filtered_seasonal[t - seasonal_periods]
        else:
            endog_adj = endog_t

        # Intermediate variables
        level_trend = filtered_level[t-1]
        if trend == 'add':
            trend_prev = damping_slope * filtered_slope[t-1]
            level_trend += trend_prev
        elif trend == 'mul':
            trend_prev = filtered_slope[t-1]**damping_slope
            level_trend *= trend_prev

        # Detrended component
        if seasonal == 'add':
            detrended = endog_t - level_trend
        elif seasonal == 'mul':
            detrended = endog_t / level_trend

        # Level
        if not missing[t - nobs_init]:
            filtered_level[t] = (smoothing_level * endog_adj +
                                 (1 - smoothing_level) * level_trend)
        else:
            filtered_level[t] = level_trend

        # Slope
        if trend == 'add':
            filtered_slope[t] = (
                smoothing_slope * (filtered_level[t] - filtered_level[t-1]) +
                (1 - smoothing_slope) * trend_prev)
        elif trend == 'mul':
            filtered_slope[t] = (
                smoothing_slope * (filtered_level[t] / filtered_level[t-1]) +
                (1 - smoothing_slope) * trend_prev)

        # Seasonal
        if seasonal is not None:
            if not missing[t - nobs_init]:
                filtered_seasonal[t] = (
                    smoothing_seasonal * detrended +
                    (1 - smoothing_seasonal) *
                    filtered_seasonal[t - seasonal_periods])
            else:
                filtered_seasonal[t] = filtered_seasonal[t - seasonal_periods]

        # Prediction
        forecasts[t] = level_trend
        if seasonal == 'add':
            forecasts[t] += filtered_seasonal[t - seasonal_periods]
        elif seasonal == 'mul':
            forecasts[t] *= filtered_seasonal[t - seasonal_periods]


class ExponentialSmoothing(tsbase.TimeSeriesModel):
    """
    Exponential Smoothing

    Parameters
    ----------
    endog : array-like
        Time series
    trend : {"add", "mul", "additive", "multiplicative", None}, optional
        Type of trend component.
    damped_trend : bool, optional
        Should the trend component be damped.
    seasonal : {"add", "mul", "additive", "multiplicative", None}, optional
        Type of seasonal component.
    seasonal_periods : int, optional
        The number of seasons to consider for the holt winters.
    initialization: {'estimated', 'heuristic', 'known'}, optional
        How to initialize the recursions. If 'known' initialization is used,
        then `initial_level` must be passed, as well as`initial_slope` and
        `initial_seasonal` if applicable. Default is estimated.
    initial_level : float, optional
        The initial level component. Only used if initialization is 'known'.
    initial_slope : float, optional
        The initial slope component. Only used if initialization is 'known'.
    initial_seasonal : array_like, optional
        The initial seasonal component. An array of length `seasonal_periods`.
        Only used if initialization is 'known'.


    References
    ----------
    [1] Hyndman, Rob, Anne B. Koehler, J. Keith Ord, and Ralph D. Snyder.
        Forecasting with exponential smoothing: the state space approach.
        Springer Science & Business Media, 2008.
    """
    def __init__(self, endog, trend=None, damped_trend=False, seasonal=None,
                 seasonal_periods=None, initialization='estimated',
                 initial_level=None, initial_slope=None, initial_seasonal=None,
                 dates=None, freq=None, missing='none'):

        # Handle alternative specifications
        spec_names = {'additive': 'add', 'add': 'add',
                      'multiplicative': 'mul', 'mul': 'mul'}
        if trend is not None:
            trend = spec_names[trend.lower()]
        if seasonal is not None:
            seasonal = spec_names[seasonal.lower()]
        if seasonal_periods is None:
            seasonal_periods = 0

        # Model parameters
        self.trend = trend
        self.damped_trend = damped_trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.trending = trend in ['mul', 'add']
        self.seasoning = seasonal in ['mul', 'add']
        if initialization is not None:
            initialization = initialization.lower()
        self.initialization = initialization

        # Save specification tuple
        # Note: we do not yet support the full state space specification, so
        # the concept of additive versus multiplicative errors does not yet
        # apply; for this reason, the zeroth element of the tuple, which would
        # usually specify the type of errors, is set to None
        specs = {None: 'N', 'add': 'A', 'mul': 'M'}
        trend_spec = specs[trend] + ('d' if damped_trend else '')
        self.specification = (None, trend_spec, specs[seasonal])

        # Initialization
        self._initial_level = None
        self._initial_slope = None
        self._initial_seasonal = None
        if self.initialization == 'known':
            self.initialize_known(initial_level, initial_slope,
                                  initial_seasonal)

        # Instantiate base class
        super(ExponentialSmoothing, self).__init__(
            endog, None, dates, freq, missing=missing)
        self.nobs = len(self.endog)
        self.missing = np.isnan(self.endog).astype(np.int32)

        # Validation
        if self.endog.ndim > 1:
            raise ValueError
        if ((self.trend == 'mul' or self.seasonal == 'mul') and
                np.any(self.endog[~self.missing.astype(bool)] <= 0.0)):
            raise NotImplementedError('Unable to correct for negative or zero'
                                      ' values.')
        if self.damped_trend and not self.trending:
            raise NotImplementedError('Can only dampen the trend component')
        if self.seasoning and self.seasonal_periods == 0:
            raise NotImplementedError('Unable to detect season automatically;'
                                      ' please specify `seasonal_periods`.')

        # Caches
        self._storage = {}
        self.use_cython = True

    @property
    def k_params(self):
        return (1 + (self.trend is not None) +
                (self.seasonal is not None) + (self.damped_trend))

    @property
    def k_init_params(self):
        if self.initialization == 'estimated':
            return (1 + (self.trend is not None) +
                    (self.seasonal is not None) * (self.seasonal_periods - 1))
        else:
            return 0

    @property
    def k_total_params(self):
        return self.k_params + self.k_init_params

    def initialize_estimated(self):
        self.initialization = 'estimated'
        self._initial_level = None
        self._initial_slope = None
        self._initial_seasonal = None

    def initialize_heuristic(self):
        self.initialization = 'heuristic'
        self._initial_level = None
        self._initial_slope = None
        self._initial_seasonal = None

    def initialize_simple(self):
        simple_specs = [('N', 'N'), ('A', 'N'), ('Ad', 'N'), ('M', 'N'),
                        ('Md', 'N'), ('A', 'A'), ('Ad', 'A'), ('A', 'M'),
                        ('Ad', 'M')]
        if self.specification[1:] not in simple_specs:
            raise NotImplementedError('Simple initialization is not available'
                                      ' for this model; consider using'
                                      ' estimated or heuristic initialization'
                                      ' instead.')

        self.initialization = 'simple'
        self._initial_level = None
        self._initial_slope = None
        self._initial_seasonal = None

    def initialize_known(self, initial_level, initial_slope=None,
                         initial_seasonal=None):
        self.initialization = 'known'
        self._initial_level = float(initial_level)
        if self.trend is not None:
            self._initial_slope = float(initial_slope)
        if self.seasonal is not None:
            self._initial_seasonal = np.asanyarray(initial_seasonal)
            k = len(self._initial_seasonal)
            if k != self.seasonal_periods:
                raise ValueError('Invalid initial seasonal component. Must'
                                 ' contain %d values; contained %d values.'
                                 % (self.seasonal_periods, k))

    def initialization_simple(self):
        # See Section 7.6 of Hyndman and Athanasopoulos
        trend = None
        seasonal = None

        # Non-seasonal
        if self.seasonal is None:
            level = self.endog[0]
            if self.trend == 'add':
                trend = self.endog[1] - self.endog[0]
            elif self.trend == 'mul':
                trend = self.endog[1] / self.endog[0]
        # Seasonal
        else:
            level = np.mean(self.endog[:self.seasonal_periods])
            m = self.seasonal_periods
            if self.trend is not None:
                trend = (pd.Series(self.endog).diff(m)[m:2 * m] / m).mean()

            if self.seasonal == 'add':
                seasonal = self.endog[:m] - level
            elif self.seasonal == 'mul':
                seasonal = self.endog[:m] / level

        return level, trend, seasonal

    def initialization_heuristic(self):
        # See Section 2.6 of Hyndman et al.
        endog = self.endog.copy()

        if self.nobs < 10:
            raise ValueError('Cannot use heuristic method with less than 10'
                             ' observations.')

        # Seasonal component
        seasonal = None
        if self.seasonal is not None:
            # Calculate the number of full cycles to use
            if self.nobs < 2 * self.seasonal_periods:
                raise ValueError('Cannot compute initial seasonals using'
                                 ' heuristic method with less than two full'
                                 ' seasonal cycles in the data.')
            # We need at least 10 periods for the level initialization
            # and we will lose self.seasonal_periods // 2 values at the
            # beginning and end of the sample, so we need at least
            # 10 + 2 * (self.seasonal_periods // 2) values
            min_obs = 10 + 2 * (self.seasonal_periods // 2)
            if self.nobs < min_obs:
                raise ValueError('Cannot use heuristic method to compute'
                                 ' initial seasonal and levels with less'
                                 ' than 10 + 2 * (seasonal_periods // 2)'
                                 ' datapoints.')
            # In some datasets we may only have 2 full cycles (but this may
            # still satisfy the above restriction that we will end up with
            # 10 seasonally adjusted observations)
            k_cycles = min(5, self.nobs // self.seasonal_periods)
            # In other datasets, 3 full cycles may not be enough to end up
            # with 10 seasonally adjusted observations
            k_cycles = max(k_cycles, np.ceil(min_obs / self.seasonal_periods))

            # Compute the moving average
            series = pd.Series(endog[:self.seasonal_periods * k_cycles])
            trend = series.rolling(self.seasonal_periods, center=True).mean()
            if self.seasonal_periods % 2 == 0:
                trend = trend.shift(-1).rolling(2).mean()

            # Detrend
            if self.seasonal == 'add':
                detrended = series - trend
            elif self.seasonal == 'mul':
                detrended = series / trend

            # Average seasonal effect
            seasonal = np.nanmean(
                detrended.values.reshape(k_cycles, self.seasonal_periods).T,
                axis=1)

            # Normalize the seasonals
            if self.seasonal == 'add':
                seasonal -= np.mean(seasonal)
            elif self.seasonal == 'mul':
                seasonal /= np.mean(seasonal)

            # Replace the data with the trend
            endog = trend.dropna().values

        # Trend / Level
        exog = np.c_[np.ones(10), np.arange(10) + 1]
        beta = np.linalg.pinv(exog).dot(endog[:10])
        level = beta[0]

        trend = None
        if self.trend == 'add':
            trend = beta[1]
        elif self.trend == 'mul':
            trend = 1 + beta[1] / beta[0]

        return level, trend, seasonal

    @property
    def param_names(self):
        param_names = ['smoothing_level']
        if self.trend is not None:
            param_names += ['smoothing_slope']
        if self.seasonal is not None:
            param_names += ['smoothing_seasonal']
        if self.damped_trend:
            param_names += ['damping_slope']

        # Initialization
        if self.initialization == 'estimated':
            param_names += ['initial_level']
            if self.trend is not None:
                param_names += ['initial_slope']
            if self.seasonal is not None:
                param_names += ['initial_seasonal.%d' % i
                                for i in range(self.seasonal_periods - 1)]

        return param_names

    @property
    def start_params(self):
        # See Hyndman p.24
        start_params = [0.1]
        if self.trend is not None:
            start_params += [0.01]
        if self.seasonal is not None:
            start_params += [0.01]
        if self.damped_trend:
            start_params += [0.99]

        # Initialization
        if self.initialization == 'estimated':
            init = self.initialization_simple()
            start_params += [init[0]]
            if self.trend is not None:
                start_params += [init[1]]
            if self.seasonal is not None:
                start_params += init[2].tolist()[:-1]

        return np.array(start_params)

    def _filter(self, params):
        # Extract the parameters
        smoothing_level = params[0]
        i = 1
        if self.trend is not None:
            smoothing_slope = params[i]
            i += 1
        else:
            smoothing_slope = 0
        if self.seasonal is not None:
            smoothing_seasonal = params[i]
            i += 1
        else:
            smoothing_seasonal = 1
        if self.damped_trend:
            damping_slope = params[i]
            i += 1
        else:
            damping_slope = 1

        # Output arrays
        nobs_init = max(1, self.seasonal_periods)
        nobs_output = self.nobs + nobs_init
        prefix, dtype, _ = find_best_blas_type((self.endog, params))

        # Initialize the storage arrays
        if prefix not in self._storage:
            self._storage[prefix] = {
                'endog': self.endog.astype(dtype),
                'filtered_level': np.zeros(nobs_output, dtype=dtype),
                'filtered_slope': np.zeros(nobs_output, dtype=dtype),
                'filtered_seasonal': np.zeros(nobs_output, dtype=dtype),
                'forecasts': np.zeros(nobs_output, dtype=dtype)
            }
        else:
            self._storage[prefix]['filtered_level'][:] = 0
            self._storage[prefix]['filtered_slope'][:] = 0
            self._storage[prefix]['filtered_seasonal'][:] = 0
            self._storage[prefix]['forecasts'][:] = 0

        # Retrieve the storage arrays
        endog = self._storage[prefix]['endog']
        filtered_level = self._storage[prefix]['filtered_level']
        filtered_slope = self._storage[prefix]['filtered_slope']
        filtered_seasonal = self._storage[prefix]['filtered_seasonal']
        forecasts = self._storage[prefix]['forecasts']

        # Initialization
        if self.initialization == 'estimated':
            init = [params[i]]
            i += 1
            if self.trend is not None:
                init += [params[i]]
                i += 1
            else:
                init += [None]
            if self.seasonal is not None:
                init_seasonal = params[i:i + self.seasonal_periods - 1]
                if self.seasonal == 'add':
                    init_seasonal = np.r_[init_seasonal, -np.sum(init_seasonal)]
                elif self.seasonal == 'mul':
                    init_seasonal = np.r_[init_seasonal, self.seasonal_periods - np.sum(init_seasonal)]
                init += [init_seasonal.tolist()]
                i += self.seasonal_periods - 1
            else:
                init += [None]
        elif self.initialization == 'simple':
            init = self.initialization_simple()
        elif self.initialization == 'heuristic':
            init = self.initialization_heuristic()
        elif self.initialization == 'known':
            init = (self._initial_level, self._initial_slope,
                    self._initial_seasonal)
        else:
            raise NotImplementedError('Invalid initialization method.')

        filtered_level[nobs_init - 1] = init[0]
        if self.trend is not None:
            filtered_slope[nobs_init-1] = init[1]
        if self.seasonal is not None:
            filtered_seasonal[:self.seasonal_periods] = init[2]

        # Filtering iterations
        if self.use_cython:
            func = prefix_es_filter_map[prefix]
        else:
            func = py_es_filter
        func(self.nobs, nobs_init, nobs_output, self.trend, self.seasonal,
             self.damped_trend, self.seasonal_periods, smoothing_level,
             smoothing_slope, smoothing_seasonal, damping_slope, endog,
             self.missing, filtered_level, filtered_slope, filtered_seasonal,
             forecasts)

        return (filtered_level[nobs_init - 1], filtered_slope[nobs_init - 1],
                filtered_seasonal[:nobs_init],
                filtered_level[nobs_init:], filtered_slope[nobs_init:],
                filtered_seasonal[nobs_init:], forecasts[nobs_init:])

    def filter(self, params, transformed=True, return_raw=False,
               results_class=None, results_wrapper_class=None):
        params = np.array(params, ndmin=1)

        if not transformed:
            params = self.transform_params(params)

        # Save the parameter names
        self.data.param_names = self.param_names

        # Get the result
        names = ['initial_level', 'initial_slope', 'initial_seasonal',
                 'filtered_level', 'filtered_slope', 'filtered_seasonal',
                 'forecasts']
        results = Bunch(**dict(zip(names, self._filter(params))))
        if self.trend is None:
            results.initial_slope = None
            results.filtered_slope = None
        if self.seasonal is None:
            results.initial_seasonal = None
            results.filtered_seasonal = None
        results = FilterResults(self, results)

        # Wrap in a results object
        return self._wrap_results(params, results, return_raw, results_class,
                                  results_wrapper_class)

    @property
    def _res_classes(self):
        return {'fit': (ExponentialSmoothingResults,
                        ExponentialSmoothingResultsWrapper)}

    def _wrap_results(self, params, result, return_raw, results_class=None,
                      wrapper_class=None):
        if not return_raw:
            # Wrap in a results object
            result_kwargs = {}

            if results_class is None:
                results_class = self._res_classes['fit'][0]
            if wrapper_class is None:
                wrapper_class = self._res_classes['fit'][1]

            res = results_class(self, params, result, **result_kwargs)
            result = wrapper_class(res)
        return result

    def transform_params(self, unconstrained):
        unconstrained = np.array(unconstrained, ndmin=1)
        constrained = np.zeros_like(unconstrained)

        # All parameters between 0 and 1
        k = self.k_params
        constrained[:k] = (1 / (1 + np.exp(-unconstrained[:k])))

        # Initial parameters are as-is
        constrained[k:] = unconstrained[k:]

        return constrained

    def untransform_params(self, constrained):
        constrained = np.array(constrained, ndmin=1)
        unconstrained = np.zeros_like(constrained)

        # All parameters between 0 and 1
        k = self.k_params
        unconstrained[:k] = np.log(constrained[0] / (1 - constrained[:k]))

        # Initial parameters are as-is
        unconstrained[k:] = constrained[k:]

        return unconstrained

    def fit(self, start_params=None, transformed=True, method='lbfgs',
            maxiter=50, full_output=1, disp=5, callback=None,
            return_params=False, warn_convergence=True):
        if start_params is None:
            start_params = self.start_params

        # Unconstrain the starting parameters
        if transformed:
            start_params = self.untransform_params(np.array(start_params))

        # Minimize the sum of squared errors
        optimizer = Optimizer()
        fargs = (False,)
        kwargs = {}
        xopt, retvals, optim_settings = optimizer._fit(
            self.sse, self.gradient_sse, start_params, fargs, kwargs,
            hessian=self.hessian_sse, method=method, disp=disp,
            maxiter=maxiter, callback=callback, retall=False,
            full_output=full_output)

        # Check for convergence problems
        if isinstance(retvals, dict):
            if warn_convergence and not retvals['converged']:
                from warnings import warn
                from statsmodels.tools.sm_exceptions import ConvergenceWarning
                warn("Optimization failed to converge. "
                     "Check sse_retvals", ConvergenceWarning)

        # Just return the fitted parameters if requested
        if return_params:
            result = self.transform_params(xopt)
        # Otherwise construct the results class if desired
        else:
            result = self.filter(xopt, transformed=False)

            result.sse_retvals = retvals
            result.sse_settings = optim_settings

        return result

    def gradient_sse(self, params, transformed=True):
        epsilon = _get_epsilon(params, 2., None, len(params))
        return approx_fprime_cs(params, self.sse, epsilon=epsilon,
                                args=(transformed,))

    def hessian_sse(self, params, transformed=True):
        """
        Hessian matrix computed by second-order complex-step differentiation
        on the `sse` function.
        """
        epsilon = _get_epsilon(params, 3., None, len(params))
        hessian = approx_hess_cs(params, self.sse, epsilon=epsilon,
                                 args=(transformed,))

        return hessian / self.nobs

    def sse(self, params, transformed=True):
        if not transformed:
            params = self.transform_params(params)

        filter_results = self.filter(params, return_raw=True)
        sse = np.nansum((self.endog - filter_results.forecasts)**2)

        return sse


class FilterResults(object):
    def __init__(self, model, results):
        # Save model, parameters
        self.model = model
        self.initialization = model.initialization

        # Save filter output
        attributes = ['initial_level', 'initial_slope', 'initial_seasonal',
                      'filtered_level', 'filtered_slope', 'filtered_seasonal',
                      'forecasts']
        for name in attributes:
            val = getattr(results, name)
            # Make copies of arrays, since multiple filter or fit calls will
            # overwrite these
            if not (val is None or np.isscalar(val)):
                val = np.copy(val)
            setattr(self, name, val)

        # Compute additional output
        self.forecasts_error = self.model.endog - self.forecasts


class ExponentialSmoothingResults(tsbase.TimeSeriesModelResults):
    def __init__(self, model, params, results, **kwargs):
        self.data = model.data

        tsbase.TimeSeriesModelResults.__init__(self, model, params,
                                               normalized_cov_params=None,
                                               scale=1.)

        # Save the filter / smoother output
        self.filter_results = results

        # Dimensions
        self.nobs = model.nobs

        # Setup the cache
        self._cache = resettable_cache()

        # Copy over arrays
        self.initial_level = results.initial_level
        self.initial_slope = results.initial_slope
        self.initial_seasonal = results.initial_seasonal

        # Make into Pandas arrays if using Pandas data
        if isinstance(self.data, PandasData):
            ix = self.data.row_labels
            self.level = pd.Series(results.filtered_level, index=ix)
            self.slope = pd.Series(results.filtered_slope, index=ix)
            self.seasonal = pd.Series(results.filtered_seasonal, index=ix)
        else:
            self.level = results.filtered_level
            self.slope = results.filtered_slope
            self.seasonal = results.filtered_seasonal

    @cache_readonly
    def aic(self):
        k = len(self.params)
        return self.nobs * np.log(self.sse / self.nobs) + k * 2

    @cache_readonly
    def aicc(self):
        k = len(self.params)
        return self.aic + (2 * (k + 2) * (k + 3)) / (self.nobs - k - 3)

    @cache_readonly
    def bic(self):
        k = len(self.params)
        return self.nobs * np.log(self.sse / self.nobs) + k * np.log(self.nobs)

    @cache_readonly
    def fittedvalues(self):
        """
        (array) The predicted values of the model. An (nobs,) array.
        """
        return self.filter_results.forecasts

    @cache_readonly
    def sse(self):
        return np.nansum(self.filter_results.forecasts_error**2)

    @cache_readonly
    def resid(self):
        """
        (array) The model residuals. An (nobs,) array.
        """
        return self.filter_results.forecasts_error

    def predict(self, start=None, end=None, dynamic=False, index=None):
        if start is None:
            start = self.model._index[0]

        # Handle start, end, dynamic
        start, end, out_of_sample, prediction_index = (
            self.model._get_prediction_index(start, end, index))

        # Handle `dynamic`
        if isinstance(dynamic, (bytes, unicode)):
            dynamic, _, _ = self.model._get_index_loc(dynamic)
        if dynamic is True:
            dynamic = 0
        if dynamic is False:
            dynamic = None

        # Short-circuit if it's all in-sample prediction
        if out_of_sample == 0 and dynamic is None:
            predictions = self.filter_results.forecasts[start:end].copy()
        # Otherwise, construct a synthetic model
        else:
            if dynamic is not None:
                # Replace the relative dynamic offset with an absolute offset
                dynamic = dynamic + start

                # Replace the end of the sample to be `dynamic`, and extend the
                # number of forecast periods
                out_of_sample = out_of_sample + (end - dynamic)
                end = dynamic

            # Synthetic endog
            endog = np.r_[self.model.endog[start:end + 1],
                          [np.nan] * out_of_sample]

            initial_slope = None
            initial_seasonal = None
            if start < 1:
                initial_level = self.initial_level
                if self.model.trend is not None:
                    initial_slope = self.initial_slope
            else:
                initial_level = self.level[start - 1]
                if self.model.trend is not None:
                    initial_slope = self.slope[start - 1]

            if start < self.model.seasonal_periods:
                initial_seasonal = np.r_[self.initial_seasonal[start:],
                                         self.seasonal[:start]]
            else:
                initial_seasonal = (self.seasonal[start -
                                    self.model.seasonal_periods:start])

            mod = ExponentialSmoothing(
                endog, trend=self.model.trend,
                damped_trend=self.model.damped_trend,
                seasonal=self.model.seasonal,
                seasonal_periods=self.model.seasonal_periods)

            mod.initialize_known(initial_level, initial_slope,
                                 initial_seasonal)
            res = mod.filter(self.params, return_raw=True)
            predictions = res.forecasts

        return predictions

    def forecast(self, steps=1):
        if isinstance(steps, (int, long)):
            end = self.nobs + steps - 1
        else:
            end = steps
        return self.predict(start=self.nobs, end=end)

    def summary(self):
        """
        Summarize the Model

        Parameters
        ----------
        alpha : float, optional
            Significance level for the confidence intervals. Default is 0.05.
        start : int, optional
            Integer of the start observation. Default is 0.
        model_name : string
            The name of the model used. Default is to use model class name.

        Returns
        -------
        summary : Summary instance
            This holds the summary table and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary
        """
        from statsmodels.iolib.summary import Summary, forg
        from statsmodels.iolib.table import SimpleTable
        from statsmodels.iolib.tableformatting import fmt_params

        # Model specification results
        model = self.model
        title = 'Exponential Smoothing Results'

        start = 0
        if self.model._index_dates:
            ix = self.model._index
            d = ix[start]
            sample = ['%02d-%02d-%02d' % (d.month, d.day, d.year)]
            d = ix[-1]
            sample += ['- ' + '%02d-%02d-%02d' % (d.month, d.day, d.year)]
        else:
            sample = [str(start), ' - ' + str(self.nobs)]

        # Standardize the model name as a list of str
        model_name = ['(' + ', '.join(self.model.specification[1:]) + ')']

        # Create the tables
        if not isinstance(model_name, list):
            model_name = [model_name]

        top_left = [('Dep. Variable:', None)]
        top_left.append(('Model:', [model_name[0]]))
        for i in range(1, len(model_name)):
            top_left.append(('', ['+ ' + model_name[i]]))
        top_left += [
            ('Date:', None),
            ('Time:', None),
            ('Sample:', [sample[0]]),
            ('', [sample[1]])
        ]

        top_right = [
            ('No. Observations:', [self.nobs]),
            ('SSE', ["%#5.3f" % self.sse]),
            ('AIC', ["%#5.3f" % self.aic]),
            ('BIC', ["%#5.3f" % self.bic]),
            ('AICC', ["%#5.3f" % self.aicc])
        ]

        format_str = lambda array: [
            ', '.join(['{0:.2f}'.format(i) for i in array])
        ]

        summary = Summary()
        summary.add_table_2cols(self, gleft=top_left, gright=top_right,
                                title=title)
        if len(self.params) > 0:
            params = self.params
            param_header = ['coef']
            params_stubs = self.data.param_names
            params_data = [[forg(params[i], prec=4)] for i in range(len(params))]

            parameter_table = SimpleTable(params_data,
                                          param_header,
                                          params_stubs,
                                          txt_fmt=fmt_params)
            summary.tables.append(parameter_table)

        # Add warnings/notes, added to text format only
        etext = []
        if etext:
            etext = ["[{0}] {1}".format(i + 1, text)
                     for i, text in enumerate(etext)]
            etext.insert(0, "Warnings:")
            summary.add_extra_txt(etext)

        return summary


class ExponentialSmoothingResultsWrapper(wrap.ResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(tsbase.TimeSeriesResultsWrapper._wrap_attrs,
                                   _attrs)
    _methods = {
        'forecast': 'dates',
    }
    _wrap_methods = wrap.union_dicts(
        tsbase.TimeSeriesResultsWrapper._wrap_methods, _methods)
wrap.populate_wrapper(ExponentialSmoothingResultsWrapper,
                      ExponentialSmoothingResults)
