"""
Trigonometric, Box-Cox, ARMA Error, Trend and Seasonal (TBATS) Model

Author: Leoyzen Liu
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

from warnings import warn
from statsmodels.compat.collections import OrderedDict

import numpy as np
from numpy.linalg import LinAlgError
from scipy import linalg
import pandas as pd
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.eval_measures import aic
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.statespace.kalman_filter import INVERT_UNIVARIATE, SOLVE_LU
from statsmodels.tsa.statespace.mlemodel import MLEResults, MLEResultsWrapper
import statsmodels.base.wrapper as wrap
from statsmodels.tools.tools import Bunch, add_constant
from statsmodels.tools.decorators import cache_readonly
from statsmodels.base.transform import BoxCox
from scipy.stats import linregress
from statsmodels.tsa.statespace.tools import (
    constrain_stationary_univariate,
    unconstrain_stationary_univariate,
    solve_discrete_lyapunov,
    companion_matrix
)
from statsmodels.tsa.stattools import _safe_arma_fit

from .tools import fourier
from .innovation import InnnovationModel


class PeriodWrapper(object):
    def __init__(self, name, attribute, dtype=np.int, multiple=False):
        self.name = name
        self.attribute = attribute
        self._attribute = '_' + attribute
        self._multiple = multiple
        self.dtype = dtype

    def __set__(self, obs, val):
        if val is None:
            k_period = 0
        elif np.isscalar(val):
            if self._multiple:
                val = np.asarray([val], dtype=self.dtype)
            else:
                val = self.dtype(val)
            k_period = 1
        else:
            if self._multiple:
                val = np.asarray(val, dtype=self.dtype)
            else:
                raise ValueError("%s must be a scalar of %s" % (
                    self.name,
                    self.dtype
                ))
            k_period = len(val)
        setattr(obs, 'k_period', k_period)
        setattr(obs, self._attribute, val)
        setattr(obs, 'period_names', ['Seasonal'] * k_period)

    def __get__(self, obj, objtype):
        return getattr(obj, self._attribute, None)


def _tbats_seasonal_matrix(periods, k):
    n_period = len(periods)
    n_k = len(k)
    tau = 2 * np.sum(k)
    if n_k != n_period:
        raise ValueError
    design = []
    transition = []
    for i, (m, k) in enumerate(zip(periods, k)):
        design = np.append(design, np.r_[np.ones(k), np.zeros(k)])
        lmbda = [2 * np.pi * (j + 1) / m for j in range(k)]
        c = np.diag(np.cos(lmbda))
        s = np.diag(np.sin(lmbda))
        A = np.bmat([[c, s], [-s, c]])
        transition.append(A)
    return tau, design, linalg.block_diag(*transition).astype(np.float32)


def _bats_seasonal_matrix(periods):
    periods = np.asarray(periods, dtype='int')
    design = []
    transition = []
    tau = np.sum(periods)
    for m in periods:
        design = np.append(design, np.r_[np.zeros(m - 1), 1])
        A = np.transpose(companion_matrix(int(m)))
        A[0, -1] = 1
        transition.append(A)
    return tau, design, linalg.block_diag(*transition).astype('int')


def _safe_tbats_fit(endog, model_kw, fit_kw, **kwargs):
    try:
        if True:
            res = TBATSModel(endog, **model_kw).fit(**fit_kw)
            return res
    except LinAlgError:
        # SVD convergence failure on badly misspecified models
        return
    except Exception as e:  # no idea what happened
        return


def _k_f_test(x, periods, ks, exog=None):
    for i, (period, k) in enumerate(zip(periods, ks)):
        if exog is None:
            exog = fourier(x, period, k, name='Seasonal' + str(i))
        else:
            exog = exog.join(
                fourier(x, period, k, name='Seasonal' + str(i)),
                how='outer'
            )
    res = OLS(x, add_constant(exog)).fit(cov_kwd={'use_t': True})
    return res


def seasonal_fourier_k_select_ic(y, periods, ic='aic', grid_search=False):
    periods = np.asarray(periods)
    ks = np.ones_like(periods, dtype=int)
    _ic = lambda obj: getattr(obj, ic)
    result = dict()
    max_k = np.ones_like(ks)

    for i, period in enumerate(periods):

        max_k[i] = period // 2
        if i != 0:
            current_k = 2
            while current_k <= max_k[i]:
                if period % current_k != 0:
                    current_k += 1
                    continue

                latter = period / current_k

                if np.any(periods[: i] % latter == 0):
                    max_k[i] = current_k - 1
                    break
                else:
                    current_k += 1

    for i, period in enumerate(periods):

        current_k = 2
        min_ic = np.inf
        while current_k <= max_k[i]:
            _ks = ks.copy()
            _ks[i] = current_k
            try:
                test = _k_f_test(y, periods, _ks)
                result.update({str(tuple(_ks.tolist())): _ic(test)})
                if test.f_pvalue > .001:
                    current_k += 1
                    continue
                if _ic(test) <= min_ic:
                    min_k = current_k
                    ks[i] = min_k
                    min_ic = _ic(test)
                elif not grid_search:
                    break
            except (ValueError, LinAlgError) as e:
                break
            finally:
                current_k += 1

    return ks


def tbats_k_order_select_ic(y, periods, use_trend=None, use_box_cox=None, damped_trend=None,
                            max_ar=3, max_ma=2, ic='aic', model='additive', **kwargs):
    from functools import partial
    from statsmodels.tsa.stattools import arma_order_select_ic
    kwargs.setdefault('model_kw', {})
    kwargs.setdefault('fit_kw', {})
    kwargs['model_kw'].setdefault('use_trend', False if use_trend is None else use_trend)
    kwargs['model_kw'].setdefault('damped_trend', False if damped_trend is None else damped_trend)
    kwargs['model_kw'].setdefault('box_cox', False if use_box_cox is None else use_box_cox)
    kwargs['model_kw'].setdefault('order', (0, 0))
    grid_search = kwargs.pop('grid_search', False)

    periods = np.asarray(periods)
    periods.sort()
    n_periods = periods.size

    ks = np.ones(n_periods, dtype=int)
    data = np.asarray(y)

    model_kw = kwargs.get('model_kw')
    fit_kw = kwargs.get('fit_kw')

    model_kw['periods'] = periods
    model_kw['k'] = ks

    calc_aic = partial(_safe_tbats_fit, y, fit_kw=fit_kw)

    min_ic = np.inf
    best_model = None
    _ic = lambda obj: getattr(obj, ic)

    if np.any(data <= 0) or use_box_cox is not True:
        model_kw['box_cox'] = False
        y2 = data.copy()
    elif use_box_cox is True:
        y2 = np.log(data)
        model_kw['box_cox'] = True

    resid, trend = hpfilter(y2, lamb=(2 * np.sin(np.pi / max(periods))) ** (-4))
    if model == 'multiplicative':
        resid = y2 / trend

    min_ic = np.inf

    ks = seasonal_fourier_k_select_ic(resid, periods, grid_search=grid_search, ic=ic)

    model_kw.update(k=ks)
    best_model = calc_aic(model_kw=model_kw)

    # Use Trend
    if use_trend is None:
        model_kw['use_trend'] = True
        mod = calc_aic(model_kw=model_kw)
        if mod and _ic(mod) < min_ic:
            model_kw['use_trend'] = True
            best_model = mod
            min_ic = _ic(mod)
            if damped_trend is None:
                model_kw['damped_trend'] = True
                mod = calc_aic(model_kw=model_kw)
                if mod and _ic(mod) < min_ic:
                    model_kw['damped_trend'] = True
                    best_model = mod
                    min_ic = _ic(mod)
                else:
                    model_kw['damped_trend'] = False
    elif use_trend is True:
        model_kw['use_trend'] = True
        mod = calc_aic(model_kw=model_kw)
        if mod:
            best_model = mod
            min_ic = _ic(mod)
            if damped_trend is None:
                model_kw['damped_trend'] = True
                mod = calc_aic(model_kw=model_kw)
                if mod and _ic(mod) < min_ic:
                    model_kw['damped_trend'] = True
                    best_model = mod
                    min_ic = _ic(mod)
                else:
                    model_kw['damped_trend'] = False

    resid = best_model.resid
    order = arma_order_select_ic(resid, max_ar=max_ar, max_ma=max_ma, ic='aic').aic_min_order
    model_kw['order'] = order
    mod = calc_aic(model_kw=model_kw)
    if mod and _ic(mod) < min_ic:
        best_model = mod
        min_ic = _ic(mod)
    else:
        model_kw['order'] = (0, 0)

    return model_kw, best_model


class TBATSModel(InnnovationModel, BoxCox):
    r"""
    Trigonometric, Box-Cox, ARMA Error, Trend and Seasonal(TBATS)

    a model of Trigonometric exponential smoothing models for seasonal data

    Parameters
    ----------
    periods:   iterable of float or None or array, optional
        The period of the seasonal component. Default is None. Can be multiple
    k:  int or None, optional, should be same length of periods
    order:  iterable, optional
        The (p,q) order of the model for the number of AR parameters, and MA parameters of error.
    period_names:    iterable or None, optional
        The names used for periods, should be the same length of periods

    """

    periods = PeriodWrapper('Seasonal Periods', 'periods', np.float, multiple=True)

    def __init__(self, endog, periods=None, k=None, order=(0, 0), period_names=None,
                 use_trend=True, damped_trend=True, box_cox=True, exog=None, mle_regression=True,
                 **kwargs):

        # Model options
        self.seasonal = periods is not None
        self.trend = use_trend
        self.boxcox = box_cox
        self.damping = damped_trend
        self.mle_regression = mle_regression
        self.autoregressive = (order != (0, 0))
        self.periods = periods
        self.k = k
        self.order = order
        k_states = 1

        # Exogenous component
        self.k_exog = 0
        if exog is not None:
            exog_is_using_pandas = _is_using_pandas(exog, None)
            if not exog_is_using_pandas:
                exog = np.asarray(exog)

            # Make sure we have 2-dimensional array
            if exog.ndim == 1:
                if not exog_is_using_pandas:
                    exog = exog[:, None]
                else:
                    exog = pd.DataFrame(exog)

            self.k_exog = exog.shape[1]
        self.regression = self.k_exog > 0

        if periods is not None:
            self.seasonal = True
            self.k = k
            k_states += int(2 * np.sum(self.k))
            self.max_period = max(self.periods)
            if period_names is not None:
                self.period_names = period_names
        else:
            self.seasonal = False
            self.max_period = 0
            self.periods = (0,)
            self.k = (0,)

        self.trend = use_trend & True
        if self.trend:
            k_states += 1

        self.autoregressive = np.sum(order) != 0
        self.order = order

        p, q = self.order

        if self.autoregressive:
            k_states += p + q

        k_states = (
            1 + self.trend +
            np.sum(self.k) * 2 +
            np.sum(self.order) +
            (not self.mle_regression) * self.k_exog
        )
        k_posdef = 1

        loglikelihood_burn = kwargs.get('loglikelihood_burn', int(self.max_period) + p + q)

        # Initialize the model base
        super(TBATSModel, self).__init__(endog=endog, exog=exog, k_states=k_states, k_posdef=k_posdef,
                                         initialization='approximate_diffuse', **kwargs)

        if box_cox:
            if np.any(self.data.endog <= 0):
                warn("To use boxcox transformation the endog must be positive")
                self.boxcox = False
            else:
                self.data.log_endog = np.log(self.data.endog)
        else:
            self.boxcox = False

        self.ssm.loglikelihood_burn = loglikelihood_burn
        self.setup()

    def setup(self):

        self.parameters = OrderedDict()
        self.parameters_obs_intercept = OrderedDict()

        # Initialize the fixed components of the state space matrices,
        i = 0  # state offset
        j = 0  # state covariance offset

        offset = 0

        self.parameters['obs_var'] = 1

        if self.boxcox:
            self.parameters['boxcox'] = 1

        # Level Setup
        self.parameters['level_alpha'] = 1
        self['design', 0, offset] = 1
        self['transition', 0, 0] = 1
        offset += 1

        # Trend Setup
        if self.trend:
            self.parameters['trend_beta'] = 1
            self['design', 0, offset] = 1
            offset += 1
            if self.damping:
                self.parameters['damping'] = 1

        if self.seasonal:
            tau, season_design, season_transition = _tbats_seasonal_matrix(self.periods, self.k)
            self['design', 0, offset: offset + tau] = season_design
            self['transition', offset: offset + tau, offset: offset + tau] = season_transition
            self.parameters['seasonal_gamma'] = 2 * self.k_period
            offset += tau

        if self.autoregressive:
            p, q = self.order
            start = 1 + self.trend
            if self.seasonal:
                start += tau
            if p:
                if p > 1:
                    np.fill_diagonal(self['transition', start + 1: start + p, start: start + p], 1)
                self['selection', start, 0] = 1
                self.parameters['ar_coef'] = p
                start += p
            if q:
                if q > 1:
                    np.fill_diagonal(self['transition', start + 1: start + q, start: start + q], 1)
                self['selection', start, 0] = 1
                self.parameters['ma_coef'] = q

        if self.regression:
            if self.mle_regression:
                self.parameters_obs_intercept['reg_coeff'] = self.k_exog
            else:
                design = np.repeat(self.ssm['design', :, :, 0], self.nobs, axis=0)
                self.ssm['design'] = design.transpose()[np.newaxis, :, :]
                self.ssm['design', 0, offset:offset + self.k_exog, :] = self.exog.transpose()
                self.ssm['transition', offset:offset + self.k_exog, offset:offset + self.k_exog] = (
                    np.eye(self.k_exog)
                )

                offset += self.k_exog

        self.parameters.update(self.parameters_obs_intercept)

        self.k_obs_intercept = sum(self.parameters_obs_intercept.values())
        self.k_params = sum(self.parameters.values())

    def initialize_state(self, lmbda=None):
        initial_state = np.zeros(self.k_states, dtype=self.ssm.dtype)
        initial_state_cov = (
            np.eye(self.k_states, dtype=self.ssm.dtype) *
            self.ssm.initial_variance
        )

        if self.boxcox and lmbda is not None:
            y = self.transform_boxcox(self.data.endog, lmbda=lmbda)
        else:
            y = self.data.endog

        if self.seasonal:
            max_period = max(self.periods)
            resid, trend = hpfilter(y, lamb=(2 * np.sin(np.pi / max_period)) ** (-4))
        else:
            resid, trend = hpfilter(y)

        slope, intercept, r_value, p_value, std_err = linregress(np.arange(trend.shape[0]), trend)

        offset = 0
        initial_state[offset] = intercept if self.trend else trend
        offset += 1

        if self.trend:
            initial_state[offset] = slope
            offset += 1

        if self.seasonal:
            ols_mod = _k_f_test(resid, self.periods, self.k)
            tau = 2 * np.sum(self.k)
            initial_state[offset: offset + tau] = ols_mod.params[1:]
            offset += tau
            resid = ols_mod.resid

        if self.autoregressive:
            start = offset
            end = start + np.sum(self.order)
            selection_stationary = self['selection', start:end, :]
            selected_state_cov_stationary = np.dot(
                np.dot(selection_stationary, self['state_cov']),
                selection_stationary.T
            )
            initial_state_cov_stationary = solve_discrete_lyapunov(
                self['transition', start:end, start:end],
                selected_state_cov_stationary
            )
            initial_state_cov[start:end, start:end] = (
                initial_state_cov_stationary
            )
        self.initialize_known(initial_state, initial_state_cov)

    def filter(self, params, **kwargs):
        kwargs.setdefault('results_class', TBATSResults)
        kwargs.setdefault('results_wrapper_class',
                          TBATSResultsWrapper)
        return super(TBATSModel, self).filter(params, **kwargs)

    def smooth(self, params, **kwargs):
        kwargs.setdefault('results_class', TBATSResults)
        kwargs.setdefault('results_wrapper_class',
                          TBATSResultsWrapper)
        return super(TBATSModel, self).smooth(params, **kwargs)

    def loglike(self, params, *args, **kwargs):
        """
        Loglikelihood evaluation

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the loglikelihood
            function.
        **kwargs
            Additional keyword arguments to pass to the Kalman filter. See
            `KalmanFilter.filter` for more details.

        Notes
        -----
        [1]_ recommend maximizing the average likelihood to avoid scale issues;
        this is done automatically by the base Model fit method.

        References
        ----------
        .. [1] Koopman, Siem Jan, Neil Shephard, and Jurgen A. Doornik. 1999.
           Statistical Algorithms for Models in State Space Using SsfPack 2.2.
           Econometrics Journal 2 (1): 107-60. doi:10.1111/1368-423X.00023.

        See Also
        --------
        update : modifies the internal state of the state space model to
                 reflect new params
        """
        # We need to handle positional arguments in two ways, in case this was
        # called by a Scipy optimization routine
        if len(args) > 0:
            argnames = ['transformed', 'complex_step']
            # the fit() method will pass a dictionary
            if isinstance(args[0], dict):
                flags = args[0]
            # otherwise, a user may have just used positional arguments...
            else:
                flags = dict(zip(argnames, args))
            transformed = flags.get('transformed', True)
            complex_step = flags.get('complex_step', True)

            for name, value in flags.items():
                if name in kwargs:
                    raise TypeError("loglike() got multiple values for keyword"
                                    " argument '%s'" % name)
        else:
            transformed = kwargs.pop('transformed', True)
            complex_step = kwargs.pop('complex_step', True)

        if not transformed:
            params = self.transform_params(params)

        self.update(params, transformed=True, complex_step=complex_step)

        if complex_step:
            kwargs['inversion_method'] = INVERT_UNIVARIATE | SOLVE_LU

        sse = self.ssm.loglike(complex_step=complex_step, **kwargs)

        lmbda = params[1]
        if self.boxcox:
            sse += (lmbda - 1) * np.nansum(self.data.log_endog[self.loglikelihood_burn:])

        return sse

    def loglikeobs(self, params, transformed=True, complex_step=False,
                   **kwargs):
        params = np.asarray(params, dtype=np.float64)
        if transformed is False:
            params = self.transform_params(params)
            transformed = True

        kwargs.setdefault('loglikelihood_burn', self.loglikelihood_burn)

        loglikelihood_burn = kwargs.get('loglikelihood_burn')

        loglikeobs = super().loglikeobs(params, transformed, complex_step, **kwargs)

        if self.boxcox:
            lmbda = params[1]
            loglikeobs += (lmbda - 1) * self.data.log_endog

        loglikeobs[:loglikelihood_burn] = 0

        return loglikeobs

    def fit(self, *args, **kwargs):
        kwargs.setdefault('maxiter', (self.k_params * 200) ** 2)
        return super().fit(*args, **kwargs)

    @property
    def start_params(self):

        if not hasattr(self, 'parameters'):
            return []

        _start_params = {}

        endog = self.endog
        exog = self.exog

        _start_params['obs_var'] = .001

        if np.any(np.isnan(endog)):
            endog = endog[~np.isnan(endog)]
            if exog is not None:
                exog = exog[~np.isnan(endog)]

        if self.boxcox:
            endog, lmbda = self.transform_boxcox(self.data.endog)
            if np.isnan(lmbda) or not (0 <= lmbda <= 1):
                _start_params['boxcox'] = .975
                endog = self.transform_boxcox(endog, .975)
            else:
                _start_params['boxcox'] = lmbda

        _start_params['level_alpha'] = .1

        # trend beta
        if self.trend:
            _start_params['trend_beta'] = .001
            if self.damping:
                _start_params['damping'] = .999

        # seasonal gammas
        if self.seasonal:
            n = self.k_period
            _start_params['seasonal_gamma'] = [.01] * 2 * n
            resid, trend = hpfilter(endog, lamb=(2 * np.sin(np.pi / max(self.periods))) ** (-4))
        else:
            resid, trend = hpfilter(endog)

        # Regression
        if self.regression and self.mle_regression:
            _start_params['reg_coeff'] = (
                np.linalg.pinv(self.exog).dot(resid).tolist()
            )
            resid = np.squeeze(
                resid - np.dot(self.exog, _start_params['reg_coeff'])
            )

        # AR
        if self.autoregressive:
            p, q = self.order
            mod = _safe_arma_fit(resid, order=self.order, trend='c', model_kw={'dates': self.data.dates}, fit_kw={})
            if p:
                if mod:
                    _start_params['ar_coef'] = mod.arparams
                else:
                    _start_params['ar_coef'] = [.001] * p

            # MA
            if q:
                if mod:
                    _start_params['ma_coef'] = mod.maparams
                else:
                    _start_params['ma_coef'] = [.001] * q

        start_params = []
        for key in self.parameters.keys():
            if np.isscalar(_start_params[key]):
                start_params.append(_start_params[key])
            else:
                start_params.extend(_start_params[key])
        return start_params

    @property
    def param_names(self):

        if not hasattr(self, 'parameters'):
            return []
        param_names = []
        p, q = self.order
        for key in self.parameters.keys():
            if key == 'obs_var':
                param_names.append('sigma2.var')
            elif key == 'level_alpha':
                param_names.append('alpha.level')
            elif key == 'trend_beta':
                param_names.append('beta.trend')
            elif key == 'damping':
                param_names.append('phi.damping')
            elif key == 'seasonal_gamma':
                for i, name in enumerate(self.period_names, 1):
                    param_names.append('gamma1.{}'.format(name))
                    param_names.append('gamma2.{}'.format(name))
            elif key == 'ar_coef':
                for i in range(p):
                    param_names.append('ar.L%d.coefs' % (i + 1))
            elif key == 'boxcox':
                param_names.append('lambda.boxcox')
            elif key == 'ma_coef':
                for i in range(q):
                    param_names.append('ma.L%d.coefs' % (i + 1))
            elif key == 'reg_coeff':
                param_names += [
                    'beta.%s.coefs' % self.exog_names[i]
                    for i in range(self.k_exog)
                    ]
            else:
                param_names.append(key)
        return param_names

    def transform_params(self, unconstrained):
        unconstrained = np.array(unconstrained, ndmin=1, dtype=np.float)
        constrained = np.zeros_like(unconstrained)

        offset = 0

        # Positive parameters: obs_cov, state_cov
        constrained[offset] = unconstrained[offset] ** 2
        offset += 1

        if self.boxcox:
            constrained[offset] = (
                1 / (1 + np.exp(-unconstrained[offset]))
            )
            offset += 1

        # Level Alpha
        alpha = (
            1 / (1 + np.exp(-unconstrained[offset]))
        )
        constrained[offset] = alpha
        offset += 1

        if self.trend:
            constrained[offset] = (
                                      1 / (1 + np.exp(-unconstrained[offset]))
                                  ) * alpha
            offset += 1

            if self.damping:
                # should between .8 ~ 1
                # low, high = 0, 1
                constrained[offset] = (
                    1 / (1 + np.exp(-unconstrained[offset]))
                )
                offset += 1

        if self.seasonal:

            max_limit = 1 - alpha

            for period, k in zip(self.periods, self.k):
                gamma1, gamma2 = unconstrained[offset: offset + 2]
                low, high = 0, max_limit / k
                gamma_sum = gamma1 + gamma2
                limit = (
                            1 / (1 + np.exp(-gamma_sum))
                        ) * (high - low) + low

                max_limit -= limit * k
                constrained[offset: offset + 2] = np.r_[gamma1 * limit / gamma_sum, gamma2 * limit / gamma_sum]
                offset += 2

        if self.autoregressive:
            p, q = self.order
            if p > 0:
                constrained[offset:offset + p] = constrain_stationary_univariate(
                    unconstrained[offset: offset + p]
                )
                offset += p

            if q > 0:
                constrained[offset:offset + q] = constrain_stationary_univariate(
                    unconstrained[offset: offset + q]
                )
                offset += q

        constrained[offset:] = unconstrained[offset:]
        return constrained

    def untransform_params(self, constrained):
        constrained = np.array(constrained, ndmin=1, dtype=np.float)
        unconstrained = np.zeros_like(constrained)

        offset = 0

        # Positive parameters: obs_cov, state_cov
        unconstrained[0] = constrained[0] ** 0.5
        offset += 1

        if self.boxcox:
            unconstrained[offset] = np.log(
                constrained[offset] / (1 - constrained[offset])
            )
            offset += 1

        # alpha
        alpha = constrained[offset]
        unconstrained[offset] = np.log(
            constrained[offset] / (1 - constrained[offset])
        )
        offset += 1

        if self.trend:
            x = constrained[offset] / alpha
            unconstrained[offset] = np.log(
                x / (1 - x)
            )
            offset += 1

            if self.damping:
                x = (constrained[offset] - 0) / 1
                unconstrained[offset] = np.log(
                    x / (1 - x)
                )
                offset += 1

        if self.seasonal:

            max_limit = 1 - alpha

            for period, k in zip(self.periods, self.k):
                gamma1, gamma2 = constrained[offset: offset + 2]
                low, high = 0, max_limit / k
                gamma_sum = gamma1 + gamma2
                x = (gamma_sum - low) / (high - low)
                limit = np.log(
                    x / (1 - x)
                )

                max_limit -= limit * k
                unconstrained[offset: offset + 2] = np.r_[gamma1 * limit / gamma_sum, gamma2 * limit / gamma_sum]
                offset += 2

        if self.autoregressive:
            p, q = self.order
            if p > 0:
                unconstrained[offset:offset + p] = unconstrain_stationary_univariate(
                    constrained[offset: offset + p]
                )
                offset += p

            if q > 0:
                unconstrained[offset:offset + q] = unconstrain_stationary_univariate(
                    constrained[offset: offset + q]
                )

                offset += q
        unconstrained[offset:] = constrained[offset:]
        return unconstrained

    def update(self, params, transformed=True, complex_step=False):
        params = super().update(params, transformed, complex_step)
        params = np.asarray(params, dtype=np.float)

        offset = matrix_offset = 0
        s_dtype = self['selection'].dtype

        # cov
        cov = params[0]
        self['state_cov', 0, 0] = cov
        offset += 1

        if self.boxcox:
            lmbda = params[offset].astype(np.float)
            offset += 1
            endog_touse = self.transform_boxcox(self.endog, lmbda)
            self.ssm.bind(endog_touse)
            self.ssm._representations = {}
            self.ssm._statespaces = {}
        else:
            lmbda = None

        # level alpha
        alpha = params[offset]
        offset += 1
        self['selection', matrix_offset, 0] = alpha
        matrix_offset += 1

        if self.trend:
            beta = params[offset]
            offset += 1

            self['selection', matrix_offset, 0] = beta
            matrix_offset += 1

            # damped
            if self.damping:
                damped = params[offset]
                offset += 1

                self['transition', :2, 1] = damped
                self['design', 0, 1] = damped

        # Seasonal gamma
        if self.seasonal:
            n = self.k_period
            tau = int(2 * np.sum(self.k))
            gamma = params[offset: offset + 2 * n]
            offset += 2 * n

            j = 0
            gamma_selection = np.zeros(tau)
            gamma_design = np.zeros(tau)
            for i, k in enumerate(self.k, 1):
                gamma_selection[j: j + 2 * k] = np.r_[
                    np.full(k, gamma[2 * i - 2], dtype=s_dtype),
                    np.full(k, gamma[2 * i - 1], dtype=s_dtype)
                ]
                gamma_design[j: j + 2 * k] = np.r_[
                    np.ones(k),
                    np.zeros(k)
                ]
                j += 2 * k

            self['selection', matrix_offset: matrix_offset + tau, 0] = gamma_selection
            self['design', 0, matrix_offset: matrix_offset + tau] = gamma_design

            matrix_offset += tau

        if self.autoregressive:
            p, q = self.order
            col_indice = matrix_offset
            # AR
            if p:
                ar = params[offset: offset + p]

                col_indice = matrix_offset
                offset += p
                self['design', 0, matrix_offset: matrix_offset + p] = ar
                self['transition', 0, matrix_offset: matrix_offset + p] = alpha * ar
                if self.trend:
                    self['transition', 1, matrix_offset: matrix_offset + p] = beta * ar
                if self.seasonal:
                    self[
                    'transition',
                    1 + self.trend: col_indice,
                    matrix_offset: matrix_offset + p
                    ] = gamma_selection[:, None] * ar
                self['transition', matrix_offset, matrix_offset: matrix_offset + p] = ar
                matrix_offset += p

            # MA
            if q:
                ma = params[offset: offset + q]

                self['design', 0, -q:] = ma
                self['transition', 0, -q:] = alpha * ma
                if self.trend:
                    self['transition', 1, -q:] = beta * ma
                if self.seasonal:
                    self['transition', 1 + self.trend: col_indice, -q:] = gamma_selection[:, None] * ma
                self['transition', col_indice + p, -q:] = ma

                offset += q

        if self.regression and self.mle_regression and True:
            self.ssm['obs_intercept'] = np.dot(
                self.data.exog,
                params[offset: offset + self.k_exog]
            )[None, :]
            offset += self.k_exog

        self.initialize_state(lmbda=lmbda)


class TBATSResults(MLEResults):
    """
    Class to hold results from fitting an unobserved components model.

    Parameters
    ----------
    model : UnobservedComponents instance
        The fitted model instance

    Attributes
    ----------
    specification : dictionary
        Dictionary including all attributes from the unobserved components
        model instance.

    See Also
    --------
    statsmodels.tsa.statespace.kalman_filter.FilterResults
    statsmodels.tsa.statespace.mlemodel.MLEResults
    """

    def __init__(self, model, params, filter_results, cov_type='opg',
                 **kwargs):
        super(TBATSResults, self).__init__(
            model, params, filter_results, cov_type, **kwargs)

        self.df_resid = np.inf  # attribute required for wald tests

        # Save _init_kwds
        self._init_kwds = self.model._get_init_kwds()

        self.specification = Bunch(**{
            # Model options
            'level': True,
            'trend': self.model.trend,
            'seasonal': self.model.seasonal,
            'arma_order': self.model.order,
            'box_cox': self.model.boxcox,
            'damped_trend': self.model.damping,
            'autoregressive': self.model.autoregressive
        })

    @cache_readonly
    def fittedvalues(self):
        fittedvalues = super(TBATSResults, self).fittedvalues
        if self.model.boxcox:
            fittedvalues = self.model.untransform_boxcox(fittedvalues, float(self.params[1]))
        return fittedvalues

    @cache_readonly
    def llf(self):
        """
        (float) The value of the log-likelihood function evaluated at `params`.
        """
        return self.model.loglike(self.params)

    @property
    def level(self):
        offset = 0
        out = Bunch(filtered=self.filtered_state[offset],
                    filtered_cov=self.filtered_state_cov[offset, offset],
                    smoothed=None, smoothed_cov=None,
                    offset=0)
        if self.smoothed_state is not None:
            out.smoothed = self.smoothed_state[offset]
        if self.smoothed_state_cov is not None:
            out.smoothed_cov = self.smoothed_state_cov[offset, offset]

        return out

    @cache_readonly
    def resid(self):
        """
        (array) The model residuals. An (nobs x k_endog) array.
        """
        # This is a (k_endog x nobs array; don't want to squeeze in case of
        # the corner case where nobs = 1 (mostly a concern in the predict or
        # forecast functions, but here also to maintain consistency)
        resid = self.data.endog - self.fittedvalues
        if resid.shape[0] == 1:
            resid = resid[0, :]
        else:
            resid = resid.T
        return resid

    @cache_readonly
    def aic(self):
        return aic(self.llf, self.nobs, self.params.nonzero()[0].shape[0])

    @cache_readonly
    def mape(self):
        return np.nanmean(np.abs(self.resid / self.fittedvalues)[self.loglikelihood_burn:])

    @cache_readonly
    def mae(self):
        return np.nanmean(np.abs(self.resid)[self.loglikelihood_burn:])

    @property
    def trend(self):
        """
        Estimates of of unobserved trend component

        Returns
        -------
        out: Bunch
            Has the following attributes:

            - `filtered`: a time series array with the filtered estimate of
                          the component
            - `filtered_cov`: a time series array with the filtered estimate of
                          the variance/covariance of the component
            - `smoothed`: a time series array with the smoothed estimate of
                          the component
            - `smoothed_cov`: a time series array with the smoothed estimate of
                          the variance/covariance of the component
            - `offset`: an integer giving the offset in the state vector where
                        this component begins
        """
        # If present, trend is always the second component of the state vector
        # (because level is always present if trend is present)
        out = None
        spec = self.specification
        if spec.trend:
            offset = int(spec.level)
            out = Bunch(filtered=self.filtered_state[offset],
                        filtered_cov=self.filtered_state_cov[offset, offset],
                        smoothed=None, smoothed_cov=None,
                        offset=offset)
            if self.smoothed_state is not None:
                out.smoothed = self.smoothed_state[offset]
            if self.smoothed_state_cov is not None:
                out.smoothed_cov = self.smoothed_state_cov[offset, offset]
        return out

    @property
    def seasonal(self):
        out = None
        spec = self.specification
        if spec.seasonal:
            k = self.model.k
            offset = int(spec.level + spec.trend)
            filtered_state = []
            filtered_state_cov = []
            smoothed_state = []
            smoothed_state_cov = []
            for i in k:
                filtered_state.append(self.filtered_state[offset: offset + i].sum(0))
                filtered_state_cov.append(
                    self.filtered_state_cov[offset: offset + i, offset: offset + i].sum(0).sum(0))
                if self.smoothed_state is not None:
                    smoothed_state.append(self.smoothed_state[offset: offset + i].sum(0))
                if self.smoothed_state_cov is not None:
                    smoothed_state_cov.append(
                        self.smoothed_state_cov[offset: offset + i, offset: offset + i].sum(0).sum(0))
                offset += 2 * i
            out = Bunch(filtered=np.array(filtered_state),
                        filtered_cov=np.array(filtered_state_cov),
                        smoothed=np.array(smoothed_state),
                        smoothed_cov=smoothed_state_cov,
                        offset=offset)
        return out

    @property
    def arma_error(self):
        out = None
        spec = self.specification
        if spec.autoregressive:
            p, q = self.model.order
            offset = int(spec.level + spec.trend)
            if spec.seasonal:
                offset += 2 * np.sum(self.model.k)

            if q > 0:
                filtered = self.filtered_state[offset] + self.filtered_state[offset + p]
                filtered_cov = self.filtered_state_cov[offset, offset] + self.filtered_state_cov[offset + p, offset + p]

            else:
                filtered = self.filtered_state[offset]
                filtered_cov = self.filtered_state_cov[offset, offset]
            out = Bunch(filtered=filtered,
                        filtered_cov=filtered_cov,
                        smoothed=None, smoothed_cov=None,
                        offset=offset)

            if self.smoothed_state is not None:
                if q > 0:
                    out.smoothed = self.smoothed_state[offset] + self.smoothed_state[offset + p]
                else:
                    out.smoothed = self.smoothed_state[offset]
            if self.smoothed_state_cov is not None:
                if q > 0:
                    out.smoothed_cov = self.smoothed_state_cov[offset, offset] + self.smoothed_state_cov[
                        offset + p, offset + p]
                else:
                    out.smoothed_cov = self.smoothed_state_cov[offset, offset]
        return out

    def plot_components(self, which=None, alpha=0.05, start=None, end=None, observed=True, level=True, trend=True,
                        seasonal=True, autoregressive=True, resid=True, fig=None, figsize=None,
                        legend_loc='upper left'):

        from scipy.stats import norm
        from statsmodels.graphics.utils import _import_mpl, create_mpl_fig
        plt = _import_mpl()
        fig = create_mpl_fig(fig, figsize)

        # Determine which results we have
        if which is None:
            which = 'filtered' if self.smoothed_state is None else 'smoothed'

        spec = self.specification
        components = OrderedDict([
            ('level', level and spec.level),
            ('trend', trend and spec.trend),
            ('arma_error', autoregressive and spec.autoregressive)
        ])

        seasonal = seasonal and spec.seasonal

        llb = self.filter_results.loglikelihood_burn

        # Number of plots
        k_plots = observed + np.sum(list(components.values())) + resid
        if seasonal:
            k_plots += len(self.model.periods)

        # Get dates, if applicable
        if hasattr(self.data, 'dates') and self.data.dates is not None:
            dates = self.data.dates._mpl_repr()
            start = self.model._get_dates_loc(self.data.dates, start) if start else llb
            end = self.model._get_dates_loc(self.data.dates, end) if end else None
        else:
            dates = np.arange(len(self.resid))

        # Get the critical value for confidence intervals
        critical_value = norm.ppf(1 - alpha / 2.)

        plot_idx = 1

        if observed:
            ax = fig.add_subplot(k_plots, 1, plot_idx)
            plot_idx += 1

            # Plot the observed dataset
            ax.plot(dates[start:end], self.data.endog[start:end], color='k',
                    label='Observed')

            # Get the predicted values and confidence intervals
            predict = self.filter_results.forecasts[0]
            std_errors = np.sqrt(self.filter_results.forecasts_error_cov[0, 0])
            ci_lower = predict - critical_value * std_errors
            ci_upper = predict + critical_value * std_errors

            if spec.box_cox:
                lmbda = self.params[1]
                predict = self.untransform_boxcox(predict, lmbda)
                ci_lower = self.untransform_boxcox(ci_lower, lmbda)
                ci_upper = self.untransform_boxcox(ci_upper, lmbda)

            # Plot
            ax.plot(dates[start:end], predict[start:end],
                    label='One-step-ahead predictions')
            ci_poly = ax.fill_between(
                dates[start:end], ci_lower[start:end], ci_upper[start:end], alpha=0.2
            )
            ci_label = '$%.3g \\%%$ confidence interval' % ((1 - alpha) * 100)

            # Proxy artist for fill_between legend entry
            # See e.g. http://matplotlib.org/1.3.1/users/legend_guide.html
            p = plt.Rectangle((0, 0), 1, 1, fc=ci_poly.get_facecolor()[0])

            # Legend
            handles, labels = ax.get_legend_handles_labels()
            handles.append(p)
            labels.append(ci_label)
            ax.legend(handles, labels, loc=legend_loc)

            ax.set_title('Predicted vs observed')

        for component, is_plotted in components.items():
            if not is_plotted:
                continue

            ax = fig.add_subplot(k_plots, 1, plot_idx)
            plot_idx += 1

            component_bunch = getattr(self, component)

            # Check for a valid estimation type
            if which not in component_bunch:
                raise ValueError('Invalid type of state estimate.')

            which_cov = '%s_cov' % which

            # Get the predicted values
            value = component_bunch[which]

            # Plot
            state_label = '%s (%s)' % (component.title(), which)
            ax.plot(dates[start:end], value[start:end], label=state_label)

            # Get confidence intervals
            if which_cov in component_bunch:
                std_errors = np.sqrt(component_bunch['%s_cov' % which])
                ci_lower = value - critical_value * std_errors
                ci_upper = value + critical_value * std_errors
                ci_poly = ax.fill_between(
                    dates[start:end], ci_lower[start:end], ci_upper[start:end], alpha=0.2
                )
                ci_label = ('$%.3g \\%%$ confidence interval'
                            % ((1 - alpha) * 100))

            # Legend
            ax.legend(loc=legend_loc)

            ax.set_title('%s component' % component.title())

        if seasonal:
            component_bunch = self.seasonal

            for i, (m, k, name) in enumerate(zip(self.model.periods, self.model.k, self.model.period_names)):
                ax = fig.add_subplot(k_plots, 1, plot_idx)
                plot_idx += 1

                which_cov = '%s_cov' % which

                # Get the predicted values
                value = component_bunch[which][i]

                # Plot
                state_label = '%s (%s, %s)' % (name, m, k)
                ax.plot(dates[start:end], value[start:end], label=state_label)

                # Get confidence intervals
                if which_cov in component_bunch:
                    std_errors = np.sqrt(component_bunch['%s_cov' % which][i])
                    ci_lower = value - critical_value * std_errors
                    ci_upper = value + critical_value * std_errors
                    ci_poly = ax.fill_between(
                        dates[start:end], ci_lower[start:end], ci_upper[start:end], alpha=0.2
                    )
                    ci_label = ('$%.3g \\%%$ confidence interval'
                                % ((1 - alpha) * 100))

                # Legend
                ax.legend(loc=legend_loc)

                ax.set_title(name + ' component')

        if resid:
            ax = fig.add_subplot(k_plots, 1, plot_idx)
            plot_idx += 1

            value = self.resid
            ax.plot(dates[start:end], value[start:end], label='Remainder')
            ax.legend(loc=legend_loc)

            ax.set_title('Residuals component')
            # ax.set_ylabel('Remainder')

        # Add a note if first observations excluded
        if llb > 0:
            text = ('Note: The first %d observations are not shown, due to'
                    ' approximate diffuse initialization.')
            fig.text(0.1, 0.01, text % llb, fontsize='large')

        return fig

    def forecast(self, steps=1, **kwargs):
        forecast_values = super().forecast(steps, **kwargs)
        if self.model.boxcox:
            forecast_values = self.untransform_boxcox(forecast_values, self.params[1])

        return forecast_values

    def get_prediction(self, start=None, end=None, dynamic=False, **kwargs):
        return super().get_prediction(start, end, dynamic, **kwargs)

    def summary(self, alpha=.05, start=None, separate_params=True, title=None, **kwargs):
        from statsmodels.iolib.summary import summary_params

        summary = super(TBATSResults, self).summary(
            alpha=alpha, start=start, model_name='TBATS',
            display_params=not separate_params, title=title
        )
        if separate_params:

            def make_table(self, mask, title, strip_end=True):
                res = (self, self.params[mask], self.bse[mask],
                       self.zvalues[mask], self.pvalues[mask],
                       self.conf_int(alpha)[mask])
                param_names = [
                    '.'.join(name.split('.')[:-1]) if strip_end else name
                    for name in
                    np.array(self.data.param_names)[mask].tolist()
                    ]

                return summary_params(res, yname=None, xname=param_names,
                                      alpha=alpha, use_t=False, title=title)

            offset = 0
            # Main Params
            mask = list()
            mask.append(offset)
            offset += 1
            if self.specification.box_cox:
                mask.append(offset)
                offset += 1

            # Level Alpha
            mask.append(offset)
            offset += 1
            title = "Main Params"
            table = make_table(self, mask, title)
            summary.tables.append(table)

            # Trend
            if self.specification.trend:
                mask = list()
                mask.append(offset)
                offset += 1

                if self.specification.damped_trend:
                    mask.append(offset)
                    offset += 1

                title = 'Trend'
                table = make_table(self, mask, title)
                summary.tables.append(table)

            # Seasonal
            if self.specification.seasonal:
                for period, k, name in zip(self.model.periods, self.model.k, self.model.period_names):
                    mask = [offset, offset + 1]
                    offset += 2
                    title = name + '({:.2f}, {:.0f})'.format(period, k)
                    table = make_table(self, mask, title)
                    summary.tables.append(table)

            if self.specification.autoregressive:
                p, q = self.model.order
                table = make_table(self, np.arange(offset, offset + p + q), 'ARMA Error' + str(self.model.order))
                offset += p + q
                summary.tables.append(table)

        return summary

    summary.__doc__ = MLEResults.summary.__doc__


class TBATSResultsWrapper(MLEResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(MLEResultsWrapper._attrs,
                                   _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(MLEResultsWrapper._methods,
                                     _methods)


wrap.populate_wrapper(TBATSResultsWrapper,
                      TBATSResults)
