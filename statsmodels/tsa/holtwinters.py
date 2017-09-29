"""
Created on Wed Jul 12 09:35:35 2017

Notes
-----
Code written using below textbook as a reference.
Results are checked against the expected outcomes in the text book.

Properties:
Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles and practice. OTexts, 2014.

Author: Terence L van Zyl

"""
import numpy as np
import pandas as pd

import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
import statsmodels.tsa.base.tsa_model as tsbase

from scipy.optimize import basinhopping, brute, minimize
from scipy.spatial.distance import sqeuclidean
try:
    from scipy.special import inv_boxcox
except ImportError:
    def inv_boxcox(x, lmbda):
        np.exp(np.log1p(lmbda * x) / lmbda) if lmbda != 0 else np.exp(x)
from scipy.stats import boxcox


def _holt_init(x, xi, p, y, l, b):
    p[xi] = x
    alpha, beta, _, l0, b0, phi = p[:6];
    alphac = 1 - alpha
    betac = 1 - beta
    y_alpha = alpha * y
    l[:] = 0; b[:] = 0;
    l[0] = l0
    b[0] = b0
    return alpha, beta, phi, alphac, betac, y_alpha


def _holt__(x, xi, p, y, l, b, s, m, n, max_seen):
    alpha, beta, phi, alphac, betac, y_alpha = _holt_init(x, xi, p, y, l, b)
    for i in range(1, n):
        l[i] = (y_alpha[i - 1]) + (alphac * (l[i - 1]))
    return sqeuclidean(l, y)

#(M,) (Md,)


def _holt_mul_dam(x, xi, p, y, l, b, s, m, n, max_seen):
    alpha, beta, phi, alphac, betac, y_alpha = _holt_init(x, xi, p, y, l, b)
    if alpha == 0.0:
        return max_seen
    if beta > alpha:
        return max_seen
    for i in range(1, n):
        l[i] = (y_alpha[i - 1]) + (alphac * (l[i - 1] * b[i - 1]**phi))
        b[i] = (beta * (l[i] / l[i - 1])) + (betac * b[i - 1]**phi)
    return sqeuclidean(l * b**phi, y)

#(A,) (Ad,)


def _holt_add_dam(x, xi, p, y, l, b, s, m, n, max_seen):
    alpha, beta, phi, alphac, betac, y_alpha = _holt_init(x, xi, p, y, l, b)
    if alpha == 0.0:
        return max_seen
    if beta > alpha:
        return max_seen
    for i in range(1, n):
        l[i] = (y_alpha[i - 1]) + (alphac * (l[i - 1] + phi * b[i - 1]))
        b[i] = (beta * (l[i] - l[i - 1])) + (betac * phi * b[i - 1])
    return sqeuclidean(l + phi * b, y)


def _holt_win_init(x, xi, p, y, l, b, s, m):
    p[xi] = x
    alpha, beta, gamma, l0, b0, phi = p[:6]; s0 = p[6:]
    alphac = 1 - alpha
    betac = 1 - beta
    gammac = 1 - gamma
    y_alpha = alpha * y
    y_gamma = gamma * y
    l[:] = 0; b[:] = 0; s[:] = 0
    l[0] = l0; b[0] = b0; s[:m] = s0
    return alpha, beta, gamma, phi, alphac, betac, gammac, y_alpha, y_gamma

#(,M)


def _holt_win__mul(x, xi, p, y, l, b, s, m, n, max_seen):
    alpha, beta, gamma, phi, alphac, betac, gammac, y_alpha, y_gamma = _holt_win_init(
        x, xi, p, y, l, b, s, m)
    if alpha == 0.0:
        return max_seen
    if gamma > 1 - alpha:
        return max_seen
    for i in range(1, n):
        l[i] = (y_alpha[i - 1] / s[i - 1]) + (alphac * (l[i - 1]))
        s[i + m - 1] = (y_gamma[i - 1] / (l[i - 1])) + (gammac * s[i - 1])
    return sqeuclidean(l * s[:-(m - 1)], y)

#(,A)


def _holt_win__add(x, xi, p, y, l, b, s, m, n, max_seen):
    alpha, beta, gamma, phi, alphac, betac, gammac, y_alpha, y_gamma = _holt_win_init(
        x, xi, p, y, l, b, s, m)
    if alpha == 0.0:
        return max_seen
    if gamma > 1 - alpha:
        return max_seen
    for i in range(1, n):
        l[i] = (y_alpha[i - 1]) - (alpha * s[i - 1]) + (alphac * (l[i - 1]))
        s[i + m - 1] = y_gamma[i - 1] - \
            (gamma * (l[i - 1])) + (gammac * s[i - 1])
    return sqeuclidean(l + s[:-(m - 1)], y)

#(A,M) (Ad,M)


def _holt_win_add_mul_dam(x, xi, p, y, l, b, s, m, n, max_seen):
    alpha, beta, gamma, phi, alphac, betac, gammac, y_alpha, y_gamma = _holt_win_init(
        x, xi, p, y, l, b, s, m)
    if alpha * beta == 0.0:
        return max_seen
    if beta > alpha or gamma > 1 - alpha:
        return max_seen
    for i in range(1, n):
        l[i] = (y_alpha[i - 1] / s[i - 1]) + \
            (alphac * (l[i - 1] + phi * b[i - 1]))
        b[i] = (beta * (l[i] - l[i - 1])) + (betac * phi * b[i - 1])
        s[i + m - 1] = (y_gamma[i - 1] / (l[i - 1] + phi *
                                          b[i - 1])) + (gammac * s[i - 1])
    return sqeuclidean((l + phi * b) * s[:-(m - 1)], y)

#(M,M) (Md,M)


def _holt_win_mul_mul_dam(x, xi, p, y, l, b, s, m, n, max_seen):
    alpha, beta, gamma, phi, alphac, betac, gammac, y_alpha, y_gamma = _holt_win_init(
        x, xi, p, y, l, b, s, m)
    if alpha * beta == 0.0:
        return max_seen
    if beta > alpha or gamma > 1 - alpha:
        return max_seen
    for i in range(1, n):
        l[i] = (y_alpha[i - 1] / s[i - 1]) + \
            (alphac * (l[i - 1] * b[i - 1]**phi))
        b[i] = (beta * (l[i] / l[i - 1])) + (betac * b[i - 1]**phi)
        s[i + m - 1] = (y_gamma[i - 1] / (l[i - 1] *
                                          b[i - 1]**phi)) + (gammac * s[i - 1])
    return sqeuclidean((l * b**phi) * s[:-(m - 1)], y)

#(A,A) (Ad,A)


def _holt_win_add_add_dam(x, xi, p, y, l, b, s, m, n, max_seen):
    alpha, beta, gamma, phi, alphac, betac, gammac, y_alpha, y_gamma = _holt_win_init(
        x, xi, p, y, l, b, s, m)
    if alpha * beta == 0.0:
        return max_seen
    if beta > alpha or gamma > 1 - alpha:
        return max_seen
    for i in range(1, n):
        l[i] = (y_alpha[i - 1]) - (alpha * s[i - 1]) + \
            (alphac * (l[i - 1] + phi * b[i - 1]))
        b[i] = (beta * (l[i] - l[i - 1])) + (betac * phi * b[i - 1])
        s[i + m - 1] = y_gamma[i - 1] - \
            (gamma * (l[i - 1] + phi * b[i - 1])) + (gammac * s[i - 1])
    return sqeuclidean((l + phi * b) + s[:-(m - 1)], y)

#(M,A) (M,Ad)


def _holt_win_mul_add_dam(x, xi, p, y, l, b, s, m, n, max_seen):
    alpha, beta, gamma, phi, alphac, betac, gammac, y_alpha, y_gamma = _holt_win_init(
        x, xi, p, y, l, b, s, m)
    if alpha * beta == 0.0:
        return max_seen
    if beta > alpha or gamma > 1 - alpha:
        return max_seen
    for i in range(1, n):
        l[i] = (y_alpha[i - 1]) - (alpha * s[i - 1]) + \
            (alphac * (l[i - 1] * b[i - 1]**phi))
        b[i] = (beta * (l[i] / l[i - 1])) + (betac * b[i - 1]**phi)
        s[i + m - 1] = y_gamma[i - 1] - \
            (gamma * (l[i - 1] * b[i - 1]**phi)) + (gammac * s[i - 1])
    return sqeuclidean((l * phi * b) + s[:-(m - 1)], y)


class HoltWintersResults(base.Results):
    def __init__(self, model, params, **kwds):
        self.data = model.data
        super(HoltWintersResults, self).__init__(model, params, **kwds)

    def predict(self, start=None, end=None):
        return self.model.predict(self.params, start, end)

    def forecast(self, steps=1):
        return self.model._predict(h=steps, **self.params).fcast
        
        
class HoltWintersResultsWrapper(wrap.ResultsWrapper):
    _attrs = {'fittedvalues': 'rows',
              'level': 'rows',
              'resid': 'rows',
              'season': 'rows',
              'slope': 'rows'
              }
    _wrap_attrs = wrap.union_dicts(wrap.ResultsWrapper._wrap_attrs,
                                   _attrs)
    _methods = {'predict': 'dates'}
    _wrap_methods = wrap.union_dicts(wrap.ResultsWrapper._wrap_methods,
                                     _methods)


wrap.populate_wrapper(HoltWintersResultsWrapper, HoltWintersResults)


class ExponentialSmoothing(tsbase.TimeSeriesModel):
    """
    Holt Winter's Exponential Smoothing

    Parameters
    ----------
    endog : array-like
        Time series
    trend : {"add", "mul", None}, optional
        Type of trend component.
    damped : bool, optional
        Should the trend component be damped.
    seasonal : {"add", "mul", None}, optional
        Type of seasonal component.
    m : int, optional
        The number of seasons to consider for the holt winters.
    """

    def __init__(
            self,
            endog,
            trend=None,
            damped=False,
            seasonal=None,
            m=None,
            dates=None,
            freq=None,
            missing='none',
            **kwargs):
        super(
            ExponentialSmoothing,
            self).__init__(
            endog,
            None,
            dates,
            freq,
            missing=missing)
        self.trend = trend
        self.damped = damped
        self.seasonal = seasonal
        self.trending = trend in ['mul', 'add']
        self.seasoning = seasonal in ['mul', 'add']
        if self.damped and not self.trending:
            raise NotImplementedError('Can only dampen the trend component')
        if self.seasoning:
            if (m is None or m == 0):
                raise NotImplementedError(
                    'Unable to detect season automatically')
            self.m = m
        else:
            self.m = 0

    def predict(self, params, start=None, end=None):
        if start is None:
            start = self._index[0]
        start, end, out_of_sample, prediction_index = self._get_prediction_index(start=start, end=end)
        if out_of_sample>0:
            res = self._predict(h=out_of_sample, **params)
        else:
            res = self._predict(h=0, **params)
        return res.fittedfcast[start:end+out_of_sample+1]

    def fit(
            self,
            alpha=None,
            beta=None,
            gamma=None,
            phi=None,
            optimized=True,
            use_boxcox=False,
            remove_bias=False,
            use_basinhopping=False):
        """
        Holt Winter's Exponential Smoothing

        Parameters
        ----------
        alpha : float, optional
            The alpha value of the simple exponential smoothing, if the value is
            set then this value will be used as the value.
        beta :  float, optional
            The beta value of the holts trend method, if the value is
            set then this value will be used as the value.
        gamma : float, optional
            The gamma value of the holt winters seasonal method, if the value is
            set then this value will be used as the value.
        phi : float, optional
            The phi value of the damped method, if the value is
            set then this value will be used as the value.
        optimized : bool, optional
            Should the values that have not been set above be optimized automatically?
        use_boxcox : {True, False, 'log'}, optional
            Should the boxcox tranform be applied to the data first? If 'log' then
            apply the log.
        remove_bias : bool, optional
            Should the bias be removed from the fcast and fitted values before being
            returned?
        use_basinhopping : bool, optional
            Should the opptimser try harder using basinhopping to find optimal values?

        Returns
        -------
        results : HoltWintersResults class
            See statsmodels.tsa.holtwinters.HoltWintersResults

        Notes
        -----
        This is a full implementation of the holt winters exponential smoothing as
        per [1]. This includes all the unstable methods as well as the stable methods.
        The implementaion of the library follows as per the R library as much as possible
        whilest still being pythonic.

        References
        ----------
        [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles and practice. OTexts, 2014.
        """
        data = self.endog
        damped = self.damped
        seasoning = self.seasoning
        trending = self.trending
        trend = self.trend
        seasonal = self.seasonal
        m = self.m
        opt = None
        phi = phi if damped else 1.0
        n = len(data)
        if use_boxcox == 'log':            
            lamda = 0.0
            y = boxcox(data, 0.0)
        elif use_boxcox:
            y, lamda = boxcox(data)
        else:
            lamda = None
            y = data.squeeze()
            if np.ndim(y) != 1:
                raise NotImplementedError('Only 1 dimensional data supported')
        l = np.zeros((n,))
        b = np.zeros((n,))
        s = np.zeros((n + m - 1,))
        p = np.zeros(6 + m)
        max_seen = np.finfo(np.double).max
        if seasoning:
            l0 = y[np.arange(n) % m == 0].mean()
            b0 = ((y[m:m + m] - y[:m]) / m).mean() if trending else None
            s0 = list(y[:m] / l0) if seasonal == 'mul' else list(y[:m] - l0)
        elif trending:
            l0 = y[0]
            b0 = y[1] / y[0] if trend == 'mul' else y[1] - y[0]
            s0 = []
        else:
            l0 = y[0]
            b0 = None
            s0 = []
        if optimized:
            init_alpha = alpha if alpha is not None else 0.5 / max(m, 1)
            init_beta = beta if beta is not None else 0.1 * init_alpha
            init_gamma = None
            init_phi = phi if phi is not None else 0.99
            # Selection of functions to optimize for approporate parameters
            func_dict = {('mul', 'add'): _holt_win_add_mul_dam,
                         ('mul', 'mul'): _holt_win_mul_mul_dam,
                         ('mul', None): _holt_win__mul,
                         ('add', 'add'): _holt_win_add_add_dam,
                         ('add', 'mul'): _holt_win_mul_add_dam,
                         ('add', None): _holt_win__add,
                         (None, 'add'): _holt_add_dam,
                         (None, 'mul'): _holt_mul_dam,
                         (None, None): _holt__}
            if seasoning:
                init_gamma = gamma if gamma is not None else 0.05 * \
                    (1 - init_alpha)
                xi = np.array([alpha is None,
                               beta is None,
                               gamma is None,
                               True,
                               trending,
                               phi is None and damped] + [True] * m)
                func = func_dict[(seasonal, trend)]
            elif trending:
                xi = np.array([alpha is None, beta is None, False,
                               True, True, phi is None and damped] + [False] * m)
                func = func_dict[(None, trend)]
            else:
                xi = np.array([alpha is None, False, False,
                               True, False, False] + [False] * m)
                func = func_dict[(None, None)]
            p[:] = [init_alpha, init_beta, init_gamma, l0, b0, init_phi] + s0

            # txi [alpha, beta, gamma, l0, b0, phi, s0,..,s_(m-1)]
            # Have a quick look in the region for a good starting place for alpha etc.
            # using guestimates for the levels
            txi = xi & np.array(
                [True, True, True, False, False, True] + [False] * m)
            bounds = np.array([(0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
                               (0.0, None), (0.0, None), (0.0, 1.0)] + [(None, None), ] * m)
            res = brute(
                func,
                bounds[txi],
                (txi,
                 p,
                 y,
                 l,
                 b,
                 s,
                 m,
                 n,
                 max_seen),
                Ns=20,
                full_output=True,
                finish=None)
            (p[txi], max_seen, grid, Jout) = res
            [alpha, beta, gamma, l0, b0, phi] = p[:6]; s0 = p[6:]
            #bounds = np.array([(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,None),(0.0,None),(0.8,1.0)] + [(None,None),]*m)
            if use_basinhopping:
                # Take a deeper look in the local minimum we are in to find the best
                # solution to parameters, maybe hop around to try escape the local
                # minimum we may be in.
                res = basinhopping(
                    func,
                    p[xi],
                    minimizer_kwargs={
                        'args': (
                            xi,
                            p,
                            y,
                            l,
                            b,
                            s,
                            m,
                            n,
                            max_seen),
                        'bounds': bounds[xi]},
                    stepsize=0.01)
            else:
                # Take a deeper look in the local minimum we are in to find the best
                # solution to parameters
                res = minimize(
                    func,
                    p[xi],
                    args=(
                        xi,
                        p,
                        y,
                        l,
                        b,
                        s,
                        m,
                        n,
                        max_seen),
                    bounds=bounds[xi])
            p[xi] = res.x
            [alpha, beta, gamma, l0, b0, phi] = p[:6]; s0 = p[6:]
            opt = res
        hwfit = self._predict(h=0,
                              alpha=alpha,
                              beta=beta,
                              gamma=gamma,
                              phi=phi,
                              l0=l0,
                              b0=b0,
                              s0=s0,
                              use_boxcox=use_boxcox,
                              lamda=lamda,
                              remove_bias=remove_bias)
        hwfit.opt = opt
        return hwfit

    def _predict(self,
                 h=None,
                 alpha=None,
                 beta=None,
                 gamma=None,
                 l0=None,
                 b0=None,
                 phi=None,
                 s0=None,
                 use_boxcox=None,
                 lamda=None,
                 remove_bias=None):
        """
        Helper prediction function

        Parameters
        ----------
        h : int, optional
            The number of time steps to forecast ahead.
        """
        # Start in sample and out of sample predictions
        h_fcast = h
        if h==0:
            h=1            
        data = self.endog
        damped = self.damped
        seasoning = self.seasoning
        trending = self.trending
        trend = self.trend
        seasonal = self.seasonal
        m = self.m
        phi = phi if damped else 1.0
        n = len(data)
        if use_boxcox == 'log':
            lamda = 0.0
            y = boxcox(data, 0.0)
        elif use_boxcox:
            y, lamda = boxcox(data)
        else:
            lamda = None
            y = data.squeeze()
            if np.ndim(y) != 1:
                raise NotImplementedError('Only 1 dimensional data supported')
        y_alpha = np.zeros((n,))
        y_gamma = np.zeros((n,))
        alphac = 1 - alpha
        y_alpha[:] = alpha * y
        if trending:
            betac = 1 - beta
        if seasoning:
            gammac = 1 - gamma
            y_gamma[:] = gamma * y
        l = np.zeros((n + h,))
        b = np.zeros((n + h,))
        s = np.zeros((n + h + m,))
        l[0] = l0
        b[0] = b0
        s[:m] = s0
        phi_h = np.cumsum(np.repeat(phi, h)**np.arange(1, h + 1)
                          ) if damped else np.arange(1, h + 1)
        trended = {
            'mul': np.multiply,
            'add': np.add,
            None: lambda l,
            b: l}[trend]
        detrend = {
            'mul': np.divide,
            'add': np.subtract,
            None: lambda l,
            b: 0}[trend]
        dampen = {
            'mul': np.power,
            'add': np.multiply,
            None: lambda b,
            phi: 0}[trend]
        if seasonal == 'mul':
            for i in range(1, n + 1):
                l[i] = y_alpha[i - 1] / s[i - 1] + \
                    (alphac * trended(l[i - 1], dampen(b[i - 1], phi)))
                if trending:
                    b[i] = (beta * detrend(l[i], l[i - 1])) + \
                        (betac * dampen(b[i - 1], phi))
                s[i + m - 1] = y_gamma[i - 1] / \
                    trended(l[i - 1], dampen(b[i - 1], phi)) + (gammac * s[i - 1])
            slope = b[:i].copy()
            season = s[m:i + m].copy()
            l[i:] = l[i]
            b[:i] = dampen(b[:i], phi)
            b[i:] = dampen(b[i], phi_h)
            s[i + m - 1:] = [s[(i - 1) + j % m] for j in range(h + 1)]
            fitted = trended(l, b) * s[:-m]
        elif seasonal == 'add':
            for i in range(1, n + 1):
                l[i] = y_alpha[i - 1] - (alpha * s[i - 1]) + \
                    (alphac * trended(l[i - 1], dampen(b[i - 1], phi)))
                if trending:
                    b[i] = (beta * detrend(l[i], l[i - 1])) + \
                        (betac * dampen(b[i - 1], phi))
                s[i + m - 1] = y_gamma[i - 1] - \
                    (gamma * trended(l[i - 1], dampen(b[i - 1], phi))) + (gammac * s[i - 1])
            slope = b[:i].copy()
            season = s[m:i + m].copy()
            l[i:] = l[i]
            b[:i] = dampen(b[:i], phi)
            b[i:] = dampen(b[i], phi_h)
            s[i + m - 1:] = [s[(i - 1) + j % m] for j in range(h + 1)]
            fitted = trended(l, b) + s[:-m]
        else:
            for i in range(1, n + 1):
                l[i] = y_alpha[i - 1] + \
                    (alphac * trended(l[i - 1], dampen(b[i - 1], phi)))
                if trending:
                    b[i] = (beta * detrend(l[i], l[i - 1])) + \
                        (betac * dampen(b[i - 1], phi))
            slope = b[:i].copy()
            season = s[m:i + m].copy()
            l[i:] = l[i]
            b[:i] = dampen(b[:i], phi)
            b[i:] = dampen(b[i], phi_h)
            fitted = trended(l, b)
        level = l[:i].copy()
        if use_boxcox or use_boxcox == 'log':            
            fitted = inv_boxcox(fitted, lamda)
            # TODO: Does it make sense to have a inv_boxcox transform of the level and trend?
            #level = inv_boxcox(level, lamda)
            #slope = inv_boxcox(slope, lamda)
            # TODO: How do we deal with the inv_boxcox transform on negative seasonal components?
            #season = inv_boxcox(season, lamda)
        SSE = sqeuclidean(fitted[:-h], data)
        # (s0 + gamma) + (b0 + beta) + (l0 + alpha) + phi
        k = m * seasoning + 2 * trending + 2 + 1 * damped
        AIC = n * np.log(SSE / n) + (k) * 2
        AICc = AIC + (2 * (k + 2) * (k + 3)) / (n - k - 3)
        BIC = n * np.log(SSE / n) + (k) * np.log(n)        
        resid = data - fitted[:-h]
        if remove_bias:
            fitted += resid.mean()
        if not damped:
            phi = None
        self.params = {'alpha': alpha,
                       'beta': beta,
                       'gamma': gamma,
                       'phi': phi,
                       'l0': l0,
                       'b0': b0,
                       's0': s0,
                       'use_boxcox': use_boxcox,
                       'lamda': lamda,
                       'remove_bias': remove_bias}
        hwfit = HoltWintersResults(self,
                                   self.params,
                                   fittedfcast=fitted,
                                   fittedvalues=fitted[:-h],
                                   fcast=fitted[-h_fcast:],
                                   SSE=SSE,
                                   level=level,
                                   slope=slope,
                                   season=season,
                                   AIC=AIC,
                                   BIC=BIC,
                                   AICc=AICc,
                                   resid=resid,
                                   k=k)
        return HoltWintersResultsWrapper(hwfit)

    @staticmethod
    def holt_winters(
            endog,
            alpha=None,
            beta=None,
            gamma=None,
            m=None,
            h=1,
            trend=None,
            damped=False,
            seasonal=None,
            phi=None,
            optimized=True,
            use_boxcox=False):
        """
        Holt Winter's Exponential Smoothing helper wrapper (...)
        Parameters
        ----------
        data : array-like
            Time series
        alpha : float, optional
            The alpha value of the simple exponential smoothing, if the value is
            set then this value will be used as the value.
        beta :  float, optional
            The beta value of the holts trend method, if the value is
            set then this value will be used as the value.
        gamma : float, optional
            The gamma value of the holt winters seasonal method, if the value is
            set then this value will be used as the value.
        m : int, optional
            The number of seasons to consider for the holt winters.
        h : int, optional
            The number of time steps to forecast ahead.
        trend : {"add", "mul", None}, optional
            Type of trend component.
        damped : bool, optional
            Should the trend component be damped.
        seasonal : {"add", "mul", None}, optional
            Type of seasonal component.
        phi : float, optional
            The phi value of the damped method, if the value is
            set then this value will be used as the value.
        optimized : bool, optional
            Should the values that have not beem set above be optimized automatically?
        use_boxcox : {True, False, 'log'}, optional
            Should the boxcox tranform be applied to the data first? If 'log' then
            apply the log.

        Returns
        -------
        results : HoltWintersResults class
            See statsmodels.tsa.holtwinters.HoltWintersResults

        Notes
        -----
        This is a full implementation of the holt winters exponential smoothing as
        per [1]. This includes all the unstable methods as well as the stable methods.
        The implementaion of the library follows as per the R library as much as possible
        whilest still being pythonic.

        References
        ----------
        [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles and practice. OTexts, 2014.
        """
        es = ExponentialSmoothing(
            endog,
            trend=trend,
            damped=damped,
            seasonal=seasonal,
            m=m)
        res = es.fit(alpha=alpha, beta=beta,
                     gamma=gamma, phi=phi, optimized=optimized,
                     use_boxcox=use_boxcox)
        return es._predict(h=h, **res.params)

    @staticmethod
    def simple_exp_smoothing(endog, alpha=None, h=1, optimized=True):
        """
        Simple Exponential Smoothing helper wrapper(...)

        Parameters
        ----------
        data : array-like
            Time series
        alpha : float, optional
            The alpha value of the simple exponential smoothing, if the value is
            set then this value will be used as the value.
        h : int, optional
            The number of time steps to forecast ahead.
        optimized : bool
            Should the values that have not been set above be optimized automatically?

        Returns
        -------
        results : HoltWintersResults class
            See statsmodels.tsa.holtwinters.HoltWintersResults

        Notes
        -----
        This is a full implementation of the simple exponential smoothing as
        per [1].

        References
        ----------
        [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles and practice. OTexts, 2014.
        """
        es = ExponentialSmoothing(endog)
        res = es.fit(alpha=alpha, optimized=optimized)
        return es._predict(h=h, **res.params)

    @staticmethod
    def holt(
            endog,
            alpha=None,
            beta=None,
            h=1,
            exponential=False,
            damped=False,
            phi=None,
            optimized=True):
        """
        Holt's Exponential Smoothing helper wrapper(...)

        Parameters
        ----------
        data : array-like
            Time series
        alpha : float, optional
            The alpha value of the simple exponential smoothing, if the value is
            set then this value will be used as the value.
        beta :  float, optional
            The beta value of the holts trend method, if the value is
            set then this value will be used as the value.
        h : int, optional
            The number of time steps to forecast ahead.
        exponential : bool, optional
            If true use a multipicative trend else use a additive trend.
        damped : bool
            Should the trend component be damped.
        phi : float, optional
            The phi value of the damped method, if the value is
            set then this value will be used as the value.
        optimized : bool, optional
            Should the values that have not been set above be optimized automatically?

        Returns
        -------
        results : HoltWintersResults class
            See statsmodels.tsa.holtwinters.HoltWintersResults

        Notes
        -----
        This is a full implementation of the holts exponential smoothing as
        per [1].

        References
        ----------
        [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles and practice. OTexts, 2014.
        """
        trend = 'mul' if exponential else 'add'
        es = ExponentialSmoothing(
            endog,
            trend=trend,
            damped=damped)
        res = es.fit(alpha=alpha, beta=beta, phi=phi, optimized=optimized)
        return es._predict(h=h, **res.params)
