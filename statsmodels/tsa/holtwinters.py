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

from statsmodels.base.model import Results
from statsmodels.base.wrapper import populate_wrapper, union_dicts, ResultsWrapper
from statsmodels.tsa.base.tsa_model import TimeSeriesModel

from scipy.optimize import basinhopping, brute, minimize
from scipy.spatial.distance import sqeuclidean
try:
    from scipy.special import inv_boxcox
except ImportError:
    def inv_boxcox(x, lmbda):
        return np.exp(np.log1p(lmbda * x) / lmbda) if lmbda != 0 else np.exp(x)
from scipy.stats import boxcox


def _holt_init(x, xi, p, y, l, b):
    """Initialization for the Holt Models"""
    p[xi] = x
    alpha, beta, _, l0, b0, phi = p[:6]
    alphac = 1 - alpha
    betac = 1 - beta
    y_alpha = alpha * y
    l[:] = 0
    b[:] = 0
    l[0] = l0
    b[0] = b0
    return alpha, beta, phi, alphac, betac, y_alpha


def _holt__(x, xi, p, y, l, b, s, m, n, max_seen):
    """
    Simple Exponential Smoothing
    Minimization Function
    (,)
    """
    alpha, beta, phi, alphac, betac, y_alpha = _holt_init(x, xi, p, y, l, b)
    for i in range(1, n):
        l[i] = (y_alpha[i - 1]) + (alphac * (l[i - 1]))
    return sqeuclidean(l, y)


def _holt_mul_dam(x, xi, p, y, l, b, s, m, n, max_seen):
    """
    Multiplicative and Multiplicative Damped 
    Minimization Function
    (M,) & (Md,)
    """
    alpha, beta, phi, alphac, betac, y_alpha = _holt_init(x, xi, p, y, l, b)
    if alpha == 0.0:
        return max_seen
    if beta > alpha:
        return max_seen
    for i in range(1, n):
        l[i] = (y_alpha[i - 1]) + (alphac * (l[i - 1] * b[i - 1]**phi))
        b[i] = (beta * (l[i] / l[i - 1])) + (betac * b[i - 1]**phi)
    return sqeuclidean(l * b**phi, y)


def _holt_add_dam(x, xi, p, y, l, b, s, m, n, max_seen):
    """
    Additive and Additive Damped 
    Minimization Function
    (A,) & (Ad,)
    """
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
    """Initialization for the Holt Winters Seasonal Models"""
    p[xi] = x
    alpha, beta, gamma, l0, b0, phi = p[:6]
    s0 = p[6:]
    alphac = 1 - alpha
    betac = 1 - beta
    gammac = 1 - gamma
    y_alpha = alpha * y
    y_gamma = gamma * y
    l[:] = 0
    b[:] = 0
    s[:] = 0
    l[0] = l0
    b[0] = b0
    s[:m] = s0
    return alpha, beta, gamma, phi, alphac, betac, gammac, y_alpha, y_gamma


def _holt_win__mul(x, xi, p, y, l, b, s, m, n, max_seen):
    """
    Multiplicative Seasonal 
    Minimization Function
    (,M)
    """
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


def _holt_win__add(x, xi, p, y, l, b, s, m, n, max_seen):
    """
    Additive Seasonal 
    Minimization Function
    (,A)
    """
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


def _holt_win_add_mul_dam(x, xi, p, y, l, b, s, m, n, max_seen):
    """
    Additive and Additive Damped with Multiplicative Seasonal 
    Minimization Function
    (A,M) & (Ad,M)
    """
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


def _holt_win_mul_mul_dam(x, xi, p, y, l, b, s, m, n, max_seen):
    """
    Multiplicative and Multiplicative Damped with Multiplicative Seasonal 
    Minimization Function
    (M,M) & (Md,M)
    """
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


def _holt_win_add_add_dam(x, xi, p, y, l, b, s, m, n, max_seen):
    """
    Additive and Additive Damped with Additive Seasonal 
    Minimization Function
    (A,A) & (Ad,A)
    """
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


def _holt_win_mul_add_dam(x, xi, p, y, l, b, s, m, n, max_seen):
    """
    Multiplicative and Multiplicative Damped with Additive Seasonal 
    Minimization Function
    (M,A) & (M,Ad)
    """
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


class HoltWintersResults(Results):
    """
    Holt Winter's Exponential Smoothing Results

    Parameters
    ----------
    model : ExponentialSmoothing instance
        The fitted model instance
    params : dictionary
        All the parameters for the Exponential Smoothing model.

    Attributes
    ----------
    specification : dictionary
        Dictionary including all attributes from the VARMAX model instance.
    params: dictionary
        All the parameters for the Exponential Smoothing model.
    fittedfcast: array
        An array of both the fitted values and forecast values.
    fittedvalues: array
        An array of the fitted values. Fitted by the Exponential Smoothing 
        model.
    fcast: array
        An array of the forecast values forecast by the Exponential Smoothing
        model.
    sse: float
        The sum of squared errors
    level: array
        An array of the levels values that make up the fitted values.
    slope: array
        An array of the slope values that make up the fitted values.
    season: array
        An array of the seaonal values that make up the fitted values.
    aic: float
        The Akaike information criterion.
    bic: float
        The Bayesian information criterion.
    aicc: float
        AIC with a correction for finite sample sizes.
    resid: array
        An array of the residuals of the fittedvalues and actual values.
    k: int
        the k parameter used to remove the bias in AIC, BIC etc.

    """

    def __init__(self, model, params, **kwds):
        self.data = model.data
        super(HoltWintersResults, self).__init__(model, params, **kwds)

    def predict(self, start=None, end=None):
        """
        In-sample prediction and out-of-sample forecasting

        Parameters
        ----------
        start : int, str, or datetime, optional
            Zero-indexed observation number at which to start forecasting, ie.,
            the first forecast is start. Can also be a date string to
            parse or a datetime type. Default is the the zeroth observation.
        end : int, str, or datetime, optional
            Zero-indexed observation number at which to end forecasting, ie.,
            the first forecast is start. Can also be a date string to
            parse or a datetime type. However, if the dates index does not
            have a fixed frequency, end must be an integer index if you
            want out of sample prediction. Default is the last observation in
            the sample.        

        Returns
        -------
        forecast : array
            Array of out of sample forecasts.
        """
        return self.model.predict(self.params, start, end)

    def forecast(self, steps=1):
        """
        Out-of-sample forecasts

        Parameters
        ----------
        steps : int
            The number of out of sample forecasts from the end of the
            sample.

        Returns
        -------
        forecast : array
            Array of out of sample forecasts
        """
        try:
            start = self.model._index[-1] + 1
            end = self.model._index[-1] + steps
            return self.model.predict(self.params, start=start, end=end)
        except ValueError:
            # May occur when the index doesn't have a freq
            return self.model._predict(h=steps, **self.params).fcastvalues


class HoltWintersResultsWrapper(ResultsWrapper):
    _attrs = {'fittedvalues': 'rows',
              'level': 'rows',
              'resid': 'rows',
              'season': 'rows',
              'slope': 'rows'}
    _wrap_attrs = union_dicts(ResultsWrapper._wrap_attrs, _attrs)
    _methods = {'predict': 'dates',
                'forecast': 'dates'}
    _wrap_methods = union_dicts(ResultsWrapper._wrap_methods, _methods)


populate_wrapper(HoltWintersResultsWrapper, HoltWintersResults)


class ExponentialSmoothing(TimeSeriesModel):
    """
    Holt Winter's Exponential Smoothing

    Parameters
    ----------
    endog : array-like
        Time series
    trend : {"add", "mul", "additive", "multiplicative", None}, optional
        Type of trend component.
    damped : bool, optional
        Should the trend component be damped.
    seasonal : {"add", "mul", "additive", "multiplicative", None}, optional
        Type of seasonal component.
    seasonal_periods : int, optional
        The number of seasons to consider for the holt winters.

    Returns
    -------
    results : ExponentialSmoothing class        

    Notes
    -----
    This is a full implementation of the holt winters exponential smoothing as
    per [1]. This includes all the unstable methods as well as the stable methods.
    The implementation of the library covers the functionality of the R 
    library as much as possible whilst still being pythonic.

    References
    ----------
    [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles and practice. OTexts, 2014.
    """

    def __init__(self, endog, trend=None, damped=False, seasonal=None,
                 seasonal_periods=None, dates=None, freq=None, missing='none'):
        super(ExponentialSmoothing, self).__init__(
            endog, None, dates, freq, missing=missing)
        if trend in ['additive', 'multiplicative']:
            trend = {'additive': 'add', 'multiplicative': 'mul'}[trend]
        self.trend = trend
        self.damped = damped
        if seasonal in ['additive', 'multiplicative']:
            seasonal = {'additive': 'add', 'multiplicative': 'mul'}[seasonal]
        self.seasonal = seasonal
        self.trending = trend in ['mul', 'add']
        self.seasoning = seasonal in ['mul', 'add']
        if (self.trend == 'mul' or self.seasonal == 'mul') and (endog <= 0.0).any():
            raise NotImplementedError(
                'Unable to correct for negative or zero values')
        if self.damped and not self.trending:
            raise NotImplementedError('Can only dampen the trend component')
        if self.seasoning:
            if (seasonal_periods is None or seasonal_periods == 0):
                raise NotImplementedError(
                    'Unable to detect season automatically')
            self.seasonal_periods = seasonal_periods
        else:
            self.seasonal_periods = 0
        self.nobs = len(self.endog)

    def predict(self, params, start=None, end=None):
        """
        Returns in-sample and out-of-sample prediction.

        Parameters
        ----------
        params : array
            The fitted model parameters.
        start : int, str, or datetime
            Zero-indexed observation number at which to start forecasting, ie.,
            the first forecast is start. Can also be a date string to
            parse or a datetime type.
        end : int, str, or datetime
            Zero-indexed observation number at which to end forecasting, ie.,
            the first forecast is start. Can also be a date string to
            parse or a datetime type.

        Returns
        -------
        predicted values : array
        """
        if start is None:
            start = self._index[-1] + 1
        start, end, out_of_sample, prediction_index = self._get_prediction_index(
            start=start, end=end)
        if out_of_sample > 0:
            res = self._predict(h=out_of_sample, **params)
        else:
            res = self._predict(h=0, **params)
        return res.fittedfcast[start:end + out_of_sample + 1]

    def fit(self, smoothing_level=None, smoothing_slope=None, smoothing_seasonal=None,
            damping_slope=None, optimized=True, use_boxcox=False, remove_bias=False,
            use_basinhopping=False):
        """
        fit Holt Winter's Exponential Smoothing

        Parameters
        ----------
        smoothing_level : float, optional
            The alpha value of the simple exponential smoothing, if the value is
            set then this value will be used as the value.
        smoothing_slope :  float, optional
            The beta value of the holts trend method, if the value is
            set then this value will be used as the value.
        smoothing_seasonal : float, optional
            The gamma value of the holt winters seasonal method, if the value is
            set then this value will be used as the value.
        damping_slope : float, optional
            The phi value of the damped method, if the value is
            set then this value will be used as the value.
        optimized : bool, optional
            Should the values that have not been set above be optimized 
            automatically?
        use_boxcox : {True, False, 'log', float}, optional
            Should the boxcox tranform be applied to the data first? If 'log' 
            then apply the log. If float then use lambda equal to float.
        remove_bias : bool, optional
            Should the bias be removed from the forecast values and fitted values 
            before being returned? Does this by enforcing average residuals equal 
            to zero.
        use_basinhopping : bool, optional
            Should the opptimser try harder using basinhopping to find optimal 
            values?

        Returns
        -------
        results : HoltWintersResults class
            See statsmodels.tsa.holtwinters.HoltWintersResults

        Notes
        -----
        This is a full implementation of the holt winters exponential smoothing as
        per [1]. This includes all the unstable methods as well as the stable methods.
        The implementation of the library covers the functionality of the R 
        library as much as possible whilst still being pythonic.

        References
        ----------
        [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles and practice. OTexts, 2014.
        """
        # Variable renames to alpha,beta, etc as this helps with following the
        # mathematical notation in general
        alpha = smoothing_level
        beta = smoothing_slope
        gamma = smoothing_seasonal
        phi = damping_slope

        data = self.endog
        damped = self.damped
        seasoning = self.seasoning
        trending = self.trending
        trend = self.trend
        seasonal = self.seasonal
        m = self.seasonal_periods
        opt = None
        phi = phi if damped else 1.0
        if use_boxcox == 'log':
            lamda = 0.0
            y = boxcox(data, lamda)
        elif isinstance(use_boxcox, float):
            lamda = use_boxcox
            y = boxcox(data, lamda)
        elif use_boxcox:
            y, lamda = boxcox(data)
        else:
            lamda = None
            y = data.squeeze()
        if np.ndim(y) != 1:
            raise NotImplementedError('Only 1 dimensional data supported')
        l = np.zeros((self.nobs,))
        b = np.zeros((self.nobs,))
        s = np.zeros((self.nobs + m - 1,))
        p = np.zeros(6 + m)
        max_seen = np.finfo(np.double).max
        if seasoning:
            l0 = y[np.arange(self.nobs) % m == 0].mean()
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
            init_beta = beta if beta is not None else 0.1 * init_alpha if trending else beta
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
                xi = np.array([alpha is None, beta is None, gamma is None,
                               True, trending, phi is None and damped] + [True] * m)
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
            res = brute(func, bounds[txi], (txi, p, y, l, b, s, m, self.nobs, max_seen),
                        Ns=20, full_output=True, finish=None)
            (p[txi], max_seen, grid, Jout) = res
            [alpha, beta, gamma, l0, b0, phi] = p[:6]
            s0 = p[6:]            
            #bounds = np.array([(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,None),(0.0,None),(0.8,1.0)] + [(None,None),]*m)
            if use_basinhopping:
                # Take a deeper look in the local minimum we are in to find the best
                # solution to parameters, maybe hop around to try escape the local
                # minimum we may be in.
                res = basinhopping(func, p[xi], minimizer_kwargs={'args': (
                    xi, p, y, l, b, s, m, self.nobs, max_seen), 'bounds': bounds[xi]}, stepsize=0.01)
            else:
                # Take a deeper look in the local minimum we are in to find the best
                # solution to parameters
                res = minimize(func, p[xi], args=(
                    xi, p, y, l, b, s, m, self.nobs, max_seen), bounds=bounds[xi])                
            p[xi] = res.x            
            [alpha, beta, gamma, l0, b0, phi] = p[:6]
            s0 = p[6:]
            opt = res
        hwfit = self._predict(h=0, smoothing_level=alpha, smoothing_slope=beta,
                              smoothing_seasonal=gamma, damping_slope=phi,
                              initial_level=l0, initial_slope=b0, initial_seasons=s0,
                              use_boxcox=use_boxcox, lamda=lamda, remove_bias=remove_bias)
        hwfit._results.mle_retvals = opt
        return hwfit

    def _predict(self, h=None, smoothing_level=None, smoothing_slope=None,
                 smoothing_seasonal=None, initial_level=None, initial_slope=None,
                 damping_slope=None, initial_seasons=None, use_boxcox=None, lamda=None, remove_bias=None):
        """
        Helper prediction function

        Parameters
        ----------
        h : int, optional
            The number of time steps to forecast ahead.
        """
        # Variable renames to alpha,beta, etc as this helps with following the
        # mathematical notation in general
        alpha = smoothing_level
        beta = smoothing_slope
        gamma = smoothing_seasonal
        phi = damping_slope

        # Start in sample and out of sample predictions
        data = self.endog
        damped = self.damped
        seasoning = self.seasoning
        trending = self.trending
        trend = self.trend
        seasonal = self.seasonal
        m = self.seasonal_periods
        phi = phi if damped else 1.0
        if use_boxcox == 'log':
            lamda = 0.0
            y = boxcox(data, 0.0)
        elif isinstance(use_boxcox, float):
            lamda = use_boxcox
            y = boxcox(data, lamda)
        elif use_boxcox:
            y, lamda = boxcox(data)
        else:
            lamda = None
            y = data.squeeze()
            if np.ndim(y) != 1:
                raise NotImplementedError('Only 1 dimensional data supported')
        y_alpha = np.zeros((self.nobs,))
        y_gamma = np.zeros((self.nobs,))
        alphac = 1 - alpha
        y_alpha[:] = alpha * y
        if trending:
            betac = 1 - beta
        if seasoning:
            gammac = 1 - gamma
            y_gamma[:] = gamma * y
        l = np.zeros((self.nobs + h + 1,))
        b = np.zeros((self.nobs + h + 1,))
        s = np.zeros((self.nobs + h + m + 1,))
        l[0] = initial_level
        b[0] = initial_slope
        s[:m] = initial_seasons
        phi_h = np.cumsum(np.repeat(phi, h + 1)**np.arange(1, h + 1 + 1)
                          ) if damped else np.arange(1, h + 1 + 1)
        trended = {'mul': np.multiply,
                   'add': np.add,
                   None: lambda l, b: l
                   }[trend]
        detrend = {'mul': np.divide,
                   'add': np.subtract,
                   None: lambda l, b: 0
                   }[trend]
        dampen = {'mul': np.power,
                  'add': np.multiply,
                  None: lambda b, phi: 0
                  }[trend]
        if seasonal == 'mul':
            for i in range(1, self.nobs + 1):
                l[i] = y_alpha[i - 1] / s[i - 1] + \
                    (alphac * trended(l[i - 1], dampen(b[i - 1], phi)))
                if trending:
                    b[i] = (beta * detrend(l[i], l[i - 1])) + \
                        (betac * dampen(b[i - 1], phi))
                s[i + m - 1] = y_gamma[i - 1] / \
                    trended(l[i - 1], dampen(b[i - 1], phi)) + \
                    (gammac * s[i - 1])
            slope = b[1:i + 1].copy()
            season = s[m:i + m].copy()
            l[i:] = l[i]
            if trending:
                b[:i] = dampen(b[:i], phi)
                b[i:] = dampen(b[i], phi_h)
            trend = trended(l, b)
            s[i + m - 1:] = [s[(i - 1) + j % m] for j in range(h + 1 + 1)]            
            fitted = trend * s[:-m]
        elif seasonal == 'add':
            for i in range(1, self.nobs + 1):
                l[i] = y_alpha[i - 1] - (alpha * s[i - 1]) + \
                    (alphac * trended(l[i - 1], dampen(b[i - 1], phi)))
                if trending:
                    b[i] = (beta * detrend(l[i], l[i - 1])) + \
                        (betac * dampen(b[i - 1], phi))
                s[i + m - 1] = y_gamma[i - 1] - \
                    (gamma * trended(l[i - 1],
                                     dampen(b[i - 1], phi))) + (gammac * s[i - 1])
            slope = b[1:i + 1].copy()
            season = s[m:i + m].copy()
            l[i:] = l[i]
            if trending:
                b[:i] = dampen(b[:i], phi)
                b[i:] = dampen(b[i], phi_h)
            trend = trended(l, b)
            s[i + m - 1:] = [s[(i - 1) + j % m] for j in range(h + 1 + 1)]            
            fitted = trend + s[:-m]
        else:
            for i in range(1, self.nobs + 1):
                l[i] = y_alpha[i - 1] + \
                    (alphac * trended(l[i - 1], dampen(b[i - 1], phi)))
                if trending:
                    b[i] = (beta * detrend(l[i], l[i - 1])) + \
                        (betac * dampen(b[i - 1], phi))
            slope = b[1:i + 1].copy()
            season = s[m:i + m].copy()
            l[i:] = l[i]
            if trending:
                b[:i] = dampen(b[:i], phi)
                b[i:] = dampen(b[i], phi_h)
            trend = trended(l, b)
            fitted = trend
        level = l[1:i + 1].copy()
        if use_boxcox or use_boxcox == 'log' or isinstance(use_boxcox, float):
            fitted = inv_boxcox(fitted, lamda)
            level = inv_boxcox(level, lamda)
            slope = detrend(trend[:i], level)
            if seasonal == 'add':
                season = (fitted - inv_boxcox(trend, lamda))[:i]
            elif seasonal == 'mul':
                season = (fitted / inv_boxcox(trend, lamda))[:i]
            else:
                pass
        sse = sqeuclidean(fitted[:-h - 1], data)
        # (s0 + gamma) + (b0 + beta) + (l0 + alpha) + phi
        k = m * seasoning + 2 * trending + 2 + 1 * damped
        aic = self.nobs * np.log(sse / self.nobs) + (k) * 2
        aicc = aic + (2 * (k + 2) * (k + 3)) / (self.nobs - k - 3)
        bic = self.nobs * np.log(sse / self.nobs) + (k) * np.log(self.nobs)
        resid = data - fitted[:-h - 1]
        if remove_bias:
            fitted += resid.mean()
        if not damped:
            phi = np.NaN
        self.params = {'smoothing_level': alpha,
                       'smoothing_slope': beta,
                       'smoothing_seasonal': gamma,
                       'damping_slope': phi,
                       'initial_level': l[0],
                       'initial_slope': b[0],
                       'initial_seasons': s[:m],
                       'use_boxcox': use_boxcox,
                       'lamda': lamda,
                       'remove_bias': remove_bias}
        hwfit = HoltWintersResults(self, self.params, fittedfcast=fitted, fittedvalues=fitted[:-h - 1],
                                   fcastvalues=fitted[-h - 1:], sse=sse, level=level,
                                   slope=slope, season=season, aic=aic, bic=bic,
                                   aicc=aicc, resid=resid, k=k)
        return HoltWintersResultsWrapper(hwfit)


class SimpleExpSmoothing(ExponentialSmoothing):
    """
    Simple Exponential Smoothing wrapper(...)

    Parameters
    ----------
    endog : array-like
        Time series

    Returns
    -------
    results : SimpleExpSmoothing class

    Notes
    -----
    This is a full implementation of the simple exponential smoothing as
    per [1].

    See Also
    ---------
    Exponential Smoothing
    Holt

    References
    ----------
    [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles and practice. OTexts, 2014.
    """

    def __init__(self, endog):
        super(SimpleExpSmoothing, self).__init__(endog)

    def fit(self, smoothing_level=None, optimized=True):
        """
        fit Simple Exponential Smoothing wrapper(...)

        Parameters
        ----------
        smoothing_level : float, optional
            The smoothing_level value of the simple exponential smoothing, if the value is
            set then this value will be used as the value.
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
        return super(SimpleExpSmoothing, self).fit(smoothing_level=smoothing_level, optimized=optimized)


class Holt(ExponentialSmoothing):
    """
    Holt's Exponential Smoothing wrapper(...)

    Parameters
    ----------
    endog : array-like
        Time series
    expoential : bool, optional
        Type of trend component.
    damped : bool, optional
        Should the trend component be damped.

    Returns
    -------
    results : Holt class        

    Notes
    -----
    This is a full implementation of the holts exponential smoothing as
    per [1].

    See Also
    ---------
    Exponential Smoothing
    Simple Exponential Smoothing

    References
    ----------
    [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles and practice. OTexts, 2014.
    """

    def __init__(self, endog, exponential=False, damped=False):
        trend = 'mul' if exponential else 'add'
        super(Holt, self).__init__(endog, trend=trend, damped=damped)

    def fit(self, smoothing_level=None, smoothing_slope=None, damping_slope=None, optimized=True):
        """
        fit Holt's Exponential Smoothing wrapper(...)

        Parameters
        ----------
        smoothing_level : float, optional
            The alpha value of the simple exponential smoothing, if the value is
            set then this value will be used as the value.
        smoothing_slope :  float, optional
            The beta value of the holts trend method, if the value is
            set then this value will be used as the value.
        damping_slope : float, optional
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
        return super(Holt, self).fit(smoothing_level=smoothing_level,
                                     smoothing_slope=smoothing_slope, damping_slope=damping_slope,
                                     optimized=optimized)
