"""
Notes
-----
Code written using below textbook as a reference.
Results are checked against the expected outcomes in the text book.

Properties:
Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles and
practice. OTexts, 2014.

Author: Terence L van Zyl
Modified: Kevin Sheppard
"""
import numpy as np
import pandas as pd
from scipy.optimize import basinhopping, brute, minimize
from scipy.spatial.distance import sqeuclidean
from scipy.special import inv_boxcox
from scipy.stats import boxcox

from statsmodels.base.model import Results
from statsmodels.base.wrapper import (populate_wrapper, union_dicts,
                                      ResultsWrapper)
from statsmodels.tools.validation import (array_like, bool_like, float_like,
                                          string_like, int_like)
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
from statsmodels.tsa.tsatools import freq_to_period
import statsmodels.tsa._exponential_smoothers as smoothers


def _holt_init(x, xi, p, y, l, b):
    """Initialization for the Holt Models"""
    p[xi.astype(np.bool)] = x
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
    p[xi.astype(np.bool)] = x
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
        s[i + m - 1] = y_gamma[i - 1] - (gamma * (l[i - 1])) + (gammac * s[i - 1])
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
        s[i + m - 1] = y_gamma[i - 1] - (gamma * (l[i - 1] + phi * b[i - 1])) + (gammac * s[i - 1])
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


SMOOTHERS = {('mul', 'add'): smoothers._holt_win_add_mul_dam,
             ('mul', 'mul'): smoothers._holt_win_mul_mul_dam,
             ('mul', None): smoothers._holt_win__mul,
             ('add', 'add'): smoothers._holt_win_add_add_dam,
             ('add', 'mul'): smoothers._holt_win_mul_add_dam,
             ('add', None): smoothers._holt_win__add,
             (None, 'add'): smoothers._holt_add_dam,
             (None, 'mul'): smoothers._holt_mul_dam,
             (None, None): smoothers._holt__}

PY_SMOOTHERS = {('mul', 'add'): _holt_win_add_mul_dam,
                ('mul', 'mul'): _holt_win_mul_mul_dam,
                ('mul', None): _holt_win__mul,
                ('add', 'add'): _holt_win_add_add_dam,
                ('add', 'mul'): _holt_win_mul_add_dam,
                ('add', None): _holt_win__add,
                (None, 'add'): _holt_add_dam,
                (None, 'mul'): _holt_mul_dam,
                (None, None): _holt__}


class HoltWintersResults(Results):
    """
    Holt Winter's Exponential Smoothing Results

    Parameters
    ----------
    model : ExponentialSmoothing instance
        The fitted model instance
    params : dict
        All the parameters for the Exponential Smoothing model.

    Attributes
    ----------
    params: dict
        All the parameters for the Exponential Smoothing model.
    params_formatted: pd.DataFrame
        DataFrame containing all parameters, their short names and a flag
        indicating whether the parameter's value was optimized to fit the data.
    fittedfcast: ndarray
        An array of both the fitted values and forecast values.
    fittedvalues: ndarray
        An array of the fitted values. Fitted by the Exponential Smoothing
        model.
    fcastvalues: ndarray
        An array of the forecast values forecast by the Exponential Smoothing
        model.
    sse: float
        The sum of squared errors
    level: ndarray
        An array of the levels values that make up the fitted values.
    slope: ndarray
        An array of the slope values that make up the fitted values.
    season: ndarray
        An array of the seasonal values that make up the fitted values.
    aic: float
        The Akaike information criterion.
    bic: float
        The Bayesian information criterion.
    aicc: float
        AIC with a correction for finite sample sizes.
    resid: ndarray
        An array of the residuals of the fittedvalues and actual values.
    k: int
        the k parameter used to remove the bias in AIC, BIC etc.
    optimized: bool
        Flag indicating whether the model parameters were optimized to fit
        the data.
    mle_retvals:  {None, scipy.optimize.optimize.OptimizeResult}
        Optimization results if the parameters were optimized to fit the data.
    """

    def __init__(self, model, params, **kwargs):
        self.data = model.data
        super(HoltWintersResults, self).__init__(model, params, **kwargs)

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
        forecast : ndarray
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
        forecast : ndarray
            Array of out of sample forecasts
        """
        try:
            freq = getattr(self.model._index, 'freq', 1)
            start = self.model._index[-1] + freq
            end = self.model._index[-1] + steps * freq
            return self.model.predict(self.params, start=start, end=end)
        except (AttributeError, ValueError):
            # May occur when the index does not have a freq
            return self.model._predict(h=steps, **self.params).fcastvalues

    def summary(self):
        """
        Summarize the fitted Model

        Returns
        -------
        smry : Summary instance
            This holds the summary table and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary
        """
        from statsmodels.iolib.summary import Summary
        from statsmodels.iolib.table import SimpleTable
        model = self.model
        title = model.__class__.__name__ + ' Model Results'

        dep_variable = 'endog'
        if isinstance(self.model.endog, pd.DataFrame):
            dep_variable = self.model.endog.columns[0]
        elif isinstance(self.model.endog, pd.Series):
            dep_variable = self.model.endog.name
        seasonal_periods = None if self.model.seasonal is None else self.model.seasonal_periods
        lookup = {'add': 'Additive', 'additive': 'Additive',
                  'mul': 'Multiplicative', 'multiplicative': 'Multiplicative', None: 'None'}
        transform = self.params['use_boxcox']
        box_cox_transform = True if transform else False
        box_cox_coeff = transform if isinstance(transform, str) else self.params['lamda']
        if isinstance(box_cox_coeff, float):
            box_cox_coeff = '{:>10.5f}'.format(box_cox_coeff)
        top_left = [('Dep. Variable:', [dep_variable]),
                    ('Model:', [model.__class__.__name__]),
                    ('Optimized:', [str(np.any(self.optimized))]),
                    ('Trend:', [lookup[self.model.trend]]),
                    ('Seasonal:', [lookup[self.model.seasonal]]),
                    ('Seasonal Periods:', [str(seasonal_periods)]),
                    ('Box-Cox:', [str(box_cox_transform)]),
                    ('Box-Cox Coeff.:', [str(box_cox_coeff)])]

        top_right = [
            ('No. Observations:', [str(len(self.model.endog))]),
            ('SSE', ['{:5.3f}'.format(self.sse)]),
            ('AIC', ['{:5.3f}'.format(self.aic)]),
            ('BIC', ['{:5.3f}'.format(self.bic)]),
            ('AICC', ['{:5.3f}'.format(self.aicc)]),
            ('Date:', None),
            ('Time:', None)]

        smry = Summary()
        smry.add_table_2cols(self, gleft=top_left, gright=top_right,
                             title=title)
        formatted = self.params_formatted  # type: pd.DataFrame

        def _fmt(x):
            abs_x = np.abs(x)
            scale = 1
            if abs_x != 0:
                scale = int(np.log10(abs_x))
            if scale > 4 or scale < -3:
                return '{:>20.5g}'.format(x)
            dec = min(7 - scale, 7)
            fmt = '{{:>20.{0}f}}'.format(dec)
            return fmt.format(x)

        tab = []
        for _, vals in formatted.iterrows():
            tab.append([_fmt(vals.iloc[1]),
                        '{0:>20}'.format(vals.iloc[0]),
                        '{0:>20}'.format(str(bool(vals.iloc[2])))])
        params_table = SimpleTable(tab, headers=['coeff', 'code', 'optimized'],
                                   title="",
                                   stubs=list(formatted.index))

        smry.tables.append(params_table)

        return smry


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
    endog : array_like
        Time series
    trend : {"add", "mul", "additive", "multiplicative", None}, optional
        Type of trend component.
    damped : bool, optional
        Should the trend component be damped.
    seasonal : {"add", "mul", "additive", "multiplicative", None}, optional
        Type of seasonal component.
    seasonal_periods : int, optional
        The number of periods in a complete seasonal cycle, e.g., 4 for
        quarterly data or 7 for daily data with a weekly cycle.

    Returns
    -------
    results : ExponentialSmoothing class

    Notes
    -----
    This is a full implementation of the holt winters exponential smoothing as
    per [1]_. This includes all the unstable methods as well as the stable
    methods. The implementation of the library covers the functionality of the
    R library as much as possible whilst still being Pythonic.

    References
    ----------
    .. [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
        and practice. OTexts, 2014.
    """

    def __init__(self, endog, trend=None, damped=False, seasonal=None,
                 seasonal_periods=None, dates=None, freq=None, missing='none'):
        super(ExponentialSmoothing, self).__init__(endog, None, dates,
                                                   freq, missing=missing)
        self.endog = self.endog
        self._y = self._data = array_like(endog, 'endog', contiguous=True,
                                          order='C')
        options = ("add", "mul", "additive", "multiplicative")
        trend = string_like(trend, 'trend', options=options, optional=True)
        if trend in ['additive', 'multiplicative']:
            trend = {'additive': 'add', 'multiplicative': 'mul'}[trend]
        self.trend = trend
        self.damped = bool_like(damped, 'damped')
        seasonal = string_like(seasonal, 'seasonal', options=options,
                               optional=True)
        if seasonal in ['additive', 'multiplicative']:
            seasonal = {'additive': 'add', 'multiplicative': 'mul'}[seasonal]
        self.seasonal = seasonal
        self.trending = trend in ['mul', 'add']
        self.seasoning = seasonal in ['mul', 'add']
        if (self.trend == 'mul' or self.seasonal == 'mul') and \
                not np.all(self._data > 0.0):
            raise ValueError('endog must be strictly positive when using'
                             'multiplicative trend or seasonal components.')
        if self.damped and not self.trending:
            raise ValueError('Can only dampen the trend component')
        if self.seasoning:
            self.seasonal_periods = int_like(seasonal_periods,
                                             'seasonal_periods', optional=True)
            if seasonal_periods is None:
                self.seasonal_periods = freq_to_period(self._index_freq)
            if self.seasonal_periods <= 1:
                raise ValueError('seasonal_periods must be larger than 1.')
        else:
            self.seasonal_periods = 0
        self.nobs = len(self.endog)

    def predict(self, params, start=None, end=None):
        """
        Returns in-sample and out-of-sample prediction.

        Parameters
        ----------
        params : ndarray
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
        predicted values : ndarray
        """
        if start is None:
            freq = getattr(self._index, 'freq', 1)
            start = self._index[-1] + freq
        start, end, out_of_sample, prediction_index = self._get_prediction_index(
            start=start, end=end)
        if out_of_sample > 0:
            res = self._predict(h=out_of_sample, **params)
        else:
            res = self._predict(h=0, **params)
        return res.fittedfcast[start:end + out_of_sample + 1]

    def fit(self, smoothing_level=None, smoothing_slope=None, smoothing_seasonal=None,
            damping_slope=None, optimized=True, use_boxcox=False, remove_bias=False,
            use_basinhopping=False, start_params=None, initial_level=None, initial_slope=None,
            use_brute=True):
        """
        Fit the model

        Parameters
        ----------
        smoothing_level : float, optional
            The alpha value of the simple exponential smoothing, if the value
            is set then this value will be used as the value.
        smoothing_slope :  float, optional
            The beta value of the Holt's trend method, if the value is
            set then this value will be used as the value.
        smoothing_seasonal : float, optional
            The gamma value of the holt winters seasonal method, if the value
            is set then this value will be used as the value.
        damping_slope : float, optional
            The phi value of the damped method, if the value is
            set then this value will be used as the value.
        optimized : bool, optional
            Estimate model parameters by maximizing the log-likelihood
        use_boxcox : {True, False, 'log', float}, optional
            Should the Box-Cox transform be applied to the data first? If 'log'
            then apply the log. If float then use lambda equal to float.
        remove_bias : bool, optional
            Remove bias from forecast values and fitted values by enforcing
            that the average residual is equal to zero.
        use_basinhopping : bool, optional
            Using Basin Hopping optimizer to find optimal values
        start_params : ndarray, optional
            Starting values to used when optimizing the fit.  If not provided,
            starting values are determined using a combination of grid search
            and reasonable values based on the initial values of the data
        initial_level : float, optional
            Value to use when initializing the fitted level.
        initial_slope : float, optional
            Value to use when initializing the fitted slope.
        use_brute : bool, optional
            Search for good starting values using a brute force (grid)
            optimizer. If False, a naive set of starting values is used.

        Returns
        -------
        results : HoltWintersResults class
            See statsmodels.tsa.holtwinters.HoltWintersResults

        Notes
        -----
        This is a full implementation of the holt winters exponential smoothing
        as per [1]. This includes all the unstable methods as well as the
        stable methods. The implementation of the library covers the
        functionality of the R library as much as possible whilst still
        being Pythonic.

        References
        ----------
        [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
            and practice. OTexts, 2014.
        """
        # Variable renames to alpha,beta, etc as this helps with following the
        # mathematical notation in general
        alpha = float_like(smoothing_level, 'smoothing_level', True)
        beta = float_like(smoothing_slope, 'smoothing_slope', True)
        gamma = float_like(smoothing_seasonal, 'smoothing_seasonal', True)
        phi = float_like(damping_slope, 'damping_slope', True)
        l0 = self._l0 = float_like(initial_level, 'initial_level', True)
        b0 = self._b0 = float_like(initial_slope, 'initial_slope', True)
        if start_params is not None:
            start_params = array_like(start_params, 'start_params',
                                      contiguous=True)
        data = self._data
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
        self._y = y
        lvls = np.zeros(self.nobs)
        b = np.zeros(self.nobs)
        s = np.zeros(self.nobs + m - 1)
        p = np.zeros(6 + m)
        max_seen = np.finfo(np.double).max
        l0, b0, s0 = self.initial_values()

        xi = np.zeros_like(p, dtype=np.bool)
        if optimized:
            init_alpha = alpha if alpha is not None else 0.5 / max(m, 1)
            init_beta = beta if beta is not None else 0.1 * init_alpha if trending else beta
            init_gamma = None
            init_phi = phi if phi is not None else 0.99
            # Selection of functions to optimize for appropriate parameters
            if seasoning:
                init_gamma = gamma if gamma is not None else 0.05 * \
                                                             (1 - init_alpha)
                xi = np.array([alpha is None, trending and beta is None, gamma is None,
                               initial_level is None, trending and initial_slope is None,
                               phi is None and damped] + [True] * m)
                func = SMOOTHERS[(seasonal, trend)]
            elif trending:
                xi = np.array([alpha is None, beta is None, False,
                               initial_level is None, initial_slope is None,
                               phi is None and damped] + [False] * m)
                func = SMOOTHERS[(None, trend)]
            else:
                xi = np.array([alpha is None, False, False,
                               initial_level is None, False, False] + [False] * m)
                func = SMOOTHERS[(None, None)]
            p[:] = [init_alpha, init_beta, init_gamma, l0, b0, init_phi] + s0
            if np.any(xi):
                # txi [alpha, beta, gamma, l0, b0, phi, s0,..,s_(m-1)]
                # Have a quick look in the region for a good starting place for alpha etc.
                # using guesstimates for the levels
                txi = xi & np.array([True, True, True, False, False, True] + [False] * m)
                txi = txi.astype(np.bool)
                bounds = ([(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, None),
                           (0.0, None), (0.0, 1.0)] + [(None, None), ] * m)
                args = (txi.astype(np.uint8), p, y, lvls, b, s, m, self.nobs,
                        max_seen)
                if start_params is None and np.any(txi) and use_brute:
                    _bounds = [bnd for bnd, flag in zip(bounds, txi) if flag]
                    res = brute(func, _bounds, args, Ns=20,
                                full_output=True, finish=None)
                    p[txi], max_seen, _, _ = res
                else:
                    if start_params is not None:
                        if len(start_params) != xi.sum():
                            msg = 'start_params must have {0} values but ' \
                                  'has {1} instead'
                            nxi, nsp = len(xi), len(start_params)
                            raise ValueError(msg.format(nxi, nsp))
                        p[xi] = start_params
                    args = (xi.astype(np.uint8), p, y, lvls, b, s, m,
                            self.nobs, max_seen)
                    max_seen = func(np.ascontiguousarray(p[xi]), *args)
                # alpha, beta, gamma, l0, b0, phi = p[:6]
                # s0 = p[6:]
                # bounds = np.array([(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,None),
                # (0.0,None),(0.8,1.0)] + [(None,None),]*m)
                args = (xi.astype(np.uint8), p, y, lvls, b, s, m, self.nobs, max_seen)
                if use_basinhopping:
                    # Take a deeper look in the local minimum we are in to find the best
                    # solution to parameters, maybe hop around to try escape the local
                    # minimum we may be in.
                    _bounds = [bnd for bnd, flag in zip(bounds, xi) if flag]
                    res = basinhopping(func, p[xi],
                                       minimizer_kwargs={'args': args, 'bounds': _bounds},
                                       stepsize=0.01)
                    success = res.lowest_optimization_result.success
                else:
                    # Take a deeper look in the local minimum we are in to find the best
                    # solution to parameters
                    _bounds = [bnd for bnd, flag in zip(bounds, xi) if flag]
                    lb, ub = np.asarray(_bounds).T.astype(np.float)
                    initial_p = p[xi]

                    # Ensure strictly inbounds
                    loc = initial_p <= lb
                    upper = ub[loc].copy()
                    upper[~np.isfinite(upper)] = 100.0
                    eps = 1e-4
                    initial_p[loc] = lb[loc] + eps * (upper - lb[loc])

                    loc = initial_p >= ub
                    lower = lb[loc].copy()
                    lower[~np.isfinite(lower)] = -100.0
                    eps = 1e-4
                    initial_p[loc] = ub[loc] - eps * (ub[loc] - lower)

                    res = minimize(func, initial_p, args=args, bounds=_bounds)
                    success = res.success

                if not success:
                    from warnings import warn
                    from statsmodels.tools.sm_exceptions import ConvergenceWarning
                    warn("Optimization failed to converge. Check mle_retvals.",
                         ConvergenceWarning)
                p[xi] = res.x
                opt = res
            else:
                from warnings import warn
                from statsmodels.tools.sm_exceptions import EstimationWarning
                message = "Model has no free parameters to estimate. Set " \
                          "optimized=False to suppress this warning"
                warn(message, EstimationWarning)

            [alpha, beta, gamma, l0, b0, phi] = p[:6]
            s0 = p[6:]

        hwfit = self._predict(h=0, smoothing_level=alpha, smoothing_slope=beta,
                              smoothing_seasonal=gamma, damping_slope=phi,
                              initial_level=l0, initial_slope=b0, initial_seasons=s0,
                              use_boxcox=use_boxcox, remove_bias=remove_bias, is_optimized=xi)
        hwfit._results.mle_retvals = opt
        return hwfit

    def initial_values(self):
        """
        Compute initial values used in the exponential smoothing recursions

        Returns
        -------
        initial_level : float
            The initial value used for the level component
        initial_slope : {float, None}
            The initial value used for the trend component
        initial_seasons : list
            The initial values used for the seasonal components

        Notes
        -----
        Convenience function the exposes the values used to initialize the
        recursions. When optimizing parameters these are used as starting
        values.

        Method used to compute the initial value depends on when components
        are included in the model.  In a simple exponential smoothing model
        without trend or a seasonal components, the initial value is set to the
        first observation. When a trend is added, the trend is initialized
        either using y[1]/y[0], if multiplicative, or y[1]-y[0]. When the
        seasonal component is added the initialization adapts to account for
        the modified structure.
        """
        y = self._y
        trend = self.trend
        seasonal = self.seasonal
        seasoning = self.seasoning
        trending = self.trending
        m = self.seasonal_periods
        l0 = self._l0
        b0 = self._b0
        if seasoning:
            l0 = y[np.arange(self.nobs) % m == 0].mean() if l0 is None else l0
            if b0 is None and trending:
                lead, lag = y[m:m + m], y[:m]
                if trend == 'mul':
                    b0 = np.exp((np.log(lead.mean()) - np.log(lag.mean())) / m)
                else:
                    b0 = ((lead - lag) / m).mean()
            s0 = list(y[:m] / l0) if seasonal == 'mul' else list(y[:m] - l0)
        elif trending:
            l0 = y[0] if l0 is None else l0
            if b0 is None:
                b0 = y[1] / y[0] if trend == 'mul' else y[1] - y[0]
            s0 = []
        else:
            if l0 is None:
                l0 = y[0]
            b0 = None
            s0 = []

        return l0, b0, s0

    def _predict(self, h=None, smoothing_level=None, smoothing_slope=None,
                 smoothing_seasonal=None, initial_level=None, initial_slope=None,
                 damping_slope=None, initial_seasons=None, use_boxcox=None, lamda=None,
                 remove_bias=None, is_optimized=None):
        """
        Helper prediction function

        Parameters
        ----------
        h : int, optional
            The number of time steps to forecast ahead.
        """
        # Variable renames to alpha, beta, etc as this helps with following the
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
        lvls = np.zeros((self.nobs + h + 1,))
        b = np.zeros((self.nobs + h + 1,))
        s = np.zeros((self.nobs + h + m + 1,))
        lvls[0] = initial_level
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
        nobs = self.nobs
        if seasonal == 'mul':
            for i in range(1, nobs + 1):
                lvls[i] = y_alpha[i - 1] / s[i - 1] + \
                       (alphac * trended(lvls[i - 1], dampen(b[i - 1], phi)))
                if trending:
                    b[i] = (beta * detrend(lvls[i], lvls[i - 1])) + \
                           (betac * dampen(b[i - 1], phi))
                s[i + m - 1] = y_gamma[i - 1] / trended(lvls[i - 1], dampen(b[i - 1], phi)) + \
                    (gammac * s[i - 1])
            slope = b[1:nobs + 1].copy()
            season = s[m:nobs + m].copy()
            lvls[nobs:] = lvls[nobs]
            if trending:
                b[:nobs] = dampen(b[:nobs], phi)
                b[nobs:] = dampen(b[nobs], phi_h)
            trend = trended(lvls, b)
            s[nobs + m - 1:] = [s[(nobs - 1) + j % m] for j in range(h + 1 + 1)]
            fitted = trend * s[:-m]
        elif seasonal == 'add':
            for i in range(1, nobs + 1):
                lvls[i] = y_alpha[i - 1] - (alpha * s[i - 1]) + \
                       (alphac * trended(lvls[i - 1], dampen(b[i - 1], phi)))
                if trending:
                    b[i] = (beta * detrend(lvls[i], lvls[i - 1])) + \
                           (betac * dampen(b[i - 1], phi))
                s[i + m - 1] = y_gamma[i - 1] - \
                    (gamma * trended(lvls[i - 1], dampen(b[i - 1], phi))) + \
                    (gammac * s[i - 1])
            slope = b[1:nobs + 1].copy()
            season = s[m:nobs + m].copy()
            lvls[nobs:] = lvls[nobs]
            if trending:
                b[:nobs] = dampen(b[:nobs], phi)
                b[nobs:] = dampen(b[nobs], phi_h)
            trend = trended(lvls, b)
            s[nobs + m - 1:] = [s[(nobs - 1) + j % m] for j in range(h + 1 + 1)]
            fitted = trend + s[:-m]
        else:
            for i in range(1, nobs + 1):
                lvls[i] = y_alpha[i - 1] + \
                       (alphac * trended(lvls[i - 1], dampen(b[i - 1], phi)))
                if trending:
                    b[i] = (beta * detrend(lvls[i], lvls[i - 1])) + \
                           (betac * dampen(b[i - 1], phi))
            slope = b[1:nobs + 1].copy()
            season = s[m:nobs + m].copy()
            lvls[nobs:] = lvls[nobs]
            if trending:
                b[:nobs] = dampen(b[:nobs], phi)
                b[nobs:] = dampen(b[nobs], phi_h)
            trend = trended(lvls, b)
            fitted = trend
        level = lvls[1:nobs + 1].copy()
        if use_boxcox or use_boxcox == 'log' or isinstance(use_boxcox, float):
            fitted = inv_boxcox(fitted, lamda)
            level = inv_boxcox(level, lamda)
            slope = detrend(trend[:nobs], level)
            if seasonal == 'add':
                season = (fitted - inv_boxcox(trend, lamda))[:nobs]
            else:  # seasonal == 'mul':
                season = (fitted / inv_boxcox(trend, lamda))[:nobs]
        sse = sqeuclidean(fitted[:-h - 1], data)
        # (s0 + gamma) + (b0 + beta) + (l0 + alpha) + phi
        k = m * seasoning + 2 * trending + 2 + 1 * damped
        aic = self.nobs * np.log(sse / self.nobs) + k * 2
        if self.nobs - k - 3 > 0:
            aicc_penalty = (2 * (k + 2) * (k + 3)) / (self.nobs - k - 3)
        else:
            aicc_penalty = np.inf
        aicc = aic + aicc_penalty
        bic = self.nobs * np.log(sse / self.nobs) + k * np.log(self.nobs)
        resid = data - fitted[:-h - 1]
        if remove_bias:
            fitted += resid.mean()
        self.params = {'smoothing_level': alpha,
                       'smoothing_slope': beta,
                       'smoothing_seasonal': gamma,
                       'damping_slope': phi if damped else np.nan,
                       'initial_level': lvls[0],
                       'initial_slope': b[0] / phi if phi > 0 else 0,
                       'initial_seasons': s[:m],
                       'use_boxcox': use_boxcox,
                       'lamda': lamda,
                       'remove_bias': remove_bias}

        # Format parameters into a DataFrame
        codes = ['alpha', 'beta', 'gamma', 'l.0', 'b.0', 'phi']
        codes += ['s.{0}'.format(i) for i in range(m)]
        idx = ['smoothing_level', 'smoothing_slope', 'smoothing_seasonal',
               'initial_level', 'initial_slope', 'damping_slope']
        idx += ['initial_seasons.{0}'.format(i) for i in range(m)]

        formatted = [alpha, beta, gamma, lvls[0], b[0], phi]
        formatted += s[:m].tolist()
        formatted = list(map(lambda v: np.nan if v is None else v, formatted))
        formatted = np.array(formatted)
        if is_optimized is None:
            optimized = np.zeros(len(codes), dtype=np.bool)
        else:
            optimized = is_optimized.astype(np.bool)
        included = [True, trending, seasoning, True, trending, damped]
        included += [True] * m
        formatted = pd.DataFrame([[c, f, o] for c, f, o in zip(codes, formatted, optimized)],
                                 columns=['name', 'param', 'optimized'],
                                 index=idx)
        formatted = formatted.loc[included]

        hwfit = HoltWintersResults(self, self.params, fittedfcast=fitted,
                                   fittedvalues=fitted[:-h - 1], fcastvalues=fitted[-h - 1:],
                                   sse=sse, level=level, slope=slope, season=season, aic=aic,
                                   bic=bic, aicc=aicc, resid=resid, k=k,
                                   params_formatted=formatted, optimized=optimized)
        return HoltWintersResultsWrapper(hwfit)


class SimpleExpSmoothing(ExponentialSmoothing):
    """
    Simple Exponential Smoothing

    Parameters
    ----------
    endog : array_like
        Time series

    Returns
    -------
    results : SimpleExpSmoothing class

    Notes
    -----
    This is a full implementation of the simple exponential smoothing as
    per [1]_.  `SimpleExpSmoothing` is a restricted version of
    :class:`ExponentialSmoothing`.

    See Also
    --------
    ExponentialSmoothing
    Holt

    References
    ----------
    .. [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
        and practice. OTexts, 2014.
    """

    def __init__(self, endog):
        super(SimpleExpSmoothing, self).__init__(endog)

    def fit(self, smoothing_level=None, optimized=True, start_params=None,
            initial_level=None, use_brute=True):
        """
        Fit the model

        Parameters
        ----------
        smoothing_level : float, optional
            The smoothing_level value of the simple exponential smoothing, if
            the value is set then this value will be used as the value.
        optimized : bool, optional
            Estimate model parameters by maximizing the log-likelihood
        start_params : ndarray, optional
            Starting values to used when optimizing the fit.  If not provided,
            starting values are determined using a combination of grid search
            and reasonable values based on the initial values of the data
        initial_level : float, optional
            Value to use when initializing the fitted level.
        use_brute : bool, optional
            Search for good starting values using a brute force (grid)
            optimizer. If False, a naive set of starting values is used.

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
        [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
            and practice. OTexts, 2014.
        """
        return super(SimpleExpSmoothing, self).fit(smoothing_level=smoothing_level,
                                                   optimized=optimized, start_params=start_params,
                                                   initial_level=initial_level,
                                                   use_brute=use_brute)


class Holt(ExponentialSmoothing):
    """
    Holt's Exponential Smoothing

    Parameters
    ----------
    endog : array_like
        Time series
    exponential : bool, optional
        Type of trend component.
    damped : bool, optional
        Should the trend component be damped.

    Returns
    -------
    results : Holt class

    Notes
    -----
    This is a full implementation of the Holt's exponential smoothing as
    per [1]_. `Holt` is a restricted version of :class:`ExponentialSmoothing`.

    See Also
    --------
    ExponentialSmoothing
    SimpleExpSmoothing

    References
    ----------
    .. [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
        and practice. OTexts, 2014.
    """

    def __init__(self, endog, exponential=False, damped=False):
        trend = 'mul' if exponential else 'add'
        super(Holt, self).__init__(endog, trend=trend, damped=damped)

    def fit(self, smoothing_level=None, smoothing_slope=None, damping_slope=None,
            optimized=True, start_params=None, initial_level=None,
            initial_slope=None, use_brute=True):
        """
        Fit the model

        Parameters
        ----------
        smoothing_level : float, optional
            The alpha value of the simple exponential smoothing, if the value
            is set then this value will be used as the value.
        smoothing_slope :  float, optional
            The beta value of the Holt's trend method, if the value is
            set then this value will be used as the value.
        damping_slope : float, optional
            The phi value of the damped method, if the value is
            set then this value will be used as the value.
        optimized : bool, optional
            Estimate model parameters by maximizing the log-likelihood
        start_params : ndarray, optional
            Starting values to used when optimizing the fit.  If not provided,
            starting values are determined using a combination of grid search
            and reasonable values based on the initial values of the data
        initial_level : float, optional
            Value to use when initializing the fitted level.
        initial_slope : float, optional
            Value to use when initializing the fitted slope.
        use_brute : bool, optional
            Search for good starting values using a brute force (grid)
            optimizer. If False, a naive set of starting values is used.

        Returns
        -------
        results : HoltWintersResults class
            See statsmodels.tsa.holtwinters.HoltWintersResults

        Notes
        -----
        This is a full implementation of the Holt's exponential smoothing as
        per [1].

        References
        ----------
        [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
            and practice. OTexts, 2014.
        """
        return super(Holt, self).fit(smoothing_level=smoothing_level,
                                     smoothing_slope=smoothing_slope, damping_slope=damping_slope,
                                     optimized=optimized, start_params=start_params,
                                     initial_level=initial_level, initial_slope=initial_slope, use_brute=use_brute)
