# pylint: disable=W0201

from statsmodels.compat.python import iteritems, string_types, range
import numpy as np
from statsmodels.tools.decorators import cache_readonly
import pandas as pd

from . import var_model as _model
from . import util
from . import plotting

FULL_SAMPLE = 0
ROLLING = 1
EXPANDING = 2


def _get_window_type(window_type):
    if window_type in (FULL_SAMPLE, ROLLING, EXPANDING):
        return window_type
    elif isinstance(window_type, string_types):
        window_type_up = window_type.upper()

        if window_type_up in ('FULL SAMPLE', 'FULL_SAMPLE'):
            return FULL_SAMPLE
        elif window_type_up == 'ROLLING':
            return ROLLING
        elif window_type_up == 'EXPANDING':
            return EXPANDING

    raise Exception('Unrecognized window type: %s' % window_type)


class DynamicVAR(object):
    """
    Estimates time-varying vector autoregression (VAR(p)) using
    equation-by-equation least squares

    Parameters
    ----------
    data : pandas.DataFrame
    lag_order : int, default 1
    window : int
    window_type : {'expanding', 'rolling'}
    min_periods : int or None
        Minimum number of observations to require in window, defaults to window
        size if None specified
    trend : {'c', 'nc', 'ct', 'ctt'}
        TODO

    Returns
    -------
    **Attributes**:

    coefs : Panel
        items : coefficient names
        major_axis : dates
        minor_axis : VAR equation names
    """
    def __init__(self, data, lag_order=1, window=None, window_type='expanding',
                 trend='c', min_periods=None):
        self.lag_order = lag_order

        self.names = list(data.columns)
        self.neqs = len(self.names)

        self._y_orig = data

        # TODO: deal with trend
        self._x_orig = _make_lag_matrix(data, lag_order)
        self._x_orig['intercept'] = 1

        (self.y, self.x, self.x_filtered, self._index,
         self._time_has_obs) = _filter_data(self._y_orig, self._x_orig)

        self.lag_order = lag_order
        self.trendorder = util.get_trendorder(trend)

        self._set_window(window_type, window, min_periods)

    def _set_window(self, window_type, window, min_periods):
        self._window_type = _get_window_type(window_type)

        if self._is_rolling:
            if window is None:
                raise Exception('Must pass window when doing rolling '
                                'regression')

            if min_periods is None:
                min_periods = window
        else:
            window = len(self.x)
            if min_periods is None:
                min_periods = 1

        self._window = int(window)
        self._min_periods = min_periods

    @cache_readonly
    def T(self):
        """
        Number of time periods in results
        """
        return len(self.result_index)

    @property
    def nobs(self):
        # Stub, do I need this?
        data = dict((eq, r.nobs) for eq, r in iteritems(self.equations))
        return pd.DataFrame(data)

    @cache_readonly
    def equations(self):
        eqs = {}
        for col, ts in iteritems(self.y):
            # TODO: Remove in favor of statsmodels implemetation
            model = pd.ols(y=ts, x=self.x, window=self._window,
                           window_type=self._window_type,
                           min_periods=self._min_periods)

            eqs[col] = model

        return eqs

    @cache_readonly
    def coefs(self):
        """
        Return dynamic regression coefficients as Panel
        """
        data = {}
        for eq, result in iteritems(self.equations):
            data[eq] = result.beta

        panel = pd.Panel.fromDict(data)

        # Coefficient names become items
        return panel.swapaxes('items', 'minor')

    @property
    def result_index(self):
        return self.coefs.major_axis

    @cache_readonly
    def _coefs_raw(self):
        """
        Reshape coefficients to be more amenable to dynamic calculations

        Returns
        -------
        coefs : (time_periods x lag_order x neqs x neqs)
        """
        coef_panel = self.coefs.copy()
        del coef_panel['intercept']

        coef_values = coef_panel.swapaxes('items', 'major').values
        coef_values = coef_values.reshape((len(coef_values),
                                           self.lag_order,
                                           self.neqs, self.neqs))

        return coef_values

    @cache_readonly
    def _intercepts_raw(self):
        """
        Similar to _coefs_raw, return intercept values in easy-to-use matrix
        form

        Returns
        -------
        intercepts : (T x K)
        """
        return self.coefs['intercept'].values

    @cache_readonly
    def resid(self):
        data = {}
        for eq, result in iteritems(self.equations):
            data[eq] = result.resid

        return pd.DataFrame(data)

    def forecast(self, steps=1):
        """
        Produce dynamic forecast

        Parameters
        ----------
        steps

        Returns
        -------
        forecasts : pandas.DataFrame
        """
        output = np.empty((self.T - steps, self.neqs))

        y_values = self.y.values
        y_index_map = dict((d, idx) for idx, d in enumerate(self.y.index))
        result_index_map = dict((d, idx) for idx, d in enumerate(self.result_index))

        coefs = self._coefs_raw
        intercepts = self._intercepts_raw

        # can only produce this many forecasts
        forc_index = self.result_index[steps:]
        for i, date in enumerate(forc_index):
            # TODO: check that this does the right thing in weird cases...
            idx = y_index_map[date] - steps
            result_idx = result_index_map[date] - steps

            y_slice = y_values[:idx]

            forcs = _model.forecast(y_slice, coefs[result_idx],
                                    intercepts[result_idx], steps)

            output[i] = forcs[-1]

        return pd.DataFrame(output, index=forc_index, columns=self.names)

    def plot_forecast(self, steps=1, figsize=(10, 10)):
        """
        Plot h-step ahead forecasts against actual realizations of time
        series. Note that forecasts are lined up with their respective
        realizations.

        Parameters
        ----------
        steps :
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(figsize=figsize, nrows=self.neqs,
                                 sharex=True)

        forc = self.forecast(steps=steps)
        dates = forc.index

        y_overlay = self.y.reindex(dates)

        for i, col in enumerate(forc.columns):
            ax = axes[i]

            y_ts = y_overlay[col]
            forc_ts = forc[col]

            y_handle = ax.plot(dates, y_ts.values, 'k.', ms=2)
            forc_handle = ax.plot(dates, forc_ts.values, 'k-')

        lines = (y_handle[0], forc_handle[0])
        labels =  ('Y', 'Forecast')
        fig.legend(lines,labels)
        fig.autofmt_xdate()

        fig.suptitle('Dynamic %d-step forecast' % steps)

        # pretty things up a bit
        plotting.adjust_subplots(bottom=0.15, left=0.10)
        plt.draw_if_interactive()

    @property
    def _is_rolling(self):
        return self._window_type == ROLLING

    @cache_readonly
    def r2(self):
        """Returns the r-squared values."""
        data = dict((eq, r.r2) for eq, r in iteritems(self.equations))
        return pd.DataFrame(data)

class DynamicPanelVAR(DynamicVAR):
    """
    Dynamic (time-varying) panel vector autoregression using panel ordinary
    least squares

    Parameters
    ----------
    """
    def __init__(self, data, lag_order=1, window=None, window_type='expanding',
                 trend='c', min_periods=None):
        self.lag_order = lag_order
        self.neqs = len(data.columns)

        self._y_orig = data

        # TODO: deal with trend
        self._x_orig = _make_lag_matrix(data, lag_order)
        self._x_orig['intercept'] = 1

        (self.y, self.x, self.x_filtered, self._index,
         self._time_has_obs) = _filter_data(self._y_orig, self._x_orig)

        self.lag_order = lag_order
        self.trendorder = util.get_trendorder(trend)

        self._set_window(window_type, window, min_periods)


def _filter_data(lhs, rhs):
    """
    Data filtering routine for dynamic VAR

    lhs : DataFrame
        original data
    rhs : DataFrame
        lagged variables

    Returns
    -------

    """
    def _has_all_columns(df):
        return np.isfinite(df.values).sum(1) == len(df.columns)

    rhs_valid = _has_all_columns(rhs)
    if not rhs_valid.all():
        pre_filtered_rhs = rhs[rhs_valid]
    else:
        pre_filtered_rhs = rhs

    index = lhs.index.union(rhs.index)
    if not index.equals(rhs.index) or not index.equals(lhs.index):
        rhs = rhs.reindex(index)
        lhs = lhs.reindex(index)

        rhs_valid = _has_all_columns(rhs)

    lhs_valid = _has_all_columns(lhs)
    valid = rhs_valid & lhs_valid

    if not valid.all():
        filt_index = rhs.index[valid]
        filtered_rhs = rhs.reindex(filt_index)
        filtered_lhs = lhs.reindex(filt_index)
    else:
        filtered_rhs, filtered_lhs = rhs, lhs

    return filtered_lhs, filtered_rhs, pre_filtered_rhs, index, valid

def _make_lag_matrix(x, lags):
    data = {}
    columns = []
    for i in range(1, 1 + lags):
        lagstr = 'L%d.'% i
        lag = x.shift(i).rename(columns=lambda c: lagstr + c)
        data.update(lag._series)
        columns.extend(lag.columns)

    return pd.DataFrame(data, columns=columns)

class Equation(object):
    """
    Stub, estimate one equation
    """

    def __init__(self, y, x):
        pass

if __name__ == '__main__':
    import pandas.util.testing as ptest

    ptest.N = 500
    data = ptest.makeTimeDataFrame().cumsum(0)

    var = DynamicVAR(data, lag_order=2, window_type='expanding')
    var2 = DynamicVAR(data, lag_order=2, window=10,
                      window_type='rolling')


