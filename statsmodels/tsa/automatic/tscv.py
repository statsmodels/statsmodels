"""Time-Series Cross Validation."""
import numpy as np
from statsmodels.tsa.base.tsa_model import TimeSeriesModel


def evaluate(endog, forecast, roll_window=30, **spec):
    """Evaluate time series cross validation.

    The function calculates the time series cross validation by making point
    forecasts over a window dataset. It uses the idea mentioned in Hyndman's
    book.

    Parameters
    ----------
    endog : list
        input containing the time series data over a period of time.

    forecast : forecast function/ TimeSeries Model
        can be a custom forecast function or a TimeSeries model.

    roll_window : int
        the length of the rolling window.

    **spec : kwargs

    Returns
    -------
    fcast_error : list
        the forecast error calculated by fitting the models for every window
        and making point forecast.

    Notes
    -----
    Reference : https://otexts.org/fpp2/accuracy.html
    https://robjhyndman.com/hyndsight/tscv/

    """
    endog = np.array(endog)
    nobs = len(endog)
    fcast_error = np.zeros(nobs) * np.nan
    for t in range(roll_window, nobs):
        training_endog = endog[t - roll_window:t]
        try:
            # If we were given a model class, create and fit it,
            # and then produce a forecast
            if isinstance(forecast, type) and issubclass(forecast, TimeSeriesModel):
                mod = forecast(training_endog, **spec)
                res = mod.fit()
                fcast = res.forecast(1)
            # Otherwise, assume we were given a function, and try
            # to call it
            else:
                fcast = forecast(training_endog, horizon=1)
            fcast_error[t] = endog[t] - np.squeeze(fcast)
        except Exception as e:
            pass
    return fcast_error
