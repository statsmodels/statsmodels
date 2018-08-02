"""Time-Series Cross Validation."""
import numpy as np
from statsmodels.tsa.base.tsa_model import TimeSeriesModel

def evaluate(endog, forecast, roll_window=30, **spec):
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
