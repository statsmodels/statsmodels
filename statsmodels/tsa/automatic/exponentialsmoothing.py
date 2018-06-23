"""Automatic selection of the Exponential Smoothing Model."""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def auto_es(endog, seasonal_periods=1):
    trends = ["add", "mul", None]
    # damped
    seasonal = ["add", "mul", None]
    min_aic = np.inf
    model = [None, None]
    for t in trends:
        if seasonal_periods > 1:
            for s in seasonal:
                try:
                    mod = sm.tsa.ExponentialSmoothing(endog, trend=t,
                                                      seasonal=s,
                                                      seasonal_periods=seasonal_periods)
                    res = mod.fit()
                    # print(t, s, res.aic)
                    if res.aic < min_aic:
                        min_aic = res.aic
                        model = [t, s]
                except Exception as e:
                    # TODO add warning
                    pass e
        else:
            try:
                mod = sm.tsa.ExponentialSmoothing(endog, trend=t,
                                                  seasonal_periods=seasonal_periods)
                res = mod.fit()
                if res.aic < min_aic:
                    min_aic = res.aic
                    model = [t, None]
            except Exception as e:
                # TODO add warning
                pass e
    return model
