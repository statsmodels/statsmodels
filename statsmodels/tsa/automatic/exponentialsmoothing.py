"""Automatic selection of the Exponential Smoothing Model."""
import numpy as np
import warnings
import statsmodels.api as sm

from statsmodels.tsa.automatic import transform


def auto_es(endog, measure='aic', seasonal_periods=1, damped=False,
            lambda_val=None, additive_only=False, alpha=None, beta=None,
            gamma=None, phi=None):
    if damped:
        trends = ["add", "mul"]
    else:
        trends = ["add", "mul", None]
    # damped
    seasonal = ["add", "mul", None]
    if lambda_val = 'auto':
        lambda_val = transform.predict_lambda(endog)
    else:
        if lambda_val not None:
            additive_only = True
    if additive_only:
        trends = ["add"]
        seasonal = ["add"]
    min_aic = np.inf
    model = [None, None]
    for t in trends:
        if seasonal_periods > 1:
            for s in seasonal:
                try:
                    mod = sm.tsa.ExponentialSmoothing(endog, trend=t,
                                                      seasonal=s,
                                                      seasonal_periods=seasonal_periods)
                    res = mod.fit(smoothing_level=alpha, smoothing_slope=beta,
                                  smoothing_seasonal=gamma, damping_slope=phi)
                    # print(t, s, res.aic)
                    if getattr(res, measure) < min_aic:
                        min_aic = getattr(res, measure)
                        model = [t, s]
                except Exception as e:
                    warnings.warn(str(e))  # TODO add warning
        else:
            try:
                mod = sm.tsa.ExponentialSmoothing(endog, trend=t,
                                                  seasonal_periods=seasonal_periods)
                res = mod.fit(smoothing_level=alpha, smoothing_slope=beta,
                              smoothing_seasonal=gamma, damping_slope=phi)
                if getattr(res, measure) < min_aic:
                    min_aic = getattr(res, measure)
                    model = [t, None]
            except Exception as e:
                warnings.warn(str(e))  # TODO add warning
    return model
