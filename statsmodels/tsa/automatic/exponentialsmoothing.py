"""Automatic selection of the Exponential Smoothing Model."""
import numpy as np
import warnings
import statsmodels.api as sm


def auto_es(endog, measure='aic', seasonal_periods=1, damped=False,
            additive_only=False, alpha=None, beta=None,
            gamma=None, phi=None):
    """Perform automatic calculation of the parameters used in ES models.

    This uses a brute force approach to traverse through all the possible
    parameters of the model and select the best model based on the measure values.

    Parameters
    ----------
    endog : list
        input contains the time series data over a period of time.
    measure : str
        specifies which information measure to use for model evaluation.
        'aic' is the default measure.
    seasonal_periods : int
        the length of a seaonal period.
    damped : boolean
        includes damped trends.
    additive_only : boolean
        Allows only additive trend and seasonal models.
    alpha : float
        Smoothing level.
    beta : float
        Smoothing slope.
    gamma : float
        Smoothing seasonal.
    phi : float
        damping slope.

    Returns
    -------
    model : pair
        Pair containing trend,seasonal component type.
    Notes
    -----
    Status : Work In Progress.

    """
    if damped:
        trends = ["add", "mul"]
    else:
        trends = ["add", "mul", None]
    # damped
    seasonal = ["add", "mul", None]
    if additive_only:
        trends = ["add"]
        seasonal = ["add"]
    min_measure = np.inf
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
                    if getattr(res, measure) < min_measure:
                        min_measure = getattr(res, measure)
                        model = [t, s]
                except Exception as e:
                    warnings.warn(str(e))
        else:
            try:
                mod = sm.tsa.ExponentialSmoothing(endog, trend=t,
                                                  seasonal_periods=seasonal_periods)
                res = mod.fit(smoothing_level=alpha, smoothing_slope=beta,
                              smoothing_seasonal=gamma, damping_slope=phi)
                if getattr(res, measure) < min_measure:
                    min_measure = getattr(res, measure)
                    model = [t, None]
            except Exception as e:
                warnings.warn(str(e))
    return model
