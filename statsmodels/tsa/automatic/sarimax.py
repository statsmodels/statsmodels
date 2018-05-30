"""automatic forecasting for SARIMAX models."""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def auto_order(endog, criteria='aic', d=0, max_order=(3, 3),
                stepwise=False, **spec):
    """Perform automatic calculation of the parameters used in SARIMAX models.

    This uses two different approaches to clacluate the parameters:
    1. Uses a brute force approach to traverse through all the possible
        parameters of the model.
    2. Uses the stepwise algorithm specified in Hyndman's Paper.

    Parameters
    ----------
    endog : list
        input contains the time series data over a period of time.
    criteria : str
        specifies which information criteria to use for model evaluation.
    d : int
        differencing factor.
    max_order : list
        maximum possible values of the parameters p and q.
    stepwise : Boolean
        specifies whether to use the stepwise algorithm or not.
    **spec : list of kwargs

    Returns
    -------
    p : the autoregressive parameter for the SARIMAX model.
    q : the moving average parameter for the SARIMAX model.

    Notes
    -----
    Status : Work In Progress.
    Citation : Hyndman, Rob J., and Yeasmin Khandakar.
         "Automatic Time Series Forecasting: The forecast Package for R."

    """
    if not stepwise:
        aic_matrix = np.zeros((max_order[0], max_order[1]))
        for p in range(max_order[0]):
            for q in range(max_order[1]):
                # fit  the model
                try:
                    mod = sm.tsa.statespace.SARIMAX(endog,
                                                    order=(p, d, q), **spec)
                    res = mod.fit(disp=False)
                except Exception as e:
                    raise e
                aic_matrix[p, q] = res.aic
                # print(res.aic)
        # print(aic_matrix)
        min_aic = aic_matrix.min()
        # print(min_aic)
        p, q = np.where(aic_matrix == min_aic)
        return p, q
    else:
        """stepwise algorithm for auto order"""
        aic_vals = np.zeros(4)
        order_init = {
                        0: [2, 2],
                        1: [0, 0],
                        2: [1, 0],
                        3: [0, 1]
                    }
        for model in range(4):
            mod = sm.tsa.statespace.SARIMAX(
                endog, order=(order_init[model][0], d, order_init[model][1]),
                **spec)
            res = mod.fit(disp=False)
            aic_vals[model] = res.aic
        # print(aic_vals)
        min_aic = aic_vals.min()
        model = int(np.where(aic_vals == min_aic)[0])
        p, q = order_init[model][0], order_init[model][1]
        new_p, new_q = p, q
        # only p varies by +1
        if(p + 1 <= max_order[0]):
            mod = sm.tsa.statespace.SARIMAX(endog, order=(p + 1, d, q), **spec)
            res = mod.fit(disp=False)
            # print(res.aic)
            if (res.aic < min_aic):
                new_p = p + 1
                min_aic = res.aic
        # only p varies by -1
        if(p - 1 >= 0):
            mod = sm.tsa.statespace.SARIMAX(endog, order=(p - 1, d, q), **spec)
            res = mod.fit(disp=False)
            # print(res.aic)
            if (res.aic < min_aic):
                new_p = p - 1
                min_aic = res.aic
        # only q varies by +1
        if(q + 1 <= max_order[1]):
            mod = sm.tsa.statespace.SARIMAX(endog, order=(p, d, q + 1), **spec)
            res = mod.fit(disp=False)
            # print(res.aic)
            if (res.aic < min_aic):
                new_q = q + 1
                min_aic = res.aic
        # only q varies by -1
        if(q - 1 >= 0):
            mod = sm.tsa.statespace.SARIMAX(endog, order=(p, d, q - 1), **spec)
            res = mod.fit(disp=False)
            # print(res.aic)
            if (res.aic < min_aic):
                new_q = q - 1
                min_aic = res.aic
        # both p and q vary by +1
        if(p + 1 <= max_order[0]) and (q + 1 <= max_order[1]):
            mod = sm.tsa.statespace.SARIMAX(endog,
                                            order=(p + 1, d, q + 1), **spec)
            res = mod.fit(disp=False)
            # print(res.aic)
            if (res.aic < min_aic):
                new_p = p + 1
                new_q = q + 1
                min_aic = res.aic
        # both p and q vary by -1
        if(p - 1 >= 0) and (q - 1 >= 0):
            mod = sm.tsa.statespace.SARIMAX(endog,
                                            order=(p - 1, d, q - 1), **spec)
            res = mod.fit(disp=False)
            # print(res.aic)
            if (res.aic < min_aic):
                new_p = p - 1
                new_q = q - 1
                min_aic = res.aic
        return new_p, new_q
