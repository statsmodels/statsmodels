"""automatic forecasting for SARIMAX models."""
import numpy as np
import pandas as pd
import warnings
import statsmodels.api as sm


def auto_order(endog, criteria='aic', d=0, max_order=(3, 3), D=0, s=0,
               max_seasonal_order=(1, 1), allow_intercept=False,
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
        aic_matrix = np.zeros((max_order[0], max_order[1],
                              max_seasonal_order[0], max_seasonal_order[1]))
        for p in range(max_order[0]):
            for q in range(max_order[1]):
                for P in range(max_seasonal_order[0]):
                    for Q in range(max_seasonal_order[1]):
                        # fit  the model
                        try:
                            if allow_intercept:
                                mod = sm.tsa.statespace.SARIMAX(
                                        endog,
                                        order=(p, d, q), trend='c',
                                        seasonal_order=(P, D, Q, s), **spec)
                                res = mod.fit(disp=False)
                                aic_matrix[p, q, P, Q] = res.aic
                            else:
                                mod = sm.tsa.statespace.SARIMAX(
                                        endog,
                                        order=(p, d, q),
                                        seasonal_order=(P, D, Q, s), **spec)
                                res = mod.fit(disp=False)
                                aic_matrix[p, q, P, Q] = res.aic
                        except Exception as e:
                            warnings.warn('Could not fit model with {},{}'
                                           .format(p, q))
                            aic_matrix[p, q] = np.inf
                # print(res.aic)
        print(aic_matrix)
        min_aic = aic_matrix.min()
        print(min_aic)
        p, q, P, Q = np.where(aic_matrix == min_aic)
        return p[0], q[0], P[0], Q[0]
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
        # print(min_aic)
        model = int(np.where(aic_vals == min_aic)[0])
        p, q = order_init[model][0], order_init[model][1]
        new_p, new_q = p, q
        new_val = [p, q]
        # print(p, q)
        order_new = {
                        0: [p+1, q],
                        1: [p-1, q],
                        2: [p, q+1],
                        3: [p, q-1],
                        4: [p+1, q+1],
                        5: [p-1, q-1]
                    }
        for model in range(6):
            try:
                if (order_new[model][0] >= 0 and
                    order_new[model][0] <= max_order[0] and
                    order_new[model][1] >= 0 and
                    order_new[model][1] <= max_order[1]):
                        mod = sm.tsa.statespace.SARIMAX(
                                endog, order=(order_new[model][0], d,
                                              order_new[model][1]), **spec)
                        res = mod.fit(disp=False)
                        # print(res.aic)
                        # print(order_new[model][0], order_new[model][1])
                        if res.aic < min_aic:
                            min_aic = res.aic
                            new_val = order_new[model]
            except Exception as e:
                warnings.warn('Could not fit model with p={}and q={}'
                              .format(order_new[model][0],
                                      order_new[model][1]))
    return new_val[0], new_val[1]
