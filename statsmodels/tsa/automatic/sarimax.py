"""automatic forecasting for SARIMAX models."""
import numpy as np
import warnings
import statsmodels.api as sm


def auto_order(endog, measure='aic', d=0, max_order=(3, 3), D=0, s=1,
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
    measure : str
        specifies which information criteria to use for model evaluation.
        'aic' is the default measure.
    d : int
        differencing factor.
    max_order : list
        maximum possible values of the parameters p and q.
    D : int
        seasonal differencing factor.
    s : int
        Number of seasonal differences.
    max_seasonal_order : list
        maximum possible values of the seasonal parameters P and Q.
    allow_intercept : boolean
        specifies whether to include the intercept or not.
    stepwise : Boolean
        specifies whether to use the stepwise algorithm or not.
    **spec : list of kwargs

    Returns
    -------
    allow_intercept : if intercept is present in the model.
    p : the autoregressive parameter for the SARIMAX model.
    d : the differencing parameter for the SARIMAX model.
    q : the moving average parameter for the SARIMAX model.
    if s>1, also returns,
    P : the seasonal autoregressive parameter for the SARIMAX model.
    D : the seasonal differencing parameter for the SARIMAX model.
    Q : the seasonal moving average parameter for the SARIMAX model.

    Notes
    -----
    Status : Work In Progress.
    Citation : Hyndman, Rob J., and Yeasmin Khandakar.
         "Automatic Time Series Forecasting: The forecast Package for R."

    Example
    --------
    from statsmodels import datasets
    macrodata = datasets.macrodata.load_pandas().data
    macrodata.index = pd.PeriodIndex(start='1959Q1', end='2009Q3', freq='Q')
    intercept, p, d, q = sarimax.auto_order(macrodata['infl'], d=0)
    """
    if not stepwise:
        n_p, n_q, n_P, n_Q = 0, 0, 0, 0
        min_aic = np.inf
        for p in range(max_order[0]):
            for q in range(max_order[1]):
                for P in range(max_seasonal_order[0]):
                    for Q in range(max_seasonal_order[1]):
                        # fit  the model
                        try:
                            if allow_intercept:
                                mod = sm.tsa.statespace.SARIMAX(
                                        endog,
                                        order=(p, d, q),
                                        seasonal_order=(P, D, Q, s),
                                        trend='c', **spec)
                                res = mod.fit(disp=False)
                                if getattr(res, measure) < min_aic:
                                    min_aic = getattr(res, measure)
                                    intercept_val = True
                                    n_p, n_q, n_P, n_Q = p, q, P, Q
                            mod = sm.tsa.statespace.SARIMAX(
                                    endog,
                                    order=(p, d, q),
                                    seasonal_order=(P, D, Q, s),
                                    **spec)
                            res = mod.fit(disp=False)
                            if getattr(res, measure) < min_aic:
                                    min_aic = getattr(res, measure)
                                    intercept_val = False
                                    n_p, n_q, n_P, n_Q = p, q, P, Q
                        except Exception as e:
                            warnings.warn('Could not fit model with {},{},{}'
                                          .format(p, d, q))
                            # aic_matrix[p, q] = np.inf
        if s > 1:
            return intercept_val, n_p, d, n_q, n_P, D, n_Q
        else:
            return intercept_val, n_p, d, n_q
    else:
        """stepwise algorithm for auto order"""
        """
        Initial Modelfitting.
        """
        if d + D <= 1:
            allow_intercept = False
        else:
            allow_intercept = True
        aic_vals = np.zeros(4)
        order_init = {
                        0: [2, 2],
                        1: [0, 0],
                        2: [1, 0],
                        3: [0, 1]
                    }
        seasonal_init = {
                            0: [1, 1],
                            1: [0, 0],
                            2: [1, 0],
                            3: [0, 1]
                        }
        for model in range(4):
            if s > 1:
                try:
                    if allow_intercept:
                        mod = sm.tsa.statespace.SARIMAX(
                                endog,
                                order=(order_init[model][0], d,
                                       order_init[model][1]), trend='c',
                                seasonal_order=(seasonal_init[model][0], D,
                                                seasonal_init[model][1], s),
                                **spec)
                        res = mod.fit(disp=False)
                        aic_vals[model] = getattr(res, measure)
                    else:
                        mod = sm.tsa.statespace.SARIMAX(
                                endog,
                                order=(order_init[model][0], d,
                                       order_init[model][1]),
                                seasonal_order=(seasonal_init[model][0], D,
                                                seasonal_init[model][1], s),
                                **spec)
                        res = mod.fit(disp=False)
                        aic_vals[model] = getattr(res, measure)
                except Exception as e:
                    aic_vals[mode] = np.inf
                    warnings.warn('Could not fit model ({},{},{})({},{},{},{})'
                                  .format(order_init[model][0], d,
                                          order_init[model][1],
                                          seasonal_init[model][0], D,
                                          seasonal_init[model][1], s))
            else:
                try:
                    if allow_intercept:
                        mod = sm.tsa.statespace.SARIMAX(
                                endog,
                                order=(order_init[model][0], d,
                                       order_init[model][1]), trend='c',
                                **spec)
                        res = mod.fit(disp=False)
                        aic_vals[model] = getattr(res, measure)
                    else:
                        mod = sm.tsa.statespace.SARIMAX(
                                endog,
                                order=(order_init[model][0], d,
                                       order_init[model][1]),
                                **spec)
                        res = mod.fit(disp=False)
                        aic_vals[model] = getattr(res, measure)
                except Exception as e:
                    aic_vals[model] = np.inf
                    warnings.warn('Could not fit model with {},{},{}'
                                  .format(order_init[model][0], d,
                                          order_init[model][0]))
        min_aic = aic_vals.min()
        model = np.where(aic_vals == min_aic)[0][0]
        p, q = order_init[model][0], order_init[model][1]
        if s > 1:
            P, Q = seasonal_init[model][0], seasonal_init[model][1]
        else:
            P, Q = 0, 0
        """
        Modifying model Parameters and refitting.
        """
        order_new = {
                        0: [p+1, q, P, Q],
                        1: [p-1, q, P, Q],
                        2: [p, q+1, P, Q],
                        3: [p, q-1, P, Q],
                        4: [p, q, P+1, Q],
                        5: [p, q, P-1, Q],
                        6: [p, q, P, Q+1],
                        7: [p, q, p, Q-1],
                        8: [p+1, q+1, P, Q],
                        9: [p-1, q-1, P, Q],
                        10: [p, q, P+1, Q+1],
                        11: [p, q, P-1, Q-1]
                    }
        new_val = [p, q, P, Q]
        if s > 1:
            for model in range(12):
                if (order_new[model][0] >= 0 and
                        order_new[model][0] <= max_order[0] and
                        order_new[model][1] >= 0 and
                        order_new[model][1] <= max_order[1] and
                        order_new[model][2] >= 0 and
                        order_new[model][2] <= max_seasonal_order[0] and
                        order_new[model][3] >= 0 and
                        order_new[model][3] <= max_seasonal_order[1]):
                    try:
                        if allow_intercept:
                            mod = sm.tsa.statespace.SARIMAX(
                                    endog,
                                    order=(order_new[model][0], d,
                                           order_new[model][1]),
                                    seasonal_order=(order_new[model][2], D,
                                                    order_new[model][3], s),
                                    trend='c', **spec)
                            res = mod.fit(disp=False)
                            if getattr(res, measure) < min_aic:
                                min_aic = getattr(res, measure)
                                new_val = order_new[model]
                        else:
                            mod = sm.tsa.statespace.SARIMAX(
                                    endog,
                                    order=(order_new[model][0], d,
                                           order_new[model][1]),
                                    seasonal_order=(order_new[model][2], D,
                                                    order_new[model][3], s),
                                    **spec)
                            res = mod.fit(disp=False)
                            if getattr(res, measure) < min_aic:
                                min_aic = getattr(res, measure)
                                new_val = order_new[model]
                    except Exception as e:
                        warnings.warn('Could not fit model ({},{},{})({},{},{},{})'
                                      .format(order_new[model][0], d,
                                              order_new[model][1],
                                              order_new[model][2], D,
                                              order_new[model][3], s))
            return allow_intercept, new_val[0], d, new_val[1], new_val[2], D, new_val[3]
        else:
            for model in range(12):
                if (order_new[model][0] >= 0 and
                    order_new[model][0] <= max_order[0] and
                    order_new[model][1] >= 0 and
                    order_new[model][1] <= max_order[1] and
                    order_new[model][2] >= 0):
                    try:
                        if allow_intercept:
                            mod = sm.tsa.statespace.SARIMAX(
                                    endog,
                                    order=(order_new[model][0], d,
                                           order_new[model][1]),
                                    trend='c', **spec)
                            res = mod.fit(disp=False)
                            if getattr(res, measure) < min_aic:
                                min_aic = getattr(res, measure)
                                new_val = order_new[model]
                        else:
                            mod = sm.tsa.statespace.SARIMAX(
                                    endog,
                                    order=(order_new[model][0], d,
                                           order_new[model][1]),
                                    **spec)
                            res = mod.fit(disp=False)
                            if getattr(res, measure) < min_aic:
                                min_aic = getattr(res, measure)
                                new_val = order_new[model]
                    except Exception as e:
                        warnings.warn('Could not fit model with p={}and q={}'
                                      .format(order_new[model][0],
                                              order_new[model][1]))
                return allow_intercept, new_val[0], d, new_val[1]
