"""
Statsmodel Eponential Smoothing
This code handles 15 different variation Standard Exponential Smoothing models

Created:     29/12/2013
Author: C Chan, dfrusdn
License: BSD(3clause)

How it works:

The main function exp_smoothing handles all the models and their variations.
There are wrappers for each model to make it simpler:

    Single Exponential Smoothing
    ----------------------------
    ses: Handles Simple Exponential Smoothing

    Double Exponential Smoothing
    ----------------------------
    brown_linear: Handles the special case for Brown Linear model(LES)
    holt_des: Handles Holt's Double Exponential Smoothing and Exponential
              trend method
    damp_es: Handles Damped-Trend Linear Exponential Smoothing and
             Multiplicative damped trend (Taylor  2003)

    Seasonal Smoothing & Triple Exponential Smoothing
    -------------------------------------------------
    seasonal_es: handles Simple Seasonal (both multiplicative
                 & additive models)
    exp_smoothing: Handles all variations of Holt-Winters Exponential
                  Smoothing multiplicative trend, additive trend,
                  multiplicative season, additive season, and dampening models
                  for all four variations. Also handles all models above.

FAQ

Q:Why are the values different from X's method?
A:Models use different values for their intitial starting point (such as
  ANOVA).
  For non-seasonal starting points we use a default values of bt = y[0]-y[1],
  st = y[0].
  For seasonal models we use the method in the NIST page for triple exponential
  smoothing http://www.itl.nist.gov/div898/handbook/pmc/section4/pmc435.htm

Q:Why are there 2 Nan's in my dataset?
A:The 2 Nans are used to backfill the forecast data to align with your
  timeseries data since values are not calculated using the forecating method.
  Error correction form for backcasting values is not implemented


TO DO: -Implement Solver for obtaining initial values

       - Decomposition to separate out trend and sasonal elements

       -Implement Error correction form for backcasting values

       - Summary elements: RMSE, Sum of squares residuals, AIC/AICc/BIC,
         Log-Likelihood, average mean square error, Hanna-Quinn,
         Mean Absolute Percentage Error, R^2, Ftest.

       -Implement other methods: Croston's method  for intermittent
        demand forecasting, Smooth Transition Exponential Smoothing (Taylor),
        single source of error model(SSOE)

       -renormalize seasonal data for multiplicative trends using methods
        in Archibald-Koehler (2003)

       -GARCH models used for variance once finish

       -Confidence bands based on "Prediction intervals for exponential
        smoothing using two new classes of state space models" (Hyndman 2003)

References
----------

::

    Exponential smoothing: The state of the art Part II, Everette S. Gardner,
        Jr. Houston, Texas 2005
    Forecasting with Exponential Smoothing: The State Space Approach,
        Hyndman, R.J., Koehler, A.B., Ord, J.K., Snyder, R.D. 2008
    Exponential Smoothing with a Damped Multiplicative Trend, James W.
        Taylor. International Journal of Forecasting, 2003
"""

import numpy as np
import statsmodels.tools.eval_measures as em


def exp_smoothing(y, alpha, gamma, delta=0, cycle=None, damp=1, initial=None,
                  trend='additive', forecast=None, season='additive',
                  output='data'):
    """
    Exponential Smoothing
    This function handles 15 different Standard Exponential Smoothing models

    Parameters
    ----------
    y: array
        Time series data
    alpha: float
        Smoothing factor for data between 0 and 1.
    gamma: non-zero integer or float
        Smoothing factor for trend generally between 0 and 1
    delta: non-zero integer or float
        Smoothing factor for season generally between 0 and 1
    cycle: int
        Length of cycles in a season. (ie: 12 for months, 4 for quarters)
    damp: non zero integer or float {default = 1}
        Autoregressive or damping parameter specifies a rate of decay
        in the trend. Generally 0<d<1
    initial: str, {'3avg'}(Optional)
        Indicate initial point for bt and y
        default:     bt = y[0]-y[1],    st = y[0]
        3avg:        Yields the average of the first 3 differences for bt.
    trend: str, {'additive','multiplicative', 'brown'}
        Indicate model type of trend default is 'additive'
        additive: uses additive models such as Holt's linear & Damped-Trend
        Linear Exponential Smoothing. Generalized as:
        s_t = a * y_t + (1-a) * (s_t-1 + b_t-1)
        b_t = g *(s_t - s_t-1) + (1 - g) * b_t-1
        multiplicative: uses multiplicative models such as Exponential trend &
        Taylor's modification of Pegels' model. Generalized as:
        s_t = a * y_t + (1-a) * (s_t-1 * b_t-1)
        b_t = g *(s_t / s_t-1) + (1 - g) * b_t-1
        brown: used to deal with the special cases in Brown linear smoothing
    forecast: int (Optional)
        Number of periods ahead.
    season: str, {'additive','multiplicative'}
        Indicate type of season default is 'additive'
    output: str, {'data', 'describe','forecast'}(Not implemented)

    Returns
    -------
    pdata: array
        Data that is smoothened using model chosen

    Notes
    -----
    This function is able to perform the following algorithms::

       * Simple Exponential Smoothing(SES)
       * Simple Seasonal models (both multiplicative and additive)
       * Brown's Linear Exponential Smoothing
       * Holt's Double Exponential Smoothing
       * Exponential trend method
       * Damped-Trend Linear Exponential Smoothing
       * Multiplicative damped trend (Taylor  2003)
       * Holt-Winters Exponential Smoothing:
       * multiplicative trend, additive trend, and damped models for both
       * multiplicative season, additive season, and damped models for both

    References
    ----------

    ::

     * Wikipedia
     * Forecasting: principles and practice by Hyndman & Athanasopoulos
     * NIST.gov http://www.itl.nist.gov/div898/handbook/pmc/section4/pmc433.htm
     * Oklahoma State SAS chapter 30 section 11
     * IBM SPSS Custom Exponential Smoothing Models
    """

    #Initialize data
    y = np.asarray(y)
    ylen = len(y)
    if ylen <= 3:
        raise ValueError("Cannot implement model, must have at least 4 "
                         "data points")
    if alpha == 0:
        raise ValueError("Cannot fit model, alpha must not be 0")
    #forcast length
    if forecast >= 1:
        ylen += 1

    #Setup array lengths
    sdata = np.zeros(ylen - 2)
    bdata = np.zeros(ylen - 2)

    # Setup seasonal values
    if type(cycle) is int:
        if ylen < 2 * cycle:
            raise ValueError("Cannot implement model, must be 2 at least "
                             "cycles long")
        #Setting b1 value
        bdata[0] = np.mean((y[cycle:2 * cycle] - y[:cycle]) / float(cycle))
        #Setup initial seasonal indicies
        #coerce cycle start lengths
        if len(y) % cycle != 0:
            cdata = y[:len(y) % cycle*-1].reshape(-1, cycle)
        else:
            cdata = y.reshape(-1, cycle)
        cdata = (cdata / cdata.mean(axis=1).reshape(-1, 1)).mean(axis=0)
        cdata = np.concatenate([cdata, np.zeros(ylen - 3)])
    else:
        #Initialize to bypass delta function with 0
        cycle = 0
        cdata = np.zeros(ylen)

    #Setting b1 value
    if gamma == 0:
        bdata[0] = 0
    elif initial == '3avg':
        bdata[0] = np.mean(abs(np.diff(y[:4])))
    else:
        bdata[0] = abs(y[0] - y[1])

    #Setting s1 value
    sdata[0] = y[0]

    #Equations & Update ylen to align array
    ylen -= 3
    for i in range(ylen):
        s = sdata[i]
        b = bdata[i]
        #handles multiplicative seasons
        if season == 'multiplicative':
            if trend == 'multiplicative':
                sdata[i + 1] = (alpha * (y[i + 2] / cdata[i]) + (1 - alpha) *
                                s * (b**damp))
                bdata[i + 1] = (gamma * (sdata[i + 1] / s) + (1 - gamma) *
                                (b ** damp))
                cdata[i + cycle] = (delta * (y[i + 2] / sdata[i + 1]) +
                                    (1 - delta) * cdata[i])
        #handles additive models
            else:
                sdata[i + 1] = (alpha * (y[i + 2] / cdata[i]) + (1 - alpha) *
                                (s + damp * b))
                bdata[i + 1] = (gamma * (sdata[i + 1] - s) + (1 - gamma) *
                                damp * b)
                cdata[i + cycle] = (delta * (y[i + 2] / sdata[i + 1]) +
                                    (1 - delta) * cdata[i])
        else:
            if trend == 'multiplicative':
                sdata[i + 1] = (alpha * (y[i + 2] - cdata[i]) + (1 - alpha) *
                                s * (b**damp))
                bdata[i + 1] = (gamma * (sdata[i + 1] / s) + (1 - gamma) *
                                (b ** damp))
                cdata[i + cycle] = (delta * (y[i + 2] - sdata[i + 1]) +
                                    (1 - delta) * cdata[i])
            #handles additive models
            else:
                sdata[i + 1] = (alpha * (y[i + 2] - cdata[i]) + (1 - alpha) *
                                (s + damp * b))
                bdata[i + 1] = (gamma * (sdata[i + 1] - s) + (1 - gamma) *
                                damp * b)
                cdata[i + cycle] = (delta * (y[i + 2] - sdata[i + 1]) +
                                    (1 - delta) * cdata[i])

    #Temporary fix: back fill data with Nan to align with y
    filx = [np.nan, np.nan]
    bdata = np.concatenate([filx, bdata])
    sdata = np.concatenate([filx, sdata])
    if season == 'multiplicative':
        if trend == 'multiplicative':
            pdata = (sdata * bdata) * cdata[:len(cdata) - cycle+3]
        else:
            pdata = (sdata + bdata) * cdata[:len(cdata) - cycle+3]
    else:
        if trend == 'multiplicative':
            pdata = sdata * bdata + cdata[:len(cdata) - cycle+3]
        #Handles special case for Brown linear
        elif trend == 'brown':
            at = 2 * sdata - bdata
            bt = alpha / (1 - alpha) * (sdata - bdata)
            sdata = at
            bdata = bt
            pdata = sdata + bdata + cdata[:len(cdata) - cycle+3]
        else:
            pdata = sdata + bdata + cdata[:len(cdata) - cycle+3]

    #Calculations for summary items (NOT USED YET)
    x1 = y[2:]
    x2 = pdata[2:len(y)]
    rmse = em.rmse(x1, x2)

    #forcast
    if forecast >= 2:
        #Configure damp
        if damp == 1:
            m = np.arange(2, forecast+1)
        else:
            m = np.cumsum(damp ** np.arange(1, forecast+1))
            m = m[1:]

        #Config season
        if cycle == 0:
            if season == 'multiplicative':
                c = 1
            else:
                c = 0
        elif forecast > cycle:
            raise NotImplementedError("Forecast beyond cycle length is "
                                      "unavailable")
        else:
            c = cdata[ylen+1:]
            c = c[:forecast-1]

        #Config trend
        if season == 'multiplicative':
            if trend == 'multiplicative':
                fdata = sdata[ylen] * (bdata[ylen] ** m) * c
            else:
                fdata = sdata[ylen] + m * bdata[ylen] * c
        else:
            if trend == 'multiplicative':
                fdata = sdata[ylen] * (bdata[ylen] ** m) + c
            else:
                fdata = sdata[ylen] + m * bdata[ylen] + c

        pdata = np.append(pdata, fdata)
    return pdata

########################################################
######################Exponential Smoothing Wrappers####
########################################################


def ses(y, alpha, forecast=None, output='data'):
    """
    Simple Exponential Smoothing (SES)
    This function is a wrapper that performs simple exponential smoothing.

    Parameters
    ----------
    y: array
        Time series data
    alpha: float
        Smoothing factor for data between 0 and 1.
    forecast: int (Optional)
        Number of periods ahead.
    output: str, {'data', 'describe','forecast'}(Not implemented)


    Returns
    -------
    pdata: array
        Data that is smoothened with forecast

    Notes
    -----
    It is used when there is no trend or seasonality.

    .. math::

      s_t = alpha * y_t + (1-a) * (s_t-1)

    Forecast equation.

    .. math::

       y_t(n) = S_t


    References
    ----------

    ::

     * Wikipedia
     * Forecasting: principles and practice by Hyndman & Athanasopoulos
     * NIST.gov
     * Oklahoma State SAS chapter 30 section 11
     * IBM SPSS Custom Exponential Smoothing Models
    """
    s_es = exp_smoothing(y, alpha, 0, 0, None, 0, None, 'additive',
                         forecast, 'additive', output)

    return s_es


################Double Exponential Smoothing###############
def brown_linear(y, alpha, forecast=None, output='data'):
    """
    Brown's Linear Exponential Smoothing (LES)
    This function a special case of the Holt's Exponential smoothing
    using alpha as the smoothing factor and smoothing trend factor.


    Parameters
    ----------
    y: array
        Time series data
    alpha: float
        Smoothing factor for data between 0 and 1.
    forecast: int (Optional)
        Number of periods ahead.
    output: str, {'data', 'describe','forecast'}(Not implemented)


    Returns
    -------
    pdata: array
        Data that is smoothened with forecast

    Notes
    -----
    It is used when there is a trend but no seasonality.

    .. math::

       s_t = a * y_t + (1 - a) * (s_t-1)
       b_t = a *(s_t - s_t-1) + (1 - a) * T_t-1
       a'=2*(s_t - b_t)
       b'=a/(1-a)*(s_t - b_t)
       F_t = a' + b'

    Forecast equation

    .. math::

       F_t+n = a' + m * b'

    References
    ----------

    ::

     * Wikipedia
     * Forecasting: principles and practice by Hyndman & Athanasopoulos
     * IBM SPSS Custom Exponential Smoothing Models
    """
    brown = exp_smoothing(y, alpha, alpha, 0, None, 1, None, 'brown',
                          forecast, 'additive', output)

    return brown


#General Double Exponential Smoothing Models
def holt_des(y, alpha, gamma, forecast=None, trend='additive',
             initial=None, output='data',):
    """
    Holt's Double Exponential Smoothing
    Use when linear trend is present with no seasonality.
    Multiplicative model is used for exponential or strong trends
    such as growth.


    Parameters
    ----------
    y: array
        Time series data
    alpha: float
        Smoothing factor for data between 0 and 1.
    gamma: non-zero integer or float
        Smoothing factor for trend generally between 0 and 1
    forecast: int (Optional)
        Number of periods ahead.
    trend: str {'addative', 'multiplicative'}
        Additive use when trend is linear
        Multiplicative is used when trend is exponential
    initial: str, {'3avg'}(Optional)
        Indicate initial point for bt and y
        default:     bt = y[0]-y[1],    st = y[0]
        3avg:        Yields the average of the first 3 differences for bt.

    output: str, {'data', 'describe','forecast'}(Not implemented)


    Returns
    -------
    pdata: array
        Data that is smoothened with forecast

    Notes
    -----
    Based on Holt's 1957 method. It is similar to ARIMA(0, 2, 2).
    Additive model Equations:

    .. math ::

       s_t=a * y_t + (1 - a)(s_t-1 + b_t-1)
       b_t=g * (s_t - s_t-1) + (1-g) * b_t-1

    Forecast (n periods):

    .. math ::

       F_t+n = S_t + m * b_t

    The multiplicative or exponential model is used for models with an
    exponential trend. (Pegels 1969, Hyndman 2002)

    References
    ----------

    ::

     * Wikipedia
     * Forecasting: principles and practice by Hyndman & Athanasopoulos
     * IBM SPSS Custom Exponential Smoothing Models
     * Exponential Smoothing with a Damped Multiplicative Trend,
       James W. Taylor. International Journal of Forecasting, 2003
    """
    holt = exp_smoothing(y, alpha, gamma, 0, None, 1, initial, trend,
                         forecast, 'additive', output)

    return holt


#Damped-Trend Linear Exponential Smoothing Models
def damp_es(y, alpha, gamma, damp=1, forecast=None, trend='additive',
            initial=None, output='data',):
    """
    Damped-Trend Linear Exponential Smoothing
    Multiplicative damped trend (Taylor  2003)
    Use when linear trend is decaying and with no seasonality
    Multiplicative model used for exponential trend decay

    Parameters
    ----------
    y: array
        Time series data
    alpha: float
        Smoothing factor for data between 0 and 1.
    gamma: non-zero integer or float
        Smoothing factor for trend generally between 0 and 1
    damp:  non-zero integer or float {default = 1}
        Specify the rate of decay in a trend.
        If d=1 identical to Holt method or multiplicative method
        if d=0 identical to simple exponential smoothing
        (Gardner and McKenzie)
        d >1 applied to srongly trending series.(Tashman and Kruk 1996)
    forecast: int (Optional)
        Number of periods ahead.
    trend: str {'addative', 'multiplicative'}
        Additive use when trend is linear
        Multiplicative is used when trend is exponential

    output: str, {'data', 'describe','forecast'}(Not implemented)


    Returns
    -------
    pdata: array
        Data that is smoothened with forecast

    Notes
    -----
    Based on the Garner McKenzie 1985 modification of the
    Holt linear method to address the tendency of overshooting
    short term trends. This added improvements in prediction
    accuracy (Makridakiset al., 1993; Makridakis & Hibon, 2000).
    It is similar to ARIMA(1, 1, 2).

    The multiplicative model is based on the modified Pegels model
    with an extra dampening parameter by Taylor in 2003.
    It slightly outperforms the Holt and Pegels models.
    Multiplicative models can be used for log transformed
    data (Pegels 1969).


    References
    ----------

    ::

     * Wikipedia
     * Forecasting: principles and practice by Hyndman & Athanasopoulos
     * IBM SPSS Custom Exponential Smoothing Models
     * Exponential Smoothing with a Damped Multiplicative Trend, James W.
       Taylor. International Journal of Forecasting, 2003

    """
    dampend = exp_smoothing(y, alpha, gamma, 0, None, damp, initial, trend,
                            forecast, 'additive', output)

    return dampend


################Seasonal Exponential Smoothing###############
def seasonal_es(y, alpha, delta, cycle, forecast=None,
                season='additive', output='data',):
    """
    Simple Seasonal Smoothing
    Use when there is a seasonal element but no trend
    Multiplicative model is used exponential seasonal components

    Parameters
    ----------
    y: array
        Time series data
    alpha: float
        Smoothing factor for data between 0 and 1.
    delta: non-zero integer or float
        Smoothing factor for trend generally between 0 and 1
    cycle: int
        Length of cycles in a season. (ie: 12 for months, 4 for quarters)
    forecast: int (Optional)
        Number of periods ahead. Note that you can only forecast up to
        1 cycle ahead.
    season: str, {'additive','multiplicative'}
        Indicate type of season default is 'additive'

    output: str, {'data', 'describe','forecast'}(Not implemented)

    Returns
    -------
    pdata: array
        Data that is smoothened with forecast

    Notes
    -----
    You need at least 2 periods of data to run seasonal algorithms.

    References
    ----------

    ::

     * Wikipedia
     * Forecasting: principles and practice by Hyndman & Athanasopoulos
     * IBM SPSS Custom Exponential Smoothing Models
     * Exponential Smoothing with a Damped Multiplicative Trend, James W.
       Taylor. International Journal of Forecasting, 2003
    """

    ssexp = exp_smoothing(y, alpha, 0, delta, cycle, 1, None, 'additive',
                          forecast, 'additive', output)

    return ssexp
