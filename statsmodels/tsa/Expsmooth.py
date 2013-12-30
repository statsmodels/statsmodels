'''
contains
--------
Exponential Smoothing
Holt Winters Exponential smoothing



License: BSD-3


TODO
----
[]Add Holt double Exponential smoothing
[]Add time series decomposition
[]Fix Holt Winters Additive model (just an if statement and change
                                    seasonal calculation from division to
                                    addition)
[]Add/improve Forcasting for Holt Winters to take multiple variables
  []Add confidence intervals to forcast
[]Add solver to fit Holt Winters parameters
[]Improve data summary RMS error, Sum of Square Residual,
'''

import numpy as np

def ExpSmoothing(y, alpha, forecast=None):
    '''
    Brown's simple exponential smoothing
    Parameters
    ----------
    y: array
        Time series data

    alpha:  float
        Smoothing factor between 0 and 1. Alpha can be calculated using
        Method of least squares
    forecast: int
        Number of periods ahead. Forcast with bootstrapping method
        s_t+1 = y_origin*a + (1-a)s_t

    Returns
    -------
    weighted data: array
        Data that is filtered using
        y_t = a * y_t + a * (1-a)^1 * y_t-1 +... + a*(1-a)^n * y_t-n


    References
    ----------
    Wikipedia
    Forecasting based on NIST equation for Bootstrapping
    http://www.itl.nist.gov/div898/handbook/pmc/section4/pmc432.htm#Single%20Exponential%20Smoothing%20with
    
    
    '''
    #Initialize data
    y = np.asarray(y)

    ylen = len(y)
    weights = alpha * ((1 - alpha)**np.arange(0, ylen))
    wdata = y * weights
    if forecast >= 0:
        fdata = np.zeros(forecast+1)
        fdata[0] = wdata[ylen - 1]
        for i in range(forecast):
            f = fdata[i]
            fdata[i + 1] = alpha * y[ylen - 1] + (1 - alpha) * f
        fdata = np.delete(fdata, 0)
        wdata = np.append(wdata, fdata)
        return wdata
    else:
        return wdata

