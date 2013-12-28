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
[]Add pandas wrapper
'''

import numpy as np
from .utils import _maybe_get_pandas_wrapper


def ExpSmoothing(y, alpha):
    '''Brown's simple exponential smoothing

    Parameters
    ----------
    y: array
        Time series data

    alpha: non zero integer or float
        Smoothing factor between 0 and 1

    Returns
    -------
    weighted data: array
        Data that is filtered using
        y_t = a * y_t + a * (1-a)^1 * y_t-1 +... + a*(1-a)^n * y_t-n


    Notes
    -------
    '''
    #Initialize data
    y=np.asarray(y)
    ylen =len(y)
    weights =np.zeros(ylen)
    wdata =np.zeros(ylen)
    weights = alpha * ((1-alpha)**np.arange(0,ylen))

    wdata=y*weights

    return wdata



def holtwinters(y, alpha, beta, gamma, c, seasonal ='multiplicative'):
    '''Holts Winter Exponential Smoothing
    aka Triple Exponential Smoothing

    Parameters
    ----------
    y: array
        Time series data

    alpha: non zero integer or float
        Smoothing for data generally between 0 and 1

    beta: non zero integer or float
        Smoothing for trend generally between 0 and 1

    gamma: non zero integer or float
        Smoothing parameter for seasonal component generally between 0 and 1

    c: integer
        Cycles or periods

    seasonal:
        'multiplicative' or 'additive' models of seasons default is multiplicative


    Returns
    -------
    smoothData: Array
        Array of Exponential smoothing values

    Notes
    -------

    '''
    #Initialize data
    y=np.asarray(y)
    ylen =len(y)
    if alpha==0:
        raise ValueError("Cannot fit model, alpha must not be 0")
    if ylen % c !=0:
        return None


    #Compute initial Bt, Initial values for the trend factor using the first two complete c periods.

    fc =float(c)
    ybar2=y[c:(2*c)].sum()/fc
    ybar1=y[:c].sum()/fc
    Bt =(ybar2 - ybar1) / fc

    #Compute for the level estimate At using Bt above.
    tbar =sum(range(1, c+1)) / fc
    At =ybar1 - Bt * tbar

    #Compute for initial indices
    idex=At + Bt*(np.arange(1, ylen+1))
    I=y/idex

    S=np.zeros(ylen+ c)
    for i in range(c):
        S[i] =(I[i] + I[i+c]) / 2.0

    #Normalize so S[i] for i in [0, c) will add to c.
    tS =c/S[:c].sum()
    S[:c] *=tS


    # Holt - winters proper ...
    smoothData =np.zeros(ylen+ c)


    if seasonal =='additive':
        for i in range(ylen):
            Atm1 =At
            Btm1 =Bt
            #Equations for additive model
            #Data
            At =alpha * (y[i] - S[i]) + (1.0-alpha) * (Atm1 + Btm1)
            #Trend
            Bt =beta * (At - Atm1) + (1- beta) * Btm1
            #Seasonal
            S[i+c] =gamma * (y[i] - At) + (1.0 - gamma) * S[i]
            smoothData[i]=At + (Bt * (i+1)) + S[i]
    else:
        #Equations for multiplicative model

        for i in range(ylen):
            Atm1 =At
            Btm1 =Bt

            #Data
            At =alpha * y[i] / S[i] + (1.0-alpha) * (Atm1 + Btm1)
            #Trend
            Bt =beta * (At - Atm1) + (1- beta) * Btm1
            #Seasonal
            S[i+c] =gamma * y[i] / At + (1.0 - gamma) * S[i]
            smoothData[i]=(At + Bt * (i+1)) * S[i]

    #Forecast for next c periods:
    #for m in range(c):
    # print "forecast:", (At + Bt* (m+1))* S[ylen + m]
    return smoothData
