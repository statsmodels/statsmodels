#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 09:35:35 2017

@author: tvzyl
"""
import numpy as np
import pandas as pd
import warnings

from statsmodels.compat.python import iteritems

from scipy.stats import boxcox
try:
    from scipy.special import inv_boxcox
except ImportError:
    inv_boxcox = lambda x, lam6da: (x**lam6da-1)/lam6da if lam6da != 0 else np.log(x)
    
from scipy.spatial.distance import sqeuclidean
from scipy.optimize import minimize, basinhopping, brute


class HoltWintersResult(object):
    def __init__(self, **kwargs):
        for key, value in iteritems(kwargs):
            setattr(self, key, value)        

def holt_init(x, xi, p, y, l, b):
        p[xi] = x
        alpha,beta,_,l0,b0,phi = p[:6];
        alphac = 1 - alpha
        betac = 1 - beta
        y_alpha = alpha*y
        l[:] = 0; b[:] = 0;
        l[0] = l0; b[0] = b0        
        return alpha,beta,phi,alphac,betac,y_alpha

def holt__(x, xi, p, y, l, b, s, m, n, max_seen):
    alpha,beta,phi,alphac,betac,y_alpha = holt_init(x, xi, p, y, l, b)
    for i in range(1, n):
        l[i] = (y_alpha[i-1]) + (alphac * (l[i-1]))
    return sqeuclidean(l, y)

#(M,) (Md,)
def holt_mul_dam(x, xi, p, y, l, b, s, m, n, max_seen):
    alpha,beta,phi,alphac,betac,y_alpha = holt_init(x, xi, p, y, l, b)        
    if alpha==0.0: return max_seen
    if beta > alpha: return max_seen
    for i in range(1, n):
        l[i] = (y_alpha[i-1]) + (alphac * (l[i-1] * b[i-1]**phi))
        b[i] = (beta * (l[i] / l[i-1])) + (betac * b[i-1]**phi)
    return sqeuclidean(l*b**phi, y)

#(A,) (Ad,)
def holt_add_dam(x, xi, p, y, l, b, s, m, n, max_seen):        
    alpha,beta,phi,alphac,betac,y_alpha = holt_init(x, xi, p, y, l, b)        
    if alpha==0.0: return max_seen
    if beta > alpha: return max_seen
    for i in range(1, n):
        l[i] = (y_alpha[i-1]) + (alphac * (l[i-1] + phi*b[i-1]))
        b[i] = (beta * (l[i] - l[i-1])) + (betac * phi*b[i-1])
    return sqeuclidean(l+phi*b, y)

def holt_win_init(x, xi, p, y, l, b, s, m):
    p[xi] = x
    alpha,beta,gamma,l0,b0,phi = p[:6]; s0 = p[6:]
    alphac = 1 - alpha
    betac = 1 - beta
    gammac = 1 - gamma
    y_alpha = alpha*y
    y_gamma = gamma*y
    l[:] = 0; b[:] = 0; s[:] = 0
    l[0] = l0; b[0] = b0; s[:m] = s0        
    return alpha,beta,gamma,phi,alphac,betac,gammac,y_alpha,y_gamma

#(,M)
def holt_win__mul(x, xi, p, y, l, b, s, m, n, max_seen):
    alpha,beta,gamma,phi,alphac,betac,gammac,y_alpha,y_gamma = holt_win_init(x, xi, p, y, l, b, s, m)        
    if alpha==0.0: return max_seen
    if gamma > 1-alpha: return max_seen
    for i in range(1, n):
        l[i] = (y_alpha[i-1] / s[i-1]) + (alphac * (l[i-1]))
        s[i+m-1] = (y_gamma[i-1] / (l[i-1])) + (gammac * s[i-1])
    return sqeuclidean(l*s[:-(m-1)], y)

#(,A)
def holt_win__add(x, xi, p, y, l, b, s, m, n, max_seen):
    alpha,beta,gamma,phi,alphac,betac,gammac,y_alpha,y_gamma = holt_win_init(x, xi, p, y, l, b, s, m)
    if alpha==0.0: return max_seen
    if gamma > 1-alpha: return max_seen
    for i in range(1, n):
        l[i] = (y_alpha[i-1]) - (alpha * s[i-1]) + (alphac * (l[i-1]))            
        s[i+m-1] = y_gamma[i-1] - (gamma * (l[i-1])) + (gammac * s[i-1])
    return sqeuclidean(l+s[:-(m-1)], y)

#(A,M) (Ad,M)
def holt_win_add_mul_dam(x, xi, p, y, l, b, s, m, n, max_seen):
    alpha,beta,gamma,phi,alphac,betac,gammac,y_alpha,y_gamma = holt_win_init(x, xi, p, y, l, b, s, m)
    if alpha*beta==0.0: return max_seen
    if beta > alpha or gamma > 1-alpha: return max_seen
    for i in range(1, n):
        l[i] = (y_alpha[i-1] / s[i-1]) + (alphac * (l[i-1] + phi*b[i-1]))
        b[i] = (beta * (l[i] - l[i-1])) + (betac * phi*b[i-1])
        s[i+m-1] = (y_gamma[i-1] / (l[i-1] + phi*b[i-1])) + (gammac * s[i-1])
    return sqeuclidean((l+phi*b)*s[:-(m-1)], y)

#(M,M) (Md,M)
def holt_win_mul_mul_dam(x, xi, p, y, l, b, s, m, n, max_seen):
    alpha,beta,gamma,phi,alphac,betac,gammac,y_alpha,y_gamma = holt_win_init(x, xi, p, y, l, b, s, m)        
    if alpha*beta==0.0: return max_seen
    if beta > alpha or gamma > 1-alpha: return max_seen
    for i in range(1, n):
        l[i] = (y_alpha[i-1] / s[i-1]) + (alphac * (l[i-1] * b[i-1]**phi))
        b[i] = (beta * (l[i] / l[i-1])) + (betac * b[i-1]**phi)
        s[i+m-1] = (y_gamma[i-1] / (l[i-1] * b[i-1]**phi)) + (gammac * s[i-1])
    return sqeuclidean((l*b**phi)*s[:-(m-1)], y)

#(A,A) (Ad,A)
def holt_win_add_add_dam(x, xi, p, y, l, b, s, m, n, max_seen):
    alpha,beta,gamma,phi,alphac,betac,gammac,y_alpha,y_gamma = holt_win_init(x, xi, p, y, l, b, s, m)        
    if alpha*beta==0.0: return max_seen
    if beta > alpha or gamma > 1-alpha: return max_seen
    for i in range(1, n):
        l[i] = (y_alpha[i-1]) - (alpha * s[i-1]) + (alphac * (l[i-1] + phi*b[i-1]))
        b[i] = (beta * (l[i] - l[i-1])) + (betac * phi*b[i-1])
        s[i+m-1] = y_gamma[i-1] - (gamma * (l[i-1] + phi*b[i-1])) + (gammac * s[i-1])
    return sqeuclidean((l+phi*b)+s[:-(m-1)], y)

#(M,A) (M,Ad)
def holt_win_mul_add_dam(x, xi, p, y, l, b, s, m, n, max_seen):
    alpha,beta,gamma,phi,alphac,betac,gammac,y_alpha,y_gamma = holt_win_init(x, xi, p, y, l, b, s, m)        
    if alpha*beta==0.0: return max_seen
    if beta > alpha or gamma > 1-alpha: return max_seen
    for i in range(1, n):
        l[i] = (y_alpha[i-1]) - (alpha * s[i-1]) + (alphac * (l[i-1] * b[i-1]**phi))
        b[i] = (beta * (l[i] / l[i-1])) + (betac * b[i-1]**phi)
        s[i+m-1] = y_gamma[i-1] - (gamma * (l[i-1] * b[i-1]**phi)) + (gammac * s[i-1])
    return sqeuclidean((l*phi*b)+s[:-(m-1)], y)        

def holt_winters(data, alpha=None, beta=None, gamma=None, m=None, h=1, trend=None, 
                 damped=False, seasonal=None, phi=None, optimized=True,
                 use_boxcox=False, remove_bias=False, use_basinhopping=False):  
    """
    Holt Winter's Exponential Smoothing

    Parameters
    ----------
    data : array-like
        Time series
    alpha : float, optional
        The alpha value of the simple exponential smoothing, if the value is
        set then this value will be used as the value.
    beta :  float, optional
        The beta value of the holts trend method, if the value is
        set then this value will be used as the value.
    gamma : float, optional
        The gamma value of the holt winters seasonal method, if the value is
        set then this value will be used as the value.
    m : int, optional
        The number of seasons to consider for the holt winters.
    h : int, optional
        The number of time steps to forecast ahead.
    trend : {"add", "mul", None}, optional
        Type of trend component.
    damped : bool, optional
        Should the trend component be damped.
    seasonal : {"add", "mul", None}, optional
        Type of seasonal component.
    phi : float, optional
        The phi value of the damped method, if the value is
        set then this value will be used as the value.
    optimized : bool, optional
        Should the values that have not beem set above be optimized automatically?
    use_boxcox : {True, False, 'log'}, optional
        Should the boxcox tranform be applied to the data first? If 'log' then 
        apply the log.
    remove_bias : bool, optional
        Should the bias be removed from the fcast and fitted values before being
        returned?
    use_basinhopping : bool, optional
        Should the opptimser try harder using basinhopping to find optimal values? 

    Returns
    -------
    results : namedtuple
        A namedtuple with fcast, fitted and other attributes

    Notes
    -----
    This is a full implementation of the holt winters exponential smoothing as 
    per [1]. This includes all the unstable methods as well as the stable methods.
    The implementaion of the library follows as per the R library as much as possible
    whilest still being pythonic.
    
    References
    ----------
    [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles and practice. OTexts, 2014.
    """
    opt=None
    trending = trend in ['mul','add']
    seasoning = seasonal in ['mul','add']    
    if damped and not trending:
        raise NotImplementedError('Can only dampen the trend component')
    phi = phi if damped else 1.0
    n = len(data)    
    if use_boxcox == True:
        y, lam6da = boxcox(data)
    elif use_boxcox == 'log':
        lam6da = 0.0
        y = boxcox(data, 0.0)
    else:
        if isinstance(data, pd.Series):
            y = data.values.squeeze()            
        else:
            y = data.squeeze()
        if np.ndim(y) != 1:
            raise NotImplementedError('Only 1 dimensional data supported') 
    y_alpha = np.zeros((n,))
    y_gamma = np.zeros((n,))
    l = np.zeros((n,))
    b = np.zeros((n,))
    if seasoning:
        if (m is None or m == 0):
            raise NotImplementedError('Unable to detect season automatically')
    else:
        m = 0
    s = np.zeros((n+m-1,))
    p = np.zeros(6+m)
    max_seen = np.finfo(np.double).max
    if seasoning:
        l0 = y[np.arange(n)%m==0].mean()
        b0 = ((y[m:m+m]-y[:m])/m).mean() if trending else None
        s0 = list(y[:m]/l0) if seasonal=='mul' else list(y[:m]-l0)
    elif trending:
        l0 = y[0]
        b0 = y[1]/y[0] if trend=='mul' else y[1]-y[0]
        s0 = []
    else:
        l0 = y[0]
        b0 = None
        s0 = []
    if optimized:        
        init_alpha = alpha if alpha is not None else 0.5/max(m,1)
        init_beta = beta if beta is not None else 0.1*init_alpha
        init_gamma = None
        init_phi = phi if phi is not None else 0.99
        #Selection of functions to optimize for approporate parameters
        func_dict = {('mul','add'):holt_win_add_mul_dam,
                     ('mul','mul'):holt_win_mul_mul_dam,
                     ('mul', None):holt_win__mul,
                     ('add','add'):holt_win_add_add_dam,
                     ('add','mul'):holt_win_mul_add_dam,
                     ('add', None):holt_win__add,
                     (None ,'add'):holt_add_dam,
                     (None ,'mul'):holt_mul_dam,
                     (None ,None ):holt__,}
        if seasoning:
            init_gamma = gamma if gamma is not None else 0.05*(1-init_alpha) 
            xi = np.array([alpha is None, beta is None, gamma is None, True, trending, phi is None and damped]+[True]*m)
            func = func_dict[(seasonal,trend)]
        elif trending:
            xi = np.array([alpha is None, beta is None, False, True, True, phi is None and damped]+[False]*m)
            func = func_dict[(None,trend)]
        else:
            xi = np.array([alpha is None, False, False, True, False, False]+[False]*m)
            func = func_dict[(None,None)]
        p[:] = [init_alpha, init_beta, init_gamma, l0, b0, init_phi] + s0
        
        #txi [alpha, beta, gamma, l0, b0, phi, s0,..,s_(m-1)]
        #Have a quick look in the region for a good starting place for alpha etc.
        #using guestimates for the levels          
        txi = xi & np.array([True, True, True, False, False, True]+[False]*m)
        bounds = np.array([(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,None),(0.0,None),(0.8,1.0)] + [(None,None),]*m)
        res = brute(func, bounds[txi], (txi, p, y, l, b, s, m, n, max_seen), Ns=20, full_output=True, finish=None)        
        (p[txi], max_seen, grid, Jout) = res        
        [alpha, beta, gamma, l0, b0, phi] = p[:6]; s0 = p[6:]        
        #bounds = np.array([(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,None),(0.0,None),(0.8,1.0)] + [(None,None),]*m)    
        if use_basinhopping:            
            #Take a deeper look in the local minimum we are in to find the best 
            #solution to parameters, maybe hop around to try escape the local
            #minimum we may be in.
            res = basinhopping(func, p[xi], minimizer_kwargs={'args':(xi, p, y, l, b, s, m, n, max_seen), 'bounds':bounds[xi]}, stepsize=0.01)
        else:
            #Take a deeper look in the local minimum we are in to find the best 
            #solution to parameters
            res = minimize(func, p[xi], args=(xi, p, y, l, b, s, m, n, max_seen), bounds=bounds[xi])
        p[xi] = res.x;
        [alpha, beta, gamma, l0, b0, phi] = p[:6]; s0 = p[6:]
        SSE = res.fun
        opt = res
    alphac = 1 - alpha
    y_alpha[:] = alpha*y
    if trending:
        betac = 1 - beta
    if seasoning:
        gammac = 1 - gamma
        y_gamma[:] = gamma*y    
    l = np.zeros((n+h,))
    b = np.zeros((n+h,))
    s = np.zeros((n+h+m,))
    l[0] = l0
    b[0] = b0
    s[:m] = s0
    phi_h = np.cumsum(np.repeat(phi,h)**np.arange(1,h+1)) if damped else np.arange(1,h+1)
    trended = {'mul':np.multiply, 'add':np.add, None: lambda l,b: l}[trend]
    detrend = {'mul':np.divide, 'add':np.subtract, None: lambda l,b: 0 }[trend]
    dampen  = {'mul':np.power, 'add':np.multiply, None: lambda b,phi: 0}[trend]
    if seasonal=='mul':
        for i in range( 1, n+1):
            l[i] = y_alpha[i-1]/s[i-1] + (alphac * trended(l[i-1], dampen(b[i-1],phi) ) )
            if trending:
                b[i] = (beta * detrend(l[i], l[i-1])) + (betac * dampen(b[i-1],phi))
            s[i+m-1] = y_gamma[i-1]/trended(l[i-1], dampen(b[i-1],phi)) + (gammac * s[i-1])
        slope = b[:i].copy()
        season = s[m:i+m].copy()
        l[i:] = l[i]
        b[:i] = dampen(b[:i],phi)
        b[i:] = dampen(b[i],phi_h)
        s[i+m-1:] = [s[(i-1)+j%m] for j in range(h+1)]
        fitted = trended(l,b)*s[:-m]
    elif seasonal=='add':        
        for i in range( 1, n+1):
            l[i] = y_alpha[i-1]-(alpha * s[i-1]) + (alphac * trended(l[i-1], dampen(b[i-1],phi)) )
            if trending:
                b[i] = (beta * detrend(l[i], l[i-1])) + (betac * dampen(b[i-1],phi))
            s[i+m-1] = y_gamma[i-1] - (gamma * trended(l[i-1], dampen(b[i-1],phi))) + (gammac * s[i-1])
        slope = b[:i].copy()
        season = s[m:i+m].copy()
        l[i:] = l[i]
        b[:i] = dampen(b[:i],phi)
        b[i:] = dampen(b[i],phi_h)
        s[i+m-1:] = [s[(i-1)+j%m] for j in range(h+1)]
        fitted = trended(l,b)+s[:-m]
    else:
        for i in range( 1, n+1):
            l[i] = y_alpha[i-1] + (alphac * trended(l[i-1], dampen(b[i-1],phi)) )
            if trending:
                b[i] = (beta * detrend(l[i], l[i-1])) + (betac * dampen(b[i-1],phi))
        slope = b[:i].copy()
        season = s[m:i+m].copy()
        l[i:] = l[i]
        b[:i] = dampen(b[:i],phi)
        b[i:] = dampen(b[i],phi_h)
        fitted = trended(l,b)
    level = l[:i].copy()
    if use_boxcox or use_boxcox == 'log':
        fitted = inv_boxcox(fitted, lam6da)
        level = inv_boxcox(level, lam6da)
        slope = inv_boxcox(slope, lam6da)
        season = inv_boxcox(season, lam6da)
    SSE = sqeuclidean(fitted[:-h], data)    
    k = m*seasoning + 2*trending + 2 + 1*damped #(s0 + gamma) + (b0 + beta) + (l0 + alpha) + phi
    AIC = n*np.log(SSE/n) + (k)*2
    AICc= AIC + (2*(k+2)*(k+3))/(n-k-3)
    BIC = n*np.log(SSE/n) + (k)*np.log(n)
    try:
        if isinstance(data, pd.Series):
            fitted = pd.Series(fitted, index=data.index.union(data.index[-1]+np.arange(1,h+1)), name=data.name)
    except ValueError:
        warnings.warn('Can not infer index as no frequency for data')
        fitted = pd.Series(fitted, name=data.name)
    resid = data-fitted[:-h]
    if remove_bias:        
        fitted += resid.mean()
    if not damped: phi = None
    return HoltWintersResult(fcast=fitted[-h:], fitted=fitted[:-h], alpha=alpha, 
               beta=beta, gamma=gamma, m=m, phi=phi, l0=l0, b0=b0, s0=s0, 
               SSE=SSE, level=level, slope=slope, season=season, opt=opt, 
               AIC=AIC, BIC=BIC, AICc=AICc, resid=resid, k=k)

def simple_exp_smoothing(data, alpha=None, h=1, optimized=True):
    """
    Simple Exponential Smoothing helper wrapper to holt_winters(...)

    Parameters
    ----------
    data : array-like
        Time series
    alpha : float, optional
        The alpha value of the simple exponential smoothing, if the value is
        set then this value will be used as the value.
    h : int, optional
        The number of time steps to forecast ahead.
    optimized : bool
        Should the values that have not been set above be optimized automatically?

    Returns
    -------
    results : namedtuple
        A namedtuple with fcast, fitted and other attributes

    Notes
    -----
    This is a full implementation of the simple exponential smoothing as 
    per [1].
    
    References
    ----------
    [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles and practice. OTexts, 2014.
    """
    return holt_winters(data, alpha=alpha, h=h, optimized=optimized)

def holt(data, alpha=None, beta=None, h=1, exponential=False, damped=False, 
         phi=None, optimized=True):
    """
    Holt's Exponential Smoothing helper wrapper to holt_winters(...)

    Parameters
    ----------
    data : array-like
        Time series
    alpha : float, optional
        The alpha value of the simple exponential smoothing, if the value is
        set then this value will be used as the value.
    beta :  float, optional
        The beta value of the holts trend method, if the value is
        set then this value will be used as the value.
    h : int, optional
        The number of time steps to forecast ahead.
    exponential : bool, optional
        If true use a multipicative trend else use a additive trend.
    damped : bool
        Should the trend component be damped.
    phi : float, optional
        The phi value of the damped method, if the value is
        set then this value will be used as the value.
    optimized : bool, optional
        Should the values that have not been set above be optimized automatically?

    Returns
    -------
    results : namedtuple
        A namedtuple with fcast, fitted and other attributes

    Notes
    -----
    This is a full implementation of the holts exponential smoothing as 
    per [1]. 
    
    References
    ----------
    [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles and practice. OTexts, 2014.
    """
    trend = 'mul' if exponential else 'add'
    return holt_winters(data, alpha=alpha, beta=beta, h=h, trend=trend, 
                        damped=damped, phi=phi, optimized=optimized)