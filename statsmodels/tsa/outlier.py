# A simple module for visualising, identifing and replace outliers
import pandas as pd
from matplotlib.pyplot import gca

from scipy.stats import norm as Gaussian
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import numpy as np 


def isoutlier(ts, find="IQR", N=2, detrend=True, dates=None, level=0):
    '''
    Find outliers in a time series.
    Parameters
    ----------
    ts : 1D array-like 
         A panda Series with a DateTime index or a 1D array
    find : {'MAD', 'IQR, 'ZScore'}
         Method for finding outliers
    N : float
        if |MAD or ZScore| > N observation is an outlier
    detrend : boolean
        remove the trend before finding ourliers?
    dates : array-like DateTime, optional
        The datetime values for the time series    
    level : int
            index level if the Series is multi-index
    Returns
    -------
    array-like boolean
        True when the observation is an outlier
    '''
    
    ts_len = len(ts)
    numpy = False
    if isinstance(ts, np.ndarray):
        ts = pd.Series(ts)
        numpy = True
        
    if detrend:
        ts = residue(ts, dates=dates, level=level) 
            
    if find == "MAD":
        outliers = abs(ts - ts.median()) > ts.mad() * (N / Gaussian.ppf(3/4.)) # reordered to avoid vector division
        
    elif find == "ZScore":
        outliers = (ts - ts.mean()) > ts.std() * N # reordered to avoid vector division
        
    elif find == "IQR": # Note uses a fixed value rather than N
        q = ts.quantile([0.25,.75])
        IQR = q[0.75] - q[0.25]
        outliers = (ts < (q[0.25] - 1.5*IQR)) | (ts > (q[0.75] + 1.5*IQR))
        
    else:
        raise ValueError('find must be one of "MAD", "ZScore" or "IQR"') 

    assert ts_len == len(outliers), "Returned result is incorrect length"
    return outliers.values if numpy else outliers

def trend(ts, dates=None, level=0):
    '''
    Return the line of best fit
    '''
    
    if (~ts.isnull()).sum() < 2 : #need at least two points
        return ts
    else:
        return rlm_ts(ts, dates=dates, level=level).fit().fittedvalues

def residue(ts, dates=None, level=0):
    '''
    Return the residue from the line of best fit
    '''
    if (~ts.isnull()).sum() < 2 : #need at least two points
        return ts
    else:
        return ts.where(ts.isnull(), rlm_ts(ts, dates=dates, level=level).fit().resid) #Necessary to deal with NaN values in ts

def rlm_ts(ts, dates=None, level=0):
    '''
    Fit a linear model using robust estimates
    '''
    df = ts.to_frame() 
    df['__X__'] = dates if dates else df.index.get_level_values(level)

    if isinstance(df['__X__'][0], pd.Timestamp):
        # ols does not work on datetime, need to create an int dependant variable 
        df['__X__'] = ((df['__X__'] - df['__X__'].min()).astype('timedelta64[s]'))
        
    return smf.rlm(formula='df[[0]] ~ __X__', data=df)


def plot(ts, trend=True, interval=False, outliers=False,  ax=None,  **kwargs):
    '''
    Plot a timeseries with optionl trend, 2 standard deviation interval and outliers
    Parameters
    ----------
    ts : pd.Series 
        A time series with a DateTime index
    trend : boolean 
        overlay trend linear as green dotted line?
    interval : boolean 
        overlay a 2 standard deviation interval? Not supported
    outliers : boolean
        Overlay outliers as red stars?
    kwargs : 
        Aguments passed to isoutler
    ax :  matplotlib.axes, optional 
        axes to draw on otherwise use current axes 
    Returns
    -------
    axes object
    '''
    
    if not ax:
        ax = gca()

    if not isinstance(ts, pd.Series):
        (ts.select_dtypes(include=[np.number]).
                   apply(plot, trend=trend, interval=interval, outliers=outliers,  ax=ax,  **kwargs))
        return ax
    
    result = rlm_ts(ts).fit()

    # Plot this first to get the better pandas timeseries drawing of dates on x axis
    ts.plot(ax=ax, label="{} ({:.0f}/yr)".format(ts.name, result.params['__X__']*60*60*24*365) if trend else ts.name)     

    if trend:
        result.fittedvalues.plot(ax=ax, style='--g', label="")
#    TODO: rlm does not support wls_prediction_std, add iterval someother way
#    if interval:
#        prstd, iv_l, iv_u = wls_prediction_std(result)
#        ax.fill_between(iv_l.index, iv_l, iv_u, color='#888888', alpha=0.25) 
    if outliers:
        df_outliers = ts[isoutlier(ts, **kwargs)]
        if len(df_outliers) > 0:
            df_outliers.plot( ax=ax, style='r*', label="")           

    return ax
            
def replace(ts, find="IQR", detrend=True, N = 2, how='NaN', **kwargs):
        ''' 
        Replace time series outliers with NaN or interpolated values. 
        Parameters
        ----------
        ts : pd.Series or Dataframe
            a time series with a DateTime index.
        find : {'MAD', 'IQR, 'ZScore'}
            Method of finding outliers
        N : int 
            if |MAD or ZScore| > N observation is an outlier
        detrend : boolean
            remove the timeseries trend before finding ourliers
        how : {'NaN', 'interpolate'}
            replace outliers with NaN or interpolated values
        Returns
        -------
        Time series with outliers replaced by NaN or values interpolated 
        from neightbours 
        '''
        if not isinstance(ts, pd.Series):
            return ts.apply(replace,  find=find, detrend=detrend, N = N, how=how, **kwargs)
 
        try:
            ts1 = ts.where(~isoutlier(ts, find=find, N=N, detrend=detrend))
            if how != 'NaN':
                ts1 = ts1.interpolate(how=how, **kwargs)

            return ts1
        except: # ts is probably not a np.numeric
            return ts