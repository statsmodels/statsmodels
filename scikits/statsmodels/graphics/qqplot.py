import numpy as np
from scipy import stats

def qqplot(data, dist=stats.norm, *args, **kwargs):
    """
    qqplot of the quantiles of x versus the quantiles/ppf of a distribution. 
    
    Can take arguments specifying the parameters for dist or fit them
    automatically. (See fit under kwargs.)

    Parameters
    ----------
    data : array-like
        1d data array
    dist : A scipy.stats or scikits.statsmodels distribution
        Compare x against dist. The default
        is scipy.stats.distributions.norm (a standard normal).
    args : Additional arguments needed to use the ppf function for dist.
    kwargs : named arguments fit, loc and scale. Unless fit=True, loc and
        scale are passed to dist.for use in dist. Default is loc=0 and scale=1.
        But if fit is True, then the parameters for dist are fit automatically. 

    Returns
    -------
    matplotlib figure.

    Examples
    --------
    >>> import scikits.statsmodels.api as sm
    >>> from matplotlib import pyplot as plt
    >>> data = sm.datasets.longley.load()
    >>> data.exog = sm.add_constant(data.exog)
    >>> mod_fit = sm.OLS(data.endog, data.exog).fit()
    >>> res = mod_fit.resid
    >>> fig = sm.qqplot(res)
    >>> plt.show()
    >>> plt.close(fig)
    >>> #qqplot against quantiles of t distribution with 4 df
    >>> import scipy.stats as stats
    >>> fig = sm.qqplot(res, stats.t, 4)
    >>> plt.show()
    >>> plt.close(fig)
    >>> #qqplot against same as above, but with mean 3 and sd 10
    >>> fig = sm.qqplot(res, stats.t, 4, loc=3,scale=10)
    >>> plt.show()
    >>> plt.close(fig)
    >>> #automatically determine parameters for t dist
    >>> #including the loc and scale
    >>> fig = sm.qqplot(res, stats.t, fit=True)
    >>> plt.show()
    >>> plt.close(fig)
    Notes
    -----
    Depends on matplotlib. If fit=True then the parameters are fit using 
    the distribution's fit( ) method. 

    """
    try:
        from matplotlib import pyplot as plt
    except:
        raise ImportError("matplotlib not installed")

    if not hasattr(dist, 'ppf'):
        raise ValueError("distribution must have a ppf method")

    nobs = data.shape[0]
    
    fit = kwargs.get('fit',False)
    if fit:
        fit_params = dist.fit(data)
        loc = fit_params[-2]
        scale = fit_params[-1]
        if len(fit_params)>2:
            dist = dist(*fit_params[:-2], loc= loc, scale=scale)
        else:
            dist = dist(loc = loc, scale= scale)
    elif args or kwargs:
        loc = kwargs.get('loc',0)
        scale = kwargs.get('scale',1)
        dist = dist(*args,loc=loc, scale=scale)

    try:
        theoretical_quantiles = dist.ppf(np.linspace(0,1,nobs+2)[1:-1])
    except:
        raise ValueError('scipy.stats distribution requires more parameters')

    sample_quantiles = np.array(data, copy=True)
    sample_quantiles.sort()
    

    plt.plot(theoretical_quantiles, sample_quantiles, 'bo')
    xlabel = "Theoretical Quantiles"
    plt.xlabel(xlabel)
    ylabel = "Sample Quantiles"
    plt.ylabel(ylabel)
    return plt.gcf()



