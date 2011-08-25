import numpy as np
from scipy import stats

def qqplot(data, dist=stats.norm, distargs=(), loc=0, scale=1, a=0, fit=False,
                line=False):
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
    distargs : tuple
        A tuple of arguments passed to dist to specify it fully
        so dist.ppf may be called.
    loc : float
        Location parameter for dist
    a : float
        Offset for the plotting position of the expected order statistic.
        The plotting positions are calculated as (i - a)/(nobs - 2*a + 1)
        for i in range(0,nobs+1)
    scale : float
        Scale parameter for dist
    fit : boolean
        If fit is false, loc, scale, and distargs are passed to the
        distribution. If fit is True then the parameters for dist
        are fit automatically using dist.fit. The quantiles are formed
        from the standardized data, after subtracting the fitted loc
        and dividing by the fitted scale.
    line : boolean
        If True a reference line is drawn on the graph. The default is to
        fit a line via OLS regression.

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
    >>> fig = sm.qqplot(res, stats.t, distargs=(4,))
    >>> plt.show()
    >>> plt.close(fig)
    >>> #qqplot against same as above, but with mean 3 and sd 10
    >>> fig = sm.qqplot(res, stats.t, distargs=(4,), loc=3, scale=10)
    >>> plt.show()
    >>> plt.close(fig)
    >>> #automatically determine parameters for t dist
    >>> #including the loc and scale
    >>> fig = sm.qqplot(res, stats.t, fit=True, line=True)
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

    if fit:
        fit_params = dist.fit(data)
        loc = fit_params[-2]
        scale = fit_params[-1]
        if len(fit_params)>2:
            dist = dist(*fit_params[:-2], loc = 0, scale = 1)
        else:
            dist = dist(loc=0, scale=1)
    elif distargs or loc != 0 or scale != 1:
        dist = dist(*distargs, **dict(loc=loc, scale=scale))

    #about 10x faster than plotting_position
    plotting_pos = lambda n, a : (np.arange(1.,n+1) - a)/(n- 2*a + 1)

    try:
        theoretical_quantiles = dist.ppf(plotting_pos(nobs, a))
    except:
        raise ValueError('distribution requires more parameters')

    sample_quantiles = np.array(data, copy=True)
    sample_quantiles.sort()
    if fit:
        sample_quantiles -= loc
        sample_quantiles /= scale


    ax = plt.gca()
    ax.set_xmargin(0.02)
    plt.plot(theoretical_quantiles, sample_quantiles, 'bo')
    if line:
        m,b = sample_quantiles.std(), sample_quantiles.mean()
        ref_line = theoretical_quantiles*m + b
        plt.plot(theoretical_quantiles, ref_line, 'r-')

    xlabel = "Theoretical Quantiles"
    plt.xlabel(xlabel)
    ylabel = "Sample Quantiles"
    plt.ylabel(ylabel)



    return plt.gcf()



