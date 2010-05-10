import numpy as np
from scipy import stats

def qqplot(data, dist=stats.distributions.norm, binom_n=None):
    """
    qqplot of the quantiles of x versus the ppf of a distribution.

    Parameters
    ----------
    data : array-like
        1d data array
    dist : scipy.stats.distribution or string
        Compare x against dist.  Strings aren't implemented yet.  The default
        is scipy.stats.distributions.norm

    Returns
    -------
    matplotlib figure.

    Examples
    --------
    >>> import scikits.statsmodels as sm
    >>> from matplotlib import pyplot as plt
    >>> data = sm.datasets.longley.Load()
    >>> data.exog = sm.add_constant(data.exog)
    >>> mod_fit = sm.OLS(data.endog, data.exog).fit()
    >>> res = mod_fit.resid
    >>> std_res = (res - res.mean())/res.std()

    Import qqplots from the sandbox

    >>> from scikits.statsmodels.sandbox.graphics import qqplot
    >>> qqplot(std_res)
    >>> plt.show()

    Notes
    -----
    Only the default arguments currently work.  Depends on matplotlib.

    """
    try:
        from matplotlib import pyplot as plt
    except:
        raise ImportError("matplotlib not installed")

    if isinstance(dist, str):
        raise NotImplementedError

    names_dist = {}
    names_dist.update({"norm_gen" : "Normal"})
    plotname = names_dist[dist.__class__.__name__]

    x = np.array(data, copy=True)
    x.sort()
    nobs = x.shape[0]
    prob = np.linspace(1./(nobs-1), 1-1./(nobs-1), nobs)
    # is the above robust for a few data points?
    quantiles = np.zeros_like(x)
    for i in range(nobs):
        quantiles[i] = stats.scoreatpercentile(x, prob[i]*100)

    # estimate shape and location using distribution.fit
    # for normal, but will have to be somewhat distribution specific
    loc,scale = dist.fit(x)
    y = dist.ppf(prob, loc=loc, scale=scale)
#    plt.figure()
    plt.scatter(y, quantiles)
    y_low = np.min((y.min(),quantiles.min()))-.25
    y_high = np.max((y.max(),quantiles.max()))+.25
    plt.plot([y.min()-.25, y.max()+.25], [y_low, y_high], 'b-')
    title = '%s - Quantile Plot' % plotname
    plt.title(title)
    xlabel = "Quantiles of %s" % plotname
    plt.xlabel(xlabel)
    ylabel = "%s Quantiles" % "Data"
    plt.ylabel(ylabel)
    plt.axis([y.min()-.25,y.max()+.25, y_low-.25, y_high+.25])
    return plt.gcf()

