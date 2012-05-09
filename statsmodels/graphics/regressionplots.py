'''Partial Regression plot and residual plots to find misspecification


Author: Josef Perktold
License: BSD-3
Created: 2011-01-23

update
2011-06-05 : start to convert example to usable functions
2011-10-27 : docstrings

'''

import numpy as np

from statsmodels.regression.linear_model import OLS
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from . import utils


__all__ = ['plot_fit', 'plot_regress_exog', 'plot_partregress', 'plot_ccpr',
           'plot_regress_exog']


def plot_fit(res, exog_idx, exog_name='', y_true=None, ax=None, fontsize='small'):
    """Plot fit against one regressor.

    This creates one graph with the scatterplot of observed values compared to
    fitted values.

    Parameters
    ----------
    res : result instance
        result instance with resid, model.endog and model.exog as attributes
    exog_idx : int
        index of regressor in exog matrix
    y_true : array_like
        (optional) If this is not None, then the array is added to the plot
    ax : Matplotlib AxesSubplot instance, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.

    Returns
    -------
    fig : Matplotlib figure instance
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.

    Notes
    -----
    This is currently very simple, no options or varnames yet.

    """
    fig, ax = utils.create_mpl_ax(ax)

    if exog_name == '':
        exog_name = 'variable %d' % exog_idx

    #maybe add option for wendog, wexog
    y = res.model.endog
    x1 = res.model.exog[:, exog_idx]
    x1_argsort = np.argsort(x1)
    y = y[x1_argsort]
    x1 = x1[x1_argsort]

    ax.plot(x1, y, 'bo', label='observed')
    if not y_true is None:
        ax.plot(x1, y_true[x1_argsort], 'b-', label='true')
        title = 'fitted versus regressor %s' % exog_name
    else:
        title = 'fitted versus regressor %s' % exog_name

    prstd, iv_l, iv_u = wls_prediction_std(res)
    ax.plot(x1, res.fittedvalues[x1_argsort], 'k-', label='fitted') #'k-o')
    #ax.plot(x1, iv_u, 'r--')
    #ax.plot(x1, iv_l, 'r--')
    ax.fill_between(x1, iv_l[x1_argsort], iv_u[x1_argsort], alpha=0.1, color='k')
    ax.set_title(title, fontsize=fontsize)

    return fig




def plot_regress_exog(res, exog_idx, exog_name='', fig=None):
    """Plot regression results against one regressor.

    This plots four graphs in a 2 by 2 figure: 'endog versus exog',
    'residuals versus exog', 'fitted versus exog' and
    'fitted plus residual versus exog'

    Parameters
    ----------
    res : result instance
        result instance with resid, model.endog and model.exog as attributes
    exog_idx : int
        index of regressor in exog matrix
    fig : Matplotlib figure instance, optional
        If given, this figure is simply returned.  Otherwise a new figure is
        created.

    Returns
    -------
    fig : matplotlib figure instance

    Notes
    -----
    This is currently very simple, no options or varnames yet.

    """

    fig = utils.create_mpl_fig(fig)

    if exog_name == '':
        exog_name = 'variable %d' % exog_idx

    #maybe add option for wendog, wexog
    #y = res.endog
    x1 = res.model.exog[:,exog_idx]

    ax = fig.add_subplot(2,2,1)
    #namestr = ' for %s' % self.name if self.name else ''
    ax.plot(x1, res.model.endog, 'o')
    ax.set_title('endog versus exog', fontsize='small')# + namestr)

    ax = fig.add_subplot(2,2,2)
    #namestr = ' for %s' % self.name if self.name else ''
    ax.plot(x1, res.resid, 'o')
    ax.axhline(y=0)
    ax.set_title('residuals versus exog', fontsize='small')# + namestr)

    ax = fig.add_subplot(2,2,3)
    #namestr = ' for %s' % self.name if self.name else ''
    ax.plot(x1, res.fittedvalues, 'o')
    ax.set_title('Fitted versus exog', fontsize='small')# + namestr)

    ax = fig.add_subplot(2,2,4)
    #namestr = ' for %s' % self.name if self.name else ''
    ax.plot(x1, res.fittedvalues + res.resid, 'o')
    ax.set_title('Fitted plus residuals versus exog', fontsize='small')# + namestr)

    fig.suptitle('Regression Plots for %s' % exog_name)

    return fig


def _partial_regression(endog, exog_i, exog_others):
    """Partial regression.

    regress endog on exog_i conditional on exog_others

    uses OLS

    Parameters
    ----------
    endog : array_like
    exog : array_like
    exog_others : array_like

    Returns
    -------
    res1c : OLS results instance

    (res1a, res1b) : tuple of OLS results instances
         results from regression of endog on exog_others and of exog_i on
         exog_others

    """
    #FIXME: This function doesn't appear to be used.
    res1a = OLS(endog, exog_others).fit()
    res1b = OLS(exog_i, exog_others).fit()
    res1c = OLS(res1a.resid, res1b.resid).fit()

    return res1c, (res1a, res1b)


def plot_partregress_ax(endog, exog_i, exog_others, varname='',
                        title_fontsize=None, ax=None):
    """Plot partial regression for a single regressor.

    Parameters
    ----------
    endog : ndarray
       endogenous or response variable
    exog_i : ndarray
        exogenous, explanatory variable
    exog_others : ndarray
        other exogenous, explanatory variables, the effect of these variables
        will be removed by OLS regression
    varname : str
        name of the variable used in the title
    ax : Matplotlib AxesSubplot instance, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.

    Returns
    -------
    fig : Matplotlib figure instance
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.

    See Also
    --------
    plot_partregress : Plot partial regression for a set of regressors.

    """
    fig, ax = utils.create_mpl_ax(ax)

    res1a = OLS(endog, exog_others).fit()
    res1b = OLS(exog_i, exog_others).fit()
    ax.plot(res1b.resid, res1a.resid, 'o')
    res1c = OLS(res1a.resid, res1b.resid).fit()
    ax.plot(res1b.resid, res1c.fittedvalues, '-', color='k')
    ax.set_title('Partial Regression plot %s' % varname,
                 fontsize=title_fontsize)# + namestr)

    return fig


def plot_partregress(endog, exog, exog_idx=None, grid=None, fig=None):
    """Plot partial regression for a set of regressors.

    Parameters
    ----------
    endog : ndarray
        endogenous or response variable
    exog : ndarray
        exogenous, regressor variables
    exog_idx : None or list of int
        (column) indices of the exog used in the plot
    grid : None or tuple of int (nrows, ncols)
        If grid is given, then it is used for the arrangement of the subplots.
        If grid is None, then ncol is one, if there are only 2 subplots, and
        the number of columns is two otherwise.
    fig : Matplotlib figure instance, optional
        If given, this figure is simply returned.  Otherwise a new figure is
        created.

    Returns
    -------
    fig : Matplotlib figure instance
        If `fig` is None, the created figure.  Otherwise `fig` itself.

    Notes
    -----
    A subplot is created for each explanatory variable given by exog_idx.
    The partial regression plot shows the relationship between the response
    and the given explanatory variable after removing the effect of all other
    explanatory variables in exog.

    See Also
    --------
    plot_partregress_ax : Plot partial regression for a single regressor.
    plot_ccpr

    References
    ----------
    See http://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/partregr.htm

    """
    fig = utils.create_mpl_fig(fig)

    #maybe add option for using wendog, wexog instead
    y = endog

    if not grid is None:
        nrows, ncols = grid
    else:
        if len(exog_idx) > 2:
            nrows = int(np.ceil(len(exog_idx)/2.))
            ncols = 2
            title_fontsize = 'small'
        else:
            nrows = len(exog_idx)
            ncols = 1
            title_fontsize = None

    k_vars = exog.shape[1]
    #this function doesn't make sense if k_vars=1

    for i,idx in enumerate(exog_idx):
        others = range(k_vars)
        others.pop(idx)
        exog_others = exog[:, others]
        ax = fig.add_subplot(nrows, ncols, i+1)
        plot_partregress_ax(y, exog[:, idx], exog_others, ax=ax)

    return fig


def plot_ccpr_ax(res, exog_idx=None, ax=None):
    """Plot CCPR against one regressor.

    Generates a CCPR (component and component-plus-residual) plot.

    Parameters
    ----------
    res : result instance
        uses exog and params of the result instance
    exog_idx : int
        (column) index of the exog used in the plot
    ax : Matplotlib AxesSubplot instance, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.

    Returns
    -------
    fig : Matplotlib figure instance
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.

    See Also
    --------
    plot_ccpr : Creates CCPR plot for multiple regressors in a plot grid.

    References
    ----------
    See http://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/ccpr.htm

    """
    fig, ax = utils.create_mpl_ax(ax)

    x1 = res.model.exog[:,exog_idx]
    #namestr = ' for %s' % self.name if self.name else ''
    x1beta = x1*res.params[1]
    ax.plot(x1, x1beta + res.resid, 'o')
    ax.plot(x1, x1beta, '-')
    ax.set_title('X_%d beta_%d plus residuals versus exog (CCPR)' % \
                                                (exog_idx, exog_idx))

    return fig


def plot_ccpr(res, exog_idx=None, grid=None, fig=None):
    """Generate CCPR plots against a set of regressors, plot in a grid.

    Generates a grid of CCPR (component and component-plus-residual) plots.

    Parameters
    ----------
    res : result instance
        uses exog and params of the result instance
    exog_idx : None or list of int
        (column) indices of the exog used in the plot
    grid : None or tuple of int (nrows, ncols)
        If grid is given, then it is used for the arrangement of the subplots.
        If grid is None, then ncol is one, if there are only 2 subplots, and
        the number of columns is two otherwise.
    fig : Matplotlib figure instance, optional
        If given, this figure is simply returned.  Otherwise a new figure is
        created.

    Returns
    -------
    fig : Matplotlib figure instance
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.

    Notes
    -----
    Partial residual plots are formed as::

        Res + Betahat(i)*Xi versus Xi

    and CCPR adds::

        Betahat(i)*Xi versus Xi

    See Also
    --------
    plot_ccpr_ax : Creates CCPR plot for a single regressor.

    References
    ----------
    See http://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/ccpr.htm

    """
    fig = utils.create_mpl_fig(fig)

    if grid is not None:
        nrows, ncols = grid
    else:
        if len(exog_idx) > 2:
            nrows = int(np.ceil(len(exog_idx)/2.))
            ncols = 2
        else:
            nrows = len(exog_idx)
            ncols = 1

    for i, idx in enumerate(exog_idx):
        ax = fig.add_subplot(nrows, ncols, i+1)
        plot_ccpr_ax(res, exog_idx=idx, ax=ax)

    return fig

def abline_plot(intercept=None, slope=None, horiz=None, vert=None,
                model_results=None, ax=None, **kwargs):
    """
    Plots a line given an intercept and slope.

    intercept : float
        The intercept of the line
    slope : float
        The slope of the line
    horiz : float or array-like
        Data for horizontal lines on the y-axis
    vert : array-like
        Data for verterical lines on the x-axis
    model_results : statsmodels results instance
        Any object that has a two-value `params` attribute. Assumed that it
        is (intercept, slope)
    ax : axes, optional
        Matplotlib axes instance
    kwargs
        Options passed to matplotlib.pyplot.plt

    Returns
    -------
    fig : Figure
        The figure given by `ax.figure` or a new instance.

    Examples
    --------
    >>> import numpy as np
    >>> import statsmodels.api as sm
    >>> np.random.seed(12345)
    >>> X = sm.add_constant(np.random.normal(0, 20, size=30), prepend=True)
    >>> y = np.dot(X, [25, 3.5]) + np.random.normal(0, 30, size=30)
    >>> mod = sm.OLS(y,X).fit()
    >>> fig = abline_plot(model_results=mod)
    >>> ax = fig.axes
    >>> ax.scatter(X[:,1], y)
    >>> ax.margins(.1)
    >>> import matplotlib.pyplot as plt
    >>> plt.show()
    """
    fig,ax = utils.create_mpl_ax(ax)

    if model_results:
        intercept, slope = model_results.params
        x = [model_results.model.exog[:,1].min(),
             model_results.model.exog[:,1].max()]
    else:
        x = None
        if not (intercept is not None and slope is not None):
            raise ValueError("specify slope and intercepty or model_results")

    if not x: # can't infer x limits
        x = ax.get_xlim()

    y = [x[0]*slope+intercept, x[1]*slope+intercept]
    ax.set_xlim(x)
    ax.set_ylim(y)

    from matplotlib.lines import Line2D

    class ABLine2D(Line2D):

        def update_datalim(self, ax):
            ax.set_autoscale_on(False)

            children = ax.get_children()
            abline = [children[i] for i in range(len(children))
                       if isinstance(children[i], ABLine2D)][0]
            x = ax.get_xlim()
            y = [x[0]*slope+intercept, x[1]*slope+intercept]
            abline.set_data(x,y)
            ax.figure.canvas.draw()

    line = ABLine2D(x, y, **kwargs)
    ax.add_line(line)
    ax.callbacks.connect('xlim_changed', line.update_datalim)
    ax.callbacks.connect('ylim_changed', line.update_datalim)


    if horiz:
        ax.hline(horiz)
    if vert:
        ax.vline(vert)
    return fig

if __name__ == '__main__':

