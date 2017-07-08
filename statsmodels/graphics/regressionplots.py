'''Partial Regression plot and residual plots to find misspecification


Author: Josef Perktold
License: BSD-3
Created: 2011-01-23

update
2011-06-05 : start to convert example to usable functions
2011-10-27 : docstrings

'''
from statsmodels.compat.python import lrange, string_types, lzip, range
import numpy as np
import pandas as pd
from patsy import dmatrix

from statsmodels.regression.linear_model import OLS, GLS, WLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.graphics import utils
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tools.tools import maybe_unwrap_results
from statsmodels.base import model

from ._regressionplots_doc import (
    _plot_added_variable_doc,
    _plot_partial_residuals_doc,
    _plot_ceres_residuals_doc)

__all__ = ['plot_fit', 'plot_regress_exog', 'plot_partregress', 'plot_ccpr',
           'plot_regress_exog', 'plot_partregress_grid', 'plot_ccpr_grid',
           'add_lowess', 'abline_plot', 'influence_plot',
           'plot_leverage_resid2', 'added_variable_resids',
           'partial_resids', 'ceres_resids', 'plot_added_variable',
           'plot_partial_residuals', 'plot_ceres_residuals']

#TODO: consider moving to influence module
def _high_leverage(results):
    #TODO: replace 1 with k_constant
    return 2. * (results.df_model + 1)/results.nobs


def add_lowess(ax, lines_idx=0, frac=.2, **lowess_kwargs):
    """
    Add Lowess line to a plot.

    Parameters
    ----------
    ax : matplotlib Axes instance
        The Axes to which to add the plot
    lines_idx : int
        This is the line on the existing plot to which you want to add
        a smoothed lowess line.
    frac : float
        The fraction of the points to use when doing the lowess fit.
    lowess_kwargs
        Additional keyword arguments are passes to lowess.

    Returns
    -------
    fig : matplotlib Figure instance
        The figure that holds the instance.
    """
    y0 = ax.get_lines()[lines_idx]._y
    x0 = ax.get_lines()[lines_idx]._x
    lres = lowess(y0, x0, frac=frac, **lowess_kwargs)
    ax.plot(lres[:, 0], lres[:, 1], 'r', lw=1.5)
    return ax.figure


def plot_fit(results, exog_idx, y_true=None, ax=None, **kwargs):
    """Plot fit against one regressor.

    This creates one graph with the scatterplot of observed values compared to
    fitted values.

    Parameters
    ----------
    results : result instance
        result instance with resid, model.endog and model.exog as attributes
    x_var : int or str
        Name or index of regressor in exog matrix.
    y_true : array_like
        (optional) If this is not None, then the array is added to the plot
    ax : Matplotlib AxesSubplot instance, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.
    kwargs
        The keyword arguments are passed to the plot command for the fitted
        values points.

    Returns
    -------
    fig : Matplotlib figure instance
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.

    Examples
    --------
    Load the Statewide Crime data set and perform linear regression with
    `poverty` and `hs_grad` as variables and `murder` as the response

    >>> import statsmodels.api as sm
    >>> import matplotlib.pyplot as plt

    >>> data = sm.datasets.statecrime.load_pandas().data
    >>> murder = data['murder']
    >>> X = data[['poverty', 'hs_grad']]

    >>> X["constant"] = 1
    >>> y = murder
    >>> model = sm.OLS(y, X)
    >>> results = model.fit()

    Create a plot just for the variable 'Poverty':

    >>> fig, ax = plt.subplots()
    >>> fig = sm.graphics.plot_fit(results, 0, ax=ax)
    >>> ax.set_ylabel("Murder Rate")
    >>> ax.set_xlabel("Poverty Level")
    >>> ax.set_title("Linear Regression")

    >>> plt.show()

    .. plot:: plots/graphics_plot_fit_ex.py

    """

    fig, ax = utils.create_mpl_ax(ax)

    exog_name, exog_idx = utils.maybe_name_or_idx(exog_idx, results.model)
    results = maybe_unwrap_results(results)

    #maybe add option for wendog, wexog
    y = results.model.endog
    x1 = results.model.exog[:, exog_idx]
    x1_argsort = np.argsort(x1)
    y = y[x1_argsort]
    x1 = x1[x1_argsort]

    ax.plot(x1, y, 'bo', label=results.model.endog_names)
    if not y_true is None:
        ax.plot(x1, y_true[x1_argsort], 'b-', label='True values')
    title = 'Fitted values versus %s' % exog_name

    prstd, iv_l, iv_u = wls_prediction_std(results)
    ax.plot(x1, results.fittedvalues[x1_argsort], 'D', color='r',
            label='fitted', **kwargs)
    ax.vlines(x1, iv_l[x1_argsort], iv_u[x1_argsort], linewidth=1, color='k',
              alpha=.7)
    #ax.fill_between(x1, iv_l[x1_argsort], iv_u[x1_argsort], alpha=0.1,
    #                    color='k')
    ax.set_title(title)
    ax.set_xlabel(exog_name)
    ax.set_ylabel(results.model.endog_names)
    ax.legend(loc='best', numpoints=1)

    return fig


def plot_regress_exog(results, exog_idx, fig=None):
    """Plot regression results against one regressor.

    This plots four graphs in a 2 by 2 figure: 'endog versus exog',
    'residuals versus exog', 'fitted versus exog' and
    'fitted plus residual versus exog'

    Parameters
    ----------
    results : result instance
        result instance with resid, model.endog and model.exog as attributes
    exog_idx : int
        index of regressor in exog matrix
    fig : Matplotlib figure instance, optional
        If given, this figure is simply returned.  Otherwise a new figure is
        created.

    Returns
    -------
    fig : matplotlib figure instance
    """

    fig = utils.create_mpl_fig(fig)

    exog_name, exog_idx = utils.maybe_name_or_idx(exog_idx, results.model)
    results = maybe_unwrap_results(results)

    #maybe add option for wendog, wexog
    y_name = results.model.endog_names
    x1 = results.model.exog[:, exog_idx]
    prstd, iv_l, iv_u = wls_prediction_std(results)

    ax = fig.add_subplot(2, 2, 1)
    ax.plot(x1, results.model.endog, 'o', color='b', alpha=0.9, label=y_name)
    ax.plot(x1, results.fittedvalues, 'D', color='r', label='fitted',
            alpha=.5)
    ax.vlines(x1, iv_l, iv_u, linewidth=1, color='k', alpha=.7)
    ax.set_title('Y and Fitted vs. X', fontsize='large')
    ax.set_xlabel(exog_name)
    ax.set_ylabel(y_name)
    ax.legend(loc='best')

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(x1, results.resid, 'o')
    ax.axhline(y=0, color='black')
    ax.set_title('Residuals versus %s' % exog_name, fontsize='large')
    ax.set_xlabel(exog_name)
    ax.set_ylabel("resid")

    ax = fig.add_subplot(2, 2, 3)
    exog_noti = np.ones(results.model.exog.shape[1], bool)
    exog_noti[exog_idx] = False
    exog_others = results.model.exog[:, exog_noti]
    from pandas import Series
    fig = plot_partregress(results.model.data.orig_endog,
                           Series(x1, name=exog_name,
                                  index=results.model.data.row_labels),
                           exog_others, obs_labels=False, ax=ax)
    ax.set_title('Partial regression plot', fontsize='large')
    #ax.set_ylabel("Fitted values")
    #ax.set_xlabel(exog_name)

    ax = fig.add_subplot(2, 2, 4)
    fig = plot_ccpr(results, exog_idx, ax=ax)
    ax.set_title('CCPR Plot', fontsize='large')
    #ax.set_xlabel(exog_name)
    #ax.set_ylabel("Fitted values + resids")

    fig.suptitle('Regression Plots for %s' % exog_name, fontsize="large")

    fig.tight_layout()

    fig.subplots_adjust(top=.90)
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


def plot_partregress(endog, exog_i, exog_others, data=None,
                     title_kwargs={}, obs_labels=True, label_kwargs={},
                     ax=None, ret_coords=False, **kwargs):
    """Plot partial regression for a single regressor.

    Parameters
    ----------
    endog : ndarray or string
       endogenous or response variable. If string is given, you can use a
       arbitrary translations as with a formula.
    exog_i : ndarray or string
        exogenous, explanatory variable. If string is given, you can use a
        arbitrary translations as with a formula.
    exog_others : ndarray or list of strings
        other exogenous, explanatory variables. If a list of strings is given,
        each item is a term in formula. You can use a arbitrary translations
        as with a formula. The effect of these variables will be removed by
        OLS regression.
    data : DataFrame, dict, or recarray
        Some kind of data structure with names if the other variables are
        given as strings.
    title_kwargs : dict
        Keyword arguments to pass on for the title. The key to control the
        fonts is fontdict.
    obs_labels : bool or array-like
        Whether or not to annotate the plot points with their observation
        labels. If obs_labels is a boolean, the point labels will try to do
        the right thing. First it will try to use the index of data, then
        fall back to the index of exog_i. Alternatively, you may give an
        array-like object corresponding to the obseveration numbers.
    labels_kwargs : dict
        Keyword arguments that control annotate for the observation labels.
    ax : Matplotlib AxesSubplot instance, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.
    ret_coords : bool
        If True will return the coordinates of the points in the plot. You
        can use this to add your own annotations.
    kwargs
        The keyword arguments passed to plot for the points.

    Returns
    -------
    fig : Matplotlib figure instance
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.
    coords : list, optional
        If ret_coords is True, return a tuple of arrays (x_coords, y_coords).

    Notes
    -----
    The slope of the fitted line is the that of `exog_i` in the full
    multiple regression. The individual points can be used to assess the
    influence of points on the estimated coefficient.

    See Also
    --------
    plot_partregress_grid : Plot partial regression for a set of regressors.
    """
    #NOTE: there is no interaction between possible missing data and
    #obs_labels yet, so this will need to be tweaked a bit for this case
    fig, ax = utils.create_mpl_ax(ax)

    # strings, use patsy to transform to data
    if isinstance(endog, string_types):
        endog = dmatrix(endog + "-1", data)

    if isinstance(exog_others, string_types):
        RHS = dmatrix(exog_others, data)
    elif isinstance(exog_others, list):
        RHS = "+".join(exog_others)
        RHS = dmatrix(RHS, data)
    else:
        RHS = exog_others
    RHS_isemtpy = False
    if isinstance(RHS, np.ndarray) and RHS.size==0:
        RHS_isemtpy = True
    elif isinstance(RHS, pd.DataFrame) and RHS.empty:
        RHS_isemtpy = True
    if isinstance(exog_i, string_types):
        exog_i = dmatrix(exog_i + "-1", data)

    # all arrays or pandas-like

    if RHS_isemtpy:
        ax.plot(endog, exog_i, 'o', **kwargs)
        fitted_line = OLS(endog, exog_i).fit()
        x_axis_endog_name = 'x' if isinstance(exog_i, np.ndarray) else exog_i.name
        y_axis_endog_name = 'y' if isinstance(endog, np.ndarray) else endog.design_info.column_names[0]
    else:
        res_yaxis = OLS(endog, RHS).fit()
        res_xaxis = OLS(exog_i, RHS).fit()
        xaxis_resid = res_xaxis.resid
        yaxis_resid = res_yaxis.resid
        x_axis_endog_name = res_xaxis.model.endog_names
        y_axis_endog_name = res_yaxis.model.endog_names
        ax.plot(xaxis_resid, yaxis_resid, 'o', **kwargs)
        fitted_line = OLS(yaxis_resid, xaxis_resid).fit()

    fig = abline_plot(0, fitted_line.params[0], color='k', ax=ax)

    if x_axis_endog_name == 'y':  # for no names regression will just get a y
        x_axis_endog_name = 'x'  # this is misleading, so use x
    ax.set_xlabel("e(%s | X)" % x_axis_endog_name)
    ax.set_ylabel("e(%s | X)" % y_axis_endog_name)
    ax.set_title('Partial Regression Plot', **title_kwargs)

    #NOTE: if we want to get super fancy, we could annotate if a point is
    #clicked using this widget
    #http://stackoverflow.com/questions/4652439/
    #is-there-a-matplotlib-equivalent-of-matlabs-datacursormode/
    #4674445#4674445
    if obs_labels is True:
        if data is not None:
            obs_labels = data.index
        elif hasattr(exog_i, "index"):
            obs_labels = exog_i.index
        else:
            obs_labels = res_xaxis.model.data.row_labels
        #NOTE: row_labels can be None.
        #Maybe we should fix this to never be the case.
        if obs_labels is None:
            obs_labels = lrange(len(exog_i))

    if obs_labels is not False:  # could be array-like
        if len(obs_labels) != len(exog_i):
            raise ValueError("obs_labels does not match length of exog_i")
        label_kwargs.update(dict(ha="center", va="bottom"))
        ax = utils.annotate_axes(lrange(len(obs_labels)), obs_labels,
                                 lzip(res_xaxis.resid, res_yaxis.resid),
                                 [(0, 5)] * len(obs_labels), "x-large", ax=ax,
                                 **label_kwargs)

    if ret_coords:
        return fig, (res_xaxis.resid, res_yaxis.resid)
    else:
        return fig


def plot_partregress_grid(results, exog_idx=None, grid=None, fig=None):
    """Plot partial regression for a set of regressors.

    Parameters
    ----------
    results : results instance
        A regression model results instance
    exog_idx : None, list of ints, list of strings
        (column) indices of the exog used in the plot, default is all.
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
    plot_partregress : Plot partial regression for a single regressor.
    plot_ccpr

    References
    ----------
    See http://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/partregr.htm

    """
    import pandas
    fig = utils.create_mpl_fig(fig)

    exog_name, exog_idx = utils.maybe_name_or_idx(exog_idx, results.model)

    #maybe add option for using wendog, wexog instead
    y = pandas.Series(results.model.endog, name=results.model.endog_names)
    exog = results.model.exog

    k_vars = exog.shape[1]
    #this function doesn't make sense if k_vars=1

    if not grid is None:
        nrows, ncols = grid
    else:
        if len(exog_idx) > 2:
            nrows = int(np.ceil(len(exog_idx)/2.))
            ncols = 2
            title_kwargs = {"fontdict" : {"fontsize" : 'small'}}
        else:
            nrows = len(exog_idx)
            ncols = 1
            title_kwargs = {}

    # for indexing purposes
    other_names = np.array(results.model.exog_names)
    for i, idx in enumerate(exog_idx):
        others = lrange(k_vars)
        others.pop(idx)
        exog_others = pandas.DataFrame(exog[:, others],
                                       columns=other_names[others])
        ax = fig.add_subplot(nrows, ncols, i+1)
        plot_partregress(y, pandas.Series(exog[:, idx],
                                          name=other_names[idx]),
                         exog_others, ax=ax, title_kwargs=title_kwargs,
                         obs_labels=False)
        ax.set_title("")

    fig.suptitle("Partial Regression Plot", fontsize="large")

    fig.tight_layout()

    fig.subplots_adjust(top=.95)
    return fig


def plot_ccpr(results, exog_idx, ax=None):
    """Plot CCPR against one regressor.

    Generates a CCPR (component and component-plus-residual) plot.

    Parameters
    ----------
    results : result instance
        A regression results instance.
    exog_idx : int or string
        Exogenous, explanatory variable. If string is given, it should
        be the variable name that you want to use, and you can use arbitrary
        translations as with a formula.
    ax : Matplotlib AxesSubplot instance, optional
        If given, it is used to plot in instead of a new figure being
        created.

    Returns
    -------
    fig : Matplotlib figure instance
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.

    See Also
    --------
    plot_ccpr_grid : Creates CCPR plot for multiple regressors in a plot grid.

    Notes
    -----
    The CCPR plot provides a way to judge the effect of one regressor on the
    response variable by taking into account the effects of the other
    independent variables. The partial residuals plot is defined as
    Residuals + B_i*X_i versus X_i. The component adds the B_i*X_i versus
    X_i to show where the fitted line would lie. Care should be taken if X_i
    is highly correlated with any of the other independent variables. If this
    is the case, the variance evident in the plot will be an underestimate of
    the true variance.

    References
    ----------
    http://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/ccpr.htm
    """
    fig, ax = utils.create_mpl_ax(ax)

    exog_name, exog_idx = utils.maybe_name_or_idx(exog_idx, results.model)
    results = maybe_unwrap_results(results)

    x1 = results.model.exog[:, exog_idx]
    #namestr = ' for %s' % self.name if self.name else ''
    x1beta = x1*results.params[exog_idx]
    ax.plot(x1, x1beta + results.resid, 'o')
    from statsmodels.tools.tools import add_constant
    mod = OLS(x1beta, add_constant(x1)).fit()
    params = mod.params
    fig = abline_plot(*params, **dict(ax=ax))
    #ax.plot(x1, x1beta, '-')
    ax.set_title('Component and component plus residual plot')
    ax.set_ylabel("Residual + %s*beta_%d" % (exog_name, exog_idx))
    ax.set_xlabel("%s" % exog_name)

    return fig


def plot_ccpr_grid(results, exog_idx=None, grid=None, fig=None):
    """Generate CCPR plots against a set of regressors, plot in a grid.

    Generates a grid of CCPR (component and component-plus-residual) plots.

    Parameters
    ----------
    results : result instance
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
    plot_ccpr : Creates CCPR plot for a single regressor.

    References
    ----------
    See http://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/ccpr.htm
    """
    fig = utils.create_mpl_fig(fig)

    exog_name, exog_idx = utils.maybe_name_or_idx(exog_idx, results.model)

    if grid is not None:
        nrows, ncols = grid
    else:
        if len(exog_idx) > 2:
            nrows = int(np.ceil(len(exog_idx)/2.))
            ncols = 2
        else:
            nrows = len(exog_idx)
            ncols = 1

    seen_constant = 0
    for i, idx in enumerate(exog_idx):
        if results.model.exog[:, idx].var() == 0:
            seen_constant = 1
            continue

        ax = fig.add_subplot(nrows, ncols, i+1-seen_constant)
        fig = plot_ccpr(results, exog_idx=idx, ax=ax)
        ax.set_title("")

    fig.suptitle("Component-Component Plus Residual Plot", fontsize="large")

    fig.tight_layout()

    fig.subplots_adjust(top=.95)
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
    >>> X = sm.add_constant(np.random.normal(0, 20, size=30))
    >>> y = np.dot(X, [25, 3.5]) + np.random.normal(0, 30, size=30)
    >>> mod = sm.OLS(y,X).fit()
    >>> fig = sm.graphics.abline_plot(model_results=mod)
    >>> ax = fig.axes[0]
    >>> ax.scatter(X[:,1], y)
    >>> ax.margins(.1)
    >>> import matplotlib.pyplot as plt
    >>> plt.show()
    """
    if ax is not None:  # get axis limits first thing, don't change these
        x = ax.get_xlim()
    else:
        x = None

    fig, ax = utils.create_mpl_ax(ax)

    if model_results:
        intercept, slope = model_results.params
        if x is None:
            x = [model_results.model.exog[:, 1].min(),
                 model_results.model.exog[:, 1].max()]
    else:
        if not (intercept is not None and slope is not None):
            raise ValueError("specify slope and intercepty or model_results")
        if x is None:
            x = ax.get_xlim()

    data_y = [x[0]*slope+intercept, x[1]*slope+intercept]
    ax.set_xlim(x)
    #ax.set_ylim(y)

    from matplotlib.lines import Line2D

    class ABLine2D(Line2D):
        def __init__(self, *args, **kwargs):
            super(ABLine2D, self).__init__(*args, **kwargs)
            self.id_xlim_callback = None
            self.id_ylim_callback = None

        def remove(self):
            ax = self.axes
            if self.id_xlim_callback:
                ax.callbacks.disconnect(self.id_xlim_callback)
            if self.id_ylim_callback:
                ax.callbacks.disconnect(self.id_ylim_callback)
            super(ABLine2D, self).remove()

        def update_datalim(self, ax):
            ax.set_autoscale_on(False)
            children = ax.get_children()
            ablines = [child for child in children if child is self]
            abline = ablines[0]
            x = ax.get_xlim()
            y = [x[0] * slope + intercept, x[1] * slope + intercept]
            abline.set_data(x, y)
            ax.figure.canvas.draw()

    # TODO: how to intercept something like a margins call and adjust?
    line = ABLine2D(x, data_y, **kwargs)
    ax.add_line(line)
    line.id_xlim_callback = ax.callbacks.connect('xlim_changed', line.update_datalim)
    line.id_ylim_callback = ax.callbacks.connect('ylim_changed', line.update_datalim)

    if horiz:
        ax.hline(horiz)
    if vert:
        ax.vline(vert)
    return fig


def influence_plot(results, external=True, alpha=.05, criterion="cooks",
                   size=48, plot_alpha=.75, ax=None, **kwargs):
    """
    Plot of influence in regression. Plots studentized resids vs. leverage.

    Parameters
    ----------
    results : results instance
        A fitted model.
    external : bool
        Whether to use externally or internally studentized residuals. It is
        recommended to leave external as True.
    alpha : float
        The alpha value to identify large studentized residuals. Large means
        abs(resid_studentized) > t.ppf(1-alpha/2, dof=results.df_resid)
    criterion : str {'DFFITS', 'Cooks'}
        Which criterion to base the size of the points on. Options are
        DFFITS or Cook's D.
    size : float
        The range of `criterion` is mapped to 10**2 - size**2 in points.
    plot_alpha : float
        The `alpha` of the plotted points.
    ax : matplotlib Axes instance
        An instance of a matplotlib Axes.

    Returns
    -------
    fig : matplotlib figure
        The matplotlib figure that contains the Axes.

    Notes
    -----
    Row labels for the observations in which the leverage, measured by the
    diagonal of the hat matrix, is high or the residuals are large, as the
    combination of large residuals and a high influence value indicates an
    influence point. The value of large residuals can be controlled using the
    `alpha` parameter. Large leverage points are identified as
    hat_i > 2 * (df_model + 1)/nobs.
    """
    fig, ax = utils.create_mpl_ax(ax)

    infl = results.get_influence()

    if criterion.lower().startswith('coo'):
        psize = infl.cooks_distance[0]
    elif criterion.lower().startswith('dff'):
        psize = np.abs(infl.dffits[0])
    else:
        raise ValueError("Criterion %s not understood" % criterion)

    # scale the variables
    #TODO: what is the correct scaling and the assumption here?
    #we want plots to be comparable across different plots
    #so we would need to use the expected distribution of criterion probably
    old_range = np.ptp(psize)
    new_range = size**2 - 8**2

    psize = (psize - psize.min()) * new_range/old_range + 8**2

    leverage = infl.hat_matrix_diag
    if external:
        resids = infl.resid_studentized_external
    else:
        resids = infl.resid_studentized_internal

    from scipy import stats

    cutoff = stats.t.ppf(1.-alpha/2, results.df_resid)
    large_resid = np.abs(resids) > cutoff
    large_leverage = leverage > _high_leverage(results)
    large_points = np.logical_or(large_resid, large_leverage)

    ax.scatter(leverage, resids, s=psize, alpha=plot_alpha)

    # add point labels
    labels = results.model.data.row_labels
    if labels is None:
        labels = lrange(len(resids))
    ax = utils.annotate_axes(np.where(large_points)[0], labels,
                             lzip(leverage, resids),
                             lzip(-(psize/2)**.5, (psize/2)**.5), "x-large",
                             ax)

    #TODO: make configurable or let people do it ex-post?
    font = {"fontsize" : 16, "color" : "black"}
    ax.set_ylabel("Studentized Residuals", **font)
    ax.set_xlabel("H Leverage", **font)
    ax.set_title("Influence Plot", **font)
    return fig


def plot_leverage_resid2(results, alpha=.05, ax=None,
                         **kwargs):
    """
    Plots leverage statistics vs. normalized residuals squared

    Parameters
    ----------
    results : results instance
        A regression results instance
    alpha : float
        Specifies the cut-off for large-standardized residuals. Residuals
        are assumed to be distributed N(0, 1) with alpha=alpha.
    ax : Axes instance
        Matplotlib Axes instance

    Returns
    -------
    fig : matplotlib Figure
        A matplotlib figure instance.
    """
    from scipy.stats import zscore, norm
    fig, ax = utils.create_mpl_ax(ax)

    infl = results.get_influence()
    leverage = infl.hat_matrix_diag
    resid = zscore(results.resid)
    ax.plot(resid**2, leverage, 'o', **kwargs)
    ax.set_xlabel("Normalized residuals**2")
    ax.set_ylabel("Leverage")
    ax.set_title("Leverage vs. Normalized residuals squared")

    large_leverage = leverage > _high_leverage(results)
    #norm or t here if standardized?
    cutoff = norm.ppf(1.-alpha/2)
    large_resid = np.abs(resid) > cutoff
    labels = results.model.data.row_labels
    if labels is None:
        labels = lrange(int(results.nobs))
    index = np.where(np.logical_or(large_leverage, large_resid))[0]
    ax = utils.annotate_axes(index, labels, lzip(resid**2, leverage),
                             [(0, 5)]*int(results.nobs), "large",
                             ax=ax, ha="center", va="bottom")
    ax.margins(.075, .075)
    return fig

def plot_added_variable(results, focus_exog, resid_type=None,
                        use_glm_weights=True, fit_kwargs=None, ax=None):
    # Docstring attached below

    model = results.model

    fig, ax = utils.create_mpl_ax(ax)

    endog_resid, focus_exog_resid =\
                 added_variable_resids(results, focus_exog,
                                       resid_type=resid_type,
                                       use_glm_weights=use_glm_weights,
                                       fit_kwargs=fit_kwargs)

    ax.plot(focus_exog_resid, endog_resid, 'o', alpha=0.6)

    ax.set_title('Added variable plot', fontsize='large')

    if type(focus_exog) is str:
        xname = focus_exog
    else:
        xname = model.exog_names[focus_exog]
    ax.set_xlabel(xname, size=15)
    ax.set_ylabel(model.endog_names + " residuals", size=15)

    return fig

plot_added_variable.__doc__ = _plot_added_variable_doc % {
    'extra_params_doc' : "results: object\n\tResults for a fitted regression model"}

def plot_partial_residuals(results, focus_exog, ax=None):
    # Docstring attached below

    model = results.model

    focus_exog, focus_col = utils.maybe_name_or_idx(focus_exog, model)

    pr = partial_resids(results, focus_exog)
    focus_exog_vals = results.model.exog[:, focus_col]

    fig, ax = utils.create_mpl_ax(ax)
    ax.plot(focus_exog_vals, pr, 'o', alpha=0.6)

    ax.set_title('Partial residuals plot', fontsize='large')

    if type(focus_exog) is str:
        xname = focus_exog
    else:
        xname = model.exog_names[focus_exog]
    ax.set_xlabel(xname, size=15)
    ax.set_ylabel("Component plus residual", size=15)

    return fig

plot_partial_residuals.__doc__ = _plot_partial_residuals_doc % {
    'extra_params_doc' : "results: object\n\tResults for a fitted regression model"}

def plot_ceres_residuals(results, focus_exog, frac=0.66, cond_means=None,
               ax=None):
    # Docstring attached below

    model = results.model

    focus_exog, focus_col = utils.maybe_name_or_idx(focus_exog, model)

    presid = ceres_resids(results, focus_exog, frac=frac,
                          cond_means=cond_means)

    focus_exog_vals = model.exog[:, focus_col]

    fig, ax = utils.create_mpl_ax(ax)
    ax.plot(focus_exog_vals, presid, 'o', alpha=0.6)

    ax.set_title('CERES residuals plot', fontsize='large')

    ax.set_xlabel(focus_exog, size=15)
    ax.set_ylabel("Component plus residual", size=15)

    return fig

plot_ceres_residuals.__doc__ = _plot_ceres_residuals_doc % {
    'extra_params_doc' : "results: object\n\tResults for a fitted regression model"}

def ceres_resids(results, focus_exog, frac=0.66, cond_means=None):
    """
    Calculate the CERES residuals (Conditional Expectation Partial
    Residuals) for a fitted model.

    Parameters
    ----------
    results : model results instance
        The fitted model for which the CERES residuals are calculated.
    focus_exog : int
        The column of results.model.exog used as the 'focus variable'.
    frac : float, optional
        Lowess smoothing parameter for estimating the conditional
        means.  Not used if `cond_means` is provided.
    cond_means : array-like, optional
        If provided, the columns of this array are the conditional
        means E[exog | focus exog], where exog ranges over some
        or all of the columns of exog other than focus exog.  If
        this is an empty nx0 array, the conditional means are
        treated as being zero.  If None, the conditional means are
        estimated.

    Returns
    -------
    An array containing the CERES residuals.

    Notes
    -----
    If `cond_means` is not provided, it is obtained by smoothing each
    column of exog (except the focus column) against the focus column.

    Currently only supports GLM, GEE, and OLS models.
    """

    model = results.model

    if not isinstance(model, (GLM, GEE, OLS)):
        raise ValueError("ceres residuals not available for %s" %
                         model.__class__.__name__)

    focus_exog, focus_col = utils.maybe_name_or_idx(focus_exog, model)

    # Indices of non-focus columns
    ix_nf = range(len(results.params))
    ix_nf = list(ix_nf)
    ix_nf.pop(focus_col)
    nnf = len(ix_nf)

    # Estimate the conditional means if not provided.
    if cond_means is None:

        # Below we calculate E[x | focus] where x is each column other
        # than the focus column.  We don't want the intercept when we do
        # this so we remove it here.
        pexog = model.exog[:, ix_nf]
        pexog -= pexog.mean(0)
        u, s, vt = np.linalg.svd(pexog, 0)
        ii = np.flatnonzero(s > 1e-6)
        pexog = u[:, ii]

        fcol = model.exog[:, focus_col]
        cond_means = np.empty((len(fcol), pexog.shape[1]))
        for j in range(pexog.shape[1]):

            # Get the fitted values for column i given the other
            # columns (skip the intercept).
            y0 = pexog[:, j]

            cf = lowess(y0, fcol, frac=frac, return_sorted=False)

            cond_means[:, j] = cf

    new_exog = np.concatenate((model.exog[:, ix_nf], cond_means), axis=1)

    # Refit the model using the adjusted exog values
    klass = model.__class__
    init_kwargs = model._get_init_kwds()
    new_model = klass(model.endog, new_exog, **init_kwargs)
    new_result = new_model.fit()

    # The partial residual, with respect to l(x2) (notation of Cook 1998)
    presid = model.endog - new_result.fittedvalues
    if isinstance(model, (GLM, GEE)):
        presid *= model.family.link.deriv(new_result.fittedvalues)
    if new_exog.shape[1] > nnf:
        presid += np.dot(new_exog[:, nnf:], new_result.params[nnf:])

    return presid

def partial_resids(results, focus_exog):
    """
    Returns partial residuals for a fitted model with respect to a
    'focus predictor'.

    Parameters
    ----------
    results : results instance
        A fitted regression model.
    focus col : int
        The column index of model.exog with respect to which the
        partial residuals are calculated.

    Returns
    -------
    An array of partial residuals.

    References
    ----------
    RD Cook and R Croos-Dabrera (1998).  Partial residual plots in
    generalized linear models.  Journal of the American Statistical
    Association, 93:442.
    """

    # TODO: could be a method of results
    # TODO: see Cook et al (1998) for a more general definition

    # The calculation follows equation (8) from Cook's paper.
    model = results.model
    resid = model.endog - results.predict()

    if isinstance(model, (GLM, GEE)):
        resid *= model.family.link.deriv(results.fittedvalues)
    elif isinstance(model, (OLS, GLS, WLS)):
        pass # No need to do anything
    else:
        raise ValueError("Partial residuals for '%s' not implemented."
                         % type(model))

    if type(focus_exog) is str:
        focus_col = model.exog_names.index(focus_exog)
    else:
        focus_col = focus_exog

    focus_val = results.params[focus_col] * model.exog[:, focus_col]

    return focus_val + resid

def added_variable_resids(results, focus_exog, resid_type=None,
                          use_glm_weights=True, fit_kwargs=None):
    """
    Residualize the endog variable and a 'focus' exog variable in a
    regression model with respect to the other exog variables.

    Parameters
    ----------
    results : regression results instance
        A fitted model including the focus exog and all other
        predictors of interest.
    focus_exog : integer or string
        The column of results.model.exog or a variable name that is
        to be residualized against the other predictors.
    resid_type : string
        The type of residuals to use for the dependent variable.  If
        None, uses `resid_deviance` for GLM/GEE and `resid` otherwise.
    use_glm_weights : bool
        Only used if the model is a GLM or GEE.  If True, the
        residuals for the focus predictor are computed using WLS, with
        the weights obtained from the IRLS calculations for fitting
        the GLM.  If False, unweighted regression is used.
    fit_kwargs : dict, optional
        Keyword arguments to be passed to fit when refitting the
        model.

    Returns
    -------
    endog_resid : array-like
        The residuals for the original exog
    focus_exog_resid : array-like
        The residuals for the focus predictor

    Notes
    -----
    The 'focus variable' residuals are always obtained using linear
    regression.

    Currently only GLM, GEE, and OLS models are supported.
    """

    model = results.model
    if not isinstance(model, (GEE, GLM, OLS)):
        raise ValueError("model type %s not supported for added variable residuals" %
                         model.__class__.__name__)

    exog = model.exog
    endog = model.endog

    focus_exog, focus_col = utils.maybe_name_or_idx(focus_exog, model)

    focus_exog_vals = exog[:, focus_col]

    # Default residuals
    if resid_type is None:
        if isinstance(model, (GEE, GLM)):
            resid_type = "resid_deviance"
        else:
            resid_type = "resid"

    ii = range(exog.shape[1])
    ii = list(ii)
    ii.pop(focus_col)
    reduced_exog = exog[:, ii]
    start_params = results.params[ii]

    klass = model.__class__

    kwargs = model._get_init_kwds()
    new_model = klass(endog, reduced_exog, **kwargs)
    args = {"start_params": start_params}
    if fit_kwargs is not None:
        args.update(fit_kwargs)
    new_result = new_model.fit(**args)
    if not new_result.converged:
        raise ValueError("fit did not converge when calculating added variable residuals")

    try:
        endog_resid = getattr(new_result, resid_type)
    except AttributeError:
        raise ValueError("'%s' residual type not available" % resid_type)

    import statsmodels.regression.linear_model as lm

    if isinstance(model, (GLM, GEE)) and use_glm_weights:
        weights = model.family.weights(results.fittedvalues)
        if hasattr(model, "data_weights"):
            weights = weights * model.data_weights
        lm_results = lm.WLS(focus_exog_vals, reduced_exog, weights).fit()
    else:
        lm_results = lm.OLS(focus_exog_vals, reduced_exog).fit()
    focus_exog_resid = lm_results.resid

    return endog_resid, focus_exog_resid
