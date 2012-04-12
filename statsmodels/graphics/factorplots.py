import numpy as np

from statsmodels.graphics.plottools import rainbow
import utils

def interaction_plot(x, trace, response, func=np.mean, ax=None, plottype='b',
                     xlabel=None, ylabel=None, colors = [], markers = [],
                     linestyles = [], legendloc='best', legendtitle=None,
                     **kwargs):
    """
    Parameters
    ----------
    x : array-like
        The `x` factor levels are the x-axis. If a `pandas.Series` is given
        its name will be used in `xlabel` if `xlabel` is None.
    trace : array-like
        The `trace` factor levels will form the trace. If `trace` is a
        `pandas.Series` its name will be used as the `legendtitle` if
        `legendtitle` is None.
    response : array-like
        The reponse variable. If a `pandas.Series` is given
        its name will be used in `ylabel` if `ylabel` is None.
    func : function
        Anything accepted by `pandas.DataFrame.aggregate`. This is applied to
        the response variable grouped by the trace levels.
    plottype : str {'line', 'scatter', 'both'}, optional
        The type of plot to return. Can be 'l', 's', or 'b'
    ax : axes, optional
        Matplotlib axes instance
    xlabel : str, optional
        Label to use for `x`. Default is 'X'. If `x` is a `pandas.Series` it
        will use the series names.
    ylabel : str, optional
        Label to use for `response`. Default is 'func of response'. If
        `response` is a `pandas.Series` it will use the series names.
    colors : list, optional
        If given, must have length == number of levels in trace.
    linestyles : list, optional
        If given, must have length == number of levels in trace.
    markers : list, optional
        If given, must have length == number of lovels in trace
    kwargs
        These will be passed to the plot command used either plot or scatter.
        If you want to control the overall plotting options, use kwargs.

    Returns
    -------
    fig : Figure
        The figure given by `ax.figure` or a new instance.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> weight = np.random.randint(1,4,size=60)
    >>> duration = np.random.randint(1,3,size=60)
    >>> days = np.log(np.random.randint(1,30, size=60))
    >>> ax = interaction_plot(weight, duration, days,
    ...             colors=['red','blue'], markers=['D','^'], ms=10)
    >>> import matplotlib.pyplot as plt
    >>> plt.show()
    """
    from pandas import DataFrame
    fig, ax = utils.create_mpl_ax(ax)

    if ylabel is None:
        try: # did we get a pandas.Series
            response_name = response.name
        except:
            response_name = 'response'
        #NOTE: py3 compatible?
        ylabel = '%s of %s' % (func.func_name, response_name)

    if xlabel is None:
        try:
            x_name = x.name
        except:
            x_name = 'X'

    if legendtitle is None:
        try:
            legendtitle = trace.name
        except:
            pass

    ax.set_ylabel(ylabel)
    ax.set_xlabel(x_name)


    data = DataFrame(dict(x=x, trace=trace, response=response))
    plot_data = data.groupby(['trace', 'x']).aggregate(func).reset_index()

    # check plot args
    n_trace = len(plot_data['trace'].unique())
    if linestyles:
        try:
            assert len(linestyles) == n_trace
        except AssertionError, err:
            raise ValueError("Must be a linestyle for each trace level")
    else: # set a default
        linestyles = ['-'] * n_trace
    if markers:
        try:
            assert len(markers) == n_trace
        except AssertionError, err:
            raise ValueError("Must be a linestyle for each trace level")
    else: # set a default
        markers = ['.'] * n_trace
    if colors:
        try:
            assert len(colors) == n_trace
        except AssertionError, err:
            raise ValueError("Must be a linestyle for each trace level")
    else: # set a default
        #TODO: how to get n_trace different colors?
        colors = rainbow(n_trace)

    if plottype == 'both' or plottype == 'b':
        for i, (values, group) in enumerate(plot_data.groupby(['trace'])):
            # trace label
            label = str(group['trace'].values[0])
            ax.plot(group['x'], group['response'], color=colors[i],
                    marker=markers[i], label=label,
                    linestyle=linestyles[i], **kwargs)
    elif plottype == 'line' or plottype == 'l':
        for i, (values, group) in enumerate(plot_data.groupby(['trace'])):
            # trace label
            label = str(group['trace'].values[0])
            ax.plot(group['x'], group['response'], color=colors[i],
                    label=label, linestyle=linestyles[i], **kwargs)
    elif plottype == 'scatter' or plottype == 's':
        for i, (values, group) in enumerate(plot_data.groupby(['trace'])):
            # trace label
            label = str(group['trace'].values[0])
            ax.scatter(group['x'], group['response'], color=colors[i],
                    label=label, marker=markers[i], **kwargs)

    else:
        raise ValueError("Plot type %s not understood" % plottype)
    ax.legend(loc=legendloc, title=legendtitle)
    ax.margins(.1)
    return fig
