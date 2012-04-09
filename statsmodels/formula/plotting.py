import numpy as np

def rainbow(n):
    """
    Returns a list of colors sampled at equal intervals over the spectrum.

    Parameters
    ----------
    n : int
        The number of colors to return

    Returns
    -------
    R : (n,3) array
        An of rows of RGB color values

    Notes
    -----
    Converts from HSV coordinates (0, 1, 1) to (1, 1, 1) to RGB. Based on
    the Sage function of the same name.
    """
    from matplotlib import colors
    R = np.ones((1,n,3))
    R[0,:,0] = np.linspace(0, 1, n, endpoint=False)
    #Note: could iterate and use colorsys.hsv_to_rgb
    return colors.hsv_to_rgb(R).squeeze()

import numpy as np
def interaction_plot(x, trace, response, func=np.mean, ax=None, plottype='b',
                     xlabel=None, ylabel=None, colors = [], markers = [],
                     linestyles = [], legendloc='best', legendtitle=None,
                     **kwargs):
    """
    Parameters
    ----------
    x : array-like
        The `x` factor levels are the x-axis
    trace : array-like
        The `trace` factor levels will form the trace
    response : array-like
        The reponse variable
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
    linestyles : list, optional
    markers : list, optional
        `colors`, `linestyles`, and `markers` must be lists of the same
        length as the number of unique trace elements. If you want to control
        the overall plotting options, use kwargs.
    kwargs
        These will be passed to the plot command used either plot or scatter.
    """
    from pandas import DataFrame
    if ax is None:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    if ylabel is None:
        try: # did we get a pandas.Series
            response_name = response.name
        except:
            response_name = 'response'
        # py3 compatible?
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
    return ax


