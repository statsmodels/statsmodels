from __future__ import division

__all__ = ['autoplot', 'facet_plot']

import numpy as np
from collections import Counter
from mpl_toolkits.mplot3d.axes3d import Axes3D

from statsmodels.graphics import utils
from statsmodels.graphics import mosaicplot
from statsmodels.graphics.boxplots import violinplot, beanplot
from statsmodels.graphics.plot_grids import _make_ellipse

from scipy.stats.kde import gaussian_kde

import patsy
import pylab as plt
import pandas as pd


##########################################################
# HELPERS FUNCTIONS
##########################################################


def _jitter(x, jitter_level=1.0):
    """add a little jitter to integer array"""
    u = x.unique()
    diff = abs(np.subtract.outer(u, u))
    diff[diff == 0] = np.max(u)
    min_diff = np.min(diff)
    return x + jitter_level * min_diff * 0.4 * (plt.rand(len(x)) - 0.5)


def _build_axes(ax, **kwargs):
    """This build the axes from information about the figure, row
    and col position, the index and the base axes from wich to obtain
    the axis
    """
    if isinstance(ax, list):
        if kwargs.get('projection', None) == '3d':
            #the new version doesn't work, patch up with the older one
            fig = ax[0]
            temp_ax = fig.add_subplot(ax[1], ax[2], ax[3])
            rect = temp_ax.get_position()
            fig.delaxes(temp_ax)
            ax = Axes3D(fig, rect)
            # remove the strange color pathch
            ax.set_axis_bgcolor(fig.get_facecolor())
        else:
            ax = ax[0].add_subplot(ax[1], ax[2], ax[3],
                                   sharex=ax[4], sharey=ax[4], **kwargs)
    fig, ax = utils.create_mpl_ax(ax)
    return fig, ax


def _make_numeric(data, ax, dir, jitter=True):
    """if the data type is categorical convert it to integer
    and set the correct ticking in the given axes
    """
    if data.dtype == object:
        states = sorted(list(data.unique()))
        indexes = dict([(v, states.index(v)) for v in states])
        data = data.apply(lambda s: indexes[s])
        data = _jitter(data, jitter)
        if ax:
            labels = range(len(states))
            ax.__getattribute__('set_{}ticks'.format(dir))(labels)
            ax.__getattribute__('set_{}ticklabels'.format(dir))(states)
    return data


def _formula_split(formula):
    """split the formula of the facet_plot into the y, x and categorical terms
    """
    # determine the facet component
    if '|' in formula:
        f = formula.split('|')[1].strip()
        formula = formula.split('|')[0]
    else:
        f = None

    #try to obtain the endog and exog variable
    if '~' in formula:
        x = formula.split('~')[1].strip()
        y = formula.split('~')[0].strip()
    else:
        x = None
        y = formula.strip()

    #if there is not exog, swith the two
    if x is None:
        x, y = y, x
    return y, x, f


def _elements4facet(facet, data):
    """obtain a list of (category, subset of the dataframe) given the facet
    """
    if facet is not None:
        facet_list = [f.strip() for f in facet.split()]
        try:
            # try to use it a a hierarchical (or simple)
            #index for the dataframe
            elements = list(data.groupby(facet_list))
        except KeyError:  # go by patsy
            # create the matrix
            matrix = patsy.dmatrix(facet, data, return_type="dataframe")
            elements = []
            # take every column of the resulting design matrix
            # and use it to split the dataframe
            for column in matrix:
                value = matrix[column]
                elements.append((column, data[value > 0]))
    # facet is none, so simply use the whole dataset
    else:
        elements = [['', data]]
    return elements


def _array4name(formula, data):
    """given a name/patsy formula obtain the dataframe data from it
    """
    result = []
    for name in formula.split('+'):
        name = name.strip()
        try:
            # try to use it as a valid index
            value = data[[name]]
        except KeyError:
            #if it fails try it as a patsy formula
            # the +0 is needed to avoid the intercept column
            value = patsy.dmatrix(name + '+0', data, return_type="dataframe")
            value.name = name
        result.append(value)
    #merge all the resulting dataframe
    void = pd.DataFrame(index=result[0].index)
    for dataframe in result:
        for col in dataframe:
            void[col] = dataframe[col]
            void[col].name = col
    result = void
    # to do monovariate plots this should return a series
    # the dataframe is kept for future implementation of
    # multivariate plots
    if len(result.columns) == 1:
        result = pd.Series(result[result.columns[0]])
    result.name = formula
    return result


def _select_rowcolsize(num_of_categories):
    """given the number of facets select the best structure of subplots
    """
    L = num_of_categories
    side_num = np.ceil(np.sqrt(L))
    col_num = side_num
    row_num = side_num
    while True:
        if (row_num - 1) * col_num >= L:
            row_num = row_num - 1
        else:
            break
    return row_num, col_num


##########################################################
# DATA PLOTTING FUNCTIONS
##########################################################


def _auto_hist(data, ax, kind=None, *args, **kwargs):
    """
    given a pandas series infers the type of data and print
    a barplot, an histogram or a density plot given the circumstances.

    the kind keyword should let the user choose the type of graphics but it's
    not implemented yet.
    """
    plot_error = TypeError("the selected plot type "
                           "({}) is not right for these data".format(kind))
    data = pd.Series(data)
    if kind in ['matrix']:
        states = sorted(list(data.unique()))
        indexes = dict([(v, states.index(v)) for v in states])
        L = len(states)
        transitions = np.zeros((L, L))
        for d0, d1 in zip(data[:-1], data[1:]):
            transitions[indexes[d0], indexes[d1]] += 1
        transitions /= sum(transitions)
        kwargs.setdefault('interpolation', 'nearest')
        kwargs.setdefault('origin', 'lower')
        kwargs.setdefault('cmap', plt.cm.binary)
        ax.imshow(transitions, *args, **kwargs)
        ax.set_xticks(range(L))
        ax.set_yticks(range(L))
        ax.set_xticklabels(states)
        ax.set_yticklabels(states)
    elif kind in ['scatter']:
        if data.dtype == object:
            states = sorted(list(data.unique()))
            indexes = dict([(v, states.index(v)) for v in states])
            data = data.apply(lambda s: indexes[s])
            ax.set_yticks(range(len(states)))
            ax.set_yticklabels(states)
        kwargs.setdefault('alpha', 0.33)
        ax.scatter(data.index, data, *args, **kwargs)
    elif kind in ['lines']:
        if data.dtype == object:
            states = sorted(list(data.unique()))
            indexes = dict([(v, states.index(v)) for v in states])
            data = data.apply(lambda s: indexes[s])
            ax.set_yticks(range(len(states)))
            ax.set_yticklabels(states)
        kwargs.setdefault('alpha', 0.33)
        ax.plot(data.index, data, *args, **kwargs)
    elif kind in ['counter'] or (not kind and data.dtype != float):
        # integer or categorical are represented
        # by the same method
        res = Counter(data)
        #obtain the categories
        key = sorted(res.keys())
        # if it's numerical fill the keys between the present values
        if data.dtype == int:
            key = range(min(key), max(key) + 1)
        val = [res[i] for i in key]
        #set the defaul options
        # if the user set some of them, his choices has the priority
        kwargs.setdefault('facecolor', '#777777')
        kwargs.setdefault('align', 'center')
        kwargs.setdefault('edgecolor', None)
        #create the bar plot for the histogram
        ax.bar(range(len(val)), val, *args, **kwargs)
        #configuration of the ticks and labels
        ax.set_ylabel('Counts')
        ax.set_xticks(range(len(val)))
        ax.set_yticks([int(i) for i in ax.get_yticks()])
        ax.set_xlim(-0.5, len(val) - 0.5)
        ax.set_xticklabels(key)
    elif kind in ['acorr']:
        if data.dtype == object:
            raise plot_error
        ax.acorr(data*1.0, *args, **kwargs)
        ax.set_ylabel('correlation')
        lim = max(abs(i) for i in ax.get_xlim())
        ax.set_xlim(-lim, lim)
        ax.set_ylim(None, 1.01)
    elif kind in ['kde'] or (not kind and data.dtype == float):
        if data.dtype == object:
            raise plot_error
        if len(data) > 1:
            # create the density plot
            # if has less than 2 point gives error
            my_pdf = gaussian_kde(data)
            x = np.linspace(np.min(data), np.max(data), 100)
            kwargs.setdefault('facecolor', '#777777')
            kwargs.setdefault('alpha', 0.66)
            ax.fill_between(x, my_pdf(x), *args, **kwargs)
        for value in data:
            # make ticks fot the data
            ax.axvline(x=value, ymin=0.0, ymax=0.1,
                       linewidth=2, color='#555555', alpha=0.33)
        #make the histogram of the data, very lightly
        kwargs.setdefault('normed', True)
        kwargs['alpha'] = 0.33
        kwargs.setdefault('edgecolor', 'k')
        kwargs['facecolor'] = '#999999'
        kwargs.setdefault('bins', int(np.sqrt(len(data))))
        ax.hist(data, *args, **kwargs)
        ax.set_ylim(0.0, None)
        ax.set_ylabel('Density')
    elif kind in ['hist']:
        if data.dtype == object:
            raise plot_error
        kwargs.setdefault('normed', True)
        kwargs.setdefault('alpha', 1.0)
        kwargs.setdefault('edgecolor', 'none')
        kwargs.setdefault('facecolor', '#777777')
        kwargs.setdefault('bins', int(np.sqrt(len(data))))
        ax.hist(data, *args, **kwargs)
        ax.set_ylim(0.0, None)
        ax.set_ylabel('Density')
    ax.set_xlabel(data.name)
    return ax


def _autoplot_num2num(x, y, ax, kind, *args, **kwargs):
    """take care of the numerical x Vs numerical y plotting
    """
    plot_error = TypeError("the selected plot type "
                           "({}) is not right for these data".format(kind))
    if kind is None or kind in ['ellipse', 'scatter']:
        kwargs.setdefault('alpha', 0.33)
        _x = _jitter(x) if x.dtype == int else x
        _y = _jitter(y) if y.dtype == int else y
        ax.scatter(_x, _y, *args, **kwargs)
    elif kind in ['lines']:
        kwargs.setdefault('alpha', 0.5)
        _x = x.order()
        _y = y[x.index]
        ax.plot(_x, _y, *args, **kwargs)
    elif kind in ['hexbin']:
        kwargs.setdefault('cmap', plt.cm.binary)
        kwargs.setdefault('gridsize', 20)
        ax.hexbin(x, y, *args, **kwargs)
    elif kind in ['boxplot']:
        data = pd.DataFrame({'x': x, 'y': y})
        levels = list(data.groupby('x')['y'])
        level_v = [v for k, v in levels]
        #level_k = [k for k, v in levels]
        ax.boxplot(level_v, *args, **kwargs)
    elif kind in ['matrix']:
        if x.dtype == float:
            nbinsx = np.round(np.sqrt(len(x)))
        else:
            nbinsx = max(x.unique()) - min(x.unique()) + 1
        Db_x = 0.5 * (max(x) - min(x)) / (nbinsx - 1)

        bins_x = np.linspace(min(x)-1-Db_x, max(x)+1+Db_x, nbinsx+3)

        if y.dtype == float:
            nbinsy = np.round(np.sqrt(len(y)))
        else:
            nbinsy = max(y.unique()) - min(y.unique()) + 1
        Db_y = 0.5 * (max(y) - min(y)) / (nbinsy - 1)

        bins_y = np.linspace(min(y)-1-Db_y, max(y)+1+Db_y, nbinsy+3)

        matrix = np.histogram2d(x, y, bins=[bins_x, bins_y])[0]
        #kwargs.setdefault('interpolation', 'nearest')
        kwargs.setdefault('origin', 'lower')
        kwargs.setdefault('cmap', plt.cm.binary)
        extent = (min(bins_x), max(bins_x), min(bins_y), max(bins_y))
        kwargs.setdefault('extent', extent)
        kwargs.setdefault('aspect', 'auto')
        ax.imshow(matrix.T, *args, **kwargs)
    else:
        raise plot_error
    #add the level to the scatterplot
    if kind is None or kind == 'ellipse':
        mean = [np.mean(x), np.mean(y)]
        cov = np.cov(x, y)
        _make_ellipse(mean, cov, ax, 0.95, 'gray')
        _make_ellipse(mean, cov, ax, 0.50, 'blue')
        _make_ellipse(mean, cov, ax, 0.05, 'purple')
    # this gives some problem I don'tunderstand about a fixedlocator
    if y.dtype == int:
        ax.set_yticks([int(i) for i in ax.get_yticks()])
    if x.dtype == int:
        ax.set_xticks([int(i) for i in ax.get_xticks()])


def _autoplot_num2cat(x, y, ax, kind, *args, **kwargs):
    """take care of the numerical x Vs categorical y plotting
    """
    plot_error = TypeError("the selected plot type "
                           "({}) is not right for these data".format(kind))
    data = pd.DataFrame({'x': x, 'f': y})
    levels = list(data.groupby('f')['x'])
    level_v = [v for k, v in levels]
    level_k = [k for k, v in levels]
    if kind is None or kind in ['boxplot']:
        kwargs.setdefault('notch', True)
        kwargs['vert'] = False
        ax.boxplot(level_v, *args, **kwargs)
        ax.set_yticklabels(level_k)
    elif kind in ['scatter']:
        levels = y.unique()
        y_ticks = range(1, 1 + len(levels))
        num4level = {k: v for k, v in zip(levels, y_ticks)}
        y = y.apply(lambda l: num4level[l])
        if x.dtype == int:
            x = _jitter(x)
        kwargs.setdefault('alpha', 0.33)
        ax.scatter(x, _jitter(y), *args, **kwargs)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(level_k)
    else:
        raise plot_error


def _autoplot_cat2num(x, y, ax, kind, *args, **kwargs):
    """take care of the categorical x Vs numerical y plotting
    """
    plot_error = TypeError("the selected plot type "
                           "({}) is not right for these data".format(kind))
    data = pd.DataFrame({'x': y, 'f': x})
    levels = list(data.groupby('f')['x'])
    level_v = [v for k, v in levels]
    level_k = [k for k, v in levels]
    if kind is None or kind in ['violinplot']:
        violinplot(level_v, labels=level_k, ax=ax, *args, **kwargs)
    elif kind in ['boxplot']:
        ax.boxplot(level_v, *args, **kwargs)
        ax.set_xticklabels(level_k)
    elif kind in ['beanplot']:
        beanplot(level_v, labels=level_k, ax=ax, *args, **kwargs)
    elif kind in ['scatter']:
        xlevels = x.unique()
        x_ticks = range(1, 1 + len(xlevels))
        num4levelx = {k: v for k, v in zip(xlevels, x_ticks)}
        _x = _jitter(x.apply(lambda l: num4levelx[l]))
        kwargs.setdefault('alpha', 0.33)
        _y = _jitter(y) if y.dtype == int else y
        ax.scatter(_x, _y, *args, **kwargs)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(level_k)
    else:
        raise plot_error


def _autoplot_cat2cat(x, y, ax, kind, *args, **kwargs):
    """take care of the categorical x Vs categorical y plotting
    """
    plot_error = TypeError("the selected plot type "
                           "({}) is not right for these data".format(kind))
    data = pd.DataFrame({'x': y, 'f': x})
    levels = list(data.groupby('f')['x'])
    level_v = [v for k, v in levels]
    level_k = [k for k, v in levels]
    if kind is None or kind in ['mosaic']:
        x_name = (x.name or 'x')
        y_name = (y.name or 'y')
        data = pd.DataFrame({x_name: x, y_name: y})
        mosaicplot.mosaic(data, index=[x_name, y_name],
                          ax=ax, *args, **kwargs)
    elif kind in ['scatter', 'matrix', 'boxplot']:
        ylevels = y.unique()
        y_ticks = range(1, 1 + len(ylevels))
        num4levely = {k: v for k, v in zip(ylevels, y_ticks)}
        xlevels = x.unique()
        x_ticks = range(1, 1 + len(xlevels))
        num4levelx = {k: v for k, v in zip(xlevels, x_ticks)}
        level_k_y = [k for k, v in data.groupby('x')['f']]
        y = y.apply(lambda l: num4levely[l])
        x = x.apply(lambda l: num4levelx[l])
        if kind in ['scatter']:
            kwargs.setdefault('alpha', 0.33)
            ax.scatter(_jitter(x), _jitter(y), *args, **kwargs)
        elif kind in ['boxplot']:
            values = [[num4levely[s] for s in v] for v in level_v]
            kwargs.setdefault('notch', True)
            ax.boxplot(values, *args, **kwargs)
            ax.set_xticklabels(level_k)
        else:
            image = np.zeros((len(y_ticks), len(x_ticks)))
            for _x, _y in zip(x, y):
                image[_y - 1, _x - 1] += 1.0
            kwargs.setdefault('interpolation', 'nearest')
            kwargs.setdefault('origin', 'lower')
            extent = (x_ticks[0] - 0.5, x_ticks[-1] + 0.5,
                      y_ticks[0] - 0.5, y_ticks[-1] + 0.5)
            kwargs.setdefault('extent', extent)
            ax.imshow(image, *args, **kwargs)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(level_k_y)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(level_k)
    else:
        raise plot_error


def _auto_multivariate(x, y, ax, kind, *args, **kwargs):
    """manage the multivariate type of plots
    """
    plot_error = TypeError("the selected plot type "
                           "({}) is not right for these data".format(kind))
    #raise NotImplementedError("support for multivariate plots"
    #                          " is not yet implemented")
    if isinstance(y, pd.Series):
            y = pd.DataFrame({y.name: y})
    if isinstance(x, pd.Series):
            x = pd.DataFrame({x.name: x})
    if len(x.columns) == 1:
        fig, ax = _build_axes(ax)
        colors = ['b', 'g', 'r', 'y', 'm', 'c', 'k']
        for column, color in zip(y, colors):
            kwargs['color'] = color
            autoplot(x[x.columns[0]], y[column],
                     ax=ax, kind=kind, *args, **kwargs)
        return ax
    if not kind or kind in ['scatter', 'lines', 'trisurf', 'wireframe']:
        fig, ax = _build_axes(ax, projection='3d')

        if len(x.columns) != 2:
            raise plot_error
        new_x = x[x.columns[0]]
        new_y = x[x.columns[1]]
        # this is ugly, but it's a workaround the uncorrect placements
        # of the labels
        linespacing = 3.0
        ax.set_xlabel('\n'+x.columns[0], linespacing=linespacing)
        ax.set_ylabel('\n'+x.columns[1], linespacing=linespacing)
        jitter = kwargs.pop('jitter', 1.0)
        new_x = _make_numeric(new_x, ax, 'x', jitter=jitter)
        new_y = _make_numeric(new_y, ax, 'y', jitter=jitter)
        colors = ['b', 'g', 'r', 'y', 'm', 'c', 'k']
        kwargs.setdefault('alpha', 0.3)
        for column, color in zip(y, colors):
            new_z = y[column]
            kwargs['label'] = column
            new_z = _make_numeric(new_z, ax, 'z', jitter=jitter)
            #if new_z.dtype == object:
            #    raise plot_error
            if not kind or kind == 'scatter':
                kwargs.setdefault('marker', 'o')
                kwargs.setdefault('linestyle', 'none')
                ax.plot(new_x, new_y, zs=new_z,
                        color=color, *args, **kwargs)
            elif kind == 'trisurf':
                ax.plot_trisurf(new_x, new_y, new_z,
                                color=color, *args, **kwargs)
            elif kind == 'wireframe':
                ax.plot_wireframe(new_x, new_y, new_z,
                                  color=color, *args, **kwargs)
            elif kind == 'lines':
                ax.plot(new_x, new_y, zs=new_z,
                        color=color, *args, **kwargs)
            else:
                raise plot_error
        # create legends or put labels, this is the choice
        if len(y.columns) > 1:
            # position the label outside the plot, and make it unique
            # among the subplots
            # make it transparent to not break the visual
            #remove all the previous legends to avoid overlappings
            for t_ax in fig.axes:
                t_ax.legend().set_visible(False)
                plt.draw()
            leg = ax.legend(bbox_to_anchor=(0, 0, 1, 1),
                            bbox_transform=fig.transFigure)
            leg.get_frame().set_alpha(0.0)
        else:
            ax.set_zlabel('\n'+y.columns[0], linespacing=linespacing)
    elif kind in ['scatter_coded']:
        fig, ax = _build_axes(ax)
        if len(x.columns) != 2:
            raise plot_error
        jitter = kwargs.pop('jitter', 1.0)
        new_x = x[x.columns[0]]
        new_x = _make_numeric(new_x, ax, 'x', jitter=jitter)
        new_y = x[x.columns[1]]
        new_y = _make_numeric(new_y, ax, 'y', jitter=jitter)
        if new_y.dtype == int:
            new_y = _jitter(new_y)
        if new_x.dtype == int:
            new_x = _jitter(new_x)
        color = y[y.columns[0]]
        if color.dtype == object:
            states = sorted(list(color.unique()))
            indexes = dict([(v, states.index(v)) for v in states])
            color = color.apply(lambda s: indexes[s])
        area = np.r_[1] if len(y.columns) == 1 else y[y.columns[1]]
        if area.dtype == object:
            states = sorted(list(area.unique()))
            indexes = dict([(v, states.index(v)) for v in states])
            area = area.apply(lambda s: indexes[s])
        area /= max(abs(area))/55.0
        area += 60
        kwargs.setdefault('alpha', 0.5)
        kwargs.setdefault('edgecolor', 'none')
        ax.scatter(new_x, new_y, s=area, c=color, *args, **kwargs)
    else:
        raise plot_error
    return ax


##########################################################
# THE PRINCIPAL FUNCTIONS
##########################################################


def autoplot(x, y=None, kind=None, ax=None, *args, **kwargs):
    """Select automatically the type of plot given the array x and y

    The rules are that if both are numeric, do a scatter
    if one is numeric and one is categorical do a boxplot
    if both are categorical do a mosaic plot. If only the x is given
    do a density plot or an histogram if the data are float or
    integer/categorical.

    the args and kwargs are redirected to the plot function

    Parameters
    ==========
    x : array, list or pandas.Series
        The main array to be plotted
    y : same as x, optional
        optional array to be plotted ad dependent variable
    ax : matplotlib.axes, optional
        the axes on which draw the plot. If None a new figure will be created
    kind : string, optional
        Describe the type of plot that should be tried on the data.
        If given and not valid will raise a TypeError

    the valid kind of plot is riassumed in this list,
    where the first element is the default option:
        -numerical exogenous:
            -isolated:
                'matrix', 'scatter', 'lines', 'hist', 'kde', 'counter', 'acorr'
            -numerical endogenous:
                'ellipse', 'scatter', 'lines', 'hexbin', 'boxplot', 'matrix'
            -categorical endogenous:
                'boxplot', 'scatter'
        -categorical exogenous:
            -isolated:
                'matrix', 'scatter', 'lines', 'counter'
            -numerical endogenous:
                'violinplot', 'scatter', 'boxplot', 'beanplot'
            -categorical endogenous:
                'mosaic', 'scatter', 'matrix', 'boxplot'

    There is at a moment only a partial support for multivariate plots:
        if the endogenous variable is monodimensional and have:
            multple exogenous, it will try to overlaps the graphs
            changing colors. Works better for scatter (the default)
        if there are two endogenous variables and one or two:
            exogenous, it will create a scatterplot with varying size
            and color of the plot.

    As can be seen the points and boxplot version can be applied in any
    case. It should be noted that even if you can, this doesn't mean
    that you should: The function will not stop the user from creating
    meaningless plots

    Returns
    =======
    ax : the axes used in the plot

    See Also
    ========
    facet_plot: create a figure with subplots of x and y divided by category

    Examples
    ========
    Similarly to the facet_plot we create an example dataset and see some
    example application
    >>> import pylab as plt
    >>> N = 300
    >>> data = pd.DataFrame({
    ...     'int_1': plt.randint(0, 5, size=N),
    ...     'int_2': plt.randint(0, 5, size=N),
    ...     'float_1': plt.randn(N),
    ...     'float_2': plt.randn(N),
    ...     'cat_1': ['aeiou'[i] for i in plt.randint(0, 5, size=N)],
    ...     'cat_2': ['BCDF'[i] for i in plt.randint(0, 4, size=N)]})

    and do some example plot
    >>> autoplot(data.float_1, data.float_2)
    >>> autoplot(data.int_2, data.float_1, kind='boxplot')
    >>> autoplot(data.float_1, data.int_2, kind='ellipse')
    >>> autoplot(data.cat_1, data.cat_2, kind='mosaic')
    """

    available_plots = ['ellipse', 'lines', 'scatter', 'hexbin',
                       'boxplot', 'violinplot', 'beanplot', 'mosaic',
                       'matrix', 'counter', 'acorr', 'kde', 'hist',
                       'trisurf', 'wireframe', 'scatter_coded']
    if kind and kind not in available_plots:
        raise TypeError("the selected plot type " +
                        "({}) is not recognized,".format(kind) +
                        "the accepted types are {}".format(available_plots))
    # pass the responsability of managing the multivariate
    # so the logic is more simple
    if isinstance(x, pd.DataFrame) or isinstance(y, pd.DataFrame):
        return _auto_multivariate(x, y, ax, kind, *args, **kwargs)
    # I create the axes only when I'm sure that it's going to be
    # a bidimensional, standard plot
    fig, ax = _build_axes(ax)
    if y is None or y is x:
        kwargs.setdefault('ax', ax)
        kwargs.setdefault('kind', kind)
        try:
            return x.__plot__(*args, **kwargs)
        except AttributeError:
            return _auto_hist(x, *args, **kwargs)
    x = pd.Series(x)
    y = pd.Series(y)
    # the exog is numerical
    if x.dtype == float or x.dtype == int:
        # the endog is numeric too, do a scatterplot
        if y.dtype == float or y.dtype == int:
            _autoplot_num2num(x, y, ax, kind, *args, **kwargs)
        # the endog is categorical, do a horizontal boxplot
        else:
            _autoplot_num2cat(x, y, ax, kind, *args, **kwargs)
    # the exog is categorical
    else:
        #if the endog is numeric do a violinplot
        if y.dtype == float or y.dtype == int:
            _autoplot_cat2num(x, y, ax, kind, *args, **kwargs)
        #otherwise do a mosaic plot
        else:
            _autoplot_cat2cat(x, y, ax, kind, *args, **kwargs)
    ax.set_xlabel(x.name)
    ax.set_ylabel(y.name)
    return ax


def facet_plot(formula, data, kind=None, subset=None,
               drop_na=True, ax=None, *args, **kwargs):
    """make a faceted plot of two variables divided into categories

    the formula should follow the sintax of the faceted plot:

        endog ~ exog | factor

    where both the endog and the factor are optionals.
    If multiple factors are inserted divided by space the cartesian
    product of their level will be used.

    All the factor will be treated as patsy formulas, but are limited to
    monovariate endogenous and exogenous variables.

    Parameters
    ==========
    formula: string formula like
        The plot description using the patsy formula syntax
    data: dict like object or pandas.Dataframe
        The dataframe over which the formula should be evaluated
    subset: boolean array, optional
        The True position indicates which rows of the dataframe should
        be used (it should be used to select only a part of the dataset)
    kind: string, optional
        The kind of plot that should be done on the graphs. If not
        specified will try and guess the best one.
    drop_na: boolean, optional
        Drop the nan values in the dataframe,defaul to True

    the args and kwargs will be redirected to the specific plot

    the valid kind of plot is riassumed in this list,
    where the first element is the default option:
        -numerical exogenous:
            -isolated:
                'matrix', 'scatter', 'lines', 'hist', 'kde', 'counter', 'acorr'
            -numerical endogenous:
                'ellipse', 'scatter', 'lines', 'hexbin', 'boxplot', 'matrix'
            -categorical endogenous:
                'boxplot', 'scatter'
        -categorical exogenous:
            -isolated:
                'matrix', 'scatter', 'lines', 'counter'
            -numerical endogenous:
                'violinplot', 'scatter', 'boxplot', 'beanplot'
            -categorical endogenous:
                'mosaic', 'scatter', 'matrix', 'boxplot'

    There is at a moment only a partial support for multivariate plots:
        if the endogenous variable is monodimensional and have:
            multple exogenous, it will try to overlaps the graphs
            changing colors. Works better for scatter (the default)
        if there are two endogenous variables and one or two:
            exogenous, it will create a scatterplot with varying size
            and color of the plot.

    Returns
    =======
    fig: the created matplotlib figure

    See Also
    ========
    autoplot:
        the function that does all the heavy lifting of guessing
        and that can be used as a standalone function. The function
        is concettually based on the lattice library of R

        - http://www.statmethods.net/advgraphs/trellis.html
        - http://cran.r-project.org/web/packages/lattice/index.html

    Examples
    ========
    Let's create a very simple dataset to work on

        >>> import pylab as plt
        >>> N = 300
        >>> data = pd.DataFrame({
        ...     'int_1': plt.randint(0, 5, size=N),
        ...     'int_2': plt.randint(0, 5, size=N),
        ...     'float_1': plt.randn(N),
        ...     'float_2': plt.randn(N),
        ...     'cat_1': ['aeiou'[i] for i in plt.randint(0, 5, size=N)],
        ...     'cat_2': ['BCDF'[i] for i in plt.randint(0, 4, size=N)]})

    basic facet_plots for variuos combinations

        >>> facet_plot('int_1', data)
        >>> facet_plot('float_1', data)
        >>> facet_plot('int_1 | cat_1', data)
        >>> facet_plot('int_1 ~ float_1 | cat_1', data)
        >>> facet_plot('float_1 ~ float_1 | cat_2', data)
        >>> facet_plot('int_1 ~ cat_1', data)

    parameters can be passed to the underlying plot function

        >>> facet_plot('float_1', data, facecolor='r', alpha=1.0)

    the split can be done even in terms of integers values

        >>> facet_plot('float_1 ~ float_2 | int_1', data)

    formulas can be used in the definition of the model

        >>> facet_plot('I(float_1 + float_2) ~ int_1', data)

    Multiple categorical variable are supported,
    using the cartesian product of the levels

        >>> facet_plot('float_1 | cat_1 cat_2', data)

    of any combination of level as categorical variables

        >>> facet_plot('float_1 | cat_1 + cat_2', data)
        >>> facet_plot('float_1 | cat_1 * cat_2', data)
        >>> facet_plot('I(float_1*4) ~ I(float_2 + 3)', data)

    At last, the possibility to insert a certain kind of plot
    for example, using all the available types on a combination
    of categorical values

        >>> facet_plot('cat_1 ~ cat_2 | int_1', data, kind='scatter')
        >>> facet_plot('cat_1 ~ cat_2 | int_1', data, kind='mosaic')
        >>> facet_plot('cat_1 ~ cat_2 | int_1', data, kind='matrix')
        >>> facet_plot('cat_1 ~ cat_2', data, kind='boxplot')

    and these are the solution for the double numerical values

        >>> facet_plot('float_1 ~ float_2 | cat_1', data, kind='scatter')
        >>> facet_plot('float_1 ~ float_2 | cat_1', data, kind='ellipse')
        >>> facet_plot('float_1 ~ float_2 | cat_1', data, kind='hexbin')
        >>> facet_plot('float_1 ~ float_2 | cat_1', data, kind='matrix')
        >>> facet_plot('float_1 ~ float_2 | cat_1', data, kind='lines')
        >>> facet_plot('float_1 ~ int_1 | cat_1', data, kind='boxplot')
    """
    if not ax:
        fig = plt.figure()
    else:
        fig, ax = utils.create_mpl_ax(ax)
    y, x, facet = _formula_split(formula)
    data = pd.DataFrame(data)
    if drop_na:
        data = data.dropna()
    # if a subset is specified use it to trim the dataframe
    if subset:
        data = data[subset]
    if ax:
        if facet:
            raise ValueError('facet are incompatibles with single axes')
        else:
            autoplot(_array4name(x, data), _array4name(y, data),
                     ax=ax, kind=kind, *args, **kwargs)
            return fig

    # obtain a list of (category, subset of the dataframe)
    elements = _elements4facet(facet, data)
    # automatically select the number of subplots as a square of this side
    L = len(elements)
    row_num, col_num = _select_rowcolsize(L)
    # for each subplot create the plot
    base_ax = None
    for idx, (level, value) in enumerate(elements):
        # choose if use the name ad a dataframe index or a patsy formula
        value_x = _array4name(x, value)
        value_y = _array4name(y, value) if y else None
        # adjust the position of the plots, shifting them
        # for a better display of the axes ticks
        idx = row_num * col_num - idx - 1
        if idx // col_num == 0:
            idx -= row_num * col_num - L
        my_row = idx // col_num
        my_col = idx % col_num

        # launch the autoplot
        # I pass the construction to it as it can decide
        # to build it as a 3d or polar axis
        ax = [fig, row_num, col_num, idx + 1, base_ax]
        ax = autoplot(value_x, value_y, ax=ax, kind=kind, *args, **kwargs)

        #ax = fig.add_subplot(row_num, col_num, idx + 1,
        #                     sharex=base_ax, sharey=base_ax)
        # all the subplots share the same axis with the first one being
        # of the same variable
        if not base_ax:
            base_ax = ax

        # remove the extremal ticks to remove overlaps
        # if the ticks have been fixed it generate a fixedLocator
        # that gives error, so just skip if it fails
        try:
            if ((value_y is not None and value_y.dtype != object)
                    or value_y is None):
                ax.locator_params(prune='both', axis='y')
        except AttributeError:
            pass

        try:
            if value_x.dtype != object:
                ax.locator_params(prune='both', axis='x')
        except AttributeError:
            pass
        # remove the superfluos info base on the columns
        if my_col and not isinstance(ax, Axes3D):
            ax.set_ylabel('')
            plt.setp(ax.get_yticklabels(), visible=False)
        # show the x labels only if it's on the last line
        if my_row != row_num - 1 and not isinstance(ax, Axes3D):
            ax.set_xlabel('')
            plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_title(level)
    fig.canvas.set_window_title(formula)
    fig.subplots_adjust(wspace=0, hspace=0.2)
    return fig

if __name__ == '__main__':
    from statsmodels.graphics.facetplot import facet_plot
    N = 1000
    data = pd.DataFrame({
        'int_1': plt.randint(0, 10, size=N),
        'int_2': plt.randint(0, 5, size=N),
        'int_3': plt.randint(1, 3, size=N),
        'float_1': 4 * plt.randn(N),
        'float_2': plt.randn(N),
        'cat_1': ['aeiou'[i] for i in plt.randint(0, 5, size=N)],
        'cat_2': ['BCDF'[i] for i in plt.randint(0, 4, size=N)],
        'sin': np.sin(np.r_[0.0:10.0:N*1j]),
        'cos': np.cos(np.r_[0.0:10.0:N*1j]),
        'lin': np.r_[0.0:10.0:N*1j],
        'lin2': 0.5*np.r_[0.0:10.0:N*1j], })
    data['float_3'] = data['float_1']+data['float_2']+plt.randn(N)
    data['float_4'] = data['float_1']*data['float_2']+plt.randn(N)

    #facet_plot('float_4 + float_3 ~ float_1 + float_2 | cat_2', data)
    #facet_plot('cat_2 ~ float_1 + float_2 | cat_1', data)
    #facet_plot('float_4 + float_3 ~ cat_1 + float_2', data)
    #facet_plot('float_4 + float_3 ~ float_1 + cat_2', data)
    facet_plot('float_4 + float_3 ~ cat_1 + cat_2', data, jitter=False)
    facet_plot('float_4 + float_3 ~ cat_1 + cat_2', data, jitter=True)
    facet_plot('float_4 + float_3 ~ cat_1 + cat_2', data, jitter=0.5)
    #facet_plot('float_4 ~ cat_1 + cat_2', data)
    #facet_plot('float_3 ~ float_1 + float_2 | cat_2', data, 'scatter')
    #facet_plot('float_4 ~ float_1 + float_2 | cat_2', data, 'lines');
    #facet_plot('float_4 ~ float_1 + float_2 | cat_2', data, 'scatter_coded');
    #facet_plot('float_4 ~ float_1 + cat_1 | cat_2', data, 'scatter_coded');
    #facet_plot('float_4 ~ float_1 + float_2 | cat_2', data, 'wireframe');
    #facet_plot('lin ~ cos + sin | cat_2', data, 'lines');
    #facet_plot('lin2 + lin ~ cos + sin | cat_2', data, 'wireframe');

    assert _formula_split('y ~ x | f') == ('y', 'x', 'f')
    assert _formula_split('y ~ x') == ('y', 'x', None)
    assert _formula_split('x | f') == (None, 'x', 'f')
    assert _formula_split('x') == (None, 'x', None)

    assert all(_array4name('int_1 + int_2',
                           data).columns == ['int_1', 'int_2'])
    assert all(_array4name('int_1 + cat_2',
                           data).columns == ['int_1', 'cat_2'])
    assert isinstance(_array4name('cat_2', data), pd.Series)
    assert isinstance(_array4name('int_1', data), pd.Series)

    #facet_plot('cat_1 ~ cat_2', data, 'boxplot')
    #autoplot(data.int_2, data.float_1, 'boxplot')
    #autoplot(data.float_1, data.float_2, 'ellipse')
    #autoplot(data.float_1, data.float_2, 'matrix')
    #facet_plot('cat_2 ~ cat_1', data)
    #facet_plot('cat_2 ~ float_1', data, 'scatter')
    #facet_plot('float_2 ~ cat_1', data, 'scatter')
    #facet_plot('float_2 ~ float_1 | cat_1', data, 'ellipse');
    #facet_plot('float_2 ~ float_1:int_3', data)
    #facet_plot('float_1', data)
    #facet_plot('int_2', data)
    #facet_plot('int_2 ~ float_1 + float_2', data)
    #facet_plot('int_1 + int_2 ~ float_1 + float_2', data)
    #facet_plot('int_1 + int_2 ~ float_1 + cat_2', data)
    #facet_plot('int_1 + int_2 ~ cat_1 + cat_2', data)
    #facet_plot('float_3 + float_4 ~ float_1 + float_2', data)
    #facet_plot('float_4 + float_3 ~ float_1 + float_2', data)
    #facet_plot('float_3 ~ float_1 + float_2', data)

    #facet_plot('int_3 ~ cat_1 + cat_2', data)
    #facet_plot('int_2 ~ float_1 + float_2', data)
    #facet_plot('int_1 ~  int_2', data, 'matrix', interpolation='nearest');

    #fig = plt.figure()
    #ax = fig.add_subplot(2, 2, 1)
    #facet_plot('cat_2 ~ cat_1', data, ax=ax)
    #this should give error
    #facet_plot('cat_2 ~ cat_1 | int_1', data, ax=ax)
    plt.show()
