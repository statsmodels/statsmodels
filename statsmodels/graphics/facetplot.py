from __future__ import division

import numpy as np
from collections import Counter

from statsmodels.graphics import mosaicplot
from statsmodels.graphics.boxplots import violinplot
from statsmodels.graphics.plot_grids import _make_ellipse

from statsmodels.api import datasets
from scipy.stats.kde import gaussian_kde

import patsy
import pylab as plt
import pandas as pd


def _auto_hist(data, ax, *args, **kwargs):
    """
    given a pandas series infers the type of data and print
    a barplot, an histogram or a density plot given the circumstances
    """
    data = pd.Series(data)
    if data.dtype == float:
        #make the histogram of the data, very lightly
        ax.hist(data, bins=int(np.sqrt(len(data))), normed=True,
            facecolor='#999999', edgecolor='k', alpha=0.33)
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
                linewidth=2, color='#555555', alpha=0.5)
        ax.set_ylim(0.0, None)
        ax.set_ylabel('Density')
    else:
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
    ax.set_xlabel(data.name)
    return ax


def _autoplot(x, y=None, ax=None, *args, **kwargs):
    """Select automatically the type of plot given the type of x and y

    basically the rules are that if both are numeric, do a scatter
    if one is numeric and one is categorical do a boxplot
    if both are categorical do a mosaic plot

    the args and kwargs are redirected to the plot function
    """
    if y is None or y is x:
        return _auto_hist(x, ax, *args, **kwargs)
    x = pd.Series(x)
    y = pd.Series(y)
    # the exog is numerical
    if x.dtype == float or x.dtype == int:
        #TODO: if both are ints should add a jitter
        # the endog is numeric too, do a scatterplot
        if y.dtype == float or y.dtype == int:
            kwargs.setdefault('alpha', 0.33)
            plt.scatter(x, y, *args, **kwargs)
            #add the level to the scatterplot
            mean = [np.mean(x), np.mean(y)]
            cov = np.cov(x, y)
            _make_ellipse(mean, cov, ax, 0.95, 'gray')
            _make_ellipse(mean, cov, ax, 0.50, 'blue')
            _make_ellipse(mean, cov, ax, 0.05, 'purple')
            if y.dtype == int:
                ax.set_yticks([int(i) for i in ax.get_yticks()])
            if x.dtype == int:
                ax.set_xticks([int(i) for i in ax.get_xticks()])
        # the endog is categorical, do a horizontal boxplot
        else:
            data = pd.DataFrame({'x': x, 'f': y})
            levels = list(data.groupby('f')['x'])
            level_v = [v for k, v in levels]
            level_k = [k for k, v in levels]
            plt.boxplot(level_v, vert=False, *args, **kwargs)
            ax.set_yticklabels(level_k)
    # the exog is categorical
    else:
        #if the endog is numeric do a violinplot
        if y.dtype == float or y.dtype == int:
            data = pd.DataFrame({'x': y, 'f': x})
            levels = list(data.groupby('f')['x'])
            level_v = [v for k, v in levels]
            level_k = [k for k, v in levels]
            #plt.boxplot(level_v, *args, **kwargs)
            #ax.set_xticklabels(level_k)
            violinplot(level_v, labels=level_k, ax=ax, *args, **kwargs)
        #otherwise do a mosaic plot
        else:
            x_name = (x.name or 'x')
            y_name = (y.name or 'y')
            data = pd.DataFrame({x_name: x, y_name: y})
            mosaicplot.mosaic(data, index=[x_name, y_name],
                    ax=ax, *args, **kwargs)
    ax.set_xlabel(x.name)
    ax.set_ylabel(y.name)


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


def _array4name(name, data):
    """given a name/patsy formula obtain the dataframe data from it
    """
    try:
        # try to use it as a valid index
        value = data[name]
    except KeyError:
        #if it fails try it as a patsy formula
        # the +0 is needed to avoid the intercept column
        value = pd.Series(patsy.dmatrix(name + '+0', data)[:, 0])
        value.name = name
    return value


def facet_plot(formula, data, subset=None, *args, **kwargs):
    """make a faceted plot of two variables divided into categories

    the formula should follow the sintax of the faceted plot:

        endog ~ exog | factor

    where both the endog and the factor are optionals.
    If multiple factors are inserted divided by space the cartesian
    product of their level will be used.

    All the factor will be treated as patsy formulas, but are limited to
    monovariate endogenous and exogenous variables.
    """
    fig = plt.figure()
    y, x, facet = _formula_split(formula)
    #if a subset is specified use it to trim the dataframe
    if subset:
        data = data[subset]
    # obtain a list of (category, subset of the dataframe)
    elements = _elements4facet(facet, data)
    # automatically select the number of subplots as a square of this side
    side_num = np.ceil(np.sqrt(len(elements)))
    #for each subplot create the plot
    for idx, (level, value) in enumerate(elements, 1):
        ax = fig.add_subplot(side_num, side_num, idx)
        #choose if use the name ad a dataframe index or a patsy formula
        value_x = _array4name(x, value)
        value_y = _array4name(y, value) if y else None
        # launch the autoplot
        _autoplot(value_x, value_y, ax, *args, **kwargs)
        ax.set_title(level)
    fig.canvas.set_window_title(formula)
    fig.tight_layout()
    return fig

if __name__ == '__main__':
    N = 20
    data = pd.DataFrame({'int_1': plt.randint(0, 5, size=N),
                         'int_2': plt.randint(0, 5, size=N),
                         'float_1': plt.randn(N),
                         'float_2': plt.randn(N),
                         'cat_1': ['lizard'] * 7 + ['dog'] * 7 + ['newt'] * 6,
                         'cat_2': (['men'] * 4 + ['women'] * 11
                                  + ['men'] * 5)})

    affair = datasets.fair.load_pandas().data
    rate_marriage = {1: '1 very poor', 2: '2 poor', 3: '3 fair',
    4: '4 good', 5: '5 very good'}
    l = lambda s: rate_marriage[s]
    affair['rate_marriage'] = affair['rate_marriage'].apply(l)
    religious = {1: '1 not', 2: '2 mildly', 3: '3 fairly', 4: '4 strongly'}
    affair['religious'] = affair['religious'].apply(lambda s: religious[s])
    occupation = {1: 'student',
        2: 'farming, agriculture; semi-skilled, or unskilled worker',
        3: 'white-colloar',
        4: ('teacher counselor social worker, nurse; artist, writers; '
            'technician, skilled worker'),
        5: 'managerial, administrative, business',
        6: 'professional with advanced degree'}
    affair['occupation'] = affair['occupation'].apply(lambda s: occupation[s])
    l = lambda s: occupation[s]
    affair['occupation_husb'] = affair['occupation_husb'].apply(l)
    affair['cheated'] = affair.affairs > 0

    #_autoplot(data.int_1)
    #_autoplot(data.cat_1)
    #_autoplot(data.float_1)
    #_autoplot(data.float_1, data.float_2)
    #_autoplot(data.float_1, data.int_2)
    #_autoplot(data.float_1, data.cat_2)
    #_autoplot(data.int_1, data.float_2)
    #_autoplot(data.int_1, data.int_2)
    #_autoplot(data.int_1, data.cat_2)
    #_autoplot(data.cat_1, data.float_2)
    #_autoplot(data.cat_1, data.int_2)
    #_autoplot(data.cat_1, data.cat_2)
    assert _formula_split('y ~ x | f') == ('y', 'x', 'f')
    assert _formula_split('y ~ x') == ('y', 'x', None)
    assert _formula_split('x | f') == (None, 'x', 'f')
    assert _formula_split('x') == (None, 'x', None)

    # basic facet_plots for variuos combinations
    #facet_plot('int_1 | cat_1', data)
    #facet_plot('int_1 ~ float_1 | cat_1', data)
    #facet_plot('int_1 ~ float_1 | cat_2', data)
    #facet_plot('int_1 ~ float_1', data)
    #facet_plot('int_1', data)
    #facet_plot('float_1', data, facecolor='r', alpha=1.0)
    #facet_plot('cat_1 ~ cat_2', data)
    #facet_plot('float_1 ~ float_2 | cat_1', data)

    # it can split even for integer levels!
    #facet_plot('float_1 ~ float_2 | int_1', data)

    #multiple classes for the categorical
    #facet_plot('float_1 | cat_1 cat_2', data)

    #categorical with patsy
    #facet_plot('float_1 | cat_1 + cat_2', data)

    #categorical with patsy version 2
    #facet_plot('float_1 | cat_1 * cat_2', data)

    #facet_plot('I(float_1*4) ~ I(float_2 + 3)', data)

    #facet_plot('yrs_married ~ religious', affair)
    #facet_plot('yrs_married ~ religious | educ', affair)
    #facet_plot('yrs_married ~ age', affair)

    import pandas as pd
    import pylab as plt
    data = pd.DataFrame({'x':plt.randn(100),
                         'y':plt.randn(100),
                         'c1': ['a']*50 + ['b']*50,
                         'c2': ['x']*40 + ['y']*50 + ['x']*10})
    facet_plot('x | c2', data)
    plt.show()