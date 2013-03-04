from __future__ import division

import numpy as np
from statsmodels.compatnp.collections import OrderedDict
from collections import Counter

from numpy import iterable, r_, cumsum, array
from statsmodels.graphics import utils
from statsmodels.graphics import mosaicplot

from scipy.stats.kde import gaussian_kde

import pylab as plt
import pandas as pd

def _auto_hist(data, ax, *args, **kwargs):
    """
    given a pandas series infers the type of data and print
    a barplot, an histogram or a density plot given the circumstances
    """
    data = pd.Series(data)
    if data.dtype == float:
        ax.hist(data, bins=int(np.sqrt(len(data))), normed=True,
            facecolor='#999999', edgecolor='k', alpha=0.33)
        my_pdf = gaussian_kde(data)
        x = np.linspace(np.min(data), np.max(data), 100)
        kwargs.setdefault('facecolor', '#777777')
        kwargs.setdefault('alpha', 0.66)
        ax.fill_between(x, my_pdf(x), *args, **kwargs)
        ax.set_ylim(0.0, None)
        ax.set_ylabel('Density')
    else:
        res = Counter(data)
        key = sorted(res.keys())
        if isinstance(key[0], int):
            key = range(min(key), max(key) + 1)
        val = [res[i] for i in key]
        kwargs.setdefault('facecolor', '#777777')
        kwargs.setdefault('align', 'center')
        kwargs.setdefault('edgecolor', None)
        ax.bar(range(len(val)), val, *args, **kwargs)
        ax.set_ylabel('Counts')
        ax.set_xticks(range(len(val)))
        ax.set_yticks([int(i) for i in ax.get_yticks()])
        ax.set_xlim(-0.5, len(val) - 0.5)
        ax.set_xticklabels(key)
    ax.set_xlabel(data.name)
    return ax

def _autoplot(x,y=None, ax=None, *args, **kwargs):
    if y is None or y is x:
        return _auto_hist(x, ax, *args, **kwargs)
    x = pd.Series(x)
    y = pd.Series(y)
    if x.dtype == float or x.dtype == int:
        #TODO: if both are ints should add a jitter
        if y.dtype == float or y.dtype == int:
            plt.scatter(x, y, *args, **kwargs)
            if y.dtype == int:
                ax.set_yticks([int(i) for i in ax.get_yticks()])
            if x.dtype == int:
                ax.set_xticks([int(i) for i in ax.get_xticks()])
        else:
            data = pd.DataFrame({'x': x, 'f': y})
            levels = list(data.groupby('f')['x'])
            level_v = [v for k, v in levels]
            level_k = [k for k, v in levels]
            plt.boxplot(level_v, vert=False, *args, **kwargs)
            ax.set_yticklabels(level_k)
    else:
        if y.dtype == float or y.dtype == int:
            data = pd.DataFrame({'x': y, 'f': x})
            levels = list(data.groupby('f')['x'])
            level_v = [v for k, v in levels]
            level_k = [k for k, v in levels]
            plt.boxplot(level_v, *args, **kwargs)
            ax.set_xticklabels(level_k)
        else:
            x_name = (x.name or 'x')
            y_name = (y.name or 'y')
            data = pd.DataFrame({x_name: x, y_name: y})
            mosaicplot.mosaic(data, index=[x_name, y_name],
                    ax=ax, *args, **kwargs)
    ax.set_xlabel(x.name)
    ax.set_ylabel(y.name)


def _formula_split(formula):
    if '|' in formula:
        f = formula.split('|')[1].strip()
        formula = formula.split('|')[0]
    else:
        f = None
    if '~' in formula:
        x = formula.split('~')[1].strip()
        y = formula.split('~')[0].strip()
    else:
        x = None
        y = formula.strip()
    return y, x, f


def facet_plot(formula, data, *args, **kwargs):
    fig = plt.figure()
    y, x, facet = _formula_split(formula)
    if x is None:
        x, y = y, x
    if facet is not None:
        elements = list(data.groupby(facet))
    else:
        elements = [['', data]]
    side_num = np.ceil(np.sqrt(len(elements)))
    for idx, (level, value) in enumerate(elements, 1):
        ax = fig.add_subplot(side_num, side_num, idx)
        if y is not None:
            _autoplot(value[x], value[y], ax, *args, **kwargs)
        else:
            _autoplot(value[x], None, ax, *args, **kwargs)
        ax.set_title(level)
    fig.canvas.set_window_title(formula)
    fig.tight_layout()
    return fig

if __name__ == '__main__':
    N = 20
    data = pd.DataFrame({'int_1': plt.randint(0,5,size=N),
                         'int_2': plt.randint(0,5,size=N),
                         'float_1': plt.randn(N),
                         'float_2': plt.randn(N),
                         'cat_1': ['lizard']*7 + ['dog']*7 + ['newt']*6,
                         'cat_2': (['men']*4 + ['women']*11
                                  + ['men']*5)})
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
    assert _formula_split('y | f') == ('y', None, 'f')
    assert _formula_split('y') == ('y', None, None)

    facet_plot('int_1 | cat_1', data)
    facet_plot('int_1 ~ float_1 | cat_1', data)
    facet_plot('int_1 ~ float_1 | cat_2', data)
    facet_plot('int_1 ~ float_1', data)
    facet_plot('int_1', data)
    facet_plot('float_1', data, facecolor='r', alpha=1.0)
    facet_plot('cat_1 ~ cat_2', data)
    plt.show()