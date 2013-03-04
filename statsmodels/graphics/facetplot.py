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
    data = np.asarray(data)
    if data.dtype == float:
        ax.hist(data, bins=int(np.sqrt(len(data))), normed=True,
            facecolor='#999999', edgecolor='k', alpha=0.33)
        my_pdf = gaussian_kde(data)
        x = np.linspace(np.min(data), np.max(data), 100)
        kwargs.setdefault('facecolor', '#777777')
        kwargs.setdefault('alpha', 0.66)
        ax.fill_between(x, my_pdf(x), *args, **kwargs)
        ax.set_ylim(0.0, None)
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
    return ax

def _autoplot(x,y=None, ax=None, *args, **kwargs):
    fig, ax = utils.create_mpl_ax(ax)
    if y is None or y is x:
        return _auto_hist(x, ax, *args, **kwargs)
    x = pd.Series(x)
    y = pd.Series(y)
    if x.dtype == float or x.dtype == int:
        #TODO: if both are ints should add a jitter
        if y.dtype == float or y.dtype == int:
            return plt.scatter(x, y, *args, **kwargs)
        else:
            data = pd.DataFrame({'x': x, 'f': y})
            levels = list(data.groupby('f')['x'])
            level_v = [v for k, v in levels]
            level_k = [k for k, v in levels]
            res = plt.boxplot(level_v, vert=False, *args, **kwargs)
            ax.set_yticklabels(level_k)
            return res
    else:
        if y.dtype == float or y.dtype == int:
            data = pd.DataFrame({'x': y, 'f': x})
            levels = list(data.groupby('f')['x'])
            level_v = [v for k, v in levels]
            level_k = [k for k, v in levels]
            res = plt.boxplot(level_v, *args, **kwargs)
            ax.set_xticklabels(level_k)
            return res
        else:
            x_name = (x.name or 'x')
            y_name = (y.name or 'y')
            data = pd.DataFrame({x_name: x, y_name: y})
            mosaicplot.mosaic(data, index=[x_name, y_name], ax=ax, *args, **kwargs)



if __name__ == '__main__':
    N = 20
    data = pd.DataFrame({'int_1': plt.randint(0,5,size=N),
                         'int_2': plt.randint(0,5,size=N),
                         'float_1': plt.randn(N),
                         'float_2': plt.randn(N),
                         'cat_1': ['cat']*13 + ['dog']*7,
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
    gender = ['male', 'male', 'male', 'female', 'female', 'female']
    pet = ['cat', 'dog', 'dog', 'cat', 'dog', 'cat']
    data = pd.DataFrame({'gender': gender, 'pet': pet})
    mosaicplot.mosaic(data, ['gender', 'pet'], title='dataframe by key 1', axes_label=False)

    plt.show()