# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:34:21 2013

@author: enrico
"""
import pandas as pd
import pylab as plt
import numpy as np
from numpy.testing import run_module_suite
from statsmodels.graphics.facetplot import facet_plot


N = 1000
data = pd.DataFrame({
    'int_1': plt.randint(0, 10, size=N),
    'int_2': plt.randint(0, 5, size=N),
    'int_3': plt.randint(1, 3, size=N),
    'float_1': 4 * plt.randn(N),
    'float_2': plt.randn(N),
    'float_5': plt.rand(N),
    'cat_1': ['aeiou'[i] for i in plt.randint(0, 5, size=N)],
    'cat_2': ['BCDF'[i] for i in plt.randint(0, 4, size=N)],
    'sin': np.sin(np.r_[0.0:10.0:N*1j]),
    'cos': np.cos(np.r_[0.0:10.0:N*1j]),
    'lin': np.r_[0.0:10.0:N*1j],
    'lin2': 0.5*np.r_[0.0:10.0:N*1j], })
data['float_3'] = data['float_1']+data['float_2']+plt.randn(N)
data['float_4'] = data['float_1']*data['float_2']+plt.randn(N)


def test_formula_split():
    from statsmodels.graphics.facetplot import _formula_split
    #test the formula split
    assert _formula_split('y ~ x | f') == ('y', 'x', 'f')
    assert _formula_split('y ~ x') == ('y', 'x', None)
    assert _formula_split('x | f') == (None, 'x', 'f')
    assert _formula_split('x') == (None, 'x', None)


# analyze_categories
def test_analyze_categories():
    from statsmodels.graphics.facetplot import _analyze_categories
    assert _analyze_categories(data.int_1) == {}
    assert _analyze_categories(data.float_1) == {}
    assert (_analyze_categories(data.cat_1) ==
            {'cat_1': ['a', 'e', 'i', 'o', 'u']})
    assert (_analyze_categories(data[['cat_1', 'float_1']])
            == {'cat_1': ['a', 'e', 'i', 'o', 'u']})
    assert (_analyze_categories(data[['cat_1', 'float_1', 'cat_2']])
            == {'cat_1': ['a', 'e', 'i', 'o', 'u'],
                'cat_2': ['B', 'C', 'D', 'F']})


def test_array4name():
    from statsmodels.graphics.facetplot import _array4name
    #test the array4name
    assert all(_array4name('int_1 + int_2',
                           data).columns == ['int_1', 'int_2'])
    assert all(_array4name('int_1 + cat_2',
                           data).columns == ['int_1', 'cat_2'])
    assert isinstance(_array4name('cat_2', data), pd.Series)
    assert isinstance(_array4name('int_1', data), pd.Series)


#######################################
# test the oracle
####################
def test_oracle():
    from statsmodels.graphics.facetplot import _oracle
    # single dimension
    assert _oracle(data.float_1, None) == 'kde'
    assert _oracle(data[['float_2', 'float_3']], None) == 'kde'
    assert _oracle(data.int_1, None) == 'counter'
    assert _oracle(data.cat_1, None) == 'counter'
    #monovariate
    assert _oracle(data.float_1, data.float_1) == 'scatter'
    assert _oracle(data.int_1, data.float_1) == 'scatter'
    assert _oracle(data.cat_1, data.float_1) == 'violinplot'
    assert _oracle(data.float_1, data.int_1) == 'scatter'
    assert _oracle(data.int_1, data.int_1) == 'scatter'
    assert _oracle(data.cat_1, data.int_1) == 'violinplot'
    assert _oracle(data.float_1, data.cat_1) == 'violinplot'
    assert _oracle(data.int_1, data.cat_1) == 'violinplot'
    assert _oracle(data.cat_1, data.cat_1) == 'mosaic'
    # multivariate...long and hard
    assert _oracle(data.float_1, data[['float_2', 'float_3']]) == 'scatter'
    assert _oracle(data.int_1, data[['float_2', 'float_3']]) == 'scatter'
    assert _oracle(data.cat_1, data[['float_2', 'float_3']]) == 'scatter'


def _test_violinplot():
    #test for the oracle and the violinplot
    #should work both on vertical and horizontal
    facet_plot('float_1 ~ cat_1 | cat_2', data)
    facet_plot('cat_1 ~ float_1 | cat_2', data)
    #should be consistent even with sparse categories
    facet_plot('float_1 ~ cat_1 | cat_1', data)
    facet_plot('cat_1 ~ float_1 | cat_1', data)
    plt.show()
    plt.close("all")


def _test_autoplot():
    #keeper function for the test of the old data_centric functions
    facet_plot('float_1', data, 'scatter')
    facet_plot('int_1', data, 'scatter')
    facet_plot('cat_1', data, 'lines')
    facet_plot('float_5 ~ cat_1', data, 'scatter')
    facet_plot('float_5 ~ float_1 | cat_1', data,
               'scatter', include_total=True)
    facet_plot('cat_2 ~ cat_1', data)
    plt.show()
    plt.close("all")


def _test_kde():
    facet_plot('float_1', data)
    facet_plot('float_1 + float_2 + float_3 | cat_2', data, include_total=True)
    facet_plot('float_1 | cat_1', data)
    facet_plot('float_1 ~ float_2 | cat_1', data, 'kde')
    plt.show()
    plt.close("all")


def _test_scatter():
    facet_plot('float_1', data, 'scatter')
    facet_plot('cat_1', data, 'scatter')
    facet_plot('float_1 + float_2 + float_3 | cat_2', data, 'scatter')
    facet_plot('float_1 ~ cat_2 | cat_1', data, 'scatter')
    facet_plot('float_1 + float_3 ~ cat_2 | cat_1', data, 'scatter')
    facet_plot('cat_2 ~ float_2 | cat_1', data, 'scatter')
    facet_plot('float_1 ~ float_2 + float_3', data, 'scatter')
    facet_plot('float_1 + float_4 ~ cat_2 + float_3', data, 'scatter')
    plt.show()
    plt.close("all")


def _test_line():
    facet_plot('float_1 + float_2', data, 'lines')
    facet_plot('cat_1', data, 'lines', jitter=0.2)
    facet_plot('float_1 | cat_2', data, 'lines')
    facet_plot('float_1 ~ cat_2 | cat_1', data, 'lines')
    facet_plot('float_1 + float_3 ~ cat_2 | cat_1', data, 'lines')
    facet_plot('lin ~ float_2 | cat_1', data, 'lines')
    facet_plot('lin ~ sin + cos', data, 'lines')
    facet_plot('float_1 + float_4 ~ cat_2 + float_3', data, 'lines')
    plt.show()
    plt.close("all")


def _test_matrix():
    facet_plot('cat_1', data, 'matrix')
    facet_plot('int_1', data, 'matrix')
    facet_plot('int_1 ~ float_2 | cat_1', data, 'matrix')
    facet_plot('float_1 ~ float_2', data, 'matrix')
    facet_plot('int_1 ~ cat_2', data, 'matrix')
    plt.show()
    plt.close("all")

if __name__ == "__main__":
    run_module_suite()

#facet_plot('float_1', data, 'scatter')
#facet_plot('int_1', data, 'scatter')
#facet_plot('cat_1', data, 'lines')
#facet_plot('float_5 ~ cat_1', data, 'scatter')
#facet_plot('float_5 ~ float_1 | cat_1', data, 'scatter', include_total=True)
#facet_plot('cat_2 ~ cat_1', data)

#facet_plot('float_4 + float_3 ~ float_1 + float_2 | cat_2', data)
#facet_plot('float_4 + float_3 ~ cat_1 + float_2', data)
#facet_plot('float_4 ~ float_1 + cat_2', data, jitter=0.5)
#facet_plot('float_4 + float_3 ~ cat_1 + cat_2', data, jitter=False)
#facet_plot('float_4 + float_3 ~ cat_1 + cat_2', data, jitter=True)
#facet_plot('float_4 + float_3 ~ cat_1 + cat_2', data, jitter=0.5)
#facet_plot('float_4 ~ cat_1 | cat_1', data)
#facet_plot('float_3 ~ float_1 + float_2 | cat_2', data, 'scatter')
#facet_plot('float_4 ~ float_1 + float_2 | cat_2', data, 'lines');
#facet_plot('float_4 ~ float_1 + float_2 | cat_2', data, 'scatter_coded');
#facet_plot('float_4 ~ float_1 + cat_1 | cat_2', data, 'scatter_coded');
#facet_plot('float_4 ~ float_1 + float_2 | cat_2', data, 'wireframe');
#facet_plot('lin ~ cos + sin | cat_2', data, 'lines');
#facet_plot('lin2 + lin:int_3 ~ cos + sin | cat_2', data, 'wireframe')



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
#facet_plot('float_3 + float_4 ~ float_1 + float_2 | cat_1', data)
#facet_plot('float_4 + float_3 ~ float_1 + float_2', data)
#facet_plot('float_3 ~ float_1 + float_2', data)

#facet_plot('int_3 ~ cat_1 + cat_2', data)
#facet_plot('int_2 ~ float_1 + float_2', data)
#facet_plot('int_1 ~  int_2', data, 'matrix', interpolation='nearest');

#fig = plt.figure()
#ax = fig.add_subplot(2, 2, 1)
#facet_plot('cat_2 ~ cat_1', data, ax=ax)
#ax = fig.add_subplot(2, 2, 2)
#facet_plot('cat_1', data, ax=ax)
#ax = fig.add_subplot(2, 2, 3)
#facet_plot('sin ~ lin', data, 'lines', ax=ax)
#ax = fig.add_subplot(2, 2, 4)
#facet_plot('float_1 ~ cat_1 + float_2', data, ax=ax, jitter=0.2)
#fig.tight_layout()
#fig.canvas.set_window_title('mixed facet_plots')
##this should give error
##facet_plot('cat_2 ~ cat_1 | int_1', data, ax=ax)

def _test_evalenvironmentcapture():
    import patsy
    from statsmodels.graphics import facetplot
    float_1 = plt.randn(10)
    facetplot._array4name('float_1', patsy.EvalEnvironment.capture().namespace)