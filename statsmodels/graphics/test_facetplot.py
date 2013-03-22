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
from nose.tools import with_setup
from nose.tools import nottest
import os


def my_setup():
    # ok, it's ugly to use globals, but it's only for this time, I promise :)
    global N
    global data
    N = 1000
    data = pd.DataFrame({
        'int_1': plt.randint(5, 15, size=N),
        'int_2': plt.randint(-3, 2, size=N),
        'int_3': plt.randint(2, 5, size=N),
        'float_1': 4 * plt.randn(N),
        'float_2': plt.randn(N),
        'float_5': plt.rand(N),
        'cat_1': ['aeiou'[i] for i in plt.randint(0, 5, size=N)],
        'cat_2': ['BCDF'[i] for i in plt.randint(0, 4, size=N)],
        'cat_3': np.r_[np.ones(N//2), np.zeros(N//2)].astype(int),
        'sin': np.sin(np.r_[0.0:10.0:N*1j]),
        'cos': np.cos(np.r_[0.0:10.0:N*1j]),
        'lin': np.r_[0.0:10.0:N*1j],
        'lin2': 0.5*np.r_[0.0:10.0:N*1j], })
    data['float_3'] = data['float_1']+data['float_2']+plt.randn(N)
    data['float_4'] = data['float_1']*data['float_2']+plt.randn(N)
    data['int_4'] = data['int_1'] + data['cat_3'] * 30
    data[u'àèéòù'] = plt.randn(N)
    data['x.1'] = plt.randn(N)
    data['x 1'] = plt.randn(N)

class base4test(object):
    def setUp(self):
        global N
        global data
        my_setup()
        self.N = N
        self.data = data

    def tearDown(self):
        plt.close("all")



@with_setup(my_setup)
def test_formula_split():
    from statsmodels.graphics.facetplot import _formula_split
    #test the formula split
    assert _formula_split('y ~ x | f') == ('y', 'x', 'f')
    assert _formula_split('y ~ x') == ('y', 'x', None)
    assert _formula_split('x | f') == (None, 'x', 'f')
    assert _formula_split('x') == (None, 'x', None)


# analyze_categories

@with_setup(my_setup)
def test_analyze_categories():
    from statsmodels.graphics.facetplot import _analyze_categories
    assert (_analyze_categories(data.int_1) ==
            {'int_1':[5, 6, 7, 8, 9, 10, 11, 12, 13, 14]})
    assert (_analyze_categories(data.float_1) ==
            {'float_1':sorted(data.float_1.unique())})
    assert (_analyze_categories(data.cat_1) ==
            {'cat_1': ['a', 'e', 'i', 'o', 'u']})
    assert (_analyze_categories(data[['cat_1', 'float_1']])
            == {'cat_1': ['a', 'e', 'i', 'o', 'u'],
                'float_1':sorted(data.float_1.unique())})
    assert (_analyze_categories(data[['cat_1', 'float_1', 'cat_2']])
            == {'cat_1': ['a', 'e', 'i', 'o', 'u'],
                'cat_2': ['B', 'C', 'D', 'F'],
                'float_1':sorted(data.float_1.unique())})


@with_setup(my_setup)
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

@with_setup(my_setup)
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


@nottest
@with_setup(my_setup)
def test_violinplot():
    #test for the oracle and the violinplot
    #should work both on vertical and horizontal
    facet_plot('float_1 ~ cat_1 | cat_2', data)
    facet_plot('cat_1 ~ float_1 | cat_2', data)
    #should be consistent even with sparse categories
    facet_plot('float_1 ~ cat_1 | cat_1', data)
    facet_plot('cat_1 ~ float_1 | cat_1', data)
    plt.show()
    plt.close("all")


@nottest
@with_setup(my_setup)
def test_boxplot():
#    #test for the boxplot
#    #should work both on vertical and horizontal
    facet_plot('float_1 ~ cat_1 | cat_2', data, kind='boxplot')
    facet_plot('cat_1 ~ float_1 | cat_2', data, kind='boxplot')
#    #should be consistent even with sparse categories
    facet_plot('float_1 ~ cat_1 | cat_1', data, kind='boxplot')
    facet_plot('cat_1 ~ float_1 | cat_1', data, kind='boxplot')
#    # management of the multivariate case
    facet_plot('float_1 + float_3 ~ cat_1 | cat_2', data, kind='boxplot')
    facet_plot('float_1 + float_3 + float_2 ~ cat_1 | cat_2', data, kind='boxplot')
    facet_plot('cat_1 ~ float_1 + float_3 + float_2 | cat_2', data, kind='boxplot')
    facet_plot('float_1 + float_3 + float_2 + float_4 ~ cat_1 | cat_2', data, kind='boxplot')
    plt.show()
    plt.close("all")


@nottest
@with_setup(my_setup)
def test_autoplot():
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


@nottest
@with_setup(my_setup)
def test_kde():
    facet_plot('float_1', data)
    facet_plot('float_1 + float_2 + float_3 | cat_2', data, include_total=True)
    facet_plot('float_1 | cat_1', data)
    facet_plot('float_1 ~ float_2 | cat_1', data, 'kde')
    plt.show()
    plt.close("all")


@nottest
@with_setup(my_setup)
def test_scatter():
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

@nottest
@with_setup(my_setup)
def test_line():
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


@nottest
@with_setup(my_setup)
def test_matrix():
    facet_plot('cat_1', data, 'matrix')
    facet_plot('int_1', data, 'matrix')
    facet_plot('int_1 ~ float_2 | cat_1', data, 'matrix')
    facet_plot('float_1 ~ float_2', data, 'matrix')
    facet_plot('int_1 ~ cat_2', data, 'matrix')
    plt.show()
    plt.close("all")


@nottest
@with_setup(my_setup)
def test_insertion_in_axes():
    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    facet_plot('cat_2 ~ cat_1', data, ax=ax)
    ax = fig.add_subplot(2, 2, 2)
    facet_plot('cat_1', data, ax=ax)
    ax = fig.add_subplot(2, 2, 3)
    facet_plot('sin ~ lin', data, 'lines', ax=ax)
    ax = fig.add_subplot(2, 2, 4)
    facet_plot('float_1 ~ cat_1 + float_2', data, ax=ax, jitter=0.2)
    fig.tight_layout()
    fig.canvas.set_window_title('mixed facet_plots')
    plt.show()
    plt.close("all")
    #this should give error
    #facet_plot('cat_2 ~ cat_1 | int_1', data, ax=ax)


@nottest
@with_setup(my_setup)
def test_create_dataframe():
    float_1 = plt.randn(100)
    gender = np.r_[np.ones(50), np.zeros(50)].astype(int)
    # can pass a normal dictionary to the creator
    facet_plot('float_1', {'float_1': float_1})
    # will drop the nan automatically
    float_1[0] = np.nan
    facet_plot('float_1', {'float_1': float_1})
    # subsetting the data
    facet_plot('float_1', {'float_1': float_1}, subset=(float_1 > 0))
    # capture the environment to explore it
    facet_plot('float_1')
    facet_plot('I(float_1**2)')
    # can I also subset it?
    facet_plot('float_1', subset=(float_1 > 0))
    # control that even mixed formulas and names don't create confusion
    facet_plot('np.log(float_1) + float_1')
    # will return error as the nan value create an error
    #facet_plot('float_1', {'float_1': float_1}, drop_na=False)
    plt.show()
    plt.close("all")


class TestSpecialNames(base4test):
    def test_names_unicode(self):
        fig = facet_plot(u'float_1 ~ àèéòù', self.data)
        assert fig.axes[0].get_xlabel() == u'àèéòù'

    def test_names_invalid_1(self):
        fig = facet_plot(u'float_1 ~ x.1', self.data)
        assert fig.axes[0].get_xlabel() == 'x.1'

    def test_names_invalid_2(self):
        fig = facet_plot(u'float_1 ~ x 1', self.data)
        assert fig.axes[0].get_xlabel() == 'x 1'

    def test_names_with_Q(self):
        fig = facet_plot(u'float_1 ~ Q("x 1")', self.data)
        assert fig.axes[0].get_xlabel() == 'Q("x 1")'

@nottest
@with_setup(my_setup)
def test_ellipse():
    facet_plot('float_1 ~ float_2 | cat_1', data, 'ellipse')
    facet_plot('float_1 ~ cat_2 | cat_1', data, 'ellipse')
    facet_plot('float_3 ~ float_1 | cat_1', data, 'ellipse')
    plt.show()
    plt.close("all")


@nottest
@with_setup(my_setup)
def test_hexbin():
    facet_plot('float_1 ~ float_2 | cat_1', data, 'hexbin')
    facet_plot('float_1 ~ cat_2 | cat_1', data, 'hexbin')
    facet_plot('float_3 ~ float_1 | cat_1', data, 'hexbin')
    plt.show()
    plt.close("all")


@nottest
@with_setup(my_setup)
def test_counter():
    #facet_plot('cat_1', data, 'counter')
    #facet_plot('cat_1 | cat_2', data, 'counter')
    #facet_plot('cat_1 | cat_1', data, 'counter')
    #facet_plot('int_1', data, 'counter')
    #facet_plot('int_1 | cat_1', data, 'counter')
    #facet_plot('int_2 | cat_1', data, 'counter')
    #facet_plot('cat_2 | cat_1', data, 'counter')
    #facet_plot('cat_2 | cat_1', data, 'counter', confidence=0)

    #facet_plot('int_1', data, 'counter', as_categorical=1)
    #facet_plot('I(int_1**2)', data, 'counter', as_categorical=1)

    # integer data with two groups of values
    # in one case I should see a separation, in the other not
    facet_plot('int_4', data, 'counter')
    facet_plot('int_4', data, 'counter', as_categorical=1)

    # same when divided into facets
    facet_plot('int_4 | cat_3', data, 'counter')
    facet_plot('int_4 | cat_3', data, 'counter', as_categorical=1)

    # now when I apply some transformation
    facet_plot('I(int_4**2)', data, 'counter')
    facet_plot('I(int_4**2)', data, 'counter', as_categorical=1)

    plt.show()
    plt.close("all")


class Test_counter(base4test):
    def test_labeling_simple_1(self):
        # the data is plit in two groups, I should have
        # the labels as normal integers
        fig = facet_plot('int_4', self.data, 'counter')
        plt.draw()
        ax = fig.axes[0]
        labels = [t.get_text().replace(u'\u2212', '-')
                  for t in ax.get_xticklabels() ]
        positions = [float(t) for t in ax.get_xticks() ]
        assert positions == [ float(t) for t in labels if t]

    def test_labeling_simple_2(self):
        # treating them as categorical they should be always
        # equal to the complete level of the variable
        fig = facet_plot('int_4', self.data, 'counter', as_categorical=1)
        plt.draw()
        ax = fig.axes[0]
        labels = [int(t.get_text().replace(u'\u2212', '-'))
                  for t in ax.get_xticklabels() ]
        expected = sorted(self.data['int_4'].unique())
        assert labels == expected

#my_setup()
#test_counter()
#plt.close("all")

if __name__ == "#__main__":
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

os.chdir('/home/enrico/lavoro/markage/dati 08-02-13')
with open('MARK-AGE Database 08.02.2013.csv', 'rb') as f:
    markage = pd.read_csv(f, encoding='latin', engine='python',
                          true_values=['Y', 'yes'], false_values=['N', 'no'],
                          na_values=['na', 'NaN', '', 'nan', ' ', 'nap'])
#rimuove le linee vuote per via della codifica e aggiusta l'indice
markage = markage[-markage.gender.isnull()]
markage.index = range(len(markage))
print markage.columns[:5]
#facet_plot('AgeClass5 | gender + subject_group ', markage,
#           'counter', as_categorical=1)
markage['AgeClass5'] = markage['AgeClass5'].astype(int)
markage['AgeClass10'] = markage['AgeClass10'].astype(int)
facet_plot('AgeClass5 | subject_group + gender', markage,
           'counter', as_categorical=True, facet_grid=True)
plt.show()
