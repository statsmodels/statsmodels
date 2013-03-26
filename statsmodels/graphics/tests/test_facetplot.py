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
import logging
from statsmodels.graphics.facetplot import _formula_terms
from statsmodels.graphics.facetplot import _beautify

#It's less pretty but make easier tests
import matplotlib as mpl
mpl.rcParams['axes.unicode_minus']=False

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
    data['x%&$1'] = plt.randn(N)

class base4test(object):
    def setUp(self):
        global N
        global data
        my_setup()
        self.N = N
        self.data = data

    def tearDown(self):
        if hasattr(self.__class__, '__show__') and self.__show__:
            plt.show()
        else:
            plt.draw()
        plt.close("all")

####################################################################
# HELPERS FUNCTIONS
####################################################################

@with_setup(my_setup)
def test_formula_split():
    from statsmodels.graphics.facetplot import _formula_split
    #test the formula split
    assert _formula_split('y ~ x | f') == ('y', 'x', 'f', None)
    assert _formula_split('y ~ x') == ('y', 'x', None, None)
    assert _formula_split('x | f') == (None, 'x', 'f', None)
    assert _formula_split('x') == (None, 'x', None, None)

    assert _formula_split('y ~ x | f ~ p') == ('y', 'x', 'f', 'p')
    assert _formula_split('y ~ x | ~ p') == ('y', 'x', None, 'p')
    assert _formula_split('x | f ~ p') == (None, 'x', 'f', 'p')
    assert _formula_split('x | ~ p') == (None, 'x', None, 'p')


@with_setup(my_setup)
def test_stack_by():
    from statsmodels.graphics.facetplot import _stack_by
    small = data[['cat_1', 'cat_2', 'cat_3']]
    reduced = _stack_by(small, 'cat_3')
    expected_col = ['Q("cat_1: 0")', 'Q("cat_1: 1")',
                    'Q("cat_2: 0")', 'Q("cat_2: 1")']
    assert sorted(reduced.columns) == expected_col
    reduced2 = _stack_by(small, ['cat_2', 'cat_3'])
    expected_col = ['Q("cat_1: B: 0")',
                    'Q("cat_1: B: 1")',
                    'Q("cat_1: C: 0")',
                    'Q("cat_1: C: 1")',
                    'Q("cat_1: D: 0")',
                    'Q("cat_1: D: 1")',
                    'Q("cat_1: F: 0")',
                    'Q("cat_1: F: 1")']
    assert sorted(reduced2.columns) == expected_col


@with_setup(my_setup)
def test_analyze_categories():
    from statsmodels.graphics.facetplot import _analyze_categories
    assert (_analyze_categories(data[['int_1']]) ==
            {'int_1':[5, 6, 7, 8, 9, 10, 11, 12, 13, 14]})
    assert (_analyze_categories(data[['float_1']]) ==
            {'float_1':sorted(data.float_1.unique())})
    assert (_analyze_categories(data[['cat_1']]) ==
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
    assert all(_array4name('int_1 + int_2', data).columns == ['int_1', 'int_2'])
    assert all(_array4name('int_1 + cat_2', data).columns == ['cat_2', 'int_1'])
    assert all(_array4name('cat_2', data).columns == ['cat_2'])
    assert all(_array4name('int_1', data).columns == ['int_1'])


@with_setup(my_setup)
def test_oracle():
    from statsmodels.graphics.facetplot import _oracle
    # single dimension
    assert _oracle(data[['float_1']], None) == 'kde'
    assert _oracle(data[['float_2', 'float_3']], None) == 'kde'
    assert _oracle(data[['int_1']], None) == 'counter'
    assert _oracle(data[['cat_1']], None) == 'counter'
    #monovariate
    assert _oracle(data[['float_1']], data[['float_1']]) == 'scatter'
    assert _oracle(data[['int_1']], data[['float_1']]) == 'scatter'
    assert _oracle(data[['cat_1']], data[['float_1']]) == 'violinplot'
    assert _oracle(data[['float_1']], data[['int_1']]) == 'scatter'
    assert _oracle(data[['int_1']], data[['int_1']]) == 'scatter'
    assert _oracle(data[['cat_1']], data[['int_1']]) == 'violinplot'
    assert _oracle(data[['float_1']], data[['cat_1']]) == 'violinplot'
    assert _oracle(data[['int_1']], data[['cat_1']]) == 'violinplot'
    assert _oracle(data[['cat_1']], data[['cat_1']]) == 'mosaic'

class Test_formula_terms(base4test):
    __show__ = True
    def test_formula_terms_1(self):
        f1 = '(a+b)'
        f = " + ".join(_formula_terms(f1))
        assert f == " + ".join(_formula_terms(f))
    def test_formula_terms_2(self):
        f1 = 'I(a+b)'
        f = " + ".join(_formula_terms(f1))
        assert f == " + ".join(_formula_terms(f))
    def test_formula_terms_3(self):
        f1 = 'I(c + d) + Intercept + a + a: b + b'
        f = " + ".join(_formula_terms(f1))
        assert f == " + ".join(_formula_terms(f))
    def test_formula_terms_4(self):
        f1 = 'I(c + d) + Intercept + a + a: b + b + 1'
        f = " + ".join(_formula_terms(f1))
        assert f == " + ".join(_formula_terms(f))
    def test_formula_terms_5(self):
        f1 = 'I(c + d) + Intercept + a + a: b + b + 0'
        f = " + ".join(_formula_terms(f1))
        assert f == " + ".join(_formula_terms(f))
    def test_formula_terms_6(self):
        f1 = 'I(c + d) + Intercept + a + a: b + b -1'
        f = " + ".join(_formula_terms(f1))
        assert f == " + ".join(_formula_terms(f))
    def test_formula_terms_7(self):
        assert 'Intercept' in _formula_terms('a+b',1)
        assert 'Intercept' in _formula_terms('a+b+1',1)
        assert 'Intercept' not in _formula_terms('a+b-1',1)
        assert 'Intercept' not in _formula_terms('a+b+0',1)

        assert 'Intercept' not in _formula_terms('a+b',0)
        assert 'Intercept' not in _formula_terms('a+b+1',0)
        assert 'Intercept' not in _formula_terms('a+b-1',0)
        assert 'Intercept' not in _formula_terms('a+b+0',0)

    def test_formula_terms_8(self):
        assert 'Intercept' in _formula_terms('Intercept+a+b',1)
        assert 'Intercept' in _formula_terms('Intercept+a+b+1',1)
        assert 'Intercept' in _formula_terms('Intercept+a+b-1',1)
        assert 'Intercept' in _formula_terms('Intercept+a+b+0',1)

        assert 'Intercept' in _formula_terms('Intercept+a+b',0)
        assert 'Intercept' in _formula_terms('Intercept+a+b+1',0)
        assert 'Intercept' in _formula_terms('Intercept+a+b-1',0)
        assert 'Intercept' in _formula_terms('Intercept+a+b+0',0)



class Test_beautify(base4test):
    def test_beutify_1(self):
        s = 'I(Q("float_1") ** 0.5) + I(float_2) ~ Q("float_3") | cat_3'
        assert 'float_1 ** 0.5 + float_2 ~ float_3 | cat_3' == _beautify(s)
    def test_beutify_2(self):
        s = "whather comes to mind, with Q, I, and () or ( op )"
        assert s == _beautify(s)
    def test_beutify_1(self):
        s = 'I(Q("float_1") ** 0.5) + I(float_2) ~ Q("float_3") | cat_3'
        assert 'I(float_1 ** 0.5) + I(float_2) ~ float_3 | cat_3' == _beautify(s,0)


####################################################################
# REAL TEST OF FACET_PLOT
####################################################################


@nottest
class Test_violin(base4test):
    __show__ = True
    def test_violin_1(self):
        facet_plot('float_1 ~ cat_1 | cat_2', data)
    def test_violin_2(self):
        facet_plot('cat_1 ~ float_1 | cat_2', data)
    def test_violin_3(self):
        facet_plot('float_1 ~ cat_1 | cat_1', data)
    def test_violin_4(self):
        facet_plot('cat_1 ~ float_1 | cat_1', data)

@nottest
class Test_boxplot(base4test):
    __show__ = True
    def test_boxplot_1(self):
        facet_plot('float_1 ~ cat_1 | cat_2', data, kind='boxplot')
    def test_boxplot_2(self):
        facet_plot('cat_1 ~ float_1 | cat_2', data, kind='boxplot')
    def test_boxplot_3(self):
        facet_plot('float_1 ~ cat_1 | cat_1', data, kind='boxplot')
    def test_boxplot_4(self):
        facet_plot('cat_1 ~ float_1 | cat_1', data, kind='boxplot')
    def test_boxplot_5(self):
        facet_plot('float_1 + float_3 ~ cat_1 | cat_2', data, kind='boxplot')
    def test_boxplot_6(self):
        facet_plot('float_1 + float_3 + float_2 ~ cat_1 | cat_2', data, kind='boxplot')
    def test_boxplot_7(self):
        facet_plot('cat_1 ~ float_1 + float_3 + float_2 | cat_2', data, kind='boxplot')
    def test_boxplot_8(self):
        facet_plot('float_1 + float_3 + float_2 + float_4 ~ cat_1 | cat_2', data, kind='boxplot')
    def test_boxplot_9(self):
        facet_plot('float_1 ~ int_1 | cat_2', data, kind='boxplot')

@nottest
class Test_kde(base4test):
    __show__ = True
    def test_kde_1(self):
        facet_plot('float_1', data)
    def test_kde_2(self):
        facet_plot('float_1 + float_2 + float_3 | cat_2', data, include_total=True)
    def test_kde_3(self):
        facet_plot('float_1 | cat_1', data)
    def test_kde_4(self):
        facet_plot('float_1 ~ float_2 | cat_1', data, 'kde')

@nottest
class Test_scatter(base4test):
    __show__ = True
    def test_scatter_1(self):
        facet_plot('float_1', data, 'scatter')
    def test_scatter_2(self):
        facet_plot('cat_1', data, 'scatter')
    def test_scatter_3(self):
        facet_plot('float_1 + float_2 + float_3 | cat_2', data, 'scatter', markersize=20)
    def test_scatter_4(self):
        facet_plot('float_1 ~ cat_2 | cat_1', data, 'scatter')
    def test_scatter_5(self):
        facet_plot('float_1 + float_3 ~ cat_2 | cat_1', data, 'scatter')
    def test_scatter_6(self):
        facet_plot('cat_2 ~ float_2 | cat_1', data, 'scatter')
    def test_scatter_7(self):
        facet_plot('float_1 ~ float_2 + float_3', data, 'scatter')
    def test_scatter_8(self):
        facet_plot('float_1 + float_4 ~ cat_2 + float_3', data, 'scatter')

@nottest
class Test_lines(base4test):
    __show__ = True
    def test_lines_1(self):
        facet_plot('float_1 + float_2', data, 'lines')
    def test_lines_2(self):
        facet_plot('cat_1', data, 'lines', jitter=0.2)
    def test_lines_3(self):
        facet_plot('float_1 | cat_2', data, 'lines')
    def test_lines_4(self):
        facet_plot('float_1 ~ cat_2 | cat_1', data, 'lines')
    def test_lines_5(self):
        facet_plot('float_1 + float_3 ~ cat_2 | cat_1', data, 'lines')
    def test_lines_6(self):
        facet_plot('lin ~ float_2 | cat_1', data, 'lines')
    def test_lines_7(self):
        facet_plot('lin ~ sin + cos', data, 'lines')
    def test_lines_8(self):
        facet_plot('float_1 + float_4 ~ cat_2 + float_3', data, 'lines')


@nottest
class Test_matrix(base4test):
    __show__ = True
    def test_matrix_1(self):
        facet_plot('cat_1', data, 'matrix')
    def test_matrix_2(self):
        facet_plot('int_1', data, 'matrix')
    def test_matrix_3(self):
        facet_plot('int_1 ~ float_2 | cat_1', data, 'matrix')
    def test_matrix_4(self):
        facet_plot('float_1 ~ float_2', data, 'matrix')
    def test_matrix_5(self):
        facet_plot('int_1 ~ cat_2', data, 'matrix')

@nottest
class Test_axes_insertion(base4test):
    __show__ = True
    def test_axes_insertion(self):
        fig = plt.figure()
        ax = fig.add_subplot(2, 2, 1)
        facet_plot('cat_2 + cat_1', self.data, ax=ax, kind='mosaic')
        ax = fig.add_subplot(2, 2, 2)
        facet_plot('cat_1', self.data, ax=ax, kind='counter')
        ax = fig.add_subplot(2, 2, 3)
        facet_plot('sin ~ lin', self.data, 'lines', ax=ax)
        ax = fig.add_subplot(2, 2, 4)
        facet_plot('float_1 ~ cat_1 + float_2', self.data, ax=ax,
                   jitter=0.2, kind='scatter')
        fig.tight_layout()
        fig.canvas.set_window_title('mixed facet_plots')
        #this should give error
        #facet_plot('cat_2 ~ cat_1 | int_1', data, ax=ax)


@nottest
class Test_create_dataframe(base4test):
    __show__ = True
    def test_database_1(self):
        float_1 = plt.randn(100)
        facet_plot('float_1', {'float_1': float_1})

    def test_database_2(self):
        float_1 = plt.randn(100)
        float_1[0] = np.nan
        facet_plot('float_1', {'float_1': float_1})

    def test_database_3(self):
        float_1 = plt.randn(100)
        float_1[0] = np.nan
        facet_plot('float_1', {'float_1': float_1}, subset=(float_1 > 0))

    def test_database_4(self):
        float_1 = plt.randn(100)
        facet_plot('float_1')

    def test_database_5(self):
        float_1 = plt.randn(100)
        facet_plot('I(float_1**2)')

    def test_database_6(self):
        float_1 = plt.randn(100)
        facet_plot('float_1', subset=(float_1 > 0))

    def test_database_7(self):
        float_1 = plt.randn(100)
        facet_plot('np.log(float_1) + float_1')


#this test will trigger some fails...
@nottest
class TestSpecialNames(base4test):
    def test_names_unicode_1(self):
        fig = facet_plot(u'float_1 ~ Q(u"àèéòù")', self.data, kind='scatter')
        assert fig.axes[0].get_xlabel() == u'àèéòù'

    def test_names_unicode_2(self):
        fig = facet_plot(u'float_1*Q(u"àèéòù") ~ float_2', self.data, kind='scatter')
        plt.show()

    def test_names_unicode_3(self):
        data = {'float_1': plt.randn(20),
                'float_2': plt.randn(20),
                'àèéòù': plt.randn(20)}
        fig = facet_plot('float_1*Q("àèéòù") ~ float_2', data, kind='scatter')
        plt.show()

    def test_names_unicode_4(self):
        data = {'float_1': plt.randn(20),
                'float_2': plt.randn(20),
                'dose μ/ml': plt.randn(20)}
        fig = facet_plot('float_1*Q("dose μ/ml") ~ float_2', data, kind='scatter')
        plt.show()

    def test_names_invalid_1(self):
        fig = facet_plot(u'float_1 ~ Q("x.1")', self.data, kind='scatter')
        assert fig.axes[0].get_xlabel() == 'x.1'

    def test_names_invalid_2(self):
        fig = facet_plot(u'float_1 ~ Q("x 1")', self.data, kind='scatter')
        assert fig.axes[0].get_xlabel() == 'x 1'

    def test_names_invalid_3(self):
        fig = facet_plot(u'float_1*Q("x 1") ~ float_2', self.data, kind='scatter')
        #assert fig.axes[0].get_xlabel() == 'x 1'

    def test_names_invalid_4(self):
        fig = facet_plot(u'float_1 ~ Q("x%&$1")', self.data, kind='scatter')
        assert fig.axes[0].get_xlabel() == 'x%&$1'


@nottest
class Test_ellipse(base4test):
    __show__ = True
    def test_ellipse_1(self):
        facet_plot('float_1 ~ float_2 | cat_1', data, 'ellipse')
    def test_ellipse_2(self):
        facet_plot('float_1 ~ cat_2 | cat_1', data, 'ellipse')
    def test_ellipse_3(self):
        facet_plot('float_3 ~ float_1 | cat_1', data, 'ellipse')

@nottest
class Test_hexbin(base4test):
    __show__ = True
    def test_hexbin_1(self):
        facet_plot('float_1 ~ float_2 | cat_1', data, 'hexbin')
    def test_hexbin_2(self):
        facet_plot('float_1 ~ cat_2 | cat_1', data, 'hexbin', cmap='winter')
    def test_hexbin_3(self):
        facet_plot('float_3 ~ float_1 | cat_1', data, 'hexbin', gridsize=10)

@nottest
class Test_counter(base4test):
    __show__ = True
    def test_counter_1(self):
        facet_plot('cat_1', data, 'counter')
    def test_counter_2(self):
        facet_plot('cat_1 | cat_2', data, 'counter')
    def test_counter_3(self):
        facet_plot('cat_1 | cat_1', data, 'counter')
    def test_counter_4(self):
        facet_plot('int_1', data, 'counter')
    def test_counter_5(self):
        facet_plot('int_1 | cat_1', data, 'counter')
    def test_counter_6(self):
        facet_plot('int_2 | cat_1', data, 'counter')
    def test_counter_7(self):
        facet_plot('cat_2 | cat_1', data, 'counter')
    def test_counter_categorical_1(self):
        facet_plot('int_4', data, 'counter')
    def test_counter_categorical_2(self):
        facet_plot('int_4', data, 'counter', as_categorical=1)
    def test_counter_categorical_3(self):
        facet_plot('int_4 | cat_3', data, 'counter')
    def test_counter_categorical_4(self):
        facet_plot('int_4 | cat_3', data, 'counter', as_categorical=1)
    def test_counter_categorical_5(self):
        facet_plot('I(int_4**2)', data, 'counter')
    def test_counter_categorical_6(self):
        facet_plot('I(int_4**2)', data, 'counter', as_categorical=1)
    def test_labeling_simple_1(self):
        # the data is plit in two groups, I should have
        # the labels as normal integers
        fig = facet_plot('int_4', self.data, 'counter')
        plt.draw()
        ax = fig.axes[0]
        labels = [t.get_text().replace(u'\u2212', '-')
                  for t in ax.get_xticklabels() ]
        positions = [float(t) for t in ax.get_xticks() ]
        assert positions[1:-1] == [ float(t) for t in labels if t]
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

@nottest
class Test_hist(base4test):
    __show__ = True
    def test_hist_simple_1(self):
        fig = facet_plot('float_1', self.data, 'hist')
    def test_hist_simple_2(self):
        fig = facet_plot('float_1 + float_2', self.data, 'hist')
    def test_hist_simple_3(self):
        fig = facet_plot('float_1 + float_2 + int_1', self.data, 'hist')

@nottest
class Test_mosaic(base4test):
    __show__ = True
    def test_hist_simple_1(self):
        fig = facet_plot('cat_1', self.data, 'mosaic')
    def test_hist_simple_2(self):
        fig = facet_plot('cat_1 + cat_2', self.data, 'mosaic')
    def test_hist_simple_3(self):
        fig = facet_plot('cat_1 + cat_2 + cat_3', self.data, 'mosaic')
    def test_hist_simple_4(self):
        fig = facet_plot('cat_1 + cat_2| cat_3', self.data, 'mosaic')


#@nottest
class Test_corr(base4test):
    __show__ = True
    def test_corr_simple_1(self):
        fig = facet_plot('float_1 | cat_1', self.data, kind='corr')
    def test_corr_simple_2(self):
        fig = facet_plot('sin | cat_1', self.data, kind='corr')
    def test_corr_simple_3(self):
        fig = facet_plot('float_1 ~ float_2 | cat_1', self.data, kind='corr')


class Test_psd(base4test):
    __show__ = True
    def test_psd_simple_1(self):
        fig = facet_plot('float_1 | cat_1', self.data, kind='psd')
    def test_psd_simple_2(self):
        fig = facet_plot('sin | cat_1', self.data, kind='psd')
    def test_psd_simple_3(self):
        fig = facet_plot('float_1 ~ float_2 | cat_1', self.data, kind='psd')


@nottest
class Test_stack_faced(base4test):
    __show__ = True
    def test_stack_faced_scatter_1(self):
        fig = facet_plot('float_1 ~ float_2 | cat_3', self.data, kind='scatter')
    def test_stack_faced_scatter_2(self):
        fig = facet_plot('float_1 ~ float_2 | ~ cat_3', self.data, kind='scatter')
    def test_stack_faced_boxplot_1(self):
        fig = facet_plot('int_4 ~ cat_2 | cat_3', self.data, kind='boxplot')
    def test_stack_faced_boxplot_2(self):
        fig = facet_plot('int_4 ~ cat_2 | ~ cat_3', self.data, kind='boxplot')
    def test_stack_faced_boxplot_3(self):
        fig = facet_plot('int_4 ~ cat_2 | cat_1 ~ cat_3', self.data, kind='boxplot')


@nottest
class Test_IQ(base4test):
    __show__ = True
    def test_IQ_1(self):
        fig = facet_plot('I(Q("float_1")**0.5) ~ float_2 | Q("cat_3")', self.data, kind='scatter')
        plt.draw()

    def test_IQ_2(self):
        fig = facet_plot('I(Q("float_1")**0.5) + I(float_2) ~ float_3 | cat_3', self.data, kind='scatter')
        plt.draw()

    def test_IQ_3(self):
        fig = facet_plot('I(Q("float_1") ** 0.5) + I(5*standardize(float_2)) ~ float_3 | I(cat_3)', self.data, kind='scatter')
        plt.draw()


@nottest
class Test_categorical(base4test):
    def test_categorical_1(self):
        facet_plot('float_1 * C(cat_1) + C(cat_2) +1', self.data, kind='_dump')
    def test_categorical_2(self):
        facet_plot('float_1 * C(cat_1) + C(cat_2) +1', self.data, kind='_dump', strict_patsy=True)


@nottest
class Test_strict_patsy(base4test):
    __show__ = False
    def test_strict_patsy_1(self):
        facet_plot('float_1', self.data, kind='_dump', strict_patsy=False)
    def test_strict_patsy_2(self):
        facet_plot('float_1', self.data, kind='_dump', strict_patsy=True)
    def test_strict_patsy_3(self):
        facet_plot('float_1 + 0 | cat_1', self.data, kind='_dump', strict_patsy=False)
    def test_strict_patsy_4(self):
        facet_plot('float_1 + 0 | cat_1', self.data, kind='_dump', strict_patsy=True)
    def test_strict_patsy_5(self):
        facet_plot('float_1 * C(int_2)', self.data, kind='_dump', strict_patsy=False)
    def test_strict_patsy_6(self):
        facet_plot('float_1 * C(int_2)', self.data, kind='_dump', strict_patsy=True)

@nottest
class Test_OLS(base4test):
    __show__ = True
    def test_ols_1(self):
        facet_plot('I(int_3 + float_3) ~ float_1 | cat_3', self.data, kind='ols', strict_patsy=False)

@nottest
class Test_mangled(base4test):
    __show__ = True
    def test_ols_1(self):
        facet_plot('float_3 ~ cat_1', self.data, kind='boxplot')
    def test_ols_2(self):
        facet_plot('float_3 ~ Q("cat_1")', self.data, kind='boxplot')


if __name__ == "__main__":
    run_module_suite()
