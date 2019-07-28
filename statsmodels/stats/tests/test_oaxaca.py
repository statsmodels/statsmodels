# -*- coding: utf-8 -*-
#from statsmodels.formula.api import ols
# STATA adds a constant no matter if you want to or not, so I cannot test for having no intercept. This also would make no sense for Oaxaca.
# All of these stata_results are from using the oaxaca command in STATA.

import numpy as np
from statsmodels.datasets.ccard.data import load, load_pandas
from statsmodels.tools.tools import add_constant
from statsmodels.stats.oaxaca import Oaxaca 

df = load()
pandas_df = load_pandas()
endog, exog = df.endog, add_constant(df.exog, prepend=False)
pd_endog, pd_exog = pandas_df.endog, add_constant(pandas_df.exog, prepend=False)


class TestOaxaca(object):
    @classmethod
    def setup_class(cls):
        cls.model = Oaxaca(endog, exog, 3)

    def test_results(self):
        stata_results = np.array([158.7504, 321.7482, 75.45371, -238.4515])
        stata_results_pooled = np.array([158.7504, 130.8095, 27.94091])
        char, coef, inter, gap = self.model.three_fold()
        unexp, exp, gap = self.model.two_fold()
        np.testing.assert_almost_equal(gap, stata_results[0], 3)
        np.testing.assert_almost_equal(char, stata_results[1], 3)
        np.testing.assert_almost_equal(coef, stata_results[2], 3)
        np.testing.assert_almost_equal(inter, stata_results[3], 3)

        np.testing.assert_almost_equal(gap, stata_results_pooled[0], 3)
        np.testing.assert_almost_equal(exp, stata_results_pooled[1], 3)
        np.testing.assert_almost_equal(unexp, stata_results_pooled[2], 3)


class TestOaxacaNoSwap(object):
    @classmethod
    def setup_class(cls):
        cls.model = Oaxaca(endog, exog, 3, swap=False)

    def test_results(self):
        stata_results = np.array([-158.7504, -83.29674, 162.9978, -238.4515])
        stata_results_pooled = np.array([-158.7504, -130.8095, -27.94091])
        char, coef, inter, gap = self.model.three_fold()
        unexp, exp, gap = self.model.two_fold()
        np.testing.assert_almost_equal(gap, stata_results[0], 3)
        np.testing.assert_almost_equal(char, stata_results[1], 3)
        np.testing.assert_almost_equal(coef, stata_results[2], 3)
        np.testing.assert_almost_equal(inter, stata_results[3], 3)

        np.testing.assert_almost_equal(gap, stata_results_pooled[0], 3)
        np.testing.assert_almost_equal(exp, stata_results_pooled[1], 3)
        np.testing.assert_almost_equal(unexp, stata_results_pooled[2], 3)


class TestOaxacaPandas(object):
    @classmethod
    def setup_class(cls):
        cls.model = Oaxaca(pd_endog, pd_exog, 'OWNRENT')

    def test_results(self):
        stata_results = np.array([158.7504, 321.7482, 75.45371, -238.4515])
        stata_results_pooled = np.array([158.7504, 130.8095, 27.94091])
        char, coef, inter, gap = self.model.three_fold()
        unexp, exp, gap = self.model.two_fold()
        np.testing.assert_almost_equal(gap, stata_results[0], 3)
        np.testing.assert_almost_equal(char, stata_results[1], 3)
        np.testing.assert_almost_equal(coef, stata_results[2], 3)
        np.testing.assert_almost_equal(inter, stata_results[3], 3)

        np.testing.assert_almost_equal(gap, stata_results_pooled[0], 3)
        np.testing.assert_almost_equal(exp, stata_results_pooled[1], 3)
        np.testing.assert_almost_equal(unexp, stata_results_pooled[2], 3)


class TestOaxacaPandasNoSwap(object):
    @classmethod
    def setup_class(cls):
        cls.model = Oaxaca(pd_endog, pd_exog, 'OWNRENT', swap=False)

    def test_results(self):
        stata_results = np.array([-158.7504, -83.29674, 162.9978, -238.4515])
        stata_results_pooled = np.array([-158.7504, -130.8095, -27.94091])
        char, coef, inter, gap = self.model.three_fold()
        unexp, exp, gap = self.model.two_fold()
        np.testing.assert_almost_equal(gap, stata_results[0], 3)
        np.testing.assert_almost_equal(char, stata_results[1], 3)
        np.testing.assert_almost_equal(coef, stata_results[2], 3)
        np.testing.assert_almost_equal(inter, stata_results[3], 3)

        np.testing.assert_almost_equal(gap, stata_results_pooled[0], 3)
        np.testing.assert_almost_equal(exp, stata_results_pooled[1], 3)
        np.testing.assert_almost_equal(unexp, stata_results_pooled[2], 3)


class TestOaxacaNoConstPassed(object):
    @classmethod
    def setup_class(cls):
        cls.model = Oaxaca(df.endog, df.exog, 3)

    def test_results(self):
        stata_results = np.array([158.7504, 321.7482, 75.45371, -238.4515])
        stata_results_pooled = np.array([158.7504, 130.8095, 27.94091])
        char, coef, inter, gap = self.model.three_fold()
        unexp, exp, gap = self.model.two_fold()
        np.testing.assert_almost_equal(gap, stata_results[0], 3)
        np.testing.assert_almost_equal(char, stata_results[1], 3)
        np.testing.assert_almost_equal(coef, stata_results[2], 3)
        np.testing.assert_almost_equal(inter, stata_results[3], 3)

        np.testing.assert_almost_equal(gap, stata_results_pooled[0], 3)
        np.testing.assert_almost_equal(exp, stata_results_pooled[1], 3)
        np.testing.assert_almost_equal(unexp, stata_results_pooled[2], 3)


class TestOaxacaNoSwapNoConstPassed(object):
    @classmethod
    def setup_class(cls):
        cls.model = Oaxaca(pandas_df.endog, pandas_df.exog,
                           3, hasconst=False, swap=False)

    def test_results(self):
        stata_results = np.array([-158.7504, -83.29674, 162.9978, -238.4515])
        stata_results_pooled = np.array([-158.7504, -130.8095, -27.94091])
        char, coef, inter, gap = self.model.three_fold()
        unexp, exp, gap = self.model.two_fold()
        np.testing.assert_almost_equal(gap, stata_results[0], 3)
        np.testing.assert_almost_equal(char, stata_results[1], 3)
        np.testing.assert_almost_equal(coef, stata_results[2], 3)
        np.testing.assert_almost_equal(inter, stata_results[3], 3)

        np.testing.assert_almost_equal(gap, stata_results_pooled[0], 3)
        np.testing.assert_almost_equal(exp, stata_results_pooled[1], 3)
        np.testing.assert_almost_equal(unexp, stata_results_pooled[2], 3)


class TestOaxacaPandasNoConstPassed(object):
    @classmethod
    def setup_class(cls):
        cls.model = Oaxaca(pandas_df.endog, pandas_df.exog, 'OWNRENT')

    def test_results(self):
        stata_results = np.array([158.7504, 321.7482, 75.45371, -238.4515])
        stata_results_pooled = np.array([158.7504, 130.8095, 27.94091])
        char, coef, inter, gap = self.model.three_fold()
        unexp, exp, gap = self.model.two_fold()
        np.testing.assert_almost_equal(gap, stata_results[0], 3)
        np.testing.assert_almost_equal(char, stata_results[1], 3)
        np.testing.assert_almost_equal(coef, stata_results[2], 3)
        np.testing.assert_almost_equal(inter, stata_results[3], 3)

        np.testing.assert_almost_equal(gap, stata_results_pooled[0], 3)
        np.testing.assert_almost_equal(exp, stata_results_pooled[1], 3)
        np.testing.assert_almost_equal(unexp, stata_results_pooled[2], 3)


class TestOaxacaPandasNoSwapNoConstPassed(object):
    @classmethod
    def setup_class(cls):
        cls.model = Oaxaca(pandas_df.endog, pandas_df.exog,
                           'OWNRENT', swap=False)

    def test_results(self):
        stata_results = np.array([-158.7504, -83.29674, 162.9978, -238.4515])
        stata_results_pooled = np.array([-158.7504, -130.8095, -27.94091])
        char, coef, inter, gap = self.model.three_fold()
        unexp, exp, gap = self.model.two_fold()
        np.testing.assert_almost_equal(gap, stata_results[0], 3)
        np.testing.assert_almost_equal(char, stata_results[1], 3)
        np.testing.assert_almost_equal(coef, stata_results[2], 3)
        np.testing.assert_almost_equal(inter, stata_results[3], 3)

        np.testing.assert_almost_equal(gap, stata_results_pooled[0], 3)
        np.testing.assert_almost_equal(exp, stata_results_pooled[1], 3)
        np.testing.assert_almost_equal(unexp, stata_results_pooled[2], 3)
