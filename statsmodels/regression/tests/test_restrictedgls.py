# -*- coding: utf-8 -*-
"""
Created on Sat May 17 07:46:12 2014

Author: Josef Perktold
License: BSD-3

"""

import os
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd

from statsmodels.regression.rls import RLS
from statsmodels.tools.tools import add_constant

cur_dir = os.path.abspath(os.path.dirname(__file__))
fn = os.path.join(cur_dir, 'rlsdata.txt')
data = pd.read_csv(fn, delimiter="\t")
data["ys"] = (data['Y'] - data['Y'].mean()) / data['Y'].std(ddof=1)


class CheckRLS(object):

    def test_consistency(self):
        res1 = self.res1
        q1 = np.dot(self.R, res1.params)
        assert_allclose(q1, self.q, rtol=1e-12, atol=1e-13)

    def test_smoke(self):
        self.res1.summary()
        self.res1.summary2()


class CheckRLSVerified(CheckRLS):

    def test_basic(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1.params, res2.params, rtol=1e-12, atol=1e-13)
        assert_allclose(res1.bse, res2.bse, rtol=1e-12, atol=1e-13)

        assert_allclose(res1.fvalue, res2.F, rtol=1e-12, atol=1e-13)
        assert_equal(res1.df_model, res2.df_m)
        assert_equal(res1.df_resid, res2.df_r)
        assert_allclose(res1.scale, res2.rmse**2, rtol=1e-12, atol=1e-13)
        assert_allclose(res1.llf, res2.ll, rtol=1e-12, atol=1e-13)


class TestRestrictedGLS1(CheckRLS):

    @classmethod
    def setup_class(cls):
        dta = data
        design = pd.concat((dta['Y'], dta['Y']**2,
                            dta[['NE', 'NC', 'W', 'S']]),
                           axis=1)
        design = add_constant(design, prepend=True)

        cls.R = [0, 0, 0, 1, 1, 1, 1]
        cls.q = [0]
        rls_mod = RLS(dta['G'], design, constr=cls.R)
        cls.res1 = rls_mod.fit()


class TestRestrictedGLS2(CheckRLS):

    @classmethod
    def setup_class(cls):
        dta = data
        # standardizing the polynomial variable avoids ill conditioned X'X
        design = pd.concat((dta['ys'], dta['ys']**2,
                            dta[['NE', 'NC', 'W', 'S']]),
                           axis=1)
        design = add_constant(design, prepend=False)  # use Stata convention

        cls.R = [0, 0, 0, 1, 1, 1, 1]
        cls.q = [1]
        rls_mod = RLS(dta['G'], design, constr=cls.R, param=cls.q)
        cls.res1 = rls_mod.fit()

        from .results import results_restrictedls as results
        cls.res2 = results.results_rls1_nonrobust


class TestRestrictedGLS3(CheckRLSVerified):

    @classmethod
    def setup_class(cls):
        dta = data
        # standardizing the polynomial variable avoids ill conditioned X'X
        design = pd.concat((dta['ys'], dta['ys']**2,
                            dta[['NE', 'NC', 'S', 'W']]),  # Note order changed
                           axis=1)
        design = add_constant(design, prepend=False)  # use Stata convention

        cls.R = [0, 0, 1, 1, 1, 1, 0]
        cls.q = [0]
        rls_mod = RLS(dta['G'], design, constr=cls.R)
        cls.res1 = rls_mod.fit()

        from .results import results_restrictedls as results
        cls.res2 = results.results_rls2_nonrobust
