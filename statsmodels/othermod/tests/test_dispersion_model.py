# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 08:37:28 2021

Author: Josef Perktod
License: BSD-3
"""

import numpy as np
from numpy.testing import assert_allclose
# import pytest

from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.genmod.families import links
from statsmodels.tools.tools import add_constant

from statsmodels.regression.linear_model import WLS
from statsmodels.miscmodels.tmodel import TLinearModel
import statsmodels.othermod.dispersion_model as odm

from statsmodels.miscmodels.tests.test_tmodel import mm

class CheckCompare():

    def test_basic(self):
        res1 = self.res1
        res2 = self.res2
        k_mean = self.k_mean
        # location
        assert_allclose(res1.params[:k_mean], res2.params[:k_mean],
                        atol=3e-5)
        assert_allclose(res1.bse[:k_mean], res2.bse[:k_mean],
                        rtol=0.01, atol=1e-5)
        assert_allclose(res1.tvalues[:k_mean], res2.tvalues[:k_mean],
                        rtol=0.01, atol=1e-5)
        assert_allclose(res1.pvalues[:k_mean], res2.pvalues[:k_mean],
                        rtol=0.01, atol=1e-5)


class TestGaussianHetWLS(CheckCompare):

    @classmethod
    def setup_class(cls):
        endog = mm.m_marietta.copy()
        # winsorize one outlier
        endog[endog > 0.25] = 0.25
        exog = add_constant(mm.CRSP)
        # mod3 = TLinearModel(endog, exog)
        # res3 = mod3.fit(method='bfgs', disp=False)

        # dropping outlier observation to get comparable params
        mod1 = odm.GaussianHet(endog, exog, exog_scale=exog)
        res1 = mod1.fit(method='bfgs', disp=False)

        m_, s_ = mod1._predict_locscale(res1.params)

        mod2 = WLS(endog, exog, weights=1 / s_)
        res2 = mod2.fit(cov_type='fixed scale', cov_kwds=dict(scale=1))

        cls.res1 = res1
        cls.res2 = res2
        cls.k_mean = exog.shape[1]


class TestTHet(CheckCompare):

    @classmethod
    def setup_class(cls):
        endog = mm.m_marietta.copy()
        # winsorize one outlier
        endog[endog > 0.25] = 0.25
        exog = add_constant(mm.CRSP)
        # mod3 = TLinearModel(endog, exog)
        # res3 = mod3.fit(method='bfgs', disp=False)

        # dropping outlier observation to get comparable params
        mod1 = odm.TLinearModelHet(endog, exog)
        res1 = mod1.fit(method='bfgs', disp=False)

        df1, m1, s1 = mod1._predict_locscale(res1.params)

        mod2 = TLinearModel(endog, exog)
        res2 = mod2.fit()

        cls.res1 = res1
        cls.res2 = res2
        cls.k_mean = exog.shape[1]
