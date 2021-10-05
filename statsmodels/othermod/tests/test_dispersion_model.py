# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 08:37:28 2021

Author: Josef Perktod
License: BSD-3
"""

import warnings

import numpy as np
from numpy.testing import assert_allclose
# import pytest

from scipy import stats

from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.genmod.families import links
from statsmodels.tools.tools import add_constant

from statsmodels.regression.linear_model import WLS
from statsmodels.miscmodels.tmodel import TLinearModel
import statsmodels.othermod.dispersion_model as odm
from statsmodels.tools.sm_exceptions import (
    DomainWarning,
    )

from statsmodels.miscmodels.tests.test_tmodel import mm
from .results import results_multilink as results_ml


class CheckCompare():

    def test_basic(self):
        res1 = self.res1
        res2 = self.res2
        k_mean = self.k_mean
        atol = getattr(self, "atol", 1e-5)
        rtol = getattr(self, "rtol", 0.01)
        # location
        assert_allclose(res1.params[:k_mean], res2.params[:k_mean],
                        atol=1e-4)  # 3e-5)
        assert_allclose(res1.bse[:k_mean], res2.bse[:k_mean],
                        rtol=rtol, atol=atol)
        assert_allclose(res1.tvalues[:k_mean], res2.tvalues[:k_mean],
                        rtol=rtol, atol=atol)
        assert_allclose(res1.pvalues[:k_mean], res2.pvalues[:k_mean],
                        rtol=rtol, atol=atol)


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

        _, s_ = mod1._predict_locscale(res1.params)

        mod2 = WLS(endog, exog, weights=1 / s_)
        res2 = mod2.fit(cov_type='fixed scale', cov_kwds=dict(scale=1))

        cls.res1 = res1
        cls.res2 = res2
        cls.k_mean = exog.shape[1]


class TestGammaHetGLM(CheckCompare):
    # TOCO: currently not a good comparison

    @classmethod
    def setup_class(cls):
        endog = mm.m_marietta.copy()
        # winsorize one outlier
        # endog[endog > 0.25] = 0.25
        exog = add_constant(mm.CRSP)
        # mod3 = TLinearModel(endog, exog)
        # res3 = mod3.fit(method='bfgs', disp=False)

        # dropping outlier observation to get comparable params
        y = (endog - endog.mean())**2
        mod1 = odm.GammaHet(y, exog, exog_scale=exog)
        res1 = mod1.fit(start_params=[0.5] * 2 + [1, 2], method='bfgs',
                        disp=False,
                        )

        _, s_ = mod1._predict_locscale(res1.params)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DomainWarning)
            mod2 = GLM(y, exog, family=families.Gamma(link=links.identity()),
                       var_weights=1 / s_)
        res2 = mod2.fit(start_params=res1.params[:2], method="bfgs", scale=1.)

        cls.res1 = res1
        cls.res2 = res2
        cls.k_mean = exog.shape[1]
        cls.atol = 0.03  # TODO: find version with better agreement
        cls.rtol = 0.2


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

        # df1, m1, s1 = mod1._predict_locscale(res1.params)

        mod2 = TLinearModel(endog, exog)
        res2 = mod2.fit()

        cls.res1 = res1
        cls.res2 = res2
        cls.k_mean = exog.shape[1]


# belowe are mainly regression tests for development


class TestJohnsonSU1():

    @classmethod
    def setup_class(cls):
        np.random.seed(987127429)
        nobs = 500
        args = (0, 1, 0, 1)
        y = stats.johnsonsu.rvs(*args, size=nobs)
        x_const = np.ones(nobs)[:, None]
        ex_trend = np.column_stack((np.ones(nobs), np.linspace(-1, 1, nobs)))

        mod = odm.Johnsonsu(y, ex_trend, exog_scale=x_const,
                            exog_extras=[None, None], k_extra=2)
        cls.res1 = mod.fit(start_params=[0.1, 0, 0, 0, 1], method="bfgs")
        cls.res2 = results_ml.results_johnsonsu_2

    def test_basic(self):
        res1 = self.res1
        res2 = self.res2

        # smoke for summary
        res1.summary()

        attr_list = [
            'aic', 'bic', 'bse', 'df_model', 'df_modelwc', 'df_resid',
            'k_constant', 'llf', 'nobs', 'params',
            'pvalues', 'scale', 'tvalues',
            ]
        # removed from list: 'normalized_cov_params',

        for attr in attr_list:
            attr1 = getattr(res1, attr)
            attr2 = getattr(res2, attr)
            assert_allclose(attr1, attr2, rtol=1e-5, atol=1e-10)
            # Note: pvalues fails without atol, smallest pvalue 9.7-32

        # lower precision across operating systems or linalg libraries
        cov1 = res1.normalized_cov_params
        cov2 = res2.normalized_cov_params
        assert_allclose(cov1, cov2, rtol=0.001, atol=1e-6)


class TestJohnsonSU2(TestJohnsonSU1):

    @classmethod
    def setup_class(cls):
        np.random.seed(987127429)
        nobs = 500
        args = (0, 1, 0, 1)
        y = stats.johnsonsu.rvs(*args, size=nobs)
        x_const = np.ones(nobs)[:, None]
        ex_trend = np.column_stack((np.ones(nobs), np.linspace(-1, 1, nobs)))

        mod = odm.Johnsonsu(y, ex_trend, exog_scale=x_const,
                            exog_extras=[x_const, x_const], k_extra=2)
        cls.res1 = mod.fit(start_params=[0.1, 0, 0, 0, 1], method="bfgs")
        cls.res2 = results_ml.results_johnsonsu_2


class TestJohnsonSU3(TestJohnsonSU1):

    @classmethod
    def setup_class(cls):
        np.random.seed(987127429)
        nobs = 500
        args = (0, 1, 0, 1)
        y = stats.johnsonsu.rvs(*args, size=nobs)
        x_const = np.ones(nobs)[:, None]
        ex_trend = np.column_stack((np.ones(nobs), np.linspace(-1, 1, nobs)))

        mod = odm.Johnsonsu(y, ex_trend, exog_scale=x_const,
                            k_extra=2)
        cls.res1 = mod.fit(start_params=[0.1, 0, 0, 0, 1], method="bfgs")
        cls.res2 = results_ml.results_johnsonsu_2


class TestJohnsonSUe():
    # with 2-column exog for second shape parameter

    @classmethod
    def setup_class(cls):
        np.random.seed(987127429)
        nobs = 500
        args = (0, 1, 0, 1)
        y = stats.johnsonsu.rvs(*args, size=nobs)
        x_const = np.ones(nobs)[:, None]
        ex_trend = np.column_stack((np.ones(nobs), np.linspace(-1, 1, nobs)))

        mod = odm.Johnsonsu(y, ex_trend, exog_scale=x_const,
                            exog_extras=[ex_trend, x_const], k_extra=2)
        cls.res1 = mod.fit(start_params=[0.1, 0, 0, 0, 0, 1], method="bfgs")
        # cls.res2 = results_ml.results_johnsonsu_2

    def test_develop(self):
        res1 = self.res1

        assert len(res1.params) == 6

        # smoke for summary
        res1.summary()
        p_names = ['const', 'x1', 'scale-1', 'a0-0', 'a0-1', 'a1-0']
        assert res1.model.exog_names == p_names


class TestJohnsonSUe2(TestJohnsonSUe):
    # with link_extras for second shape parameter

    @classmethod
    def setup_class(cls):
        np.random.seed(987127429)
        nobs = 500
        args = (0, 1, 0, 1)
        y = stats.johnsonsu.rvs(*args, size=nobs)
        x_const = np.ones(nobs)[:, None]
        ex_trend = np.column_stack((np.ones(nobs), np.linspace(-1, 1, nobs)))

        mod = odm.Johnsonsu(y, ex_trend, exog_scale=x_const,
                            exog_extras=[ex_trend, x_const],
                            link_extras=[families.links.identity(),
                                         families.links.Log()],
                            k_extra=2)
        cls.res1 = mod.fit(start_params=[0.1, 0, 0, 0, 1, 2], method="bfgs")

    def tes_links(self):
        res1 = self.res1
        mod1 = res1.model
        assert isinstance(mod1.link, families.links.identity)
        assert isinstance(mod1.link_scale, families.links.Log)
        assert isinstance(mod1.link_extras[0], families.links.identity)
        assert isinstance(mod1.link_extras[1], families.links.Log)
