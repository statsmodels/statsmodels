# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 20:33:02 2018

Author: Josef Perktold
"""

import numpy as np
from numpy.testing import assert_allclose
from statsmodels.regression.linear_model import OLS
from statsmodels.regression.special_linear_model import OLSVectorized


class TestOLSVectorized(object):

    @classmethod
    def setup_class(cls):
        np.random.seed(3)
        sig_e = 0.1

        exog = (np.repeat(np.arange(3), 20)[:,None] ==
                np.arange(3)).astype(float)

        params_true = np.array([[1, 1, 1], [1, 1, 1],
                                [1, 1.5, 2], [1, 1.5, 2]]).T

        y_true = exog.dot(params_true)
        endog = y_true + sig_e * np.random.randn(*y_true.shape)

        cls.res1 = OLSVectorized(endog, exog).fit()
        cls.res2_list = [OLS(endog[:, i], exog).fit() for i in range(4)]

    def test_attributes(self):
        res1 = self.res1
        res2_list = self.res2_list


        attrs = ['params', 'scale', 'bse', 'tvalues', 'pvalues', 'ssr', 'centered_tss',
                 'rsquared', 'llf', 'aic', 'bic', 'fvalue', 'f_pvalue']
        for attr in attrs:
            value = getattr(res1, attr)
            res2_value = np.array([getattr(r, attr) for r in res2_list]).T
            assert_allclose(value, res2_value, rtol=1e-13)

    def test_methods(self):
        res1 = self.res1
        res2_list = self.res2_list

        ci1 = res1.conf_int()
        ci2 = np.array([r.conf_int() for r in res2_list])
        ci2 = np.rollaxis(ci2, 0, 3)
        assert_allclose(ci1, ci2, rtol=1e-13)

        # smoke test
        res1.summary()

    def test_wald(self):
        res1 = self.res1
        res2_list = self.res2_list

        tt1 = res1.t_test('x1=x2')
        tt2_list = [r.t_test('x1=x2') for r in res2_list]
        attrs = ['effect', 'sd', 'tvalue', 'pvalue', 'df_denom', 'statistic']
        for attr in attrs:
            value = getattr(tt1, attr)
            res2_value = np.array([getattr(t, attr) for t in tt2_list]).squeeze()
            assert_allclose(value, res2_value, rtol=1e-12, err_msg=attr)

        ci1 = tt1.conf_int()
        ci2 = np.array([t.conf_int() for t in tt2_list]).squeeze()
        assert_allclose(ci1, ci2, rtol=1e-12)

        # smoke test
        tt1.summary()
