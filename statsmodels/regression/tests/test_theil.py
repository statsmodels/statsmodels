# -*- coding: utf-8 -*-
"""
Created on Mon May 05 17:29:56 2014

Author: Josef Perktold
"""

import os
import numpy as np
from scipy import stats

from numpy.testing import assert_allclose, assert_equal, assert_warns

from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.tools.tools import add_constant
from statsmodels.tools.sm_exceptions import InvalidTestWarning

from statsmodels.sandbox.regression.penalized import TheilGLS


class TestTheilTextile(object):

    @classmethod
    def setup_class(cls):

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(cur_dir, "results", "theil_textile_predict.csv")
        cls.res_predict = np.recfromtxt(filepath, delimiter=",")


        names = "year	lconsump	lincome	lprice".split()

        data = np.array('''\
        1923	1.99651	1.98543	2.00432
        1924	1.99564	1.99167	2.00043
        1925	2	2	2
        1926	2.04766	2.02078	1.95713
        1927	2.08707	2.02078	1.93702
        1928	2.07041	2.03941	1.95279
        1929	2.08314	2.04454	1.95713
        1930	2.13354	2.05038	1.91803
        1931	2.18808	2.03862	1.84572
        1932	2.18639	2.02243	1.81558
        1933	2.20003	2.00732	1.78746
        1934	2.14799	1.97955	1.79588
        1935	2.13418	1.98408	1.80346
        1936	2.22531	1.98945	1.72099
        1937	2.18837	2.0103	1.77597
        1938	2.17319	2.00689	1.77452
        1939	2.2188	2.0162	1.78746'''.split(), float).reshape(-1, 4)


        endog = data[:, 1]
        # constant at the end to match Stata
        exog = np.column_stack((data[:, 2:], np.ones(endog.shape[0])))

        #prior(lprice -0.7 0.15 lincome 1 0.15) cov(lprice lincome -0.01)
        r_matrix = np.array([[1, 0, 0], [0, 1, 0]])
        r_mean = [1, -0.7]

        cov_r = np.array([[0.15**2, -0.01], [-0.01, 0.15**2]])
        mod = TheilGLS(endog, exog, r_matrix, q_matrix=r_mean, sigma_prior=cov_r)
        cls.res1 = mod.fit(cov_type='data-prior')
        from .results import results_theil_textile as resmodule
        cls.res2 = resmodule.results_theil_textile


    def test_basic(self):
        pt = self.res2.params_table[:,:6].T
        params2, bse2, tvalues2, pvalues2, ci_low, ci_upp = pt
        assert_allclose(self.res1.params, params2, rtol=2e-6)

        #TODO tgmixed seems to use scale from initial OLS, not from final res
        # np.sqrt(res.scale / res_ols.scale)
        # see below mse_resid which is equal to scale
        corr_fact = 0.9836026210570028
        assert_allclose(self.res1.bse / corr_fact, bse2, rtol=2e-6)
        assert_allclose(self.res1.tvalues  * corr_fact, tvalues2, rtol=2e-6)
        # pvalues are very small
        assert_allclose(self.res1.pvalues, pvalues2, atol=2e-6)
        assert_allclose(self.res1.pvalues, pvalues2, rtol=0.7)
        ci = self.res1.conf_int()
        # not scale corrected
        assert_allclose(ci[:,0], ci_low, rtol=0.01)
        assert_allclose(ci[:,1], ci_upp, rtol=0.01)
        assert_allclose(self.res1.rsquared, self.res2.r2, rtol=2e-6)

        # Note: tgmixed is using k_exog for df_resid
        corr_fact = self.res1.df_resid / self.res2.df_r
        assert_allclose(np.sqrt(self.res1.mse_resid * corr_fact), self.res2.rmse, rtol=2e-6)


    def test_other(self):
        tc = self.res1.test_compatibility()
        assert_allclose(np.squeeze(tc[0]), self.res2.compat, rtol=2e-6)
        assert_allclose(np.squeeze(tc[1]), self.res2.pvalue, rtol=2e-6)

        frac = self.res1.share_data()
        assert_allclose(frac, self.res2.frac_sample, rtol=2e-6)


    def test_no_penalization(self):
        res_ols = OLS(self.res1.model.endog, self.res1.model.exog).fit()
        res_theil = self.res1.model.fit(lambd=0, cov_type='data-prior')
        assert_allclose(res_theil.params, res_ols.params, rtol=1e-11)
        assert_allclose(res_theil.bse, res_ols.bse, rtol=1e-11)


    def test_smoke(self):
        self.res1.summary()
