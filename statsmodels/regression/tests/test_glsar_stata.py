# -*- coding: utf-8 -*-
"""Testing GLSAR against STATA

Created on Wed May 30 09:25:24 2012

Author: Josef Perktold
"""

import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose

from statsmodels.regression.linear_model import GLSAR
from statsmodels.tools.tools import add_constant
from statsmodels.datasets import macrodata


class CheckStataResultsMixin(object):

    def test_params_table(self):
        res, results = self.res, self.results
        assert_almost_equal(res.params, results.params, 3)
        assert_almost_equal(res.bse, results.bse, 3)
        #assert_almost_equal(res.tvalues, results.tvalues, 3) 0.0003
        assert_allclose(res.tvalues, results.tvalues, atol=0, rtol=0.004)
        assert_allclose(res.pvalues, results.pvalues, atol=1e-7, rtol=0.004)

class CheckStataResultsPMixin(CheckStataResultsMixin):

    def test_predicted(self):
        res, results = self.res, self.results
        assert_allclose(res.fittedvalues, results.fittedvalues, rtol=0.002)
        predicted = res.predict(res.model.exog) #should be equal
        assert_allclose(predicted, results.fittedvalues, rtol=0.0016)
        #not yet
        #assert_almost_equal(res.fittedvalues_se, results.fittedvalues_se, 4)

class TestGLSARCorc(CheckStataResultsPMixin):

    @classmethod
    def setup_class(self):
        d2 = macrodata.load().data
        g_gdp = 400*np.diff(np.log(d2['realgdp']))
        g_inv = 400*np.diff(np.log(d2['realinv']))
        exogg = add_constant(np.c_[g_gdp, d2['realint'][:-1]], prepend=False)

        mod1 = GLSAR(g_inv, exogg, 1)
        self.res = mod1.iterative_fit(5)

        from results.macro_gr_corc_stata import results
        self.results = results

    def test_rho(self):
        assert_almost_equal(self.res.model.rho, self.results.rho, 3)  # pylint: disable-msg=E1101


if __name__=="__main__":
    import nose
    nose.runmodule(argv=[__file__,'-vvs','-x',#'--pdb', '--pdb-failure'
                         ],
                   exit=False)
