"""
Tests for panel models

Status:
    - rsquared_adj: ???
    - conf_int: SM is right R's plm is wrong (no df correction in plm)
    - wrong f_pvalue (see especially between)
    - fvalue: why do I need to divide by 2 except for within? 2-tailed?

Stata:

insheet using Grunfeld.csv, clear
encode firm, generate(firmn)
xtset firm year
xtreg invest value capital, be

"""

import os
import statsmodels.api as sm
import numpy as np
from numpy.testing import *
import statsmodels
from statsmodels.regression.panel import PanelLM
from patsy import dmatrices

DECIMAL_14 = 14
DECIMAL_10 = 10
DECIMAL_9 = 9
DECIMAL_4 = 4
DECIMAL_3 = 3
DECIMAL_2 = 2
DECIMAL_1 = 1
DECIMAL_0 = 0

class CheckModelResults(object):
    """
    res2 should be the test results from RModelWrap
    or the results as defined in model_results_data
    """
    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL_4)

    def test_dof(self):
                assert_equal(self.res1.df_resid, self.res2.df_resid)

#    def test_conf_int(self):
        #assert_almost_equal(self.res1.conf_int(), self.res2.conf_int, DECIMAL_3)

    def test_tstat(self):
        assert_almost_equal(self.res1.tvalues, self.res2.tvalues, DECIMAL_4)

    def pvalues(self):
        assert_almost_equal(self.res1.pvalues, self.res2.pvalues, DECIMAL_4)

    def test_normalized_cov_params(self):
        pass

    def test_bse(self):
        assert_almost_equal(self.res1.bse, self.res2.bse, DECIMAL_4)

    def test_predict(self):
        assert_almost_equal(self.res1.predict(),
                            self.res2.fittedvalues, DECIMAL_4)

    def test_nobs(self):
        assert_almost_equal(self.res1.nobs,
                            self.res2.nobs, DECIMAL_4)

    def test_rsquared(self):
        assert_almost_equal(self.res1.rsquared,
                            self.res2.rsquared, DECIMAL_4)

    #def test_rsquared_adj(self):
        #assert_almost_equal(self.res1.rsquared_adj,
                            #self.res2.rsquared_adj, DECIMAL_4)

    def test_fvalue(self):
        assert_almost_equal(self.res1.fvalue,
                            self.res2.fvalue, DECIMAL_4)

    def test_f_pvalue(self):
        assert_almost_equal(self.res1.f_pvalue,
                            self.res2.f_pvalue, DECIMAL_4)

    def test_resid(self):
        assert_almost_equal(self.res1.resid,
                            self.res2.residuals, DECIMAL_4)


class TestWithin(CheckModelResults):
    import statsmodels
    @classmethod
    def setupClass(cls):
        from results_panel import within
        data = statsmodels.datasets.grunfeld.load_pandas().data
        data.firm = data.firm.apply(lambda x: x.lower())
        data = data.set_index(['firm', 'year'])
        data = data.sort()
        y, X = dmatrices("invest ~ value + capital - 1", data=data,
                return_type='dataframe')
        cls.res1 = PanelLM(y, X, method='within').fit(disp=0)
        res2 = within
        cls.res2 = res2

class TestBetween(CheckModelResults):
    import statsmodels
    @classmethod
    def setupClass(cls):
        from results_panel import between
        data = statsmodels.datasets.grunfeld.load_pandas().data
        data.firm = data.firm.apply(lambda x: x.lower())
        data = data.set_index(['firm', 'year'])
        data = data.sort()
        y, X = dmatrices("invest ~ value + capital", data=data,
                return_type='dataframe')
        cls.res1 = PanelLM(y, X, method='between').fit(disp=0)
        res2 = between
        cls.res2 = res2

class TestRandom(CheckModelResults):
    import statsmodels
    @classmethod
    def setupClass(cls):
        from results_panel import swar1w
        data = statsmodels.datasets.grunfeld.load_pandas().data
        data.firm = data.firm.apply(lambda x: x.lower())
        data = data.set_index(['firm', 'year'])
        data = data.sort()
        y, X = dmatrices("invest ~ value + capital", data=data,
                return_type='dataframe')
        cls.res1 = PanelLM(y, X, method='swar').fit(disp=0)
        res2 = swar1w
        cls.res2 = res2

class TestPooling(CheckModelResults):
    import statsmodels
    @classmethod
    def setupClass(cls):
        from results_panel import pooling
        data = statsmodels.datasets.grunfeld.load_pandas().data
        data.firm = data.firm.apply(lambda x: x.lower())
        data = data.set_index(['firm', 'year'])
        data = data.sort()
        y, X = dmatrices("invest ~ value + capital", data=data,
                return_type='dataframe')
        cls.res1 = PanelLM(y, X, method='pooling').fit(disp=0)
        res2 = pooling
        cls.res2 = res2


#if __name__ == "__main__":
    #import nose
    #nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb'],
            #exit=False)
