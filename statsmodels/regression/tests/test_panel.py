"""
Tests for panel models

Stata:

insheet using Grunfeld.csv, clear double
encode firm, generate(firmn)
xtset firmn year
xtreg invest value capital, be

"""

import os
import statsmodels.api as sm
import numpy as np
import numpy.testing as npt
import statsmodels
from statsmodels.regression.panel import PanelLM
from statsmodels.datasets import grunfeld
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
        npt.assert_almost_equal(self.res1.params, self.res2.params, DECIMAL_4)

    def test_dof(self):
        npt.assert_equal(self.res1.df_resid, self.res2.df_resid)
        if self.res1.model.method != 'pooling':
            npt.assert_equal(self.res1.df_model, self.res2.df_model)

    def test_conf_int(self):
        if self.res1.model.method == "pooling":
            # we report t values but R results are norm, so check against that
            from scipy.stats import norm
            q = norm.ppf(1 - .025)
            params, bse = self.res1.params, self.res1.bse
            lower = params - q * bse
            upper = params + q * bse
            conf_int = np.asarray(zip(lower,upper))
            npt.assert_almost_equal(conf_int, self.res2.conf_int, 4)
        else:
            npt.assert_almost_equal(self.res1.conf_int(), self.res2.conf_int,
                                    DECIMAL_4)

    def test_tstat(self):
        npt.assert_almost_equal(self.res1.tvalues, self.res2.tvalues,
                                DECIMAL_4)

    def pvalues(self):
        npt.assert_almost_equal(self.res1.pvalues, self.res2.pvalues,
                                DECIMAL_4)

    def test_normalized_cov_params(self):
        pass

    def test_bse(self):
        npt.assert_almost_equal(self.res1.bse, self.res2.bse, DECIMAL_4)

    def test_predict(self):
        npt.assert_almost_equal(self.res1.predict(),
                            self.res2.fittedvalues, DECIMAL_4)

    def test_nobs(self):
        npt.assert_almost_equal(self.res1.nobs,
                            self.res2.nobs, DECIMAL_4)

    def test_rsquared(self):
        npt.assert_almost_equal(self.res1.rsquared,
                            self.res2.rsquared, DECIMAL_4)

    def test_resid(self):
        npt.assert_almost_equal(self.res1.resid,
                            self.res2.residuals, DECIMAL_4)


class FixedEffectsMixin(object):
    def test_sum_of_squares(self):
        npt.assert_almost_equal(self.res1.ssr, self.res2.ssr)
        npt.assert_almost_equal(self.res1.ess, self.res2.ess)
        #centered and uncentered tss?
        npt.assert_almost_equal(self.res1.centered_tss,
                                self.res1.centered_tss, 4)

    def test_fvalue(self):
        npt.assert_almost_equal(self.res1.fvalue,
                            self.res2.fvalue, DECIMAL_4)

    def test_f_pvalue(self):
        npt.assert_almost_equal(self.res1.f_pvalue,
                            self.res2.f_pvalue, DECIMAL_4)

    def test_rsquared_adj(self):
        npt.assert_almost_equal(self.res1.rsquared_adj,
                                self.res2.rsquared_adj, DECIMAL_4)

class TestWithin(CheckModelResults, FixedEffectsMixin):
    @classmethod
    def setupClass(cls):
        from results_panel import within
        data = grunfeld.load_pandas().data
        data.firm = data.firm.apply(lambda x: x.lower())
        data = data.set_index(['firm', 'year'])
        data = data.sort()
        y, X = dmatrices("invest ~ value + capital - 1", data=data,
                return_type='dataframe')
        cls.res1 = PanelLM(y, X, method='within').fit(disp=0)
        res2 = within
        cls.res2 = res2

    def test_sigma(self):
        npt.assert_almost_equal(self.res1.std_dev_groups,
                                self.res2.std_dev_groups, 4)
        npt.assert_almost_equal(self.res1.std_dev_resid,
                                self.res2.std_dev_resid, 4)
        npt.assert_almost_equal(self.res1.std_dev_overall,
                                self.res2.std_dev_overall, 4)
        npt.assert_almost_equal(self.res1.rho, self.res2.rho, 4)

    def test_constant(self):
        npt.assert_almost_equal(self.res1.constant, self.res2.constant, 4)

    def test_corr(self):
        npt.assert_almost_equal(self.res1.corr, self.res2.corr, 2)

    def test_other_resids(self):
        npt.assert_almost_equal(self.res1.resid_groups,
                                self.res2.resid_groups, 4)
        npt.assert_almost_equal(self.res1.resid_combined,
                                self.res2.resid_combined, 4)
        npt.assert_almost_equal(self.res1.resid_overall,
                                self.res2.resid_overall, 4)

    def test_fittedvalues(self):
        npt.assert_almost_equal(self.res1.fittedvalues,
                                self.res2.fittedvalues_stata, 3)

    def test_rsquared_other(self):
        npt.assert_almost_equal(self.res1.rsquared_overall,
                                self.res2.rsquared_overall, 4)
        npt.assert_almost_equal(self.res1.rsquared_between,
                                self.res2.rsquared_between, 4)
        npt.assert_almost_equal(self.res1.rsquared_within,
                                self.res2.rsquared_within, 4)

    def test_loglike(self):
        npt.assert_almost_equal(self.res1.llf, self.res2.llf, 4)

class TestBetween(CheckModelResults, FixedEffectsMixin):
    @classmethod
    def setupClass(cls):
        from results_panel import between
        data = grunfeld.load_pandas().data
        data.firm = data.firm.apply(lambda x: x.lower())
        data = data.set_index(['firm', 'year'])
        data = data.sort()
        y, X = dmatrices("invest ~ value + capital", data=data,
                return_type='dataframe')
        cls.res1 = PanelLM(y, X, method='between').fit(disp=0)
        res2 = between
        cls.res2 = res2

    def test_rsquared_other(self):
        npt.assert_almost_equal(self.res1.rsquared_overall,
                                self.res2.rsquared_overall, 4)
        npt.assert_almost_equal(self.res1.rsquared_between,
                                self.res2.rsquared_between, 4)
        npt.assert_almost_equal(self.res1.rsquared_within,
                                self.res2.rsquared_within, 4)

    def test_rmse(self):
        npt.assert_almost_equal(self.res1.rmse, self.res2.rmse, 4)

    def test_fittedvalues(self):
        npt.assert_almost_equal(self.res1.fittedvalues,
                                self.res2.fittedvalues_stata, 4)

    def test_other_resids(self):
        npt.assert_almost_equal(self.res1.resid_groups,
                                self.res2.resid_groups, 4)
        npt.assert_almost_equal(self.res1.resid_combined,
                                self.res2.resid_combined, 4)
        npt.assert_almost_equal(self.res1.resid_overall,
                                self.res2.resid_overall, 4)

    def test_loglike(self):
        npt.assert_almost_equal(self.res1.llf, self.res2.llf, 4)

class TestRandomSWAR(CheckModelResults):
    @classmethod
    def setupClass(cls):
        from results_panel import swar1w
        data = grunfeld.load_pandas().data
        data.firm = data.firm.apply(lambda x: x.lower())
        data = data.set_index(['firm', 'year'])
        data = data.sort()
        y, X = dmatrices("invest ~ value + capital", data=data,
                return_type='dataframe')
        cls.res1 = PanelLM(y, X, method='swar').fit(disp=0)
        res2 = swar1w
        cls.res2 = res2

    def test_chi2(self):
        npt.assert_almost_equal(self.res1.chi2, self.res2.fvalue*2)

    def test_chi2_pvalue(self):
        npt.assert_almost_equal(self.res1.chi2_pvalue, self.res2.chi2_pvalue)

    def test_other_resids(self):
        npt.assert_almost_equal(self.res1.resid_groups,
                                self.res2.resid_groups, 4)
        npt.assert_almost_equal(self.res1.resid_combined,
                                self.res2.resid_combined, 4)
        npt.assert_almost_equal(self.res1.resid_overall,
                                self.res2.resid_overall, 4)

    def test_fittedvalues(self):
        npt.assert_almost_equal(self.res1.fittedvalues,
                                self.res2.fittedvalues_stata, 3)

    def test_wresid(self):
        npt.assert_almost_equal(self.res1.wresid,
                                self.res2.wresid, 4)


class TestPooling(CheckModelResults, FixedEffectsMixin):
    @classmethod
    def setupClass(cls):
        from results_panel import pooling
        data = grunfeld.load_pandas().data
        data.firm = data.firm.apply(lambda x: x.lower())
        data = data.set_index(['firm', 'year'])
        data = data.sort()
        y, X = dmatrices("invest ~ value + capital", data=data,
                return_type='dataframe')
        cls.res1 = PanelLM(y, X, method='pooling').fit(disp=0)
        res2 = pooling
        cls.res2 = res2


class XestTwoWay(CheckModelResults, FixedEffectsMixin):
    pass

class XestMLE(CheckModelResults):
    @classmethod
    def setupClass(cls):
        #from results_panel import mle_results
        data = grunfeld.load_pandas().data
        data.firm = data.firm.str.lower()
        data = data.set_index(['firm', 'year'])
        data = data.sort()
        y = data['invest']
        data['const'] = 1
        X = data[['const', 'value', 'capital']]
        cls.res1 = PanelLM(y, X, method='mle').fit(disp=0)
        #cls.res2 = mle_results

#TODO:
#        check constants
#        check different data (put these tests in panel/base/tests
#        check different variances
#        test prediction

#if __name__ == "__main__":
    #import nose
    #nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb'],
            #exit=False)
