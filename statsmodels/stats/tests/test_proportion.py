# -*- coding: utf-8 -*-
"""

Created on Fri Mar 01 14:56:56 2013

Author: Josef Perktold
"""

import numpy as np
from numpy.testing import assert_almost_equal

from statsmodels.stats.proportion import confint_proportion
import statsmodels.stats.proportion as smprop

class Holder(object):
    pass

def test_confint_proportion():
    from results.results_proportion import res_binom, res_binom_methods
    methods = {'agresti_coull' : 'agresti-coull',
               'normal' : 'asymptotic',
               'beta' : 'exact',
               'wilson' : 'wilson',
               'jeffrey' : 'bayes'
               }

    for case in res_binom:
        count, nobs = case
        for method in methods:
            idx = res_binom_methods.index(methods[method])
            res_low = res_binom[case].ci_low[idx]
            res_upp = res_binom[case].ci_upp[idx]
            if np.isnan(res_low) or np.isnan(res_upp):
                continue
            ci = confint_proportion(count, nobs, alpha=0.05, method=method)

            assert_almost_equal(ci, [res_low, res_upp], decimal=6,
                                err_msg=repr(case) + method)

class CheckProportionMixin(object):
    def test_proptest(self):
        # equality of k-samples
        pt = smprop.proportions_chisquare(self.n_success, self.nobs, value=None)
        assert_almost_equal(pt[0], self.res_prop_test.statistic, decimal=13)
        assert_almost_equal(pt[1], self.res_prop_test.p_value, decimal=13)

        # several against value
        pt = smprop.proportions_chisquare(self.n_success, self.nobs,
                                    value=self.res_prop_test_val.null_value[0])
        assert_almost_equal(pt[0], self.res_prop_test_val.statistic, decimal=13)
        assert_almost_equal(pt[1], self.res_prop_test_val.p_value, decimal=13)

        # one proportion against value
        pt = smprop.proportions_chisquare(self.n_success[0], self.nobs[0],
                                    value=self.res_prop_test_1.null_value)
        assert_almost_equal(pt[0], self.res_prop_test_1.statistic, decimal=13)
        assert_almost_equal(pt[1], self.res_prop_test_1.p_value, decimal=13)

    def test_pairwiseproptest(self):
        ppt = smprop.proportions_chisquare_allpairs(self.n_success, self.nobs,
                                  value=None, multitest_method=None)
        assert_almost_equal(ppt.pvals_raw, self.res_ppt_pvals_raw)
        ppt = smprop.proportions_chisquare_allpairs(self.n_success, self.nobs,
                                  value=None, multitest_method='h')
        assert_almost_equal(ppt.pval_corrected(), self.res_ppt_pvals_holm)

class TestProportion(CheckProportionMixin):
    def setup(self):
        self.n_success = np.array([ 73,  90, 114,  75])
        self.nobs = np.array([ 86,  93, 136,  82])

        self.res_ppt_pvals_raw = np.array([
                 0.00533824886503131, 0.8327574849753566, 0.1880573726722516,
                 0.002026764254350234, 0.1309487516334318, 0.1076118730631731
                ])
        self.res_ppt_pvals_holm = np.array([
                 0.02669124432515654, 0.8327574849753566, 0.4304474922526926,
                 0.0121605855261014, 0.4304474922526926, 0.4304474922526926
                ])

        res_prop_test = Holder()
        res_prop_test.statistic = 11.11938768628861
        res_prop_test.parameter = 3
        res_prop_test.p_value = 0.011097511366581344
        res_prop_test.estimate = np.array([
             0.848837209302326, 0.967741935483871, 0.838235294117647,
             0.9146341463414634
            ]).reshape(4,1, order='F')
        res_prop_test.null_value = '''NULL'''
        res_prop_test.conf_int = '''NULL'''
        res_prop_test.alternative = 'two.sided'
        res_prop_test.method = '4-sample test for equality of proportions ' + \
                               'without continuity correction'
        res_prop_test.data_name = 'smokers2 out of patients'
        self.res_prop_test = res_prop_test

        #> pt = prop.test(smokers2, patients, p=rep(c(0.9), 4), correct=FALSE)
        #> cat_items(pt, "res_prop_test_val.")
        res_prop_test_val = Holder()
        res_prop_test_val.statistic = np.array([
             13.20305530710751
            ]).reshape(1,1, order='F')
        res_prop_test_val.parameter = np.array([
             4
            ]).reshape(1,1, order='F')
        res_prop_test_val.p_value = 0.010325090041836
        res_prop_test_val.estimate = np.array([
             0.848837209302326, 0.967741935483871, 0.838235294117647,
             0.9146341463414634
            ]).reshape(4,1, order='F')
        res_prop_test_val.null_value = np.array([
             0.9, 0.9, 0.9, 0.9
            ]).reshape(4,1, order='F')
        res_prop_test_val.conf_int = '''NULL'''
        res_prop_test_val.alternative = 'two.sided'
        res_prop_test_val.method = '4-sample test for given proportions without continuity correction'
        res_prop_test_val.data_name = 'smokers2 out of patients, null probabilities rep(c(0.9), 4)'
        self.res_prop_test_val = res_prop_test_val

        #> pt = prop.test(smokers2[1], patients[1], p=0.9, correct=FALSE)
        #> cat_items(pt, "res_prop_test_1.")
        res_prop_test_1 = Holder()
        res_prop_test_1.statistic = 2.501291989664086
        res_prop_test_1.parameter = 1
        res_prop_test_1.p_value = 0.113752943640092
        res_prop_test_1.estimate = 0.848837209302326
        res_prop_test_1.null_value = 0.9
        res_prop_test_1.conf_int = np.array([0.758364348004061,
                                             0.9094787701686766])
        res_prop_test_1.alternative = 'two.sided'
        res_prop_test_1.method = '1-sample proportions test without continuity correction'
        res_prop_test_1.data_name = 'smokers2[1] out of patients[1], null probability 0.9'
        self.res_prop_test_1 = res_prop_test_1


def test_binom_test():
    #> bt = binom.test(51,235,(1/6),alternative="less")
    #> cat_items(bt, "binom_test_less.")
    binom_test_less = Holder()
    binom_test_less.statistic = 51
    binom_test_less.parameter = 235
    binom_test_less.p_value = 0.982022657605858
    binom_test_less.conf_int = [0, 0.2659460862574313]
    binom_test_less.estimate = 0.2170212765957447
    binom_test_less.null_value = 1. / 6
    binom_test_less.alternative = 'less'
    binom_test_less.method = 'Exact binomial test'
    binom_test_less.data_name = '51 and 235'

    #> bt = binom.test(51,235,(1/6),alternative="greater")
    #> cat_items(bt, "binom_test_greater.")
    binom_test_greater = Holder()
    binom_test_greater.statistic = 51
    binom_test_greater.parameter = 235
    binom_test_greater.p_value = 0.02654424571169085
    binom_test_greater.conf_int = [0.1735252778065201, 1]
    binom_test_greater.estimate = 0.2170212765957447
    binom_test_greater.null_value = 1. / 6
    binom_test_greater.alternative = 'greater'
    binom_test_greater.method = 'Exact binomial test'
    binom_test_greater.data_name = '51 and 235'

    #> bt = binom.test(51,235,(1/6),alternative="t")
    #> cat_items(bt, "binom_test_2sided.")
    binom_test_2sided = Holder()
    binom_test_2sided.statistic = 51
    binom_test_2sided.parameter = 235
    binom_test_2sided.p_value = 0.0437479701823997
    binom_test_2sided.conf_int = [0.1660633298083073, 0.2752683640289254]
    binom_test_2sided.estimate = 0.2170212765957447
    binom_test_2sided.null_value = 1. / 6
    binom_test_2sided.alternative = 'two.sided'
    binom_test_2sided.method = 'Exact binomial test'
    binom_test_2sided.data_name = '51 and 235'

    alltests = [('larger', binom_test_greater),
                ('smaller', binom_test_less),
                ('2-sided', binom_test_2sided)]

    for alt, res0 in alltests:
        # only p-value is returned
        res = smprop.binom_test_stat(51, 235, p=1. / 6, alternative=alt)
        #assert_almost_equal(res[0], res0.statistic)
        assert_almost_equal(res, res0.p_value)



if __name__ == '__main__':
    test_confint_proportion()
