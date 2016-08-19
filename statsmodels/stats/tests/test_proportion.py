# -*- coding: utf-8 -*-
"""

Created on Fri Mar 01 14:56:56 2013

Author: Josef Perktold
"""
import warnings

import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_array_less,
                           assert_raises, assert_allclose)

from statsmodels.stats.proportion import (proportion_confint,
                                          multinomial_proportions_confint)
import statsmodels.stats.proportion as smprop
from statsmodels.tools.sm_exceptions import HypothesisTestWarning


class Holder(object):
    pass



def test_confint_proportion():
    from .results.results_proportion import res_binom, res_binom_methods
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
            ci = proportion_confint(count, nobs, alpha=0.05, method=method)

            assert_almost_equal(ci, [res_low, res_upp], decimal=6,
                                err_msg=repr(case) + method)

def test_samplesize_confidenceinterval_prop():
    #consistency test for samplesize to achieve confidence_interval
    nobs = 20
    ci = smprop.proportion_confint(12, nobs, alpha=0.05, method='normal')
    res = smprop.samplesize_confint_proportion(12./nobs, (ci[1] - ci[0]) / 2)
    assert_almost_equal(res, nobs, decimal=13)

def test_proportion_effect_size():
    # example from blog
    es = smprop.proportion_effectsize(0.5, 0.4)
    assert_almost_equal(es, 0.2013579207903309, decimal=13)

def test_confint_multinomial_proportions():
    from .results.results_multinomial_proportions import res_multinomial

    for ((method, description), values) in res_multinomial.items():
        cis = multinomial_proportions_confint(values.proportions, 0.05,
                                              method=method)
        assert_almost_equal(
            values.cis, cis, decimal=values.precision,
            err_msg='"%s" method, %s' % (method, description))

def test_multinomial_proportions_errors():
    # Out-of-bounds values for alpha raise a ValueError
    for alpha in [-.1, 0, 1, 1.1]:
        assert_raises(ValueError, multinomial_proportions_confint,
                      [5] * 50, alpha=alpha)

    assert_raises(ValueError, multinomial_proportions_confint,
                  np.arange(50) - 1)
    # Any unknown method is reported.
    for method in ['unknown_method', 'sisok_method', 'unknown-glaz']:
        assert_raises(NotImplementedError, multinomial_proportions_confint,
                      [5] * 50, method=method)

def  test_confint_multinomial_proportions_zeros():
    # test when a count is zero or close to zero
    # values from R MultinomialCI
    ci01 = np.array([
     0.09364718, 0.1898413,
     0.00000000, 0.0483581,
     0.13667426, 0.2328684,
     0.10124019, 0.1974343,
     0.10883321, 0.2050273,
     0.17210833, 0.2683024,
     0.09870919, 0.1949033]).reshape(-1,2)

    ci0 = np.array([
    0.09620253, 0.19238867,
    0.00000000, 0.05061652,
    0.13924051, 0.23542664,
    0.10379747, 0.19998360,
    0.11139241, 0.20757854,
    0.17468354, 0.27086968,
    0.10126582, 0.19745196]).reshape(-1,2)

    # the shifts are the differences between "LOWER(SG)"  "UPPER(SG)" and
    # "LOWER(C+1)" "UPPER(C+1)" in verbose printout
    # ci01_shift = np.array([0.002531008, -0.002515122])  # not needed
    ci0_shift = np.array([0.002531642, 0.002515247])

    p = [56, 0.1, 73, 59, 62, 87, 58]
    ci_01 = smprop.multinomial_proportions_confint(p, 0.05,
                                                   method='sison_glaz')
    p = [56, 0, 73, 59, 62, 87, 58]
    ci_0 = smprop.multinomial_proportions_confint(p, 0.05,
                                                  method='sison_glaz')

    assert_allclose(ci_01, ci01, atol=1e-5)
    assert_allclose(ci_0, np.maximum(ci0 - ci0_shift, 0), atol=1e-5)
    assert_allclose(ci_01, ci_0, atol=5e-4)


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
                                  multitest_method=None)
        assert_almost_equal(ppt.pvals_raw, self.res_ppt_pvals_raw)
        ppt = smprop.proportions_chisquare_allpairs(self.n_success, self.nobs,
                                  multitest_method='h')
        assert_almost_equal(ppt.pval_corrected(), self.res_ppt_pvals_holm)

        pptd = smprop.proportions_chisquare_pairscontrol(self.n_success,
                                  self.nobs, multitest_method='hommel')
        assert_almost_equal(pptd.pvals_raw, ppt.pvals_raw[:len(self.nobs) - 1],
                            decimal=13)


    def test_number_pairs_1493(self):
        ppt = smprop.proportions_chisquare_allpairs(self.n_success[:3],
                                                    self.nobs[:3],
                                                    multitest_method=None)

        assert_equal(len(ppt.pvals_raw), 3)
        idx = [0, 1, 3]
        assert_almost_equal(ppt.pvals_raw, self.res_ppt_pvals_raw[idx])


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

    # GH 2969
    def test_default_values(self):
        count = np.array([5, 12])
        nobs = np.array([83, 99])
        stat, pval = smprop.proportions_ztest(count, nobs, value=None)
        assert_almost_equal(stat, -1.4078304151258787)
        assert_almost_equal(pval, 0.15918129181156992)

    # GH 2779
    def test_scalar(self):
        count = 5
        nobs = 83
        value = 0.05
        stat, pval = smprop.proportions_ztest(count, nobs, value=value)
        assert_almost_equal(stat, 0.392126026314)
        assert_almost_equal(pval, 0.694965098115)

        assert_raises(ValueError, smprop.proportions_ztest, count, nobs, value=None)


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
                ('two-sided', binom_test_2sided)]

    for alt, res0 in alltests:
        # only p-value is returned
        res = smprop.binom_test(51, 235, prop=1. / 6, alternative=alt)
        #assert_almost_equal(res[0], res0.statistic)
        assert_almost_equal(res, res0.p_value, decimal=13)

    # R binom_test returns Copper-Pearson confint
    ci_2s = smprop.proportion_confint(51, 235, alpha=0.05, method='beta')
    ci_low, ci_upp = smprop.proportion_confint(51, 235, alpha=0.1,
                                               method='beta')
    assert_almost_equal(ci_2s, binom_test_2sided.conf_int, decimal=13)
    assert_almost_equal(ci_upp, binom_test_less.conf_int[1], decimal=13)
    assert_almost_equal(ci_low, binom_test_greater.conf_int[0], decimal=13)


def test_binom_rejection_interval():
    # consistency check with binom_test
    # some code duplication but limit checks are different
    alpha = 0.05
    nobs = 200
    prop = 12./20
    alternative='smaller'
    ci_low, ci_upp = smprop.binom_test_reject_interval(prop, nobs, alpha=alpha,
                                                       alternative=alternative)
    assert_equal(ci_upp, nobs)
    pval = smprop.binom_test(ci_low, nobs, prop=prop,
                                  alternative=alternative)
    assert_array_less(pval, alpha)
    pval = smprop.binom_test(ci_low + 1, nobs, prop=prop,
                                  alternative=alternative)
    assert_array_less(alpha, pval)

    alternative='larger'
    ci_low, ci_upp = smprop.binom_test_reject_interval(prop, nobs, alpha=alpha,
                                                       alternative=alternative)
    assert_equal(ci_low, 0)
    pval = smprop.binom_test(ci_upp, nobs, prop=prop,
                                  alternative=alternative)
    assert_array_less(pval, alpha)
    pval = smprop.binom_test(ci_upp - 1, nobs, prop=prop,
                                  alternative=alternative)
    assert_array_less(alpha, pval)

    alternative='two-sided'
    ci_low, ci_upp = smprop.binom_test_reject_interval(prop, nobs, alpha=alpha,
                                                       alternative=alternative)
    pval = smprop.binom_test(ci_upp, nobs, prop=prop,
                                  alternative=alternative)
    assert_array_less(pval, alpha)
    pval = smprop.binom_test(ci_upp - 1, nobs, prop=prop,
                                  alternative=alternative)
    assert_array_less(alpha, pval)
    pval = smprop.binom_test(ci_upp, nobs, prop=prop,
                                  alternative=alternative)
    assert_array_less(pval, alpha)

    pval = smprop.binom_test(ci_upp - 1, nobs, prop=prop,
                                  alternative=alternative)
    assert_array_less(alpha, pval)



def test_binom_tost():
    # consistency check with two different implementation,
    # proportion_confint is tested against R
    # no reference case from other package available
    ci = smprop.proportion_confint(10, 20, method='beta', alpha=0.1)
    bt = smprop.binom_tost(10, 20, *ci)
    assert_almost_equal(bt, [0.05] * 3, decimal=12)

    ci = smprop.proportion_confint(5, 20, method='beta', alpha=0.1)
    bt = smprop.binom_tost(5, 20, *ci)
    assert_almost_equal(bt, [0.05] * 3, decimal=12)

    # vectorized, TODO: observed proportion = 0 returns nan
    ci = smprop.proportion_confint(np.arange(1, 20), 20, method='beta',
                                   alpha=0.05)
    bt = smprop.binom_tost(np.arange(1, 20), 20, *ci)
    bt = np.asarray(bt)
    assert_almost_equal(bt, 0.025 * np.ones(bt.shape), decimal=12)

def test_power_binom_tost():
    # comparison numbers from PASS manual
    p_alt = 0.6 + np.linspace(0, 0.09, 10)
    power = smprop.power_binom_tost(0.5, 0.7, 500, p_alt=p_alt, alpha=0.05)
    res_power = np.array([0.9965,  0.9940,  0.9815,  0.9482,  0.8783,  0.7583,
                          0.5914,  0.4041,  0.2352,  0.1139])
    assert_almost_equal(power, res_power, decimal=4)

    rej_int = smprop.binom_tost_reject_interval(0.5, 0.7, 500)
    res_rej_int = (269, 332)
    assert_equal(rej_int, res_rej_int)

    # TODO: actual alpha=0.0489  for all p_alt above

    # another case
    nobs = np.arange(20, 210, 20)
    power = smprop.power_binom_tost(0.4, 0.6, nobs, p_alt=0.5, alpha=0.05)
    res_power = np.array([ 0.,  0.,  0.,  0.0889,  0.2356,  0.3517,  0.4457,
                           0.6154,  0.6674,  0.7708])
    # TODO: I currently don't impose power>=0, i.e np.maximum(power, 0)
    assert_almost_equal(np.maximum(power, 0), res_power, decimal=4)

def test_power_ztost_prop():
    power = smprop.power_ztost_prop(0.1, 0.9, 10, p_alt=0.6, alpha=0.05,
                         discrete=True, dist='binom')[0]
    assert_almost_equal(power, 0.8204, decimal=4) # PASS example

    with warnings.catch_warnings():  # python >= 2.6
        warnings.simplefilter("ignore", HypothesisTestWarning)
        power = smprop.power_ztost_prop(0.4, 0.6, np.arange(20, 210, 20),
                                        p_alt=0.5, alpha=0.05, discrete=False,
                                        dist='binom')[0]

        res_power = np.array([ 0., 0., 0., 0.0889, 0.2356, 0.4770, 0.5530,
            0.6154,  0.7365,  0.7708])
        # TODO: I currently don't impose power>=0, i.e np.maximum(power, 0)
        assert_almost_equal(np.maximum(power, 0), res_power, decimal=4)

        # with critval_continuity correction
        power = smprop.power_ztost_prop(0.4, 0.6, np.arange(20, 210, 20),
                                        p_alt=0.5, alpha=0.05, discrete=False,
                                        dist='binom', variance_prop=None,
                                        continuity=2, critval_continuity=1)[0]

        res_power = np.array([0., 0., 0., 0.0889, 0.2356, 0.3517, 0.4457,
                              0.6154, 0.6674, 0.7708])
        # TODO: I currently don't impose power>=0, i.e np.maximum(power, 0)
        assert_almost_equal(np.maximum(power, 0), res_power, decimal=4)

        power = smprop.power_ztost_prop(0.4, 0.6, np.arange(20, 210, 20),
                                        p_alt=0.5, alpha=0.05, discrete=False,
                                        dist='binom', variance_prop=0.5,
                                        critval_continuity=1)[0]

        res_power = np.array([0., 0., 0., 0.0889, 0.2356, 0.3517, 0.4457,
                              0.6154, 0.6674, 0.7112])
        # TODO: I currently don't impose power>=0, i.e np.maximum(power, 0)
        assert_almost_equal(np.maximum(power, 0), res_power, decimal=4)

def test_ztost():
    xfair = np.repeat([1,0], [228, 762-228])

    # comparing to SAS last output at
    # http://support.sas.com/documentation/cdl/en/procstat/63104/HTML/default/viewer.htm#procstat_freq_sect028.htm
    # confidence interval for tost
    # generic ztost is moved to weightstats
    from statsmodels.stats.weightstats import zconfint, ztost
    ci01 = zconfint(xfair, alpha=0.1, ddof=0)
    assert_almost_equal(ci01,  [0.2719, 0.3265], 4)
    res = ztost(xfair, 0.18, 0.38, ddof=0)

    assert_almost_equal(res[1][0], 7.1865, 4)
    assert_almost_equal(res[2][0], -4.8701, 4)
    assert_array_less(res[0], 0.0001)


def test_power_ztost_prop_norm():
    # regression test for normal distribution
    # from a rough comparison, the results and variations look reasonable
    power = smprop.power_ztost_prop(0.4, 0.6, np.arange(20, 210, 20),
                                    p_alt=0.5, alpha=0.05, discrete=False,
                                    dist='norm', variance_prop=0.5,
                                    continuity=0, critval_continuity=0)[0]

    res_power = np.array([0., 0., 0., 0.11450013,  0.27752006, 0.41495922,
                          0.52944621,  0.62382638,  0.70092914,  0.76341806])
    # TODO: I currently don't impose power>=0, i.e np.maximum(power, 0)
    assert_almost_equal(np.maximum(power, 0), res_power, decimal=4)

    # regression test for normal distribution
    power = smprop.power_ztost_prop(0.4, 0.6, np.arange(20, 210, 20),
                                    p_alt=0.5, alpha=0.05, discrete=False,
                                    dist='norm', variance_prop=0.5,
                                    continuity=1, critval_continuity=0)[0]

    res_power = np.array([0., 0., 0.02667562,  0.20189793,  0.35099606,
                          0.47608598,  0.57981118,  0.66496683,  0.73427591,
                          0.79026127])
    # TODO: I currently don't impose power>=0, i.e np.maximum(power, 0)
    assert_almost_equal(np.maximum(power, 0), res_power, decimal=4)

    # regression test for normal distribution
    power = smprop.power_ztost_prop(0.4, 0.6, np.arange(20, 210, 20),
                                    p_alt=0.5, alpha=0.05, discrete=True,
                                    dist='norm', variance_prop=0.5,
                                    continuity=1, critval_continuity=0)[0]

    res_power = np.array([0., 0., 0., 0.08902071,  0.23582284, 0.35192313,
                          0.55312718,  0.61549537,  0.66743625,  0.77066806])
    # TODO: I currently don't impose power>=0, i.e np.maximum(power, 0)
    assert_almost_equal(np.maximum(power, 0), res_power, decimal=4)

    # regression test for normal distribution
    power = smprop.power_ztost_prop(0.4, 0.6, np.arange(20, 210, 20),
                                    p_alt=0.5, alpha=0.05, discrete=True,
                                    dist='norm', variance_prop=0.5,
                                    continuity=1, critval_continuity=1)[0]

    res_power = np.array([0., 0., 0., 0.08902071,  0.23582284, 0.35192313,
                          0.44588687,  0.61549537,  0.66743625,  0.71115563])
    # TODO: I currently don't impose power>=0, i.e np.maximum(power, 0)
    assert_almost_equal(np.maximum(power, 0), res_power, decimal=4)

    # regression test for normal distribution
    power = smprop.power_ztost_prop(0.4, 0.6, np.arange(20, 210, 20),
                                    p_alt=0.5, alpha=0.05, discrete=True,
                                    dist='norm', variance_prop=None,
                                    continuity=0, critval_continuity=0)[0]

    res_power = np.array([0., 0., 0., 0., 0.15851942, 0.41611758,
                          0.5010377 ,  0.5708047 ,  0.70328247,  0.74210096])
    # TODO: I currently don't impose power>=0, i.e np.maximum(power, 0)
    assert_almost_equal(np.maximum(power, 0), res_power, decimal=4)


def test_proportion_ztests():
    # currently only consistency test with proportions chisquare
    # Note: alternative handling is generic

    res1 = smprop.proportions_ztest(15, 20., value=0.5, prop_var=0.5)
    res2 = smprop.proportions_chisquare(15, 20., value=0.5)
    assert_almost_equal(res1[1], res2[1], decimal=13)

    res1 = smprop.proportions_ztest(np.asarray([15, 10]), np.asarray([20., 20]),
                                 value=0, prop_var=None)
    res2 = smprop.proportions_chisquare(np.asarray([15, 10]), np.asarray([20., 20]))
    # test only p-value
    assert_almost_equal(res1[1], res2[1], decimal=13)

if __name__ == '__main__':
    test_confint_proportion()
