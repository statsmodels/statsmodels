

from numpy.testing import assert_allclose, assert_equal

# we cannot import test_poisson_2indep directly, pytest treats that as test
import statsmodels.stats.rates as smr
from statsmodels.stats.rates import etest_poisson_2indep


def test_twosample_poisson():
    # testing against two examples in Gu et al

    # example 1
    count1, n1, count2, n2 = 60, 51477.5, 30, 54308.7

    s1, pv1 = smr.test_poisson_2indep(count1, n1, count2, n2, method='wald')
    pv1r = 0.000356
    assert_allclose(pv1, pv1r*2, rtol=0, atol=5e-6)
    assert_allclose(s1, 3.384913, atol=0, rtol=5e-6)  # regression test

    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2, method='score')
    pv2r = 0.000316
    assert_allclose(pv2, pv2r*2, rtol=0, atol=5e-6)
    assert_allclose(s2, 3.417402, atol=0, rtol=5e-6)  # regression test

    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2, method='sqrt')
    pv2r = 0.000285
    assert_allclose(pv2, pv2r*2, rtol=0, atol=5e-6)
    assert_allclose(s2, 3.445485, atol=0, rtol=5e-6)  # regression test

    # two-sided
    # example2
    # I don't know why it's only 2.5 decimal agreement, rounding?
    count1, n1, count2, n2 = 41, 28010, 15, 19017
    s1, pv1 = smr.test_poisson_2indep(count1, n1, count2, n2, method='wald',
                                      ratio_null=1.5)
    pv1r = 0.2309
    assert_allclose(pv1, pv1r*2, rtol=0, atol=5e-3)
    assert_allclose(s1, 0.735447, atol=0, rtol=5e-6)  # regression test

    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2, method='score',
                                      ratio_null=1.5)
    pv2r = 0.2398
    assert_allclose(pv2, pv2r*2, rtol=0, atol=5e-3)
    assert_allclose(s2, 0.706631, atol=0, rtol=5e-6)  # regression test

    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2, method='sqrt',
                                      ratio_null=1.5)
    pv2r = 0.2499
    assert_allclose(pv2, pv2r*2, rtol=0, atol=5e-3)
    assert_allclose(s2, 0.674401, atol=0, rtol=5e-6)  # regression test

    # one-sided
    # example 1 onesided
    count1, n1, count2, n2 = 60, 51477.5, 30, 54308.7

    s1, pv1 = smr.test_poisson_2indep(count1, n1, count2, n2, method='wald',
                                      alternative='larger')
    pv1r = 0.000356
    assert_allclose(pv1, pv1r, rtol=0, atol=5e-6)

    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2, method='score',
                                      alternative='larger')
    pv2r = 0.000316
    assert_allclose(pv2, pv2r, rtol=0, atol=5e-6)

    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2, method='sqrt',
                                      alternative='larger')
    pv2r = 0.000285
    assert_allclose(pv2, pv2r, rtol=0, atol=5e-6)

    # 'exact-cond', 'cond-midp'

    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2,
                                      method='exact-cond',
                                      ratio_null=1, alternative='larger')
    pv2r = 0.000428  # typo in Gu et al, switched pvalues between C and M
    assert_allclose(pv2, pv2r, rtol=0, atol=5e-4)

    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2,
                                      method='cond-midp',
                                      ratio_null=1, alternative='larger')
    pv2r = 0.000310
    assert_allclose(pv2, pv2r, rtol=0, atol=5e-4)

    _, pve1 = etest_poisson_2indep(count1, n1, count2, n2,
                                   method='score',
                                   ratio_null=1, alternative='larger')
    pve1r = 0.000298
    assert_allclose(pve1, pve1r, rtol=0, atol=5e-4)

    _, pve1 = etest_poisson_2indep(count1, n1, count2, n2,
                                   method='wald',
                                   ratio_null=1, alternative='larger')
    pve1r = 0.000298
    assert_allclose(pve1, pve1r, rtol=0, atol=5e-4)

    # example2 onesided
    # I don't know why it's only 2.5 decimal agreement, rounding?
    count1, n1, count2, n2 = 41, 28010, 15, 19017
    s1, pv1 = smr.test_poisson_2indep(count1, n1, count2, n2, method='wald',
                                      ratio_null=1.5, alternative='larger')
    pv1r = 0.2309
    assert_allclose(pv1, pv1r, rtol=0, atol=5e-4)

    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2, method='score',
                                      ratio_null=1.5, alternative='larger')
    pv2r = 0.2398
    assert_allclose(pv2, pv2r, rtol=0, atol=5e-4)

    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2, method='sqrt',
                                      ratio_null=1.5, alternative='larger')
    pv2r = 0.2499
    assert_allclose(pv2, pv2r, rtol=0, atol=5e-4)

    # 'exact-cond', 'cond-midp'

    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2,
                                      method='exact-cond',
                                      ratio_null=1.5, alternative='larger')
    pv2r = 0.2913
    assert_allclose(pv2, pv2r, rtol=0, atol=5e-4)

    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2,
                                      method='cond-midp',
                                      ratio_null=1.5, alternative='larger')
    pv2r = 0.2450
    assert_allclose(pv2, pv2r, rtol=0, atol=5e-4)

    _, pve2 = etest_poisson_2indep(count1, n1, count2, n2,
                                   method='score',
                                   ratio_null=1.5, alternative='larger')
    pve2r = 0.2453
    assert_allclose(pve2, pve2r, rtol=0, atol=5e-4)

    _, pve2 = etest_poisson_2indep(count1, n1, count2, n2,
                                   method='wald',
                                   ratio_null=1.5, alternative='larger')
    pve2r = 0.2453
    assert_allclose(pve2, pve2r, rtol=0, atol=5e-4)


def test_twosample_poisson_r():
    # testing against R package `exactci
    from .results.results_rates import res_pexact_cond, res_pexact_cond_midp

    # example 1 from Gu
    count1, n1, count2, n2 = 60, 51477.5, 30, 54308.7

    res2 = res_pexact_cond
    res1 = smr.test_poisson_2indep(count1, n1, count2, n2, method='exact-cond')
    assert_allclose(res1.pvalue, res2.p_value, rtol=1e-13)
    assert_allclose(res1.ratio, res2.estimate, rtol=1e-13)
    assert_equal(res1.ratio_null, res2.null_value)

    res2 = res_pexact_cond_midp
    res1 = smr.test_poisson_2indep(count1, n1, count2, n2, method='cond-midp')
    assert_allclose(res1.pvalue, res2.p_value, rtol=0, atol=5e-6)
    assert_allclose(res1.ratio, res2.estimate, rtol=1e-13)
    assert_equal(res1.ratio_null, res2.null_value)

    # one-sided
    # > pe = poisson.exact(c(60, 30), c(51477.5, 54308.7), r=1.2,
    #                      alternative="less", tsmethod="minlike", midp=TRUE)
    # > pe$p.value
    pv2 = 0.9949053964701466
    rest = smr.test_poisson_2indep(count1, n1, count2, n2, method='cond-midp',
                                   ratio_null=1.2, alternative='smaller')
    assert_allclose(rest.pvalue, pv2, rtol=1e-12)
    # > pe = poisson.exact(c(60, 30), c(51477.5, 54308.7), r=1.2,
    #           alternative="greater", tsmethod="minlike", midp=TRUE)
    # > pe$p.value
    pv2 = 0.005094603529853279
    rest = smr.test_poisson_2indep(count1, n1, count2, n2, method='cond-midp',
                                   ratio_null=1.2, alternative='larger')
    assert_allclose(rest.pvalue, pv2, rtol=1e-12)
    # > pe = poisson.exact(c(60, 30), c(51477.5, 54308.7), r=1.2,
    #           alternative="greater", tsmethod="minlike", midp=FALSE)
    # > pe$p.value
    pv2 = 0.006651774552714537
    rest = smr.test_poisson_2indep(count1, n1, count2, n2, method='exact-cond',
                                   ratio_null=1.2, alternative='larger')
    assert_allclose(rest.pvalue, pv2, rtol=1e-12)
    # > pe = poisson.exact(c(60, 30), c(51477.5, 54308.7), r=1.2,
    #                      alternative="less", tsmethod="minlike", midp=FALSE)
    # > pe$p.value
    pv2 = 0.9964625674930079
    rest = smr.test_poisson_2indep(count1, n1, count2, n2, method='exact-cond',
                                   ratio_null=1.2, alternative='smaller')
    assert_allclose(rest.pvalue, pv2, rtol=1e-12)


def test_tost_poisson():

    count1, n1, count2, n2 = 60, 51477.5, 30, 54308.7
    # # central conf_int from R exactci
    low, upp = 1.339735721772650, 3.388365573616252

    pv, _, _ = smr.tost_poisson_2indep(count1, n1, count2, n2, low, upp,
                                       method="exact-cond")

    assert_allclose(pv, 0.025, rtol=1e-12)
    methods = ['wald', 'score', 'sqrt', 'exact-cond', 'cond-midp']

    # test that we are in the correct range for other methods
    for meth in methods:
        pv, _, _ = smr.tost_poisson_2indep(count1, n1, count2, n2, low, upp,
                                           method=meth)
        assert_allclose(pv, 0.025, atol=0.01)
