

from __future__ import division
import numpy as np

from statsmodels.stats.rates import poisson_twosample, etest_twopoisson
from numpy.testing import assert_allclose


def test_twosample_poisson():
    # testing against two examples in Gu et al

    print('\ntwo-sided')
    # example 1
    count1, n1, count2, n2 = 60, 51477.5, 30, 54308.7

    s1, pv1 = poisson_twosample(count1, n1, count2, n2, method='wald')
    pv1r = 0.000356
    assert_allclose(pv1, pv1r*2, rtol=0, atol=5e-6)
    print('wald', s1, pv1 / 2)   # one sided in the "right" direction

    s2, pv2 = poisson_twosample(count1, n1, count2, n2, method='score')
    pv2r = 0.000316
    assert_allclose(pv2, pv2r*2, rtol=0, atol=5e-6)
    print('score', s2, pv2 / 2)   # one sided in the "right" direction

    s2, pv2 = poisson_twosample(count1, n1, count2, n2, method='sqrt')
    pv2r = 0.000285
    assert_allclose(pv2, pv2r*2, rtol=0, atol=5e-6)
    print('sqrt', s2, pv2 / 2)   # one sided in the "right" direction

    print('\ntwo-sided')
    # example2
    # I don't know why it's only 2.5 decimal agreement, rounding?
    count1, n1, count2, n2 = 41, 28010, 15, 19017
    s1, pv1 = poisson_twosample(count1, n1, count2, n2, method='wald', ratio_null=1.5)
    pv1r = 0.2309
    assert_allclose(pv1, pv1r*2, rtol=0, atol=5e-3)
    print('wald', s1, pv1 / 2)   # one sided in the "right" direction

    s2, pv2 = poisson_twosample(count1, n1, count2, n2, method='score', ratio_null=1.5)
    pv2r = 0.2398
    assert_allclose(pv2, pv2r*2, rtol=0, atol=5e-3)
    print('score', s2, pv2 / 2)   # one sided in the "right" direction

    s2, pv2 = poisson_twosample(count1, n1, count2, n2, method='sqrt', ratio_null=1.5)
    pv2r = 0.2499
    assert_allclose(pv2, pv2r*2, rtol=0, atol=5e-3)
    print('sqrt', s2, pv2 / 2)   # one sided in the "right" direction

    print('\none-sided')
    # example 1 onesided
    count1, n1, count2, n2 = 60, 51477.5, 30, 54308.7

    s1, pv1 = poisson_twosample(count1, n1, count2, n2, method='wald', alternative='larger')
    pv1r = 0.000356
    assert_allclose(pv1, pv1r, rtol=0, atol=5e-6)
    print('wald', s1, pv1)   # one sided in the "right" direction

    s2, pv2 = poisson_twosample(count1, n1, count2, n2, method='score', alternative='larger')
    pv2r = 0.000316
    assert_allclose(pv2, pv2r, rtol=0, atol=5e-6)
    print('score', s2, pv2)   # one sided in the "right" direction

    s2, pv2 = poisson_twosample(count1, n1, count2, n2, method='sqrt', alternative='larger')
    pv2r = 0.000285
    assert_allclose(pv2, pv2r, rtol=0, atol=5e-6)
    print('sqrt', s2, pv2)   # one sided in the "right" direction

    # 'exact-cond', 'cond-midp'

    s2, pv2 = poisson_twosample(count1, n1, count2, n2, method='exact-cond',
                                ratio_null=1, alternative='larger')
    pv2r = 0.000428  # typo in Gu et al, switched pvalues between C and M
    assert_allclose(pv2, pv2r, rtol=0, atol=5e-4)
    print('exact-cond', s2, pv2)   # one sided in the "right" direction

    s2, pv2 = poisson_twosample(count1, n1, count2, n2, method='cond-midp',
                                ratio_null=1, alternative='larger')
    pv2r = 0.000310
    assert_allclose(pv2, pv2r, rtol=0, atol=5e-4)
    print('cond-midp', s2, pv2)   # one sided in the "right" direction

    ste1, pve1 = etest_twopoisson(count1, n1, count2, n2, method='score',
                                      ratio_null=1, alternative='larger')
    pve1r = 0.000298
    assert_allclose(pve1, pve1r, rtol=0, atol=5e-4)
    print('E-test score', pve1)

    ste1, pve1 = etest_twopoisson(count1, n1, count2, n2, method='wald',
                                      ratio_null=1, alternative='larger')
    pve1r = 0.000298
    assert_allclose(pve1, pve1r, rtol=0, atol=5e-4)
    print('E-test wald', pve1)

    print('\none-sided')
    # example2 onesided
    # I don't know why it's only 2.5 decimal agreement, rounding?
    count1, n1, count2, n2 = 41, 28010, 15, 19017
    s1, pv1 = poisson_twosample(count1, n1, count2, n2, method='wald',
                                ratio_null=1.5, alternative='larger')
    pv1r = 0.2309
    assert_allclose(pv1, pv1r, rtol=0, atol=5e-4)
    print('wald', s1, pv1)   # one sided in the "right" direction

    s2, pv2 = poisson_twosample(count1, n1, count2, n2, method='score',
                                ratio_null=1.5, alternative='larger')
    pv2r = 0.2398
    assert_allclose(pv2, pv2r, rtol=0, atol=5e-4)
    print('score', s2, pv2)   # one sided in the "right" direction

    s2, pv2 = poisson_twosample(count1, n1, count2, n2, method='sqrt',
                                ratio_null=1.5, alternative='larger')
    pv2r = 0.2499
    assert_allclose(pv2, pv2r, rtol=0, atol=5e-4)
    print('score', s2, pv2)   # one sided in the "right" direction

    # 'exact-cond', 'cond-midp'

    s2, pv2 = poisson_twosample(count1, n1, count2, n2, method='exact-cond',
                                ratio_null=1.5, alternative='larger')
    pv2r = 0.2913
    assert_allclose(pv2, pv2r, rtol=0, atol=5e-4)
    print('exact-cond', s2, pv2)   # one sided in the "right" direction

    s2, pv2 = poisson_twosample(count1, n1, count2, n2, method='cond-midp',
                                ratio_null=1.5, alternative='larger')
    pv2r = 0.2450
    assert_allclose(pv2, pv2r, rtol=0, atol=5e-4)
    print('cond-midp', s2, pv2)   # one sided in the "right" direction

    ste2, pve2 = etest_twopoisson(count1, n1, count2, n2, method='score',
                                      ratio_null=1.5, alternative='larger')
    pve2r = 0.2453
    assert_allclose(pve2, pve2r, rtol=0, atol=5e-4)
    print('E-test score', pve2)

    ste2, pve2 = etest_twopoisson(count1, n1, count2, n2, method='wald',
                                      ratio_null=1.5, alternative='larger')
    pve2r = 0.2453
    assert_allclose(pve2, pve2r, rtol=0, atol=5e-4)
    print('E-test wald', pve2)
