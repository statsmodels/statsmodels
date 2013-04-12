'''Tests for multipletests and fdr pvalue corrections

Author : Josef Perktold


['b', 's', 'sh', 'hs', 'h', 'fdr_i', 'fdr_n', 'fdr_tsbh']
are tested against R:multtest

'hommel' is tested against R stats p_adjust (not available in multtest

'fdr_gbs', 'fdr_2sbky' I did not find them in R, currently tested for
    consistency only

'''

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_

from statsmodels.stats.multitest import (multipletests, fdrcorrection,
                                         fdrcorrection_twostage)
from statsmodels.stats.multicomp import tukeyhsd

pval0 = np.array([0.838541367553 , 0.642193923795 , 0.680845947633 ,
        0.967833824309 , 0.71626938238 , 0.177096952723 , 5.23656777208e-005 ,
        0.0202732688798 , 0.00028140506198 , 0.0149877310796])

res_multtest = np.array([[  5.2365677720800003e-05,   5.2365677720800005e-04,
                  5.2365677720800005e-04,   5.2365677720800005e-04,
                  5.2353339704891422e-04,   5.2353339704891422e-04,
                  5.2365677720800005e-04,   1.5337740764175588e-03],
               [  2.8140506198000000e-04,   2.8140506197999998e-03,
                  2.5326455578199999e-03,   2.5326455578199999e-03,
                  2.8104897961789277e-03,   2.5297966317768816e-03,
                  1.4070253098999999e-03,   4.1211324652269442e-03],
               [  1.4987731079600001e-02,   1.4987731079600000e-01,
                  1.1990184863680001e-01,   1.1990184863680001e-01,
                  1.4016246580579017e-01,   1.1379719679449507e-01,
                  4.9959103598666670e-02,   1.4632862843720582e-01],
               [  2.0273268879800001e-02,   2.0273268879799999e-01,
                  1.4191288215860001e-01,   1.4191288215860001e-01,
                  1.8520270949069695e-01,   1.3356756197485375e-01,
                  5.0683172199499998e-02,   1.4844940238274187e-01],
               [  1.7709695272300000e-01,   1.0000000000000000e+00,
                  1.0000000000000000e+00,   9.6783382430900000e-01,
                  8.5760763426056130e-01,   6.8947825122356643e-01,
                  3.5419390544599999e-01,   1.0000000000000000e+00],
               [  6.4219392379499995e-01,   1.0000000000000000e+00,
                  1.0000000000000000e+00,   9.6783382430900000e-01,
                  9.9996560644133570e-01,   9.9413539782557070e-01,
                  8.9533672797500008e-01,   1.0000000000000000e+00],
               [  6.8084594763299999e-01,   1.0000000000000000e+00,
                  1.0000000000000000e+00,   9.6783382430900000e-01,
                  9.9998903512635740e-01,   9.9413539782557070e-01,
                  8.9533672797500008e-01,   1.0000000000000000e+00],
               [  7.1626938238000004e-01,   1.0000000000000000e+00,
                  1.0000000000000000e+00,   9.6783382430900000e-01,
                  9.9999661886871472e-01,   9.9413539782557070e-01,
                  8.9533672797500008e-01,   1.0000000000000000e+00],
               [  8.3854136755300002e-01,   1.0000000000000000e+00,
                  1.0000000000000000e+00,   9.6783382430900000e-01,
                  9.9999998796038225e-01,   9.9413539782557070e-01,
                  9.3171263061444454e-01,   1.0000000000000000e+00],
               [  9.6783382430900000e-01,   1.0000000000000000e+00,
                  1.0000000000000000e+00,   9.6783382430900000e-01,
                  9.9999999999999878e-01,   9.9413539782557070e-01,
                  9.6783382430900000e-01,   1.0000000000000000e+00]])


res_multtest2_columns = ['rawp', 'Bonferroni', 'Holm', 'Hochberg', 'SidakSS', 'SidakSD',
                'BH', 'BY', 'ABH', 'TSBH_0.05']

rmethods = {'rawp':(0,'pval'), 'Bonferroni':(1,'b'), 'Holm':(2,'h'),
            'Hochberg':(3,'sh'), 'SidakSS':(4,'s'), 'SidakSD':(5,'hs'),
            'BH':(6,'fdr_i'), 'BY':(7,'fdr_n'),
            'TSBH_0.05':(9, 'fdr_tsbh')}

NA = np.nan
# all rejections, except for Bonferroni and Sidak
res_multtest2 = np.array([
     0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.012, 0.024, 0.036, 0.048,
     0.06, 0.072, 0.012, 0.02, 0.024, 0.024, 0.024, 0.024, 0.012, 0.012,
     0.012, 0.012, 0.012, 0.012, 0.01194015976019192, 0.02376127616613988,
     0.03546430060660932, 0.04705017875634587, 0.058519850599,
     0.06987425045000606, 0.01194015976019192, 0.01984063872102404,
     0.02378486270400004, 0.023808512, 0.023808512, 0.023808512, 0.012,
     0.012, 0.012, 0.012, 0.012, 0.012, 0.0294, 0.0294, 0.0294, 0.0294,
     0.0294, 0.0294, NA, NA, NA, NA, NA, NA, 0, 0, 0, 0, 0, 0
    ]).reshape(6,10, order='F')

res_multtest3 = np.array([
     0.001, 0.002, 0.003, 0.004, 0.005, 0.05, 0.06, 0.07, 0.08, 0.09, 0.01,
     0.02, 0.03, 0.04, 0.05, 0.5, 0.6, 0.7, 0.8, 0.9, 0.01, 0.018, 0.024,
     0.028, 0.03, 0.25, 0.25, 0.25, 0.25, 0.25, 0.01, 0.018, 0.024, 0.028,
     0.03, 0.09, 0.09, 0.09, 0.09, 0.09, 0.00995511979025177,
     0.01982095664805061, 0.02959822305108317, 0.03928762649718986,
     0.04888986953422814, 0.4012630607616213, 0.4613848859051006,
     0.5160176928207072, 0.5656115457763677, 0.6105838818818925,
     0.00995511979025177, 0.0178566699880266, 0.02374950634358763,
     0.02766623106147537, 0.02962749064373438, 0.2262190625000001,
     0.2262190625000001, 0.2262190625000001, 0.2262190625000001,
     0.2262190625000001, 0.01, 0.01, 0.01, 0.01, 0.01, 0.08333333333333334,
     0.0857142857142857, 0.0875, 0.0888888888888889, 0.09,
     0.02928968253968254, 0.02928968253968254, 0.02928968253968254,
     0.02928968253968254, 0.02928968253968254, 0.2440806878306878,
     0.2510544217687075, 0.2562847222222222, 0.2603527336860670,
     0.2636071428571428, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, 0.005,
     0.005, 0.005, 0.005, 0.005, 0.04166666666666667, 0.04285714285714286,
     0.04375, 0.04444444444444445, 0.045
    ]).reshape(10,10, order='F')


class CheckMultiTestsMixin(object):
    def test_multi_pvalcorrection(self):
        #test against R package multtest mt.rawp2adjp

        res_multtest = self.res2
        pval0 = res_multtest[:,0]

        for k,v in rmethods.items():
            if v[1] in self.methods:
                reject, pvalscorr = multipletests(pval0,
                                                  alpha=self.alpha,
                                                  method=v[1])[:2]
                assert_almost_equal(pvalscorr, res_multtest[:,v[0]], 15)
                assert_equal(reject, pvalscorr <= self.alpha)

        pvalscorr = np.sort(fdrcorrection(pval0, method='n')[1])
        assert_almost_equal(pvalscorr, res_multtest[:,7], 15)
        pvalscorr = np.sort(fdrcorrection(pval0, method='i')[1])
        assert_almost_equal(pvalscorr, res_multtest[:,6], 15)

class TestMultiTests1(CheckMultiTestsMixin):
    def __init__(self):
        self.methods =  ['b', 's', 'sh', 'hs', 'h', 'fdr_i', 'fdr_n']
        self.alpha = 0.1
        self.res2 = res_multtest

class TestMultiTests2(CheckMultiTestsMixin):
    # case: all hypothesis rejected (except 'b' and 's'
    def __init__(self):
        self.methods =  ['b', 's', 'sh', 'hs', 'h', 'fdr_i', 'fdr_n']
        self.alpha = 0.05
        self.res2 = res_multtest2

class TestMultiTests3(CheckMultiTestsMixin):
    def __init__(self):
        self.methods =  ['b', 's', 'sh', 'hs', 'h', 'fdr_i', 'fdr_n',
                         'fdr_tsbh']
        self.alpha = 0.05
        self.res2 = res_multtest3

def test_pvalcorrection_reject():
    # consistency test for reject boolean and pvalscorr

    for alpha in [0.01, 0.05, 0.1]:
        for method in ['b', 's', 'sh', 'hs', 'h', 'hommel', 'fdr_i', 'fdr_n',
                       'fdr_tsbky', 'fdr_tsbh', 'fdr_gbs']:
            for ii in range(11):
                pval1 = np.hstack((np.linspace(0.0001, 0.0100, ii),
                                   np.linspace(0.05001, 0.11, 10 - ii)))
                # using .05001 instead of 0.05 to avoid edge case issue #768
                reject, pvalscorr = multipletests(pval1, alpha=alpha,
                                                  method=method)[:2]
                #print 'reject.sum', v[1], reject.sum()
                msg = 'case %s %3.2f rejected:%d\npval_raw=%r\npvalscorr=%r' % (
                                 method, alpha, reject.sum(), pval1, pvalscorr)
                #assert_equal(reject, pvalscorr <= alpha, err_msg=msg)
                yield assert_equal, reject, pvalscorr <= alpha, msg


def test_hommel():
    #tested agains R stats p_adjust(pval0, method='hommel')
    pval0 = np.array(
              [ 0.00116,  0.00924,  0.01075,  0.01437,  0.01784,  0.01918,
                0.02751,  0.02871,  0.03054,  0.03246,  0.04259,  0.06879,
                0.0691 ,  0.08081,  0.08593,  0.08993,  0.09386,  0.09412,
                0.09718,  0.09758,  0.09781,  0.09788,  0.13282,  0.20191,
                0.21757,  0.24031,  0.26061,  0.26762,  0.29474,  0.32901,
                0.41386,  0.51479,  0.52461,  0.53389,  0.56276,  0.62967,
                0.72178,  0.73403,  0.87182,  0.95384])

    result_ho = np.array(
              [ 0.0464            ,  0.25872           ,  0.29025           ,
                0.3495714285714286,  0.41032           ,  0.44114           ,
                0.57771           ,  0.60291           ,  0.618954          ,
                0.6492            ,  0.7402725000000001,  0.86749           ,
                0.86749           ,  0.8889100000000001,  0.8971477777777778,
                0.8993            ,  0.9175374999999999,  0.9175374999999999,
                0.9175374999999999,  0.9175374999999999,  0.9175374999999999,
                0.9175374999999999,  0.95384           ,  0.9538400000000001,
                0.9538400000000001,  0.9538400000000001,  0.9538400000000001,
                0.9538400000000001,  0.9538400000000001,  0.9538400000000001,
                0.9538400000000001,  0.9538400000000001,  0.9538400000000001,
                0.9538400000000001,  0.9538400000000001,  0.9538400000000001,
                0.9538400000000001,  0.9538400000000001,  0.9538400000000001,
                0.9538400000000001])

    rej, pvalscorr, _, _ = multipletests(pval0, alpha=0.1, method='ho')
    assert_almost_equal(pvalscorr, result_ho, 15)
    assert_equal(rej, result_ho < 0.1)  #booleans

def test_fdr_bky():
    # test for fdrcorrection_twostage
    # example from BKY
    pvals = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344, 0.0459,
             0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.000 ]

    #no test for corrected p-values, but they are inherited
    #same number of rejection as in BKY paper:
    #single step-up:4, two-stage:8, iterated two-step:9
    #also alpha_star is the same as theirs for TST
    #print fdrcorrection0(pvals, alpha=0.05, method='indep')
    #print fdrcorrection_twostage(pvals, alpha=0.05, iter=False)
    res_tst = fdrcorrection_twostage(pvals, alpha=0.05, iter=False)
    assert_almost_equal([0.047619, 0.0649], res_tst[-1][:2],3) #alpha_star for stage 2
    assert_equal(8, res_tst[0].sum())
    #print fdrcorrection_twostage(pvals, alpha=0.05, iter=True)

def test_tukeyhsd():
    #example multicomp in R p 83

    res = '''\
    pair      diff        lwr        upr       p adj
    P-M   8.150000 -10.037586 26.3375861 0.670063958
    S-M  -3.258333 -21.445919 14.9292527 0.982419709
    T-M  23.808333   5.620747 41.9959194 0.006783701
    V-M   4.791667 -13.395919 22.9792527 0.931020848
    S-P -11.408333 -29.595919  6.7792527 0.360680099
    T-P  15.658333  -2.529253 33.8459194 0.113221634
    V-P  -3.358333 -21.545919 14.8292527 0.980350080
    T-S  27.066667   8.879081 45.2542527 0.002027122
    V-S   8.050000 -10.137586 26.2375861 0.679824487
    V-T -19.016667 -37.204253 -0.8290806 0.037710044
    '''

    res = np.array([[ 8.150000,  -10.037586, 26.3375861, 0.670063958],
                     [-3.258333,  -21.445919, 14.9292527, 0.982419709],
                     [23.808333,    5.620747, 41.9959194, 0.006783701],
                     [ 4.791667,  -13.395919, 22.9792527, 0.931020848],
                     [-11.408333, -29.595919,  6.7792527, 0.360680099],
                     [15.658333,  -2.529253,  33.8459194, 0.113221634],
                     [-3.358333, -21.545919,  14.8292527, 0.980350080],
                     [27.066667,   8.879081,  45.2542527, 0.002027122],
                     [ 8.050000, -10.137586,  26.2375861, 0.679824487],
                     [-19.016667, -37.204253, -0.8290806, 0.037710044]])

    m_r = [94.39167, 102.54167,  91.13333, 118.20000,  99.18333]
    myres = tukeyhsd(m_r, 6, 110.8, alpha=0.05, df=4)
    from numpy.testing import assert_almost_equal, assert_equal
    pairs, reject, meandiffs, std_pairs, confint, q_crit = myres[:6]
    assert_almost_equal(meandiffs, res[:, 0], decimal=5)
    assert_almost_equal(confint, res[:, 1:3], decimal=2)
    assert_equal(reject, res[:, 3]<0.05)

