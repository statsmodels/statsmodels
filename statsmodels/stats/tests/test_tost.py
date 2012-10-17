# -*- coding: utf-8 -*-
"""

Created on Wed Oct 17 09:48:34 2012

Author: Josef Perktold
"""

import numpy as np
import statsmodels.stats.weightstats as smws

from numpy.testing import assert_almost_equal

class Holder(object):
    pass

raw_clinic = '''\
1     1 2.84 4.00 3.45 2.55 2.46
2     1 2.51 3.26 3.10 2.82 2.48
3     1 2.41 4.14 3.37 2.99 3.04
4     1 2.95 3.42 2.82 3.37 3.35
5     1 3.14 3.25 3.31 2.87 3.41
6     1 3.79 4.34 3.88 3.40 3.16
7     1 4.14 4.97 4.25 3.43 3.06
8     1 3.85 4.31 3.92 3.58 3.91
9     1 3.02 3.11 2.20 2.24 2.28
10    1 3.45 3.41 3.80 3.86 3.91
11    1 5.37 5.02 4.59 3.99 4.27
12    1 3.81 4.21 4.08 3.18 1.86
13    1 4.19 4.59 4.79 4.17 2.60
14    1 3.16 5.30 4.69 4.83 4.51
15    1 3.84 4.32 4.25 3.87 2.93
16    2 2.60 3.76 2.86 2.41 2.71
17    2 2.82 3.66 3.20 2.49 2.49
18    2 2.18 3.65 3.87 3.00 2.65
19    2 3.46 3.60 2.97 1.80 1.74
20    2 4.01 3.48 4.42 3.06 2.76
21    2 3.04 2.87 2.87 2.71 2.87
22    2 3.47 3.24 3.47 3.26 3.14
23    2 4.06 3.92 3.18 3.06 1.74
24    2 2.91 3.99 3.06 2.02 3.18
25    2 3.59 4.21 4.02 3.26 2.85
26    2 4.51 4.21 3.78 2.63 1.92
27    2 3.16 3.31 3.28 3.25 3.52
28    2 3.86 3.61 3.28 3.19 3.09
29    2 3.31 2.97 3.76 3.18 2.60
30    2 3.02 2.73 3.87 3.50 2.93'''.split()
clinic = np.array(raw_clinic, float).reshape(-1,7)
    

#t = tost(-clinic$var2[16:30] + clinic$var2[1:15], eps=0.6)
tost_clinic_paired = Holder()
tost_clinic_paired.sample = 'paired'
tost_clinic_paired.mean_diff = 0.5626666666666665
tost_clinic_paired.se_diff = 0.2478276410785118
tost_clinic_paired.alpha = 0.05
tost_clinic_paired.ci_diff = (0.1261653305099018, 0.999168002823431)
tost_clinic_paired.df = 14
tost_clinic_paired.epsilon = 0.6
tost_clinic_paired.result = 'not rejected'
tost_clinic_paired.p_value = 0.4412034046017588
tost_clinic_paired.check_me = (0.525333333333333, 0.6)

#> t = tost(-clinic$var1[16:30] + clinic$var1[1:15], eps=0.6)
#> cat_items(t, prefix="tost_clinic_paired_1.")
tost_clinic_paired_1 = Holder()
tost_clinic_paired_1.mean_diff = 0.1646666666666667
tost_clinic_paired_1.se_diff = 0.1357514067862445
tost_clinic_paired_1.alpha = 0.05
tost_clinic_paired_1.ci_diff = (-0.0744336620516462, 0.4037669953849797)
tost_clinic_paired_1.df = 14
tost_clinic_paired_1.epsilon = 0.6
tost_clinic_paired_1.result = 'rejected'
tost_clinic_paired_1.p_value = 0.003166881489265175
tost_clinic_paired_1.check_me = (-0.2706666666666674, 0.600000000000001)


#> t = tost(clinic$var2[1:15], clinic$var2[16:30], eps=0.6)
#> cat_items(t, prefix="tost_clinic_indep.")
tost_clinic_indep = Holder()
tost_clinic_indep.sample = 'independent'
tost_clinic_indep.mean_diff = 0.562666666666666
tost_clinic_indep.se_diff = 0.2149871904637392
tost_clinic_indep.alpha = 0.05
tost_clinic_indep.ci_diff = (0.194916250699966, 0.930417082633366)
tost_clinic_indep.df = 24.11000151062728
tost_clinic_indep.epsilon = 0.6
tost_clinic_indep.result = 'not rejected'
tost_clinic_indep.p_value = 0.4317936812594803
tost_clinic_indep.check_me = (0.525333333333332, 0.6)

#> t = tost(clinic$var1[1:15], clinic$var1[16:30], eps=0.6)
#> cat_items(t, prefix="tost_clinic_indep_1.")
tost_clinic_indep_1 = Holder()
tost_clinic_indep_1.sample = 'independent'
tost_clinic_indep_1.mean_diff = 0.1646666666666667
tost_clinic_indep_1.se_diff = 0.2531625991083627
tost_clinic_indep_1.alpha = 0.05
tost_clinic_indep_1.ci_diff = (-0.2666862980722534, 0.596019631405587)
tost_clinic_indep_1.df = 26.7484787582315
tost_clinic_indep_1.epsilon = 0.6
tost_clinic_indep_1.result = 'rejected'
tost_clinic_indep_1.p_value = 0.04853083976236974
tost_clinic_indep_1.check_me = (-0.2706666666666666, 0.6)

#pooled variance
#> t = tost(clinic$var1[1:15], clinic$var1[16:30], eps=0.6, var.equal = TRUE)
#> cat_items(t, prefix="tost_clinic_indep_1_pooled.")
tost_clinic_indep_1_pooled = Holder()
tost_clinic_indep_1_pooled.mean_diff = 0.1646666666666667
tost_clinic_indep_1_pooled.se_diff = 0.2531625991083628
tost_clinic_indep_1_pooled.alpha = 0.05
tost_clinic_indep_1_pooled.ci_diff = (-0.2659960620757337, 0.595329395409067)
tost_clinic_indep_1_pooled.df = 28
tost_clinic_indep_1_pooled.epsilon = 0.6
tost_clinic_indep_1_pooled.result = 'rejected'
tost_clinic_indep_1_pooled.p_value = 0.04827315100761467
tost_clinic_indep_1_pooled.check_me = (-0.2706666666666666, 0.6)

#> t = tost(clinic$var2[1:15], clinic$var2[16:30], eps=0.6, var.equal = TRUE)
#> cat_items(t, prefix="tost_clinic_indep_2_pooled.")
tost_clinic_indep_2_pooled = Holder()
tost_clinic_indep_2_pooled.mean_diff = 0.562666666666666
tost_clinic_indep_2_pooled.se_diff = 0.2149871904637392
tost_clinic_indep_2_pooled.alpha = 0.05
tost_clinic_indep_2_pooled.ci_diff = (0.1969453064978777, 0.928388026835454)
tost_clinic_indep_2_pooled.df = 28
tost_clinic_indep_2_pooled.epsilon = 0.6
tost_clinic_indep_2_pooled.result = 'not rejected'
tost_clinic_indep_2_pooled.p_value = 0.43169347692374
tost_clinic_indep_2_pooled.check_me = (0.525333333333332, 0.6)


res1 = smws.tost_paired(clinic[:15, 2], clinic[15:, 2], -0.6, 0.6, transform=None)
res2 = smws.tost_paired(clinic[:15, 3], clinic[15:, 3], -0.6, 0.6, transform=None)
res = smws.tost_ind(clinic[:15, 3], clinic[15:, 3], -0.6, 0.6, usevar='separate')


class CheckTost(object):
    
    def test_pval(self):
        assert_almost_equal(self.res1.pvalue, self.res2.p_value, decimal=13)
        #assert_almost_equal(self.res1.df, self.res2.df, decimal=13)

class TestTostp1(CheckTost):
    def __init__(self):
        self.res2 = tost_clinic_paired_1
        x, y = clinic[:15, 2], clinic[15:, 2]
        self.res1 = Holder()
        res = smws.tost_paired(x, y, -0.6, 0.6, transform=None)
        self.res1.pvalue = res[0]
        #self.res1.df = res[1][-1] not yet

class TestTostp2(CheckTost):
    def __init__(self):
        self.res2 = tost_clinic_paired
        x, y = clinic[:15, 3], clinic[15:, 3]
        self.res1 = Holder()
        res = smws.tost_paired(x, y, -0.6, 0.6, transform=None)
        self.res1.pvalue = res[0]

class TestTosti1(CheckTost):
    def __init__(self):
        self.res2 = tost_clinic_indep_1
        x, y = clinic[:15, 2], clinic[15:, 2]
        self.res1 = Holder()
        res = smws.tost_ind(x, y, -0.6, 0.6, usevar='separate')
        self.res1.pvalue = res[0]

class TestTosti2(CheckTost):
    def __init__(self):
        self.res2 = tost_clinic_indep
        x, y = clinic[:15, 3], clinic[15:, 3]
        self.res1 = Holder()
        res = smws.tost_ind(x, y, -0.6, 0.6, usevar='separate')
        self.res1.pvalue = res[0]

class TestTostip1(CheckTost):
    def __init__(self):
        self.res2 = tost_clinic_indep_1_pooled
        x, y = clinic[:15, 2], clinic[15:, 2]
        self.res1 = Holder()
        res = smws.tost_ind(x, y, -0.6, 0.6, usevar='pooled')
        self.res1.pvalue = res[0]

class TestTostip2(CheckTost):
    def __init__(self):
        self.res2 = tost_clinic_indep_2_pooled
        x, y = clinic[:15, 3], clinic[15:, 3]
        self.res1 = Holder()
        res = smws.tost_ind(x, y, -0.6, 0.6, usevar='pooled')
        self.res1.pvalue = res[0]

if __name__ == '__main__':
    for cls in [TestTostp1, TestTostp2, TestTosti1, TestTosti2,
                TestTostip1, TestTostip2]:
        print cls
        tt = cls()
        tt.test_pval()


