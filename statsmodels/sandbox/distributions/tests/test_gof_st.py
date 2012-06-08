# -*- coding: utf-8 -*-
"""

Created on Fri Jun 08 15:48:23 2012

Author: Josef Perktold
License : BSD-3

"""


print pvalue_st(t, cutoff, coef)

alpha = [0.15, 0.10, 0.05, 0.025, 0.01]
a2_exp = [0.916, 1.062, 1.321, 1.591, 1.959]

for t, aa in zip(a2_exp, alpha):
    print pvalue_st(t, cutoff, coef), aa

for t, aa in zip(crit_normal_a2, alpha):
    print pvalue_st(t, acut2, acoef2), aa

#compare with linear interpolation
from scipy import interpolate
nu2 = interpolate.interp1d(crit_normal_u2, alpha)
#check
assert_equal(nu2(crit_normal_u2), alpha)

for b in np.linspace(0, 1, 11):
    z = crit_normal_u2[:-1] + b * np.diff(crit_normal_u2)
    print (nu2(z) - [pvalue_st(zz, ucut2, ucoef2)[1] for zz in z])
    assert_almost_equal(nu2(z),
                        [pvalue_st(zz, ucut2, ucoef2)[1] for zz in z],
                        decimal=2)

for te in ['w2', 'u2', 'a2']:
    assert_almost_equal([pvalue_expon(z, te) for z in crit_expon[te]],
                         alpha, decimal=2)
    assert_almost_equal([pvalue_normal(z, te) for z in crit_normal[te]],
                         alpha, decimal=2)


#created for copying to R with
#np.random.seed(9768)
#xx = np.round(1000 * stats.expon.rvs(size=20), 3).astype(int)
xx = np.array([1580,  179, 1471,  328,  492, 1008, 1412, 4820, 2840,  559,
               223, 871,  791,  837, 1722, 1247,  985, 4378,  620,  530])
x = xx / 1000.

ge = GOFExpon(x)

b_list = [0, 0.3, 0.35, 0.5, 2]  #chosen to get a good spread of pvalues

#the following doesn't work well because for some case the tests differ too
#much
for b in b_list:
    ge2 = GOFExpon(x + b * x**2)
    ad = ge2.get_test('a2')
    for ti in ['d', 'v', 'w2', 'u2']:
        oth = ge2.get_test(ti)
        #check pvalues
        if oth[1] == 0.15:  #upper boundary for pval of d and v
            if not (ti == 'v' and b in [0.5]):  #skip one test for Kuiper
                assert_array_less(0.11, ad[1])
        elif oth[1] == 0.01:  #upper boundary for pval of d and v
            #if not ti == 'v':  #skip for Kuiper
            assert_array_less(ad[1], 0.01)
        else:
            #assert_almost_equal(ad[1], oth[1], 1)
            #assert_array_less(np.abs(ad[1] / oth[1] - 1), 0.6) #25)
            #assert_array_less(np.abs(oth[1] - ad[1]) / ad[1]**2, 1)
            assert_array_less(np.abs(ad[1] - oth[1]), 0.01 + 0.6 * oth[1]) #25)

#b in rows, ti in columns
res_r = np.array([0.1564240118194638, 0.09436760924796966, 0.6314329797982694,
                  0.1793511287764876, 0.1506515020772339, 0.887774219046726,
                  0.184369741403067, 0.1697126047249459, 0.978323412566048,
                  0.2055857239850029, 0.225018414134194, 1.247031661161905,
                  0.285678730877463, 0.510830687544145, 2.808327043589259]
                  ) #.reshape(-1,3)


res_gof = [ GOFExpon(x + b * x**2).get_test(ti)[0] for b in b_list for ti in ['d', 'w2', 'a2']]

assert_almost_equal(res_gof, res_r, 7)

