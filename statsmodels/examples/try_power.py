# -*- coding: utf-8 -*-
"""

Created on Sat Mar 02 14:38:17 2013

Author: Josef Perktold
"""

from __future__ import print_function
import numpy as np

import statsmodels.stats.power as smp
import statsmodels.stats.proportion as smpr

sigma=1; d=0.3; nobs=80; alpha=0.05
print(smp.normal_power(d, nobs/2, 0.05))
print(smp.NormalIndPower().power(d, nobs, 0.05))
print(smp.NormalIndPower().solve_power(effect_size=0.3, nobs1=80, alpha=0.05, power=None))
print(0.475100870572638, 'R')

norm_pow = smp.normal_power(-0.01, nobs/2, 0.05)
norm_pow_R = 0.05045832927039234
#value from R: >pwr.2p.test(h=0.01,n=80,sig.level=0.05,alternative="two.sided")
print('norm_pow', norm_pow, norm_pow - norm_pow_R)

norm_pow = smp.NormalIndPower().power(0.01, nobs, 0.05, alternative="larger")
norm_pow_R = 0.056869534873146124
#value from R: >pwr.2p.test(h=0.01,n=80,sig.level=0.05,alternative="greater")
print('norm_pow', norm_pow, norm_pow - norm_pow_R)

# Note: negative effect size is same as switching one-sided alternative
# TODO: should I switch to larger/smaller instead of "one-sided" options
norm_pow = smp.NormalIndPower().power(-0.01, nobs, 0.05, alternative="larger")
norm_pow_R = 0.0438089705093578
#value from R: >pwr.2p.test(h=0.01,n=80,sig.level=0.05,alternative="less")
print('norm_pow', norm_pow, norm_pow - norm_pow_R)


#Note: I use n_bins and ddof instead of df
# pwr.chisq.test(w=0.289,df=(4-1),N=100,sig.level=0.05)
chi2_pow = smp.GofChisquarePower().power(0.289, 100, 4, 0.05)
chi2_pow_R = 0.675077657003721
print('chi2_pow', chi2_pow, chi2_pow - chi2_pow_R)

chi2_pow = smp.GofChisquarePower().power(0.01, 100, 4, 0.05)
chi2_pow_R = 0.0505845519208533
print('chi2_pow', chi2_pow, chi2_pow - chi2_pow_R)

chi2_pow = smp.GofChisquarePower().power(2, 100, 4, 0.05)
chi2_pow_R = 1
print('chi2_pow', chi2_pow, chi2_pow - chi2_pow_R)

chi2_pow = smp.GofChisquarePower().power(0.9, 100, 4, 0.05)
chi2_pow_R = 0.999999999919477
print('chi2_pow', chi2_pow, chi2_pow - chi2_pow_R, 'lower precision ?')

chi2_pow = smp.GofChisquarePower().power(0.8, 100, 4, 0.05)
chi2_pow_R = 0.999999968205591
print('chi2_pow', chi2_pow, chi2_pow - chi2_pow_R)

def cohen_es(*args, **kwds):
    print("You better check what's a meaningful effect size for your question.")


#BUG: after fixing 2.sided option, 2 rejection areas
tt_pow = smp.TTestPower().power(effect_size=0.01, nobs=nobs, alpha=0.05)
tt_pow_R = 0.05089485285965
# value from> pwr.t.test(d=0.01,n=80,sig.level=0.05,type="one.sample",alternative="two.sided")
print('tt_pow', tt_pow, tt_pow - tt_pow_R)
