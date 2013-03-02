# -*- coding: utf-8 -*-
"""

Created on Sat Mar 02 14:38:17 2013

Author: Josef Perktold
"""

import numpy as np

import statsmodels.stats.power as smp
import statsmodels.stats._proportion as smpr

sigma=1; d=0.3; nobs=80; alpha=0.05
print smp.normal_power(d, nobs/2, 0.05)
print smp.NormalIndPower().power(d, nobs, 0.05)
print smp.NormalIndPower().solve_power(effect_size=0.3, nobs1=80, alpha=0.05, beta=None)
print 0.475100870572638, 'R'

es_abs = smp.normal_power(-0.01, nobs/2, 0.05, abs_effect=True)
es_abs_R = 0.05045832927039234
print 'es_abs', es_abs, es_abs - es_abs_R

#something strange
# R seems to assume the effect size could be in either side
# this would be correct if effect size d is abs(d)
# check SAS, this looks weird to me, or I don't understand 2-sided power
# added abs_effect option to smp.normal_power but not to class
'''
>>> smp.normal_power(0.01, nobs/2, 0.05) + smp.normal_power(-0.01, nobs/2, 0.05)
0.050458329270392233
>>> _ - 0.05045832927039234
-1.0408340855860843e-16
value from R: >pwr.2p.test(h=0.01,n=80,sig.level=0.05,alternative="two.sided")
'''
