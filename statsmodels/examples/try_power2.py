# -*- coding: utf-8 -*-
"""

Created on Wed Mar 13 13:06:14 2013

Author: Josef Perktold
"""

from __future__ import print_function
from statsmodels.stats.power import TTestPower, TTestIndPower, tt_solve_power

if __name__ == '__main__':
    effect_size, alpha, power = 0.5, 0.05, 0.8

    ttest_pow = TTestPower()
    print('\nroundtrip - root with respect to all variables')
    print('\n       calculated, desired')

    nobs_p = ttest_pow.solve_power(effect_size=effect_size, nobs=None, alpha=alpha, power=power)
    print('nobs  ', nobs_p)
    print('effect', ttest_pow.solve_power(effect_size=None, nobs=nobs_p, alpha=alpha, power=power), effect_size)

    print('alpha ', ttest_pow.solve_power(effect_size=effect_size, nobs=nobs_p, alpha=None, power=power), alpha)
    print('power  ', ttest_pow.solve_power(effect_size=effect_size, nobs=nobs_p, alpha=alpha, power=None), power)

    print('\nroundtrip - root with respect to all variables')
    print('\n       calculated, desired')

    print('nobs  ', tt_solve_power(effect_size=effect_size, nobs=None, alpha=alpha, power=power), nobs_p)
    print('effect', tt_solve_power(effect_size=None, nobs=nobs_p, alpha=alpha, power=power), effect_size)

    print('alpha ', tt_solve_power(effect_size=effect_size, nobs=nobs_p, alpha=None, power=power), alpha)
    print('power  ', tt_solve_power(effect_size=effect_size, nobs=nobs_p, alpha=alpha, power=None), power)

    print('\none sided')
    nobs_p1 = tt_solve_power(effect_size=effect_size, nobs=None, alpha=alpha, power=power, alternative='larger')
    print('nobs  ', nobs_p1)
    print('effect', tt_solve_power(effect_size=None, nobs=nobs_p1, alpha=alpha, power=power, alternative='larger'), effect_size)
    print('alpha ', tt_solve_power(effect_size=effect_size, nobs=nobs_p1, alpha=None, power=power, alternative='larger'), alpha)
    print('power  ', tt_solve_power(effect_size=effect_size, nobs=nobs_p1, alpha=alpha, power=None, alternative='larger'), power)

    #start_ttp = dict(effect_size=0.01, nobs1=10., alpha=0.15, power=0.6)

    ttind_solve_power = TTestIndPower().solve_power

    print('\nroundtrip - root with respect to all variables')
    print('\n       calculated, desired')

    nobs_p2 = ttind_solve_power(effect_size=effect_size, nobs1=None, alpha=alpha, power=power)
    print('nobs  ', nobs_p2)
    print('effect', ttind_solve_power(effect_size=None, nobs1=nobs_p2, alpha=alpha, power=power), effect_size)
    print('alpha ', ttind_solve_power(effect_size=effect_size, nobs1=nobs_p2, alpha=None, power=power), alpha)
    print('power  ', ttind_solve_power(effect_size=effect_size, nobs1=nobs_p2, alpha=alpha, power=None), power)
    print('ratio  ', ttind_solve_power(effect_size=effect_size, nobs1=nobs_p2, alpha=alpha, power=power, ratio=None), 1)

    print('\ncheck ratio')
    print('smaller power', ttind_solve_power(effect_size=effect_size, nobs1=nobs_p2, alpha=alpha, power=0.7, ratio=None), '< 1')
    print('larger power ', ttind_solve_power(effect_size=effect_size, nobs1=nobs_p2, alpha=alpha, power=0.9, ratio=None), '> 1')
