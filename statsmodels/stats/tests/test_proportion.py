# -*- coding: utf-8 -*-
"""

Created on Fri Mar 01 14:56:56 2013

Author: Josef Perktold
"""

import numpy as np
from numpy.testing import assert_almost_equal

from statsmodels.stats.proportion import confint_proportion



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

if __name__ == '__main__':
    test_confint_proportion()
