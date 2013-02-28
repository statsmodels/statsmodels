# -*- coding: utf-8 -*-
"""

Created on Thu Feb 28 13:24:59 2013

Author: Josef Perktold
"""

import numpy as np
from numpy.testing import assert_almost_equal

from statsmodels.stats.gof import chisquare_power


def test_chisquare_power():
    from results.results_power import pwr_chisquare
    for case in pwr_chisquare.values():
        power = chisquare_power(case.w, case.N, case.df + 1,
                                alpha=case.sig_level)
        assert_almost_equal(power, case.power, decimal=6,
                            err_msg=repr(vars(case)))
