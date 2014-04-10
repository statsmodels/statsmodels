# -*- coding: utf-8 -*-
"""

Created on Sat Jul 06 15:44:57 2013

Author: Josef Perktold
"""

from __future__ import print_function
from statsmodels.compat.python import iteritems
import numpy as np
from numpy.testing import assert_almost_equal
from statsmodels.datasets import macrodata

import statsmodels.tsa.stattools as tsa_stats

# some example data
mdata = macrodata.load().data
mdata = mdata[['realgdp','realcons']]
data = mdata.view((float,2))
data = np.diff(np.log(data), axis=0)

#R: lmtest:grangertest
r_result = [0.243097, 0.7844328, 195, 2]  #f_test
gr = tsa_stats.grangercausalitytests(data[:,1::-1], 2, verbose=False)
assert_almost_equal(r_result, gr[2][0]['ssr_ftest'], decimal=7)
assert_almost_equal(gr[2][0]['params_ftest'], gr[2][0]['ssr_ftest'],
                    decimal=7)

lag = 2
print('\nTest Results for %d lags' % lag)
print()
print('\n'.join(['%-20s statistic: %f6.4   p-value: %f6.4' % (k, res[0], res[1])
                 for k, res in iteritems(gr[lag][0]) ]))

print('\n Results for auxiliary restricted regression with two lags')
print()
print(gr[lag][1][0].summary())

print('\n Results for auxiliary unrestricted regression with two lags')
print()
print(gr[lag][1][1].summary())
