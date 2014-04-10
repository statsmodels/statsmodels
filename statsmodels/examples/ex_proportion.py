# -*- coding: utf-8 -*-
"""

Created on Sun Apr 21 07:59:26 2013

Author: Josef Perktold
"""

from __future__ import print_function
from statsmodels.compat.python import lmap
import numpy as np
import statsmodels.stats.proportion as sms
import statsmodels.stats.weightstats as smw

from numpy.testing import assert_almost_equal


# Region, Eyes, Hair, Count
ss = '''\
1 blue  fair   23  1 blue  red     7  1 blue  medium 24
1 blue  dark   11  1 green fair   19  1 green red     7
1 green medium 18  1 green dark   14  1 brown fair   34
1 brown red     5  1 brown medium 41  1 brown dark   40
1 brown black   3  2 blue  fair   46  2 blue  red    21
2 blue  medium 44  2 blue  dark   40  2 blue  black   6
2 green fair   50  2 green red    31  2 green medium 37
2 green dark   23  2 brown fair   56  2 brown red    42
2 brown medium 53  2 brown dark   54  2 brown black  13'''

dta0 = np.array(ss.split()).reshape(-1,4)
dta = np.array(lmap(tuple, dta0.tolist()), dtype=[('Region', int), ('Eyes', 'S6'), ('Hair', 'S6'), ('Count', int)])

xfair = np.repeat([1,0], [228, 762-228])

# comparing to SAS last output at
# http://support.sas.com/documentation/cdl/en/procstat/63104/HTML/default/viewer.htm#procstat_freq_sect028.htm
# confidence interval for tost
ci01 = smw.confint_ztest(xfair, alpha=0.1)
assert_almost_equal(ci01,  [0.2719, 0.3265], 4)
res = smw.ztost(xfair, 0.18, 0.38)

assert_almost_equal(res[1][0], 7.1865, 4)
assert_almost_equal(res[2][0], -4.8701, 4)

nn = np.arange(200, 351)
pow_z = sms.power_ztost_prop(0.5, 0.72, nn, 0.6, alpha=0.05)
pow_bin = sms.power_ztost_prop(0.5, 0.72, nn, 0.6, alpha=0.05, dist='binom')
import matplotlib.pyplot as plt
plt.plot(nn, pow_z[0], label='normal')
plt.plot(nn, pow_bin[0], label='binomial')
plt.legend(loc='lower right')
plt.title('Proportion Equivalence Test: Power as function of sample size')
plt.xlabel('Number of Observations')
plt.ylabel('Power')

plt.show()

