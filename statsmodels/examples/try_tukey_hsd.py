# -*- coding: utf-8 -*-
"""

Created on Wed Mar 28 15:34:18 2012

Author: Josef Perktold
"""
from io import StringIO

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from scipy import stats

from statsmodels.stats.libqsturng import qsturng
from statsmodels.stats.multicomp import tukeyhsd
import statsmodels.stats.multicomp as multi


ss = '''\
  43.9  1   1
  39.0  1   2
  46.7  1   3
  43.8  1   4
  44.2  1   5
  47.7  1   6
  43.6  1   7
  38.9  1   8
  43.6  1   9
  40.0  1  10
  89.8  2   1
  87.1  2   2
  92.7  2   3
  90.6  2   4
  87.7  2   5
  92.4  2   6
  86.1  2   7
  88.1  2   8
  90.8  2   9
  89.1  2  10
  68.4  3   1
  69.3  3   2
  68.5  3   3
  66.4  3   4
  70.0  3   5
  68.1  3   6
  70.6  3   7
  65.2  3   8
  63.8  3   9
  69.2  3  10
  36.2  4   1
  45.2  4   2
  40.7  4   3
  40.5  4   4
  39.3  4   5
  40.3  4   6
  43.2  4   7
  38.7  4   8
  40.9  4   9
  39.7  4  10'''

#idx   Treatment StressReduction
ss2 = '''\
1     mental               2
2     mental               2
3     mental               3
4     mental               4
5     mental               4
6     mental               5
7     mental               3
8     mental               4
9     mental               4
10    mental               4
11  physical               4
12  physical               4
13  physical               3
14  physical               5
15  physical               4
16  physical               1
17  physical               1
18  physical               2
19  physical               3
20  physical               3
21   medical               1
22   medical               2
23   medical               2
24   medical               2
25   medical               3
26   medical               2
27   medical               3
28   medical               1
29   medical               3
30   medical               1'''

ss3 = '''\
1 24.5
1 23.5
1 26.4
1 27.1
1 29.9
2 28.4
2 34.2
2 29.5
2 32.2
2 30.1
3 26.1
3 28.3
3 24.3
3 26.2
3 27.8'''

cylinders = np.array([8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 6, 6, 6, 4, 4,
                    4, 4, 4, 4, 6, 8, 8, 8, 8, 4, 4, 4, 4, 8, 8, 8, 8, 6, 6, 6, 6, 4, 4, 4, 4, 6, 6,
                    6, 6, 4, 4, 4, 4, 4, 8, 4, 6, 6, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                    4, 4, 4, 4, 4, 4, 4, 6, 6, 4, 6, 4, 4, 4, 4, 4, 4, 4, 4])
cyl_labels = np.array(['USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'France',
    'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'Japan', 'USA', 'USA', 'USA', 'Japan',
    'Germany', 'France', 'Germany', 'Sweden', 'Germany', 'USA', 'USA', 'USA', 'USA', 'USA', 'Germany',
    'USA', 'USA', 'France', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'Germany',
    'Japan', 'USA', 'USA', 'USA', 'USA', 'Germany', 'Japan', 'Japan', 'USA', 'Sweden', 'USA', 'France',
    'Japan', 'Germany', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA',
    'Germany', 'Japan', 'Japan', 'USA', 'USA', 'Japan', 'Japan', 'Japan', 'Japan', 'Japan', 'Japan', 'USA',
    'USA', 'USA', 'USA', 'Japan', 'USA', 'USA', 'USA', 'Germany', 'USA', 'USA', 'USA'])

dta = np.recfromtxt(StringIO(ss), names=("Rust","Brand","Replication"))
dta2 = np.recfromtxt(StringIO(ss2), names = ("idx", "Treatment", "StressReduction"))
dta3 = np.recfromtxt(StringIO(ss3), names = ("Brand", "Relief"))

#print tukeyhsd(dta['Brand'], dta['Rust'])

def get_thsd(mci):
    var_ = np.var(mci.groupstats.groupdemean(), ddof=len(mci.groupsunique))
    means = mci.groupstats.groupmean
    nobs = mci.groupstats.groupnobs
    resi = tukeyhsd(means, nobs, var_, df=None, alpha=0.05, q_crit=qsturng(0.95, len(means), (nobs-1).sum()))
    print(resi[4])
    var2 = (mci.groupstats.groupvarwithin() * (nobs - 1)).sum() \
                                                        / (nobs - 1).sum()
    assert_almost_equal(var_, var2, decimal=14)
    return resi

mc = multi.MultiComparison(dta['Rust'], dta['Brand'])
res = mc.tukeyhsd()
print(res)

mc2 = multi.MultiComparison(dta2['StressReduction'], dta2['Treatment'])
res2 = mc2.tukeyhsd()
print(res2)

mc2s = multi.MultiComparison(dta2['StressReduction'][3:29], dta2['Treatment'][3:29])
res2s = mc2s.tukeyhsd()
print(res2s)
res2s_001 = mc2s.tukeyhsd(alpha=0.01)
#R result
tukeyhsd2s = np.array([1.888889,0.8888889,-1,0.2658549,-0.5908785,-2.587133,3.511923,2.368656,0.5871331,0.002837638,0.150456,0.1266072]).reshape(3,4, order='F')
assert_almost_equal(res2s_001.confint, tukeyhsd2s[:,1:3], decimal=3)

mc3 = multi.MultiComparison(dta3['Relief'], dta3['Brand'])
res3 = mc3.tukeyhsd()
print(res3)

tukeyhsd4 = multi.MultiComparison(cylinders, cyl_labels, group_order=["Sweden", "Japan", "Germany", "France", "USA"])
res4 = tukeyhsd4.tukeyhsd()
print(res4)
try:
    import matplotlib.pyplot as plt
    fig = res4.plot_simultaneous("USA")
    plt.show()
except Exception as e:
    print(e)

for mci in [mc, mc2, mc3]:
    get_thsd(mci)

print(mc2.allpairtest(stats.ttest_ind, method='b')[0])

'''same as SAS:
>>> np.var(mci.groupstats.groupdemean(), ddof=3)
4.6773333333333351
>>> var_ = np.var(mci.groupstats.groupdemean(), ddof=3)
>>> tukeyhsd(means, nobs, var_, df=None, alpha=0.05, q_crit=qsturng(0.95, 3, 12))[4]
array([[ 0.95263648,  8.24736352],
       [-3.38736352,  3.90736352],
       [-7.98736352, -0.69263648]])
>>> tukeyhsd(means, nobs, var_, df=None, alpha=0.05, q_crit=3.77278)[4]
array([[ 0.95098508,  8.24901492],
       [-3.38901492,  3.90901492],
       [-7.98901492, -0.69098508]])
'''

ss5 = '''\
Comparisons significant at the 0.05 level are indicated by ***.
BRAND
Comparison	Difference
Between
Means	Simultaneous 95% Confidence Limits	 Sign.
2 - 3	4.340	0.691	7.989	***
2 - 1	4.600	0.951	8.249	***
3 - 2	-4.340	-7.989	-0.691	***
3 - 1	0.260	-3.389	3.909	 -
1 - 2	-4.600	-8.249	-0.951	***
1 - 3	-0.260	-3.909	3.389	'''

ss5 = '''\
2 - 3	4.340	0.691	7.989	***
2 - 1	4.600	0.951	8.249	***
3 - 2	-4.340	-7.989	-0.691	***
3 - 1	0.260	-3.389	3.909	 -
1 - 2	-4.600	-8.249	-0.951	***
1 - 3	-0.260	-3.909	3.389	'''

dta5 = np.recfromtxt(StringIO(ss5), names = ('pair', 'mean', 'lower', 'upper', 'sig'), delimiter='\t')

sas_ = dta5[[1,3,2]]
confint1 = res3.confint
confint2 = sas_[['lower','upper']].view(float).reshape((3,2))
assert_almost_equal(confint1, confint2, decimal=2)
reject1 = res3.reject
reject2 = sas_['sig'] == '***'
assert_equal(reject1, reject2)
meandiff1 = res3.meandiffs
meandiff2 = sas_['mean']
assert_almost_equal(meandiff1, meandiff2, decimal=14)
