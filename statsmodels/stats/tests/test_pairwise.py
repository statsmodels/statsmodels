# -*- coding: utf-8 -*-
"""

Created on Wed Mar 28 15:34:18 2012

Author: Josef Perktold
"""

from statsmodels.compatnp.py3k import BytesIO, asbytes
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from statsmodels.stats.libqsturng import qsturng

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

ss5 = '''\
2 - 3\t4.340\t0.691\t7.989\t***
2 - 1\t4.600\t0.951\t8.249\t***
3 - 2\t-4.340\t-7.989\t-0.691\t***
3 - 1\t0.260\t-3.389\t3.909\t-
1 - 2\t-4.600\t-8.249\t-0.951\t***
1 - 3\t-0.260\t-3.909\t3.389\t'''

#accommodate recfromtxt for python 3.2, requires bytes
ss = asbytes(ss)
ss2 = asbytes(ss2)
ss3 = asbytes(ss3)
ss5 = asbytes(ss5)

dta = np.recfromtxt(BytesIO(ss), names=("Rust","Brand","Replication"))
dta2 = np.recfromtxt(BytesIO(ss2), names = ("idx", "Treatment", "StressReduction"))
dta3 = np.recfromtxt(BytesIO(ss3), names = ("Brand", "Relief"))
dta5 = np.recfromtxt(BytesIO(ss5), names = ('pair', 'mean', 'lower', 'upper', 'sig'), delimiter='\t')
sas_ = dta5[[1,3,2]]

from statsmodels.stats.multicomp import (tukeyhsd, pairwise_tukeyhsd,
                                         MultiComparison)
#import statsmodels.sandbox.stats.multicomp as multi
#print tukeyhsd(dta['Brand'], dta['Rust'])

def get_thsd(mci, alpha=0.05):
    var_ = np.var(mci.groupstats.groupdemean(), ddof=len(mci.groupsunique))
    means = mci.groupstats.groupmean
    nobs = mci.groupstats.groupnobs
    resi = tukeyhsd(means, nobs, var_, df=None, alpha=alpha,
                    q_crit=qsturng(1-alpha, len(means), (nobs-1).sum()))
    #print resi[4]
    var2 = (mci.groupstats.groupvarwithin() * (nobs - 1)).sum() \
                                                        / (nobs - 1).sum()
    assert_almost_equal(var_, var2, decimal=14)
    return resi

class CheckTuckeyHSDMixin(object):

    @classmethod
    def setup_class_(self):
        self.mc = MultiComparison(self.endog, self.groups)
        self.res = self.mc.tukeyhsd(alpha=self.alpha)

    def test_multicomptukey(self):
        meandiff1 = self.res[1][2]
        assert_almost_equal(meandiff1, self.meandiff2, decimal=14)

        confint1 = self.res[1][4]
        assert_almost_equal(confint1, self.confint2, decimal=2)

        reject1 = self.res[1][1]
        assert_equal(reject1, self.reject2)

    def test_group_tukey(self):
        res_t = get_thsd(self.mc,alpha=self.alpha)
        assert_almost_equal(res_t[4], self.confint2, decimal=2)

    def test_shortcut_function(self):
        #check wrapper function
        res = pairwise_tukeyhsd(self.endog, self.groups, alpha=self.alpha)
        assert_almost_equal(res[1][4], self.res[1][4], decimal=14)


class TestTuckeyHSD2(CheckTuckeyHSDMixin):

    @classmethod
    def setup_class(self):
        #balanced case
        self.endog = dta2['StressReduction']
        self.groups = dta2['Treatment']
        self.alpha = 0.05
        self.setup_class_() #in super

        #from R
        tukeyhsd2s = np.array([ 1.5,1,-0.5,0.3214915,
                               -0.1785085,-1.678509,2.678509,2.178509,
                                0.6785085,0.01056279,0.1079035,0.5513904]
                                ).reshape(3,4, order='F')
        self.meandiff2 = tukeyhsd2s[:, 0]
        self.confint2 = tukeyhsd2s[:, 1:3]
        pvals = tukeyhsd2s[:, 3]
        self.reject2 = pvals < 0.05


class TestTuckeyHSD2s(CheckTuckeyHSDMixin):

    @classmethod
    def setup_class(self):
        #unbalanced case
        self.endog = dta2['StressReduction'][3:29]
        self.groups = dta2['Treatment'][3:29]
        self.alpha = 0.01
        self.setup_class_()

        #from R
        tukeyhsd2s = np.array(
                [1.8888888888888889, 0.888888888888889, -1, 0.2658549,
                 -0.5908785, -2.587133, 3.511923, 2.368656,
                 0.5871331, 0.002837638, 0.150456, 0.1266072]
                ).reshape(3,4, order='F')
        self.meandiff2 = tukeyhsd2s[:, 0]
        self.confint2 = tukeyhsd2s[:, 1:3]
        pvals = tukeyhsd2s[:, 3]
        self.reject2 = pvals < 0.01


class TestTuckeyHSD3(CheckTuckeyHSDMixin):

    @classmethod
    def setup_class(self):
        #SAS case
        self.endog = dta3['Relief']
        self.groups = dta3['Brand']
        self.alpha = 0.05
        self.setup_class_()
        #super(self, self).setup_class_()
        #CheckTuckeyHSD.setup_class_()

        self.meandiff2 = sas_['mean']
        self.confint2 = sas_[['lower','upper']].view(float).reshape((3,2))
        self.reject2 = sas_['sig'] == asbytes('***')

