import numpy as np
import unittest
#from scikits.statsmodels.iolib.table import SimpleTable, default_txt_fmt

import sys
sys.path.append("/Users/vmd/Dropbox/Statsmodels/Descriptive-Stats/scikits/statsmodels/iolib")
sys.path.append("/Users/vmd/Dropbox/Statsmodels/Descriptive-Stats/scikits/statsmodels/tests")
from table import SimpleTable, default_txt_fmt
from test_regression import TestOLS

class TestSimpleTable(unittest.TestCase):
    def test_SimpleTable_1(self):
        """Basic test, test_SimpleTable_1"""
        desired = \
'''=====================
      header1 header2
---------------------
stub1 1.30312 2.73999
stub2 1.95038 2.65765
---------------------'''
        test1data = [[1.30312, 2.73999],[1.95038, 2.65765]]
        test1stubs = ('stub1', 'stub2')
        test1header = ('header1', 'header2')
        actual = SimpleTable(test1data, test1header, test1stubs,
                             txt_fmt=default_txt_fmt)
        self.assertEqual(desired, str(actual))

    def test_SimpleTable_2(self):
        """ Test SimpleTable.extend_right()"""
        desired = \
'''=============================================================
           header s1 header d1            header s2 header d2
-------------------------------------------------------------
stub R1 C1  10.30312  10.73999 stub R1 C2  50.95038  50.65765
stub R2 C1  90.30312  90.73999 stub R2 C2  40.95038  40.65765
-------------------------------------------------------------'''
        data1 = [[10.30312, 10.73999], [90.30312, 90.73999]]
        data2 = [[50.95038, 50.65765], [40.95038, 40.65765]]
        stubs1 = ['stub R1 C1', 'stub R2 C1']
        stubs2 = ['stub R1 C2', 'stub R2 C2']
        header1 = ['header s1', 'header d1']
        header2 = ['header s2', 'header d2']
        actual1 = SimpleTable(data1, header1, stubs1, txt_fmt=default_txt_fmt)
        actual2 = SimpleTable(data2, header2, stubs2, txt_fmt=default_txt_fmt)
        actual1.extend_right(actual2)
        self.assertEqual(desired, str(actual1))

    def test_SimpleTable_3(self):
        """ Test SimpleTable.extend() as in extend down"""
        desired = \
'''==============================
           header s1 header d1
------------------------------
stub R1 C1  10.30312  10.73999
stub R2 C1  90.30312  90.73999
           header s2 header d2
------------------------------
stub R1 C2  50.95038  50.65765
stub R2 C2  40.95038  40.65765
------------------------------'''
        data1 = [[10.30312, 10.73999], [90.30312, 90.73999]]
        data2 = [[50.95038, 50.65765], [40.95038, 40.65765]]
        stubs1 = ['stub R1 C1', 'stub R2 C1']
        stubs2 = ['stub R1 C2', 'stub R2 C2']
        header1 = ['header s1', 'header d1']
        header2 = ['header s2', 'header d2']
        actual1 = SimpleTable(data1, header1, stubs1, txt_fmt=default_txt_fmt)
        actual2 = SimpleTable(data2, header2, stubs2, txt_fmt=default_txt_fmt)
        actual1.extend(actual2)
        self.assertEqual(desired, str(actual1))

    def test_regression_summary(self):
        from scikits.statsmodels.regression import TesOLS
        desired = \
'''================================================================================
Dependent Variable: Y
Model: OLS
Method: Least Squares
Date: Sun, 02 May 2010
Time: 22:02:39
# obs:           16
Df residuals:     9
Df model:         6
================================================================================
variable       coefficient       std. error     t-statistic       prob.
================================================================================
X.0                15.0619          84.9149          0.1774       0.8631
X.1             -0.0358192        0.0334910          -1.070       0.3127
X.2               -2.02023         0.488400          -4.136       0.002535
X.3               -1.03323         0.214274          -4.822       0.0009444
X.4             -0.0511041         0.226073         -0.2261       0.8262
X.5                1829.15          455.478           4.016       0.003037
X.6           -3.48226e+06          890420.          -3.911       0.003560
================================================================================
Models stats                         Residual stats
================================================================================
R-squared            0.99548          Durbin-Watson stat   2.559
Adjusted R-squared   0.99247          Omnibus stat         0.7486
F-statistic          330.29           Prob(Omnibus stat)   0.6878
Prob (F-statistic)   4.984e-10         JB stat              0.3528
Log likelihood       -109.62          Prob(JB)             0.8383
AIC criterion        233.23           Skew                 0.42
BIC criterion        238.6            Kurtosis             2.434
================================================================================'''
        aregression = TestOLS()
        summarytable = aregression.res1.summary()
        print('###')
        print(summarytable)
        print('###')
        print(desired)
        print('###')

        self.assertEqual(desired, str(summarytable))

if __name__ == "__main__":
    unittest.main()



