import numpy as np
import unittest
from scikits.statsmodels.iolib.table import SimpleTable, default_txt_fmt

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

    def test_SimpleTable_4(self):
        """Basic test, test_SimpleTable_4
        test uses custom txt_fmt"""
        test_fmt = dict(
            data_fmts = ["%s"],
            data_fmt = "%s",  #deprecated; use data_fmts
            empty_cell = '*',
            colwidths = 15,
            colsep='*',
            row_pre = '*',
            row_post = '*',
            table_dec_above='*',
            table_dec_below='*',
            header_dec_below='*',
            header_fmt = '%s',
            stub_fmt = '%s',
            title_align='r',
            header_align = 'c',
            data_aligns = "l",
            stubs_align = "r",
            fmt = 'txt'
        )
        desired = \
'''*************************************************
*       *       *    header1    *    header2    *
*************************************************
*          stub1*1.30312        *2.73999        *
*          stub2*1.95038        *2.65765        *
*************************************************'''
        test1data = [[1.30312, 2.73999],[1.95038, 2.65765]]
        test1stubs = ('stub1', 'stub2')
        test1header = ('header1', 'header2')
        actual = SimpleTable(test1data, test1header, test1stubs,
                             txt_fmt=test_fmt)
        print('###')
        print(actual)
        print('###')
        self.assertEqual(desired, str(actual))

##    def test_regression_summary(self):
##        from scikits.statsmodels.tests.test_regression import TestOLS
##        desired = \
##'''   Summary of Regression Results
##====================================
##Dependent Variable:        Y
##Model:                    OLS
##Method:              Least Squares
##Date:               Sun, 02 May 2010
##Time:                   23:30:28
### obs:                    16.0
##Df residuals:             9.0
##Df model:                 6.0
##------------------------------------
##=====================================================================
##      coefficient       std. error     t-statistic        prob.
##---------------------------------------------------------------------
##X.0  15.0618722715    84.9149257747   0.177376028231  0.863140832808
##X.1 -0.0358191792926 0.0334910077722  -1.06951631722  0.312681061092
##X.2  -2.02022980382   0.488399681652  -4.13642735594 0.0025350917341
##X.3  -1.03322686717   0.214274163162  -4.82198531045 0.00094436676416
##X.4 -0.0511041056536  0.226073200069 -0.226051144664  0.826211795764
##X.5  1829.15146461    455.478499142   4.01588981271  0.00303680334162
##X.6  -3482258.6346    890420.383607   -3.91080291816 0.00356040366371
##---------------------------------------------------------------------
##===================================================================
##                       Models stats                  Residual stats
##-------------------------------------------------------------------
##R-squared:            0.995479004577  Durbin-Watson: 2.55948768928
##Adjusted R-squared:   0.992465007629  Omnibus:       0.748615075597
##F-statistic:          330.285339235   Prob(Omnibus): 0.687765365455
##Prob (F-statistic): 4.98403052871e-10 JB:            0.352772786021
##Log likelihood:       -109.617434808  Prob(JB):      0.838294009804
##AIC criterion:        233.234869617   Skew:          0.419983800891
##BIC criterion:        238.642990673   Kurtosis:      2.43373344891
##-------------------------------------------------------------------'''
##        aregression = TestOLS()
##        summarytable = aregression.res1.summary()
##        print('###')
##        print(summarytable)
##        print('###')
##        print(desired)
##        print('###')
##
##        self.assertEqual(desired, str(summarytable))

if __name__ == "__main__":
    unittest.main()



