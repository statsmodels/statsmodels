import numpy as np
import numpy.testing as nt
from numpy.testing import assert_string_equal, assert_
from scikits.statsmodels.iolib.table import SimpleTable, default_txt_fmt

class test_formatting(object):
    def testSimpleTable_1(self):
        #desired = '\n=======================\n+      header1 header2-\n+stub1 1.30312 2.73999-\n+stub2 1.95038 2.65765-\n_______________________\n\n'
        desired = \
'''
=====================
      header1 header2
---------------------
stub1 1.30312 2.73999
stub2 1.95038 2.65765
---------------------
'''
        test1data = [[1.30312, 2.73999],[1.95038, 2.65765]]
        test1stubs = ('stub1', 'stub2')
        test1header = ('header1', 'header2')
        actual = SimpleTable(test1data, test1header, test1stubs,
                             txt_fmt=default_txt_fmt)
        print '###'
        print(actual)
        print '###'
        print desired
        print '###'
        assert_string_equal(desired, str(actual))#,
#                err_msg='On fail both tables are printed desired, actual'+
#                desired + str(actual))

    def test_SimpleTable_2(self):
        desired = '\n=====================================================================================\n+           header s1            header d1            header s2            header d2-\n+          stub R1 C1              10.3031           stub R1 C2                20.74-\n+          stub R2 C1              10.9504           stub R2 C2              20.6577-\n_____________________________________________________________________________________\n\n'
        data1 = [[10.30312, 20.73999]]
        data2 = [[10.95038, 20.65765]]
        stubs1 = ('stub R1 C1', 'stub R2 C1')
        stubs2 = ('stub R1 C2', 'stub R2 C2')
        header1 = ('header s1', 'header d1')
        header2 = ('header s2', 'header d2')
        actual1 = SimpleTable(data1, header1, stubs1, txt_fmt=default_txt_fmt)
        actual2 = SimpleTable(data2, header2, stubs2, txt_fmt=default_txt_fmt)
        actual_both = actual1.extend_right(actual2)
        print '###'
        print(actual_both)
        print '###'
        assert_(desired, str(actual_both))#,
#            err_msg='Basic test of Multiple stub columns.\n On fail both tables are printed desired, actual'+
#            desired + str(actual_both))

if __name__ == "__main__":
    nt.run_module_suite()


