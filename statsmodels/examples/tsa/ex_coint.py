
from __future__ import print_function
from statsmodels.tsa.tests.test_stattools import CheckCoint, TestCoint_t


#test whether t-test for cointegration equals that produced by Stata

tst = TestCoint_t()
print(tst.test_tstat())
