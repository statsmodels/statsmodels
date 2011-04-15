from scikits.statsmodels.tsa.tests_stattools import CheckCoint, TestCoint_t


#test whether t-test for cointegration equals that produced by Stata

tst = TestCoint_t()
print tst.test_tstat()
