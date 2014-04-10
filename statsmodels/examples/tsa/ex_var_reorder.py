
from __future__ import print_function
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.tests.test_var import TestVARResults

test_VAR = TestVARResults()
test_VAR.test_reorder()
