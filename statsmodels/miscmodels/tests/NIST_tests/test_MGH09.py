
import numpy as np
import statsmodels.api as sm
from numpy.testing import assert_almost_equal, assert_
from MGH09_testclass import TestNonlinearLS

class TestMGH09(TestNonlinearLS):

    def test_basic(self):
        res1 = self.res_start1
        res2 = self.res_start2
        res1J = self.resJ_start1
        res2J = self.resJ_start2
        certified = self.Cert_parameters
        assert_almost_equal(res1.params,certified)
        assert_almost_equal(res2.params,certified)
        assert_almost_equal(res1J.params,certified)
        assert_almost_equal(res2J.params,certified)
