import numpy as np
from numpy.random import standard_normal
from numpy.testing import (assert_equal, assert_array_equal,
                           assert_almost_equal, assert_string_equal, TestCase)
from nose.tools import (assert_true, assert_false, assert_raises)
from statsmodels.base.transform import (BoxCox)


class TestTransform(TestCase):

    def test_nonpositive(self):
        bc = BoxCox()
        x = [1, -1, 1]

        assert_raises(ValueError, bc.transform_boxcox, x)



