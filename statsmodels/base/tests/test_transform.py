import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_warns,
                           assert_raises, dec, assert_)
from statsmodels.base.transform import (BoxCox)
from statsmodels.datasets import macrodata


class SetupBoxCox(object):
    data = macrodata.load()
    x = data.data['realgdp']
    bc = BoxCox()


class TestTransform(SetupBoxCox):

    def test_nonpositive(self):
        y = [1, -1, 1]
        assert_raises(ValueError, self.bc.transform_boxcox, y)

    def test_invalid_bounds(self):
        assert_raises(ValueError, self.bc._est_lambda, self.x, (-3, 2, 3))

    def test_unclear_methods(self):
        # Both _est_lambda and untransform have a method argument that should
        # be tested.
        assert_raises(ValueError, self.bc._est_lambda,
                      self.x, (-1, 2), 2, 'test')
        assert_raises(ValueError, self.bc.untransform_boxcox,
                      self.x, 1, 'test')


if __name__=="__main__":
    import nose
    import numpy as np
    np.testing.run_module_suite()
