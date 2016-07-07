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
        # Testing negative values
        y = [1, -1, 1]
        assert_raises(ValueError, self.bc.transform_boxcox, y)

        # Testing nonzero
        y = [1, 0, 1]
        assert_raises(ValueError, self.bc.transform_boxcox, y)

    def test_invalid_bounds(self):
        # more than two bounds
        assert_raises(ValueError, self.bc._est_lambda, self.x, (-3, 2, 3))

        # upper bound <= lower bound
        assert_raises(ValueError, self.bc._est_lambda, self.x, (2, -1))

    def test_unclear_methods(self):
        # Both _est_lambda and untransform have a method argument that should
        # be tested.
        assert_raises(ValueError, self.bc._est_lambda,
                      self.x, (-1, 2), 'test')
        assert_raises(ValueError, self.bc.untransform_boxcox,
                      self.x, 1, 'test')

    def test_unclear_scale_parameter(self):
        # bc.guerrero allows for 'mad' and 'sd', for the MAD and Standard
        # Deviation, respectively
        assert_raises(ValueError, self.bc._est_lambda,
                      self.x, scale='test')

        # Next, check if mad/sd work:
        self.bc._est_lambda(self.x, scale='mad')
        self.bc._est_lambda(self.x, scale='MAD')

        self.bc._est_lambda(self.x, scale='sd')
        self.bc._est_lambda(self.x, scale='SD')

    def test_valid_guerrero(self):
        # `l <- BoxCox.lambda(x, method="guerrero")` on a ts object
        # with frequency 4 (BoxCox.lambda defaults to 2, but we use
        # Guerrero and Perera (2004) as a guideline)
        lmbda = self.bc._est_lambda(self.x, method='guerrero', R=4)
        assert_almost_equal(lmbda, 0.507624, 4)

        # `l <- BoxCox.lambda(x, method="guerrero")` with the default grouping
        # parameter (namely, R=2).
        lmbda = self.bc._est_lambda(self.x, method='guerrero', R=2)
        assert_almost_equal(lmbda, 0.513893, 4)

    def test_boxcox_transformation_methods(self):
        return True