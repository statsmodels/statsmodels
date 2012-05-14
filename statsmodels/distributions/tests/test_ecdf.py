import numpy as np
import numpy.testing as npt
from statsmodels.distributions import StepFunction, monotone_fn_inverter

class TestDistributions(npt.TestCase):

    def test_StepFunction(self):
        x = np.arange(20)
        y = np.arange(20)
        f = StepFunction(x, y)
        npt.assert_almost_equal(f( np.array([[3.2,4.5],[24,-3.1],[3.0, 4.0]])),
                                             [[ 3, 4], [19, 0],  [2, 3]])

    def test_StepFunctionBadShape(self):
        x = np.arange(20)
        y = np.arange(21)
        self.assertRaises(ValueError, StepFunction, x, y)
        x = np.zeros((2, 2))
        y = np.zeros((2, 2))
        self.assertRaises(ValueError, StepFunction, x, y)

    def test_StepFunctionValueSideRight(self):
        x = np.arange(20)
        y = np.arange(20)
        f = StepFunction(x, y, side='right')
        npt.assert_almost_equal(f( np.array([[3.2,4.5],[24,-3.1],[3.0, 4.0]])),
                                             [[ 3, 4], [19, 0],  [3, 4]])

    def test_StepFunctionRepeatedValues(self):
        x = [1, 1, 2, 2, 2, 3, 3, 3, 4, 5]
        y = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        f = StepFunction(x, y)
        npt.assert_almost_equal(f([1, 2, 3, 4, 5]), [0, 7, 10, 13, 14])
        f2 = StepFunction(x, y, side='right')
        npt.assert_almost_equal(f2([1, 2, 3, 4, 5]), [7, 10, 13, 14, 15])

    def test_monotone_fn_inverter(self):
        x = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        fn = lambda x : 1./x
        y = fn(np.array(x))
        f = monotone_fn_inverter(fn, x)
        npt.assert_array_equal(f.y, x[::-1])
        npt.assert_array_equal(f.x, y[::-1])

