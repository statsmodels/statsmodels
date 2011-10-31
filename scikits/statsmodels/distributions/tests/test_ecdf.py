import numpy as np
import numpy.testing as npt
from scikits.statsmodels.distributions import StepFunction

class TestDistributions(npt.TestCase):

    def test_StepFunction(self):
        x = np.arange(20)
        y = np.arange(20)
        f = StepFunction(x, y)
        npt.assert_almost_equal(f( np.array([[3.2,4.5],[24,-3.1]]) ), 
                                             [[ 3, 4], [19, 0]])

    def test_StepFunctionBadShape(self):
        x = np.arange(20)
        y = np.arange(21)
        self.assertRaises(ValueError, StepFunction, x, y)
        x = np.zeros((2, 2))
        y = np.zeros((2, 2))
        self.assertRaises(ValueError, StepFunction, x, y)
