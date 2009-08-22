"""
Test functions for models.tools
"""

import numpy as np
import numpy.random as R
from numpy.testing import *

from scikits.statsmodels import tools

class TestTools(TestCase):

    def test_recipr(self):
        X = np.array([[2,1],[-1,0]])
        Y = tools.recipr(X)
        assert_almost_equal(Y, np.array([[0.5,1],[0,0]]))

    def test_recipr0(self):
        X = np.array([[2,1],[-4,0]])
        Y = tools.recipr0(X)
        assert_almost_equal(Y, np.array([[0.5,1],[-0.25,0]]))

    def test_rank(self):
        X = R.standard_normal((40,10))
        self.assertEquals(tools.rank(X), 10)

        X[:,0] = X[:,1] + X[:,2]
        self.assertEquals(tools.rank(X), 9)

    def test_fullrank(self):
        X = R.standard_normal((40,10))
        X[:,0] = X[:,1] + X[:,2]

        Y = tools.fullrank(X)
        self.assertEquals(Y.shape, (40,9))
        self.assertEquals(tools.rank(Y), 9)

        X[:,5] = X[:,3] + X[:,4]
        Y = tools.fullrank(X)
        self.assertEquals(Y.shape, (40,8))
        self.assertEquals(tools.rank(Y), 8)

    def test_StepFunction(self):
        x = np.arange(20)
        y = np.arange(20)
        f = tools.StepFunction(x, y)
        assert_almost_equal(f( np.array([[3.2,4.5],[24,-3.1]]) ), [[ 3, 4], [19, 0]])

    def test_StepFunctionBadShape(self):
        x = np.arange(20)
        y = np.arange(21)
        self.assertRaises(ValueError, tools.StepFunction, x, y)
        x = np.zeros((2, 2))
        y = np.zeros((2, 2))
        self.assertRaises(ValueError, tools.StepFunction, x, y)



