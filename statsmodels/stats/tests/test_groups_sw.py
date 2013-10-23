# -*- coding: utf-8 -*-
"""Test for a helper function for PanelHAC robust covariance

the functions should be rewritten to make it more efficient

Created on Thu May 17 21:09:41 2012

Author: Josef Perktold
"""
import numpy as np
from numpy.testing import assert_equal, assert_raises
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.grouputils import Group, GroupSorted

class CheckPanelLagMixin(object):

    def calculate(self):
        self.g = g = GroupSorted(self.gind)  # pylint: disable-msg=W0201
        self.alla = [(lag, sw.lagged_groups(self.x, lag, g.groupidx))  # pylint: disable-msg=W0201
                         for lag in range(5)]

    def test_values(self):
        for lag, (y0, ylag) in self.alla:
            assert_equal(y0, self.alle[lag].T)
            assert_equal(y0, ylag + lag)

    def test_raises(self):
        mlag = self.mlag
        assert_raises(ValueError, sw.lagged_groups, self.x, mlag,
                      self.g.groupidx)


class TestBalanced(CheckPanelLagMixin):

    def __init__(self):
        self.gind = np.repeat([0,1,2], 5)
        self.mlag = 5
        x = np.arange(15)
        x += 10**self.gind
        self.x = x[:,None]
        #expected result
        self.alle = {
            0 : np.array([[  1,   2,   3,   4,   5,  15,  16,  17,  18,  19,
                           110, 111, 112, 113, 114]]),
            1 : np.array([[  2,   3,   4,   5,  16,  17,  18,  19, 111, 112,
                           113, 114]]),
            2 : np.array([[  3,   4,   5,  17,  18,  19, 112, 113, 114]]),
            3 : np.array([[  4,   5,  18,  19, 113, 114]]),
            4 : np.array([[  5,  19, 114]])
            }
        self.calculate()

class TestUnBalanced(CheckPanelLagMixin):

    def __init__(self):
        self.gind = gind = np.repeat([0,1,2], [3, 5, 10])
        self.mlag = 10  #maxlag
        x = np.arange(18)
        x += 10**gind
        self.x = x[:,None]

        #expected result
        self.alle = {
            0 : np.array([[  1,   2,   3,  13,  14,  15,  16,  17, 108, 109,
                           110, 111, 112, 113, 114, 115, 116, 117]]),
            1 : np.array([[  2,   3,  14,  15,  16,  17, 109, 110, 111, 112,
                           113, 114, 115, 116, 117]]),
            2 : np.array([[  3,  15,  16,  17, 110, 111, 112, 113, 114, 115,
                           116, 117]]),
            3 : np.array([[ 16,  17, 111, 112, 113, 114, 115, 116, 117]]),
            4 : np.array([[ 17, 112, 113, 114, 115, 116, 117]]),
            5 : np.array([[113, 114, 115, 116, 117]]),
            }
        self.calculate()

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb-failures'], exit=False)
