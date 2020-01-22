# -*- coding: utf-8 -*-
"""

Created on Fri Aug 16 13:41:12 2013

Author: Josef Perktold
"""

import numpy as np
from numpy.testing import assert_equal, assert_raises

from scipy import stats
import statsmodels.stats.robust_compare as stats

# taken from scipy and adjusted
class Test_Trim(object):
    # test trim functions
    def test_trim1(self):
        a = np.arange(11)
        assert_equal(stats.trim1(a, 0.1), np.arange(10))
        assert_equal(stats.trim1(a, 0.2), np.arange(9))
        assert_equal(stats.trim1(a, 0.2, tail='left'), np.arange(2,11))
        assert_equal(stats.trim1(a, 3/11., tail='left'), np.arange(3,11))

    def test_trimboth(self):
        a = np.arange(11)
        a2 = np.arange(24).reshape(6, 4)
        a3 = np.arange(24).reshape(6, 4, order='F')
        assert_equal(stats.trimboth(a, 3/11.), np.arange(3,8))
        assert_equal(stats.trimboth(a, 0.2), np.array([2, 3, 4, 5, 6, 7, 8]))

        assert_equal(stats.trimboth(a2, 0.2),
                     np.arange(4,20).reshape(4,4))
        assert_equal(stats.trimboth(a3, 2/6.),
               np.array([[2, 8, 14, 20],[3, 9, 15, 21]]))
        assert_raises(ValueError, stats.trimboth,
               np.arange(24).reshape(4,6).T, 4/6.)

    def test_trim_mean(self):
        a = np.array([ 4,  8,  2,  0,  9,  5, 10,  1,  7,  3,  6])
        idx = np.array([3, 5, 0, 1, 2, 4])
        a2 = np.arange(24).reshape(6, 4)[idx, :]
        a3 = np.arange(24).reshape(6, 4, order='F')[idx, :]
        assert_equal(stats.trim_mean(a3, 2/6.),
                        np.array([2.5, 8.5, 14.5, 20.5]))
        assert_equal(stats.trim_mean(a2, 2/6.),
                        np.array([10., 11., 12., 13.]))
        idx4 = np.array([1, 0, 3, 2])
        a4 = np.arange(24).reshape(4, 6)[idx4, :]
        assert_equal(stats.trim_mean(a4, 2/6.),
                        np.array([9., 10., 11., 12., 13., 14.]))
        # shuffled arange(24)
        a = np.array([ 7, 11, 12, 21, 16,  6, 22,  1,  5,  0, 18, 10, 17,  9,
                      19, 15, 23, 20,  2, 14,  4, 13,  8,  3])
        assert_equal(stats.trim_mean(a, 2/6.), 11.5)
        assert_equal(stats.trim_mean([5,4,3,1,2,0], 2/6.), 2.5)

        # check axis argument
        np.random.seed(1234)
        a = np.random.randint(20, size=(5, 6, 4, 7))
        for axis in [0, 1, 2, 3, -1]:
            res1 = stats.trim_mean(a, 2/6., axis=axis)
            res2 = stats.trim_mean(np.rollaxis(a, axis), 2/6.)
            assert_equal(res1, res2)

        res1 = stats.trim_mean(a, 2/6., axis=None)
        res2 = stats.trim_mean(a.ravel(), 2/6.)
        assert_equal(res1, res2)


tt = Test_Trim()
#tt.test_trim1()
tt.test_trimboth()
tt.test_trim_mean()
