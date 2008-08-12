"""
Test functions for models.robust.scale
"""

import numpy.random as R
from numpy.testing import *

import neuroimaging.fixes.scipy.stats.models.robust.scale as scale

W = R.standard_normal

class TestScale(TestCase):

    # FIXME: Figure out why this is failing and fix.
    @dec.skipknownfailure
    def test_MAD(self):
        X = W((40,10))
        m = scale.MAD(X)
        self.assertEquals(m.shape, (10,))

    # FIXME: Figure out why this is failing and fix.
    @dec.skipknownfailure
    def test_MADaxes(self):
        X = W((40,10,30))
        m = scale.MAD(X, axis=0)
        self.assertEquals(m.shape, (10,30))

        m = scale.MAD(X, axis=1)
        self.assertEquals(m.shape, (40,30))

        m = scale.MAD(X, axis=2)
        self.assertEquals(m.shape, (40,10))

        m = scale.MAD(X, axis=-1)
        self.assertEquals(m.shape, (40,10))

    # FIXME: Fix the axis length bug in stats.models.robust.scale.huber
    #     Then resolve ticket #587
    @dec.skipknownfailure
    def test_huber(self):
        X = W((40,10))
        m = scale.huber(X)
        self.assertEquals(m.shape, (10,))

    # FIXME: Fix the axis length bug in stats.models.robust.scale.huber
    @dec.skipknownfailure
    def test_huberaxes(self):
        X = W((40,10,30))
        m = scale.huber(X, axis=0)
        self.assertEquals(m.shape, (10,30))

        m = scale.huber(X, axis=1)
        self.assertEquals(m.shape, (40,30))

        m = scale.huber(X, axis=2)
        self.assertEquals(m.shape, (40,10))

        m = scale.huber(X, axis=-1)
        self.assertEquals(m.shape, (40,10))
