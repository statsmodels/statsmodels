"""
Test functions for models.robust.scale
"""

import numpy as np
import numpy.random as R
import numpy.testing as nptest
import nose.tools

# Example from Section 5.5, Venables & Ripley (2002)

chem = np.array([2.20, 2.20, 2.4, 2.4, 2.5, 2.7, 2.8, 2.9, 3.03, 3.03, 3.10, 3.37, \
                     3.4, 3.4, 3.4, 3.5, 3.6, 3.7, 3.7, 3.7,3.7, 3.77, 5.28, 28.95])

import neuroimaging.fixes.scipy.stats.models.robust.scale as scale
reload(scale)

W = R.standard_normal

def test_chem():
    """
    MAD test from chem data set, using value in Venables & Ripley
    """
    yield nptest.assert_almost_equal, np.mean(chem), 4.2804, 4
    yield nptest.assert_almost_equal, np.median(chem), 3.385, 4
    yield nptest.assert_almost_equal, scale.MAD(chem), 0.52632, 5
    yield nptest.assert_almost_equal, scale.huber(chem)[0], 3.20549, 5
    yield nptest.assert_almost_equal, scale.huber(chem)[1], 0.67365, 5

    # the default Huber uses a one-step version of
    # the 'full' version using HuberT with t=1.5

    n = scale.norms.HuberT()
    n.t = 1.5
    h = scale.Huber(norm=n)
    yield nptest.assert_almost_equal, scale.huber(chem)[0], h(chem)[0], 5
    yield nptest.assert_almost_equal, scale.huber(chem)[1], h(chem)[1], 5

    hh = scale.Huber(norm=scale.norms.Hampel())
    yield nptest.assert_almost_equal, hh(chem)[0], 3.17434, 5
    yield nptest.assert_almost_equal, hh(chem)[1], 0.66783, 5

def test_MAD():
    X = W((40,10))
    m = scale.MAD(X)
    nose.tools.assert_equals(m.shape, (10,))

def test_MADaxes():
    X = W((40,10,30))
    m = scale.MAD(X, axis=0)
    yield nose.tools.assert_equals, m.shape, (10,30)

    m = scale.MAD(X, axis=1)
    yield nose.tools.assert_equals, m.shape, (40,30)

    m = scale.MAD(X, axis=2)
    yield nose.tools.assert_equals, m.shape, (40,10)

    m = scale.MAD(X, axis=-1)
    yield nose.tools.assert_equals, m.shape, (40,10)



# FIXME: Fix the axis length bug in stats.models.robust.scale.huber
#     Then resolve ticket #587
#@dec.knownfailureif(True)
def test_huber():
    X = W((40,10))
    h = scale.Huber(niter=50)
    m, s = h(X)
    yield nose.tools.assert_equals, m.shape, (10,)

def test_huberaxes():
    X = W((40,10,30))
    h = scale.Huber(niter=500, tol=1.0e-05)
    m, s = h(X, axis=0)
    yield nose.tools.assert_equals, m.shape, (10,30)

    m, s = h(X, axis=1)
    yield nose.tools.assert_equals, m.shape, (40,30)

    m, s = h(X, axis=2)
    yield nose.tools.assert_equals, m.shape, (40,10)

    m, s = h(X, axis=-1)
    yield nose.tools.assert_equals, m.shape, (40,10)
