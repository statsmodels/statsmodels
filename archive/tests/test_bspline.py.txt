import warnings

import numpy as np
from nipy.testing import *

bsp = None

def setup():
    # Suppress warnings during tests to reduce noise
    warnings.simplefilter("ignore")
    # import bspline module after suppressing UserWarnings
    global bsp
    import nipy.fixes.scipy.stats.models.bspline as bsp

def teardown():
    # Clear list of warning filters
    warnings.resetwarnings()


class TestBSpline(TestCase):

    def test1(self):
        b = bsp.BSpline(np.linspace(0,10,11), x=np.linspace(0,10,101))
        old = b._basisx.shape
        b.x = np.linspace(0,10,51)
        new = b._basisx.shape
        self.assertEqual((old[0], 51), new)

    # FIXME: Have no idea what this test does.  It's here to simply verify the
    # C extension is working (in a technical sense, not functional).
    def test_basis(self):
        b = bsp.BSpline(np.linspace(0,1,11))
        x = np.array([0.4, 0.5])
        v = b.basis(x, lower=0, upper=13)
        t = np.array([[ 0.        ,  0.        ],
                      [ 0.        ,  0.        ],
                      [ 0.        ,  0.        ],
                      [ 0.        ,  0.        ],
                      [ 0.16666667,  0.        ],
                      [ 0.66666667,  0.16666667],
                      [ 0.16666667,  0.66666667],
                      [ 0.        ,  0.16666667],
                      [ 0.        ,  0.        ],
                      [ 0.        ,  0.        ],
                      [ 0.        ,  0.        ],
                      [ 0.        ,  0.        ],
                      [ 0.        ,  0.        ]])
        assert_array_almost_equal(v, t, decimal=6)

    # FIXME: Have no idea what this test does.  It's here to simply verify the
    # C extension is working (in a technical sense, not functional).
    def test_gram(self):
        b = bsp.BSpline(np.linspace(0,1,11))
        grm = b.gram()
        assert grm.shape == (4, 13)
