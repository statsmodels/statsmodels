"""
Tests for iolib/foreign.py
"""

from numpy.testing import *
import numpy as np
import scikits.statsmodels.api as sm
import os

# Test precisions
DECIMAL_4 = 4
DECIMAL_3 = 3


def test_genfromdta():
    """
    Test genfromdta vs. results/macrodta.npy created with genfromtxt.
    """
#NOTE: Stata handles data very oddly.  Round tripping from csv to dta
#    to ndarray 2710.349 (csv) -> 2510.2491 (stata) -> 2710.34912109375
#    (dta/ndarray)
    curdir = os.path.dirname(os.path.abspath(__file__))
    #res2 = np.load(curdir+'/results/macrodata.npy')
    #res2 = res2.view((float,len(res2[0])))
    from results.macrodata import macrodata_result as res2
    res1 = sm.iolib.genfromdta(curdir+'/../../datasets/macrodata/macrodata.dta')
    #res1 = res1.view((float,len(res1[0])))
    assert_array_equal(res1 == res2, True)

if __name__ == "__main__":
    import nose
    nose.runmodule(argv=[__file__,'-vvs','-x','--pdb'],
                       exit=False)
