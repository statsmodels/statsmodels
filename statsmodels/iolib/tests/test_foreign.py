"""
Tests for iolib/foreign.py
"""

from numpy.testing import *
import numpy as np
import statsmodels.api as sm
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

def test_genfromdta_pandas():
    from pandas.util.testing import assert_frame_equal
    dta = sm.datasets.macrodata.load_pandas().data
    curdir = os.path.dirname(os.path.abspath(__file__))
    res1 = sm.iolib.genfromdta(curdir+'/../../datasets/macrodata/macrodata.dta',
                        pandas=True)
    res1 = res1.astype(float)
    assert_frame_equal(res1, dta)


if __name__ == "__main__":
    import nose
    nose.runmodule(argv=[__file__,'-vvs','-x','--pdb'],
                       exit=False)
