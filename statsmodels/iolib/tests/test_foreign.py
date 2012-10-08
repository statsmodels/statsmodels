"""
Tests for iolib/foreign.py
"""

from StringIO import StringIO
from numpy.testing import *
import numpy as np
import statsmodels.api as sm
import os
from statsmodels.iolib.foreign import StataWriter, genfromdta
from statsmodels.datasets import macrodata
from pandas import DataFrame
import pandas.util.testing as ptesting

# Test precisions
DECIMAL_4 = 4
DECIMAL_3 = 3


def test_genfromdta():
    #Test genfromdta vs. results/macrodta.npy created with genfromtxt.
    #NOTE: Stata handles data very oddly.  Round tripping from csv to dta
    #    to ndarray 2710.349 (csv) -> 2510.2491 (stata) -> 2710.34912109375
    #    (dta/ndarray)
    curdir = os.path.dirname(os.path.abspath(__file__))
    #res2 = np.load(curdir+'/results/macrodata.npy')
    #res2 = res2.view((float,len(res2[0])))
    from results.macrodata import macrodata_result as res2
    res1 = genfromdta(curdir+'/../../datasets/macrodata/macrodata.dta')
    #res1 = res1.view((float,len(res1[0])))
    assert_array_equal(res1 == res2, True)

def test_genfromdta_pandas():
    from pandas.util.testing import assert_frame_equal
    dta = macrodata.load_pandas().data
    curdir = os.path.dirname(os.path.abspath(__file__))
    res1 = sm.iolib.genfromdta(curdir+'/../../datasets/macrodata/macrodata.dta',
                        pandas=True)
    res1 = res1.astype(float)
    assert_frame_equal(res1, dta)

def test_stata_writer_structured():
    buf = StringIO()
    dta = macrodata.load().data
    dtype = dta.dtype
    dta = dta.astype(np.dtype([('year', 'i8'),
                               ('quarter', 'i4')] + dtype.descr[2:]))
    writer = StataWriter(buf, dta)
    writer.write_file()
    buf.seek(0)
    dta2 = genfromdta(buf)
    assert_array_equal(dta, dta2)

def test_stata_writer_array():
    buf = StringIO()
    dta = macrodata.load().data
    dta = DataFrame.from_records(dta)
    dta.columns = ["v%d" % i for i in range(1,15)]
    writer = StataWriter(buf, dta.values)
    writer.write_file()
    buf.seek(0)
    dta2 = genfromdta(buf)
    dta = dta.to_records(index=False)
    assert_array_equal(dta, dta2)

def test_stata_writer_missing():
    pass

def test_stata_writer_pandas():
    buf = StringIO()
    dta = macrodata.load().data
    dtype = dta.dtype
    #as of 0.9.0 pandas only supports i8 and f8
    dta = dta.astype(np.dtype([('year', 'i8'),
                               ('quarter', 'i8')] + dtype.descr[2:]))
    dta = DataFrame.from_records(dta)
    writer = StataWriter(buf, dta)
    writer.write_file()
    buf.seek(0)
    dta2 = genfromdta(buf)
    ptesting.assert_frame_equal(dta.reset_index(), DataFrame.from_records(dta2))


def test_stata_writer_unicode():
    # make sure to test with characters outside the latin-1 encoding
    pass

def test_stata_writer_datetime():
    pass


if __name__ == "__main__":
    import nose
    nose.runmodule(argv=[__file__,'-vvs','-x','--pdb'],
                       exit=False)
