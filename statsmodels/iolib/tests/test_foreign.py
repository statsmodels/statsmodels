"""
Tests for iolib/foreign.py
"""
from statsmodels.compat.testing import skipif
import os
import warnings
from datetime import datetime

from numpy.testing import assert_array_equal, assert_, assert_equal, dec
import numpy as np
from pandas import DataFrame, isnull
import pandas.util.testing as ptesting

from statsmodels.compat.python import BytesIO, asbytes
import statsmodels.api as sm
from statsmodels.iolib.foreign import (StataWriter, genfromdta,
            _datetime_to_stata_elapsed, _stata_elapsed_date_to_datetime)
from statsmodels.datasets import macrodata


from statsmodels.compat.pandas import version as pandas_version
pandas_old = pandas_version < '0.9'

# Test precisions
DECIMAL_4 = 4
DECIMAL_3 = 3

curdir = os.path.dirname(os.path.abspath(__file__))


def test_genfromdta():
    #Test genfromdta vs. results/macrodta.npy created with genfromtxt.
    #NOTE: Stata handles data very oddly.  Round tripping from csv to dta
    #    to ndarray 2710.349 (csv) -> 2510.2491 (stata) -> 2710.34912109375
    #    (dta/ndarray)
    from .results.macrodata import macrodata_result as res2
    res1 = genfromdta(curdir+'/../../datasets/macrodata/macrodata.dta')
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
    buf = BytesIO()
    dta = macrodata.load().data
    dtype = dta.dtype
    dta = dta.astype(np.dtype([('year', int),
                               ('quarter', int)] + dtype.descr[2:]))
    writer = StataWriter(buf, dta)
    writer.write_file()
    buf.seek(0)
    dta2 = genfromdta(buf)
    assert_array_equal(dta, dta2)

def test_stata_writer_array():
    buf = BytesIO()
    dta = macrodata.load().data
    dta = DataFrame.from_records(dta)
    dta.columns = ["v%d" % i for i in range(1,15)]
    writer = StataWriter(buf, dta.values)
    writer.write_file()
    buf.seek(0)
    dta2 = genfromdta(buf)
    dta = dta.to_records(index=False)
    assert_array_equal(dta, dta2)

def test_missing_roundtrip():
    buf = BytesIO()
    dta = np.array([(np.nan, np.inf, "")],
                      dtype=[("double_miss", float), ("float_miss", np.float32),
                              ("string_miss", "a1")])
    writer = StataWriter(buf, dta)
    writer.write_file()
    buf.seek(0)
    dta = genfromdta(buf, missing_flt=np.nan)
    assert_(isnull(dta[0][0]))
    assert_(isnull(dta[0][1]))
    assert_(dta[0][2] == asbytes(""))

    dta = genfromdta(os.path.join(curdir, "results/data_missing.dta"),
            missing_flt=-999)
    assert_(np.all([dta[0][i] == -999 for i in range(5)]))

def test_stata_writer_pandas():
    buf = BytesIO()
    dta = macrodata.load().data
    dtype = dta.dtype
    #as of 0.9.0 pandas only supports i8 and f8
    dta = dta.astype(np.dtype([('year', 'i8'),
                               ('quarter', 'i8')] + dtype.descr[2:]))
    dta4 = dta.astype(np.dtype([('year', 'i4'),
                               ('quarter', 'i4')] + dtype.descr[2:]))
    dta = DataFrame.from_records(dta)
    dta4 = DataFrame.from_records(dta4)
    # dta is int64 'i8'  given to Stata writer
    writer = StataWriter(buf, dta)
    writer.write_file()
    buf.seek(0)
    dta2 = genfromdta(buf)
    dta5 = DataFrame.from_records(dta2)
    # dta2 is int32 'i4'  returned from Stata reader

    if dta5.dtypes[1] is np.dtype('int64'):
        ptesting.assert_frame_equal(dta.reset_index(), dta5)
    else:
        # don't check index because it has different size, int32 versus int64
        ptesting.assert_frame_equal(dta4, dta5[dta5.columns[1:]])

def test_stata_writer_unicode():
    # make sure to test with characters outside the latin-1 encoding
    pass

def test_genfromdta_datetime():
    results = [(datetime(2006, 11, 19, 23, 13, 20), 1479596223000,
            datetime(2010, 1, 20), datetime(2010, 1, 8), datetime(2010, 1, 1),
            datetime(1974, 7, 1), datetime(2010, 1, 1), datetime(2010, 1, 1)),
        (datetime(1959, 12, 31, 20, 3, 20), -1479590, datetime(1953, 10, 2),
            datetime(1948, 6, 10), datetime(1955, 1, 1), datetime(1955, 7, 1),
            datetime(1955, 1, 1), datetime(2, 1, 1))]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        dta = genfromdta(os.path.join(curdir, "results/time_series_examples.dta"))
        assert_(len(w) > 0)  # should get a warning for that format.

    assert_array_equal(dta[0].tolist(), results[0])
    assert_array_equal(dta[1].tolist(), results[1])

    with warnings.catch_warnings(record=True):
        dta = genfromdta(os.path.join(curdir, "results/time_series_examples.dta"),
                         pandas=True)

    assert_array_equal(dta.iloc[0].tolist(), results[0])
    assert_array_equal(dta.iloc[1].tolist(), results[1])

def test_date_converters():
    ms = [-1479597200000, -1e6, -1e5, -100, 1e5, 1e6, 1479597200000]
    days = [-1e5, -1200, -800, -365, -50, 0, 50, 365, 800, 1200, 1e5]
    weeks = [-1e4, -1e2, -53, -52, -51, 0, 51, 52, 53, 1e2, 1e4]
    months = [-1e4, -1e3, -100, -13, -12, -11, 0, 11, 12, 13, 100, 1e3, 1e4]
    quarter = [-100, -50, -5, -4, -3, 0, 3, 4, 5, 50, 100]
    half = [-50, 40, 30, 10, 3, 2, 1, 0, 1, 2, 3, 10, 30, 40, 50]
    year = [1, 50, 500, 1000, 1500, 1975, 2075]
    for i in ms:
        assert_equal(_datetime_to_stata_elapsed(
                     _stata_elapsed_date_to_datetime(i, "tc"), "tc"), i)
    for i in days:
        assert_equal(_datetime_to_stata_elapsed(
                     _stata_elapsed_date_to_datetime(i, "td"), "td"), i)
    for i in weeks:
        assert_equal(_datetime_to_stata_elapsed(
                     _stata_elapsed_date_to_datetime(i, "tw"), "tw"), i)
    for i in months:
        assert_equal(_datetime_to_stata_elapsed(
                     _stata_elapsed_date_to_datetime(i, "tm"), "tm"), i)
    for i in quarter:
        assert_equal(_datetime_to_stata_elapsed(
                     _stata_elapsed_date_to_datetime(i, "tq"), "tq"), i)
    for i in half:
        assert_equal(_datetime_to_stata_elapsed(
                     _stata_elapsed_date_to_datetime(i, "th"), "th"), i)
    for i in year:
        assert_equal(_datetime_to_stata_elapsed(
                     _stata_elapsed_date_to_datetime(i, "ty"), "ty"), i)

@skipif(pandas_old, 'pandas too old')
def test_datetime_roundtrip():
    dta = np.array([(1, datetime(2010, 1, 1), 2),
                    (2, datetime(2010, 2, 1), 3),
                    (4, datetime(2010, 3, 1), 5)],
                    dtype=[('var1', float), ('var2', object), ('var3', float)])
    buf = BytesIO()
    writer = StataWriter(buf, dta, {"var2" : "tm"})
    writer.write_file()
    buf.seek(0)
    dta2 = genfromdta(buf)
    assert_equal(dta, dta2)

    dta = DataFrame.from_records(dta)
    buf = BytesIO()
    writer = StataWriter(buf, dta, {"var2" : "tm"})
    writer.write_file()
    buf.seek(0)
    dta2 = genfromdta(buf, pandas=True)
    ptesting.assert_frame_equal(dta, dta2.drop('index', axis=1))


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, '-vvs', '-x', '--pdb'])
