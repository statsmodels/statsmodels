import os
import numpy as np
from numpy.testing import assert_, assert_array_equal
import pytest

from statsmodels.datasets import get_rdataset, webuse, check_internet, utils

cur_dir = os.path.dirname(os.path.abspath(__file__))


def test_get_rdataset():
    # smoke test
    test_url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/cars.csv"
    internet_available = check_internet(test_url)
    if not internet_available:
        pytest.skip('Unable to retrieve file - skipping test')
    duncan = get_rdataset("Duncan", "carData", cache=cur_dir)
    assert_(isinstance(duncan, utils.Dataset))
    duncan = get_rdataset("Duncan", "carData", cache=cur_dir)
    assert_(duncan.from_cache)

    # test writing and reading cache
    guerry = get_rdataset("Guerry", "HistData", cache=cur_dir)
    assert_(guerry.from_cache is False)
    guerry2 = get_rdataset("Guerry", "HistData", cache=cur_dir)
    assert_(guerry2.from_cache is True)
    fn = "raw.githubusercontent.com,vincentarelbundock,Rdatasets,master,csv,HistData,Guerry.csv.zip"
    os.remove(os.path.join(cur_dir, fn))
    fn = "raw.githubusercontent.com,vincentarelbundock,Rdatasets,master,doc,HistData,rst,Guerry.rst.zip"
    os.remove(os.path.join(cur_dir, fn))


def test_webuse():
    # test copied and adjusted from iolib/tests/test_foreign
    from statsmodels.iolib.tests.results.macrodata import macrodata_result as res2
    res2 = np.array([list(row) for row in res2])
    base_gh = "https://github.com/statsmodels/statsmodels/raw/master/statsmodels/datasets/macrodata/"
    internet_available = check_internet(base_gh)
    if not internet_available:
        pytest.skip('Unable to retrieve file - skipping test')
    res1 = webuse('macrodata', baseurl=base_gh, as_df=False)
    assert_array_equal(res1, res2)


def test_webuse_pandas():
    # test copied and adjusted from iolib/tests/test_foreign
    from pandas.util.testing import assert_frame_equal
    from statsmodels.datasets import macrodata
    dta = macrodata.load_pandas().data
    base_gh = "https://github.com/statsmodels/statsmodels/raw/master/statsmodels/datasets/macrodata/"
    internet_available = check_internet(base_gh)
    if not internet_available:
        pytest.skip('Unable to retrieve file - skipping test')
    res1 = webuse('macrodata', baseurl=base_gh)
    res1 = res1.astype(float)
    assert_frame_equal(res1, dta.astype(float))
