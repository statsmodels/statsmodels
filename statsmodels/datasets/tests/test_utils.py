import os
import sys
from statsmodels.datasets import get_rdataset, webuse, check_internet
from numpy.testing import assert_, assert_array_equal, dec

cur_dir = os.path.dirname(os.path.abspath(__file__))

def test_get_rdataset():
    # smoke test
    if sys.version_info[0] >= 3:
        #NOTE: there's no way to test both since the cached files were
        #created with Python 2.x, they're strings, but Python 3 expects
        #bytes and the index file path is hard-coded so both can't live
        #side by side
        pass
        #duncan = get_rdataset("Duncan-py3", "car", cache=cur_dir)
    else:
        duncan = get_rdataset("Duncan", "car", cache=cur_dir)
        assert_(duncan.from_cache)

#internet_available = check_internet()
#@dec.skipif(not internet_available)
def t_est_webuse():
    # test copied and adjusted from iolib/tests/test_foreign
    from statsmodels.iolib.tests.results.macrodata import macrodata_result as res2
    #base_gh = "http://github.com/statsmodels/statsmodels/raw/master/statsmodels/datasets/macrodata/"
    base_gh = "http://statsmodels.sourceforge.net/devel/_static/"
    res1 = webuse('macrodata', baseurl=base_gh, as_df=False)
    assert_array_equal(res1 == res2, True)

#@dec.skipif(not internet_available)
def t_est_webuse_pandas():
    # test copied and adjusted from iolib/tests/test_foreign
    from pandas.util.testing import assert_frame_equal
    from statsmodels.datasets import macrodata
    dta = macrodata.load_pandas().data
    base_gh = "http://github.com/statsmodels/statsmodels/raw/master/statsmodels/datasets/macrodata/"
    res1 = webuse('macrodata', baseurl=base_gh)
    res1 = res1.astype(float)
    assert_frame_equal(res1, dta)
