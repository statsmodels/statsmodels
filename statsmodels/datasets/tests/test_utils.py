import os
import sys
from statsmodels.datasets import get_rdataset
from numpy.testing import assert_

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
