import os
from statsmodels.datasets import get_rdataset
from numpy.testing import assert_

cur_dir = os.path.dirname(os.path.abspath(__file__))

def test_get_rdataset():
    # smoke test
    duncan = get_rdataset("Duncan", "car", cache=cur_dir)
    assert_(duncan.from_cache)
