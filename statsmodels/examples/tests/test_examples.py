import os

import pytest

from statsmodels.examples.run_all import filelist, run_example

validated = [x for x in filelist
             if os.path.split(x)[-1] in ["try_tukey_hsd.py"]]


@pytest.mark.smoke
@pytest.mark.parametrize('path', validated)
def test_example(path):
    # TODO: re-write so that we get a useful exception message on failure
    rc = run_example(path)
    assert rc == 0
