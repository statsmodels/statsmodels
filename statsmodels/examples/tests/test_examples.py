import pytest

from statsmodels.examples.run_all import filelist, run_example


@pytest.mark.parametrize('path', filelist)
def test_example(path):
    # TODO: re-write so that we get a useful exception message on failure
    rc = run_example(path)
    assert rc == 0
