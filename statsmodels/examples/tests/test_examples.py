import pytest

from statsmodels.examples.run_all import filelist, run_example

test_files = [x for x in filelist]
for n, path in enumerate(test_files):
    if 'kernel' in path:
        # In local runs totalling 2015.08 seconds, the slowest files are:
        #  ex_kernel_regression_sigtest         752.50s
        #  ex_kernel_test_functional_li_wang    749.86s
        #  ex_kernel_regression3                117.04s
        #  ex_emplike_3                         50.02s
        #  ex_kernel_semilinear_dgp             44.30s
        #  ex_kernel_test_functional            42.97s
        #  ex_emplike_1                         36.73s
        #  ex_kernel_singleindex_dgp            27.73s
        # All others were below 20 seconds, most of them single-digit.
        param = pytest.param(path, marks=pytest.mark.slow)
        test_files[n] = param
    elif 'koul_and_mc' in path:
        param = pytest.param(
            path,
            marks=pytest.mark.xfail(reason="rverify.csv file does not exist",
                                    raises=IOError, strict=True))
        test_files[n] = param
    elif 'ex_sandwich2' in path:
        param = pytest.param(
            path,
            marks=pytest.mark.xfail(reason="Invalid stata file",
                                    raises=ValueError, strict=True))
        test_files[n] = param


@pytest.mark.parametrize('path', test_files)
def test_example(path):
    # TODO: re-write so that we get a useful exception message on failure
    rc = run_example(path)
    assert rc == 0
