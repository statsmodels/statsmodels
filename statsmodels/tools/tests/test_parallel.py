from math import sqrt
import warnings

from numpy import arange, testing
import pytest

from statsmodels.tools.parallel import parallel_func


@pytest.mark.thread_unsafe(reason="uses joblib which is not thread safe")
def test_parallel():
    x = arange(10.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parallel, p_func, n_jobs = parallel_func(sqrt, n_jobs=-1, verbose=0)
        y = parallel(p_func(i**2) for i in range(10))
    testing.assert_equal(x, y)
