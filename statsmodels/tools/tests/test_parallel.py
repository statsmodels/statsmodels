from statsmodels.tools.parallel import parallel_func
from statsmodels.tools.sm_exceptions import ModuleUnavailableWarning
from pandas.util.testing import assert_produces_warning
from numpy import arange, testing
from math import sqrt

def test_parallel():
    x = arange(10.)
    with assert_produces_warning(ModuleUnavailableWarning):
        parallel, p_func, n_jobs = parallel_func(sqrt, n_jobs=-1, verbose=0)
        y = parallel(p_func(i**2) for i in range(10))
    testing.assert_equal(x,y)
