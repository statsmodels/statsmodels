from scikits.statsmodels.tools.parallel import parallel_func
from numpy import arange, testing
from math import sqrt

def test_parrallel():
    x = arange(10.)
    parallel, p_func, n_jobs = parallel_func(sqrt, n_jobs=-1, verbose=0)
    y = parallel(p_func(i**2) for i in range(10))
    testing.assert_equal(x,y)
