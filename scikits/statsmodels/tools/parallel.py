"""Parralle util function
"""

# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: Simplified BSD


def parallel_func(func, n_jobs, verbose=5):
    """Return parallel instance with delayed function

    Util function to use joblib only if available

    Parameters
    ----------
    func: callable
        A function
    n_jobs: int
        Number of jobs to run in parallel
    verbose: int
        Verbosity level

    Returns
    -------
    parallel: instance of joblib.Parallel or list
        The parallel object
    my_func: callable
        func if not parallel or delayed(func)
    n_jobs: int
        Number of jobs >= 0
    """
    try:
        from sklearn.externals.joblib import Parallel, delayed
        parallel = Parallel(n_jobs, verbose=verbose)
        my_func = delayed(func)

        if n_jobs == -1:
            try:
                import multiprocessing
                n_jobs = multiprocessing.cpu_count()
            except ImportError:
                print "multiprocessing not installed. Cannot run in parallel."
                n_jobs = 1

    except ImportError:
        print "joblib not installed. Cannot run in parallel."
        n_jobs = 1
        my_func = func
        parallel = list
    return parallel, my_func, n_jobs
