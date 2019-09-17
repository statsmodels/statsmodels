"""
Examples corresponding to sandbox.stats.multicomp
"""
import numpy as np
from scipy import stats

from statsmodels.compat.python import lzip
from statsmodels.iolib.table import SimpleTable
from statsmodels.stats.multitest import multipletests, _ecdf as ecdf

from statsmodels.sandbox.stats.multicomp import (
    maxzero, maxzerodown, fdrcorrection_bak
)


def example_fdr_bonferroni():
    x1 = [1, 1, 1, 0, -1, -1, -1, 0, 1, 1, -1, 1]
    print(lzip(np.arange(len(x1)), x1))
    print(maxzero(x1))
    # Expected output from these last two prints:
    # [(0, 1), (1, 1), (2, 1), (3, 0), (4, -1), (5, -1), (6, -1), (7, 0), \
    #  (8, 1), (9, 1), (10, -1), (11, 1)]
    # (11, array([ 3,  7, 11]))

    print(maxzerodown(-np.array(x1)))

    locs = np.linspace(0, 1, 10)
    locs = np.array([0.]*6 + [0.75]*4)
    rvs = locs + stats.norm.rvs(size=(20, 10))
    tt, tpval = stats.ttest_1samp(rvs, 0)
    tpval_sortind = np.argsort(tpval)
    tpval_sorted = tpval[tpval_sortind]

    reject = tpval_sorted < ecdf(tpval_sorted)*0.05
    reject2 = max(np.nonzero(reject))
    print(reject)
    print(reject2)

    res = np.array(lzip(np.round(rvs.mean(0), 4),
                        np.round(tpval, 4),
                        reject[tpval_sortind.argsort()]),
                   dtype=[('mean', float),
                          ('pval', float),
                          ('reject', np.bool8)])
    print(SimpleTable(res, headers=res.dtype.names))
    print(fdrcorrection_bak(tpval, alpha=0.05))
    print(reject)

    print('\nrandom example')
    print('bonf', multipletests(tpval, alpha=0.05, method='bonf'))
    print('sidak', multipletests(tpval, alpha=0.05, method='sidak'))
    print('hs', multipletests(tpval, alpha=0.05, method='hs'))
    print('sh', multipletests(tpval, alpha=0.05, method='sh'))
    pvals = np.array([0.002, 0.0045, 0.006, 0.008,  0.0085,
                      0.009, 0.0175, 0.025, 0.1055, 0.535])
    print('\nexample from lecture notes')
    for meth in ['bonf', 'sidak', 'hs', 'sh']:
        print(meth)
        print(multipletests(pvals, alpha=0.05, method=meth))
