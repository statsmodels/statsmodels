from __future__ import print_function
import numpy as np
from statsmodels.compat.numpy import np_matrix_rank


def add_indep(x, varnames, dtype=None):
    '''
    construct array with independent columns

    x is either iterable (list, tuple) or instance of ndarray or a subclass of it.
    If x is an ndarray, then each column is assumed to represent a variable with
    observations in rows.
    '''
    #TODO: this needs tests for subclasses

    if isinstance(x, np.ndarray) and x.ndim == 2:
        x = x.T

    nvars_orig = len(x)
    nobs = len(x[0])
    #print('nobs, nvars_orig', nobs, nvars_orig)
    if not dtype:
        dtype = np.asarray(x[0]).dtype
    xout = np.zeros((nobs, nvars_orig), dtype=dtype)
    count = 0
    rank_old = 0
    varnames_new = []
    varnames_dropped = []
    keepindx = []
    for (xi, ni) in zip(x, varnames):
        #print(xi.shape, xout.shape)
        xout[:,count] = xi
        rank_new = np_matrix_rank(xout)
        #print(rank_new)
        if  rank_new > rank_old:
            varnames_new.append(ni)
            rank_old = rank_new
            count += 1
        else:
            varnames_dropped.append(ni)

    return xout[:,:count], varnames_new

if __name__ == '__main__':
    x1 = np.array([0,0,0,0,0,1,1,1,2,2,2])
    x2 = np.array([0,0,0,0,0,1,1,1,1,1,1])
    x0 = np.ones(len(x2))
    x = np.column_stack([x0, x1[:,None]*np.arange(3), x2[:,None]*np.arange(2)])
    varnames = ['const'] + ['var1_%d' %i for i in np.arange(3)] \
                         + ['var2_%d' %i for i in np.arange(2)]
    xo,vo = add_indep(x, varnames)
    print(xo.shape)





