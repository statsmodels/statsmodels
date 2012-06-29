import numpy as np
import KernelFunctions as kf
import scipy.optimize as opt
from scipy.integrate import *


kernel_func = dict(wangryzin=kf.WangRyzin, aitchisonaitken=kf.AitchisonAitken,
                 epanechnikov=kf.Epanechnikov, gaussian=kf.Gaussian,
                 gauss_convolution=kf.Gaussian_Convolution,
                 wangryzin_convolution=kf.WangRyzin_Convolution,
                 aitchisonaitken_convolution=kf.AitchisonAitken_Convolution)

Convolution_Kernels = dict(gauss_convolution=kf.Gaussian_Convolution,
                           wangryzin_convolution=kf.WangRyzin_Convolution)


def IntegrateSingle(val, pdf):
    f1 = lambda x: pdf(edat=np.asarray(x))
    return quad(f1, -np.Inf, val)[0]


def IntegrateDbl(val, pdf):
    f2 = lambda x, y: pdf(edat=np.asarray([y, x]))
    return dblquad(f2, -np.Inf, val[0], lambda x: -np.Inf, lambda x: val[1])[0]


class LeaveOneOut(object):
    # Written by Skipper
    """
    Generator to give leave one out views on X

    Parameters
    ----------
    X : array-like
        2d array

    Examples
    --------
    >>> X = np.random.normal(0,1,[10,2])
    >>> loo = LeaveOneOut(X)
    >>> for x in loo:
    ...    print x

    Notes
    -----
    A little lighter weight than sklearn LOO. We don't need test index.
    Also passes views on X, not the index.
    """
    def __init__(self, X):
        self.X = np.asarray(X)

    def __iter__(self):
        X = self.X
        N, K = np.shape(X)

        for i in xrange(N):
            index = np.ones(N, dtype=np.bool)
            index[i] = False
            yield X[index, :]

def GPKE(bw, tdat, edat, var_type, ckertype='gaussian',
         okertype='wangryzin', ukertype='aitchisonaitken'):
    """
    Returns the non-normalized Generalized Product Kernel Estimator

    Parameters
    ----------
    bw: array-like
        The user-specified bandwdith parameters
    tdat: 1D or 2d array
        The training data
    edat: 1d array
        The evaluation points at which the kernel estimation is performed
    var_type: str
        The variable type (continuous, ordered, unordered)
    ckertype: str
        The kernel used for the continuous variables
    okertype: str
        The kernel used for the ordered discrete variables
    ukertype: str
        The kernel used for the unordered discrete variables
    """
    var_type = np.asarray(list(var_type))
    iscontinuous = np.where(var_type == 'c')[0]
    isordered = np.where(var_type == 'o')[0]
    isunordered = np.where(var_type == 'u')[0]
    edat = np.asarray(edat)
    edat = np.squeeze(edat)

##    if tdat.ndim > 1:
##        N, K = np.shape(tdat)
##    else:
##        K = 1
##        N = np.shape(tdat)[0]
##        tdat = tdat.reshape([N, K])
##
##    if edat.ndim > 1:
##        N_edat = np.shape(edat)[0]
##    else:
##        N_edat = 1
##        edat = edat.reshape([N_edat, K])

    K = len(var_type)
    if tdat.ndim == 1 and K == 1:  # one variable many observations
        N = np.size(tdat)
        #N_edat = np.size(edat)
    elif tdat.ndim == 1 and K > 1:
        N = 1

    else:
        N, K = np.shape(tdat)
    tdat = tdat.reshape([N, K])
    if edat.ndim == 1 and K > 1:  # one obs many vars
        N_edat = 1
    elif edat.ndim == 1 and K == 1:  # one obs one var
        N_edat = np.size(edat)

    else:
        N_edat = np.shape(edat)[0]  # ndim >1 so many obs many vars
        assert np.shape(edat)[1] == K

    edat = edat.reshape([N_edat, K])

    bw = np.reshape(np.asarray(bw), (K,))  # must remain 1-D for indexing to work
    dens = np.empty([N_edat, 1])

    for i in xrange(N_edat):

        Kval = np.concatenate((
        kernel_func[ckertype](bw[iscontinuous], tdat[:, iscontinuous], edat[i, iscontinuous]),
        kernel_func[okertype](bw[isordered], tdat[:, isordered], edat[i, isordered]),
        kernel_func[ukertype](bw[isunordered], tdat[:, isunordered], edat[i, isunordered])
        ), axis=1)

        dens[i] = np.sum(np.prod(Kval, axis=1)) * 1. / (np.prod(bw[iscontinuous]))
    return dens




def GPKE3(bw, tdat, edat, var_type, ckertype='gaussian',
          okertype='wangryzin', ukertype='aitchisonaitken'):
    # Rename !

    var_type = np.asarray(list(var_type))
    iscontinuous = np.where(var_type == 'c')[0]
    isordered = np.where(var_type == 'o')[0]
    isunordered = np.where(var_type == 'u')[0]
    K = len(var_type)
    if tdat.ndim == 1 and K == 1:  # one variable many observations
        N = np.size(tdat)
        #N_edat = np.size(edat)
    elif tdat.ndim == 1 and K > 1:
        N = 1

    else:
        N, K = np.shape(tdat)

    tdat = tdat.reshape([N, K])

    if edat.ndim == 1 and K > 1:  # one obs many vars
        N_edat = 1
    elif edat.ndim == 1 and K == 1:  # one obs one var
        N_edat = np.size(edat)
    else:
        N_edat = np.shape(edat)[0]  # ndim >1 so many obs many vars
        assert np.shape(edat)[1] == K

    edat = edat.reshape([N_edat, K])

    bw = np.reshape(np.asarray(bw), (K, ))  # must remain 1-D for indexing to work
    # dens = np.empty([N, 1])
    Kval = np.concatenate((
    kernel_func[ckertype](bw[iscontinuous], tdat[:, iscontinuous], edat[:, iscontinuous]),
    kernel_func[okertype](bw[isordered], tdat[:, isordered], edat[:, isordered]),
    kernel_func[ukertype](bw[isunordered], tdat[:, isunordered], edat[:, isunordered])
    ), axis=1)

    dens = np.prod(Kval, axis=1) * 1. / (np.prod(bw[iscontinuous]))
    return dens
