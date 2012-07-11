import numpy as np
import KernelFunctions as kf
import scipy.optimize as opt
from scipy.integrate import *


kernel_func = dict(wangryzin=kf.WangRyzin, aitchisonaitken=kf.AitchisonAitken,
                 epanechnikov=kf.Epanechnikov, gaussian=kf.Gaussian,
                 gauss_convolution=kf.Gaussian_Convolution,
                 wangryzin_convolution=kf.WangRyzin_Convolution,
                 aitchisonaitken_convolution=kf.AitchisonAitken_Convolution,
                   gaussian_cdf=kf.Gaussian_cdf,
                   aitchisonaitken_cdf=kf.AitchisonAitken_cdf,
                   wangryzin_cdf=kf.WangRyzin_cdf)

Convolution_Kernels = dict(gauss_convolution=kf.Gaussian_Convolution,
                           wangryzin_convolution=kf.WangRyzin_Convolution)

  

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
            
def _get_type_pos(var_type):
    var_type = np.asarray(list(var_type))
    iscontinuous = np.where(var_type == 'c')[0]
    isordered = np.where(var_type == 'o')[0]
    isunordered = np.where(var_type == 'u')[0]
    return iscontinuous, isordered, isunordered

def _get_shape(tdat, edat, var_type):
    var_type = np.asarray(list(var_type))
    K = len(var_type)
    if tdat.ndim == 1 and K == 1:  # one variable many observations
        N = np.size(tdat)
    elif tdat.ndim == 1 and K > 1:
        N = 1
    else:
        N, k = np.shape(tdat)
        assert K == k

    if edat.ndim == 1 and K > 1:  # one obs many vars
        N_edat = 1
    elif edat.ndim == 1 and K == 1:  # one obs one var
        N_edat = np.size(edat)
    else:
        N_edat = np.shape(edat)[0]  # ndim >1 so many obs many vars
        assert np.shape(edat)[1] == K
    return K, N, N_edat
    

    
def GPKE(bw, tdat, edat, var_type, ckertype='gaussian',
         okertype='wangryzin', ukertype='aitchisonaitken', tosum=True):
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

    Returns
    -------
    dens: array-like
        The dens estimator

    Notes
    -----

    The formula for the multivariate kernel estimator for the pdf is:

    .. math:: f(x)=\frac{1}{nh_{1}...h_{q}}\sum_{i=1}^{n}K\left(\frac{X_{i}-x}{h}\right)

    where
    
    .. math:: K\left(\frac{X_{i}-x}{h}\right)=k\left(\frac{X_{i1}-x_{1}}{h_{1}}\right)\times k\left(\frac{X_{i2}-x_{2}}{h_{2}}\right)\times...\times k\left(\frac{X_{iq}-x_{q}}{h_{q}}\right)
    
    """
    iscontinuous, isordered, isunordered = _get_type_pos(var_type)
    tdat = np.asarray(tdat)
    edat = np.asarray(edat)

    K, N, N_edat = _get_shape(tdat, edat, var_type)

    tdat = tdat.reshape([N, K])
    edat = edat.reshape([N_edat, K])

    bw = np.reshape(np.asarray(bw), (K,))  # must remain 1-D for indexing to work
    dens = np.empty([N, N_edat])

    for i in xrange(N_edat):

        Kval = np.concatenate((
        kernel_func[ckertype](bw[iscontinuous], tdat[:, iscontinuous], edat[i, iscontinuous]),
        kernel_func[okertype](bw[isordered], tdat[:, isordered], edat[i, isordered]),
        kernel_func[ukertype](bw[isunordered], tdat[:, isunordered], edat[i, isunordered])
        ), axis=1)

        dens[:,i] = np.prod(Kval, axis=1) * 1. / (np.prod(bw[iscontinuous]))
    if tosum:
        return np.sum(dens, axis=0)
    else:
        return dens

def PKE(bw, tdat, edat, var_type, ckertype='gaussian',
          okertype='wangryzin', ukertype='aitchisonaitken'):
    """
    The product of appropriate kernels

    Used in the calculation of IMSE for the conditional KDE

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

    Returns
    -------
    dens: array-like
        The dens estimator
    Notes
    -----
    This is similar to GPKE but adapted specifically for the CKDE
    cross-validation least squares. It also doesn't sum across the kernel
    values
    """
    # TODO: See if you can substitute instead of the GPKE method
    iscontinuous, isordered, isunordered = _get_type_pos(var_type)
    tdat = np.asarray(tdat)
    edat = np.asarray(edat)

    K, N, N_edat = _get_shape(tdat, edat, var_type)

    tdat = tdat.reshape([N, K])
    edat = edat.reshape([N_edat, K])

    bw = np.reshape(np.asarray(bw), (K,))  # must remain 1-D for indexing to work

    Kval = np.concatenate((
    kernel_func[ckertype](bw[iscontinuous], tdat[:, iscontinuous], edat[:, iscontinuous]),
    kernel_func[okertype](bw[isordered], tdat[:, isordered], edat[:, isordered]),
    kernel_func[ukertype](bw[isunordered], tdat[:, isunordered], edat[:, isunordered])
    ), axis=1)

    dens = np.prod(Kval, axis=1) * 1. / (np.prod(bw[iscontinuous]))
    return dens

