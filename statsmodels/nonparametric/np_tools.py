import numpy as np

from . import kernels


kernel_func = dict(wangryzin=kernels.WangRyzin,
                   aitchisonaitken=kernels.AitchisonAitken,
                   gaussian=kernels.Gaussian,
                   gauss_convolution=kernels.Gaussian_Convolution,
                   wangryzin_convolution=kernels.WangRyzin_Convolution,
                   aitchisonaitken_convolution=kernels.AitchisonAitken_Convolution,
                   gaussian_cdf=kernels.Gaussian_cdf,
                   aitchisonaitken_cdf=kernels.AitchisonAitken_cdf,
                   wangryzin_cdf=kernels.WangRyzin_cdf)


class LeaveOneOut(object):
    """
    Generator to give leave-one-out views on X.

    Parameters
    ----------
    X : array-like
        2-D array.

    Examples
    --------
    >>> X = np.random.normal(0, 1, [10,2])
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


def adjust_shape(dat, K):
    """ Returns NxK array so that it can be used with gpke()."""
    dat = np.asarray(dat)
    if dat.ndim > 2:
        dat = np.squeeze(dat)
    if dat.ndim == 1 and K > 1:  # one obs many vars
        N = 1
    elif dat.ndim == 1 and K == 1:  # one obs one var
        N = len(dat)
    else:
        if np.shape(dat)[0] == K and np.shape(dat)[1] != K:
            dat = dat.T
        N = np.shape(dat)[0]  # ndim >1 so many obs many vars
        assert np.shape(dat)[1] == K

    dat = np.reshape(dat, (N, K))
    return dat


def gpke(bw, tdat, edat, var_type, ckertype='gaussian',
         okertype='wangryzin', ukertype='aitchisonaitken', tosum=True):
    """
    Returns the non-normalized Generalized Product Kernel Estimator

    Parameters
    ----------
    bw: array-like
        The user-specified bandwidth parameters.
    tdat: 1D or 2-D ndarray
        The training data.
    edat: 1-D ndarray
        The evaluation points at which the kernel estimation is performed.
    var_type: str, optional
        The variable type (continuous, ordered, unordered).
    ckertype: str, optional
        The kernel used for the continuous variables.
    okertype: str, optional
        The kernel used for the ordered discrete variables.
    ukertype: str, optional
        The kernel used for the unordered discrete variables.
    tosum : bool, optional
        Whether or not to sum the calculated array of densities.  Default is
        True.

    Returns
    -------
    dens: array-like
        The generalized product kernel density estimator.

    Notes
    -----
    The formula for the multivariate kernel estimator for the pdf is:

    .. math:: f(x)=\frac{1}{nh_{1}...h_{q}}\sum_{i=1}^
                        {n}K\left(\frac{X_{i}-x}{h}\right)

    where

    .. math:: K\left(\frac{X_{i}-x}{h}\right) =
                k\left( \frac{X_{i1}-x_{1}}{h_{1}}\right)\times
                k\left( \frac{X_{i2}-x_{2}}{h_{2}}\right)\times...\times
                k\left(\frac{X_{iq}-x_{q}}{h_{q}}\right)
    """
    iscontinuous, isordered, isunordered = _get_type_pos(var_type)
    K = len(var_type)
    N = np.shape(tdat)[0]
    # must remain 1-D for indexing to work
    bw = np.reshape(np.asarray(bw), (K,))
    Kval = np.concatenate((
        kernel_func[ckertype](bw[iscontinuous],
                            tdat[:, iscontinuous], edat[:, iscontinuous]),
        kernel_func[okertype](bw[isordered], tdat[:, isordered],
                            edat[:, isordered]),
        kernel_func[ukertype](bw[isunordered], tdat[:, isunordered],
                            edat[:, isunordered])), axis=1)

    dens = np.prod(Kval, axis=1) * 1. / (np.prod(bw[iscontinuous]))
    if tosum:
        return np.sum(dens, axis=0)
    else:
        return dens
