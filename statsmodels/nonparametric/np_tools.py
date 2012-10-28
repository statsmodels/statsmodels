import numpy as np
import kernels as kernels


kernel_func = dict(wangryzin=kernels.wang_ryzin,
                   aitchisonaitken=kernels.aitchison_aitken,
                   gaussian=kernels.gaussian,
                   aitchison_aitken_reg = kernels.aitchison_aitken_reg,
                   wangryzin_reg = kernels.wang_ryzin_reg,
                   gauss_convolution=kernels.gaussian_convolution,
                   wangryzin_convolution=kernels.wang_ryzin_convolution,
                   aitchisonaitken_convolution=kernels.aitchison_aitken_convolution,
                   gaussian_cdf=kernels.gaussian_cdf,
                   aitchisonaitken_cdf=kernels.aitchison_aitken_cdf,
                   wangryzin_cdf=kernels.wang_ryzin_cdf,
                   d_gaussian=kernels.d_gaussian)


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
        nobs, K = np.shape(X)

        for i in xrange(nobs):
            index = np.ones(nobs, dtype=np.bool)
            index[i] = False
            yield X[index, :]


def _get_type_pos(var_type):
    ix_cont = np.array([c == 'c' for c in var_type])
    ix_ord = np.array([c == 'o' for c in var_type])
    ix_unord = np.array([c == 'u' for c in var_type])
    return ix_cont, ix_ord, ix_unord


def adjust_shape(dat, K):
    """ Returns an array of shape (nobs, K) for use with `gpke`."""
    dat = np.asarray(dat)
    if dat.ndim > 2:
        dat = np.squeeze(dat)
    if dat.ndim == 1 and K > 1:  # one obs many vars
        nobs = 1
    elif dat.ndim == 1 and K == 1:  # one obs one var
        nobs = len(dat)
    else:
        if np.shape(dat)[0] == K and np.shape(dat)[1] != K:
            dat = dat.T

        nobs = np.shape(dat)[0]  # ndim >1 so many obs many vars

    dat = np.reshape(dat, (nobs, K))
    return dat


def gpke(bw, data, data_predict, var_type, ckertype='gaussian',
         okertype='wangryzin', ukertype='aitchisonaitken', tosum=True):
    """
    Returns the non-normalized Generalized Product Kernel Estimator

    Parameters
    ----------
    bw: 1-D ndarray
        The user-specified bandwidth parameters.
    data: 1D or 2-D ndarray
        The training data.
    data_predict: 1-D ndarray
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
    kertypes = dict(c=ckertype, o=okertype, u=ukertype)
    #Kval = []
    #for ii, vtype in enumerate(var_type):
    #    func = kernel_func[kertypes[vtype]]
    #    Kval.append(func(bw[ii], data[:, ii], data_predict[ii]))

    #Kval = np.column_stack(Kval)

    Kval = np.empty(data.shape)
    for ii, vtype in enumerate(var_type):
        func = kernel_func[kertypes[vtype]]
        Kval[:, ii] = func(bw[ii], data[:, ii], data_predict[ii])

    iscontinuous = np.array([c == 'c' for c in var_type])
    dens = Kval.prod(axis=1) / np.prod(bw[iscontinuous])
    if tosum:
        return dens.sum(axis=0)
    else:
        return dens

