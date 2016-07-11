from statsmodels.regression.linear_model import OLS
import numpy as np


def calc_nodewise_row(wexog, idx, alpha):
    """calculates the nodewise_row values for the idxth variable, used to
    estimate approx_inv_cov.

    Parameters
    ----------
    wexog : array-like
        The weighted design matrix for the current partition.
    idx : scalar
        Index of the current variable.
    alpha : scalar or array-like
        The penalty weight.  If a scalar, the same penalty weight
        applies to all variables in the model.  If a vector, it
        must have the same length as `params`, and contains a
        penalty weight for each coefficient.

    Returns
    -------
    An array-like object of length p-1

    Notes
    -----

    nodewise_row_i = arg min 1/(2n) ||wexog_i - wexog_-i gamma||_2^2
                             + alpha ||gamma||_1
    """

    p = wexog.shape[1]
    # handle array alphas
    if not np.isscalar(alpha):
        alpha = alpha[ind]

    ind = list(range(p))
    ind.pop(idx)

    tmod = OLS(wexog[:, idx], wexog[:, ind])

    nodewise_row = tmod.fit_regularized(alpha=alpha).params

    return nodewise_row


def calc_nodewise_weight(wexog, nodewise_row, idx, alpha):
    """calculates the nodewise_weightvalue for the idxth variable, used to
    estimate approx_inv_cov.

    Parameters
    ----------
    wexog : array-like
        The weighted design matrix for the current partition.
    nodewise_row : array-like
        The nodewise_row values for the current variable.
    idx : scalar
        Index of the current variable
    alpha : scalar or array-like
        The penalty weight.  If a scalar, the same penalty weight
        applies to all variables in the model.  If a vector, it
        must have the same length as `params`, and contains a
        penalty weight for each coefficient.

    Returns
    -------
    A scalar

    Notes
    -----

    nodewise_weight_i = sqrt(1/n ||wexog,i - wexog_-i nodewise_row||_2^2
                             + alpha ||nodewise_row||_1)
    """

    n, p = wexog.shape
    # handle array alphas
    if not np.isscalar(alpha):
        alpha = alpha[ind]

    ind = list(range(p))
    ind.pop(idx)

    d = np.linalg.norm(wexog[:, idx] - wexog[:, ind].dot(nodewise_row))**2
    d = np.sqrt(d / n + alpha * np.linalg.norm(nodewise_row, 1))
    return d


def calc_approx_inv_cov(nodewise_row_l, nodewise_weight_l):
    """calculates the approximate inverse covariance matrix

    Parameters
    ----------
    nodewise_row_l : list
        A list of array-like object where each object corresponds to
        the nodewise_row values for the corresponding variable, should
        be length p.
    nodewise_weight_l : list
        A list of scalars where each scalar corresponds to the nodewise_weight
        value for the corresponding variable, should be length p.

    Returns
    ------
    An array-like object, p x p matrix

    Notes
    -----

    nwr = nodewise_row
    nww = nodewise_weight

    approx_inv_cov_j = - 1 / nww_j [nwr_j,1,...,1,...nwr_j,p]
    """

    p = len(nodewise_weight_l)

    approx_inv_cov = np.eye(p)
    for idx in range(p):
        ind = list(range(p))
        ind.pop(idx)
        approx_inv_cov[idx,ind] = nodewise_row_l[idx]
    approx_inv_cov *= -1 / nodewise_weight_l[:, None]**2

    return approx_inv_cov
