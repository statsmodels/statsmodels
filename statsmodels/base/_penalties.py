"""
A collection of smooth penalty functions.

Penalties on vectors take a vector argument and return a scalar
penalty.  The gradient of the penalty is a vector with the same shape
as the input value.

Penalties on covariance matrices take two arguments: the matrix and
its inverse, both in unpacked (square) form.  The returned penalty is
a scalar, and the gradient is returned as a vector that contains the
gradient with respect to the free elements in the lower triangle of
the covariance matrix.

All penalties are subtracted from the log-likelihood, so greater
penalty values correspond to a greater degree of penalization.

The penaties should be smooth so that they can be subtracted from log
likelihood functions and optimized using standard methods (i.e. L1
penalties do not belong here).
"""

import numpy as np

class Penalty(object):
    """
    A class for representing a scalar-value penalty.

    Parameters
    ----------
    wts : array-like
        A vector of weights that determines the weight of the penalty
        for each parameter.


    Notes
    -----
    The class has a member called `alpha` that scales the weights.
    """

    def __init__(self, wts):
        self.wts = wts
        self.alpha = 1.

    def func(self, params):
        """
        A penalty function on a vector of parameters.

        Parameters
        ----------
        params : array-like
            A vector of parameters.

        Returns
        -------
        A scalar penaty value; greater values imply greater
        penalization.
        """
        raise NotImplementedError

    def grad(self, params):
        """
        The gradient of a penalty function.

        Parameters
        ----------
        params : array-like
            A vector of parameters

        Returns
        -------
        The gradient of the penalty with respect to each element in
        `params`.
        """
        raise NotImplementedError


class L2(Penalty):
    """
    The L2 (ridge) penalty.
    """

    def __init__(self, wts=None):
        if wts is None:
            self.wts = 1.
        else:
            self.wts = wts
        self.alpha = 1.

    def func(self, params):
        return np.sum(self.wts * self.alpha * params**2)

    def grad(self, params):
        return 2 * self.wts * self.alpha * params


class PseudoHuber(Penalty):
    """
    The pseudo-Huber penalty.
    """

    def __init__(self, dlt, wts=None):
        self.dlt = dlt
        if wts is None:
            self.wts = 1.
        else:
            self.wts = wts
        self.alpha = 1.

    def func(self, params):
        v = np.sqrt(1 + (params / self.dlt)**2)
        v -= 1
        v *= self.dlt**2
        return np.sum(self.wts * self.alpha * v)

    def grad(self, params):
        v = np.sqrt(1 + (params / self.dlt)**2)
        return params * self.wts * self.alpha / v


class CovariancePenalty(object):

    def __init__(self, wt):
        self.wt = wt

    def func(self, mat, mat_inv):
        """
        Parameters
        ----------
        mat : square matrix
            The matrix to be penalized.
        mat_inv : square matrix
            The inverse of `mat`.

        Returns
        -------
        A scalar penalty value
        """
        raise NotImplementedError

    def grad(self, mat, mat_inv):
        """
        Parameters
        ----------
        mat : square matrix
            The matrix to be penalized.
        mat_inv : square matrix
            The inverse of `mat`.

        Returns
        -------
        A vector containing the gradient of the penalty
        with respect to each element in the lower triangle
        of `mat`.
        """
        raise NotImplementedError


class PSD(CovariancePenalty):
    """
    A penalty that converges to +infinity as the argument matrix
    approaches the boundary of the domain of symmetric, positive
    definite matrices.
    """

    def func(self, mat, mat_inv):
        try:
            cy = np.linalg.cholesky(mat)
        except np.linalg.LinAlgError:
            return np.inf
        return -2 * self.wt * np.sum(np.log(np.diag(cy)))

    def grad(self, mat, mat_inv):
        cy = mat_inv.copy()
        cy = 2*cy - np.diag(np.diag(cy))
        i,j = np.tril_indices(mat.shape[0])
        return -self.wt * cy[i,j]
