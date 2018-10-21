__author__ = 'Luca Puggini: <lucapuggio@gmail.com>'

import numpy as np
from scipy.linalg import block_diag


## this class will be later removed and taken from another push
class Penalty(object):
    """
    A class for representing a scalar-value penalty.
    Parameters
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


class UnivariateGamPenalty(Penalty):
    __doc__ = """
    Penalty for Generalized Additive Models class

    Parameters
    -----------
    alpha : float
        the penalty term

    wts: TODO: I do not know!

    cov_der2: the covariance matrix of the second derivative of the basis matrix

    der2: The second derivative of the basis function

    Attributes
    -----------
    alpha : float
        the penalty term

    wts: TODO: I do not know!

    cov_der2: the covariance matrix of the second derivative of the basis matrix

    der2: The second derivative of the basis function

    n_samples: The number of samples used during the estimation



    """

    def __init__(self, univariate_smoother, wts=1, alpha=1):
        self.wts = wts  # should we keep wts????
        self.alpha = alpha
        self.univariate_smoother = univariate_smoother
        self.n_samples, self.n_columns = self.univariate_smoother.n_samples, self.univariate_smoother.dim_basis

    def func(self, params):
        '''
        1) params are the coefficients in the regression model
        2) der2  is the second derivative of the splines basis
        '''

        if self.univariate_smoother.der2_basis_ is not None:
            # The second derivative of the estimated regression function
            f = np.dot(self.univariate_smoother.der2_basis_, params)
            return self.alpha * np.sum(f ** 2) / self.n_samples
        else:
            f = params.dot(self.univariate_smoother.cov_der2_.dot(params))
            return self.alpha * f / self.n_samples

    def grad(self, params):
        '''
        1) params are the coefficients in the regression model
        2) der2  is the second derivative of the splines basis
        3) cov_der2 is obtained as np.dot(der2.T, der2)
        '''

        return 2 * self.alpha * np.dot(self.univariate_smoother.cov_der2_, params) / self.n_samples

    def deriv2(self, params):
        return 2 * self.alpha * self.univariate_smoother.cov_der2_ / self.n_samples


class MultivariateGamPenalty(Penalty):
    __doc__ = """
    GAM penalty for multivariate regression

    Parameters
    -----------
    cov_der2: list of matrices
     is a list of squared matrix of shape (size_base, size_base)

    der2: list of matrices
     is a list of matrix of shape (n_samples, size_base)

    alpha: array-like
     list of doubles. Each one representing the penalty
          for each function

    wts: array-like
     is a list of doubles of the same length of alpha

    """

    def __init__(self, multivariate_smoother, alpha, wts=None, start_idx=0):

        if len(multivariate_smoother.smoothers_) != len(alpha):
            raise ValueError('all the input values should be list of the same length. len(smoothers_)=',
                             len(multivariate_smoother.smoothers_), ' len(alphas)=', len(alpha))

        self.multivariate_smoother = multivariate_smoother
        self.dim_basis = self.multivariate_smoother.dim_basis
        self.k_variables = self.multivariate_smoother.k_variables
        self.n_samples = self.multivariate_smoother.n_samples
        self.alpha = alpha
        self.start_idx = start_idx
        self.k_params = start_idx + self.dim_basis

        # TODO: Review this
        if wts is None:
            self.wts = [1] * len(alpha)
        else:
            self.wts = wts

        self.mask = [np.array([False] * self.k_params)
                     for _ in range(self.k_variables)]
        param_count = start_idx
        for i, smoother in enumerate(self.multivariate_smoother.smoothers_):
            # the mask[i] contains a vector of length k_columns. The index
            # corresponding to the i-th input variable are set to True.
            self.mask[i][param_count: param_count + smoother.dim_basis] = True
            param_count += smoother.dim_basis

        self.gp = []
        for i in range(self.k_variables):
            gp = UnivariateGamPenalty(wts=self.wts[i], alpha=self.alpha[i],
                                      univariate_smoother=self.multivariate_smoother.smoothers_[i])
            self.gp.append(gp)

        return

    def func(self, params):
        cost = 0
        for i in range(self.k_variables):
            params_i = params[self.mask[i]]
            cost += self.gp[i].func(params_i)

        return cost

    def grad(self, params):
        grad = [np.zeros(self.start_idx)]
        for i in range(self.k_variables):
            params_i = params[self.mask[i]]
            grad.append(self.gp[i].grad(params_i))

        return np.concatenate(grad)

    def deriv2(self, params):
        deriv2 = np.zeros((self.start_idx, self.start_idx))
        for i in range(self.k_variables):
            params_i = params[self.mask[i]]
            deriv2 = block_diag(deriv2, self.gp[i].deriv2(params_i))

        return deriv2
