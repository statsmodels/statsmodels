# -*- coding: utf-8 -*-
"""
Penalty classes for Generalized Additive Models

Author: Luca Puggini
Author: Josef Perktold

"""

import numpy as np
from scipy.linalg import block_diag
from statsmodels.base._penalties import Penalty


class UnivariateGamPenalty(Penalty):
    __doc__ = """
    Penalty for Generalized Additive Models class

    Parameters
    -----------
    alpha : float
        the penalty term

    weights: TODO: I do not know!

    cov_der2: the covariance matrix of the second derivative of the basis
        matrix

    der2: The second derivative of the basis function

    Attributes
    -----------
    alpha : float
        the penalty term

    weights: TODO: I do not know!

    cov_der2: the covariance matrix of the second derivative of the basis matrix

    der2: The second derivative of the basis function

    n_samples: The number of samples used during the estimation

    """

    def __init__(self, univariate_smoother, weights=1, alpha=1):
        self.weights = weights  # should we keep weights????
        self.alpha = alpha
        self.univariate_smoother = univariate_smoother
        self.n_samples = self.univariate_smoother.n_samples
        self.n_columns = self.univariate_smoother.dim_basis

    def func(self, params, alpha=None):
        """evaluate penalization at params

        1) params are the coefficients in the regression model
        2) der2  is the second derivative of the splines basis
        """
        if alpha is None:
            alpha = self.alpha

        if self.univariate_smoother.der2_basis_ is not None:
            # The second derivative of the estimated regression function
            f = np.dot(self.univariate_smoother.der2_basis_, params)
            return alpha * np.sum(f ** 2) / self.n_samples
        else:
            f = params.dot(self.univariate_smoother.cov_der2_.dot(params))
            return alpha * f / self.n_samples

    def deriv(self, params, alpha=None):
        """evaluate derivative of penalty with respect to params

        1) params are the coefficients in the regression model
        2) der2  is the second derivative of the splines basis
        3) cov_der2 is obtained as np.dot(der2.T, der2)
        """
        if alpha is None:
            alpha = self.alpha

        d = 2 * alpha * np.dot(self.univariate_smoother.cov_der2_, params)
        d /= self.n_samples
        return d

    def deriv2(self, params, alpha=None):
        """evaluate second derivative of penalty with respect to params
        """
        if alpha is None:
            alpha = self.alpha

        d2 = 2 * alpha * self.univariate_smoother.cov_der2_
        d2 /= self.n_samples
        return  d2


    def penalty_matrix(self, alpha=None):
        if alpha is None:
            alpha = self.alpha

        return alpha * self.univariate_smoother.cov_der2_


class MultivariateGamPenalty(Penalty):
    __doc__ = """
    GAM penalty for multivariate regression

    Parameters
    ----------
    multivariate_smoother : instance of multivariate smoother
    alpha: array-like
        list of doubles. Each one representing the penalty
        for each function
    weights: array-like
        is a list of doubles of the same length of alpha
    start_idx : int
        number of parameters that come before the smooth terms. If the model has
        a linear component, then the parameters for the smooth components start
        at ``start_index``.

    """

    def __init__(self, multivariate_smoother, alpha, weights=None, start_idx=0):

        if len(multivariate_smoother.smoothers_) != len(alpha):
            raise ValueError('all the input values should be list of the same '
                             'length. len(smoothers_)=',
                             len(multivariate_smoother.smoothers_),
                             ' len(alphas)=', len(alpha))

        self.multivariate_smoother = multivariate_smoother
        self.dim_basis = self.multivariate_smoother.dim_basis
        self.k_variables = self.multivariate_smoother.k_variables
        self.n_samples = self.multivariate_smoother.n_samples
        self.alpha = alpha
        self.start_idx = start_idx
        self.k_params = start_idx + self.dim_basis

        # TODO: Review this,
        if weights is None:
            # weights should hanve length params
            self.weights = np.ones(self.k_params)
        else:
            self.weights = weights

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
            gp = UnivariateGamPenalty(weights=self.weights[i],
                alpha=self.alpha[i],
                univariate_smoother=self.multivariate_smoother.smoothers_[i])
            self.gp.append(gp)

    def func(self, params, alpha=None):
        if alpha is None:
            alpha = [None] * self.k_variables

        cost = 0
        for i in range(self.k_variables):
            params_i = params[self.mask[i]]
            cost += self.gp[i].func(params_i, alpha=alpha[i])

        return cost

    def deriv(self, params, alpha=None):
        if alpha is None:
            alpha = [None] * self.k_variables

        grad = [np.zeros(self.start_idx)]
        for i in range(self.k_variables):
            params_i = params[self.mask[i]]
            grad.append(self.gp[i].deriv(params_i, alpha=alpha[i]))

        return np.concatenate(grad)

    def deriv2(self, params, alpha=None):
        if alpha is None:
            alpha = [None] * self.k_variables

        deriv2 = [np.zeros((self.start_idx, self.start_idx))]
        for i in range(self.k_variables):
            params_i = params[self.mask[i]]
            deriv2.append(self.gp[i].deriv2(params_i, alpha=alpha[i]))

        return block_diag(*deriv2)

    def penalty_matrix(self, alpha=None):
        if alpha is None:
            alpha = self.alpha

        s_all = [np.zeros((self.start_idx, self.start_idx))]
        for i in range(self.k_variables):
            s_all.append(self.gp[i].penalty_matrix(alpha=alpha[i]))

        return block_diag(*s_all)
