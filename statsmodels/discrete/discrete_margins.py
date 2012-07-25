#Splitting out maringal effects to see if they can be generalized

import numpy as np

def margeff_cov_params(params, exog, cov_params, at, derivative, dummy_ind,
                       count_ind):
    """
    Computes the variance-covariance of marginal effects by the delta method.

    Parameters
    ----------
    params : array-like
        estimated model parameters
    exog : array-like
        exogenous variables at which to calculate the derivative
    cov_params : array-like
        The variance-covariance of the parameters
    at : str
       Options are:

        - 'overall', The average of the marginal effects at each
          observation.
        - 'mean', The marginal effects at the mean of each regressor.
        - 'median', The marginal effects at the median of each regressor.
        - 'zero', The marginal effects at zero for each regressor.
        - 'all', The marginal effects at each observation.

        Only overall has any effect here.

    derivative : function or array-like
        If a function, it returns the marginal effects of the model with
        respect to the exogenous variables evaluated at exog. Expected to be
        called derivative(params, exog). This will be numerically
        differentiated. Otherwise, it can be the Jacobian of the marginal
        effects with respect to the parameters.
    dummy_ind : array-like
        Indices of the columns of exog that contain dummy variables
    count_ind : array-like
        Indices of the columns of exog that contain count variables

    Notes
    -----
    For continuous regressors, the variance-covariance is given by

    Asy. Var[MargEff] = [d margeff / d params] V [d margeff / d params]'

    where V is the parameter variance-covariance.

    The outer Jacobians are computed via numerical differentiation if
    derivative is a function.
    """
    if callable(derivative):
        from statsmodels.sandbox.regression.numdiff import approx_fprime_cs
        jacobian_mat = approx_fprime_cs(params, derivative, args=(exog,))
        if at == 'overall':
            jacobian_mat = np.mean(jacobian_mat, axis=1)
    else:
        jacobian_mat = derivative
    #NOTE: this won't go through for at == 'all'
    return np.dot(np.dot(jacobian_mat, cov_params), jacobian_mat.T)

def margeff_cov_with_se(params, exog, cov_params, at, derivative, dummy_ind,
               count_ind):
    """
    See margeff_cov_params.

    Same function but returns both the covariance of the marginal effects
    and their standard errors.
    """
    cov_me = margeff_cov_params(params, exog, cov_params, at,
                                              derivative, dummy_ind,
                                              count_ind)
    return cov_me, np.sqrt(np.diag(cov_me))

def margeff():
    pass
