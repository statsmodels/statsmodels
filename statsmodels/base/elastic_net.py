import numpy as np
import statsmodels.base.wrapper as wrap


def _gen_npfuncs(k, L1_wt, alpha, loglike_kwds, score_kwds, hess_kwds):
    """
    Negative penalized log-likelihood functions.

    Returns the negative penalized log-likelihood, its derivative, and
    its Hessian.  The penalty only includes the smooth (L2) term.

    All three functions have arguments (x, model), where ``x`` is a
    point in the parmeter space and ``model`` is an arbitrary model.
    """

    def nploglike(params, model):
        nobs = model.nobs
        pen = alpha[k] * (1 - L1_wt) * np.sum(params**2) / 2
        llf = model.loglike(np.r_[params], **loglike_kwds)
        return - llf / nobs + pen

    def npscore(params, model):
        nobs = model.nobs
        l2_grad = alpha[k] * (1 - L1_wt) * params
        gr = -model.score(np.r_[params], **score_kwds)[0] / nobs
        return gr + l2_grad

    def nphess(params, model):
        nobs = model.nobs
        pen_hess = alpha[k] * (1 - L1_wt)
        return -model.hessian(np.r_[params], **hess_kwds)[0,0] / nobs + pen_hess

    return nploglike, npscore, nphess



def _fit(model, method="coord_descent", maxiter=100, alpha=0.,
         L1_wt=1., start_params=None, cnvrg_tol=1e-7, zero_tol=1e-8,
         return_object=False, loglike_kwds=None, score_kwds=None,
         hess_kwds=None, **kwargs):
    """
    Return a regularized fit to a regression model.

    Parameters
    ----------
    model : model object
        A statsmodels object implementing ``log-like``, ``score``, and
        ``hessian``.
    method :
        Only the coordinate descent algorithm is implemented.
    maxiter : integer
        The maximum number of iteration cycles (an iteration cycle
        involves running coordinate descent on all variables).
    alpha : scalar or array-like
        The penalty weight.  If a scalar, the same penalty weight
        applies to all variables in the model.  If a vector, it
        must have the same length as `params`, and contains a
        penalty weight for each coefficient.
    L1_wt : scalar
        The fraction of the penalty given to the L1 penalty term.
        Must be between 0 and 1 (inclusive).  If 0, the fit is
        a ridge fit, if 1 it is a lasso fit.
    start_params : array-like
        Starting values for `params`.
    cnvrg_tol : scalar
        If `params` changes by less than this amount (in sup-norm)
        in once iteration cycle, the algorithm terminates with
        convergence.
    zero_tol : scalar
        Any estimated coefficient smaller than this value is
        replaced with zero.
    return_object : bool
        If False, only the parameter estimates are returned.
    loglike_kwds : dict-like or None
        Keyword arguments for the log-likelihood function.
    score_kwds : dict-like or None
        Keyword arguments for the score function.
    hess_kwds : dict-like or None
        Keyword arguments for the Hessian function.

    Returns
    -------
    If `return_object` is true, a results object of the same type
    returned by `model.fit`, otherise returns the estimated parameter
    vector.

    Notes
    -----
    The ``elastic net`` penalty is a combination of L1 and L2
    penalties.

    The function that is minimized is:

    -loglike/n + alpha*((1-L1_wt)*|params|_2^2/2 + L1_wt*|params|_1)

    where |*|_1 and |*|_2 are the L1 and L2 norms.

    The computational approach used here is to obtain a quadratic
    approximation to the smooth part of the target function:

    -loglike/n + alpha*(1-L1_wt)*|params|_2^2/2

    then optimize the L1 penalized version of this function along
    a coordinate axis.
    """

    k_exog = model.exog.shape[1]
    n_exog = model.exog.shape[0]

    loglike_kwds = {} if loglike_kwds is None else loglike_kwds
    score_kwds = {} if score_kwds is None else score_kwds
    hess_kwds = {} if hess_kwds is None else hess_kwds

    if np.isscalar(alpha):
        alpha = alpha * np.ones(k_exog)

    # Define starting params
    if start_params is None:
        params = np.zeros(k_exog)
    else:
        params = start_params.copy()

    converged = False
    btol = 1e-8
    params_zero = np.zeros(len(params), dtype=bool)

    init_args = {k : getattr(model, k) for k in model._init_keys
                 if k != "offset" and hasattr(model, k)}

    fgh_list = [_gen_npfuncs(k, L1_wt, alpha, loglike_kwds, score_kwds, hess_kwds)
                for k in range(k_exog)]

    for itr in range(maxiter):

        # Sweep through the parameters
        params_save = params.copy()
        for k in range(k_exog):

            # Under the active set method, if a parameter becomes
            # zero we don't try to change it again.
            if params_zero[k]:
                continue

            # Set the offset to account for the variables that are
            # being held fixed in the current coordinate
            # optimization.
            params0 = params.copy()
            params0[k] = 0
            offset = np.dot(model.exog, params0)
            if hasattr(model, "offset") and model.offset is not None:
                offset += model.offset

            # Create a one-variable model for optimization.
            model_1var = model.__class__(model.endog, model.exog[:, k], offset=offset,
                                         **init_args)

            func, grad, hess = fgh_list[k]
            params[k] = _opt_1d(func, grad, hess, model_1var, params[k], alpha[k]*L1_wt, tol=btol)

            # Update the active set
            if itr > 0 and np.abs(params[k]) < zero_tol:
                params_zero[k] = True
                params[k] = 0.

        # Check for convergence
        pchange = np.max(np.abs(params - params_save))
        if pchange < cnvrg_tol:
            converged = True
            break

    # Set approximate zero coefficients to be exactly zero
    params *= np.abs(params) >= zero_tol

    if not return_object:
        return params

    # Fit the reduced model to get standard errors and other
    # post-estimation results.
    ii = np.flatnonzero(params)
    cov = np.zeros((k_exog, k_exog))
    if len(ii) > 0:
        model1 = model.__class__(model.endog, model.exog[:, ii],
                               **kwargs)
        rslt = model1.fit()
        cov[np.ix_(ii, ii)] = rslt.normalized_cov_params
    else:
        model1 = model.__class__(model.endog, model.exog[:, 0], **kwargs)
        rslt = model1.fit()
        cov[np.ix_(ii, ii)] = rslt.normalized_cov_params

    # fit may return a results or a results wrapper
    if issubclass(rslt.__class__, wrap.ResultsWrapper):
        klass = rslt._results.__class__
    else:
        klass = rslt.__class__

    # Not all models have a scale
    if hasattr(rslt, 'scale'):
        scale = rslt.scale
    else:
        scale = 1.

    refit = klass(model, params, cov, scale=scale)
    refit.regularized = True

    return refit


def _opt_1d(func, grad, hess, model, start, L1_wt, tol):
    """
    One-dimensional helper for elastic net.

    Parameters:
    -----------
    func : function
        A smooth function of a single variable to be optimized
        with L1 penaty.
    grad : function
        The gradient of `func`.
    hess : function
        The Hessian of `func`.
    model : statsmodels model
        The model being fit.
    start : real
        A starting value for the function argument
    L1_wt : non-negative real
        The weight for the L1 penalty function.
    tol : non-negative real
        A convergence threshold.

    Notes
    -----
    ``func``, ``grad``, and ``hess`` have arguments (x, model), where
    ``x`` is a point in the parameter space and ``model`` is the model
    being fit.

    If the log-likelihood for the model is exactly quadratic, the
    global minimum is returned in one step.  Otherwise numerical
    bisection is used.

    Returns:
    --------
    The argmin of the objective function.
    """

    from scipy.optimize import brent

    x = start
    f = func(x, model)
    b = grad(x, model)
    c = hess(x, model)
    d = b - c*x

    if L1_wt > np.abs(d):
        return 0.

    if d >= 0:
        h = (L1_wt - b) / c
    elif d < 0:
        h = -(L1_wt + b) / c

    f1 = func(x + h, model) + L1_wt*np.abs(x + h)
    if f1 <= f + L1_wt*np.abs(x) + 1e-10:
        return x + h

    x_opt = brent(func, args=(model,), brack=(x-1, x+1), tol=tol)
    return x_opt
