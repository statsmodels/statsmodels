"""
Holds files for l1 regularization of LikelihoodModel, using cvxopt.
"""
import numpy as np
import pdb
from cvxopt import solvers, matrix
# pdb.set_trace


def _fit_l1_cvxopt_cp(
        f, score, start_params, args, kwargs, disp=None, maxiter=100,
        callback=None, retall=False, full_output=False, hess=None):
    """
    Solve the l1 regularized problem using cvxopt.solvers.cp

    Parameters
    ----------
    All the usual parameters from LikelhoodModel.fit

    Special kwargs
    ----------
    alpha : non-negative scalar or numpy array (same size as parameters)
        The weight multiplying the l1 penalty term
    trim_params : boolean (default True)
        Set small parameters to zero
    trim_tol : float
        Set parameters whose absolute value < trim_tol to zero
    abstol : float
        absolute accuracy (default: 1e-7).
    reltol : float
        relative accuracy (default: 1e-6).
    feastol : float
        tolerance for feasibility conditions (default: 1e-7).
    refinement : int
        number of iterative refinement steps when solving KKT equations
        (default: 1).
    """

    if callback:
        print "Callback will be ignored with l1_cvxopt_cp"
    start_params = np.array(start_params).ravel('F')

    ## Extract arguments
    # K is total number of covariates, possibly including a leading constant.
    K = len(start_params)
    # The regularization parameter
    alpha = np.array(kwargs['alpha']).ravel('F')
    assert alpha.min() >= 0
    # The start point
    x0 = np.append(start_params, np.fabs(start_params))
    x0 = matrix(x0, (2 * K, 1))

    ## Wrap up functions for cvxopt
    f_0 = lambda x: objective_func(f, x, K, alpha, *args)
    Df = lambda x: fprime(score, x, K, alpha)
    G = get_G(K)  # Inequality constraint matrix, Gx \leq h
    h = matrix(0.0, (2 * K, 1))  # RHS in inequality constraint
    H = lambda x, z: hessian_wrapper(hess, x, z, K)

    ## Define the optimization function
    def F(x=None, z=None):
        if x is None:
            return 0, x0
        elif z is None:
            return f_0(x), Df(x)
        else:
            return f_0(x), Df(x), H(x, z)

    ## Convert optimization settings to cvxopt form
    solvers.options['show_progress'] = retall
    solvers.options['maxiters'] = maxiter
    if 'abstol' in kwargs:
        solvers.options['abstol'] = kwargs['abstol']
    if 'reltol' in kwargs:
        solvers.options['reltol'] = kwargs['reltol']
    if 'feastol' in kwargs:
        solvers.options['feastol'] = kwargs['feastol']
    if 'refinement' in kwargs:
        solvers.options['refinement'] = kwargs['refinement']

    ### Call the optimizer
    results = solvers.cp(F, G, h)

    ### Post-process
    trim_params = kwargs.setdefault('trim_params', True)
    if trim_params:
        trim_tol = kwargs.setdefault('trim_tol', 1e-4)
        results = do_trim_params(results, K, alpha, trim_tol)

    ### Pack up return values for statsmodels
    # TODO These retvals are returned as mle_retvals...but the fit wasn't ML
    if full_output:
        x = np.array(results['x']).ravel()
        params = x[:K]
        fopt = f_0(x)
        gopt = float('nan')  # Objective is non-differentiable
        hopt = float('nan')
        iterations = float('nan')
        converged = 'True' if results['status'] == 'optimal'\
            else results['status']
        retvals = {
            'fopt': fopt, 'converged': converged, 'iterations': iterations,
            'gopt': gopt, 'hopt': hopt}
    else:
        x = np.array(results['x']).ravel()
        params = x[:K]

    ### Return results
    if full_output:
        return params, retvals
    else:
        return params


def do_trim_params(results, K, alpha, trim_tol):
    """
    Trims (sets = 0) params that are within trim_tol of zero.
    If alpha[i] == 0, then don't trim the ith param.
    """
    ## Extract params from the results
    x_arr = np.array(results['x'])
    ## Trim the small params
    # Don't bother triming the dummy variables 'u'
    for i in xrange(K):
        if abs(x_arr[i]) < trim_tol and alpha[i] != 0:
            x_arr[i] = 0.0
    ## Replenish results
    results['x'] = matrix(x_arr)
    ## Return
    return results


def objective_func(f, x, K, alpha, *args):
    """
    The regularized objective function.
    """
    x_arr = np.array(x)
    params = x_arr[:K].ravel()
    u = x_arr[K:]
    # Call the numpy version
    objective_func_arr = f(params, *args) + (alpha * u).sum()
    # Return
    return matrix(objective_func_arr)


def fprime(score, x, K, alpha):
    """
    The regularized derivative.
    """
    x_arr = np.array(x)
    params = x_arr[:K].ravel()
    # Call the numpy version
    # The derivative just appends a vector of constants
    fprime_arr = np.append(score(params), alpha * np.ones(K))
    # Return
    return matrix(fprime_arr, (1, 2 * K))


def get_G(K):
    """
    The linear inequality constraint matrix.
    """
    I = np.eye(K)
    A = np.concatenate((-I, -I), axis=1)
    B = np.concatenate((I, -I), axis=1)
    C = np.concatenate((A, B), axis=0)
    # Return
    return matrix(C)


def hessian_wrapper(hess, x, z, K):
    """
    Wraps the hessian up in the form for cvxopt.
    """
    x_arr = np.array(x)
    params = x_arr[:K].ravel()
    zh_x = np.array(z[0]) * hess(params)
    zero_mat = np.zeros(zh_x.shape)
    A = np.concatenate((zh_x, zero_mat), axis=1)
    B = np.concatenate((zero_mat, zero_mat), axis=1)
    zh_x_ext = np.concatenate((A, B), axis=0)
    return matrix(zh_x_ext, (2 * K, 2 * K))
