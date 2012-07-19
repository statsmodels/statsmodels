"""
Holds files for the lasso regularization
"""
import numpy as np
from scipy.optimize import fmin_slsqp
import pdb
# pdb.set_trace


def _fit_lasso(f, score, start_params, fargs, kwargs, disp=None, maxiter=100, 
        callback=None, retall=False, full_output=False, hess=None):
    """
    """

    if callback:
        print "Callback will be ignored with lasso"

    ### Extract values
    P = len(start_params)
    if 'constant' in kwargs:
        if kwargs['constant']:
            K = P-1
    alpha = kwargs['alpha']
    epsilon = kwargs.setdefault('epsilon', None)
    # Some display parameters
    if disp or retall:
        if disp:
            disp_slsqp = 1
        if retall:
            disp_slsqp = 2
    else:
        disp_slsqp = 0
    ### Set up the optimization functions
    ## We recover twice the parameters in this constrained optimization
    ## We will break x into params and the dummy variable 'u'
    ## params = x[:K], u = x[K:]
    # The function to be minimized
    func = lambda x : f(x[:K]) + alpha * x[K:].sum()
    # The start point
    x0 = np.append(start_params, np.fabs(start_params))
    # The derivative just appends a vector of constants
    fprime = lambda x : np.append(score(x[:K]), alpha * np.ones(K))
    # All entries in this vector must be \geq 0 in a feasible solution
    f_ieqcons = lambda x : np.append(x[:K]+x[K:], x[K:]-x[:K])

    results = fmin_slsqp(func, x0, f_ieqcons=f_ieqcons, fprime=fprime, args=fargs, 
            iter=maxiter, disp=disp_slsqp, full_output=full_output, epsilon=epsilon)

    ### The return values for statsmodels optimizers
    if full_output:
        x, fx, its, imode, smode = results
        xopt = np.array(x[:K])
        fopt = fx
        converged = smode
        iterations = its
        gopt = score(xopt)
        hopt = hess(xopt)
        retvals = {'fopt':fopt, 'converged':converged, 'iterations':iterations, 
                'gopt':gopt, 'hopt':hopt}
        return xopt, retvals
    else:
        xopt = np.array(results)
        return xopt

#def func(x, *args):
#    params = x[:K]
