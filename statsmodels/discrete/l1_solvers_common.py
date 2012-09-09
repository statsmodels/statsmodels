"""
Holds common functions for l1 solvers.
"""
import numpy as np


def QC_results(params, alpha, score, kwargs):
    """
    Theory dictates that one of two conditions holds:
        i) abs(score[i]) == alpha[i]  and  params[i] != 0
        ii) abs(score[i]) <= alpha[i]  and  params[i] == 0
    QC_results checks to see that (ii) holds, within QC_tol

    QC_results also checks for nan or results of the wrong shape.

    Parameters
    ----------
    params : np.ndarray
        model parameters.  Not including the added variables x_added.
    alpha : np.ndarray
        regularization coefficients
    score : function
        Gradient of unregularized objective function
    kwargs : Dictionary
        The usual **kwargs passed to calling function.

    Returns
    -------
    passed : Boolean
        True if QC check passed
    QC_dict : Dictionary
        Keys are fprime, alpha, params, passed_array

    Prints
    ------
    Warning message if QC check fails.
    """
    ## Extract kwargs
    QC_tol = kwargs.setdefault('QC_tol', 0.03)

    ## Check for fatal errors
    assert not np.isnan(params).max()
    assert (params == params.ravel('F')).min(), \
        "params should have already been 1-d"

    ## Start the theory compliance check
    fprime = score(params)
    k_params = len(params)

    passed_array = np.array([True] * k_params)
    for i in xrange(k_params):
        if alpha[i] > 0:
            # If |fprime| is too big, then something went wrong
            if (abs(fprime[i]) - alpha[i]) / alpha[i] > QC_tol:
                passed_array[i] = False
    QC_dict = dict(
        fprime=fprime, alpha=alpha, params=params, passed_array=passed_array)
    passed = passed_array.min()
    if not passed:
        num_failed = (passed_array == False).sum()
        message = 'QC check did not pass for %d out of %d parameters' % (
            num_failed, k_params)
        message += '\nTry increasing solver accuracy or number of iterations'\
            ', decreasing alpha, or switch solvers'
        print message

    return passed, QC_dict


def do_trim_params(params, k_params, alpha, score, passed, kwargs):
    """
    Trims (set to zero) params that are zero at the theoretical minimum.
    Uses heuristics to account for the solver not actually finding the minimum.

    In all cases, if alpha[i] == 0, then don't trim the ith param.
    In all cases, do nothing with the added variables.

    Parameters
    ----------
    params : np.ndarray
        model parameters.  Not including added variables.
    k_params : Int
        Number of parameters
    alpha : np.ndarray
        regularization coefficients
    score : Function.
        score(params) should return a 1-d vector of derivatives of the
        unpenalized objective function.
    passed : Boolean
        True if the QC check passed
    kwargs : Dictionary
        The usual **kwargs passed to calling function.

    Returns
    -------
    params : np.ndarray
        Trimmed model parameters
    trimmed : np.ndarray of Booleans
        trimmed[i] == True if the ith parameter was trimmed.
    """
    ## Extract kwargs
    trim_mode = kwargs.setdefault('trim_mode', 'auto')
    size_trim_tol = kwargs.setdefault('size_trim_tol', 1e-4)
    auto_trim_tol = kwargs.setdefault('auto_trim_tol', 0.01)

    ## Trim the small params
    trimmed = [False] * k_params

    if trim_mode == 'off':
        trimmed = np.array([False] * k_params)
    elif trim_mode == 'auto' and not passed:
        print "Could not trim params automatically due to failed QC "\
            "check.  Trimming using trim_mode == 'size' will still work."
        trimmed = np.array([False] * k_params)
    elif trim_mode == 'auto' and passed:
        fprime = score(params)
        for i in xrange(k_params):
            if alpha[i] != 0:
                if (alpha[i] - abs(fprime[i])) / alpha[i] > auto_trim_tol:
                    params[i] = 0.0
                    trimmed[i] = True
    elif trim_mode == 'size':
        for i in xrange(k_params):
            if alpha[i] != 0:
                if abs(params[i]) < size_trim_tol:
                    params[i] = 0.0
                    trimmed[i] = True
    else:
        raise Exception(
            "trim_mode == %s, which is not recognized" % (trim_mode))

    return params, np.asarray(trimmed)
