from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy import newaxis

from ..compat.python import range


def _bootstrap(fitted_kde, nb_samples, eval_fct, CIs):
    if CIs is not None:
        CIs = np.asarray(CIs)
        if np.any((CIs > 1) | (CIs < 0)):
            raise ValueError("Error, confidence interval values must be "
                             "between 0 and 1 only")
    exog = fitted_kde.exog
    fitted = fitted_kde.copy()
    npts = fitted_kde.npts
    sampling = np.random.randint(0, npts-1, size=(nb_samples, npts))
    results = []
    for i in range(nb_samples):
        fitted.exog = exog[sampling[i]]
        results.append(eval_fct(fitted))
    results = np.concatenate([r[..., newaxis] for r in results], axis=-1)
    results.sort(axis=-1)
    if CIs is not None:
        med_CIs = CIs / 2
        # Compute exact position of the CI
        max_index = nb_samples-1
        low_CIs = max_index*(.5 - med_CIs)
        high_CIs = max_index*(.5 + med_CIs)
        # Compute integer positions
        lower_low_CIs = np.floor(low_CIs).astype(int)
        upper_low_CIs = np.ceil(low_CIs).astype(int)
        ratio_low_CIs = low_CIs - lower_low_CIs
        lower_high_CIs = np.floor(high_CIs).astype(int)
        upper_high_CIs = np.ceil(high_CIs).astype(int)
        ratio_high_CIs = high_CIs - lower_high_CIs
        # Compute CIs
        low_CIs = (results[..., lower_low_CIs] * (1-ratio_low_CIs) +
                   results[..., upper_low_CIs] * ratio_low_CIs)
        high_CIs = (results[..., lower_high_CIs] * (1-ratio_high_CIs) +
                    results[..., upper_high_CIs] * ratio_high_CIs)
        return np.concatenate((low_CIs[..., newaxis], high_CIs[..., newaxis]),
                              axis=-1)
    return results


def bootstrap_grid(fitted_kde, nb_samples, CIs=None,
                   bootstrapped_function=None, adjust_bw=1., fct_args={}):
    """
    Compute the confidence interval on the fitted KDE using bootstrapping.

    Parameters
    ----------
    fitted_kde: object
        Result of fitting a KDE object
    nb_samples: int
        Number of sampling to perform for the bootstrapping
    CIs: list of float or None
        If not None, this is the list of confidence intervals to return (e.g.
        values between 0 and 1). See the return values for details on the
        output.
    bootstrapped_function: None or string or callable
        If None, the ``grid`` method of the fitted_kde object will be used. If
        a string is given, the corresponding method will be used. At last, if
        another callable is used, it should accept as first argument the
        fitted_kde object and return two values: the mesh and the values.
    adjust_bw: float or ndarray or callable
        If a float or an ndarray of same size and shape as the fitted KDE
        bandwidth, the bandwidth of the fitted kde will be scaled by the
        given value. If a callable, it accepts the fitted_kde and the function
        to bootstrap as arguments and returns the scaling of the bandwidth,
        either as a float or an ndarray.
    fct_args: dictionary
        Dictionary of arguments given to the bootstrapped function.

    Returns
    -------
    mesh : `statsmodels.kernel_methods.kde_utils.Grid`
        Grid object giving the positions of all elements
    intervals: ndarray
        If ``CIs`` is None, this returns a ndarray of dimensions
        `grid.shape + (nb_samples,)`, where for each grid position, we obtain
        the ordered list of possible values.
        If ``CIs`` is specified, this returns a ndarray of dimensions
        `(grid.shape, len(CIs), 2)`, where for each grid position, and each
        confidence interval, we have a pair of value giving the range of values
        for this CI.
    """
    # Run a first time to get the exact grid
    if bootstrapped_function is None:
        bootstrapped_function = type(fitted_kde).grid
    elif isinstance(bootstrapped_function, str):
        bootstrapped_function = getattr(type(fitted_kde),
                                        bootstrapped_function)

    def eval_fct(fitted):
        return bootstrapped_function(fitted, **fct_args)[1]
    if callable(adjust_bw):
        adjust_bw = adjust_bw(fitted_kde, eval_fct)
    grid, _ = bootstrapped_function(fitted_kde, **fct_args)
    adjusted_kde = fitted_kde.copy()
    adjusted_kde.bandwidth *= adjust_bw
    return grid, _bootstrap(adjusted_kde, nb_samples, eval_fct, CIs)


def bootstrap(fitted_kde, eval_points, nb_samples, CIs=None,
              bootstrapped_function=None, adjust_bw=1., fct_args={}):
    """
    Compute the confidence interval on the fitted KDE using bootstrapping.

    Parameters
    ----------
    fitted_kde: object
        Result of fitting a KDE object
    eval_points: ndarray
        This should be a list of points, suitable to use in the ``__call__``
        method of the ``fitted_kde`` object.
    nb_samples: int
        Number of sampling to perform for the bootstrapping
    CIs: list of float or None
        If not None, this is the list of confidence intervals to return (e.g.
        values between 0 and 1). See the return values for details on the
        output.
    bootstrapped_function: None or string or callable
        If None, the ``__call__`` method of the fitted_kde object will be used.
        If a string is given, the corresponding method will be used. At last,
        if another callable is used, it should accept as first argument the
        fitted_kde object, as second a 1D or 2D array of points and return a
        ndarray with as many values as points given as input.
    adjust_bw: float or ndarray or callable
        If a float or an ndarray of same size and shape as the fitted KDE
        bandwidth, the bandwidth of the fitted kde will be scaled by the given
        value. if a callable, it accepts the fitted_kde and the bootstrap
        function as arguments and returns the scaling of the bandwidth, either
        as a float or an ndarray.
    fct_args: dictionary
        Dictionary of arguments given to the bootstrapped function.

    Returns
    -------
    intervals: ndarray
        If ``CIs`` is None, this returns a ndarray of dimensions
        `(len(eval_points.shape), nb_samples)`, where for each point position,
        we obtain the ordered list of possible values.
        If ``CIs`` is specified, this returns a ndarray of dimensions
        `(len(eval_points.shape), len(CIs), 2)`, where for each grid position,
        and each confidence interval, we have a pair of value giving the range
        of value for this CI.
    """
    if bootstrapped_function is None:
        bootstrapped_function = type(fitted_kde).__call__
    elif isinstance(bootstrapped_function, str):
        bootstrapped_function = getattr(type(fitted_kde),
                                        bootstrapped_function)

    def eval_fct(fitted):
        return bootstrapped_function(fitted, eval_points, **fct_args)
    if callable(adjust_bw):
        adjust_bw = adjust_bw(fitted_kde, eval_fct)
    adjusted_kde = fitted_kde.copy()
    adjusted_kde.bandwidth *= adjust_bw
    return _bootstrap(adjusted_kde, nb_samples, eval_fct, CIs)
