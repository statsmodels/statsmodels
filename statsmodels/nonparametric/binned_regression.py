# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 15:21:54 2017

Author: Josef Perktold
License: BSD-3

"""
from __future__ import division
import numpy as np
from scipy import special


def get_kernel(power_coef=3):
    """kernel of symmetric beta family parameterized by power coefficient

    Parameters
    ----------
    power_coef : int
       This is the power used in the symmetric beta family.

       - power = 0 is uniform
       - power = 1 is Epanechnikov
       - power = 2 is biweight
       - power = 3 is triweight
       - power -> inf is Gaussian

       Warning: no special case handling for 0 or inf yet.

    Returns
    -------
    kern_func : function
        kernel function that takes a single argument

    """

    tmp = 2 * power_coef + 2
    const = special.gamma(tmp) * special.gamma(power_coef + 1)**(-2) * 2**(-tmp)
    def kernel(x):
        res = const * np.maximum(1. - x*x, 0.)**power_coef
        return res

    return kernel


class LocalPolynomialFitResults(object):
    """Results class for univariate local polynomial regression



    """
    def __init__(self, projector, endog, fitted_values, **kwds):
        self.endog = endog
        self.projector = projector
        self.fittedvalues = fitted_values
        self.__dict__.update(kwds)

    def plot(self, ax=None, endog_true=None):
        """basic plot of data and fittedvalues

        ax not used yet, always creates a figure
        """
        proj = self.projector
        x = proj.exog
        y = self.endog / proj.weights # + 1e-15)
        fitted = self.fittedvalues
        nobs = proj.nobs
        bwi = proj.bwi

        import matplotlib.pyplot as plt
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax.plot(x, y, 'o', color='cyan', alpha=0.5)
        if endog_true is not None:
            plt.plot(x, endog_true, 'k-', lw=3, label='true')
        miss = len(y) - len(fitted)
        idx0 = miss // 2
        idx1 = len(y) - (miss - idx0)
        ax.plot(x[idx0: idx1], fitted, 'r-', lw=2, label='linear')
        ax.legend()
        ax.set_title('Local Polynomial Regression (nobs=%d, bwi=%d)' %
                     (nobs, bwi))
        return ax.figure


class BinnedLocalPolynomialProjector(object):
    """Locally constant or locally linear polynomial kernel regression.

    assumes exog is on an equal spaced grid, see Binner


    Parameters
    ----------
    exog : array_like, 1-D
        explanatory variable assumed to be on equal space grid
    poly_degree : int, 0 or 1
       degree 0 is local constant and degree 1 is local linear regression.
       Higher degrees are currently not implemented
    weights: float or array_like
       frequency weights, e.g. based on count in bins
    window_length : int
       The window is symmetric of odd length with window_length // 2 on each
       half window plus the center point.
       E.g. window_lenght equal to 10 or 11 both result in a window with
       11 points.
    kernel : None or callable
       currently only the default kernel supported which is a symmetric beta
       kernel, see power_coef
    power_coef : int
       This is the power used in the symmetric beta family.

       - power = 0 is uniform
       - power = 1 is Epanechnikov
       - power = 2 is biweight
       - power = 3 is triweight
       - power -> inf is Gaussian

       Warning: no special case handling for 0 or inf yet.


    Notes
    -----
    Status: experimental, missing options and features, no input checks,
    insufficient checks for empty bins resulting in nan or inf.


    """

    def __init__(self, exog, poly_degree=1, weights=1.,
                 window_length=None, kernel=None, power_coef=3):

        self.exog = exog = np.asarray(exog)
        self.nobs = exog.shape[0]
        if kernel is None:
            kernel = get_kernel(power_coef=power_coef)
        nobs = exog.shape[0]
        width = 1. / nobs
        if window_length is None:
            bwi = nobs // 10
        else:
            bwi = window_length

        bw = bwi * width
        dist = np.arange(-(bwi // 2), bwi // 2 + 1) * width
        w0 = kernel(dist / bw)
        w1 = w0 * dist
        w2 = w1 * dist
        self.w0, self.w1 = w0, w1

        if np.isscalar(weights):
            tmp = weights
            weights = np.empty(nobs, dtype=np.float64)
            weights.fill(tmp)
        self.weights = weights

        mode = 'same' #'valid'
        s0 = np.convolve(weights, w0, mode=mode)
        s1 = np.convolve(weights, w1, mode=mode)
        s2 = np.convolve(weights, w2, mode=mode)
        self.s0, self.s1, self.s2 = s0, s1, s2

        # attach what else we might need, currently for checking
        self.bwi = bwi


    def project(self, endog, is_sum=True):
        """project or smooth a series

        currently endog has to be 1-D

        Parameters
        ----------
        endog : array_like
             this has to match the exog that has been provided
        is_sum : bool
             In the computation we need the sum of endog at each bin.

        Returns
        -------
        res : LocalPolynomialFitResults instance

        """
        w0, w1 = self.w0, self.w1
        s0, s1, s2 = self.s0, self.s1, self.s2
        if is_sum:
            y = np.asarray(endog)
        else:
            y = np.asarray(endog) * self.weights
        mode = 'same'
        t0 = np.convolve(y, w0, mode=mode)
        t1 = np.convolve(y, w1, mode=mode)
        #t2 = np.convolve(y, w2, mode=mode)  #not used for constant and linear

        m0 = t0 / s0
        m1 = (s2 * t0 - s1 * t1) / (s2 * s0 - s1**2)
        res = LocalPolynomialFitResults(self, y, m1, fitted_locpoly0=m0)
        return res

    smooth = project  # alias for smoothers


class Binner(object):
    """class for linear binning

    Linear binning assigns a fraction of the observation to each
    neighboring bin center

    Similar to numpy histogram, the bins are open on the upper end, except for
    a correction at the last bin which includes the upper bin boundary.


    Parameters
    ----------
    x : array_like
        binning variable, linear binning fraction is base on the distance of
        this variable to the bins
    n_bins : int
        the number of grid points is currently n_bins + 1
    xmin : None or float
        smallest bin value. If None, then the smallest value of x is used.
    xmax : None or float
        largest bin value. If None, then the largest value of x is used.
        To avoid extra bins for observations that are at xmax, xmax is corrected
        by 1e-15 so the upper boundary is included in the last bin.

    Attributes
    ----------
    d : float
        distance between grid points
    idx : integer array
        index array for the assignment of observations to bin
    rem : array
        fraction of each observation for the linear binning
    bin_center : array
        center of the bins, the length corresponds to the number of grid points
        n_bins + 1.

    Notes
    -----
    Status: It works, but there might be some redefinition in the number and
    values of the grid.

    """

    # TODO: needs better attribute names

    def __init__(self, x, n_bins=500, xmin=None, xmax=None):
        # needs options for points (x < xmin) and (x >= xmax)
        # currently clipping
        x = np.asarray(x)
        if xmin is None:
            xmin = x.min()
        if xmax is None:
            xmax = x.max() + 1e-15
        else:
            # does this always make a copy?
            x = np.clip(x, xmin, xmax - 1e-15)

        d = (xmax - xmin) / n_bins
        idx, rem = divmod(x - xmin, d)
        idx = idx.astype(np.int64)
        xcenter = xmin + (np.arange(n_bins + 1) + 0.5) * d   #bin center
        # attach
        self.n_bins = n_bins
        self.d = d
        self.idx = idx
        self.rem = rem
        self.bin_center = xcenter

    def bin_data(self, x=1.):
        """discretize or bin a new series base on bins and bin fractions

        This uses the same fraction to allocate to neighboring bins as the
        original variable.

        Parameters
        ----------
        x : array_like, 1-D
            new data series to be binned. It needs to correspond to the data
            series used in creating the instance. Currently restricted to one
            dimensional arrays because of the use of numpy bincount.

        Returns
        -------
        binned : ndarray
            data binned to the grid. The magnitudes correspond to the sum of
            all fractional values that are assigned to a bin.

        """
        n_bins = self.n_bins
        rem = self.rem
        idx = self.idx
        binned = np.zeros(n_bins + 1)
        binned[:-1] += np.bincount(idx, x * (1 - rem), minlength=n_bins - 1)
        binned[1:] += np.bincount(idx, x *rem, minlength=n_bins - 1)
        return binned


def _leave_kth_out(k, endog, exog, projector_kwds, start=0):
    """experimental version to leave kth bin or observation out

    Computes the mean squared error of the out-of-sample prediction for the
    left out variables. Every k-th observation or bin starting at `start` is
    left out for the estimation.
    This works by setting the weight left out bins to essentially zero and does
    not change the number and sequence of bins.

    Parameters
    ----------
    k : int
        every k-th bin is left out by setting the weights of it to 1e-15
    endog : array
        dependent or endogenous variable
    exog : array
        independent or exogenous variable
    projector_kwds : dict
        keywords transmitted to the projector class
    start : int
        The index of the first observation to be left out

    Returns
    -------
    mse : float
        mean squared error of the prediction of the left out observations

    """
    exog = np.asarray(exog)
    nobs = exog.shape[0]
    weights = projector_kwds.pop('weights')
    mask = np.ones(nobs, np.bool_)
    mask[start::k] = 0
    # TODO: use weights = 0 or epsilon to avoid zero division
    weights_m = weights * mask + 1e-15   # not inplace
    endog_m = endog * mask
    locpol = BinnedLocalPolynomialProjector(exog, weights=weights_m,
                                            **projector_kwds)
    res = locpol.project(endog_m)
    m1 = res.fittedvalues
    #out of sample mse
    fitted_sum = m1[~mask]
    weights_nm = weights[~mask]
    if not np.isscalar(weights):
        fitted_sum *= weights_nm
    elif weights != 1:
        # why do/should we support scalar weights != 1 ?
        fitted_sum *= weights
    resid = endog[~mask] - fitted_sum
    mse = (resid**2).sum() / weights_nm.sum()
    return mse


def _cross_validation(window_lengths, k, endog, exog, projector_kwds):
    """experimental leave k-th out cross validation for bandwidth choice

    This computes the mean squared error mse for out-of-sample prediction.

    Parameters
    ----------
    window_lengths : iterable
        Iterable of all window_lengths (bandwidth parameter) for which the
        leave k-th out cross-validation error is computed.
    k : int
        every k-th bin is left out by setting the weights of it to 1e-15
    endog : array
        dependent or endogenous variable
    exog : array
        independent or exogenous variable
    projector_kwds : dict
        keywords transmitted to the projector class

    Returns
    -------
    res : ndarray, 2-D
        This array contains the window length in the first column and the
        cross-validation mse in the second column for each window specified in
        `window_lengths`.


    Notes
    -----
    This needs more options to control the cross-validation. Currently it
    undersmooths in small samples.
    """
    res = []
    if len(endog) < 202:
        incr = 1
    else:
        incr = max(1, k // 3)
    for k_win in window_lengths:
        kwds = projector_kwds.copy()
        kwds['window_length'] = k_win
        # todo: add start option, or do we want full k
        mse = 0

        start_all = np.arange(0, k, incr)   #[0, k // 2]
        for start in start_all:
            mse += _leave_kth_out(k, endog, exog, kwds.copy(), start=start)
            #print('cv loop', k_win, start, mse)  #for checking or debugging
        res.append([k_win, mse / len(start_all)])

    return np.asarray(res)


def fit_loclin_cvbw(endog, exog, weights=1., n_bins=None, shift_loss=0.1,
                    max_window_length=50, **projector_kwds):
    """convenience function to get local linear regression results

    This function combines binning, cross-validate bandwidth search and returns a
    LocalPolynomialFitResults with the local linear kernel regression base on the
    minimum MSE window_length.

    Warning: This oversmooths in small samples.

    Parameters
    ----------

    endog : array
        dependent or endogenous variable
    exog : array
        independent or exogenous variable
    weights : array or float
        frequency weights if data is already binned. This is currently ignored
        if binning is performed by this function, i.e. if n_bins > 0.
    n_bins : None or int
        If None, then no binning is performed, and the data is assumed without
        verification to be on an equal spaced grid.
    shift_loss : float
        This increases the window length if the relative loss is smaller than shift_loss.
        In small samples setting for example shift_loss = 0.2 helps to avoid undersmoothing
        in some cases.
    projector_kwds : dict
        keywords transmitted to the projector class

    Returns
    -------
    res : LocalPolynomialFitResults instance
        results of the local linear kernel regression based on the minimum
        MSE window length.

    Notes
    -----
    Note: If observation are densely distributed on a compact support, then the
    method will can give good results without binning and unequal spaced
    observations.

    TODO: add options to control cross-validation

    """
    if n_bins is not None:
        binn = Binner(exog, n_bins=n_bins) #, xmin=0, xmax=1)
        exog_ = binn.bin_center
        endog_ = binn.bin_data(endog)
        #assert len(exog_) == len(endog_)
        weights = binn.bin_data()

    else:
        exog_ = exog
        endog_ = endog

    win_lens = np.arange(5, max_window_length + 1, 2)
    res_cv = _cross_validation(win_lens, 10, endog_, exog_,
                            dict(weights=weights))
    idx_best = np.nanargmin(res_cv[:,1])
    bwi_best, mse = res_cv[idx_best]
    projector_kwds['window_length'] = bwi_best
    if shift_loss > 0 and idx_best != len(res_cv) - 1:
        for idx, r in enumerate(res_cv[idx_best+1:]):
            if r[1] > mse * (1 + shift_loss):
                break

        projector_kwds['window_length'] = res_cv[idx_best + idx, 0]

    locpoly = BinnedLocalPolynomialProjector(exog_, weights=weights,
                                            **projector_kwds)
    res = locpoly.project(endog_)
    res.results_crossval = res_cv
    return res
