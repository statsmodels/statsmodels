#cython: boundscheck = False
#cython: wraparound = False
#cython: cdivision = True

'''
Univariate lowess function, like in R.

References
----------
Hastie, Tibshirani, Friedman. (2009) The Elements of Statistical Learning: Data
Mining, Inference, and Prediction, Second Edition: Chapter 6.

Cleveland, W.S. (1979) "Robust Locally Weighted Regression and Smoothing
Scatterplots". Journal of the American Statistical Association 74 (368): 829-836.
'''

cimport numpy as np
import numpy as np
from cpython cimport bool
cimport cython

# there's no fmax in math.h with windows SDK apparently
cdef inline double fmax(double x, double y): return x if x >= y else y

DTYPE = np.double
ctypedef np.double_t DTYPE_t

def lowess(np.ndarray[DTYPE_t, ndim = 1] endog,
           np.ndarray[DTYPE_t, ndim = 1] exog,
           double frac = 2.0 / 3.0,
           Py_ssize_t it = 3,
           double delta = 0.0):
    '''lowess(endog, exog, frac=2.0/3.0, it=3, delta=0.0)
    LOWESS (Locally Weighted Scatterplot Smoothing)

    A lowess function that outs smoothed estimates of endog
    at the given exog values from points (exog, endog)

    Parameters
    ----------
    endog : 1-D numpy array
        The y-values of the observed points
    exog : 1-D numpy array
        The x-values of the observed points. exog has to be increasing.
    frac : float
        Between 0 and 1. The fraction of the data used
        when estimating each y-value.
    it : int
        The number of residual-based reweightings
        to perform.
    delta : float
        Distance within which to use linear-interpolation
        instead of weighted regression.

    Returns
    -------
    out : numpy array
        A numpy array with two columns. The first column
        is the sorted x values and the second column the
        associated estimated y-values.

    Notes
    -----
    This lowess function implements the algorithm given in the
    reference below using local linear estimates.

    Suppose the input data has N points. The algorithm works by
    estimating the `smooth` y_i by taking the frac*N closest points
    to (x_i,y_i) based on their x values and estimating y_i
    using a weighted linear regression. The weight for (x_j,y_j)
    is tricube function applied to |x_i-x_j|.

    If it > 1, then further weighted local linear regressions
    are performed, where the weights are the same as above
    times the _lowess_bisquare function of the residuals. Each iteration
    takes approximately the same amount of time as the original fit,
    so these iterations are expensive. They are most useful when
    the noise has extremely heavy tails, such as Cauchy noise.
    Noise with less heavy-tails, such as t-distributions with df>2,
    are less problematic. The weights downgrade the influence of
    points with large residuals. In the extreme case, points whose
    residuals are larger than 6 times the median absolute residual
    are given weight 0.

    delta can be used to save computations. For each x_i, regressions
    are skipped for points closer than delta. The next regression is
    fit for the farthest point within delta of x_i and all points in
    between are estimated by linearly interpolating between the two
    regression fits.

    Judicious choice of delta can cut computation time considerably
    for large data (N > 5000). A good choice is delta = 0.01 *
    range(exog).

    Some experimentation is likely required to find a good
    choice of frac and iter for a particular dataset.

    References
    ----------
    Cleveland, W.S. (1979) "Robust Locally Weighted Regression
    and Smoothing Scatterplots". Journal of the American Statistical
    Association 74 (368): 829-836.

    Examples
    --------
    The below allows a comparison between how different the fits from
    lowess for different values of frac can be.

    >>> import numpy as np
    >>> import statsmodels.api as sm
    >>> lowess = sm.nonparametric.lowess
    >>> x = np.random.uniform(low = -2*np.pi, high = 2*np.pi, size=500)
    >>> y = np.sin(x) + np.random.normal(size=len(x))
    >>> z = lowess(y, x)
    >>> w = lowess(y, x, frac=1./3)

    This gives a similar comparison for when it is 0 vs not.

    >>> import numpy as np
    >>> import scipy.stats as stats
    >>> import statsmodels.api as sm
    >>> lowess = sm.nonparametric.lowess
    >>> x = np.random.uniform(low = -2*np.pi, high = 2*np.pi, size=500)
    >>> y = np.sin(x) + stats.cauchy.rvs(size=len(x))
    >>> z = lowess(y, x, frac= 1./3, it=0)
    >>> w = lowess(y, x, frac=1./3)

    '''
    cdef:
        Py_ssize_t n
        int k
        Py_ssize_t robiter, i, left_end, right_end
        int last_fit_i,
        np.ndarray[DTYPE_t, ndim = 1] x, y
        np.ndarray[DTYPE_t, ndim = 1] y_fit
        np.ndarray[DTYPE_t, ndim = 1] weights
        np.ndarray[DTYPE_t, ndim = 1] resid_weights

    y = endog   # now just alias
    x = exog

    if not 0 <= frac <= 1:
           raise ValueError("Lowess `frac` must be in the range [0,1]!")

    n = x.shape[0]

    # The number of neighbors in each regression.
    # round up if close to integer
    k =  int(frac * n + 1e-10)

    # frac should be set, so that 2 <= k <= n.
    # Conform them instead of throwing error.
    if k < 2:
        k = 2
    if k > n:
        k = n

    y_fit = np.zeros(n, dtype = DTYPE)
    resid_weights = np.zeros(n, dtype = DTYPE)

    it += 1 # Add one to it for initial run.
    for robiter in xrange(it):
        i = 0
        last_fit_i = -1
        left_end = 0
        right_end = k
        y_fit = np.zeros(n, dtype = DTYPE)

        # 'do' Fit y[i]'s 'until' the end of the regression
        while True:
            # Re-initialize the weights for each point x[i].
            weights = np.zeros(n, dtype = DTYPE)

            # Describe the neighborhood around the current x[i].
            left_end, right_end, radius = update_neighborhood(x, i, n,
                                                              left_end,
                                                              right_end)

            # Calculate the weights for the regression in this neighborhood.
            # Determine if at least some weights are positive, so a regression
            # is ok.
            reg_ok = calculate_weights(x, weights, resid_weights, i, left_end,
                                       right_end, radius, robiter > 0)

            # If ok, run the regression
            calculate_y_fit(x, y, i, y_fit, weights, left_end, right_end,
                            reg_ok)

            # If we skipped some points (because of how delta was set), go back
            # and fit them by linear interpolation.
            if last_fit_i < (i - 1):
                interpolate_skipped_fits(x, y_fit, i, last_fit_i)

            # Update the last fit counter to indicate we've now fit this point.
            # Find the next i for which we'll run a regression.
            i, last_fit_i = update_indices(x, y_fit, delta, i, n, last_fit_i)

            if last_fit_i >= n-1:
                break

        # Calculate residual weights, but do not bother on the last iteration.
        if robiter < it - 1:
            resid_weights = calculate_residual_weights(y, y_fit)

    return np.array([x, y_fit]).T


def update_neighborhood(np.ndarray[DTYPE_t, ndim = 1] x,
                        Py_ssize_t i,
                        Py_ssize_t n,
                        Py_ssize_t left_end,
                        Py_ssize_t right_end):
    '''
    Find the indices bounding the k-nearest-neighbors of the current point.

    Parameters
    ----------
    x: 1-D numpy array
        The input x-values
    i: indexing integer
        The index of the point currently being fit.
    n: indexing integer
        The length of the input vectors, x and y.
    left_end: indexing integer
        The index of the left-most point in the neighborhood
        of x[i-1] (the previously-fit point).
    right_end: indexing integer
        The index of the right-most point in the neighborhood
        of x[i-1]. Non-inclusive, s.t. the neighborhood is
        x[left_end] <= x < x[right_end].
    radius: float
        The radius of the current neighborhood. The larger of
        distances between x[i] and its left-most or right-most
        neighbor.

    Returns
    -------
    left_end: indexing integer
        The index of the left-most point in the neighborhood
              of x[i] (the current point).
    right_end: indexing integer
        The index of the right-most point in the neighborhood
               of x[i]. Non-inclusive, s.t. the neighborhood is
               x[left_end] <= x < x[right_end].
    radius: float
        The radius of the current neighborhood. The larger of
        distances between x[i] and its left-most or right-most
        neighbor.
    '''

    cdef double radius
    # A subtle loop. Start from the current neighborhood range:
    # [left_end, right_end). Shift both ends rightwards by one
    # (so that the neighborhood still contains k points), until
    # the current point is in the center (or just to the left of
    # the center) of the neighborhood. This neighborhood will
    # contain the k-nearest neighbors of x[i].
    #
    # Once the right end hits the end of the data, hold the
    # neighborhood the same for the remaining x[i]s.
    while True:
        if right_end < n:

            if (x[i] > (x[left_end] + x[right_end]) / 2.0):
                left_end += 1
                right_end += 1
            else:
                break
        else:
            break

    radius = fmax(x[i] - x[left_end], x[right_end-1] - x[i])

    return left_end, right_end, radius

cdef bool calculate_weights(np.ndarray[DTYPE_t, ndim = 1] x,
                            np.ndarray[DTYPE_t, ndim = 1] weights,
                            np.ndarray[DTYPE_t, ndim = 1] resid_weights,
                            Py_ssize_t i,
                            Py_ssize_t left_end,
                            Py_ssize_t right_end,
                            double radius,
                            bool use_resid_weights):
    '''

    Parameters
    ----------
    x: 1-D vector
        The input x-values.
    weights: 1-D numpy array
        The vector of regression weights.
    resid_weights: 1-D numpy array
        The vector of residual weights from the last iteration.
    i: indexing integer
        The index of the point currently being fit.
    left_end: indexing integer
        The index of the left-most point in the neighborhood of
        x[i].
    right_end: indexing integer
        The index of the right-most point in the neighborhood
        of x[i]. Non-inclusive, s.t. the neighborhood is
        x[left_end] <= x < x[right_end].
    radius: float
        The radius of the current neighborhood. The larger of
        distances between x[i] and its left-most or right-most
        neighbor.
    use_resid_weights: bool
        If True, multiply the x-distance weights by the residual
        weights from the last iteration of regressions. Set to
        False on the first iteration (since there are no residuals
        yet) and True on the subsequent ``robustifying`` iterations.


    Returns
    -------
    reg_ok: bool
        If True, at least some points have positive weight, and the
        regression will be run. If False, the regression is skipped
        and y_fit[i] is set to equal y[i].
    Also, changes elements of weights in-place.
    '''

    cdef:
        np.ndarray[DTYPE_t, ndim = 1] x_j = x[left_end:right_end]
        np.ndarray[DTYPE_t, ndim = 1] dist_i_j = np.abs(x_j - x[i]) / radius
        bool reg_ok = True
        double sum_weights

    # Assign the distance measure to the weights, then apply the tricube
    # function to change in-place.
    # use_resid_weights will be False on the first iteration, then True
    # on the subsequent ones, after some residuals have been calculated.
    weights[left_end:right_end] = dist_i_j
    if use_resid_weights == False:
        tricube(weights[left_end:right_end])
    if use_resid_weights == True:
        tricube(weights[left_end:right_end])
        weights[left_end:right_end] = (weights[left_end:right_end] *
                                          resid_weights[left_end:right_end])

    sum_weights = np.sum(weights[left_end:right_end])

    if sum_weights <= 0.0 or (np.sum(weights[left_end:right_end] != 0) == 1):
        # 2nd condition checks if only 1 local weight is non-zero, which
        # will give a divisor of zero in calculate_y_fit
        # see 1960
        reg_ok = False
    else:
        weights[left_end:right_end] = weights[left_end:right_end] / sum_weights

    return reg_ok


cdef void calculate_y_fit(np.ndarray[DTYPE_t, ndim = 1] x,
                          np.ndarray[DTYPE_t, ndim = 1] y,
                          Py_ssize_t i,
                          np.ndarray[DTYPE_t, ndim = 1] y_fit,
                          np.ndarray[DTYPE_t, ndim = 1] weights,
                          Py_ssize_t left_end,
                          Py_ssize_t right_end,
                          bool reg_ok):
    '''
    Calculate smoothed/fitted y-value by weighted regression.

    Parameters
    ----------
    x: 1-D numpy array
        The vector of input x-values.
    y: 1-D numpy array
        The vector of input y-values.
    i: indexing integer
        The index of the point currently being fit.
    y_fit: 1-D numpy array
        The vector of fitted y-values.
    weights: 1-D numpy array
        The vector of regression weights.
    left_end: indexing integer
        The index of the left-most point in the neighborhood of
        x[i].
    right_end: indexing integers
        The index of the right-most point in the neighborhood
        of x[i]. Non-inclusive, s.t. the neighborhood is
        x[left_end] <= x < x[right_end].
    reg_ok: bool
        If True, at least some points have positive weight, and the
        regression will be run. If False, the regression is skipped
        and y_fit[i] is set to equal y[i].

    Returns
    -------
    Nothing. Changes y_fit[i] in-place.

    Notes
    -----
    No regression function (e.g. lstsq) is called. Instead "projection
    vector" p_i_j is calculated, and y_fit[i] = sum(p_i_j * y[j]) = y_fit[i]
    for j s.t. x[j] is in the neighborhood of x[i]. p_i_j is a function of
    the weights, x[i], and its neighbors.
    '''

    cdef:
       double sum_weighted_x = 0, weighted_sqdev_x = 0, p_i_j

    if reg_ok == False:
        y_fit[i] = y[i]
    else:
        for j in xrange(left_end, right_end):
            sum_weighted_x += weights[j] * x[j]
        for j in xrange(left_end, right_end):
            weighted_sqdev_x += weights[j] * (x[j] - sum_weighted_x) ** 2
        for j in xrange(left_end, right_end):
            p_i_j = weights[j] * (1.0 + (x[i] - sum_weighted_x) *
                             (x[j] - sum_weighted_x) / weighted_sqdev_x)
            y_fit[i] += p_i_j * y[j]

cdef void interpolate_skipped_fits(np.ndarray[DTYPE_t, ndim = 1] x,
                                   np.ndarray[DTYPE_t, ndim = 1] y_fit,
                                   Py_ssize_t i,
                                   Py_ssize_t last_fit_i):
    '''
    Calculate smoothed/fitted y by linear interpolation between the current
    and previous y fitted by weighted regression.
    Called only if delta > 0.

    Parameters
    ----------
    x: 1-D numpy array
        The vector of input x-values.
    y_fit: 1-D numpy array
        The vector of fitted y-values
    i: indexing integer
        The index of the point currently being fit by weighted
        regression.
    last_fit_i: indexing integer
        The index of the last point fit by weighted regression.

    Returns
    -------
    Nothing: changes elements of y_fit in-place.
    '''

    cdef np.ndarray[DTYPE_t, ndim = 1] a

    a = x[(last_fit_i + 1): i] - x[last_fit_i]
    a =  a / (x[i] - x[last_fit_i])
    y_fit[(last_fit_i + 1): i] = a * y_fit[i] + (1.0 - a) * y_fit[last_fit_i]


def update_indices(np.ndarray[DTYPE_t, ndim = 1] x,
                   np.ndarray[DTYPE_t, ndim = 1] y_fit,
                   double delta,
                   Py_ssize_t i,
                   Py_ssize_t n,
                   Py_ssize_t last_fit_i):
    '''
    Update the counters of the local regression.

    Parameters
    ----------
    x : 1-D numpy array
        The vector of input x-values.
    y_fit : 1-D numpy array
        The vector of fitted y-values
    delta : float
        Indicates the range of x values within which linear
        interpolation should be used to estimate y_fit instead
        of weighted regression.
    i : indexing integer
        The index of the current point being fit.
    n : indexing integer
        The length of the input vectors, x and y.
    last_fit_i : indexing integer
        The last point at which y_fit was calculated.

    Returns
    -------
    i : indexing integer
        The next point at which to run a weighted regression.
    last_fit_i : indexing integer
        The updated last point at which y_fit was calculated

    Notes
    -----
    The relationship between the outputs is s.t. x[i+1] >
    x[last_fit_i] + delta.

    '''
    cdef:
        Py_ssize_t k
        double cutpoint

    last_fit_i = i
    k = last_fit_i
    # For most points within delta of the current point, we skip the
    # weighted linear regression (which save much computation of
    # weights and fitted points). Instead, we'll jump to the last
    # point within delta, fit the weighted regression at that point,
    # and linearly interpolate in between.

    # This loop increments until we fall just outside of delta distance,
    # copying the results for any repeated x's along the way.
    cutpoint = x[last_fit_i] + delta
    for k in range(last_fit_i + 1, n):
        if x[k] > cutpoint:
            break
        if x[k] == x[last_fit_i]:
            # if tied with previous x-value, just use the already
            # fitted y, and update the last-fit counter.
            y_fit[k] = y_fit[last_fit_i]
            last_fit_i = k

    # i, which indicates the next point to fit the regression at, is
    # either one prior to k (since k should be the first point outside
    # of delta) or is just incremented + 1 if k = i+1. This insures we
    # always step forward.
    i = max(k-1, last_fit_i + 1)

    return i, last_fit_i


def calculate_residual_weights(np.ndarray[DTYPE_t, ndim = 1] y,
                              np.ndarray[DTYPE_t, ndim = 1] y_fit):
    '''
    Calculate residual weights for the next `robustifying` iteration.

    Parameters
    ----------
    y: 1-D numpy array
        The vector of actual input y-values.
    y_fit: 1-D numpy array
        The vector of fitted y-values from the current
        iteration.

    Returns
    -------
    resid_weights: 1-D numpy array
        The vector of residual weights, to be used in the
        next iteration of regressions.
    '''

    std_resid = np.abs(y - y_fit)
    median = np.median(std_resid)
    if median == 0:
        std_resid[std_resid > 0] = 1
    else:
        std_resid /= 6.0 * median

    # Some trimming of outlier residuals.
    std_resid[std_resid >= 1.0] = 1.0
    #std_resid[std_resid >= 0.999] = 1.0
    #std_resid[std_resid <= 0.001] = 0.0

    resid_weights = bisquare(std_resid)

    return resid_weights


cdef void tricube(np.ndarray[DTYPE_t, ndim = 1] x):
    '''
    The tri-cubic function (1 - x**3)**3. Used to weight neighboring
    points along the x-axis based on their distance to the current point.

    Parameters
    ----------
    x: 1-D numpy array
        A vector of neighbors` distances from the current point,
        in units of the neighborhood radius.

    Returns
    -------
    Nothing. Changes array elements in-place
    '''

    # fast_array_cube is an elementwise, in-place cubed-power
    # operator.
    fast_array_cube(x)
    x[:] = np.negative(x)
    x += 1
    fast_array_cube(x)


cdef void fast_array_cube(np.ndarray[DTYPE_t, ndim = 1] x):
    '''
    A fast, elementwise, in-place cube operator. Called by the
    tricube function.

    Parameters
    ----------
    x: 1-D numpy array

    Returns
    -------
    Nothing. Changes array elements in-place.
    '''

    x2 = x*x
    x *= x2


def bisquare(np.ndarray[DTYPE_t, ndim = 1] x):
    '''
    The bi-square function (1 - x**2)**2.

    Used to weight the residuals in the `robustifying`
    iterations. Called by the calculate_residual_weights function.

    Parameters
    ----------
    x: 1-D numpy array
        A vector of absolute regression residuals, in units of
        6 times the median absolute residual.

    Returns
    -------
    A 1-D numpy array of residual weights.
    '''

    return (1.0 - x**2)**2
