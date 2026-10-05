"""using scipy signal and numpy correlate to calculate some time series
statistics

original developer notes

see also scikits.timeseries  (movstat is partially inspired by it)
added 2009-08-29
timeseries moving stats are in c, autocorrelation similar to here
I thought I saw moving stats somewhere in python, maybe not)


TODO

moving statistics
- filters do not handle boundary conditions nicely (correctly ?)
e.g., minimum order filter uses 0 for out of bounds value
-> append and prepend with last resp. first value
- enhance for nd arrays, with axis = 0



Note: Equivalence for 1D signals
>>> np.all(signal.correlate(x,[1,1,1],'valid')==np.correlate(x,[1,1,1]))
True
>>> np.all(ndimage.filters.correlate(x,[1,1,1], origin = -1)[:-3+1]==np.correlate(x,[1,1,1]))
True

# multidimensional, but, it looks like it uses common filter across time series, no VAR
ndimage.filters.correlate(np.vstack([x,x]),np.array([[1,1,1],[0,0,0]]), origin = 1)
ndimage.filters.correlate(x,[1,1,1],origin = 1))
ndimage.filters.correlate(np.vstack([x,x]),np.array([[0.5,0.5,0.5],[0.5,0.5,0.5]]), \
origin = 1)

>>> np.all(ndimage.filters.correlate(np.vstack([x,x]),np.array([[1,1,1],[0,0,0]]), origin = 1)[0]==\
ndimage.filters.correlate(x,[1,1,1],origin = 1))
True
>>> np.all(ndimage.filters.correlate(np.vstack([x,x]),np.array([[0.5,0.5,0.5],[0.5,0.5,0.5]]), \
origin = 1)[0]==ndimage.filters.correlate(x,[1,1,1],origin = 1))


update
2009-09-06: cosmetic changes, rearrangements
"""

import numpy as np
from numpy.testing import assert_array_equal
from scipy import signal


def expandarr(x, k):
    # make it work for 2D or nD with axis
    kadd = k
    if np.ndim(x) == 2:
        kadd = (kadd, np.shape(x)[1])
    return np.r_[np.ones(kadd) * x[0], x, np.ones(kadd) * x[-1]]


def movorder(x, order="med", windsize=3, lag="lagged"):
    """moving order statistics

    Parameters
    ----------
    x : ndarray
       time series data
    order : float or 'med', 'min', 'max'
       which order statistic to calculate
    windsize : int
       window size
    lag : 'lagged', 'centered', or 'leading'
       location of window relative to current position

    Returns
    -------
    filtered array


    """

    # if windsize is even should it raise ValueError
    if lag == "lagged":
        lead = windsize // 2
    elif lag == "centered":
        lead = 0
    elif lag == "leading":
        lead = -windsize // 2 + 1
    else:
        raise ValueError
    if np.isfinite(order):  # if np.isnumber(order):
        ord = order  # note: ord is a builtin function
    elif order == "med":
        ord = (windsize - 1) / 2
    elif order == "min":
        ord = 0
    elif order == "max":
        ord = windsize - 1
    else:
        raise ValueError

    # return signal.order_filter(x,np.ones(windsize),ord)[:-lead]
    xext = expandarr(x, windsize)
    # np.r_[np.ones(windsize)*x[0],x,np.ones(windsize)*x[-1]]
    return signal.order_filter(xext, np.ones(windsize), ord)[
        windsize - lead : -(windsize + lead)
    ]


def check_movorder():
    """graphical test for movorder"""
    import matplotlib.pylab as plt

    x = np.arange(1, 10)
    xo = movorder(x, order="max")
    assert_array_equal(xo, x)
    x = np.arange(10, 1, -1)
    xo = movorder(x, order="min")
    assert_array_equal(xo, x)
    assert_array_equal(movorder(x, order="min", lag="centered")[:-1], x[1:])

    tt = np.linspace(0, 2 * np.pi, 15)
    x = np.sin(tt) + 1
    xo = movorder(x, order="max")
    plt.figure()
    plt.plot(tt, x, ".-", tt, xo, ".-")
    plt.title("moving max lagged")
    xo = movorder(x, order="max", lag="centered")
    plt.figure()
    plt.plot(tt, x, ".-", tt, xo, ".-")
    plt.title("moving max centered")
    xo = movorder(x, order="max", lag="leading")
    plt.figure()
    plt.plot(tt, x, ".-", tt, xo, ".-")
    plt.title("moving max leading")


# identity filter
# >>> signal.order_filter(x,np.ones(1),0)
# array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])
# median filter
# signal.medfilt(np.sin(x), kernel_size=3)
# >>> plt.figure()
# <matplotlib.figure.Figure object at 0x069BBB50>
# >>> x=np.linspace(0,3,100);plt.plot(x,np.sin(x),x,signal.medfilt(np.sin(x), kernel_size=3))

# remove old version
# def movmeanvar(x, windowsize=3, valid='same'):
#    '''
#    this should also work along axis or at least for columns
#    '''
#    n = x.shape[0]
#    x = expandarr(x, windowsize - 1)
#    takeslice = slice(windowsize-1, n + windowsize-1)
#    avgkern = (np.ones(windowsize)/float(windowsize))
#    m = np.correlate(x, avgkern, 'same')# [takeslice]
#    print(m.shape)
#    print(x.shape)
#    xm = x - m
#    v = np.correlate(x*x, avgkern, 'same') - m**2
#    v1 = np.correlate(xm*xm, avgkern, valid) # not correct for var of window
# #>>> np.correlate(xm*xm,np.array([1,1,1])/3.0,'valid')-np.correlate(xm*xm,np.array([1,1,1])/3.0,'valid')**2
#    return m[takeslice], v[takeslice], v1


def movmean(x, windowsize=3, lag="lagged"):
    """moving window mean


    Parameters
    ----------
    x : ndarray
       time series data
    windowsize : int
       window size
    lag : 'lagged', 'centered', or 'leading'
       location of window relative to current position

    Returns
    -------
    mk : ndarray
        moving mean, with same shape as x


    Notes
    -----
    for leading and lagging the data array x is extended by the closest value of the array


    """
    return movmoment(x, 1, windowsize=windowsize, lag=lag)


def movvar(x, windowsize=3, lag="lagged"):
    """moving window variance


    Parameters
    ----------
    x : ndarray
       time series data
    windowsize : int
       window size
    lag : 'lagged', 'centered', or 'leading'
       location of window relative to current position

    Returns
    -------
    mk : ndarray
        moving variance, with same shape as x


    """
    m1 = movmoment(x, 1, windowsize=windowsize, lag=lag)
    m2 = movmoment(x, 2, windowsize=windowsize, lag=lag)
    return m2 - m1 * m1


def movmoment(x, k, windowsize=3, lag="lagged"):
    """non-central moment


    Parameters
    ----------
    x : ndarray
       time series data
    k : int
       order of the moment
    windowsize : int
       window size
    lag : 'lagged', 'centered', or 'leading'
       location of window relative to current position

    Returns
    -------
    mk : ndarray
        k-th moving non-central moment, with same shape as x


    Notes
    -----
    If data x is 2d, then moving moment is calculated for each
    column.

    """

    windsize = windowsize
    # if windsize is even should it raise ValueError
    if lag == "lagged":
        # lead = -0 + windsize # windsize//2
        lead = -0  # + (windsize-1) + windsize//2
        sl = slice((windsize - 1) or None, -2 * (windsize - 1) or None)
    elif lag == "centered":
        lead = -windsize // 2  # 0#-1 #+ #(windsize-1)
        sl = slice(
            (windsize - 1) + windsize // 2 or None,
            -(windsize - 1) - windsize // 2 or None,
        )
    elif lag == "leading":
        # lead = -windsize +1#+1 #+ (windsize-1)#//2 +1
        lead = -windsize + 2  # -windsize//2 +1
        sl = slice(
            2 * (windsize - 1) + 1 + lead or None,
            -(2 * (windsize - 1) + lead) + 1 or None,
        )
    else:
        raise ValueError

    avgkern = np.ones(windowsize) / float(windowsize)
    xext = expandarr(x, windsize - 1)
    # Note: expandarr increases the array size by 2*(windsize-1)

    # sl = slice(2*(windsize-1)+1+lead or None, -(2*(windsize-1)+lead)+1 or None)
    print(sl)

    if xext.ndim == 1:
        return np.correlate(xext**k, avgkern, "full")[sl]
        # return np.correlate(xext**k, avgkern, 'same')[windsize-lead:-(windsize+lead)]
    else:
        print(xext.shape)
        print(avgkern[:, None].shape)

        # try first with 2d along columns, possibly ndim with axis
        return signal.correlate(xext**k, avgkern[:, None], "full")[sl, :]


# x=0.5**np.arange(10);xm=x-x.mean();a=np.correlate(xm,[1],'full')
# x=0.5**np.arange(3);np.correlate(x,x,'same')
# >>> x=0.5**np.arange(10);xm=x-x.mean();a=np.correlate(xm,xo,'full')
#
# >>> xo=np.ones(10);d=np.correlate(xo,xo,'full')
# >>> xo
# xo=np.ones(10);d=np.correlate(xo,xo,'full')
# >>> x=np.ones(10);xo=x-x.mean();a=np.correlate(xo,xo,'full')
# >>> xo=np.ones(10);d=np.correlate(xo,xo,'full')
# >>> d
# array([  1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,   9.,
#         8.,   7.,   6.,   5.,   4.,   3.,   2.,   1.])


# def ccovf():
#    pass
#    # x=0.5**np.arange(10);xm=x-x.mean();a=np.correlate(xm,xo,'full')

__all__ = ["movmean", "movmoment", "movorder", "movvar"]
