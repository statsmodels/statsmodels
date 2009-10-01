
'''using scipy signal and numpy correlate to calculate some time series
statistics

see also scikits.timeseries  (movstat is partially inspired by it)
(added 2009-08-29:
timeseries moving stats are in c, autocorrelation similar to here
I thought I saw moving stats somewhere in python, maybe not)


TODO:

moving statistics
* filters don't handle boundary conditions nicely (correctly ?)
  e.g. minimum order filter uses 0 for out of bounds value
  -> append and prepend with last resp. first value
* enhance for nd arrays, with axis = 0



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


update:
2009-09-06: cosmetic changes, rearrangements
'''


import numpy as np
from scipy import signal

from numpy.testing import assert_array_equal, assert_array_almost_equal

import scikits.statsmodels as sm


def expandarr(x,k):
    #make it work for 2D or nD with axis
    kadd = k
    if np.ndim(x) == 2:
        kadd = (kadd, np.shape(x)[1])
    return np.r_[np.ones(kadd)*x[0],x,np.ones(kadd)*x[-1]]

def movorder(x, order = 'med', windsize=3, lag='lagged'):
    '''moving order statistics

    Parameters
    ----------
    x : array
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


    '''

    #if windsize is even should it raise ValueError
    if lag == 'lagged':
        lead = windsize//2
    elif lag == 'centered':
        lead = 0
    elif lag == 'leading':
        lead = -windsize//2 +1
    else:
        raise ValueError
    if np.isfinite(order) == True: #if np.isnumber(order):
        ord = order   # note: ord is a builtin function
    elif order == 'med':
        ord = (windsize - 1)/2
    elif order == 'min':
        ord = 0
    elif order == 'max':
        ord = windsize - 1
    else:
        raise ValueError

    #return signal.order_filter(x,np.ones(windsize),ord)[:-lead]
    xext = expandarr(x, windsize)
    #np.r_[np.ones(windsize)*x[0],x,np.ones(windsize)*x[-1]]
    return signal.order_filter(xext,np.ones(windsize),ord)[windsize-lead:-(windsize+lead)]

def check_movorder():
    '''graphical test for movorder'''
    import matplotlib.pylab as plt
    x = np.arange(1,10)
    xo = movorder(x, order='max')
    assert_array_equal(xo, x)
    x = np.arange(10,1,-1)
    xo = movorder(x, order='min')
    assert_array_equal(xo, x)
    assert_array_equal(movorder(x, order='min', lag='centered')[:-1], x[1:])

    tt = np.linspace(0,2*np.pi,15)
    x = np.sin(tt) + 1
    xo = movorder(x, order='max')
    plt.figure()
    plt.plot(tt,x,'.-',tt,xo,'.-')
    plt.title('moving max lagged')
    xo = movorder(x, order='max', lag='centered')
    plt.figure()
    plt.plot(tt,x,'.-',tt,xo,'.-')
    plt.title('moving max centered')
    xo = movorder(x, order='max', lag='leading')
    plt.figure()
    plt.plot(tt,x,'.-',tt,xo,'.-')
    plt.title('moving max leading')

# identity filter
##>>> signal.order_filter(x,np.ones(1),0)
##array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])
# median filter
##signal.medfilt(np.sin(x), kernel_size=3)
##>>> plt.figure()
##<matplotlib.figure.Figure object at 0x069BBB50>
##>>> x=np.linspace(0,3,100);plt.plot(x,np.sin(x),x,signal.medfilt(np.sin(x), kernel_size=3))

# remove old version
##def movmeanvar(x, windowsize=3, valid='same'):
##    '''
##    this should also work along axis or at least for columns
##    '''
##    n = x.shape[0]
##    x = expandarr(x, windowsize - 1)
##    takeslice = slice(windowsize-1, n + windowsize-1)
##    avgkern = (np.ones(windowsize)/float(windowsize))
##    m = np.correlate(x, avgkern, 'same')#[takeslice]
##    print m.shape
##    print x.shape
##    xm = x - m
##    v = np.correlate(x*x, avgkern, 'same') - m**2
##    v1 = np.correlate(xm*xm, avgkern, valid) #not correct for var of window
###>>> np.correlate(xm*xm,np.array([1,1,1])/3.0,'valid')-np.correlate(xm*xm,np.array([1,1,1])/3.0,'valid')**2
##    return m[takeslice], v[takeslice], v1

def movmean(x, windowsize=3, lag='lagged'):
    return movmoment(x, 1, windowsize=windowsize, lag=lag)

def movvar(x, windowsize=3, lag='lagged'):
    m1 = movmoment(x, 1, windowsize=windowsize, lag=lag)
    m2 = movmoment(x, 2, windowsize=windowsize, lag=lag)
    return m2 - m1*m1

def movmoment(x, k, windowsize=3, lag='lagged'):
    '''non-central moment


    Parameters
    ----------
    x : array
       time series data
    windsize : int
       window size
    lag : 'lagged', 'centered', or 'leading'
       location of window relative to current position

    Returns
    -------
    mk : array
        k-th moving non-central moment, with same shape as x


    '''

    windsize = windowsize
    #if windsize is even should it raise ValueError
    if lag == 'lagged':
        lead = windsize//2
    elif lag == 'centered':
        lead = 0
    elif lag == 'leading':
        lead = -windsize//2 +1
    else:
        raise ValueError

    avgkern = (np.ones(windowsize)/float(windowsize))
    xext = expandarr(x, windsize-1)

    if xext.ndim == 1:
        return np.correlate(xext**k, avgkern, 'full')[windsize-lead:-(windsize+lead)]
    else:
        print xext.shape
        print avgkern[:,None].shape

        # try first with 2d along columns, possibly ndim with axis
        return signal.correlate(xext**k, avgkern[:,None], 'full')[windsize-lead:-(windsize+lead),:]


#None of the acovf, ... are tested; starting index? orientation?
def acovf(x, unbiased=True):
    ''' autocovariance for 1D
    '''
    n = len(x)
    xo = x - x.mean();
    if unbiased:
        xi = np.ones(n);
        d = np.correlate(xi, xi, 'full')
    else:
        d = n
    return (np.correlate(xo, xo, 'full') / d)[n-1:]

def ccovf(x, y, unbiased=True):
    ''' crosscovariance for 1D
    '''
    n = len(x)
    xo = x - x.mean();
    yo = y - y.mean();
    if unbiased:
        xi = np.ones(n);
        d = np.correlate(xi, xi, 'full')
    else:
        d = n
    return (np.correlate(xo,yo,'full') / d)[n-1:]

def acf(x, unbiased=True):
    '''autocorrelation function for 1d'''
    avf = acovf(x, unbiased=unbiased)
    return avf/avf[0]

def ccf(x, y, unbiased=True):
    '''cross-correlation function for 1d'''
    cvf = ccovf(x, y, unbiased=unbiased)
    return cvf / (np.std(x) * np.std(y))


def pacf_yw(x, maxlag=20, method='unbiased'):
    '''Partial autocorrelation estimated with non-recursive yule_walker

    Parameters
    ----------
    x : 1d array
        observations of time series for which pacf is calculated
    maxlag : int
        largest lag for which pacf is returned
    method : 'unbiased' (default) or 'mle'
        method for the autocovariance calculations in yule walker

    Returns
    -------
    pacf : 1d array
        partial autocorrelations, maxlag+1 elements

    Notes
    -----

    '''
    xm = x - x.mean()
    pacf = [1.]
    for k in range(1, maxlag+1):
        pacf.append(sm.regression.yule_walker(x, k, method=method)[0][-1])
    return np.array(pacf)

def pacf_ols(x, maxlag=20):
    '''Partial autocorrelation estimated with non-recursive OLS

    Parameters
    ----------
    x : 1d array
        observations of time series for which pacf is calculated
    maxlag : int
        largest lag for which pacf is returned

    Returns
    -------
    pacf : 1d array
        partial autocorrelations, maxlag+1 elements
    '''
    from scikits.statsmodels.sandbox.tools.tools_tsa import lagmat
    xlags = lagmat(x-x.mean(), maxlag)
    pacfols = [1.]
    for k in range(1, maxlag+1):
        res = sm.OLS(xlags[k:,0], xlags[k:,1:k+1]).fit()
        #print res.params
        pacfols.append(res.params[-1])
    return np.array(pacfols)



#x=0.5**np.arange(10);xm=x-x.mean();a=np.correlate(xm,[1],'full')
#x=0.5**np.arange(3);np.correlate(x,x,'same')
##>>> x=0.5**np.arange(10);xm=x-x.mean();a=np.correlate(xm,xo,'full')
##
##>>> xo=np.ones(10);d=np.correlate(xo,xo,'full')
##>>> xo
##xo=np.ones(10);d=np.correlate(xo,xo,'full')
##>>> x=np.ones(10);xo=x-x.mean();a=np.correlate(xo,xo,'full')
##>>> xo=np.ones(10);d=np.correlate(xo,xo,'full')
##>>> d
##array([  1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,   9.,
##         8.,   7.,   6.,   5.,   4.,   3.,   2.,   1.])


##def ccovf():
##    pass
##    #x=0.5**np.arange(10);xm=x-x.mean();a=np.correlate(xm,xo,'full')

__all__ = ['movorder', 'movmean', 'movvar', 'movmoment', 'acovf', 'ccovf',
           'acf', 'ccf', 'pacf_yw', 'pacf_ols']

if __name__ == '__main__':


    T = 20
    K = 2
    x = np.column_stack([np.arange(T)]*K)
    aav = acovf(x[:,0])
    print aav[0] == np.var(x[:,0])
    aac = acf(x[:,0])



    print '\ncheckin moving mean and variance'
    nobs = 10
    x = np.arange(nobs)
    ws = 3
    ave = np.array([ 0., 1/3., 1., 2., 3., 4., 5., 6., 7., 8.,
                  26/3., 9])
    va = np.array([[ 0.        ,  0.        ],
                   [ 0.22222222,  0.88888889],
                   [ 0.66666667,  2.66666667],
                   [ 0.66666667,  2.66666667],
                   [ 0.66666667,  2.66666667],
                   [ 0.66666667,  2.66666667],
                   [ 0.66666667,  2.66666667],
                   [ 0.66666667,  2.66666667],
                   [ 0.66666667,  2.66666667],
                   [ 0.66666667,  2.66666667],
                   [ 0.22222222,  0.88888889],
                   [ 0.        ,  0.        ]])
    ave2d = np.c_[ave, 2*ave]
    print movmean(x, windowsize=ws, lag='lagged')
    print movvar(x, windowsize=ws, lag='lagged')
    print [np.var(x[i-ws:i]) for i in range(ws, nobs)]
    m1 = movmoment(x, 1, windowsize=3, lag='lagged')
    m2 = movmoment(x, 2, windowsize=3, lag='lagged')
    print m1
    print m2
    print m2 - m1*m1

    # this implicitly also tests moment
    assert_array_almost_equal(va[ws-1:,0],
                    movvar(x, windowsize=3, lag='leading'))
    assert_array_almost_equal(va[ws//2:-ws//2+1,0],
                    movvar(x, windowsize=3, lag='centered'))
    assert_array_almost_equal(va[:-ws+1,0],
                    movvar(x, windowsize=ws, lag='lagged'))



    print '\nchecking moving moment for 2d (columns only)'
    x2d = np.c_[x, 2*x]
    print movmoment(x2d, 1, windowsize=3, lag='centered')
    print movmean(x2d, windowsize=ws, lag='lagged')
    print movvar(x2d, windowsize=ws, lag='lagged')
    assert_array_almost_equal(va[ws-1:,:],
                    movvar(x2d, windowsize=3, lag='leading'))
    assert_array_almost_equal(va[ws//2:-ws//2+1,:],
                    movvar(x2d, windowsize=3, lag='centered'))
    assert_array_almost_equal(va[:-ws+1,:],
                    movvar(x2d, windowsize=ws, lag='lagged'))

    assert_array_almost_equal(ave2d[ws-1:],
                    movmoment(x2d, 1, windowsize=3, lag='leading'))
    assert_array_almost_equal(ave2d[ws//2:-ws//2+1],
                    movmoment(x2d, 1, windowsize=3, lag='centered'))
    assert_array_almost_equal(ave2d[:-ws+1],
                    movmean(x2d, windowsize=ws, lag='lagged'))

    from scipy import ndimage
    print ndimage.filters.correlate1d(x2d, np.array([1,1,1])/3., axis=0)
