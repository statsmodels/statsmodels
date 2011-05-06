"""
Univariate Kernel Density Estimators

References
----------
Racine, Jeff. (2008) "Nonparametric Econometrics: A Primer," Foundation and
    Trends in Econometrics: Vol 3: No 1, pp1-88.
    http://dx.doi.org/10.1561/0800000009

http://en.wikipedia.org/wiki/Kernel_%28statistics%29

Silverman, B.W.  Density Estimation for Statistics and Data Anaylsis.
"""
import numpy as np
#class KDensity(object):
#    def __init__(self, X, kernel="epa" bwidth=""):

#def subset(x, limit):
#    """
#    Returns selector for X, given limit.
#
#    """
    #TODO: finish docstring.
#    return np.

def linbin(X,a,b,M, trunc=1):
    """
    Linear Binning as described in Fan and Marron (1994)
    """
    gcnts = np.zeros(M)
    delta = (b-a)/(M-1)

    for x in X:
        lxi = ((x - a)/delta) # +1
        li = int(lxi)
        rem = lxi - li
        if li > 1 and li < M:
            gcnts[li] = gcnts[li] + 1-rem
            gcnts[li+1] = gcnts[li+1] + rem
        if li > M and trunc == 0:
            gcnts[M] = gncts[M] + 1

    return gcnts

#This is much faster, might be able to speed it up still?
def counts(x,v):
    """
    Counts the number of elements of x that fall within v

    Notes
    -----
    Using np.digitize and np.bincount
    """
    idx = np.digitize(x,v)
    return np.bincount(idx, minlength=len(v))

#def counts(x,v):
#    """
#    Counts the number of elements of x that fall within v
#
#    Notes
#    -----
#    Flattens both x and v.
#    """
#    x = x.flatten()
#    v = v.flatten()
#    c = []
#    nv = np.r_[-np.inf,v]
#    for i in range(1,len(nv)):
#        t = np.sum(np.logical_and(x<=nv[i],x>nv[i-1]))
#        c.append(t)
#    return np.asarray(c)


def kdesum(x,axis=0):
    return np.asarray([np.sum(x[i] - x, axis) for i in range(len(x))])

def kdensity(X, kernel="epa", bw=None, weights=None, gridsize=None, axis=0, clip=(-np.inf,np.inf), cut=3):
    """
    Rosenblatz-Parzen univariate kernel desnity estimator

    Parameters
    ----------
    X : array-like
    kernel : str
        "bi" for biweight
        "cos" for cosine
        "epa" for Epanechnikov, default
        "epa2" for alternative Epanechnikov
        "gauss" for Gaussian.
        "par" for Parzen
        "rect" for rectangular
        "tri" for triangular
    bw : str, int
        If None, the bandwidth uses the rule of thumb for the given kernel.
        ie., h = c*nobs**(-1/5.) where c = (see Racine 2.6)
        gridsize : int
        If gridsize is None, min(len(X), 512) is used.  Note that this number
        is rounded up to the next highest power of 2.

    Notes
    -----
    Weights aren't implemented yet.
    Does not use FFT.
    Should actually only work for 1 column.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:,None]
    X = X[np.logical_and(X>clip[0], X<clip[1])] # won't work for two columns.
                                                # will affect underlying data?
    nobs = float(len(X)) # after trim

    # if bw is None, select optimal bandwidth for kernel
    if bw == None:
        if kernel.lower() == "gauss":
            c = 1.0592 * np.std(X, axis=axis, ddof=1)
        if kernel.lower() == "epa":
            c = 1.0487 * np.std(X, axis=axis, ddof=1) # is this correct?
#TODO: can use windows from scipy.signal?
        h = c * nobs**(-1/5.)
    else:
        h = bw

    if gridsize == None:
        gridsize = np.max((nobs,512.))
    # round gridsize up to the next power of 2
    gridsize = 2**np.ceil(np.log2(gridsize))
    # define mesh
    grid = np.linspace(np.min(X,axis) - cut*bw,np.max(X,axis)+cut*bw,gridsize)
    # this will fail for not 1 column
    if grid.ndim == 1:
        grid = grid[:,None]

    k = (X.T - grid)/h  # uses broadcasting
# res = np.repeat(x,n).reshape(m,n).T - np.repeat(xi,m).reshape(n,m))/h
    if kernel.lower() == "epa":
        k = np.zeros_like(grid) + np.less_equal(np.abs(k),
                np.sqrt(5)) * 3/(4*np.sqrt(5)) * (1-.2*k**2)
#        k = (.15/np.sqrt(5))*(5-k**2)/h
#        k[k<0] = 0
    if kernel.lower() == "gauss":
        k = 1/np.sqrt(2*np.pi)*np.exp(-.5*k**2)
#        k = np.clip(k,1e12,0)
#TODO:
    if weights == None:
        q = nobs
        q = 1
        weights = 1
    return np.mean(1/(q*h)*weights*k,1),k/(q*h)*weights
#TODO: need to check this
#    return k.mean(1),k

def kdensityf(X, kernel="epa", bw=None, weights=None, gridsize=None, clip=(-np.inf,np.inf), cut=3):
    """
    Rosenblatz-Parzen univariate kernel desnity estimator

    Parameters
    ----------
    X : array-like
    kernel : str
        "bi" for biweight
        "cos" for cosine
        "epa" for Epanechnikov, default
        "epa2" for alternative Epanechnikov
        "gauss" for Gaussian.
        "par" for Parzen
        "rect" for rectangular
        "tri" for triangular
    bw : str, int
        If None, the bandwidth uses the rule of thumb for the given kernel.
        ie., h = c*nobs**(-1/5.) where c = (see Racine 2.6)
        gridsize : int
        If gridsize is None, min(len(X), 512) is used.  Note that this number
        is rounded up to the next highest power of 2.

    Notes
    -----
    Weights aren't implemented yet.
    Uses FFT.
    Should actually only work for 1 column.
    DOesn't work yet
    """
    X = np.asarray(X)
    X = X[np.logical_and(X>clip[0], X<clip[1])] # won't work for two columns.
                                                # will affect underlying data?
    nobs = float(len(X)) # after trim
    if bw == None:
        if kernel.lower() == "gauss":
            c = 1.0592 * np.std(X, ddof=1)
        if kernel.lower() == "epa":
            c = 1.0487 * np.std(X, ddof=1) # is this correct?
#TODO: can use windows from scipy.signal?
        bw = c * nobs**(-1/5.)


    if gridsize == None:
        gridsize = np.max((nobs,512.))
    # round gridsize up to the next power of 2
    gridsize = 2**np.ceil(np.log2(gridsize))

    # Discretize the data on an M-element grid over [a,b] to find the weight seq.
    # define grid
    grid,delta = np.linspace(np.min(X)-cut*bw,np.max(X)+cut*bw,gridsize,retstep=True)
    RANGE = np.max(X)+cut*bw - np.min(X)-cut*bw
    # sort the data
    X.sort(0)
    # how fine is the data vis-a-vis the grid?
    count = counts(X,grid)

    # make a weights array
# weighting according to Silverman
    wt = np.zeros_like(grid)    #xi_{k} in Silverman
    j = 0
    for k in range(int(gridsize-1)):
        if count[k]>0: # there are points of X in the grid here
            Xingrid = X[j:j+count[k]] # get all these points
            # get weights at grid[k],grid[k+1]
            wt[k] += np.sum(grid[k+1]-Xingrid)
            wt[k+1] += np.sum(Xingrid-grid[k])
            j += count[k]
    wt /= (nobs)*delta**2 # normalize wt to sum to 1/delta

# Weights according to Hall and Jones
# still use count
#    wt = np.zeros_like(grid[:gridsize/2])


#    print nobs
#    print len(wt)
    assert np.allclose(np.sum(wt), 1/delta)

    # step 2 compute FFT of the weights
#    y = np.fft.rfft(wt) # RFFT uses opposite sign vs. Silverman and doesn't
                        # normalize by len(wt)
# so Silverman's definition corresponds to
    y = np.fft.irfft(wt, n = gridsize)
#    y = np.fft.ifft(wt, n = gridsize)
    # put in order expected
    # see Monro (1976) AS 97 (FORRT) real parts first then the imaginary parts.
    # brute force
    #   don't have to reorder if you use irfft
#    y_neworder = np.zeros_like(grid)
#    for i in range(int(gridsize)):
#        if i <= gridsize/2:
#            y_neworder[i] = np.real(y[i])
#        else:
#            y_neworder[i] = np.imag(y[i-(gridsize/2)])

    ell = np.arange(-gridsize/2.,gridsize/2.)
    s = 2*ell*np.pi/RANGE

    # step 3 and 4 for optimal bandwidt compute zstar and the density estimate f
#   3.59

# if use irfft then you don't have to reorder
    zstar = np.exp(-.5*bw**2*s**2)*y
#    zstar = np.exp(-.5*bw**2*s**2)*y_neworder
    f = np.fft.rfft(zstar)
# and given the defintion in Silverman and numpy then f should be
#    f = np.real(np.fft.fft(zstar))
    return f

if __name__ == "__main__":
    import numpy as np
    np.random.seed(12345)
    xi = np.random.randn(100)
    f,k = kdensity(xi, kernel="gauss", bw=.372735)
    f2 = kdensityf(xi, kernel="gauss")

# do Fourier method by hand
    X = xi.copy()
    nobs = len(X)
#    c = 1.0592 * np.std(X, axis=0, ddof=1)
    c = .9 * np.std(X, axis=0, ddof=1)
    bw = c * nobs ** -.2
    gridsize = 512
    cut = 3
    a,b = X.min() - cut*bw, X.max()+cut*bw
    grid,delta = np.linspace(a,b,gridsize,retstep=True)
    wt = np.zeros_like(grid)    #xi_{k} in Silverman
    j = 0
    count = counts(X,grid)
    RANGE = np.max(X)+cut*bw - np.min(X)-cut*bw
    for k in range(int(gridsize-1)):
        if count[k]>0: # there are points of X in the grid here
            Xingrid = X[j:j+count[k]] # get all these points
            # get weights at grid[k],grid[k+1]
            wt[k] += np.sum(grid[k+1]-Xingrid)
            wt[k+1] += np.sum(Xingrid-grid[k])
            j += count[k]
    wt /= (nobs)*delta**2 # normalize wt to sum to 1/delta
    y = np.fft.ifft(wt, n = gridsize)
    ell = np.arange(-gridsize/2.,gridsize/2.)
    s = 2*ell*np.pi/RANGE
    zstar = np.exp(-.5*bw**2*s**2)*y
    f2 = np.fft.fft(zstar)



"""
    # Hall and Jones example
    c = [4,2,1,2,0,0,0,0]
    kappa = [7,5,2,0,0,0,2,5]

    C = np.fft.fft(c)
    K = np.fft.fft(kappa)
    F_tilde = C*K
    f_tilde = np.fft.ifft(F_tilde)
    # only keep first 4 because last 4 have wrap-around error
    f = f_tilde[:4]
    # this is our density estimate.
    # can this be done with fftconvolve?
    #from scipy import signal
#TODO: note that scipy.fftpack looks to be more "standard" notation
"""
