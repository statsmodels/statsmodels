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
import kernel as kernels
import bandwidths #TODO: change to absolute import

#### Kernels Switch for estimators ####

kernel_switch = dict(gau=kernels.Gaussian, epa=kernels.Epanechnikov,
                    uni=kernels.Uniform, tri=kernels.Triangular,
                    biw=kernels.Biweight, triw=kernels.Triweight,
                    cos=kernels.Cosine)

#### Convenience Functions to be moved to kerneltools ####

def forrt(X,m=None):
    """
    RFFT with order like Munro (1976) FORTT routine.
    """
    if m is None:
        m = len(X)
    y = np.fft.rfft(X,m)/m
    return np.r_[y.real,y[1:-1].imag]

def revrt(X,m=None):
    """
    Inverse of forrt. Equivalent to Munro (1976) REVRT routine.
    """
    if m is None:
        m = len(X)
    y = X[:m/2+1] + np.r_[0,X[m/2+1:],0]*1j
    return np.fft.irfft(y)*m

def silverman_transform(bw, M, RANGE):
    """
    FFT of Gaussian kernel following to Silverman AS 176.

    Notes
    -----
    Underflow is intentional as a dampener.
    """
    J = np.arange(M/2+1)
    FAC1 = 2*(np.pi*bw/RANGE)**2
    JFAC = J**2*FAC1
    BC = 1 - 1./3 * (J*1./M*np.pi)**2
    FAC = np.exp(-JFAC)/BC
    kern_est = np.r_[FAC,FAC[1:-1]]
    return kern_est

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

def counts(x,v):
    """
    Counts the number of elements of x that fall within the grid points v

    Notes
    -----
    Using np.digitize and np.bincount
    """
    idx = np.digitize(x,v)
    try: # numpy 1.6
        return np.bincount(idx, minlength=len(v))
    except:
        bc = np.bincount(idx)
        return np.r_[bc,np.zeros(len(v)-len(bc))]


def kdesum(x,axis=0):
    return np.asarray([np.sum(x[i] - x, axis) for i in range(len(x))])

# global dict?
bandwidth_funcs = dict(scott=bandwidths.bw_scott,silverman=bandwidths.bw_silverman)

def select_bandwidth(X, bw, kernel):
    """
    Selects bandwidth
    """
    bw = bw.lower()
    if bw not in ["scott","silverman"]:
        raise ValueError("Bandwidth %s not understood" % bw)
#TODO: uncomment checks when we have non-rule of thumb bandwidths for diff. kernels
#    if kernel == "gauss":
    return bandwidth_funcs[bw](X)
#    else:
#        raise ValueError("Only Gaussian Kernels are currently supported")

#### Kernel Density Estimator Class ###

#TODO: should be able to extend to multivariate
class KDE(object):
    """
    Kernel Density Estimator

    Parameters
    ----------
    endog : array-like
        The variable for which the density estimate is desired.
    """
    def __init__(self, endog):
        self.endog = np.asarray(endog)

    def fit(self, kernel="gau", bw="scott", fft=True, weights=None, gridsize=None,
            adjust=1, cut=3, clip=(-np.inf, np.inf)):
        """
        Attach the density estimate to the KDE class.

        Parameters
        ----------
        kernel : str
            The Kernel to be used. Choices are
            - "biw" for biweight
            - "cos" for cosine
            - "epa" for Epanechnikov
            - "gauss" for Gaussian.
            - "tri" for triangular
            - "triw" for triweight
            - "uni" for uniform
        bw : str, float
            "scott" - 1.059 * A * nobs ** (-1/5.), where A is
                    min(std(X),IQR/1.34)
            "silverman" - .9 * A * nobs ** (-1/5.), where A is
                    min(std(X),IQR/1.34)
            If a float is given, it is the bandwidth.
        fft : bool
            Whether or not to use FFT. FFT implementation is more
            computationally efficient. However, only the Gaussian kernel
            is implemented.
        gridsize : int
            If gridsize is None, max(len(X), 50) is used.
        cut : float
            Defines the length of the grid past the lowest and highest values
            of X so that the kernel goes to zero. The end points are
            -/+ cut*bw*{min(X) or max(X)}
        adjust : float
            An adjustment factor for the bw. Bandwidth becomes bw * adjust.
        """
        try:
            bw = float(bw)
            self.bw_method = "user-given"
        except:
            self.bw_method = bw
        endog = self.endog

        if fft:
            if kernel != "gau":
                from warnings import warn
                msg = "Only Gaussian kernel is available for FFT. Using "
                mgs += "kernel ='gau'"
                warn(msg)
            density, grid, bw = kdensityfft(endog, kernel=kernel, bw=bw,
                    adjust=adjust, weights=weights, gridsize=gridsize,
                    clip=clip, cut=cut)
        else:
            density, grid, bw = kdensity(endog, kernel=kernel, bw=bw,
                    adjust=adjust, weights=weights, gridsize=gridsize,
                    clip=clip, cut=cut)
        self.density = density
        self.support = grid
        self.bw = bw


#### Kernel Density Estimator Functions ####

def kdensity(X, kernel="gauss", bw="scott", weights=None, gridsize=None,
                adjust=1, clip=(-np.inf,np.inf), cut=3, retgrid=True):
    """
    Rosenblatz-Parzen univariate kernel desnity estimator

    Parameters
    ----------
    X : array-like
        The variable for which the density estimate is desired.
    kernel : str
        The Kernel to be used. Choices are
        - "biw" for biweight
        - "cos" for cosine
        - "epa" for Epanechnikov
        - "gauss" for Gaussian.
        - "tri" for triangular
        - "triw" for triweight
        - "uni" for uniform
    bw : str, float
        "scott" - 1.059 * A * nobs ** (-1/5.), where A is min(std(X),IQR/1.34)
        "silverman" - .9 * A * nobs ** (-1/5.), where A is min(std(X),IQR/1.34)
        If a float is given, it is the bandwidth.
    gridsize : int
        If gridsize is None, max(len(X), 50) is used.
    cut : float
        Defines the length of the grid past the lowest and highest values of X
        so that the kernel goes to zero. The end points are
        -/+ cut*bw*{min(X) or max(X)}


    Notes
    -----
    Creates an intermediate (`gridsize` x `gridsize`) array. Use FFT for a more
    computationally efficient version.
    Weights aren't implemented yet.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:,None]
    X = X[np.logical_and(X>clip[0], X<clip[1])]

    nobs = float(len(X)) # after trim

    # if bw is None, select optimal bandwidth for kernel
    try:
        bw = float(bw)
    except:
        bw = select_bandwidth(X, bw, kernel)
    bw *= adjust

    if gridsize == None:
        gridsize = max(nobs,50) # don't need to resize if no FFT

    a = np.min(X,axis=0) - cut*bw
    b = np.max(X,axis=0) + cut*bw
    grid = np.linspace(a, b, gridsize)

    k = (X.T - grid[:,None])/bw  # uses broadcasting

    # instantiate kernel class
    kern = kernel_switch[kernel](h=bw)
    k = kern(k) # estimate density
    k[k<0] = 0 # get rid of any negative values

# res = np.repeat(x,n).reshape(m,n).T - np.repeat(xi,m).reshape(n,m))/h
#    if kernel.lower() == "epa":
#        k = np.zeros_like(grid) + np.less_equal(np.abs(k),
#                np.sqrt(5)) * 3/(4*np.sqrt(5)) * (1-.2*k**2)
##        k = (.15/np.sqrt(5))*(5-k**2)/h
##        k[k<0] = 0
#    if kernel.lower() == "gauss":
#        k = 1/np.sqrt(2*np.pi)*np.exp(-.5*k**2)
##        k = np.clip(k,1e12,0)
##        kern = kernels.Gaussian(h=bw)
##        k = kern(k)

    if weights == None: #TODO: observation weights should go before estimation
        weights = 1

#    dens = np.mean(1/bw*weights*k,1)
    dens = np.mean(k,1)/bw
    if retgrid:
        return dens, grid, bw
    else:
        return dens, bw

def kdensityfft(X, kernel="gauss", bw="scott", weights=None, gridsize=None,
        adjust=1, clip=(-np.inf,np.inf), cut=3, retgrid=True):
    """
    Rosenblatz-Parzen univariate kernel desnity estimator

    Parameters
    ----------
    X : array-like
        The variable for which the density estimate is desired.
    kernel : str
        "bi" for biweight
        "cos" for cosine
        "epa" for Epanechnikov, default
        "epa2" for alternative Epanechnikov
        "gauss" for Gaussian.
        "par" for Parzen
        "rect" for rectangular
        "tri" for triangular
        ONLY GAUSSIAN IS CURRENTLY IMPLEMENTED.
    bw : str, float
        "scott" - 1.059 * A * nobs ** (-1/5.), where A is min(std(X),IQR/1.34)
        "silverman" - .9 * A * nobs ** (-1/5.), where A is min(std(X),IQR/1.34)
        If a float is given, it is the bandwidth.
    adjust : float
        An adjustment factor for the bw. Bandwidth becomes bw * adjust.
    gridsize : int
        If gridsize is None, min(len(X), 512) is used.  Note that this number
        is rounded up to the next highest power of 2.
    cut : float
        Defines the length of the grid past the lowest and highest values of X
        so that the kernel goes to zero. The end points are
        -/+ cut*bw*{X.min() or X.max()}

    Notes
    -----
    Only the default kernel is implemented. Weights aren't implemented yet.
    This follows Silverman (1982) with changes suggested by Jones and Lotwick
    (1984). However, the discretization step is replaced by linear binning
    of Fan and Marron (1994).

    References
    ---------- ::

    Fan, J. and J.S. Marron. (1994) `Fast implementations of nonparametric
        curve estimators`. Journal of Computational and Graphical Statistics.
        3.1, 35-56.
    Jones, M.C. and H.W. Lotwick. (1984) `Remark AS R50: A Remark on Algorithm
        AS 176. Kernal Density Estimation Using the Fast Fourier Transform`.
        Journal of the Royal Statistical Society. Series C. 33.1, 120-2.
    Silverman, B.W. (1982) `Algorithm AS 176. Kernel density estimation using
        the Fast Fourier Transform. Journal of the Royal Statistical Society.
        Series C. 31.2, 93-9.
    """
    X = np.asarray(X)
    X = X[np.logical_and(X>clip[0], X<clip[1])] # won't work for two columns.
                                                # will affect underlying data?
    try:
        bw = float(bw)
    except:
        bw = select_bandwidth(X, bw, kernel) # will cross-val fit this pattern?
    bw *= adjust

    nobs = float(len(X)) # after trim

    # 1 Make grid and discretize the data
    if gridsize == None:
        gridsize = np.max((nobs,512.))
    gridsize = 2**np.ceil(np.log2(gridsize)) # round to next power of 2

    a = np.min(X)-cut*bw
    b = np.max(X)+cut*bw
    grid,delta = np.linspace(a,b,gridsize,retstep=True)
    RANGE = b-a

# This is the Silverman binning function, but I believe it's buggy (SS)

# weighting according to Silverman
#    count = counts(X,grid)
#    binned = np.zeros_like(grid)    #xi_{k} in Silverman
#    j = 0
#    for k in range(int(gridsize-1)):
#        if count[k]>0: # there are points of X in the grid here
#            Xingrid = X[j:j+count[k]] # get all these points
#            # get weights at grid[k],grid[k+1]
#            binned[k] += np.sum(grid[k+1]-Xingrid)
#            binned[k+1] += np.sum(Xingrid-grid[k])
#            j += count[k]
#    binned /= (nobs)*delta**2 # normalize binned to sum to 1/delta

#NOTE: THE ABOVE IS WRONG, JUST TRY WITH LINEAR BINNING
    binned = linbin(X,a,b,gridsize)/(delta*nobs)

    # step 2 compute FFT of the weights, using Munro (1976) FFT convention
    y = forrt(binned)

    # step 3 and 4 for optimal bw compute zstar and the density estimate f
    # don't have to redo the above if just changing bw, ie., for cross val

#NOTE: silverman_transform is the closed form solution of the FFT of the
#gaussian kernel. Not yet sure how to generalize it.
    zstar = silverman_transform(bw, nobs, RANGE)*y # 3.49 in Silverman
                                                   # 3.50 w Gaussian kernel
    f = revrt(zstar)
    if retgrid:
        return f, grid, bw
    else:
        return f, bw

if __name__ == "__main__":
    import numpy as np
    np.random.seed(12345)
    xi = np.random.randn(100)
    f,grid, bw1 = kdensity(xi, kernel="gauss", bw=.372735, retgrid=True)
    f2, bw2 = kdensityfft(xi, kernel="gauss", bw="silverman",retgrid=False)

# do some checking vs. silverman algo.
# you need denes.f, http://lib.stat.cmu.edu/apstat/176
#NOTE: I (SS) made some changes to the Fortran
# and the FFT stuff from Munro http://lib.stat.cmu.edu/apstat/97o
# then compile everything and link to denest with f2py
#Make pyf file as usual, then compile shared object
#f2py denest.f -m denest2 -h denest.pyf
#edit pyf
#-c flag makes it available to other programs, fPIC builds a shared library
#/usr/bin/gfortran -Wall -c -fPIC fft.f
#f2py -c denest.pyf ./fft.o denest.f

    try:
        from denest2 import denest
        a = -3.4884382032045504
        b = 4.3671504686785605
        RANGE = b - a
        bw = bandwidths.bw_silverman(xi)

        ft,smooth,ifault,weights,smooth1 = denest(xi,a,b,bw,np.zeros(512),np.zeros(512),0,
                np.zeros(512), np.zeros(512))
# We use a different binning algo, so only accurate up to 3 decimal places
        np.testing.assert_almost_equal(f2, smooth, 3)
#NOTE: for debugging
#        y2 = forrt(weights)
#        RJ = np.arange(512/2+1)
#        FAC1 = 2*(np.pi*bw/RANGE)**2
#        RJFAC = RJ**2*FAC1
#        BC = 1 - RJFAC/(6*(bw/((b-a)/M))**2)
#        FAC = np.exp(-RJFAC)/BC
#        SMOOTH = np.r_[FAC,FAC[1:-1]] * y2

#        dens = revrt(SMOOTH)

    except:
#        ft = np.loadtxt('./ft_silver.csv')
#        smooth = np.loadtxt('./smooth_silver.csv')
        print "Didn't get the estimates from the Silverman algorithm"
