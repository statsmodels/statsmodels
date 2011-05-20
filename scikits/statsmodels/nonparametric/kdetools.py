#### Convenience Functions to be moved to kerneltools ####
import numpy as np

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
