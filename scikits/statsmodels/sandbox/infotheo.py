"""
Information Theoretic and Entropy Measures

References
----------
Golan, As. 2008. "Information and Entropy Econometrics -- A Review and
    Synthesis." Foundations And Trends in Econometrics 2(1-2), 1-145.

Golan, A., Judge, G., and Miller, D.  1996.  Maximum Entropy Econometrics.
    Wiley & Sons, Chichester.
"""
#For MillerMadow correction
#Miller, G. 1955. Note on the bias of information estimates. Info. Theory
#    Psychol. Prob. Methods II-B:95-100.

#For ChaoShen method
#Chao, A., and T.-J. Shen. 2003. Nonparametric estimation of Shannon's index of diversity when
#there are unseen species in sample. Environ. Ecol. Stat. 10:429-443.
#Good, I. J. 1953. The population frequencies of species and the estimation of population parameters.
#Biometrika 40:237-264.
#Horvitz, D.G., and D. J. Thompson. 1952. A generalization of sampling without replacement from a finute universe. J. Am. Stat. Assoc. 47:663-685.

#For NSB method
#Nemenman, I., F. Shafee, and W. Bialek. 2002. Entropy and inference, revisited. In: Dietterich, T.,
#S. Becker, Z. Gharamani, eds. Advances in Neural Information Processing Systems 14: 471-478.
#Cambridge (Massachusetts): MIT Press.

#For shrinkage method
#Dougherty, J., Kohavi, R., and Sahami, M. (1995). Supervised and unsupervised discretization of
#continuous features. In International Conference on Machine Learning.
#Yang, Y. and Webb, G. I. (2003). Discretization for naive-bayes learning: managing discretization
#bias and variance. Technical Report 2003/131 School of Computer Science and Software Engineer-
#ing, Monash University.

from scipy import maxentropy, stats
import numpy as np
from matplotlib import pyplot as plt

#TODO: change these to use maxentutils so that over/underflow is handled
#with the logsumexp.

from scipy.maxentropy import logsumexp

#The below was taken from Warren's post
#http://mail.scipy.org/pipermail/scipy-user/2009-October/022931.html
def my_logsumexp(a, axis=None):
    """
    """
    if axis is None:
        # Use the scipy.maxentropy version.
        return logsumexp(a)
    a = asarray(a)
    shp = list(a.shape)
    shp[axis] = 1
    a_max = a.max(axis=axis)
    s = log(exp(a - a_max.reshape(shp)).sum(axis=axis))
    lse  = a_max + s
    return lse


def _isproperdist(X):
    """
    Checks to see if `X` is a proper probability distribution
    """
    X = np.asarray(X)
    if not np.allclose(np.sum(X), 1) or not np.all(X>=0) or not np.all(X<=1):
        return False
    else:
        return True

def discretize(X, method="ef", nbins=None):
    """
    Discretize `X`

    Parameters
    ----------
    bins : int, optional
        Number of bins.  Default is floor(sqrt(N))
    method : string
        "ef" is equal-frequency binning
        "ew" is equal-width binning

    Examples
    --------
    """
    nobs = len(X)
    if nbins == None:
        nbins = np.floor(np.sqrt(nobs))
    if method == "ef":
        discrete = np.ceil(nbins * stats.rankdata(X)/nobs)
    if method == "ew":
        width = np.max(X) - np.min(X)
        width = np.floor(width/nbins)
        svec, ivec = stats.fastsort(X)
        discrete = np.zeros(nobs)
        binnum = 1
        base = svec[0]
        discrete[ivec[0]] = binnum
        for i in xrange(1,nobs):
            if svec[i] < base + width:
                discrete[ivec[i]] = binnum
            else:
                base = svec[i]
                binnum += 1
                discrete[ivec[i]] = binnum
    return discrete
#TODO: looks okay but needs more robust tests for corner cases



def logbasechange(a,b):
    """
    There is a one-to-one transformation of the entropy value from
    a log base b to a log base a :

    H_{b}(X)=log_{b}(a)[H_{a}(X)]

    Returns
    -------
    log_{b}(a)
    """
    return np.log(b)/np.log(a)

def natstobits(X):
    """
    Converts from nats to bits
    """
    return logbasechange(np.e, 2) * X

def bitstonats(X):
    """
    Converts from bits to nats
    """
    return logbasechange(2, np.e) * X

# Shannon's entropy
def shannonentropy(px, logbase=2):
    """
    This is Shannon's entropy

    Parameters
    -----------
    logbase, int or np.e
        The base of the log
    px : 1d or 2d array_like
        Can be a discrete probability distribution, a 2d joint distribution,
        or a sequence of probabilities.

    Returns
    -----
    For log base 2 (bits) given a discrete distribution
        H(p) = sum(px * log2(1/px) = -sum(pk*log2(px)) = E[log2(1/p(X))]

    For log base 2 (bits) given a joint distribution
        H(px,py) = -sum_{k,j}*w_{kj}log2(w_{kj})

    Notes
    -----
    shannonentropy(0) is defined as 0
    """
    px = np.asarray(px)
    if not np.all(px <= 1) or not np.all(px >= 0):
        raise ValueError, "px does not define proper distribution"


    entropy = -np.sum(np.nan_to_num(px*np.log2(px)))
    if logbase != 2:
        return logbasechange(2,logbase) * entropy
    else:
        return entropy

# Shannon's information content
def shannoninfo(px, logbase=2):
    """
    Shannon's information

    Parameters
    ----------
    px : float or array-like
        `px` is a discrete probability distribution

    Returns
    -------
    For logbase = 2
    np.log2(px)
    """
    px = np.asarray(px)
    if not np.all(px <= 1) or not np.all(px >= 0):
        raise ValueError, "px does not define proper distribution"
    if logbase != 2:
        return - logbasechange(2,logbase) * np.log2(px)
    else:
        return - np.log2(px)

def condentropy(px, py, pxpy=None, logbase=2):
    """
    Return the conditional entropy of X given Y.

    Parameters
    ----------
    px : array-like
    py : array-like
    pxpy : array-like, optional
        If pxpy is None, the distributions are assumed to be independent
        and conendtropy(px,py) = shannonentropy(px)
    logbase : int or np.e

    Returns
    -------
    sum_{kj}log(q_{j}/w_{kj}

    where q_{j} = Y[j]
    and w_kj = X[k,j]
    """
    if not _isproperdist(px) or not _isproperdist(py):
        raise ValueError, "px or py is not a proper probability distribution"
    if pxpy != None and not _isproperdist(pxpy):
        raise ValueError, "pxpy is not a proper joint distribtion"
    if pxpy == None:
        pxpy = np.outer(py,px)

    condent = np.sum(pxpy * np.nan_to_num(np.log2(py/pxpy)))
    if logbase == 2:
        return condent
    else:
        return logbasechange(2, logbase) * condent

def mutualinfo(px,py,pxpy, logbase=2):
    """
    Returns the mutual information between X and Y.

    Parameters
    ----------
    px : array-like
        Discrete probability distribution of random variable X
    py : array-like
        Discrete probability distribution of random variable Y
    pxpy : 2d array-like
        The joint probability distribution of random variables X and Y.
        Note that if X and Y are independent then the mutual information
        is zero.
    logbase : int or np.e, optional
        Default is 2 (bits)

    Returns
    -------
    shannonentropy(px) - condentropy(px,py,pxpy)
    """
    if not _isproperdist(px) or not _isproperdist(py):
        raise ValueError, "px or py is not a proper probability distribution"
    if pxpy != None and not _isproperdist(pxpy):
        raise ValueError, "pxpy is not a proper joint distribtion"
    if pxpy == None:
        pxpy = np.outer(py,px)
    return shannonentropy(px, logbase=logbase) - condentropy(px,py,pxpy,
            logbase=logbase)

def corrent(px,py,pxpy,logbase=2):
    """
    An information theoretic correlation measure.

    Reflects linear and nonlinear correlation between two random variables
    X and Y, characterized by the discrete probability distributions px and py
    respectively.

    Parameters
    ----------
    px : array-like
        Discrete probability distribution of random variable X
    py : array-like
        Discrete probability distribution of random variable Y
    pxpy : 2d array-like, optional
        Joint probability distribution of X and Y.  If pxpy is None, X and Y
        are assumed to be independent.
    logbase : int or np.e, optional
        Default is 2 (bits)

    Returns
    -------
    mutualinfo(px,py,pxpy,logbase=logbase)/shannonentropy(py,logbase=logbase)

    Notes
    -----
    This is also equivalent to

    corrent(px,py,pxpy) = 1 - condent(px,py,pxpy)/shannonentropy(py)
    """
    return mutualinfo(px,py,pxpy,logbase=logbase)/shannonentropy(py,
            logbase=logbase)

def covent(px,py,pxpy,logbase=2):
    """
    An information theoretic covariance measure.

    Reflects linear and nonlinear correlation between two random variables
    X and Y, characterized by the discrete probability distributions px and py
    respectively.

    Parameters
    ----------
    px : array-like
        Discrete probability distribution of random variable X
    py : array-like
        Discrete probability distribution of random variable Y
    pxpy : 2d array-like, optional
        Joint probability distribution of X and Y.  If pxpy is None, X and Y
        are assumed to be independent.
    logbase : int or np.e, optional
        Default is 2 (bits)

    Returns
    -------
    mutualinfo(px,py,pxpy,logbase=logbase)/shannonentropy(py,logbase=logbase)

    Notes
    -----
    This is also equivalent to

    corrent(px,py,pxpy) = 1 - condent(px,py,pxpy)/shannonentropy(py)

    """




if __name__ == "__main__":
    print "From Golan (2008) \"Information and Entropy Econometrics -- A Review \
and Synthesis"
    print "Table 3.1"
    # Examples from Golan (2008)

    X = [.2,.2,.2,.2,.2]
    Y = [.322,.072,.511,.091,.004]

    for i in X:
        print shannoninfo(i)
    for i in Y:
        print shannoninfo(i)
    print shannonentropy(X)
    print shannonentropy(Y)

    p = [1e-5,1e-4,.001,.01,.1,.15,.2,.25,.3,.35,.4,.45,.5]

    plt.subplot(111)
    plt.ylabel("Information")
    plt.xlabel("Probability")
    x = np.linspace(0,1,100001)
    plt.plot(x, shannoninfo(x))
#    plt.show()

    plt.subplot(111)
    plt.ylabel("Entropy")
    plt.xlabel("Probability")
    x = np.linspace(0,1,101)
    plt.plot(x, map(shannonentropy, zip(x,1-x)))
#    plt.show()

    # define a joint probability distribution
    # from Golan (2008) table 3.3
    w = np.array([[0,0,1./3],[1/9.,1/9.,1/9.],[1/18.,1/9.,1/6.]])
    # table 3.4
    px = w.sum(0)
    py = w.sum(1)
    H_X = shannonentropy(px)
    H_Y = shannonentropy(py)
    H_XY = shannonentropy(w)
    H_XgivenY = condentropy(px,py,w)
    H_YgivenX = condentropy(py,px,w)
# note that cross-entropy is not a distance measure as the following shows
    D_YX = logbasechange(2,np.e)*stats.entropy(px, py)
    D_XY = logbasechange(2,np.e)*stats.entropy(py, px)
    I_XY = mutualinfo(px,py,w)
    print "Table 3.3"
    print H_X,H_Y, H_XY, H_XgivenY, H_YgivenX, D_YX, D_XY, I_XY

    print "discretize functions"
    X=np.array([21.2,44.5,31.0,19.5,40.6,38.7,11.1,15.8,31.9,25.8,20.2,14.2,24.0,21.0,
        11.3,18.0,16.3,22.2,7.8,27.8,16.3,35.1,14.9,17.1,28.2,16.4,16.5,46.0,9.5,18.8,
        32.1,26.1,16.1,7.3,21.4,20.0,29.3,14.9,8.3,22.5,12.8,26.9,25.5,22.9,11.2,20.7,
        26.2,9.3,10.8,15.6])





