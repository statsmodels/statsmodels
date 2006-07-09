import numpy as N
from numpy.linalg import pinv

from neuroimaging import traits
import scipy.stats
from scipy.special import gamma, gammaln, beta
from scipy import factorial

def binomial(n, j):
    """
    Binomial coefficient:

           n!
       ---------
       (n-j)! j!
    """

    if n <= j or n == 0:
        return 0.
    elif j == 0:
        return 1.
    return 1./(beta(n-j+1,j+1)*(n+1))

def rho(x, order, m=N.inf):
    """
    EC densities for T and Gaussian (m=inf) random fields. They are in the CJS paper....
    """
    x = N.asarray(x, N.float64)
    j = order - 1

    f = x * 0
    if j >= 0:
        for l in range(int(N.floor(j/2.))+1):
            tmp = N.power(x, j-2*l) / (factorial(j-2*l) * factorial(l) * N.power(-2.0, l))

            if N.isfinite(m):
                tmp *= N.power(1 + x**2/m, -(m-1-2*l)/2.) * N.power(2./m, (j-2*l)/2.) * N.exp(gammaln((m+j-2*l)/2.) - gammaln(m/2.))
            f += tmp
        f = f * factorial(j)

        if not N.isfinite(m):
            f *= N.exp(-x**2/2)
        return f / N.sqrt(2*N.pi)
    else:
        if not N.isfinite(m):
            return scipy.stats.norm.sf(x)
        else:
            return scipy.stats.t.sf(x, m)

class LKSet(traits.HasTraits):
    """
    A simple class that exists only to compute the intrinsic volumes of
    products of sets (that themselves have intrinsic volumes, of course).
    """

    order = traits.Int(0)

    def __init__(self, LK=[1]):
        self.LK = N.asarray(LK, N.float64)
        if isinstance(LK, LKSet):
            self.LK = LK.LK
        self.order = self.LK.ndim

    def __str__(self):
        return str(self.LK)

    def __mul__(self, other):

        if not isinstance(other, LKSet):
            raise ValueError, 'expecting an LKSet instance'
        order = self.order + other.order - 1
        LK = N.zeros(order, N.float64)

        for i in range(order):
            for j in range(i+1):
                try:
                    LK[i] += self.LK[j] * other.LK[i-j]
                except:
                    pass
        return LKSet(LK)


class ECcone(LKSet):
    """
    A class that takes the intrinsic volumes of a set and gives the
    EC approximation to the supremum distribution of a unit variance Gaussian
    process with these intrinsic volumes. This is the basic building block of
    all of the EC densities.
    """

    def __init__(self, LK=[1], m=N.inf, search=None):
        self.m = m
        LKSet.__init__(self, LK=LK)
        if search:
            self.search = LKSet(search)
        self.order = self.LK.shape[0]

    def __call__(self, x, j=0, search=None):

        if search is None:
            search = self.search
        if search is None:
            search = LKSet([1.])
        x = N.asarray(x, N.float64)
        f = N.zeros(x.shape, N.float64)
        for j in range(LKSet(search).LK.shape[0]):
            _rho = 0.
            for i in range(self.order):
                _x = self.LK[i] * rho(x, i+j, m=self.m) * N.power(1 + x**2 / self.m, j/2.)

                _rho += self.LK[i] * rho(x, i+j, m=self.m) * N.power(1 + x**2 / self.m, j/2.)
            _rho /= N.power(2*N.pi, j/2.)
            f += search[j] * _rho
        return f

    def pvalue(self, x, search=None):
        return self(x, search=search)

    def density(self, x, j):
        """
        The j-th EC density.
        """
        search = N.zeros((j+1), N.float64)
        search[-1] = 1.
        return self(x, search=search)

    def polynomial(self, x, j):
        """
        Polynomial part of the j-th EC density.
        """
        x = N.asarray(x, N.float64)
        search = N.zeros((j+1), N.float64)
        search[-1] = 1.
        if N.isfinite(self.m):
            f = N.power(1. + x**2 / self.m, (self.m - 1)/ 2.)
        else:
            f = N.exp(x**2/2)
        return self(x, search=search) * f

Gaussian = ECcone

def LKsphere(n, j, r=1):
    """
    Return LK_j(S_r(R^n)), the j-th Lipschitz Killing
    curvature of the sphere of radius r in R^n.
    """

    if j < n:
        if n-1 == j:
            return 2 * N.power(N.pi, n/2.) * N.power(r, n-1) / gamma(n/2.)

        if (n-1-j)%2 == 0:

            return 2 * binomial(n-1, j) * LKsphere(n,n-1) * N.power(r, j) / LKsphere(n-j,n-j-1)
        else:
            return 0
    else:
        return 0

def LKball(n, j, r=1):
    """
    Return LK_j(B_n(r)), the j-th Lipschitz Killing
    curvature of the ball of radius r in R^n.
    """

    if j <= n:
        if n == j:
            return N.power(N.pi, n/2.) * N.power(r, n) / gamma(n/2. + 1.)
        else:
            return binomial(n, j) * N.power(r, j) * LKball(n,n) / LKball(n-j,n-j)
    else:
        return 0

def volume2ball(vol, d=3):
    """
    Approximate intrinsic volumes of a set with a given volume by those of a ball with a given dimension and equal volume.
    """

    if d > 0:
        LK = N.zeros((d+1,), N.float64)
        r = N.power(vol * 1. / LKball(d, d), 1./d)

        for j in range(d+1):
            LK[j] = LKball(d, j, r=r)
    else:
        LK = [1]
    return LKSet(LK=LK)

class ChiSquared(ECcone):

    """
    EC densities for a Chi-Squared(n) random field.
    """

    def __init__(self, n=1, **extra):
        self.n = n
        LK = N.zeros(n, N.float64)
        for i in range(n):
            LK[i] = LKsphere(n, i)
        ECcone.__init__(self, LK, **extra)

    def __call__(self, x, **extra):
        return ECcone.__call__(self, N.sqrt(x), **extra)

class TStat(ECcone):
    """
    EC densities for a t random field.
    """

    def __init__(self, m=4, **extra):
        ECcone.__init__(self, [1], m=m, **extra)

class FStat(ChiSquared):

    """
    EC densities for a F random field.
    """

    def __init__(self, n=1, m=4, **extra):
        self.n = n
        ChiSquared.__init__(self, n, m=m, **extra)

    def __call__(self, x, **extra):
        return ECcone.__call__(self, N.sqrt(x * self.n), **extra)

class Roy(FStat):
    """
    Roy's maximum root: maximize an F_{n,m} statistic over a sphere
    of dimension k.
    """

    def __init__(self, n=1, m=4, k=1, **extra):
        FStat.__init__(m=m, n=n, **extra)
        self.sphere = LKSet([LKsphere(k,i) for i in range(self.k)])
        self.k = k

    def __call__(self, x, search=None):

        if search is None:
            search = self.sphere
        else:
            search = LKSet(search) * self.sphere
        return FStat.__call__(self, x, search=search)

class Hotelling(FStat):
    """
    Hotelling's T^2: maximize an F_{1,m}=T_m^2 statistic over a sphere
    of dimension k.
    """

    def __init__(self, m=4, k=1, **extra):
        FStat.__init__(m=m, n=1, **extra)
        self.sphere = LKSet([LKsphere(k,i) for i in range(self.k)])
        self.k = k

    def __call__(self, x, search=None):

        if search is None:
            search = self.sphere
        else:
            search = LKSet(search) * self.sphere
        return FStat.__call__(self, x, search=search)

class OneSidedF(FStat):

    def __call__(self, x, search=None):
        d1 = FStat.__call__(self, x, search=search)
        self.m -= 1
        d2 = FStat.__call__(self, x, search=search)
        self.m += 1
        return (d1 - d2) / 2.

class ChiBarSquared(ChiSquared):

    def _getLK(self):
        x = N.linspace(0, 2 * self.n, 100)
        sf = 0.
        g = Gaussian()
        for i in range(1, self.n+1):
            sf += binomial(self.n, i) * scipy.stats.chi.sf(x, i) / N.power(2., self.n)

        d = N.transpose(N.array([g.density(N.sqrt(x), j) for j in range(self.n)]))
        c = N.dot(pinv(d), sf)
        sf += 1. / N.power(2, self.n)
        self.LK = LKSet(c)
        print self.LK, sf[0], 'after'

    def __init__(self, n=1, **extra):
        ChiSquared.__init__(self, n=n, **extra)
        print self.LK, 'now'
        self._getLK()

    def __call__(self, x, search=None):

        if search is None:
            search = self.stat
        else:
            search = LKSet(search) * self.stat
        return FStat.__call__(self, x, search=search)

