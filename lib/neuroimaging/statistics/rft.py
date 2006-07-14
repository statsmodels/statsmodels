import numpy as N
from numpy.linalg import pinv

from neuroimaging import traits
import scipy.stats
from scipy.special import gamma, gammaln, beta, hermitenorm
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

def Q(dim, dfd=N.inf):
    """

    If dfd == inf (the default), then
    Q(dim) is the (dim-1)-st Hermite polynomial

    H_j(x) = (-1)^j * e^{x^2/2} * (d^j/dx^j e^{-x^2/2})

    If dfd != inf, then it is the polynomial Q defined in

    Worsley, K.J. (1994). 'Local maxima and the expected Euler
    characteristic of excursion sets of \chi^2, F and t fields.'
    Advances in Applied Probability, 26:13-42.

    """

    m = dfd
    j = dim

    if j > 0:
        poly = hermitenorm(j-1)
        poly = N.poly1d(N.around(poly.c))
        if N.isfinite(m):
            for l in range((j-1)/2+1):
                f = N.exp(gammaln((m+1)/2.) - gammaln((m+2-j+2*l)/2.)
                                   - 0.5*(j-1-2*l)*(N.log(m/2.)))
                poly.c[2*l] *= f
        return poly
    else:
        raise ValueError, 'Q only defined for j > 0'

def K(dim=4, dfn=7, dfd=N.inf):
    """
    Determine the polynomial K in

    Worsley, K.J. (1994). 'Local maxima and the expected Euler
    characteristic of excursion sets of \chi^2, F and t fields.'
    Advances in Applied Probability, 26:13-42.

    with an additional factor in front

    \frac{\Gamma(dfn+dfd-dim)/2}{\Gamma(dfd/2)} * (m/2)^(n-N/2)

    If dfd=inf, return the limiting polynomial.

    """

    def lbinom(n, j):
        return gammaln(n+1) - gammaln(j+1) - gammaln(n-j+1)

    m = dfd
    n = dfn
    D = dim

    rhoF=N.zeros((D,D), N.float64)
    rhoCT=N.zeros((D,D), N.float64)
    k = N.arange(D)

    coef = 0

    for j in range(int(N.floor((D-1)/2.)+1)):
        if N.isfinite(m):
            t = (gammaln((m+n-D)/2.+j) - # first factor
                 gammaln(j+1) -
                 gammaln((m+n-D)/2.))
            t += lbinom(m-1, k-j) - k * N.log(m)
        else:
            _t = N.power(2., -j) / factorial(k-j)
            t = N.log(_t)
            t[N.isinf(_t)] = -N.inf
        t += lbinom(n-1, D-1-j-k)
        coef += (-1)**(D-1) * factorial(D-1) * N.exp(t) * N.power(-1.*n, k)

    return N.poly1d(coef[::-1])

def rho(x, dim, df=N.inf):
    """
    EC densities for T and Gaussian (df=inf) random fields.
    """

    m = df

    if dim > 0:
        x = N.asarray(x, N.float64)
        q = Q(dim-1, dfd=df)(x)

        if N.isfinite(m):
            q *= N.power(1 + x**2/m, -(m-1)/2.)
        else:
            q *= N.exp(-x**2/2)

        return q * N.power(2*N.pi, -(dim+1)/2.)
    else:
        if N.isfinite(m):
            return scipy.stats.t.sf(x, df)
        else:
            return scipy.stats.norm.sf(x)

def F(x, dim, dfd=N.inf, dfn=1):
    """
    EC densities for F and Chi^2 (dfd=inf) random fields.
    """

    m = dfd
    n = dfn
    D = dim

    if dim > 0:
        x = N.asarray(x, N.float64)
        k = K(dim=dim, dfd=dfd, dfn=dfn)(x)

        if N.isfinite(m):
            f = x*n/m
            t = -N.log(1 + f) * (m-n-2) / 2.
            t += N.log(f) * (n-D) / 2.
            t += gammaln((m+n-D)/2.) - gammaln(m/2.) - gammaln(n/2.)
        else:
            f = x*n
            t = N.log(f/2) * (n-D) / 2. - f/2.
        t += N.log(2*N.pi) * D / 2. + N.log(2) * (D-2)/2.
        k *= N.exp(t)

        return k
    else:
        if N.isfinite(m):
            return scipy.stats.f.sf(x, dfn, dfd)
        else:
            return scipy.stats.chi.sf(x, dfn)

class IntrinsicVolumes(traits.HasTraits):
    """
    A simple class that exists only to compute the intrinsic volumes of
    products of sets (that themselves have intrinsic volumes, of course).
    """

    order = traits.Int(0)

    def __init__(self, mu=[1]):
        self.mu = N.asarray(mu, N.float64)
        if isinstance(mu, IntrinsicVolumes):
            self.mu = mu.mu
        self.order = self.mu.ndim

    def __str__(self):
        return str(self.mu)

    def __mul__(self, other):

        if not isinstance(other, IntrinsicVolumes):
            raise ValueError, 'expecting an IntrinsicVolumes instance'
        order = self.order + other.order - 1
        mu = N.zeros(order, N.float64)

        for i in range(order):
            for j in range(i+1):
                try:
                    mu[i] += self.mu[j] * other.mu[i-j]
                except:
                    pass
        return IntrinsicVolumes(mu)

class ECcone(IntrinsicVolumes):
    """
    A class that takes the intrinsic volumes of a set and gives the
    EC approximation to the supremum distribution of a unit variance Gaussian
    process with these intrinsic volumes. This is the basic building block of
    all of the EC densities.
    """

    def __init__(self, mu=[1], m=N.inf, search=None):
        self.m = m
        IntrinsicVolumes.__init__(self, mu=mu)
        if search:
            self.search = IntrinsicVolumes(search)
        self.order = self.mu.shape[0]

    def __call__(self, x, j=0, search=None):

        if search is None:
            search = self.search
        if search is None:
            search = IntrinsicVolumes([1.])
        x = N.asarray(x, N.float64)
        f = N.zeros(x.shape, N.float64)
        for j in range(IntrinsicVolumes(search).mu.shape[0]):
            _rho = 0.
            for i in range(self.order):
                _x = self.mu[i] * rho(x, i+j, m=self.m) * N.power(1 + x**2 / self.m, j/2.)

                _rho += self.mu[i] * rho(x, i+j, m=self.m) * N.power(1 + x**2 / self.m, j/2.)
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

def mu_sphere(n, j, r=1):
    """
    Return mu_j(S_r(R^n)), the j-th Lipschitz Killing
    curvature of the sphere of radius r in R^n.
    """

    if j < n:
        if n-1 == j:
            return 2 * N.power(N.pi, n/2.) * N.power(r, n-1) / gamma(n/2.)

        if (n-1-j)%2 == 0:

            return 2 * binomial(n-1, j) * mu_sphere(n,n-1) * N.power(r, j) / mu_sphere(n-j,n-j-1)
        else:
            return 0
    else:
        return 0

def mu_ball(n, j, r=1):
    """
    Return mu_j(B_n(r)), the j-th Lipschitz Killing
    curvature of the ball of radius r in R^n.
    """

    if j <= n:
        if n == j:
            return N.power(N.pi, n/2.) * N.power(r, n) / gamma(n/2. + 1.)
        else:
            return binomial(n, j) * N.power(r, j) * mu_ball(n,n) / mu_ball(n-j,n-j)
    else:
        return 0

def volume2ball(vol, d=3):
    """
    Approximate intrinsic volumes of a set with a given volume by those of a ball with a given dimension and equal volume.
    """

    if d > 0:
        mu = N.zeros((d+1,), N.float64)
        r = N.power(vol * 1. / mu_ball(d, d), 1./d)

        for j in range(d+1):
            mu[j] = mu_ball(d, j, r=r)
    else:
        mu = [1]
    return IntrinsicVolumes(mu=mu)

class ChiSquared(ECcone):

    """
    EC densities for a Chi-Squared(n) random field.
    """

    def __init__(self, n=1, **extra):
        self.n = n
        mu = N.zeros(n, N.float64)
        for i in range(n):
            mu[i] = mu_sphere(n, i)
        ECcone.__init__(self, mu, **extra)

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
        self.sphere = IntrinsicVolumes([mu_sphere(k,i) for i in range(k)])
        self.k = k

    def __call__(self, x, search=None):

        if search is None:
            search = self.sphere
        else:
            search = IntrinsicVolumes(search) * self.sphere
        return FStat.__call__(self, x, search=search)

class Hotelling(FStat):
    """
    Hotelling's T^2: maximize an F_{1,m}=T_m^2 statistic over a sphere
    of dimension k.
    """

    def __init__(self, m=4, k=1, **extra):
        FStat.__init__(m=m, n=1, **extra)
        self.sphere = IntrinsicVolumes([mu_sphere(k,i) for i in range(k)])
        self.k = k

    def __call__(self, x, search=None):

        if search is None:
            search = self.sphere
        else:
            search = IntrinsicVolumes(search) * self.sphere
        return FStat.__call__(self, x, search=search)

class OneSidedF(FStat):

    def __call__(self, x, search=None):
        d1 = FStat.__call__(self, x, search=search)
        self.m -= 1
        d2 = FStat.__call__(self, x, search=search)
        self.m += 1
        return (d1 - d2) / 2.

class ChiBarSquared(ChiSquared):

    def _getmu(self):
        x = N.linspace(0, 2 * self.n, 100)
        sf = 0.
        g = Gaussian()
        for i in range(1, self.n+1):
            sf += binomial(self.n, i) * scipy.stats.chi.sf(x, i) / N.power(2., self.n)

        d = N.transpose(N.array([g.density(N.sqrt(x), j) for j in range(self.n)]))
        c = N.dot(pinv(d), sf)
        sf += 1. / N.power(2, self.n)
        self.mu = IntrinsicVolumes(c)
        print self.mu, sf[0], 'after'

    def __init__(self, n=1, **extra):
        ChiSquared.__init__(self, n=n, **extra)
        print self.mu, 'now'
        self._getmu()

    def __call__(self, x, search=None):

        if search is None:
            search = self.stat
        else:
            search = IntrinsicVolumes(search) * self.stat
        return FStat.__call__(self, x, search=search)

