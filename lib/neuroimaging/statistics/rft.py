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
        return N.poly1d(poly.c)
    else:
        raise ValueError, 'Q defined only for dim > 0'

def K(dim=4, dfn=7, dfd=N.inf):
    """
    Determine the polynomial K in

    Worsley, K.J. (1994). 'Local maxima and the expected Euler
    characteristic of excursion sets of \chi^2, F and t fields.'
    Advances in Applied Probability, 26:13-42.

    If dfd=inf, return the limiting polynomial.

    """

    def lbinom(n, j):
        return gammaln(n+1) - gammaln(j+1) - gammaln(n-j+1)

    m = dfd
    n = dfn
    D = dim

    k = N.arange(D)

    coef = 0

    for j in range(int(N.floor((D-1)/2.)+1)):
        if N.isfinite(m):
            t = (gammaln((m+n-D)/2.+j) -
                 gammaln(j+1) -
                 gammaln((m+n-D)/2.))
            t += lbinom(m-1, k-j) - k * N.log(m)
        else:
            _t = N.power(2., -j) / (factorial(k-j) * factorial(j))
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
        q = Q(dim, dfd=df)(x)

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

    m = float(dfd)
    n = float(dfn)
    D = float(dim)

    if dim > 0:
        x = N.asarray(x, N.float64)
        k = K(dim=dim, dfd=dfd, dfn=dfn)(x)

        if N.isfinite(m):
            f = x*n/m
            t = -N.log(1 + f) * (m+n-2.) / 2.
            t += N.log(f) * (n-D) / 2.
            t += gammaln((m+n-D)/2.) - gammaln(m/2.)
        else:
            f = x*n
            t = N.log(f/2.) * (n-D) / 2. - f/2.

        t -= N.log(2*N.pi) * D / 2. + N.log(2) * (D-2)/2. + gammaln(n/2.)
        k *= N.exp(t)

        return k
    else:
        if N.isfinite(m):
            return scipy.stats.f.sf(x, dfn, dfd)
        else:
            return scipy.stats.chi.sf(x, dfn)


class ECquasi(N.poly1d):

    """
    A subclass of poly1d consisting of polynomials with a premultiplier of the
    form

    (1 + x^2/m)^-exponent

    where m is a non-negative float (possibly infinity, in which the
    function is a polynomial) and exponent is a non-negative multiple of 1/2.

    These arise often in the EC densities.

    >>> import numpy
    >>> x = numpy.linspace(0,1,101)

    >>> a = ECquasi([3,4,5])
    >>> a
    ECquasi([3, 4, 5],m=inf, exponent=0.000000)
    >>> a(3) == 3**2+1
    True

    >>> b = ECquasi(a.coeffs, m=30, exponent=4)
    >>> numpy.allclose(b(x), a(x) * numpy.power(1+x**2/30, -4))
    True


    """


    def __init__(self, c_or_r, r=0, exponent=None, m=None):
        N.poly1d.__init__(self, c_or_r, r=r, variable='x')

        if exponent is None and not hasattr(self, 'exponent'): self.exponent = 0
        elif not hasattr(self, 'exponent'): self.exponent = exponent


        if m is None and not hasattr(self, 'm'): self.m = N.inf
        elif not hasattr(self, 'm'): self.m = m

        if not N.isfinite(self.m): self.exponent = 0.

    def denom_poly(self):
        """
        This is the base of the premultiplier: (1+x^2/m).

        >>> b = ECquasi([3,4,20], m=30, exponent=4)
        >>> d = b.denom_poly()
        >>> d
        poly1d([ 0.03333333,  0.        ,  1.        ])
        >>> numpy.allclose(d.c, [1./b.m,0,1])
        True

        """
        return N.poly1d([1./self.m, 0, 1])

    def change_exponent(self, _pow):
        """
        Multiply top and bottom by an integer multiple of the
        self.denom_poly.

        >>> b = ECquasi([3,4,20], m=30, exponent=4)
        >>> x = numpy.linspace(0,1,101)
        >>> c = b.change_exponent(3)
        >>> c
        ECquasi([  3.70370370e-05,   0.00000000e+00,   3.37037037e-03,
        0.00000000e+00,   1.03333333e-01,   0.00000000e+00,
        1.10000000e+00,   0.00000000e+00,   1.00000000e+00],m=30.000000, exponent=7.000000)
        >>> numpy.allclose(c(x), b(x))
        True
        """

        if N.isfinite(self.m):
            _denom_poly = self.denom_poly()
            if int(_pow) != _pow or _pow < 0:
                raise ValueError, 'expecting a non-negative integer'
            p = _denom_poly**int(_pow)
            exponent = self.exponent + _pow
            coeffs = N.polymul(self, p).coeffs
            return ECquasi(coeffs, exponent=exponent, m=self.m)
        else:
            return ECquasi(self.coeffs, exponent=self.exponent, m=self.m)

    def __setattr__(self, key, val):
        if key == 'exponent':
            if 2*float(val) % 1 == 0:
                self.__dict__[key] = float(val)
            else: raise ValueError, 'expecting multiple of a half, got %f' % val
        elif key == 'm':
            if float(val) > 0 or val == inf:
                self.__dict__[key] = val
            else: raise ValueError, 'expecting positive float or inf'
        else: poly1d.__setattr__(self, key, val)

    def compatible(self, other):
        """
        Check whether the degrees of freedom of two instances are equal
        so that they can be multiplied together.

        >>> b = ECquasi([3,4,20], m=30, exponent=4)
        >>> x = numpy.linspace(0,1,101)
        >>> c = b.change_exponent(3)
        >>> b.compatible(c)
        True
        >>> d = ECquasi([3,4,20])
        >>> b.compatible(d)
        False
        """

        if self.m != other.m:
            raise ValueError, 'quasi polynomials are not compatible, m disagrees'
        return True

    def __add__(self, other):
        """
        Add two compatible ECquasi instances together.

        >>> b = ECquasi([3,4,20], m=30, exponent=4)
        >>> c = ECquasi([1], m=30, exponent=4)
        >>> b+c
        ECquasi([ 3,  4, 21],m=30.000000, exponent=4.000000)

        >>> d = ECquasi([1], m=30, exponent=3)
        >>> b+d
        ECquasi([  3.03333333,   4.        ,  21.        ],m=30.000000, exponent=4.000000)
        """

        other = ECquasi(other)
        if self.compatible(other):
            if N.isfinite(self.m):
                M = max(self.exponent, other.exponent)
                q1 = self.change_exponent(M-self.exponent)
                q2 = other.change_exponent(M-other.exponent)
                p = N.poly1d.__add__(q1, q2)
                return ECquasi(p.coeffs,
                               exponent=M,
                               m=self.m)
            else:
                p = N.poly1d.__add__(self, other)
                return ECquasi(p.coeffs,
                               exponent=0,
                               m=self.m)

    def __mul__(self, other):
        """
        Multiply two compatible ECquasi instances together.

        >>> b=ECquasi([3,4,20], m=30, exponent=4)
        >>> c=ECquasi([1,2], m=30, exponent=4.5)
        >>> b*c
        ECquasi([ 3, 10, 28, 40],m=30.000000, exponent=8.500000)
        """

        if N.isscalar(other):
            return ECquasi(self.coeffs * other,
                           m=self.m,
                           exponent=self.exponent)
        elif self.compatible(other):
            other = ECquasi(other)
            p = N.poly1d.__mul__(self, other)
            return ECquasi(p.coeffs,
                           exponent=self.exponent+other.exponent,
                           m=self.m)

    def __call__(self, val):
        """
        Evaluate the ECquasi instance.

        >>> import numpy
        >>> x = numpy.linspace(0,1,101)

        >>> a = ECquasi([3,4,5])
        >>> a
        ECquasi([3, 4, 5],m=inf, exponent=0.000000)
        >>> a(3) == 3**2+1
        True

        >>> b = ECquasi(a.coeffs, m=30, exponent=4)
        >>> numpy.allclose(b(x), a(x) * numpy.power(1+x**2/30, -4))
        True
        """

        n = N.poly1d.__call__(self, val)
        _p = self.denom_poly()(val)
        return n / N.power(_p, self.exponent)

    def __div__(self, other):
        raise NotImplementedError

    def __eq__(self, other):
        return (N.poly1d.__eq__(self, other) and
                self.m == other.m and
                self.exponent == other.exponent)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __pow__(self, _pow):
        """
        Power of a ECquasi instance.

        >>> b = ECquasi([3,4,5],m=10, exponent=3)
        >>> b**2
        ECquasi([ 9, 24, 46, 40, 25],m=10.000000, exponent=6.000000)
        >>>
        """
        p = N.poly1d.__pow__(self, int(_pow))
        q = ECquasi(p, m=self.m, exponent=_pow*self.exponent)
        return q

    def __sub__(self, other):
        """
        Subtract other from self.

        >>> b = ECquasi([3,4,20], m=30, exponent=4)
        >>> c = ECquasi([1,2], m=30, exponent=4)
        >>> b-c
        ECquasi([ 3,  3, 18],m=30.000000, exponent=4.000000)

        """

    def __repr__(self):
        vals = repr(self.coeffs)
        vals = vals[6:-1]
        if N.isfinite(self.m):
            return "ECquasi(%s,m=%f, exponent=%f)" % (vals, self.m, self.exponent)
        else:
            return "ECquasi(%s,m=%s, exponent=%f)" % (vals, `self.m`, self.exponent)

    __str__ = __repr__
    __rsub__ = __sub__
    __rmul__ = __mul__
    __rdiv__ = __div__

    def deriv(self, m=1):
        """
        Evaluate derivative of ECquasi.

        >>> a = ECquasi([3,4,5])
        >>> a.deriv(m=2)
        ECquasi([6],m=inf, exponent=0.000000)

        >>> b = ECquasi([3,4,5],m=10, exponent=3)
        >>> b.deriv()
        ECquasi([-1.2, -2. ,  3. ,  4. ],m=10.000000, exponent=4.000000)
        """

        if m == 1:
            if N.isfinite(self.m):
                q1 = ECquasi(N.poly1d.deriv(self, m=1),
                             m=self.m,
                             exponent=self.exponent)
                q2 = ECquasi(N.poly1d.__mul__(self, self.denom_poly().deriv(m=1)),
                             m = self.m,
                             exponent=self.exponent+1)
                return q1 - self.exponent * q2
            else:
                return ECquasi(N.poly1d.deriv(self, m=1),
                               m=N.inf,
                               exponent=0)
        else:
            d = selfy.deriv(m=1)
            return d.deriv(m=m-1)

class fnsum:

    def __init__(self, *items):
        self.items = list(items)

    def __call__(self, x):
        v = 0
        for q in self.items:
            v += q(x)
        return v

class IntrinsicVolumes(traits.HasTraits):
    """
    A simple class that exists only to compute the intrinsic volumes of
    products of sets (that themselves have intrinsic volumes, of course).
    """

    order = traits.Int(0, desc="Dimension of cone's parameter set, i.e. order of largest non-zero intrinsic volume.")

    def __init__(self, mu=[1]):

        if isinstance(mu, IntrinsicVolumes):
            mu = mu.mu
        self.mu = N.asarray(mu, N.float64)
        self.order = self.mu.shape[0]-1

    def __str__(self):
        return str(self.mu)

    def __mul__(self, other):

        if not isinstance(other, IntrinsicVolumes):
            raise ValueError, 'expecting an IntrinsicVolumes instance'
        order = self.order + other.order + 1
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

    If product is not None, then this product (an instance
    of IntrinsicVolumes) will effectively
    be prepended to the search region in any call, but it will
    also affect the (quasi-)polynomial part of the EC density. For
    instance, Hotelling's T^2 random field has a sphere as product,
    as does Roy's maximum root.

    """

    def __init__(self, mu=[1], dfd=N.inf, search=[1], product=[1]):
        self.dfd = dfd
        IntrinsicVolumes.__init__(self, mu=mu)
        self.product = IntrinsicVolumes(product)
        self.search = IntrinsicVolumes(search)

    def __call__(self, x, search=None):
        """
        If search=[1], raising dim gives the EC densities.
        """

        x = N.asarray(x, N.float64)

        if search is None:
            search = self.search
        else:
            search = IntrinsicVolumes(search)

        search *= self.product
        if N.isfinite(self.dfd):
            q_even = ECquasi([0], m=self.dfd, exponent=0)
            q_odd = ECquasi([0], m=self.dfd, exponent=0.5)
        else:
            q_even = N.poly1d([0])
            q_odd = N.poly1d([0])

        for k in range(search.mu.shape[0]):
            if k > 0:
                q = self.quasi(k)
                c = float(search.mu[k]) * N.power(2*N.pi, -(k+1)/2.)
                if N.isfinite(self.dfd):
                    q_even += q[0] * c
                    q_odd += q[1] * c
                else:
                    q_even += q * c

        _rho = q_even(x) + q_odd(x)

        if N.isfinite(self.dfd):
            _rho *= N.power(1 + x**2/self.dfd, -(self.dfd-1)/2.)
        else:
            _rho *= N.exp(-x**2/2.)


        if search.mu[0] * self.mu[0] != 0.:
            # tail probability is not "quasi-polynomial"
            if not N.isfinite(self.dfd):
                P = scipy.stats.norm.sf
            else:
                P = lambda x: scipy.stats.t.sf(x, self.dfd)
            _rho += P(x) * search.mu[0] * self.mu[0]
        return _rho

    def pvalue(self, x, search=None):
        return self(x, search=search)

    def integ(self, m=None, k=None):
        raise NotImplementedError # this could be done with scipy.stats.t,
                                  # at least m=1

    def density(self, x, dim):
        """
        The EC density in dimension dim.
        """

        return self(x, search=[0]*dim+[1])


    def _quasi_polynomials(self, dim):
        """
        Generate a list of quasi-polynomials for use in
        EC density calculation.
        """

        c = self.mu / N.power(2*N.pi, N.arange(self.order+1.)/2.)
        p = self.product.mu

        quasi_polynomials = []

        for k in range(c.shape[0]):
            if k+dim > 0:
                _q = ECquasi(Q(k+dim, dfd=self.dfd),
                             m=self.dfd,
                             exponent=k/2.)
                _q *= float(c[k])
                quasi_polynomials.append(_q)
        return quasi_polynomials

    def quasi(self, dim):
        """
        (Quasi-)polynomial parts of the EC density in dimension dim
        ignoring a factor of (2\pi)^{-(dim+1)/2} in front.

        """

        q_even = ECquasi([0], m=self.dfd, exponent=0)
        q_odd = ECquasi([0], m=self.dfd, exponent=0.5)

        quasi_polynomials = self._quasi_polynomials(dim)
        for k in range(len(quasi_polynomials)):
            _q = quasi_polynomials[k]
            if k % 2 == 0:
                q_even += _q
            else:
                q_odd += _q

        if not N.isfinite(self.dfd):
            q_even += q_odd
            return N.poly1d(q_even.coeffs)

        else:
            return (q_even, q_odd)

Gaussian = ECcone

def mu_sphere(n, j, r=1):
    """
    Return mu_j(S_r(R^n)), the j-th Lipschitz Killing
    curvature of the sphere of radius r in R^n.

    From Chapter 6 of

    Adler & Taylor, 'Random Fields and Geometry'. 2006.

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

def spherical_search(n, r=1):
    """
    A spherical search region of radius r.
    """

    return IntrinsicVolumes([mu_sphere(n,j,r=r) for j in range(n)])

def ball_search(n, r=1):
    """
    A ball-shaped search region of radius r.
    """
    return IntrinsicVolumes([mu_ball(n,j,r=r) for j in range(n+1)])

def volume2ball(vol, d=3):
    """
    Approximate intrinsic volumes of a set with a given volume by those of a ball with a given dimension and equal volume.
    """

    if d > 0:
        mu = N.zeros((d+1,), N.float64)
        r = N.power(vol * 1. / mu_ball(d, d), 1./d)
        return ball_search(d, r=r)
    else:
        return IntrinsicVolumes([1])

class ChiSquared(ECcone):

    """
    EC densities for a Chi-Squared(n) random field.
    """

    def __init__(self, dfn, dfd=N.inf, search=[1]):
        self.dfn = dfn
        ECcone.__init__(self, mu=spherical_search(self.dfn), search=search, dfd=dfd)

    def __call__(self, x, search=None):

        return ECcone.__call__(self, N.sqrt(x), search=search)

class TStat(ECcone):
    """
    EC densities for a t random field.
    """

    def __init__(self, dfd=N.inf, search=[1]):
        ECcone.__init__(self, mu=[1], dfd=dfd, search=search)

class FStat(ChiSquared):

    """
    EC densities for a F random field.
    """

    def __call__(self, x, search=None):
        return ECcone.__call__(self, N.sqrt(x * self.dfn), search=search)

class Roy(FStat):
    """
    Roy's maximum root: maximize an F_{n,m} statistic over a sphere
    of dimension k.
    """

    def __init__(self, dfn=1, dfd=N.inf, k=1, search=[1]):
        FStat.__init__(dfd=dfd, dfn=dfn, search=search)
        self.sphere = IntrinsicVolumes([mu_sphere(k,i) for i in range(k)])
        self.k = k

    def __call__(self, x, search=None):

        if search is None:
            search = self.sphere
        else:
            search = IntrinsicVolumes(search) * self.sphere
        return FStat.__call__(self, x, search=search)

class Hotelling(ECcone):
    """
    Hotelling's T^2: maximize an F_{1,m}=T_m^2 statistic over a sphere
    of dimension k.
    """

    def __init__(self, dfd=N.inf, k=1, search=[1]):
        product = spherical_search(k).mu / 2
        self.k = k
        ECcone.__init__(self, mu=spherical_search(1), search=search, dfd=dfd, product=product)

    def __call__(self, x, search=None):
        return ECcone.__call__(self, N.sqrt(x), search=search)


class OneSidedF(FStat):

    def __call__(self, x, dim=0, search=[1]):
        d1 = FStat.__call__(self, x, dim=dim, search=search)
        self.dfd -= 1
        d2 = FStat.__call__(self, x, dim=dim, search=search)
        self.dfd += 1
        return (d1 - d2) / 2.

class ChiBarSquared(ChiSquared):

    def _getmu(self):
        x = N.linspace(0, 2 * self.dfn, 100)
        sf = 0.
        g = Gaussian()
        for i in range(1, self.dfn+1):
            sf += binomial(self.dfn, i) * scipy.stats.chi.sf(x, i) / N.power(2., self.dfn)

        d = N.transpose(N.array([g.density(N.sqrt(x), j) for j in range(self.dfn)]))
        c = N.dot(pinv(d), sf)
        sf += 1. / N.power(2, self.dfn)
        self.mu = IntrinsicVolumes(c)

    def __init__(self, dfn=1, search=[1]):
        ChiSquared.__init__(self, dfn=dfn, search=search)
        self._getmu()

    def __call__(self, x, dim=0, search=[1]):

        if search is None:
            search = self.stat
        else:
            search = IntrinsicVolumes(search) * self.stat
        return FStat.__call__(self, x, dim=dim, search=search)

