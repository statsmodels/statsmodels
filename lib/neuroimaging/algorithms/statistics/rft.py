__docformat__ = 'restructuredtext'

import numpy as N
from numpy.linalg import pinv

from scipy import stats
from scipy.misc import factorial
from scipy.special import gamma, gammaln, beta, hermitenorm

def binomial(n, j):
    """
    Binomial coefficient::

           n!
       ---------
       (n-j)! j!
    """

    if n <= j or n == 0:
        return 0.
    elif j == 0:
        return 1.
    return 1./(beta(n-j+1, j+1)*(n+1))

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

class ECquasi(N.poly1d):

    """
    A subclass of poly1d consisting of polynomials with a premultiplier of the
    form

    (1 + x^2/m)^-exponent

    where m is a non-negative float (possibly infinity, in which case the
    function is a polynomial) and exponent is a non-negative multiple of 1/2.

    These arise often in the EC densities.

    >>> import numpy
    >>> from neuroimaging.algorithms.statistics.rft import ECquasi
    >>> x = numpy.linspace(0,1,101)

    >>> a = ECquasi([3,4,5])
    >>> a
    ECquasi([3, 4, 5],m=inf, exponent=0.000000)
    >>> a(3) == 3*3**2 + 4*3 + 5
    True

    >>> b = ECquasi(a.coeffs, m=30, exponent=4)
    >>> numpy.allclose(b(x), a(x) * numpy.power(1+x**2/30, -4))
    True
    >>>
    """


    def __init__(self, c_or_r, r=0, exponent=None, m=None):
        N.poly1d.__init__(self, c_or_r, r=r, variable='x')

        if exponent is None and not hasattr(self, 'exponent'):
            self.exponent = 0
        elif not hasattr(self, 'exponent'):
            self.exponent = exponent


        if m is None and not hasattr(self, 'm'):
            self.m = N.inf
        elif not hasattr(self, 'm'):
            self.m = m

        if not N.isfinite(self.m):
            self.exponent = 0.

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
        ECquasi([  1.11111111e-04,   1.48148148e-04,   1.07407407e-02,
                 1.33333333e-02,   3.66666667e-01,   4.00000000e-01,
                 5.00000000e+00,   4.00000000e+00,   2.00000000e+01],m=30.000000, exponent=7.000000)
        >>> numpy.allclose(c(x), b(x))
        True
        >>>
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
            if float(val) > 0 or val == N.inf:
                self.__dict__[key] = val
            else: raise ValueError, 'expecting positive float or inf'
        else: N.poly1d.__setattr__(self, key, val)

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
        >>>
        """

        if self.m != other.m:
            #raise ValueError, 'quasi polynomials are not compatible, m disagrees'
            return False
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
        >>>
        """
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
        >>>
        """

        if N.isscalar(other):
            return ECquasi(self.coeffs * other,
                           m=self.m,
                           exponent=self.exponent)
        elif self.compatible(other):
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
        >>> a(3) == 3*3**2 + 4*3 + 5
        True

        >>> b = ECquasi(a.coeffs, m=30, exponent=4)
        >>> numpy.allclose(b(x), a(x) * numpy.power(1+x**2/30, -4))
        True
        >>>
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
        >>> print b-c
        ECquasi([ 3,  3, 18],m=30.000000, exponent=4.000000)
        >>>
        """
        return self + (other * -1)

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
        >>>
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
            d = self.deriv(m=1)
            return d.deriv(m=m-1)

class fnsum:

    def __init__(self, *items):
        self.items = list(items)

    def __call__(self, x):
        v = 0
        for q in self.items:
            v += q(x)
        return v

class IntrinsicVolumes:
    """
    A simple class that exists only to compute the intrinsic volumes of
    products of sets (that themselves have intrinsic volumes, of course).
    """

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
        mu = N.zeros(order)

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
        Get expected EC for a search region (default is self.search which
        itself defaults to [1] giving the survival function.
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
                P = stats.norm.sf
            else:
                P = lambda x: stats.t.sf(x, self.dfd)
            _rho += P(x) * search.mu[0] * self.mu[0]
        return _rho

    def pvalue(self, x, search=None):
        return self(x, search=search)

    def integ(self, m=None, k=None):
        raise NotImplementedError # this could be done with stats.t,
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
            if _q.exponent % 1 == 0:
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

class FStat(ECcone):

    """
    EC densities for a F random field.
    """

    def __init__(self, dfn, dfd=N.inf, search=[1]):
        self.dfn = dfn
        ECcone.__init__(self, mu=spherical_search(self.dfn), search=search, dfd=dfd)

    def __call__(self, x, search=None):
        return ECcone.__call__(self, N.sqrt(x * self.dfn), search=search)


class Roy(ECcone):
    """
    Roy's maximum root: maximize an F_{dfd,dfn} statistic over a sphere
    of dimension k.
    """

    def __init__(self, dfn=1, dfd=N.inf, k=1, search=[1]):
        product = spherical_search(k)
        self.k = k
        self.dfn = dfn
        ECcone.__init__(self, mu=spherical_search(self.dfn),
                        search=search, dfd=dfd, product=product)

    def __call__(self, x, search=None):
        return ECcone.__call__(self, N.sqrt(x * self.dfn), search=search)

class MultilinearForm(ECcone):
    """
    Maximize a multivariate Gaussian form, maximized over spheres
    of dimension dims. See

    Kuri, S. & Takemura, A. (2001).
    'Tail probabilities of the maxima of multilinear forms and
    their applications.' Ann, Statist. 29(2): 328-371.
    """

    def __init__(self, *dims, **keywords):
        product = IntrinsicVolumes([1])

        if keywords.has_key('search'):
            search = keywords['search']
        else:
            search = [1]

        for d in dims:
            product *= spherical_search(d)
        product.mu /= 2.**(len(dims)-1)

        ECcone.__init__(self, search=search, product=product)

class Hotelling(ECcone):
    """
    Hotelling's T^2: maximize an F_{1,dfd}=T_dfd^2 statistic over a sphere
    of dimension k.
    """

    def __init__(self, dfd=N.inf, k=1, search=[1]):
        product = spherical_search(k)
        self.k = k
        ECcone.__init__(self, mu=[1], search=search, dfd=dfd, product=product)

    def __call__(self, x, search=None):
        return ECcone.__call__(self, N.sqrt(x), search=search)


class OneSidedF(ECcone):

    """

    EC densities for one-sided F statistic in

    Worsley, K.J. & Taylor, J.E. (2005). 'Detecting fMRI activation
    allowing for unknown latency of the hemodynamic response.'
    Neuroimage, 29,649-654.

    """

    def __init__(self, dfn, dfd=N.inf, search=[1]):
        self.dfn = dfn
        self.regions = [spherical_search(dfn), spherical_search(dfn-1)]
        ECcone.__init__(self, mu=spherical_search(self.dfn), search=search, dfd=dfd)

    def __call__(self, x, search=None):
        IntrinsicVolumes.__init__(self, self.regions[0])
        d1 = ECcone.__call__(self, N.sqrt(x * self.dfn), search=search)
        IntrinsicVolumes.__init__(self, self.regions[1])
        d2 = ECcone.__call__(self, N.sqrt(x * (self.dfn-1)), search=search)
        self.mu = self.regions[0].mu
        return (d1 - d2) * 0.5

class ChiBarSquared(ChiSquared):

    def _getmu(self):
        x = N.linspace(0, 2 * self.dfn, 100)
        sf = 0.
        g = Gaussian()
        for i in range(1, self.dfn+1):
            sf += binomial(self.dfn, i) * stats.chi.sf(x, i) / N.power(2., self.dfn)

        d = N.array([g.density(N.sqrt(x), j) for j in range(self.dfn)])
        c = N.dot(pinv(d.T), sf)
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

def scale_space(region, interval, kappa=1.):
    """
    Work out intrinsic volumes of region x interval in the
    scale space model. See

    Siegmund, D.O and Worsley, K.J. (1995). 'Testing for a signal
    with unknown location and scale in a stationary Gaussian random
    field.'  Annals of Statistics, 23:608-639.

    and

    Taylor, J.E. & Worsley, K.J. (2005). 'Random fields of multivariate
    test statistics, with applications to shape analysis and fMRI.'

    (available on http://www.math.mcgill.ca/keith


    """

    w1, w2 = interval
    region = IntrinsicVolumes(region)

    D = region.order

    out = N.zeros((D+2,))

    out[0] = region.mu[0]
    for i in range(1, D+2):
        if i < D+1:
            out[i] = (1./w1 + 1./w2) * region.mu[i] * 0.5
        for j in range(int(N.floor((D-i+1)/2.)+1)):
            denom = (i + 2*j - 1.)
            # w^-i/i when i=0
            # according to Keith Worsley the 2005 paper has a typo
            if denom == 0:
                f = N.log(w2/w1)
            else:
                f = (w1**(-i-2*j+1) - w2**(-i-2*j+1)) / denom
            f *= kappa**((1-2*j)/2.) * (-1)**j * factorial(int(denom))
            f /= (1 - 2*j) * (4*N.pi)**j * factorial(j) * factorial(i-1)
            out[i] += region.mu[int(denom)] * f
    return IntrinsicVolumes(out)
