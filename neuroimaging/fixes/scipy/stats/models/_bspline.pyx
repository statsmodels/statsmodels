cimport c_python
cimport c_numpy
import numpy

cdef extern from "stdlib.h":
    cdef int abs(int a)

# constants used in quadrature integration below
# using nq=18 will give exact quadrature
# integrals up to splines of order???

from scipy.special.orthogonal import p_roots

# Numpy must be initialized
c_numpy.import_array()

def basis(c_numpy.ndarray x,
          c_numpy.ndarray knots,
          int m,
          int lower=0,
          int upper=-1,
          int deriv=0,
          double delta=1.0e-06,
          add_knots=True):

    """


    Parameters
    ----------
    x : array, points where the B-splines are to be evaluated
    knots : array, internal knots of the B-spline (see below for the
            boundary knots), assumed to be sorted
    m : integer, order of the B-spline, m=4 corresponds to cubic B-spline
    lower : integer, which of the basis functions to compute, lower limit
    upper : integer, which of the basis functions to compute, upper limit
            if < 0, defaults to (knots.shape[0]+m-2) when knots are
            added
    deriv : integer, order of derivative of B-spline to compute
    delta : float, offset to define boundary knots at right hand endpoint
    add_knots : prepend and append boundary knots using offset at right?

    Returns
    -------
    B : array of shape (upper-lower,) + x.shape
        Compute m-th order B-spline basis functions with specified internal
        knots at the points x. If deriv > 0, it is the
        derivative of the basis functions.

        The i-th row of B are the values B^(deriv)(m,i+1)(x) where B(m,i+1) is
        defined on p.161 of the first reference below.

    Notes
    -----

    Boundary knots are added using the following rule:

    tau = (m-1)*[knots[0]] + knots + [knots[-1]+delta*(knots.max()-knots.min())]*(m-1)

    The offset delta is arbitrary and ensures that all of the basis
    functions are continuously twice differentiable up to knots[-1].

    Values are computed using the recursive definition:

    C(m,i)(x) = ((x-tau[i])*C(m-1,i)(x)/(tau[i+m-1]-tau[i]) +
                 (tau[i+m] - x)*C(m-1,i+1)(x)/(tau[i+m]-tau[i+1]))

    where C(m,i) is the the function B(m,i+1)(x) in
    the first reference below (p.161).

    Derivatives are computed using a similar recursive definition

    D(m,i,j)(x) = (((x-tau[i])*(j <= 0) + (m-1) * (j > 0)) *
                   D(m-1,i,max(j-1,0))(x)/(tau[i+m-1]-tau[i]) +
                   ((tau[i+m]-x)*(j <= 0) - (m-1) * (j > 0)) *
                   D(m-1,i+1,max(j-1,0))(x)/(tau[i+m-1]-tau[i]))
    where D(m,i,j) is the j-th derivative of the function B(m,i+1)(x)
    in the first reference below (p.161).

    The recursive definition for the first derivative can be found
    many places, for instance in the second reference below. Extending
    it to higher order derivatives is not too difficult.

    Reference
    ---------

    1) Hastie, Tibshirani and Friedman (2001). "The Elements of Statistical
    Learning." Springer-Verlag. 536 pages. Chapter 5.

    2) Prochaszkova, J. "Derivative of a B-spline function."
    http://mat.fsv.cvut.cz/gcg/sbornik/prochazkova.pdf
    """

    cdef int nbasis, index
    cdef long ntotal = 1
    cdef c_numpy.ndarray b, B, nknots
    cdef double f0, f1
    cdef double *resultd
    cdef double *place, *b0, *b1
    cdef double d0, d1

    delta *= knots.max() - knots.min()
    if add_knots:
        nknots = numpy.hstack([[knots[0]]*(m-1), knots, [knots[-1]+delta]*(m-1)])
    else:
        nknots = knots

    if deriv < 0:
        raise ValueError, 'order of derivative must be non-negative'
    if lower < 0:
        raise ValueError, 'lower index must be non-negative'

    if add_knots:
        if upper < 0:
            upper = knots.shape[0]+m-3
        if upper >= knots.shape[0]+m-2:
            raise ValueError, 'upper index must be < %d' % (knots.shape[0]+m-2,)
    else:
        if upper < 0:
            upper = knots.shape[0] - m - 1

    for i from 0 <= i < x.nd:
        ntotal *= x.dimensions[i]

    nbasis = upper + 1 - lower

    B = numpy.zeros((nbasis,) + x.shape)
    resultd = <double *>B.data

    if m == 1:
        for i from 0 <= i < nbasis:
            index = i + lower
            place = resultd + (<int>i)*ntotal
            for k from 0 <= k < ntotal:
                place[k] = <double> (x[k] >= nknots[index]) * (x[k] < nknots[index+1])

    else:

        b = basis(x, nknots, m-1, lower, upper+1, add_knots=False, deriv=max(deriv-1,0))
        for i from 0 <= i < nbasis:
            index = i + lower

            b0 = <double *>b.data + (<int>i)*ntotal
            b1 = <double *>b.data + (<int>i + 1)*ntotal
            place = resultd + (<int>i)*ntotal

            d0 = 1. / (nknots[index+m-1] - nknots[index])
            if d0 == numpy.inf:
                d0 = 0.
            d1 = 1. / (nknots[index+m] - nknots[index+1])
            if d1 == numpy.inf:
                d1 = 0.

            for k from 0 <= k < ntotal:
                f0 = ((deriv == 0) * (x[k] - nknots[index]) + (deriv > 0) * (m-1)) * d0
                f1 = ((deriv == 0) * (nknots[index+m] - x[k]) - (deriv > 0) * (m-1)) * d1
                place[k] = f0*b0[k] + f1*b1[k]

    return numpy.squeeze(B)

def gram(c_numpy.ndarray knots,
          int m, int dl=2, int dr=2,
          double delta=1.0e-06):
    """

    Parameters
    ----------
    knots : array, internal knots of the B-spline (see below for the
            boundary knots), assumed to be sorted
    m : integer, order of the B-spline, m=4 corresponds to cubic B-spline
    dl : order of derivative of left hand side to compute
    dr : order of derivative of right hand side to compute
    delta : float, offset to define boundary knots at right hand endpoint

    Returns
    -------

    result : float, the bilinear functional given by
        integral_knots[0]^knots[1] D(m,l+m-1,dl)(x) D(m,r+m-1,dr)(x) dx

    See Also
    --------
    basis : function defining the B-spline basis functions, the functions
            D(m,i,j) are defined in its docstring

    Notes
    -----

    The integral is computed using Gaussian quadrature on the
    intervals determined by the interior knots. On these intervals,
    the integrand is a polynomial of order j=(m-1-dl)+(m-1-dr),
    hence Gaussian quadrature can be used with the number of points
    given by ceil(j+1/2.)

    """

    j = (m-1-dl) + (m-1-dr) # order of polynomials
    nq = numpy.ceil((j+1.)/2) # number of points for Gaussian quadrature to be exact

    nk = knots.shape[0]+m-2
    qx, qw = p_roots(nq)

    nknots = numpy.hstack([[knots[0]]*(m-1), knots, [knots[-1]+delta]*(m-1)])

    qw = numpy.multiply.outer(nknots[1:] - nknots[:-1], qw) / 2.
    xs = (numpy.multiply.outer(nknots[1:], qx + 1) +
          numpy.multiply.outer(nknots[:-1], 1 - qx)) / 2.

    xs.shape = numpy.product(xs.shape)

    Bl = numpy.zeros((nk, m, qx.shape[0]))
    Br = numpy.zeros((nk, m, qx.shape[0]))

    gram = numpy.zeros((m,nk))

    for k from 0 <= k < knots.shape[0]+m-2:
        z = xs[k*qx.shape[0]:(k+m)*qx.shape[0]]
        bl = basis(z, nknots, m,
                   lower=k,
                   upper=k,
                   deriv=dl,
                   add_knots=False)
        bl.shape = (m, qx.shape[0])
        Bl[k] = bl
        if dl != dr:
            br = basis(z, nknots, m, deriv=dr,
                       upper=k, lower=k,
                       add_knots=False)
            br.shape = (m, qx.shape[0])
            Br[k] = br
        else:
            Br[k] = Bl[k]

        for j from 0 <= j <= min(m-1, k):
            gram[j,k-j] = ((Bl[k-j,j:] * Br[k,:(m-j)]) * qw[k:(k+m-j)]).sum()

    return gram


def invband(c_numpy.ndarray M):
    '''
    Parameter
    ---------

    M : array, Cholesky decomposition of a symmetric positive definite banded
        matrix A in lower diagonal form.

    Returns
    -------

    inv : first (m+1) bands of inv(A)

    Notes
    -----

    Algorithm taken from

    Hutchison, M. and Hoog, F. "Smoothing noisy data with spline functions."
    Numerische Mathematik, 47(1), 99-106.


    '''

    cdef c_numpy.ndarray inv
    cdef double *Md, *invd
    cdef int idx, idy

    m, n = M.shape

    Md = <double *>M.data

    inv = numpy.zeros((M.shape[0], M.shape[1]))
    invd = <double *>inv.data

    inv[0] = 1. / (M[0]**2)
    for i in range(1, M.shape[0]):
        M[i] /= M[0]
    M[0] = 1.

    for i from n-1 >= i >= 0:
        I = min(<int>n-i, <int>m)
        for l from 1 <= l < I:
            for k from 1 <= k < I:
                idx = abs(k - l)
                idy = <int>i + min(l,k)
                # invM[i,i+l] -= M[i,i+k] * invM[i+k,i+l]
#                inv[l,i] -= M[k,i] * inv[idx,idy]
                invd[l*(<int>n)+i] -= Md[k*(<int>n)+i] * invd[idx*(<int>n)+idy]
        for l from 1 <= l < I:
            # invM[i,i] -= M[i,i+l] * invM[i,i+k]
#            inv[0,i] -= M[l,i] * inv[l,i]
            invd[i] -= Md[l*(<int>n)+i] * invd[l*(<int>n)+i]


    return inv

def _trace_symbanded(a, b, lower=0):
    """
    Compute the trace(ab) for two upper or banded real symmetric matrices
    stored either in either upper or lower form.

    INPUTS:
       a, b    -- two banded real symmetric matrices (either lower or upper)
       lower   -- if True, a and b are assumed to be the lower half


    OUTPUTS: trace
       trace   -- trace(ab)

    """

    if lower:
        t = _zero_triband(a * b, lower=1)
        return t[0].sum() + 2 * t[1:].sum()
    else:
        t = _zero_triband(a * b, lower=0)
        return t[-1].sum() + 2 * t[:-1].sum()


def _zero_triband(a, lower=0):
    """
    Explicitly zero out unused elements of a real symmetric banded matrix.

    INPUTS:
       a   -- a real symmetric banded matrix (either upper or lower hald)
       lower   -- if True, a is assumed to be the lower half

    """

    nrow, ncol = a.shape
    if lower:
        for i in range(nrow): a[i,(ncol-i):] = 0.
    else:
        for i in range(nrow): a[i,0:i] = 0.
    return a

def _band2array(a, lower=0, symmetric=False, hermitian=False):
    """
    Take an upper or lower triangular banded matrix and return a
    numpy array.

    INPUTS:
       a         -- a matrix in upper or lower triangular banded matrix
       lower     -- is the matrix upper or lower triangular?
       symmetric -- if True, return the original result plus its transpose
       hermitian -- if True (and symmetric False), return the original
                    result plus its conjugate transposed

    """

    n = a.shape[1]
    r = a.shape[0]
    _a = 0

    if not lower:
        for j in range(r):
            _b = numpy.diag(a[r-1-j],k=j)[j:(n+j),j:(n+j)]
            _a += _b
            if symmetric and j > 0: _a += _b.T
            elif hermitian and j > 0: _a += _b.conjugate().T
    else:
        for j in range(r):
            _b = numpy.diag(a[j],k=j)[0:n,0:n]
            _a += _b
            if symmetric and j > 0: _a += _b.T
            elif hermitian and j > 0: _a += _b.conjugate().T
        _a = _a.T

    return _a


def _upper2lower(ub):
    """
    Convert upper triangular banded matrix to lower banded form.

    INPUTS:
       ub  -- an upper triangular banded matrix

    OUTPUTS: lb
       lb  -- a lower triangular banded matrix with same entries
              as ub
    """

    lb = numpy.zeros(ub.shape, ub.dtype)
    nrow, ncol = ub.shape
    for i in range(ub.shape[0]):
        lb[i,0:(ncol-i)] = ub[nrow-1-i,i:ncol]
        lb[i,(ncol-i):] = ub[nrow-1-i,0:i]
    return lb

def _lower2upper(lb):
    """
    Convert lower triangular banded matrix to upper banded form.

    INPUTS:
       lb  -- a lower triangular banded matrix

    OUTPUTS: ub
       ub  -- an upper triangular banded matrix with same entries
              as lb
    """

    ub = numpy.zeros(lb.shape, lb.dtype)
    nrow, ncol = lb.shape
    for i in range(lb.shape[0]):
        ub[nrow-1-i,i:ncol] = lb[i,0:(ncol-i)]
        ub[nrow-1-i,0:i] = lb[i,(ncol-i):]
    return ub

def _triangle2unit(tb, lower=0):
    """
    Take a banded triangular matrix and return its diagonal and the
    unit matrix: the banded triangular matrix with 1's on the diagonal,
    i.e. each row is divided by the corresponding entry on the diagonal.

    INPUTS:
       tb    -- a lower triangular banded matrix
       lower -- if True, then tb is assumed to be lower triangular banded,
                in which case return value is also lower triangular banded.

    OUTPUTS: d, b
       d     -- diagonal entries of tb
       b     -- unit matrix: if lower is False, b is upper triangular
                banded and its rows of have been divided by d,
                else lower is True, b is lower triangular banded
                and its columns have been divided by d.

    """

    if lower: d = tb[0].copy()
    else: d = tb[-1].copy()

    if lower: return d, (tb / d)
    else:
        l = _upper2lower(tb)
        return d, _lower2upper(l / d)

__all__ = ["basis", "gram", "invband", "gram2"]

