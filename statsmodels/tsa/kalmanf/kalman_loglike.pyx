# cython: profile=True
from numpy cimport float64_t, ndarray, complex128_t, complex64_t
from numpy import log as nplog
from numpy import identity, dot, kron, zeros, pi, exp, eye, sum, empty, ones
from numpy.linalg import pinv
cimport cython

ctypedef float64_t DOUBLE
ctypedef complex128_t COMPLEX128
ctypedef complex64_t COMPLEX64

cdef extern from "math.h":
    double log(double x)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def kalman_filter_double(ndarray[DOUBLE, ndim=1] y,
                   unsigned int k, unsigned int p, unsigned int q,
                  unsigned int r, unsigned int nobs,
                   ndarray[DOUBLE, ndim=2] Z_mat,
                   ndarray[DOUBLE, ndim=2] R_mat,
                   ndarray[DOUBLE, ndim=2] T_mat):
    """
    Cython version of the Kalman filter recursions for an ARMA process.
    """
    m = Z_mat.shape[1]
    # store forecast-errors
    v = zeros((nobs,1))
    # store variance of forecast errors
    F = ones((nobs,1))
    loglikelihood = zeros((1,1))
    cdef int i = 0
    # initial state
#    cdef np.ndarray[DOUBLE, ndim=2] alpha = zeros((m,1))
    alpha = zeros((m,1))
    # initial variance
    P = dot(pinv(identity(m**2)-kron(T_mat, T_mat)),dot(R_mat,
            R_mat.T).ravel('F')).reshape(r,r, order='F')
    F_mat = 0
    while not F_mat == 1 and i < nobs:
        # Predict
        v_mat = y[i] - dot(Z_mat,alpha) # one-step forecast error
        v[i] = v_mat
        F_mat = dot(dot(Z_mat, P), Z_mat.T)
        F[i] = F_mat
        Finv = 1./F_mat # always scalar for univariate series
        K = dot(dot(dot(T_mat,P),Z_mat.T),Finv) # Kalman Gain Matrix
        # update state
        alpha = dot(T_mat, alpha) + dot(K,v_mat)
        L = T_mat - dot(K,Z_mat)
        P = dot(dot(T_mat, P), L.T) + dot(R_mat, R_mat.T)
        loglikelihood += log(F_mat)
        i+=1
    for i in xrange(i,nobs):
        v_mat = y[i] - dot(Z_mat,alpha)
        v[i] = v_mat
        alpha = dot(T_mat, alpha) + dot(K, v_mat)
    return v, F, loglikelihood

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def kalman_filter_complex(ndarray[COMPLEX128, ndim=1] y,
                   unsigned int k, unsigned int p, unsigned int q,
                  unsigned int r, unsigned int nobs,
                   ndarray[DOUBLE, ndim=2] Z_mat,
                   ndarray[COMPLEX128, ndim=2] R_mat,
                   ndarray[COMPLEX128, ndim=2] T_mat):
    """
    Cython version of the Kalman filter recursions for an ARMA process.
    """
    m = Z_mat.shape[1]
    # store forecast-errors
    v = zeros((nobs,1), dtype=complex)
    # store variance of forecast errors
    F = ones((nobs,1), dtype=complex)
    loglikelihood = zeros((1,1), dtype=complex)
    cdef int i = 0
    # initial state
#    cdef np.ndarray[DOUBLE, ndim=2] alpha = zeros((m,1))
    alpha = zeros((m,1))
    # initial variance
    P = dot(pinv(identity(m**2)-kron(T_mat, T_mat)),dot(R_mat,
            R_mat.T).ravel('F')).reshape(r,r, order='F')
    F_mat = 0
    while not F_mat == 1 and i < nobs:
        # Predict
        v_mat = y[i] - dot(Z_mat,alpha) # one-step forecast error
        v[i] = v_mat
        F_mat = dot(dot(Z_mat, P), Z_mat.T)
        F[i] = F_mat
        Finv = 1./F_mat # always scalar for univariate series
        K = dot(dot(dot(T_mat,P),Z_mat.T),Finv) # Kalman Gain Matrix
        # update state
        alpha = dot(T_mat, alpha) + dot(K,v_mat)
        L = T_mat - dot(K,Z_mat)
        P = dot(dot(T_mat, P), L.T) + dot(R_mat, R_mat.T)
        loglikelihood += nplog(F_mat)
        i+=1
    for i in xrange(i,nobs):
        v_mat = y[i] - dot(Z_mat,alpha)
        v[i] = v_mat
        alpha = dot(T_mat, alpha) + dot(K, v_mat)
    return v,F,loglikelihood

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def kalman_loglike_double(ndarray[DOUBLE, ndim=1] y,
                   unsigned int k, unsigned int p, unsigned int q,
                  unsigned int r, unsigned int nobs,
                   ndarray[DOUBLE, ndim=2] Z_mat,
                   ndarray[DOUBLE, ndim=2] R_mat,
                   ndarray[DOUBLE, ndim=2] T_mat):
    """
    Cython version of the Kalman filter recursions for an ARMA process.
    """
    v, F, loglikelihood = kalman_filter_double(y,k,p,q,r,nobs,Z_mat,R_mat,T_mat)
    sigma2 = 1./nobs * sum(v**2 / F)
    loglike = -.5 *(loglikelihood + nobs*log(sigma2))
    loglike -= nobs/2. * (log(2*pi) + 1)
    return loglike, sigma2

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def kalman_loglike_complex(ndarray[COMPLEX128, ndim=1] y,
                   unsigned int k, unsigned int p, unsigned int q,
                  unsigned int r, unsigned int nobs,
                   ndarray[DOUBLE, ndim=2] Z_mat,
                   ndarray[COMPLEX128, ndim=2] R_mat,
                   ndarray[COMPLEX128, ndim=2] T_mat):
    """
    Cython version of the Kalman filter recursions for an ARMA process.
    """
    v,F,loglikelihood = kalman_filter_complex(y,k,p,q,r,nobs,Z_mat,R_mat,T_mat)
    sigma2 = 1./nobs * sum(v**2 / F)
    loglike = -.5 *(loglikelihood + nobs*log(sigma2))
    loglike -= nobs/2. * (log(2*pi) + 1)
    return loglike, sigma2
