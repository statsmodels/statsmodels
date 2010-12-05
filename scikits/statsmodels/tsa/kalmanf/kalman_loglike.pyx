cimport numpy as np
from numpy import dot, identity, kron, log, zeros, pi, exp, eye, sum
from numpy.linalg import inv
from scipy.linalg.fblas import dgemm, dger
cimport cython

ctypedef np.float64_t DOUBLE

# call from KalmanFilter.loglike would be on line 531
#fast_kalman_loglike.loglike(params, k, p, q, r, Z_mat, R_mat, T_mat)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def kalman_loglike(np.ndarray[DOUBLE, ndim=1] y,
                   unsigned int k, unsigned int p, unsigned int q,
                   unsigned int r, unsigned int nobs,
                   np.ndarray[DOUBLE, ndim=2] Z_mat,
                   np.ndarray[DOUBLE, ndim=2] R_mat,
                   np.ndarray[DOUBLE, ndim=2] T_mat):
    """
    Cython version of the Kalman filter recursions for an ARMA process.
    """
    cdef unsigned int m = Z_mat.shape[1]

    # store forecast-errors
    cdef np.ndarray[DOUBLE, ndim=2] v = zeros((nobs,1))
    # store variance of forecast errors
    cdef np.ndarray[DOUBLE, ndim=2] F = zeros((nobs, 1))
    cdef np.ndarray[DOUBLE, ndim=2] v_mat
    cdef np.ndarray[DOUBLE, ndim=2] F_mat
    cdef np.ndarray[DOUBLE, ndim=2] Finv
    cdef np.ndarray[DOUBLE, ndim=2] K
    cdef np.ndarray[DOUBLE, ndim=2] L
    cdef np.ndarray[DOUBLE, ndim=2] loglikelihood = zeros((1,1))
    cdef np.ndarray[DOUBLE, ndim=2] loglike
    cdef unsigned int i
    cdef double sigma2

    # initial state
    cdef np.ndarray[DOUBLE, ndim=2] alpha = zeros((m,1))

    # initial variance
    cdef np.ndarray[DOUBLE, ndim=2] P = dot(inv(identity(m**2)-kron(T_mat,
                                        T_mat)),dot(R_mat,
                                        R_mat.T).ravel('F')).reshape(r,r,
                                        order='F')



    for i in xrange(nobs):
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

    sigma2 = 1./nobs * sum(v**2 / F)
    loglike = -.5 *(loglikelihood + nobs*log(sigma2))
    loglike -= nobs/2. * (log(2*pi) + 1)
    return loglike, sigma2


