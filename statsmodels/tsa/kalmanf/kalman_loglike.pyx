from numpy cimport float64_t, ndarray, complex128_t
from numpy import log as nplog
from numpy import (identity, dot, kron, pi, sum, zeros_like, ones,
                   asfortranarray)
from numpy.linalg import pinv
cimport cython
cimport numpy as cnp

cnp.import_array()

# included in Cython numpy headers
from numpy cimport PyArray_ZEROS
from scipy.linalg.cython_blas cimport dgemm, dgemv, zgemm, zgemv


ctypedef float64_t DOUBLE
ctypedef complex128_t dcomplex
cdef int FORTRAN = 1

cdef extern from "math.h":
    double log(double x)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def kalman_filter_double(double[:] y not None,
                         unsigned int k, unsigned int p, unsigned int q,
                         int r, unsigned int nobs,
                         double[::1, :] Z_mat,
                         double[::1, :] R_mat,
                         double[::1, :] T_mat):
    """
    Cython version of the Kalman filter recursions for an ARMA process.
    """
    cdef cnp.npy_intp yshape[2]
    yshape[0] = <cnp.npy_intp> nobs
    yshape[1] = <cnp.npy_intp> 1
    cdef cnp.npy_intp mshape[2]
    mshape[0] = <cnp.npy_intp> r
    mshape[1] = <cnp.npy_intp> 1
    cdef cnp.npy_intp r2shape[2]
    r2shape[0] = <cnp.npy_intp> r
    r2shape[1] = <cnp.npy_intp> r

    cdef:
        int one = 1  # univariate filter
        int ldz = Z_mat.strides[1] / sizeof(DOUBLE)
        int ldt = T_mat.strides[1] / sizeof(DOUBLE)
        int ldr = R_mat.strides[1] / sizeof(DOUBLE)
        # forecast errors
        ndarray[DOUBLE, ndim=2] v = PyArray_ZEROS(2, yshape, cnp.NPY_DOUBLE,
                                                  FORTRAN)
        # store variance of forecast errors
        double[::1, :] F = ones((nobs,1), order='F')  # variance of forecast errors
        double loglikelihood = 0
        int i = 0
        # initial state
        double[::1, :] alpha = PyArray_ZEROS(2, mshape, cnp.NPY_DOUBLE, FORTRAN)
        int lda = alpha.strides[1] / sizeof(DOUBLE)
        # initial variance
        double[::1, :] P = asfortranarray(dot(pinv(identity(r**2) -
                                                   kron(T_mat, T_mat)),
                                              dot(R_mat,
                                                  R_mat.T).ravel('F')
                                              ).reshape(r, r, order='F'))
        int ldp = P.strides[1] / sizeof(DOUBLE)
        double F_mat = 0.
        double Finv = 0.
        double v_mat = 0
        double[::1, :] K = PyArray_ZEROS(2, r2shape, cnp.NPY_DOUBLE, FORTRAN)
        int ldk = K.strides[1] / sizeof(DOUBLE)

        # pre-allocate some tmp arrays for the dgemm calls
        # T_mat rows x P cols
        double[::1, :] tmp1 = PyArray_ZEROS(2, r2shape, cnp.NPY_DOUBLE, FORTRAN)
        int ldt1 = tmp1.strides[1] / sizeof(DOUBLE)
        double[::1, :] tmp2 = zeros_like(alpha, order='F')  # K rows x v_mat cols
        int ldt2 = tmp2.strides[1] / sizeof(DOUBLE)
        double[::1, :] L = zeros_like(T_mat, order='F')
        int ldl = L.strides[1] / sizeof(DOUBLE)
        # T_mat rows x P cols
        double[::1, :] tmp3 = PyArray_ZEROS(2, r2shape, cnp.NPY_DOUBLE, FORTRAN)
        int ldt3 = tmp3.strides[1] / sizeof(DOUBLE)

        double alph = 1.0
        double beta = 0.0

    # NOTE: not sure about just checking F_mat[0, 0], did not appear to work
    while not F_mat == 1. and i < nobs:
        # Predict
        # Z_mat is just a selector matrix
        v_mat = y[i] - alpha[0, 0]
        v[i, 0] = v_mat

        # one-step forecast error covariance
        # Z_mat is just a selector matrix
        F_mat = P[0, 0]
        F[i,0] = F_mat
        Finv = 1. / F_mat  # always scalar for univariate series

        # compute Kalman Gain, K equivalent to:
        #   K = np.dot(np.dot(np.dot(T_mat, P), Z_mat.T), Finv)
        # in two steps:
        #   tmp1 = np.dot(T_mat, P)
        #   K = Finv * tmp1.dot(Z_mat.T)
        dgemm("N", "N", &r, &r, &r,
              &alph, &T_mat[0, 0], &ldt, &P[0, 0], &ldp,
              &beta, &tmp1[0, 0], &ldt1)
        dgemv("N", &r, &r,
              &Finv, &tmp1[0, 0], &ldt1, &Z_mat[0, 0], &one,
              &beta, &K[0, 0], &one)

        # update state; equivalent to:
        #   alpha = np.dot(T_mat, alpha) + np.dot(K, v_mat)
        # in two steps:
        #   tmp2 = np.dot(T_mat, alpha)
        #   alpha = tmp2 + np.dot(K, v_mat)
        dgemv("N", &r, &r,
              &alph, &T_mat[0, 0], &ldt, &alpha[0, 0], &one,
              &beta, &tmp2[0, 0], &one)
        for ii in range(r):
            alpha[ii, 0] = K[ii, 0] * v_mat + tmp2[ii, 0]

        # The next block is equivalent to:
        #   L = T_mat - np.dot(K, Z_mat)
        # in two steps:
        #   L = np.dot(K, Z_mat)
        #   L = T_mat - L
        dgemm("N", "N", &r, &r, &one,
              &alph, &K[0, 0], &ldk, &Z_mat[0, 0], &ldz,
              &beta, &L[0, 0], &ldl)
        for jj in range(r):
            for kk in range(r):
                L[jj, kk] = T_mat[jj, kk] - L[jj, kk]

        # The next block computes P equivalent to:
        #   P = np.dot(np.dot(T_mat, P), L.T) + np.dot(R_mat, R_mat.T)
        # in three steps:
        #   tmp5 = np.dot(R_mat, R_mat.T)
        #   tmp3 = np.dot(T_mat, P)
        #   P = np.dot(tmp3, L.T) + tmp5
        dgemm("N", "N", &r, &r, &r,
              &alph, &T_mat[0, 0], &ldt, &P[0, 0], &ldp,
              &beta, &tmp3[0, 0], &ldt3)
        dgemm("N", "T", &r, &r, &one,
              &alph, &R_mat[0, 0], &ldr, &R_mat[0, 0], &ldr,
              &beta, &P[0, 0], &ldp)
        dgemm("N", "T", &r, &r, &r,
              &alph, &tmp3[0, 0], &ldt3, &L[0, 0], &ldl,
              &alph, &P[0, 0], &ldp)

        loglikelihood += log(F_mat)
        i += 1

    for i in xrange(i,nobs):
        v_mat = y[i] - alpha[0, 0]
        v[i, 0] = v_mat
        # The next block is equivalent to:
        #   alpha = np.dot(T_mat, alpha) + np.dot(K, v_mat)
        # in two steps:
        #   tmp2 = np.dot(T_mat, alpha)
        #   alpha = np.dot(K, v_mat) + tmp2
        dgemm("N", "N", &r, &one, &r,
              &alph, &T_mat[0, 0], &ldt, &alpha[0, 0], &lda,
              &beta, &tmp2[0, 0], &ldt2)
        for ii in range(r):
            alpha[ii, 0] = K[ii, 0] * v_mat + tmp2[ii, 0]
    return v, F, loglikelihood


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def kalman_filter_complex(dcomplex[:] y,
                          unsigned int k, unsigned int p, unsigned int q,
                          int r, unsigned int nobs,
                          dcomplex[::1, :] Z_mat,
                          dcomplex[::1, :] R_mat,
                          dcomplex[::1, :] T_mat):
    """
    Cython version of the Kalman filter recursions for an ARMA process.
    """
    cdef cnp.npy_intp yshape[2]
    yshape[0] = <cnp.npy_intp> nobs
    yshape[1] = <cnp.npy_intp> 1
    cdef cnp.npy_intp mshape[2]
    mshape[0] = <cnp.npy_intp> r
    mshape[1] = <cnp.npy_intp> 1
    cdef cnp.npy_intp r2shape[2]
    r2shape[0] = <cnp.npy_intp> r
    r2shape[1] = <cnp.npy_intp> r


    cdef:
        int one = 1
        int ldz = Z_mat.strides[1] / sizeof(dcomplex)
        int ldt = T_mat.strides[1] / sizeof(dcomplex)
        int ldr = R_mat.strides[1] / sizeof(dcomplex)
        # forecast errors
        ndarray[complex, ndim=2] v = PyArray_ZEROS(2, yshape,
                                                   cnp.NPY_CDOUBLE, FORTRAN)
        # store variance of forecast errors
        dcomplex[::1, :] F = ones((nobs, 1), dtype=complex, order='F')
        dcomplex loglikelihood = 0 + 0j
        int i = 0
        # initial state
        dcomplex[::1, :] alpha = PyArray_ZEROS(2, mshape,
                                               cnp.NPY_CDOUBLE, FORTRAN)
        int lda = alpha.strides[1] / sizeof(dcomplex)
        # initial variance
        dcomplex[::1, :] P = asfortranarray(dot(pinv(identity(r**2) - kron(T_mat, T_mat)),
                                            dot(R_mat, R_mat.T).ravel('F')).reshape(r, r, order='F'))
        int ldp = P.strides[1] / sizeof(dcomplex)
        dcomplex F_mat = 0
        dcomplex Finv = 0
        dcomplex v_mat = 0
        dcomplex[::1, :] K = PyArray_ZEROS(2, mshape, cnp.NPY_CDOUBLE, FORTRAN)
        int ldk = K.strides[1] / sizeof(dcomplex)

        # pre-allocate some tmp arrays for the dgemm calls
        dcomplex[::1, :] tmp1 = PyArray_ZEROS(2, r2shape,
                                              cnp.NPY_CDOUBLE, FORTRAN)
        int ldt1 = tmp1.strides[1] / sizeof(dcomplex)
        dcomplex[::1, :] tmp2 = zeros_like(alpha, order='F')
        int ldt2 = tmp2.strides[1] / sizeof(dcomplex)
        dcomplex[::1, :] L = zeros_like(T_mat, dtype=complex, order='F')
        int ldl = L.strides[1] / sizeof(dcomplex)
        # T_mat rows x P cols
        dcomplex[::1, :] tmp3 = PyArray_ZEROS(2, r2shape,
                                              cnp.NPY_CDOUBLE, FORTRAN)
        int ldt3 = tmp3.strides[1] / sizeof(dcomplex)

        dcomplex alph = 1 + 0j
        dcomplex beta = 0

    while F_mat != 1 and i < nobs:
        # Z_mat is just a selector matrix
        v_mat = y[i] - alpha[0,0]
        v[i, 0] = v_mat

        # one-step forecast error covariance
        # Z_mat is just a selctor matrix so the below is equivalent
        F_mat = P[0, 0]
        F[i, 0] = F_mat
        Finv = 1. / F_mat  # always scalar for univariate series

        # compute Kalman Gain K, equivalent to:
        #   K = np.dot(np.dot(np.dot(T_mat, P), Z_mat.T), Finv)
        # in two steps:
        #   tmp1 = np.dot(T_mat, P)
        #   K = Finv * tmp1.dot(Z_mat.T)
        zgemm("N", "N", &r, &r, &r,
              &alph, &T_mat[0, 0], &ldt, &P[0, 0], &ldp,
              &beta, &tmp1[0, 0], &ldt1)
        zgemv("N", &r, &r,
              &Finv, &tmp1[0, 0], &ldt1, &Z_mat[0, 0], &one,
              &beta, &K[0, 0], &one)

        # update state, equivalent to:
        #   alpha = np.dot(T_mat, alpha) + np.dot(K, v_mat)
        # in two steps:
        #   tmp2 = np.dot(T_mat, alpha)
        #   alpha = tmp2 + np.dot(K, v_mat)
        zgemv("N", &r, &r,
              &alph, &T_mat[0, 0], &ldt, &alpha[0, 0], &one,
              &beta, &tmp2[0, 0], &one)
        for ii in range(r):
            alpha[ii, 0] = K[ii, 0] * v_mat + tmp2[ii, 0]

        # The next block is equivalent to:
        #   L = T_mat - np.dot(K, Z_mat)
        # in two steps:
        #   L = np.dot(K, Z_mat)
        #   L = T_mat - L
        zgemm("N", "N", &r, &r, &one,
              &alph, &K[0, 0], &ldk, &Z_mat[0, 0], &ldz,
              &beta, &L[0, 0], &ldl)
        for jj in range(r):
            for kk in range(r):
                L[jj, kk] = T_mat[jj, kk] - L[jj, kk]

        # The next block computes P equivalent to:
        #   P = np.dot(np.dot(T_mat, P), L.T) + np.dot(R_mat, R_mat.T)
        # in three steps:
        #   tmp5 = np.dot(R_mat, R_mat.T)
        #   tmp3 = np.dot(T_mat, P)
        #   P = np.dot(tmp3, L.T) + tmp5
        zgemm("N", "N", &r, &r, &r,
              &alph, &T_mat[0, 0], &ldt, &P[0, 0], &ldp,
              &beta, &tmp3[0, 0], &ldt3)
        zgemm("N", "T", &r, &r, &one,
              &alph, &R_mat[0, 0], &ldr, &R_mat[0, 0], &ldr,
              &beta, &P[0, 0], &ldp)
        zgemm("N", "T", &r, &r, &r,
              &alph, &tmp3[0, 0], &ldt3, &L[0, 0], &ldl,
              &alph, &P[0, 0], &ldp)

        loglikelihood += nplog(F_mat)
        i += 1

    for i in xrange(i, nobs):
        # Z_mat is just a selector
        v_mat = y[i] - alpha[0, 0]
        v[i, 0] = v_mat
        # The next block is equivalent to:
        #   alpha = np.dot(T_mat, alpha) + np.dot(K, v_mat)
        # in two steps:
        #   tmp2 = np.dot(T_mat, alpha)
        #   alpha = np.dot(K, v_mat) + tmp2
        zgemm("N", "N", &r, &one, &r,
              &alph, &T_mat[0, 0], &ldt, &alpha[0, 0], &lda,
              &beta, &tmp2[0, 0], &ldt2)
        for ii in range(r):
            alpha[ii, 0] = K[ii, 0] * v_mat + tmp2[ii, 0]

    return v, F, loglikelihood


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def kalman_loglike_double(double[:] y, unsigned int k, unsigned int p,
                          unsigned int q, int r, unsigned int nobs,
                          double[::1, :] Z_mat,
                          double[::1, :] R_mat,
                          double[::1, :] T_mat):
    """
    Cython version of the Kalman filter recursions for an ARMA process.
    """
    v, F, loglikelihood = kalman_filter_double(y, k, p, q, r, nobs,
                                               Z_mat, R_mat, T_mat)
    sigma2 = 1. / nobs * sum(v**2 / F)
    loglike = -.5 * (loglikelihood + nobs * log(sigma2))
    loglike -= nobs / 2. * (log(2 * pi) + 1)
    return loglike, sigma2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def kalman_loglike_complex(dcomplex[:] y, unsigned int k, unsigned int p,
                           unsigned int q, int r, unsigned int nobs,
                           dcomplex[::1, :] Z_mat,
                           dcomplex[::1, :] R_mat,
                           dcomplex[::1, :] T_mat):
    """
    Cython version of the Kalman filter recursions for an ARMA process.
    """
    v, F, loglikelihood = kalman_filter_complex(y, k, p, q, r, nobs,
                                                Z_mat, R_mat, T_mat)
    sigma2 = 1. / nobs * sum(v**2 / F)
    loglike = -.5 * (loglikelihood + nobs * nplog(sigma2))
    loglike -= nobs / 2. * (log(2 * pi) + 1)
    return loglike, sigma2
