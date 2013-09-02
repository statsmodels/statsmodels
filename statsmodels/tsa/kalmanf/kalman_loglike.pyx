from numpy cimport float64_t, ndarray, complex128_t, complex64_t
from numpy import log as nplog
from numpy import identity, dot, kron, pi, eye, sum, zeros, zeros_like, ones, empty, empty_like, asarray, complex128
from numpy.linalg import pinv
cimport cython
cimport numpy as cnp
from blas cimport dgemm, zgemm

cnp.import_array()

ctypedef float64_t DOUBLE
ctypedef complex128_t dcomplex
ctypedef complex64_t COMPLEX64

cdef extern from "math.h":
    double log(double x)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def kalman_filter_double(double[:] y not None,
                   unsigned int k, unsigned int p, unsigned int q,
                  unsigned int r, unsigned int nobs,
                  double[:,:] Z_mat,
                  double[:,:] R_mat,
                  double[:,:] T_mat):
    """
    Cython version of the Kalman filter recursions for an ARMA process.
    """
    cdef int m = Z_mat.shape[1]
    # store forecast-errors
    #cdef double[:,:] v = zeros((nobs,1))
    cdef ndarray[DOUBLE, ndim=2] v = zeros((nobs, 1))
    # store variance of forecast errors
    cdef double[:,:] F = ones((nobs,1))
    #cdef double[:,:] loglikelihood = zeros((1,1))
    cdef double loglikelihood = 0
    cdef int i = 0
    # initial state
    cdef double[:,:] alpha = zeros((m,1))
    # initial variance
    cdef double[:,:] P = dot(pinv(identity(m**2)-kron(T_mat, T_mat)),dot(R_mat,
            R_mat.T).ravel('C')).reshape(r,r, order='C')
    cdef double[:,:] F_mat = zeros((1,1))
    cdef double[:,:] Finv = zeros((1,1))
    #cdef double[:,:] v_mat = zeros((1,1))
    #cdef ndarray[DOUBLE, ndim=2] v_mat = cnp.PyArray_Zeros(2, [1,1], cnp.NPY_FLOAT64,
    #                                                       0)
    cdef ndarray[DOUBLE, ndim=2] v_mat = zeros((1,1))
    cdef double[:,:] K = zeros((r,1))

    # pre-allocate some tmp arrays for the dgemm calls
    cdef double[:,:] tmp1 = zeros((1, r)) # Z_mat rows x P cols
    cdef double[:,:] tmp2 = zeros((r, r)) # T_mat rows x P cols
    cdef double[:,:] tmp3 = zeros((r, 1)) # T_mat rows x 1
    cdef double[:,:] tmp4 = zeros_like(alpha) # K rows x v_mat cols
    cdef double[:,:] L = zeros_like(T_mat)
    cdef double[:,:] tmp5 = zeros((r, r))
    cdef double[:,:] tmp6 = zeros((r, r)) # T_mat rows x P cols

    #NOTE: not sure about just checking F_mat[0,0], didn't appear to work
    while not F_mat[0,0] == 1. and i < nobs:
        #print i
        # Predict
        #dgemm(101, 111, 111, 1, 1, m, 1.0,
        #      &Z_mat[0,0], Z_mat.strides[0] / sizeof(DOUBLE),
        #      &alpha[0,0], alpha.strides[0] / sizeof(DOUBLE),
        #      0.0, &v_mat[0,0], v_mat.strides[0] / sizeof(DOUBLE))
        dgemm(101, 111, 111, 1, 1, m, 1.0,
              &Z_mat[0,0], Z_mat.strides[0] / sizeof(DOUBLE),
              &alpha[0,0], alpha.strides[0] / sizeof(DOUBLE),
              0.0, <DOUBLE *>v_mat.data, v_mat.strides[0] / sizeof(DOUBLE))

        v_mat[0,0] = y[i] - v_mat[0,0] # copies?
        #print asarray(<DOUBLE[:1,:1] *> &v_mat[0,0])
        #print "v_mat: ", asarray(<DOUBLE[:1, :1] *> &v_mat[0,0])

        # one-step forecast error
        # colon should make a copy
        v[i, 0] = v_mat[0,0]
        #dcopy(1, <DOUBLE *>v_mat.data, 1, &v[i,0], 1)
        #if i > 1:
        #    print asarray(<DOUBLE[:1,:1] *> &v[i-1,0])
        #print asarray(<DOUBLE[:1,:1] *> &v_mat[0,0])
        dgemm(101, 111, 111, 1, r, m, 1.0,
                &Z_mat[0,0], Z_mat.strides[0] / sizeof(DOUBLE),
                &P[0,0], P.strides[0] / sizeof(DOUBLE),
                0.0, &tmp1[0,0], tmp1.strides[0] / sizeof(DOUBLE))
        dgemm(101, 111, 112, 1, 1, r, 1.0,
              &tmp1[0,0], tmp1.strides[0] / sizeof(DOUBLE),
              &Z_mat[0,0], Z_mat.strides[0] / sizeof(DOUBLE),
              0.0,
              &F_mat[0,0], F_mat.strides[0] / sizeof(DOUBLE))
        F[i,0] = F_mat[0,0]
        Finv[0,0] = 1./F_mat[0,0] # always scalar for univariate series
        # compute Kalman Gain, K
        # K = dot(dot(dot(T_mat,P),Z_mat.T),Finv)
        # tmp2 = dot(T_mat, P)
        # tmp3 = dot(tmp2, Z_mat.T)
        # K = dot(tmp3, Finv)

        #print "Finv: ", asarray(<DOUBLE[:1, :1] *> &Finv[0,0])
        dgemm(101, 111, 111, r, r, r, 1.0,
                &T_mat[0,0], T_mat.strides[0] / sizeof(DOUBLE),
                &P[0,0], P.strides[0] / sizeof(DOUBLE), 0.0,
                &tmp2[0,0], tmp2.strides[0] / sizeof(DOUBLE))
        #print "tmp2: ", asarray(<DOUBLE[:r, :r] *> &tmp2[0,0])
        dgemm(101, 111, 112, r, 1, r, 1.0, &tmp2[0,0],
              tmp2.strides[0] / sizeof(DOUBLE),
              &Z_mat[0,0], Z_mat.strides[0] / sizeof(DOUBLE), 0.0,
              &tmp3[0,0], tmp3.strides[0] / sizeof(DOUBLE))

        #print "tmp3: ", asarray(<DOUBLE[:r, :1] *> &tmp3[0,0])
        dgemm(101, 111, 111, r, 1, 1, 1.0, &tmp3[0,0],
              tmp3.strides[0] / sizeof(DOUBLE), &Finv[0,0],
              Finv.strides[0] / sizeof(DOUBLE),
              0.0, &K[0,0], K.strides[0] / sizeof(DOUBLE))
        #print "K: ", asarray(<DOUBLE[:r, :1] *> &K[0,0])

        # update state
        #alpha = dot(T_mat, alpha) + dot(K, v_mat)
        #dot(T_mat, alpha)
        #NOTE: might be able to do away with tmp4 and use alpha - don't know
        # if you can update it in place while doing multiplication
        #print "alpha: ", asarray(<DOUBLE[:m, :1] *> &alpha[0,0])
        #print "T_mat: ", asarray(<DOUBLE[:m, :m] *> &T_mat[0,0])
        dgemm(101, 111, 111, r, 1, r, 1.0, &T_mat[0,0],
                T_mat.strides[0] / sizeof(DOUBLE),
                &alpha[0,0], alpha.strides[0] / sizeof(DOUBLE),
              0.0, &tmp4[0,0], tmp4.strides[0] / sizeof(DOUBLE))
        #print "tmp4: ", asarray(<DOUBLE[:m, :1] *> &tmp4[0,0])

        #dot(K, v_mat) + tmp4
        dgemm(101, 111, 111, r, 1, 1, 1.0, &K[0,0],
                K.strides[0] / sizeof(DOUBLE), &v_mat[0,0],
                v_mat.strides[0] / sizeof(DOUBLE), 0.0,
                &alpha[0,0], alpha.strides[0] / sizeof(DOUBLE))
        #print "aalpha: ", asarray(<DOUBLE[:m, :1] *> &alpha[0,0])
        # alpha += tmp4
        #daxpy(r, 1.0, &tmp4[0,0], 1, &alpha[0,0], 1)
        for ii in range(m):
            alpha[ii,0] = alpha[ii,0] + tmp4[ii,0]
        #print "alpha: ", asarray(<DOUBLE[:m, :1] *> &alpha[0,0])

        #L = T_mat - dot(K,Z_mat)
        dgemm(101, 111, 111, r, r, 1, 1.0,
                &K[0,0], K.strides[0] / sizeof(DOUBLE),
                &Z_mat[0,0], Z_mat.strides[0] / sizeof(DOUBLE), 0.0, &L[0,0],
                L.strides[0] / sizeof(DOUBLE))

        # L = T_mat - L
        # L = -(L - T_mat)
        #L = asarray(<DOUBLE[:r,:r] *> &T_mat[0,0]) - asarray(<DOUBLE[:r,:r] *> &L[0,0])
        for jj in range(r):
            for kk in range(r):
                L[jj,kk] = T_mat[jj,kk] - L[jj,kk]
        #daxpy(r, -1.0, &T_mat[0,0], 1, &L[0,0], 1)
        #dscal(r, -1.0, &L[0,0], 1)
        #print "L: ", asarray(<DOUBLE[:4,:4] *> &L[0,0])

        #P = dot(dot(T_mat, P), L.T) + dot(R_mat, R_mat.T)
        # tmp5 = dot(R_mat, R_mat.T)
        # tmp6 = dot(T_mat, P)
        # P = dot(tmp6, L.T) + tmp5
        dgemm(101, 111, 112, r, r, 1, 1.0, &R_mat[0,0],
              R_mat.strides[0] / sizeof(DOUBLE),
              &R_mat[0,0], R_mat.strides[0] / sizeof(DOUBLE), 0.0,
              &tmp5[0,0], tmp5.strides[0] / sizeof(DOUBLE))
        dgemm(101, 111, 111, r, r, r, 1.0, &T_mat[0,0],
                T_mat.strides[0] / sizeof(DOUBLE), &P[0,0],
                P.strides[0] / sizeof(DOUBLE),
              0.0, &tmp6[0,0], tmp6.strides[0] / sizeof(DOUBLE))
        #print "tmp6: ", asarray(<DOUBLE[:r,:r] *> &tmp6[0,0])

        dgemm(101, 111, 112, r, r, r, 1.0, &tmp6[0,0],
                tmp6.strides[0] / sizeof(DOUBLE), &L[0,0],
                L.strides[0] / sizeof(DOUBLE), 0.0,
              &P[0,0], P.strides[0] / sizeof(DOUBLE))
        #print "PP: ", asarray(<DOUBLE[:r,:r] *> &P[0,0])

        # 101 = c-order, 122 - lower triangular of R, 111 - no trans (XX')
        #dsyrk(101, 122, 111, r, 1, 1.0, &R_mat[0,0],
        #      R_mat.strides[0] / sizeof(DOUBLE), 1.0, &P[0,0],
        #      P.strides[0] / sizeof(DOUBLE) )
        #daxpy(r, 1.0, &tmp5[0,0], 1, &P[0,0], 1)

        #NOTE: can probably replace this once we get proper support for
        # symmetric P
        #P = asarray(<DOUBLE[:r,:r] *> &P[0,0]) + asarray(<DOUBLE[:r,:r] *> &tmp5[0,0])
        for jj in range(r):
            for kk in range(r):
                P[jj, kk] = P[jj, kk] + tmp5[jj, kk]

        #print "P: ", asarray(<DOUBLE[:r,:r] *> &P[0,0])
        loglikelihood += log(F_mat[0,0])
        #print loglikelihood
        #print
        i+=1

    for i in xrange(i,nobs):
        dgemm(101, 111, 111, 1, 1, m, 1.0,
              &Z_mat[0,0], Z_mat.strides[0] / sizeof(DOUBLE),
              &alpha[0,0], alpha.strides[0] / sizeof(DOUBLE),
              0.0, <DOUBLE *>v_mat.data, v_mat.strides[0] / sizeof(DOUBLE))

        v_mat[0,0] = y[i] - v_mat[0,0]
        #print asarray(<DOUBLE[:1,:1] *> &v_mat[0,0])
        # colon should induce a copy?
        v[i, 0] = v_mat[0,0]
        #alpha = dot(T_mat, alpha) + dot(K, v_mat)
        dgemm(101, 111, 111, r, 1, r, 1.0, &T_mat[0,0],
                T_mat.strides[0] / sizeof(DOUBLE),
                &alpha[0,0], alpha.strides[0] / sizeof(DOUBLE),
              0.0, &tmp4[0,0], tmp4.strides[0] / sizeof(DOUBLE))
        #dot(K, v_mat) + tmp4
        dgemm(101, 111, 111, r, 1, 1, 1.0, &K[0,0],
                K.strides[0] / sizeof(DOUBLE), &v_mat[0,0],
                v_mat.strides[0] / sizeof(DOUBLE), 0.0,
                &alpha[0,0], alpha.strides[0] / sizeof(DOUBLE))
        # alpha += tmp4
        #daxpy(r, 1.0, &tmp4[0,0], 1, &alpha[0,0], 1)
        #alpha = asarray(<DOUBLE[:m,:1] *> &alpha[0,0]) + asarray(<DOUBLE[:m,:1] *> &tmp4[0,0])
        for ii in range(m):
            alpha[ii,0] = alpha[ii,0] + tmp4[ii,0]
    #return asarray(<DOUBLE[:nobs, :1] *> &v[0,0]), F, loglikelihood
    return v, F, loglikelihood

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def kalman_filter_complex(dcomplex[:] y,
                   unsigned int k, unsigned int p, unsigned int q,
                  unsigned int r, unsigned int nobs,
                  dcomplex[:,:] Z_mat,
                  dcomplex[:,:] R_mat,
                  dcomplex[:,:] T_mat):
    """
    Cython version of the Kalman filter recursions for an ARMA process.
    """
    cdef int m = Z_mat.shape[1]
    # store forecast-errors
    #cdef dcomplex[:,:] v = zeros((nobs,1), dtype=complex)
    cdef ndarray[complex, ndim=2] v = zeros((nobs, 1), dtype=complex)
    # store variance of forecast errors
    cdef dcomplex[:,:] F = ones((nobs,1), dtype=complex)
    #cdef double[:,:] loglikelihood = zeros((1,1))
    cdef dcomplex loglikelihood = 0 + 0j
    cdef int i = 0
    # initial state
    cdef dcomplex[:,:] alpha = zeros((m,1), dtype=complex)
    # initial variance
    cdef dcomplex[:,:] P = dot(pinv(identity(m**2)-kron(T_mat, T_mat)),
                        dot(R_mat,R_mat.T).ravel('C')).reshape(r,r, order='C')
    cdef dcomplex[:,:] F_mat = zeros((1,1), dtype=complex)
    cdef dcomplex[:,:] Finv = zeros((1,1), dtype=complex)
    #cdef dcomplex[:,:] v_mat = zeros((1,1), dtype=complex)
    cdef ndarray[complex, ndim=2] v_mat = zeros((1,1), dtype=complex)
    cdef dcomplex[:,:] K = zeros((r,1), dtype=complex)

    # pre-allocate some tmp arrays for the dgemm calls
    cdef dcomplex[:,:] tmp1 = zeros((1, r), dtype=complex)
    cdef dcomplex[:,:] tmp2 = zeros((r, r), dtype=complex)
    cdef dcomplex[:,:] tmp3 = zeros((r, 1), dtype=complex)
    cdef dcomplex[:,:] tmp4 = zeros_like(alpha)
    cdef dcomplex[:,:] L = zeros_like(T_mat, dtype=complex)
    cdef dcomplex[:,:] tmp5 = zeros((r, r), dtype=complex)
    cdef dcomplex[:,:] tmp6 = zeros((r, r), dtype=complex) # T_mat rows x P cols

    cdef double *alph = [1.0, 0.0]
    cdef double *beta = [0.0, 0.0]

    while not F_mat[0,0] == 1 and i < nobs:
        #print
        #print i
        #print "Z_mat: ", asarray(<dcomplex[:1, :m] *> &Z_mat[0,0])
        #print "alpha: ", asarray(<dcomplex[:m, :1] *> &alpha[0,0])
        #print "v_mat: ", asarray(<dcomplex[:1, :1] *> &v_mat[0,0])
        zgemm(101, 111, 111, Z_mat.shape[0], alpha.shape[1], Z_mat.shape[1],
                alph,
              &Z_mat[0,0], Z_mat.strides[0] / sizeof(dcomplex),
              &alpha[0,0], alpha.strides[0] / sizeof(dcomplex),
              beta, <dcomplex *>v_mat.data,
              v_mat.strides[0] / sizeof(dcomplex))
        #print "v_mat: ", asarray(<dcomplex[:1, :1] *> &v_mat[0,0])

        v_mat[0,0] = y[i] - v_mat[0,0] # copies?
        #print asarray(<dcomplex[:1,:1] *> &v_mat[0,0])
        #print "v_mat: ", asarray(<dcomplex[:1, :1] *> &v_mat[0,0])

        # one-step forecast error
        # colon should make a copy
        v[i, 0] = v_mat[0,0]
        #print "v_mat: ", asarray(<dcomplex[:1, :1] *> &v_mat[0,0])
        zgemm(101, 111, 111, 1, r, m, alph,
                &Z_mat[0,0], Z_mat.strides[0] / sizeof(dcomplex),
                &P[0,0], P.strides[0] / sizeof(dcomplex),
                beta, &tmp1[0,0], tmp1.strides[0] / sizeof(dcomplex))
        zgemm(101, 111, 112, 1, 1, r, alph,
              &tmp1[0,0], tmp1.strides[0] / sizeof(dcomplex),
              &Z_mat[0,0], Z_mat.strides[0] / sizeof(dcomplex),
              beta,
              &F_mat[0,0], F_mat.strides[0] / sizeof(dcomplex))
        #print "F_mat: ", asarray(<dcomplex[:1,:1] *> &F_mat[0,0])
        F[i,0] = F_mat[0,0]
        Finv[0,0] = 1./F_mat[0,0] # always scalar for univariate series
        # compute Kalman Gain, K
        # K = dot(dot(dot(T_mat,P),Z_mat.T),Finv)
        # tmp2 = dot(T_mat, P)
        # tmp3 = dot(tmp2, Z_mat.T)
        # K = dot(tmp3, Finv)

        #print "Finv: ", asarray(<dcomplex[:1, :1] *> &Finv[0,0])
        zgemm(101, 111, 111, r, r, r, alph,
                &T_mat[0,0], T_mat.strides[0] / sizeof(dcomplex),
                &P[0,0], P.strides[0] / sizeof(dcomplex), beta,
                &tmp2[0,0], tmp2.strides[0] / sizeof(dcomplex))
        #print "tmp2: ", asarray(<dcomplex[:r, :r] *> &tmp2[0,0])
        zgemm(101, 111, 112, r, 1, r, alph, &tmp2[0,0],
              tmp2.strides[0] / sizeof(dcomplex),
              &Z_mat[0,0], Z_mat.strides[0] / sizeof(dcomplex), beta,
              &tmp3[0,0], tmp3.strides[0] / sizeof(dcomplex))

        #print "tmp3: ", asarray(<dcomplex[:r, :1] *> &tmp3[0,0])
        zgemm(101, 111, 111, r, 1, 1, alph, &tmp3[0,0],
              tmp3.strides[0] / sizeof(dcomplex), &Finv[0,0],
              Finv.strides[0] / sizeof(dcomplex),
              beta, &K[0,0], K.strides[0] / sizeof(dcomplex))
        #print "K: ", asarray(<dcomplex[:r, :1] *> &K[0,0])

        # update state
        #alpha = dot(T_mat, alpha) + dot(K, v_mat)
        #dot(T_mat, alpha)
        #NOTE: might be able to do away with tmp4 and use alpha - don't know
        # if you can update it in place while doing multiplication
        #print "alpha: ", asarray(<dcomplex[:m, :1] *> &alpha[0,0])
        #print "T_mat: ", asarray(<dcomplex[:m, :m] *> &T_mat[0,0])
        zgemm(101, 111, 111, r, 1, r, alph, &T_mat[0,0],
                T_mat.strides[0] / sizeof(dcomplex),
                &alpha[0,0], alpha.strides[0] / sizeof(dcomplex),
              beta, &tmp4[0,0], tmp4.strides[0] / sizeof(dcomplex))
        #print "tmp4: ", asarray(<dcomplex[:m, :1] *> &tmp4[0,0])

        #dot(K, v_mat) + tmp4
        zgemm(101, 111, 111, r, 1, 1, alph, &K[0,0],
                K.strides[0] / sizeof(dcomplex), &v_mat[0,0],
                v_mat.strides[0] / sizeof(dcomplex), beta,
                &alpha[0,0], alpha.strides[0] / sizeof(dcomplex))
        #print "aalpha: ", asarray(<dcomplex[:m, :1] *> &alpha[0,0])
        # alpha += tmp4
        #daxpy(r, alph, &tmp4[0,0], 1, &alpha[0,0], 1)
        for ii in range(m):
            alpha[ii,0] = alpha[ii,0] + tmp4[ii,0]

        #L = T_mat - dot(K,Z_mat)
        zgemm(101, 111, 111, r, r, 1, alph,
                &K[0,0], K.strides[0] / sizeof(dcomplex),
                &Z_mat[0,0], Z_mat.strides[0] / sizeof(dcomplex), beta, &L[0,0],
                L.strides[0] / sizeof(dcomplex))

        # L = T_mat - L
        # L = -(L - T_mat)
        #L = asarray(<dcomplex[:r,:r] *> &T_mat[0,0]) - asarray(<dcomplex[:r,:r] *> &L[0,0])
        for jj in range(r):
            for kk in range(r):
                L[jj, kk] = T_mat[jj,kk] - L[jj,kk]
        #daxpy(r, -alph, &T_mat[0,0], 1, &L[0,0], 1)
        #dscal(r, -alph, &L[0,0], 1)
        #print "L: ", asarray(<dcomplex[:4,:4] *> &L[0,0])

        #P = dot(dot(T_mat, P), L.T) + dot(R_mat, R_mat.T)
        # tmp5 = dot(R_mat, R_mat.T)
        # tmp6 = dot(T_mat, P)
        # P = dot(tmp6, L.T) + tmp5
        zgemm(101, 111, 112, r, r, 1, alph, &R_mat[0,0],
              R_mat.strides[0] / sizeof(dcomplex),
              &R_mat[0,0], R_mat.strides[0] / sizeof(dcomplex), beta,
              &tmp5[0,0], tmp5.strides[0] / sizeof(dcomplex))
        zgemm(101, 111, 111, r, r, r, alph, &T_mat[0,0],
                T_mat.strides[0] / sizeof(dcomplex), &P[0,0],
                P.strides[0] / sizeof(dcomplex),
              beta, &tmp6[0,0], tmp6.strides[0] / sizeof(dcomplex))
        #print "tmp6: ", asarray(<dcomplex[:r,:r] *> &tmp6[0,0])

        zgemm(101, 111, 112, r, r, r, alph, &tmp6[0,0],
                tmp6.strides[0] / sizeof(dcomplex), &L[0,0],
                L.strides[0] / sizeof(dcomplex), beta,
              &P[0,0], P.strides[0] / sizeof(dcomplex))
        #print "PP: ", asarray(<dcomplex[:r,:r] *> &P[0,0])

        # 101 = c-order, 122 - lower triangular of R, 111 - no trans (XX')
        #dsyrk(101, 122, 111, r, 1, alph, &R_mat[0,0],
        #      R_mat.strides[0] / sizeof(dcomplex), alph, &P[0,0],
        #      P.strides[0] / sizeof(dcomplex) )
        #daxpy(r, alph, &tmp5[0,0], 1, &P[0,0], 1)

        #NOTE: can probably replace this once we get proper support for
        # symmetric P
        #P = asarray(<dcomplex[:r,:r] *> &P[0,0]) + asarray(<dcomplex[:r,:r] *> &tmp5[0,0])
        for jj in range(r):
            for kk in range(r):
                P[jj,kk] = P[jj, kk] + tmp5[jj, kk]


        loglikelihood += nplog(F_mat[0,0])
        i+=1

    for i in xrange(i,nobs):
        zgemm(101, 111, 111, 1, 1, m, alph,
              &Z_mat[0,0], Z_mat.strides[0] / sizeof(dcomplex),
              &alpha[0,0], alpha.strides[0] / sizeof(dcomplex),
              beta, <dcomplex *>v_mat.data,
              v_mat.strides[0] / sizeof(dcomplex))

        v_mat[0,0] = y[i] - v_mat[0,0]
        #print asarray(<dcomplex[:1,:1] *> &v_mat[0,0])
        # colon should induce a copy?
        v[i, 0] = v_mat[0,0]
        #alpha = dot(T_mat, alpha) + dot(K, v_mat)
        zgemm(101, 111, 111, r, 1, r, alph, &T_mat[0,0],
                T_mat.strides[0] / sizeof(dcomplex),
                &alpha[0,0], alpha.strides[0] / sizeof(dcomplex),
              beta, &tmp4[0,0], tmp4.strides[0] / sizeof(dcomplex))
        #dot(K, v_mat) + tmp4
        zgemm(101, 111, 111, r, 1, 1, alph, &K[0,0],
                K.strides[0] / sizeof(dcomplex), &v_mat[0,0],
                v_mat.strides[0] / sizeof(dcomplex), beta,
                &alpha[0,0], alpha.strides[0] / sizeof(dcomplex))
        # alpha += tmp4
        #daxpy(r, alph, &tmp4[0,0], 1, &alpha[0,0], 1)
        #alpha = asarray(<dcomplex[:m,:1] *> &alpha[0,0]) + asarray(<dcomplex[:m,:1] *> &tmp4[0,0])
        for ii in range(m):
            alpha[ii,0] = alpha[ii,0] + tmp4[ii,0]
    return v, F, loglikelihood

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def kalman_loglike_double(double[:] y,
                   unsigned int k, unsigned int p, unsigned int q,
                  unsigned int r, unsigned int nobs,
                  double[:,:] Z_mat,
                  double[:,:] R_mat,
                  double[:,:] T_mat):
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
def kalman_loglike_complex(dcomplex[:] y,
                   unsigned int k, unsigned int p, unsigned int q,
                  unsigned int r, unsigned int nobs,
                  dcomplex[:,:] Z_mat,
                  dcomplex[:,:] R_mat,
                  dcomplex[:,:] T_mat):
    """
    Cython version of the Kalman filter recursions for an ARMA process.
    """
    v,F,loglikelihood = kalman_filter_complex(y,k,p,q,r,nobs,Z_mat,R_mat,T_mat)
    sigma2 = 1./nobs * sum(v**2 / F)
    loglike = -.5 *(loglikelihood + nobs*nplog(sigma2))
    loglike -= nobs/2. * (log(2*pi) + 1)
    return loglike, sigma2
