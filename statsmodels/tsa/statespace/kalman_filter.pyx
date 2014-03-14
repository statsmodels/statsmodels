"""
Multivariate Kalman Filter (Cython and BLAS/LAPACK)

Author: Chad Fulton
License: Simplified-BSD

"""

import numpy as np
cimport numpy as np
cimport cython

from cpython cimport PyCObject_AsVoidPtr
import scipy
__import__('scipy.linalg.blas')
__import__('scipy.linalg.lapack')

from libc.math cimport log as dlog, abs as dabs
cdef extern from "complex.h":
    complex clog(complex x)
    np.complex64_t clogf(np.complex64_t x)
    complex cabs(complex x)
    np.complex64_t cabsf(np.complex64_t x)

from blas_lapack cimport *

#cdef ssymm_t *ssymm = <ssymm_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.ssymm._cpointer)
cdef sgemm_t *sgemm = <sgemm_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.sgemm._cpointer)
cdef sgemv_t *sgemv = <sgemv_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.sgemv._cpointer)
cdef scopy_t *scopy = <scopy_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.scopy._cpointer)
cdef saxpy_t *saxpy = <saxpy_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.saxpy._cpointer)
cdef sdot_t *sdot = <sdot_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.sdot._cpointer)
cdef sgetrf_t *sgetrf = <sgetrf_t*>PyCObject_AsVoidPtr(scipy.linalg.lapack.sgetrf._cpointer)
cdef sgetri_t *sgetri = <sgetri_t*>PyCObject_AsVoidPtr(scipy.linalg.lapack.sgetri._cpointer)
cdef spotrf_t *spotrf = <spotrf_t*>PyCObject_AsVoidPtr(scipy.linalg.lapack.spotrf._cpointer)

#cdef dsymm_t *dsymm = <dsymm_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.dsymm._cpointer)
cdef dgemm_t *dgemm = <dgemm_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.dgemm._cpointer)
cdef dgemv_t *dgemv = <dgemv_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.dgemv._cpointer)
cdef dcopy_t *dcopy = <dcopy_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.dcopy._cpointer)
cdef daxpy_t *daxpy = <daxpy_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.daxpy._cpointer)
cdef ddot_t *ddot = <ddot_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.ddot._cpointer)
cdef dgetrf_t *dgetrf = <dgetrf_t*>PyCObject_AsVoidPtr(scipy.linalg.lapack.dgetrf._cpointer)
cdef dgetri_t *dgetri = <dgetri_t*>PyCObject_AsVoidPtr(scipy.linalg.lapack.dgetri._cpointer)
cdef dpotrf_t *dpotrf = <dpotrf_t*>PyCObject_AsVoidPtr(scipy.linalg.lapack.dpotrf._cpointer)

#cdef csymm_t *csymm = <csymm_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.csymm._cpointer)
cdef cgemm_t *cgemm = <cgemm_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.cgemm._cpointer)
cdef cgemv_t *cgemv = <cgemv_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.cgemv._cpointer)
cdef ccopy_t *ccopy = <ccopy_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.ccopy._cpointer)
cdef caxpy_t *caxpy = <caxpy_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.caxpy._cpointer)
cdef cgetrf_t *cgetrf = <cgetrf_t*>PyCObject_AsVoidPtr(scipy.linalg.lapack.cgetrf._cpointer)
cdef cgetri_t *cgetri = <cgetri_t*>PyCObject_AsVoidPtr(scipy.linalg.lapack.cgetri._cpointer)
cdef cpotrf_t *cpotrf = <cpotrf_t*>PyCObject_AsVoidPtr(scipy.linalg.lapack.cpotrf._cpointer)

#cdef zsymm_t *zsymm = <zsymm_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.zsymm._cpointer)
cdef zgemm_t *zgemm = <zgemm_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.zgemm._cpointer)
cdef zgemv_t *zgemv = <zgemv_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.zgemv._cpointer)
cdef zcopy_t *zcopy = <zcopy_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.zcopy._cpointer)
cdef zaxpy_t *zaxpy = <zaxpy_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.zaxpy._cpointer)
cdef zgetrf_t *zgetrf = <zgetrf_t*>PyCObject_AsVoidPtr(scipy.linalg.lapack.zgetrf._cpointer)
cdef zgetri_t *zgetri = <zgetri_t*>PyCObject_AsVoidPtr(scipy.linalg.lapack.zgetri._cpointer)
cdef zpotrf_t *zpotrf = <zpotrf_t*>PyCObject_AsVoidPtr(scipy.linalg.lapack.zpotrf._cpointer)

# Kalman Filter: Single Precision
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef skalman_filter(np.float32_t [::1,:]   y,  # nxT+1    (data: endogenous, observed)
                     np.float32_t [::1,:,:] H,  # nxkxT+1  (parameters)
                     np.float32_t [:]       mu, # kx0      (parameters)
                     np.float32_t [::1,:]   F,  # kxk      (parameters)
                     np.float32_t [::1,:]   R,  # nxn      (parameters: covariance matrix)
                     np.float32_t [::1,:]   G,  # kxg      (parameters)
                     np.float32_t [::1,:]   Q_star,  # gxg      (parameters: covariance matrix)
                     np.float32_t [::1,:]   z=None,  # rxT+1    (data: weakly exogenous, observed)
                     np.float32_t [::1,:]   A=None,  # nxr      (parameters)
                     np.float32_t [:]       beta_tt_init=None,
                     np.float32_t [::1,:]   P_tt_init=None):

    cdef np.float32_t [::1,:,:] P_tt, P_tt1, f_tt1, gain, f_inv
    cdef int [::1,:] ipiv
    cdef np.float32_t [::1,:] beta_tt, beta_tt1, y_tt1, eta_tt1, tmp, work, PHT, Q
    cdef double [:] ll
    cdef np.float32_t det, tol = 10e-20
    cdef:
        int i
        int t
        int T = y.shape[1]
        int n = y.shape[0]
        int r = 0
        int g = Q_star.shape[0]
        int k = mu.shape[0]
        int time_varying_H = H.shape[2] == T
        int H_idx = 0
        int converged = 0
    cdef:
        int kn = k*n
        int k2 = k**2
        int n2 = n**2
        int info # return code fro sgetri, sgetrf
        int inc = 1 # incrementer for sgemv
        int ldwork = max(k2, n2) # number of rows/columns in work array
        int lwork = ldwork**2 # size of work array for sgetri
    cdef:
        np.float32_t alpha = 1.0 # first scalar multiple on sgemv, sgemm
        np.float32_t beta = 0.0 # second scalar multiple on sgemv, sgemm
        np.float32_t gamma = -1.0 # scalar multiple for saxpy
        np.float32_t delta = -0.5 # scalar multiple for log calculation

    # Check if we have an exog matrix
    if z is not None and A is not None:
        r = z.shape[0]

    # Allocate memory for variables
    beta_tt = np.zeros((k,T+1), np.float32, order="F")    # T+1xk
    P_tt = np.zeros((k,k,T+1), np.float32, order="F")     # T+1xkxk
    beta_tt1 = np.zeros((k,T+1), np.float32, order="F")   # T+1xk
    P_tt1 = np.zeros((k,k,T+1), np.float32, order="F")    # T+1xkxk
    y_tt1 = np.zeros((n,T+1), np.float32, order="F")      # T+1xn
    eta_tt1 = np.zeros((n,T+1), np.float32, order="F")    # T+1xn
    f_tt1 = np.zeros((n,n,T+1), np.float32, order="F")    # T+1xnxn
    gain = np.zeros((k,n,T+1), np.float32, order="F")     # T+1xkxn
    ll = np.zeros((T+1,), float)                          # T+1
    work = np.zeros((ldwork,ldwork), np.float32, order="F")
    ipiv = np.empty((ldwork,ldwork), np.int32, order="F")
    PHT = np.empty((k,n), np.float32, order="F")
    f_inv = np.empty((n,n,T+1), np.float32, order="F")
    Q = np.zeros((k,k), np.float32, order="F")
    
    # Get Q = G Q^* G'
    sgemm("N", "N", &k, &g, &g, &alpha, &G[0,0], &k, &Q_star[0,0], &g, &beta, &work[0,0], &ldwork)
    sgemm("N", "T", &k, &g, &g, &alpha, &work[0,0], &ldwork, &G[0,0], &k, &beta, &Q[0,0], &k)

    # Initial values
    if beta_tt_init is None:
        #beta_tt[:,0] = np.linalg.inv(np.eye(k) - F).dot(mu) # kxk * kx1 = kx1
        beta_tt_init = np.zeros((k,), np.float32, order="F")
        tmp = np.array(np.eye(k), np.float32, order="F") - F
        sgetrf(&k, &k, &tmp[0,0], &k, &ipiv[0,0], &info)
        sgetri(&k, &tmp[0,0], &k, &ipiv[0,0], &work[0,0], &lwork, &info)
        sgemv("N",&k,&k,&alpha,&tmp[0,0],&k,&mu[0],&inc,&beta,&beta_tt_init[0],&inc)
    beta_tt[::1,0] = beta_tt_init[::1]

    if P_tt_init is None:
        #P_tt[0] = np.linalg.inv(np.eye(k**2) - np.kron(F,F)).dot(Q.reshape(Q.size, 1)).reshape(k,k) # kxk
        P_tt_init = np.zeros((k,k), np.float32, order="F")
        tmp = np.array(np.eye(k2) - np.kron(F,F), np.float32, order="F")
        sgetrf(&k2, &k2, &tmp[0,0], &k2, &ipiv[0,0], &info)
        sgetri(&k2, &tmp[0,0], &k2, &ipiv[0,0], &work[0,0], &lwork, &info)
        sgemv("N",&k2,&k2,&alpha,&tmp[0,0],&k2,&Q[0,0],&inc,&beta,&P_tt_init[0,0],&inc)
    P_tt[::1,:,0] = P_tt_init[::1,:]

    # Redefine the tmp array
    tmp = np.empty((ldwork,ldwork), np.float32, order="F")

    # Iterate forwards
    for t in range(1,T+1):
        if time_varying_H:
            H_idx = t-1

        # Prediction
        #beta_tt1[t] = mu + np.dot(F, beta_tt[t-1])
        #beta_tt1[::1,t] = mu[::1]
        scopy(&k, &mu[0], &inc, &beta_tt1[0,t], &inc)
        sgemv("N",&k,&k,&alpha,&F[0,0],&k,&beta_tt[0,t-1],&inc,&alpha,&beta_tt1[0,t],&inc)

        #P_tt1[t] = np.dot(F, P_tt[t-1]).dot(F.T) + Q
        if converged:
            scopy(&k2, &P_tt1[0,0,t-1], &inc, &P_tt1[0,0,t], &inc)
        else:
            #P_tt1[::1,:,t] = Q[::1,:]
            sgemm("N", "N", &k, &g, &g, &alpha, &G[0,0], &k, &Q[0,0], &g, &beta, &tmp[0,0], &ldwork)
            sgemm("N", "T", &k, &g, &g, &alpha, &tmp[0,0], &ldwork, &G[0,0], &k, &beta, &P_tt1[0,0,t], &k)
            #scopy(&k2, &Q[0,0], &inc, &P_tt1[0,0,t], &inc)
            #ssymm("R", "L", &k, &k, &alpha, &F[0,0], &k, &P_tt[0,0,t-1], &k, &beta, &tmp[0,0], &ldwork)
            sgemm("N", "N", &k, &k, &k, &alpha, &F[0,0], &k, &P_tt[0,0,t-1], &k, &beta, &tmp[0,0], &ldwork)
            sgemm("N", "T", &k, &k, &k, &alpha, &tmp[0,0], &ldwork, &F[0,0], &k, &alpha, &P_tt1[0,0,t], &k)

        #y_tt1[t] = np.dot(H[:,:,H_idx], beta_tt1[:,t]) + np.dot(A,z[:,t-1])
        sgemv("N", &n, &k, &alpha, &H[0,0,H_idx], &n, &beta_tt1[0,t], &inc, &beta, &y_tt1[0,t], &inc)
        if r > 0:
            # z[0] corresponds to z[t=1]
            sgemv("N", &n, &r, &alpha, &A[0,0], &n, &z[0,t-1], &inc, &alpha, &y_tt1[0,t], &inc)

        #eta_tt1[::1,t] = y[::1,t-1] - y_tt1[:,t]
        #eta_tt1[::1,t] = y[::1,t-1] # y[0] corresponds to y[t=1]
        scopy(&n, &y[0,t-1], &inc, &eta_tt1[0,t], &inc)
        saxpy(&n, &gamma, &y_tt1[0,t], &inc, &eta_tt1[0,t], &inc)

        if converged:
            scopy(&n2, &f_tt1[0,0,t-1], &inc, &f_tt1[0,0,t], &inc)
            scopy(&n2, &f_inv[0,0,t-1], &inc, &f_inv[0,0,t], &inc)
        else:
            #PHT = np.dot(P_tt1[t], H[:,:,H_idx].T) # kxn
            #print np.dot(P_tt1[:,:,t], H[:,:,H_idx].T) # taking .T here crashes the program for some reason
            sgemm("N", "T", &k, &n, &k, &alpha, &P_tt1[0,0,t], &k, &H[0,0,H_idx], &n, &beta, &PHT[0,0], &k)

            #f_tt1[t] = np.dot(H[:,:,H_idx], PHT) + R
            #f_tt1[::1,:,t] = R[::1,:]
            scopy(&n2, &R[0,0], &inc, &f_tt1[0,0,t], &inc)
            sgemm("N", "N", &n, &n, &k, &alpha, &H[0,0,H_idx], &n, &PHT[0,0], &k, &alpha, &f_tt1[0,0,t], &n)

            #f_inv = np.linalg.inv(f_tt1[t])
            #f_inv[::1,:] = f_tt1[::1,:,t]
            if n == 1:
                det = dabs(f_tt1[0,0,t])
                f_inv[0,0,t] = 1/f_tt1[0,0,t]
            else:
                scopy(&n2, &f_tt1[0,0,t], &inc, &f_inv[0,0,t], &inc)
                sgetrf(&n, &n, &f_inv[0,0,t], &n, &ipiv[0,0], &info)
                det = 1
                for i in range(n):
                    if not ipiv[i,0] == i+1:
                        det *= -1*f_inv[i,i,t]
                    else:
                        det *= f_inv[i,i,t]
                # Now complete taking the inverse
                sgetri(&n, &f_inv[0,0,t], &n, &ipiv[0,0], &work[0,0], &lwork, &info)

        # Log-likelihood as byproduct
        #ll[t] -0.5*log(2*np.pi*np.linalg.det(f_tt1[:,:,t])) - 0.5*np.dot(np.dot(eta_tt1[:,t].T, f_inv), eta_tt1[:,t])
        # ^ this doesn't work, crashes for some reason; probably related to taking .T as it did above
        ll[t] = -0.5*dlog(2*np.pi*det)
        sgemv("N",&n,&n,&alpha,&f_inv[0,0,t],&n,&eta_tt1[0,t],&inc,&beta,&tmp[0,0],&inc)
        ll[t] += -0.5*float(sdot(&n, &eta_tt1[0,t], &inc, &tmp[0,0], &inc))

        # Updating
        #gain[t] = np.dot(PHT, f_inv) # kxn * nxn = kxn
        if converged:
            scopy(&kn, &gain[0,0,t-1], &inc, &gain[0,0,t], &inc)
        else:
            sgemm("N", "N", &k, &n, &n, &alpha, &PHT[0,0], &k, &f_inv[0,0,t], &n, &beta, &gain[0,0,t], &k)

        #beta_tt[t] = np.dot(gain[:,:,t], eta_tt1[:,t]) + beta_tt1[:,t] # kxn * nx1 + kx1
        #beta_tt[::1,t] = beta_tt1[::1,t]
        scopy(&k, &beta_tt1[0,t], &inc, &beta_tt[0,t], &inc)
        sgemv("N",&k,&n,&alpha,&gain[0,0,t],&k,&eta_tt1[0,t],&inc,&alpha,&beta_tt[0,t],&inc)

        #P_tt[t] =  -1* gain[t].dot(H_view).dot(P_tt1[t]) + P_tt1[t] # kxn * nxk * kxk + kxk
        if converged:
            scopy(&k2, &P_tt[0,0,t-1], &inc, &P_tt[0,0,t], &inc)
        else:
            #P_tt[::1,:,t] = P_tt1[::1,:,t]
            scopy(&k2, &P_tt1[0,0,t], &inc, &P_tt[0,0,t], &inc)
            sgemm("N", "N", &k, &k, &n, &alpha, &gain[0,0,t], &k, &H[0,0,H_idx], &n, &beta, &tmp[0,0], &ldwork)
            sgemm("N", "N", &k, &k, &k, &gamma, &tmp[0,0], &ldwork, &P_tt1[0,0,t], &k, &alpha, &P_tt[0,0,t], &k)

        # Check if we have converged (by finding the determinant of P
        if not converged and not time_varying_H:
            scopy(&k2, &P_tt[0,0,t], &inc, &tmp[0,0], &inc)
            saxpy(&k2, &gamma, &P_tt[0,0,t-1], &inc, &tmp[0,0], &inc)
            if sdot(&k2, &tmp[0,0], &inc, &tmp[0,0], &inc) < tol:
                converged = 1

    return beta_tt, P_tt, beta_tt1, P_tt1, y_tt1, eta_tt1, f_tt1, f_inv, gain, ll

# Kalman Filter: Double Precision
# TODO add G
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef dkalman_filter(double [::1,:]   y,  # nxT+1    (data: endogenous, observed)
                     double [::1,:,:] H,  # nxkxT+1  (parameters)
                     double [:]       mu, # kx0      (parameters)
                     double [::1,:]   F,  # kxk      (parameters)
                     double [::1,:]   R,  # nxn      (parameters: covariance matrix)
                     double [::1,:]   G,  # kxg      (parameters)
                     double [::1,:]   Q_star,  # gxg      (parameters: covariance matrix)
                     double [::1,:]   z=None,  # rxT+1    (data: weakly exogenous, observed)
                     double [::1,:]   A=None,  # nxr      (parameters)
                     double [:]       beta_tt_init=None,
                     double [::1,:]   P_tt_init=None):

    cdef double [::1,:,:] P_tt, P_tt1, f_tt1, gain, f_inv
    cdef int [::1,:] ipiv
    cdef double [::1,:] beta_tt, beta_tt1, y_tt1, eta_tt1, tmp, work, PHT, Q
    cdef double [:] ll
    cdef double det, tol = 10e-20
    cdef:
        int i
        int t
        int T = y.shape[1]
        int n = y.shape[0]
        int r = 0
        int g = Q_star.shape[0]
        int k = mu.shape[0]
        int time_varying_H = H.shape[2] == T
        int H_idx = 0
        int converged = 0
    cdef:
        int kn = k*n
        int k2 = k**2
        int n2 = n**2
        int info # return code for dgetri, dgetrf
        int inc = 1 # incrementer for dgemv
        int ldwork = max(k2, n2) # number of rows/columns in work array
        int lwork = ldwork**2 # size of work array for dgetri
    cdef:
        double alpha = 1.0 # first scalar multiple on dgemv, dgemm
        double beta = 0.0 # second scalar multiple on dgemv, dgemm
        double gamma = -1.0 # scalar multiple for daxpy
        double delta = -0.5 # scalar multiple for log calculation

    # Check if we have an exog matrix
    if z is not None and A is not None:
        r = z.shape[0]

    # Allocate memory for variables
    beta_tt = np.zeros((k,T+1), float, order="F")    # T+1xk
    P_tt = np.zeros((k,k,T+1), float, order="F")     # T+1xkxk
    beta_tt1 = np.zeros((k,T+1), float, order="F")   # T+1xk
    P_tt1 = np.zeros((k,k,T+1), float, order="F")    # T+1xkxk
    y_tt1 = np.zeros((n,T+1), float, order="F")      # T+1xn
    eta_tt1 = np.zeros((n,T+1), float, order="F")    # T+1xn
    f_tt1 = np.zeros((n,n,T+1), float, order="F")    # T+1xnxn
    gain = np.zeros((k,n,T+1), float, order="F")     # T+1xkxn
    ll = np.zeros((T+1,), float)                     # T+1
    work = np.zeros((ldwork,ldwork), float, order="F")
    ipiv = np.empty((ldwork,ldwork), np.int32, order="F")
    PHT = np.empty((k,n), float, order="F")
    f_inv = np.empty((n,n,T+1), float, order="F")
    Q = np.zeros((k,k), float, order="F")
    
    # Get Q = G Q^* G'
    dgemm("N", "N", &k, &g, &g, &alpha, &G[0,0], &k, &Q_star[0,0], &g, &beta, &work[0,0], &ldwork)
    dgemm("N", "T", &k, &g, &g, &alpha, &work[0,0], &ldwork, &G[0,0], &k, &beta, &Q[0,0], &k)

    # Initial values
    if beta_tt_init is None:
        #beta_tt[:,0] = np.linalg.inv(np.eye(k) - F).dot(mu) # kxk * kx1 = kx1
        beta_tt_init = np.zeros((k,), float, order="F")
        tmp = np.array(np.eye(k), float, order="F") - F
        dgetrf(&k, &k, &tmp[0,0], &k, &ipiv[0,0], &info)
        dgetri(&k, &tmp[0,0], &k, &ipiv[0,0], &work[0,0], &lwork, &info)
        dgemv("N",&k,&k,&alpha,&tmp[0,0],&k,&mu[0],&inc,&beta,&beta_tt_init[0],&inc)
    beta_tt[::1,0] = beta_tt_init[::1]

    if P_tt_init is None:
        #P_tt[0] = np.linalg.inv(np.eye(k**2) - np.kron(F,F)).dot(Q.reshape(Q.size, 1)).reshape(k,k) # kxk
        P_tt_init = np.zeros((k,k), float, order="F")
        tmp = np.array(np.eye(k2) - np.kron(F,F), float, order="F")
        dgetrf(&k2, &k2, &tmp[0,0], &k2, &ipiv[0,0], &info)
        dgetri(&k2, &tmp[0,0], &k2, &ipiv[0,0], &work[0,0], &lwork, &info)
        dgemv("N",&k2,&k2,&alpha,&tmp[0,0],&k2,&Q[0,0],&inc,&beta,&P_tt_init[0,0],&inc)
    P_tt[::1,:,0] = P_tt_init[::1,:]

    # Redefine the tmp array
    tmp = np.zeros((ldwork,ldwork), float, order="F")

    # Iterate forwards
    for t in range(1,T+1):
        if time_varying_H:
            H_idx = t-1

        # Prediction
        #beta_tt1[t] = mu + np.dot(F, beta_tt[t-1])
        #beta_tt1[::1,t] = mu[::1]
        dcopy(&k, &mu[0], &inc, &beta_tt1[0,t], &inc)
        dgemv("N",&k,&k,&alpha,&F[0,0],&k,&beta_tt[0,t-1],&inc,&alpha,&beta_tt1[0,t],&inc)

        #P_tt1[t] = np.dot(F, P_tt[t-1]).dot(F.T) + Q
        if converged:
            dcopy(&k2, &P_tt1[0,0,t-1], &inc, &P_tt1[0,0,t], &inc)
        else:
            #P_tt1[::1,:,t] = Q[::1,:]
            dcopy(&k2, &Q[0,0], &inc, &P_tt1[0,0,t], &inc)
            #dsymm("R", "L", &k, &k, &alpha, &F[0,0], &k, &P_tt[0,0,t-1], &k, &beta, &tmp[0,0], &ldwork)
            dgemm("N", "N", &k, &k, &k, &alpha, &F[0,0], &k, &P_tt[0,0,t-1], &k, &beta, &tmp[0,0], &ldwork)
            dgemm("N", "T", &k, &k, &k, &alpha, &tmp[0,0], &ldwork, &F[0,0], &k, &alpha, &P_tt1[0,0,t], &k)

        #y_tt1[t] = np.dot(H[:,:,H_idx], beta_tt1[:,t]) + np.dot(A,z[:,t-1])
        dgemv("N", &n, &k, &alpha, &H[0,0,H_idx], &n, &beta_tt1[0,t], &inc, &beta, &y_tt1[0,t], &inc)
        if r > 0:
            # z[0] corresponds to z[t=1]
            dgemv("N", &n, &r, &alpha, &A[0,0], &n, &z[0,t-1], &inc, &alpha, &y_tt1[0,t], &inc)

        #eta_tt1[::1,t] = y[::1,t-1] - y_tt1[:,t]
        #eta_tt1[::1,t] = y[::1,t-1] # y[0] corresponds to y[t=1]
        dcopy(&n, &y[0,t-1], &inc, &eta_tt1[0,t], &inc)
        daxpy(&n, &gamma, &y_tt1[0,t], &inc, &eta_tt1[0,t], &inc)

        if converged:
            dcopy(&n2, &f_tt1[0,0,t-1], &inc, &f_tt1[0,0,t], &inc)
            dcopy(&n2, &f_inv[0,0,t-1], &inc, &f_inv[0,0,t], &inc)
        else:
            #PHT = np.dot(P_tt1[t], H[:,:,H_idx].T) # kxn
            #print np.dot(P_tt1[:,:,t], H[:,:,H_idx].T) # taking .T here crashes the program for some reason
            dgemm("N", "T", &k, &n, &k, &alpha, &P_tt1[0,0,t], &k, &H[0,0,H_idx], &n, &beta, &PHT[0,0], &k)
    
            #f_tt1[t] = np.dot(H[:,:,H_idx], PHT) + R
            #f_tt1[::1,:,t] = R[::1,:]
            dcopy(&n2, &R[0,0], &inc, &f_tt1[0,0,t], &inc)
            dgemm("N", "N", &n, &n, &k, &alpha, &H[0,0,H_idx], &n, &PHT[0,0], &k, &alpha, &f_tt1[0,0,t], &n)
    
            #f_inv = np.linalg.inv(f_tt1[t])
            #f_inv[::1,:] = f_tt1[::1,:,t]
            if n == 1:
                det = dabs(f_tt1[0,0,t])
                f_inv[0,0,t] = 1/f_tt1[0,0,t]
            else:
                dcopy(&n2, &f_tt1[0,0,t], &inc, &f_inv[0,0,t], &inc)
                dgetrf(&n, &n, &f_inv[0,0,t], &n, &ipiv[0,0], &info)
                det = 1
                for i in range(n):
                    if not ipiv[i,0] == i+1:
                        det *= -1*f_inv[i,i,t]
                    else:
                        det *= f_inv[i,i,t]
                # Now complete taking the inverse
                dgetri(&n, &f_inv[0,0,t], &n, &ipiv[0,0], &work[0,0], &lwork, &info)

        # Log-likelihood as byproduct
        #ll[t] -0.5*log(2*np.pi*np.linalg.det(f_tt1[:,:,t])) - 0.5*np.dot(np.dot(eta_tt1[:,t].T, f_inv), eta_tt1[:,t])
        # ^ this doesn't work, crashes for some reason; probably related to taking .T as it did above
        ll[t] = -0.5*dlog(2*np.pi*det)
        dgemv("N",&n,&n,&alpha,&f_inv[0,0,t],&n,&eta_tt1[0,t],&inc,&beta,&tmp[0,0],&inc)
        ll[t] += -0.5*ddot(&n, &eta_tt1[0,t], &inc, &tmp[0,0], &inc)

        # Updating
        #gain[t] = np.dot(PHT, f_inv) # kxn * nxn = kxn
        if converged:
            dcopy(&kn, &gain[0,0,t-1], &inc, &gain[0,0,t], &inc)
        else:
            dgemm("N", "N", &k, &n, &n, &alpha, &PHT[0,0], &k, &f_inv[0,0,t], &n, &beta, &gain[0,0,t], &k)

        #beta_tt[t] = np.dot(gain[:,:,t], eta_tt1[:,t]) + beta_tt1[:,t] # kxn * nx1 + kx1
        #beta_tt[::1,t] = beta_tt1[::1,t]
        dcopy(&k, &beta_tt1[0,t], &inc, &beta_tt[0,t], &inc)
        dgemv("N",&k,&n,&alpha,&gain[0,0,t],&k,&eta_tt1[0,t],&inc,&alpha,&beta_tt[0,t],&inc)

        #P_tt[t] =  -1* gain[t].dot(H_view).dot(P_tt1[t]) + P_tt1[t] # kxn * nxk * kxk + kxk
        if converged:
            dcopy(&k2, &P_tt[0,0,t-1], &inc, &P_tt[0,0,t], &inc)
        else:
            #P_tt[::1,:,t] = P_tt1[::1,:,t]
            dcopy(&k2, &P_tt1[0,0,t], &inc, &P_tt[0,0,t], &inc)
            dgemm("N", "N", &k, &k, &n, &alpha, &gain[0,0,t], &k, &H[0,0,H_idx], &n, &beta, &tmp[0,0], &ldwork)
            dgemm("N", "N", &k, &k, &k, &gamma, &tmp[0,0], &ldwork, &P_tt1[0,0,t], &k, &alpha, &P_tt[0,0,t], &k)
        
        # Check if we have converged (by finding the determinant of P
        if not converged and not time_varying_H:
            dcopy(&k2, &P_tt[0,0,t], &inc, &tmp[0,0], &inc)
            daxpy(&k2, &gamma, &P_tt[0,0,t-1], &inc, &tmp[0,0], &inc)
            if ddot(&k2, &tmp[0,0], &inc, &tmp[0,0], &inc) < tol:
                converged = 1

    return beta_tt, P_tt, beta_tt1, P_tt1, y_tt1, eta_tt1, f_tt1, f_inv, gain, ll

cpdef ckalman_filter(
                    np.complex64_t [::1,:]   y,  # nxT+1    (data: endogenous, observed)
                    np.complex64_t [::1,:,:] H,  # nxkxT+1  (parameters)
                    np.complex64_t [:]       mu, # kx0      (parameters)
                    np.complex64_t [::1,:]   F,  # kxk      (parameters)
                    np.complex64_t [::1,:]   R,  # nxn      (parameters: covariance matrix)
                    np.complex64_t [::1,:]   G,  # kxg      (parameters)
                    np.complex64_t [::1,:]   Q_star,  # gxg      (parameters: covariance matrix)
                    np.complex64_t [::1,:]   z=None,  # rxT+1    (data: weakly exogenous, observed)
                    np.complex64_t [::1,:]   A=None,  # nxr      (parameters)
                    np.complex64_t [:]       beta_tt_init=None,
                    np.complex64_t [::1,:]   P_tt_init=None):

    cdef np.complex64_t [::1,:,:] P_tt, P_tt1, f_tt1, gain, f_inv
    cdef int [::1,:] ipiv
    cdef np.complex64_t [::1,:] beta_tt, beta_tt1, y_tt1, eta_tt1, tmp, work, PHT, Q
    cdef np.complex64_t [:] ll
    cdef np.complex64_t det
    cdef double tol = 10e-20
    cdef:
        int i
        int t
        int T = y.shape[1]
        int n = y.shape[0]
        int r = 0
        int g = Q_star.shape[0]
        int k = mu.shape[0]
        int time_varying_H = H.shape[2] == T
        int H_idx = 0
        int converged = 0
    cdef:
        int kn = k*n
        int k2 = k**2
        int n2 = n**2
        int info # return code fro dgetri, dgetrf
        int inc = 1 # incrementer for dgemv
        int ldwork = max(k2, n2) # number of rows/columns in work array
        int lwork = ldwork**2 # size of work array for dgetri
    cdef:
        np.complex64_t alpha = 1.0 # first scalar multiple on dgemv, dgemm
        np.complex64_t beta = 0.0 # second scalar multiple on dgemv, dgemm
        np.complex64_t gamma = -1.0 # scalar multiple for daxpy
        np.complex64_t delta = -0.5 # scalar multiple for log calculation

    # Check if we have an exog matrix
    if z is not None and A is not None:
        r = z.shape[0]

    # Allocate memory for variables
    beta_tt = np.zeros((k,T+1), np.complex64, order="F")    # T+1xk
    P_tt = np.zeros((k,k,T+1), np.complex64, order="F")     # T+1xkxk
    beta_tt1 = np.zeros((k,T+1), np.complex64, order="F")   # T+1xk
    P_tt1 = np.zeros((k,k,T+1), np.complex64, order="F")    # T+1xkxk
    y_tt1 = np.zeros((n,T+1), np.complex64, order="F")      # T+1xn
    eta_tt1 = np.zeros((n,T+1), np.complex64, order="F")    # T+1xn
    f_tt1 = np.zeros((n,n,T+1), np.complex64, order="F")    # T+1xnxn
    gain = np.zeros((k,n,T+1), np.complex64, order="F")     # T+1xkxn
    ll = np.zeros((T+1,), np.complex64)                     # T+1
    work = np.zeros((ldwork,ldwork), np.complex64, order="F")
    ipiv = np.empty((ldwork,ldwork), np.int32, order="F")
    PHT = np.empty((k,n), np.complex64, order="F")
    f_inv = np.empty((n,n), np.complex64, order="F")
    Q = np.zeros((k,k), np.complex64, order="F")
    
    # Get Q = G Q^* G'
    cgemm("N", "N", &k, &g, &g, &alpha, &G[0,0], &k, &Q_star[0,0], &g, &beta, &work[0,0], &ldwork)
    cgemm("N", "T", &k, &g, &g, &alpha, &work[0,0], &ldwork, &G[0,0], &k, &beta, &Q[0,0], &k)

    # Initial values
    if beta_tt_init is None:
        #beta_tt[:,0] = np.linalg.inv(np.eye(k) - F).dot(mu) # kxk * kx1 = kx1
        beta_tt_init = np.zeros((k,), np.complex64, order="F")
        tmp = np.array(np.eye(k), np.complex64, order="F") - F
        cgetrf(&k, &k, &tmp[0,0], &k, &ipiv[0,0], &info)
        cgetri(&k, &tmp[0,0], &k, &ipiv[0,0], &work[0,0], &lwork, &info)
        cgemv("N",&k,&k,&alpha,&tmp[0,0],&k,&mu[0],&inc,&beta,&beta_tt_init[0],&inc)
    beta_tt[::1,0] = beta_tt_init[::1]

    if P_tt_init is None:
        #P_tt[0] = np.linalg.inv(np.eye(k**2) - np.kron(F,F)).dot(Q.reshape(Q.size, 1)).reshape(k,k) # kxk
        P_tt_init = np.zeros((k,k), np.complex64, order="F")
        tmp = np.array(np.eye(k2) - np.kron(F,F), np.complex64, order="F")
        cgetrf(&k2, &k2, &tmp[0,0], &k2, &ipiv[0,0], &info)
        cgetri(&k2, &tmp[0,0], &k2, &ipiv[0,0], &work[0,0], &lwork, &info)
        cgemv("N",&k2,&k2,&alpha,&tmp[0,0],&k2,&Q[0,0],&inc,&beta,&P_tt_init[0,0],&inc)
    P_tt[::1,:,0] = P_tt_init[::1,:]

    # Redefine the tmp array
    tmp = np.empty((ldwork,ldwork), np.complex64, order="F")

    # Iterate forwards
    for t in range(1,T+1):
        if time_varying_H:
            H_idx = t-1

        # Prediction
        #beta_tt1[t] = mu + np.dot(F, beta_tt[t-1])
        #beta_tt1[::1,t] = mu[::1]
        ccopy(&k, &mu[0], &inc, &beta_tt1[0,t], &inc)
        cgemv("N",&k,&k,&alpha,&F[0,0],&k,&beta_tt[0,t-1],&inc,&alpha,&beta_tt1[0,t],&inc)

        #P_tt1[t] = np.dot(F, P_tt[t-1]).dot(F.T) + Q
        if converged:
            ccopy(&k2, &P_tt1[0,0,t-1], &inc, &P_tt1[0,0,t], &inc)
        else:
            #P_tt1[::1,:,t] = Q[::1,:]
            cgemm("N", "N", &k, &g, &g, &alpha, &G[0,0], &k, &Q[0,0], &g, &beta, &tmp[0,0], &ldwork)
            cgemm("N", "T", &k, &g, &g, &alpha, &tmp[0,0], &ldwork, &G[0,0], &k, &beta, &P_tt1[0,0,t], &k)
            #ccopy(&k2, &Q[0,0], &inc, &P_tt1[0,0,t], &inc)
            #csymm("R", "L", &k, &k, &alpha, &F[0,0], &k, &P_tt[0,0,t-1], &k, &beta, &tmp[0,0], &ldwork)
            cgemm("N", "N", &k, &k, &k, &alpha, &F[0,0], &k, &P_tt[0,0,t-1], &k, &beta, &tmp[0,0], &ldwork)
            cgemm("N", "T", &k, &k, &k, &alpha, &tmp[0,0], &ldwork, &F[0,0], &k, &alpha, &P_tt1[0,0,t], &k)

        #y_tt1[t] = np.dot(H[:,:,H_idx], beta_tt1[:,t]) + np.dot(A,z[:,t-1])
        cgemv("N", &n, &k, &alpha, &H[0,0,H_idx], &n, &beta_tt1[0,t], &inc, &beta, &y_tt1[0,t], &inc)
        if r > 0:
            # z[0] corresponds to z[t=1]
            cgemv("N", &n, &r, &alpha, &A[0,0], &n, &z[0,t-1], &inc, &alpha, &y_tt1[0,t], &inc)

        #eta_tt1[::1,t] = y[::1,t-1] - y_tt1[:,t]
        #eta_tt1[::1,t] = y[::1,t-1] # y[0] corresponds to y[t=1]
        ccopy(&n, &y[0,t-1], &inc, &eta_tt1[0,t], &inc)
        caxpy(&n, &gamma, &y_tt1[0,t], &inc, &eta_tt1[0,t], &inc)

        if converged:
            ccopy(&n2, &f_tt1[0,0,t-1], &inc, &f_tt1[0,0,t], &inc)
            ccopy(&n2, &f_inv[0,0,t-1], &inc, &f_inv[0,0,t], &inc)
        else:
            #PHT = np.dot(P_tt1[t], H[:,:,H_idx].T) # kxn
            #print np.dot(P_tt1[:,:,t], H[:,:,H_idx].T) # taking .T here crashes the program for some reason
            cgemm("N", "T", &k, &n, &k, &alpha, &P_tt1[0,0,t], &k, &H[0,0,H_idx], &n, &beta, &PHT[0,0], &k)

            #f_tt1[t] = np.dot(H[:,:,H_idx], PHT) + R
            #f_tt1[::1,:,t] = R[::1,:]
            ccopy(&n2, &R[0,0], &inc, &f_tt1[0,0,t], &inc)
            cgemm("N", "N", &n, &n, &k, &alpha, &H[0,0,H_idx], &n, &PHT[0,0], &k, &alpha, &f_tt1[0,0,t], &n)

            #f_inv = np.linalg.inv(f_tt1[t])
            #f_inv[::1,:] = f_tt1[::1,:,t]
            if n == 1:
                det = cabsf(f_tt1[0,0,t])
                f_inv[0,0,t] = 1/f_tt1[0,0,t]
            else:
                ccopy(&n2, &f_tt1[0,0,t], &inc, &f_inv[0,0,t], &inc)
                cgetrf(&n, &n, &f_inv[0,0,t], &n, &ipiv[0,0], &info)
                det = 1
                for i in range(n):
                    if not ipiv[i,0] == i+1:
                        det *= -1*f_inv[i,i,t]
                    else:
                        det *= f_inv[i,i,t]
                # Now complete taking the inverse
                cgetri(&n, &f_inv[0,0,t], &n, &ipiv[0,0], &work[0,0], &lwork, &info)

        # Log-likelihood as byproduct
        #ll[t] -0.5*log(2*np.pi*np.linalg.det(f_tt1[:,:,t])) - 0.5*np.dot(np.dot(eta_tt1[:,t].T, f_inv), eta_tt1[:,t])
        # ^ this doesn't work, crashes for some reason; probably related to taking .T as it did above
        ll[t] = -0.5*clogf(2*np.pi*det)
        cgemv("N",&n,&n,&alpha,&f_inv[0,0,t],&n,&eta_tt1[0,t],&inc,&beta,&tmp[0,0],&inc)
        # ll[t] += -0.5*zdotu(&n, &eta_tt1[0,t], &inc, &tmp[0,0], &inc)
        # ^ zdotu, cdotu don't work, give a segfault 11, not sure why
        cgemv("N",&inc,&n,&alpha,&eta_tt1[0,t],&inc,&tmp[0,0],&inc,&beta,&work[0,0],&inc)
        ll[t] += -0.5*work[0,0]

        # Updating
        #gain[t] = np.dot(PHT, f_inv) # kxn * nxn = kxn
        if converged:
            ccopy(&kn, &gain[0,0,t-1], &inc, &gain[0,0,t], &inc)
        else:
            cgemm("N", "N", &k, &n, &n, &alpha, &PHT[0,0], &k, &f_inv[0,0,t], &n, &beta, &gain[0,0,t], &k)

        #beta_tt[t] = np.dot(gain[:,:,t], eta_tt1[:,t]) + beta_tt1[:,t] # kxn * nx1 + kx1
        #beta_tt[::1,t] = beta_tt1[::1,t]
        ccopy(&k, &beta_tt1[0,t], &inc, &beta_tt[0,t], &inc)
        cgemv("N",&k,&n,&alpha,&gain[0,0,t],&k,&eta_tt1[0,t],&inc,&alpha,&beta_tt[0,t],&inc)

        #P_tt[t] =  -1* gain[t].dot(H_view).dot(P_tt1[t]) + P_tt1[t] # kxn * nxk * kxk + kxk
        if converged:
            ccopy(&k2, &P_tt[0,0,t-1], &inc, &P_tt[0,0,t], &inc)
        else:
            #P_tt[::1,:,t] = P_tt1[::1,:,t]
            ccopy(&k2, &P_tt1[0,0,t], &inc, &P_tt[0,0,t], &inc)
            cgemm("N", "N", &k, &k, &n, &alpha, &gain[0,0,t], &k, &H[0,0,H_idx], &n, &beta, &tmp[0,0], &ldwork)
            cgemm("N", "N", &k, &k, &k, &gamma, &tmp[0,0], &ldwork, &P_tt1[0,0,t], &k, &alpha, &P_tt[0,0,t], &k)

        # Check if we have converged (by finding the determinant of P
        if not converged and not time_varying_H:
            ccopy(&k2, &P_tt[0,0,t], &inc, &tmp[0,0], &inc)
            caxpy(&k2, &gamma, &P_tt[0,0,t-1], &inc, &tmp[0,0], &inc)
            cgemv("N",&inc,&k2,&alpha,&tmp[0,0],&inc,&tmp[0,0],&inc,&beta,&work[0,0],&inc)
            if <float> cabs(work[0,0]) < tol:
                converged = 1

    return beta_tt, P_tt, beta_tt1, P_tt1, y_tt1, eta_tt1, f_tt1, f_inv, gain, ll

cpdef zkalman_filter(
                    complex [::1,:]   y,  # nxT+1    (data: endogenous, observed)
                    complex [::1,:,:] H,  # nxkxT+1  (parameters)
                    complex [:]       mu, # kx0      (parameters)
                    complex [::1,:]   F,  # kxk      (parameters)
                    complex [::1,:]   R,  # nxn      (parameters: covariance matrix)
                    complex [::1,:]   G,  # kxg      (parameters)
                    complex [::1,:]   Q_star,  # gxg      (parameters: covariance matrix)
                    complex [::1,:]   z=None,  # rxT+1    (data: weakly exogenous, observed)
                    complex [::1,:]   A=None,  # nxr      (parameters)
                    complex [:]       beta_tt_init=None,
                    complex [::1,:]   P_tt_init=None):

    cdef complex [::1,:,:] P_tt, P_tt1, f_tt1, gain, f_inv
    cdef int [::1,:] ipiv
    cdef complex [::1,:] beta_tt, beta_tt1, y_tt1, eta_tt1, tmp, work, PHT, Q
    cdef complex [:] ll
    cdef complex det
    cdef double tol = 10e-20
    cdef:
        int i
        int t
        int T = y.shape[1]
        int n = y.shape[0]
        int r = 0
        int g = Q_star.shape[0]
        int k = mu.shape[0]
        int time_varying_H = H.shape[2] == T
        int H_idx = 0
        int converged = 0
    cdef:
        int kn = k*n
        int k2 = k**2
        int n2 = n**2
        int info # return code fro dgetri, dgetrf
        int inc = 1 # incrementer for dgemv
        int ldwork = max(k2, n2) # number of rows/columns in work array
        int lwork = ldwork**2 # size of work array for dgetri
    cdef:
        complex alpha = 1.0 # first scalar multiple on dgemv, dgemm
        complex beta = 0.0 # second scalar multiple on dgemv, dgemm
        complex gamma = -1.0 # scalar multiple for daxpy
        complex delta = -0.5 # scalar multiple for log calculation

    # Check if we have an exog matrix
    if z is not None and A is not None:
        r = z.shape[0]

    # Allocate memory for variables
    beta_tt = np.zeros((k,T+1), complex, order="F")    # T+1xk
    P_tt = np.zeros((k,k,T+1), complex, order="F")     # T+1xkxk
    beta_tt1 = np.zeros((k,T+1), complex, order="F")   # T+1xk
    P_tt1 = np.zeros((k,k,T+1), complex, order="F")    # T+1xkxk
    y_tt1 = np.zeros((n,T+1), complex, order="F")      # T+1xn
    eta_tt1 = np.zeros((n,T+1), complex, order="F")    # T+1xn
    f_tt1 = np.zeros((n,n,T+1), complex, order="F")    # T+1xnxn
    gain = np.zeros((k,n,T+1), complex, order="F")     # T+1xkxn
    ll = np.zeros((T+1,), complex)                     # T+1
    work = np.zeros((ldwork,ldwork), complex, order="F")
    ipiv = np.empty((ldwork,ldwork), np.int32, order="F")
    PHT = np.empty((k,n), complex, order="F")
    f_inv = np.empty((n,n,T+1), complex, order="F")
    Q = np.zeros((k,k), complex, order="F")
    
    # Get Q = G Q^* G'
    zgemm("N", "N", &k, &g, &g, &alpha, &G[0,0], &k, &Q_star[0,0], &g, &beta, &work[0,0], &ldwork)
    zgemm("N", "T", &k, &g, &g, &alpha, &work[0,0], &ldwork, &G[0,0], &k, &beta, &Q[0,0], &k)

    # Initial values
    if beta_tt_init is None:
        #beta_tt[:,0] = np.linalg.inv(np.eye(k) - F).dot(mu) # kxk * kx1 = kx1
        beta_tt_init = np.zeros((k,), complex, order="F")
        tmp = np.array(np.eye(k), complex, order="F") - F
        zgetrf(&k, &k, &tmp[0,0], &k, &ipiv[0,0], &info)
        zgetri(&k, &tmp[0,0], &k, &ipiv[0,0], &work[0,0], &lwork, &info)
        zgemv("N",&k,&k,&alpha,&tmp[0,0],&k,&mu[0],&inc,&beta,&beta_tt_init[0],&inc)
    beta_tt[::1,0] = beta_tt_init[::1]

    if P_tt_init is None:
        #P_tt[0] = np.linalg.inv(np.eye(k**2) - np.kron(F,F)).dot(Q.reshape(Q.size, 1)).reshape(k,k) # kxk
        P_tt_init = np.zeros((k,k), complex, order="F")
        tmp = np.array(np.eye(k2) - np.kron(F,F), complex, order="F")
        zgetrf(&k2, &k2, &tmp[0,0], &k2, &ipiv[0,0], &info)
        zgetri(&k2, &tmp[0,0], &k2, &ipiv[0,0], &work[0,0], &lwork, &info)
        zgemv("N",&k2,&k2,&alpha,&tmp[0,0],&k2,&Q[0,0],&inc,&beta,&P_tt_init[0,0],&inc)
    P_tt[::1,:,0] = P_tt_init[::1,:]

    # Define the tmp array
    tmp = np.empty((ldwork,ldwork), complex, order="F")

    # Iterate forwards
    for t in range(1,T+1):
        if time_varying_H:
            H_idx = t-1

        # Prediction
        #beta_tt1[t] = mu + np.dot(F, beta_tt[t-1])
        #beta_tt1[::1,t] = mu[::1]
        zcopy(&k, &mu[0], &inc, &beta_tt1[0,t], &inc)
        zgemv("N",&k,&k,&alpha,&F[0,0],&k,&beta_tt[0,t-1],&inc,&alpha,&beta_tt1[0,t],&inc)

        #P_tt1[t] = np.dot(F, P_tt[t-1]).dot(F.T) + Q
        if converged:
            zcopy(&k2, &P_tt1[0,0,t-1], &inc, &P_tt1[0,0,t], &inc)
        else:
            #P_tt1[::1,:,t] = Q[::1,:]
            zgemm("N", "N", &k, &g, &g, &alpha, &G[0,0], &k, &Q[0,0], &g, &beta, &tmp[0,0], &ldwork)
            zgemm("N", "T", &k, &g, &g, &alpha, &tmp[0,0], &ldwork, &G[0,0], &k, &beta, &P_tt1[0,0,t], &k)
            #zcopy(&k2, &Q[0,0], &inc, &P_tt1[0,0,t], &inc)
            #zsymm("R", "L", &k, &k, &alpha, &F[0,0], &k, &P_tt[0,0,t-1], &k, &beta, &tmp[0,0], &ldwork)
            zgemm("N", "N", &k, &k, &k, &alpha, &F[0,0], &k, &P_tt[0,0,t-1], &k, &beta, &tmp[0,0], &ldwork)
            zgemm("N", "T", &k, &k, &k, &alpha, &tmp[0,0], &ldwork, &F[0,0], &k, &alpha, &P_tt1[0,0,t], &k)

        #y_tt1[t] = np.dot(H[:,:,H_idx], beta_tt1[:,t]) + np.dot(A,z[:,t-1])
        zgemv("N", &n, &k, &alpha, &H[0,0,H_idx], &n, &beta_tt1[0,t], &inc, &beta, &y_tt1[0,t], &inc)
        if r > 0:
            # z[0] corresponds to z[t=1]
            zgemv("N", &n, &r, &alpha, &A[0,0], &n, &z[0,t-1], &inc, &alpha, &y_tt1[0,t], &inc)

        #eta_tt1[::1,t] = y[::1,t-1] - y_tt1[:,t]
        #eta_tt1[::1,t] = y[::1,t-1] # y[0] corresponds to y[t=1]
        zcopy(&n, &y[0,t-1], &inc, &eta_tt1[0,t], &inc)
        zaxpy(&n, &gamma, &y_tt1[0,t], &inc, &eta_tt1[0,t], &inc)

        if converged:
            zcopy(&n2, &f_tt1[0,0,t-1], &inc, &f_tt1[0,0,t], &inc)
            zcopy(&n2, &f_inv[0,0,t-1], &inc, &f_inv[0,0,t], &inc)
        else:
            #PHT = np.dot(P_tt1[t], H[:,:,H_idx].T) # kxn
            #print np.dot(P_tt1[:,:,t], H[:,:,H_idx].T) # taking .T here crashes the program for some reason
            zgemm("N", "T", &k, &n, &k, &alpha, &P_tt1[0,0,t], &k, &H[0,0,H_idx], &n, &beta, &PHT[0,0], &k)

            #f_tt1[t] = np.dot(H[:,:,H_idx], PHT) + R
            #f_tt1[::1,:,t] = R[::1,:]
            zcopy(&n2, &R[0,0], &inc, &f_tt1[0,0,t], &inc)
            zgemm("N", "N", &n, &n, &k, &alpha, &H[0,0,H_idx], &n, &PHT[0,0], &k, &alpha, &f_tt1[0,0,t], &n)

            #f_inv = np.linalg.inv(f_tt1[t])
            #f_inv[::1,:] = f_tt1[::1,:,t]
            if n == 1:
                det = cabs(f_tt1[0,0,t])
                f_inv[0,0,t] = 1/f_tt1[0,0,t]
            else:
                zcopy(&n2, &f_tt1[0,0,t], &inc, &f_inv[0,0,t], &inc)
                zgetrf(&n, &n, &f_inv[0,0,t], &n, &ipiv[0,0], &info)
                det = 1
                for i in range(n):
                    if not ipiv[i,0] == i+1:
                        det *= -1*f_inv[i,i,t]
                    else:
                        det *= f_inv[i,i,t]
                # Now complete taking the inverse
                zgetri(&n, &f_inv[0,0,t], &n, &ipiv[0,0], &work[0,0], &lwork, &info)

        # Log-likelihood as byproduct
        #ll[t] -0.5*log(2*np.pi*np.linalg.det(f_tt1[:,:,t])) - 0.5*np.dot(np.dot(eta_tt1[:,t].T, f_inv), eta_tt1[:,t])
        # ^ this doesn't work, crashes for some reason; probably related to taking .T as it did above
        ll[t] = -0.5*clog(2*np.pi*det)
        zgemv("N",&n,&n,&alpha,&f_inv[0,0,t],&n,&eta_tt1[0,t],&inc,&beta,&tmp[0,0],&inc)
        # ll[t] += -0.5*zdotu(&n, &eta_tt1[0,t], &inc, &tmp[0,0], &inc)
        # ^ zdotu, cdotu don't work, give a segfault 11, not sure why
        zgemv("N",&inc,&n,&alpha,&eta_tt1[0,t],&inc,&tmp[0,0],&inc,&beta,&work[0,0],&inc)
        ll[t] += -0.5*work[0,0]

        # Updating
        #gain[t] = np.dot(PHT, f_inv) # kxn * nxn = kxn
        if converged:
            zcopy(&kn, &gain[0,0,t-1], &inc, &gain[0,0,t], &inc)
        else:
            zgemm("N", "N", &k, &n, &n, &alpha, &PHT[0,0], &k, &f_inv[0,0,t], &n, &beta, &gain[0,0,t], &k)

        #beta_tt[t] = np.dot(gain[:,:,t], eta_tt1[:,t]) + beta_tt1[:,t] # kxn * nx1 + kx1
        #beta_tt[::1,t] = beta_tt1[::1,t]
        zcopy(&k, &beta_tt1[0,t], &inc, &beta_tt[0,t], &inc)
        zgemv("N",&k,&n,&alpha,&gain[0,0,t],&k,&eta_tt1[0,t],&inc,&alpha,&beta_tt[0,t],&inc)

        #P_tt[t] =  -1* gain[t].dot(H_view).dot(P_tt1[t]) + P_tt1[t] # kxn * nxk * kxk + kxk
        if converged:
            zcopy(&k2, &P_tt[0,0,t-1], &inc, &P_tt[0,0,t], &inc)
        else:
            #P_tt[::1,:,t] = P_tt1[::1,:,t]
            zcopy(&k2, &P_tt1[0,0,t], &inc, &P_tt[0,0,t], &inc)
            zgemm("N", "N", &k, &k, &n, &alpha, &gain[0,0,t], &k, &H[0,0,H_idx], &n, &beta, &tmp[0,0], &ldwork)
            zgemm("N", "N", &k, &k, &k, &gamma, &tmp[0,0], &ldwork, &P_tt1[0,0,t], &k, &alpha, &P_tt[0,0,t], &k)

        # Check if we have converged (by finding the determinant of P
        if not converged and not time_varying_H:
            zcopy(&k2, &P_tt[0,0,t], &inc, &tmp[0,0], &inc)
            zaxpy(&k2, &gamma, &P_tt[0,0,t-1], &inc, &tmp[0,0], &inc)
            zgemv("N",&inc,&k2,&alpha,&tmp[0,0],&inc,&tmp[0,0],&inc,&beta,&work[0,0],&inc)
            if <float> cabs(work[0,0]) < tol:
                converged = 1

    return beta_tt, P_tt, beta_tt1, P_tt1, y_tt1, eta_tt1, f_tt1, f_inv, gain, ll
