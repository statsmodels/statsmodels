import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport log
from cpython cimport PyCObject_AsVoidPtr
import scipy
__import__('scipy.linalg.blas')
__import__('scipy.linalg.lapack')

ctypedef int dgemm_t(
    # Compute C := alpha*A*B + beta*C
    char *transa,  # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    char *transb,  # {'T','C'}: o(B)=B'; {'N'}: o(B)=B
    int *m,        # Rows of o(A)    (and of C)
    int *n,        # Columns of o(B) (and of C)
    int *k,        # Columns of o(A) / Rows of o(B)
    double *alpha, # Scalar multiple
    double *a,     # Matrix A: mxk
    int *lda,      # The size of the first dimension of A (in memory)
    double *b,     # Matrix B: kxn
    int *ldb,      # The size of the first dimension of B (in memory)
    double *beta,  # Scalar multiple
    double *c,     # Matrix C: mxn
    int *ldc       # The size of the first dimension of C (in memory)
)

ctypedef int dgemv_t(
    # Compute C := alpha*A*x + beta*y
    char *trans,   # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    int *m,        # Rows of o(A)
    int *n,        # Columns of o(A) / min(len(x))
    double *alpha, # Scalar multiple
    double *a,     # Matrix A: mxn
    int *lda,      # The size of the first dimension of A (in memory)
    double *x,     # Vector x, min(len(x)) = n
    int *incx,     # The increment between elements of x (usually 1)
    double *beta,  # Scalar multiple
    double *y,     # Vector y, min(len(y)) = m
    int *incy      # The increment between elements of y (usually 1)
)

ctypedef int daxpy_t(
    # Compute y := alpha*x + y
    int *n,        # Columns of o(A) / min(len(x))
    double *alpha, # Scalar multiple
    double *x,     # Vector x, min(len(x)) = n
    int *incx,     # The increment between elements of x (usually 1)
    double *y,     # Vector y, min(len(y)) = m
    int *incy      # The increment between elements of y (usually 1)
)

ctypedef double ddot_t(
    # Compute DDOT := x.T * y
    int *n,        # Length of vectors
    double *x,     # Vector x, min(len(x)) = n
    int *incx,     # The increment between elements of x (usually 1)
    double *y,     # Vector y, min(len(y)) = m
    int *incy      # The increment between elements of y (usually 1)
)


ctypedef int dgetrf_t(
    int *m,        # Rows of A
    int *n,        # Columns of A
    double *a,     # Matrix A: mxn
    int *lda,      # The size of the first dimension of A (in memory)
    int *ipiv,     # Matrix P: mxn (the pivot indices)
    int *info      # 0 if success, otherwise an error code (integer)
)

ctypedef int dgetri_t(
    int *n,        # Order of A
    double *a,     # Matrix A: nxn (the LUP decomposed matrix from dgetrf)
    int *lda,      # The size of the first dimension of A (in memory)
    int *ipiv,     # Matrix P: nxn (the pivot indices from the LUP decomposition)
    double *work,  # Matrix: nxn (a workspace for the inversion, optimal size=nxn)
    int *lwork,    # Number of elements in the workspace: optimal is n**2
    int *info      # 0 if success, otherwise an error code (integer)
)

cdef dgemm_t *dgemm = <dgemm_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.dgemm._cpointer)
cdef dgemv_t *dgemv = <dgemv_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.dgemv._cpointer)
cdef daxpy_t *daxpy = <daxpy_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.daxpy._cpointer)
cdef ddot_t *ddot = <ddot_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.ddot._cpointer)
cdef dgetrf_t *dgetrf = <dgetrf_t*>PyCObject_AsVoidPtr(scipy.linalg.lapack.dgetrf._cpointer)
cdef dgetri_t *dgetri = <dgetri_t*>PyCObject_AsVoidPtr(scipy.linalg.lapack.dgetri._cpointer)

# Kalman Filter
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef kalman_filter(double [::1,:]   y,  # nxT+1    (data: endogenous, observed)
                    double [::1,:]   z,  # rxT+1    (data: weakly exogenous, observed)
                    double [::1,:]   A,  # nxr      (parameters)
                    double [::1,:,:] H,  # nxkxT+1  (parameters)
                    double [:]       mu, # kx0      (parameters)
                    double [::1,:]   F,  # kxk      (parameters)
                    double [::1,:]   R,  # nxn      (parameters: covariance matrix)
                    double [::1,:]   Q,  # kxk      (parameters: covariance matrix)
                    double [:]       beta_tt_init=None,
                    double [::1,:]   P_tt_init=None):
    
    cdef double [::1,:,:] P_tt, P_tt1, f_tt1, gain
    cdef int [::1,:] ipiv
    cdef double [::1,:] beta_tt, beta_tt1, y_tt1, eta_tt1, tmp, work, PHT, f_inv
    cdef double [:] ll
    cdef double det
    cdef:
        int i
        int t
        int T = y.shape[1]-1
        int n = y.shape[0]
        int r = z.shape[0]
        int k = mu.shape[0]
        int time_varying_H = H.shape[2] == T+1
        int H_idx = 0
    cdef:
        int k2 = k**2
        int n2 = n**2
        int info # return code fro dgetri, dgetrf
        int inc = 1 # incrementer for dgemv
        int ldwork = max(k2, n2) # number of rows/columns in work array
        int lwork = ldwork**2 # size of work array for dgetri
    cdef:
        double alpha = 1.0 # first scalar multiple on dgemv, dgemm
        double beta = 0.0 # second scalar multiple on dgemv, dgemm
        double gamma = -1.0 # scalar multiple for daxpy
        double delta = -0.5 # scalar multiple for log calculation
    
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
    work = np.empty((ldwork,ldwork), float, order="F")
    ipiv = np.empty((ldwork,ldwork), np.int32, order="F")
    PHT = np.empty((k,n), float, order="F")
    f_inv = np.empty((n,n), float, order="F")
    
    # Initial values
    if beta_tt_init is None:
        #beta_tt[:,0] = np.linalg.inv(np.eye(k) - F).dot(mu) # kxk * kx1 = kx1
        tmp = np.array(np.eye(k), float, order="F") - F
        dgetrf(&k, &k, &tmp[0,0], &ldwork, &ipiv[0,0], &info)
        dgetri(&k, &tmp[0,0], &ldwork, &ipiv[0,0], &work[0,0], &lwork, &info)
        dgemv("N",&k,&k,&alpha,&tmp[0,0],&ldwork,&mu[0],&inc,&beta,&beta_tt[0,0],&inc)
    else:
        beta_tt[::1,0] = beta_tt_init[::1]

    if P_tt_init is None:
        #P_tt[0] = np.linalg.inv(np.eye(k**2) - np.kron(F,F)).dot(Q.reshape(Q.size, 1)).reshape(k,k) # kxk
        tmp = np.array(np.eye(k2) - np.kron(F,F), float, order="F")
        dgetrf(&k2, &k2, &tmp[0,0], &ldwork, &ipiv[0,0], &info)
        dgetri(&k2, &tmp[0,0], &ldwork, &ipiv[0,0], &work[0,0], &lwork, &info)
        dgemv("N",&k2,&k2,&alpha,&tmp[0,0],&ldwork,&Q[0,0],&inc,&beta,&P_tt[0,0,0],&inc)
    else:
        P_tt[::1,:,0] = P_tt_init[::1,:]

    # Redefine the tmp array
    tmp = np.empty((ldwork,ldwork), float, order="F")
    
    # Iterate forwards
    for t in range(1,T+1):
        if time_varying_H:
            H_idx = t
            
        # Prediction
        #beta_tt1[t] = mu + np.dot(F, beta_tt[t-1])
        beta_tt1[::1,t] = mu[::1]
        dgemv("N",&k,&k,&alpha,&F[0,0],&k,&beta_tt[0,t-1],&inc,&alpha,&beta_tt1[0,t],&inc)
        
        #P_tt1[t] = np.dot(F, P_tt[t-1]).dot(F.T) + Q
        P_tt1[::1,:,t] = Q[::1,:]
        dgemm("N", "N", &k, &k, &k, &alpha, &F[0,0], &k, &P_tt[0,0,t-1], &k, &beta, &tmp[0,0], &ldwork)
        dgemm("N", "T", &k, &k, &k, &alpha, &tmp[0,0], &ldwork, &F[0,0], &k, &alpha, &P_tt1[0,0,t], &k)
        
        #y_tt1[t] = np.dot(H[:,:,H_idx], beta_tt1[:,t]) + np.dot(A,z[:,t])
        dgemv("N", &n, &k, &alpha, &H[0,0,H_idx], &n, &beta_tt1[0,t], &inc, &beta, &y_tt1[0,t], &inc)
        dgemv("N", &n, &r, &alpha, &A[0,0], &n, &z[0,t], &inc, &alpha, &y_tt1[0,t], &inc)
        
        #eta_tt1[::1,t] = y[::1,t] - y_tt1[:,t]
        eta_tt1[::1,t] = y[::1,t]
        daxpy(&n, &gamma, &y_tt1[0,t], &inc, &eta_tt1[0,t], &inc)
        
        #PHT = np.dot(P_tt1[t], H[:,:,H_idx].T) # kxn
        #print np.dot(P_tt1[:,:,t], H[:,:,H_idx].T) # taking .T here crashes the program for some reason
        dgemm("N", "T", &k, &n, &k, &alpha, &P_tt1[0,0,t], &k, &H[0,0,H_idx], &n, &beta, &PHT[0,0], &k)
        
        #f_tt1[t] = np.dot(H[:,:,H_idx], PHT) + R
        f_tt1[::1,:,t] = R[::1,:]
        dgemm("N", "N", &n, &n, &k, &alpha, &H[0,0,H_idx], &n, &PHT[0,0], &k, &alpha, &f_tt1[0,0,t], &n)
        
        #f_inv = np.linalg.inv(f_tt1[t])
        f_inv[::1,:] = f_tt1[::1,:,t]
        dgetrf(&n, &n, &f_inv[0,0], &n, &ipiv[0,0], &info)
        det = 1
        for i in range(n):
            if not ipiv[i,0] == i+1:
                det *= -1*f_inv[i,i]
            else:
                det *= f_inv[i,i]
        # Now complete taking the inverse
        dgetri(&n, &f_inv[0,0], &n, &ipiv[0,0], &work[0,0], &lwork, &info)
        
        # Log-likelihood as byproduct
        #ll[t] -0.5*log(2*np.pi*np.linalg.det(f_tt1[:,:,t])) - 0.5*np.dot(np.dot(eta_tt1[:,t].T, f_inv), eta_tt1[:,t])
        # ^ this doesn't work, crashes for some reason; probably related to taking .T as it did above
        ll[t] = -0.5*log(2*np.pi*det)
        dgemv("N",&n,&n,&alpha,&f_inv[0,0],&n,&eta_tt1[0,t],&inc,&beta,&tmp[0,0],&inc)
        ll[t] += -0.5*ddot(&n, &eta_tt1[0,t], &inc, &tmp[0,0], &inc)
    
        # Updating
        #gain[t] = np.dot(PHT, f_inv) # kxn * nxn = kxn
        dgemm("N", "N", &k, &n, &n, &alpha, &PHT[0,0], &k, &f_inv[0,0], &n, &beta, &gain[0,0,t], &k)
        
        #beta_tt[t] = np.dot(gain[:,:,t], eta_tt1[:,t]) + beta_tt1[:,t] # kxn * nx1 + kx1
        beta_tt[::1,t] = beta_tt1[::1,t]
        dgemv("N",&k,&n,&alpha,&gain[0,0,t],&k,&eta_tt1[0,t],&inc,&alpha,&beta_tt[0,t],&inc)
        
        #P_tt[t] =  -1* gain[t].dot(H_view).dot(P_tt1[t]) + P_tt1[t] # kxn * nxk * kxk + kxk
        P_tt[::1,:,t] = P_tt1[::1,:,t]
        dgemm("N", "N", &k, &k, &n, &alpha, &gain[0,0,t], &k, &H[0,0,H_idx], &n, &beta, &tmp[0,0], &ldwork)
        dgemm("N", "N", &k, &k, &k, &gamma, &tmp[0,0], &ldwork, &P_tt1[0,0,t], &k, &alpha, &P_tt[0,0,t], &k)
        
    return beta_tt, P_tt, beta_tt1, P_tt1, y_tt1, eta_tt1, f_tt1, gain, ll

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef kalman_filter_iteration(double [:]       y,  # nx0    (data: endogenous, observed)
                             double [:]       z,  # rx0    (data: weakly exogenous, observed)
                             double [::1,:]   A,  # nxr    (parameters)
                             double [::1,:]   H,  # nxk    (parameters)
                             double [:]       mu, # kx0    (parameters)
                             double [::1,:]   F,  # kxk    (parameters)
                             double [::1,:]   R,  # nxn    (parameters: covariance matrix)
                             double [::1,:]   Q,  # kxk    (parameters: covariance matrix)
                             double [:]       beta_tt,
                             double [::1,:]   P_tt,
                             double [:]       beta_tt1,
                             double [::1,:]   P_tt1,
                             double [:]       y_tt1,
                             double [:]       eta_tt1,
                             double [::1,:]   f_tt1,
                             double [::1,:]   gain,
                             double [::1,:]   tmp,
                             double [::1,:]   work,
                             int    [::1,:]   ipiv,
                             double [::1,:]   PHT,
                             double [::1,:]   f_inv):
    cdef double [::1,:] P_tt_out
    cdef double [:] beta_tt_out
    cdef double det, ll
    cdef:
        int i
        int t
        int T = y.shape[1]-1
        int n = y.shape[0]
        int r = z.shape[0]
        int k = mu.shape[0]
        int time_varying_H = H.shape[2] == T+1
        int H_idx = 0
    cdef:
        int k2 = k**2
        int n2 = n**2
        int info # return code fro dgetri, dgetrf
        int inc = 1 # incrementer for dgemv
        int ldwork = max(k2, n2) # number of rows/columns in work array
        int lwork = ldwork**2 # size of work array for dgetri
    cdef:
        double alpha = 1.0 # first scalar multiple on dgemv, dgemm
        double beta = 0.0 # second scalar multiple on dgemv, dgemm
        double gamma = -1.0 # scalar multiple for daxpy
        double delta = -0.5 # scalar multiple for log calculation

    beta_tt_out = np.empty((k,), float)
    P_tt_out = np.empty((k,k), float)

    # Prediction
    #beta_tt1[t] = mu + np.dot(F, beta_tt[t-1])
    beta_tt1[::1] = mu[::1]
    # Note: the beta_tt passed in the arguments should be from t-1
    dgemv("N",&k,&k,&alpha,&F[0,0],&k,&beta_tt[0],&inc,&alpha,&beta_tt1[0],&inc)

    #P_tt1[t] = np.dot(F, P_tt[t-1]).dot(F.T) + Q
    P_tt1[::1,:] = Q[::1,:]
    # Note: the P_tt passed in the arguments should be from t-1
    dgemm("N", "N", &k, &k, &k, &alpha, &F[0,0], &k, &P_tt[0,0], &k, &beta, &tmp[0,0], &ldwork)
    dgemm("N", "T", &k, &k, &k, &alpha, &tmp[0,0], &ldwork, &F[0,0], &k, &alpha, &P_tt1[0,0], &k)

    #y_tt1[t] = np.dot(H[:,:,H_idx], beta_tt1[:,t]) + np.dot(A,z[:,t])
    dgemv("N", &n, &k, &alpha, &H[0,0], &n, &beta_tt1[0], &k, &beta, &y_tt1[0], &n)
    dgemv("N", &n, &r, &alpha, &A[0,0], &n, &z[0], &r, &alpha, &y_tt1[0], &n)

    #eta_tt1[::1,t] = y[::1,t] - y_tt1[:,t]
    eta_tt1[::1] = y[::1]
    daxpy(&n, &gamma, &y_tt1[0], &inc, &eta_tt1[0], &inc)

    #PHT = np.dot(P_tt1[t], H[:,:,H_idx].T) # kxn
    #print np.dot(P_tt1[:,:,t], H[:,:,H_idx].T) # taking .T here crashes the program for some reason
    dgemm("N", "T", &k, &n, &k, &alpha, &P_tt1[0,0], &k, &H[0,0], &n, &beta, &PHT[0,0], &k)

    #f_tt1[t] = np.dot(H[:,:,H_idx], PHT) + R
    f_tt1[::1,:] = R[::1,:]
    dgemm("N", "N", &n, &n, &k, &alpha, &H[0,0], &n, &PHT[0,0], &k, &alpha, &f_tt1[0,0], &n)

    #f_inv = np.linalg.inv(f_tt1[t])
    f_inv[::1,:] = f_tt1[::1,:]
    dgetrf(&n, &n, &f_inv[0,0], &n, &ipiv[0,0], &info)
    det = 1
    for i in range(n):
        if not ipiv[i,0] == i+1:
            det *= -1*f_inv[i,i]
        else:
            det *= f_inv[i,i]
    # Now complete taking the inverse
    dgetri(&n, &f_inv[0,0], &n, &ipiv[0,0], &work[0,0], &lwork, &info)

    # Log-likelihood as byproduct
    #ll[t] -0.5*log(2*np.pi*np.linalg.det(f_tt1[:,:,t])) - 0.5*np.dot(np.dot(eta_tt1[:,t].T, f_inv), eta_tt1[:,t])
    # ^ this doesn't work, crashes for some reason; probably related to taking .T as it did above
    ll = -0.5*log(2*np.pi*det)
    dgemv("N",&k,&k,&alpha,&f_inv[0,0],&k,&eta_tt1[0],&inc,&beta,&tmp[0,0],&inc)
    ll += -0.5*ddot(&n, &eta_tt1[0], &inc, &tmp[0,0], &inc)

    # Updating
    #gain[t] = np.dot(PHT, f_inv) # kxn * nxn = kxn
    dgemm("N", "N", &k, &n, &n, &alpha, &PHT[0,0], &k, &f_inv[0,0], &n, &beta, &gain[0,0], &k)

    #beta_tt[t] = np.dot(gain[:,:,t], eta_tt1[:,t]) + beta_tt1[:,t] # kxn * nx1 + kx1
    beta_tt_out[::1] = beta_tt1[::1]
    dgemv("N",&k,&n,&alpha,&gain[0,0],&k,&eta_tt1[0],&inc,&alpha,&beta_tt_out[0],&inc)

    #P_tt[t] =  -1* gain[t].dot(H_view).dot(P_tt1[t]) + P_tt1[t] # kxn * nxk * kxk + kxk
    P_tt_out[::1,:] = P_tt1[::1,:]
    dgemm("N", "N", &k, &k, &n, &alpha, &gain[0,0], &k, &H[0,0], &n, &beta, &tmp[0,0], &ldwork)
    dgemm("N", "N", &k, &k, &k, &gamma, &tmp[0,0], &ldwork, &P_tt1[0,0], &k, &alpha, &P_tt_out[0,0], &k)

    return ll, beta_tt_out, P_tt_out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double kalman_filter_loglike(double [::1,:]   y,  # nxT+1    (data: endogenous, observed)
                            double [::1,:]   z,  # rxT+1    (data: weakly exogenous, observed)
                            double [::1,:]   A,  # nxr      (parameters)
                            double [::1,:,:] H,  # nxkxT+1  (parameters)
                            double [:]       mu, # kx0      (parameters)
                            double [::1,:]   F,  # kxk      (parameters)
                            double [::1,:]   R,  # nxn      (parameters: covariance matrix)
                            double [::1,:]   Q): # kxk      (parameters: covariance matrix)

    cdef double [::1,:] beta_tt
    cdef double [::1,:,:] P_tt
    cdef double [::1,:] P_tt_out
    cdef double [:] beta_tt_out
    cdef int [::1,:] ipiv
    cdef double [::1,:] P_tt1, f_tt1, gain, tmp, work, PHT, f_inv
    cdef double [:] ll, beta_tt1, y_tt1, eta_tt1,
    cdef double det
    cdef:
        int i
        int t
        int T = y.shape[1]-1
        int n = y.shape[0]
        int r = z.shape[0]
        int k = mu.shape[0]
        int time_varying_H = H.shape[2] == T+1
        int H_idx = 0
    cdef:
        int k2 = k**2
        int n2 = n**2
        int info # return code fro dgetri, dgetrf
        int inc = 1 # incrementer for dgemv
        int ldwork = max(k2, n2) # number of rows/columns in work array
        int lwork = ldwork**2 # size of work array for dgetri
    cdef:
        double alpha = 1.0 # first scalar multiple on dgemv, dgemm
        double beta = 0.0 # second scalar multiple on dgemv, dgemm
        double gamma = -1.0 # scalar multiple for daxpy
        double delta = -0.5 # scalar multiple for log calculation

    # Allocate memory for variables
    beta_tt = np.zeros((k,T+1), float, order="F")    # T+1xk
    P_tt = np.zeros((k,k,T+1), float, order="F")     # T+1xkxk

    beta_tt1 = np.zeros((k,), float, order="F")      # kx0
    P_tt1 = np.zeros((k,k), float, order="F")        # kxk
    y_tt1 = np.zeros((n), float, order="F")          # nx0
    eta_tt1 = np.zeros((n), float, order="F")        # nx0
    f_tt1 = np.zeros((n,n), float, order="F")        # nxn
    gain = np.zeros((k,k), float, order="F")         # kxk
    ll = np.zeros((T+1,), float)                     # T+1
    tmp = np.empty((ldwork,ldwork), float, order="F")
    work = np.empty((ldwork,ldwork), float, order="F")
    ipiv = np.empty((ldwork,ldwork), np.int32, order="F")
    PHT = np.empty((k,n), float, order="F")
    f_inv = np.empty((k,k), float, order="F")

    # Initial values
    #beta_tt[:,0] = np.linalg.inv(np.eye(k) - F).dot(mu) # kxk * kx1 = kx1
    #print np.linalg.inv(np.eye(k) - F).dot(mu) # kxk * kx1 = kx1
    tmp[:k,:k] = np.eye(k) - F
    dgetrf(&k, &k, &tmp[0,0], &ldwork, &ipiv[0,0], &info)
    dgetri(&k, &tmp[0,0], &ldwork, &ipiv[0,0], &work[0,0], &lwork, &info)
    dgemv("N",&k,&k,&alpha,&tmp[0,0],&ldwork,&mu[0],&inc,&beta,&beta_tt[0,0],&inc)
    #print np.asarray(beta_tt[:,0])

    #P_tt[0] = np.linalg.inv(np.eye(k**2) - np.kron(F,F)).dot(Q.reshape(Q.size, 1)).reshape(k,k) # kxk
    tmp[:k2,:k2] = np.eye(k**2) - np.kron(F,F)
    dgetrf(&k2, &k2, &tmp[0,0], &ldwork, &ipiv[0,0], &info)
    dgetri(&k2, &tmp[0,0], &ldwork, &ipiv[0,0], &work[0,0], &lwork, &info)
    dgemv("N",&k2,&k2,&alpha,&tmp[0,0],&ldwork,&Q[0,0],&inc,&beta,&P_tt[0,0,0],&inc)

    # Iterate forwards
    for t in range(1,T+1):
        if time_varying_H:
            H_idx = t

        ll[t], beta_tt_out, P_tt_out = kalman_filter_iteration(
            y[:,t],  # nx0    (data: endogenous, observed)
            z[:,t],  # rx0    (data: weakly exogenous, observed)
            A,  # nxr    (parameters)
            H[:,:,H_idx],  # nxk    (parameters)
            mu, # kx0    (parameters)
            F,  # kxk    (parameters)
            R,  # nxn    (parameters: covariance matrix)
            Q,  # kxk    (parameters: covariance matrix)
            beta_tt[:,t-1],
            P_tt[:,:,t-1],
            beta_tt1,
            P_tt1,
            y_tt1,
            eta_tt1,
            f_tt1,
            gain,
            tmp,
            work,
            ipiv,
            PHT,
            f_inv
        )

        beta_tt[::1,t] = beta_tt_out[::1]
        P_tt[::1,:,t] = P_tt_out[::1,:]
    return np.sum(ll)


# Kalman Filter
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def naive_kalman_filter(np.ndarray[np.float64_t, ndim=2] y,  # T+1xn    (data: endogenous, observed)
                  np.ndarray[np.float64_t, ndim=2] z,  # T+1xr    (data: weakly exogenous, observed)
                  np.ndarray[np.float64_t, ndim=2] A,  # nxr      (parameters)
                  np.ndarray[np.float64_t, ndim=3] H,  # nxkxT+1  (parameters)
                  np.ndarray[np.float64_t, ndim=2] mu, # kx1      (parameters)
                  np.ndarray[np.float64_t, ndim=2] F,  # kxk      (parameters)
                  np.ndarray[np.float64_t, ndim=2] R,  # kxk      (parameters: covariance matrix)
                  np.ndarray[np.float64_t, ndim=2] G,  # kxg      (parameters)
                  np.ndarray[np.float64_t, ndim=2] Q): # gxg      (parameters: covariance matrix)
    
    cdef np.ndarray[np.float64_t, ndim=3] beta_tt, P_tt, beta_tt1, P_tt1, f_tt1, gain
    cdef np.ndarray[np.float64_t, ndim=2] y_tt1, eta_tt1, PHT, f_inv
    cdef np.ndarray[np.float64_t, ndim=1] ll
    cdef int T = y.shape[0]-1, n = y.shape[1], r = z.shape[1], k = mu.shape[0], g = G.shape[1], t
    cdef int time_varying_H = H.shape[2] == T+1
    cdef double [:, :] H_view
    
    # Allocate memory for variables
    beta_tt = np.zeros((T+1,k,1))  # T+1xkx1
    P_tt = np.zeros((T+1,k,k))     # T+1xkxk
    beta_tt1 = np.zeros((T+1,k,k)) # T+1xkxk
    P_tt1 = np.zeros((T+1,k,k))    # T+1xkxk
    y_tt1 = np.zeros((T+1,n))      # T+1xn
    eta_tt1 = np.zeros((T+1,n))    # T+1xn
    f_tt1 = np.zeros((T+1,k,k))    # T+1xkxk
    gain = np.zeros((T+1,k,k))     # T+1xkxk
    ll = np.zeros((T+1,))          # T+1
    PHT = np.zeros((k,n))
    f_inv = np.zeros((k,k))
    
    # Initial values
    beta_tt[0] = np.linalg.inv(np.eye(k) - F).dot(mu) # kxk * kx1 = kx1
    #vec_Q = Q.reshape(Q.size, 1)
    P_tt[0] = np.linalg.inv(np.eye(k**2) - np.kron(F,F)).dot(Q.reshape(Q.size, 1)).reshape(k,k) # kxk
    #P_tt[0] = vec_P_00.reshape(k,k)
    
    # Calculations out of the loop
    # - Maybe can precalculate Az_t? but maybe no benefit, since z_t is T+1 x r rather than 1 x r..

    # Iterate forwards
    H_view = H[:,:,0]
    for t in range(1,T+1):
        if time_varying_H:
            H_view = H[:,:,t]
        
        # Prediction
        beta_tt1[t] = mu + F.dot(beta_tt[t-1])
        P_tt1[t]    = F.dot(P_tt[t-1]).dot(F.T) + Q
        y_tt1[t]    = np.dot(H_view, beta_tt1[t]) + A.dot(z[t])
        eta_tt1[t]  = y[t] - y_tt1[t]
        PHT = P_tt1[t].dot(H_view.T)
        f_tt1[t]    = np.dot(H_view, PHT) + R
        f_inv = np.linalg.inv(f_tt1[t])
        
        # Log-likelihood as byproduct
        ll[t] = -0.5*log(2*np.pi*np.linalg.det(f_tt1[t])) - 0.5*eta_tt1[t].T.dot(f_inv).dot(eta_tt1[t])
        
        # Updating
        gain[t] = PHT.dot(f_inv)
        beta_tt[t] = beta_tt1[t] + gain[t].dot(eta_tt1[t])
        P_tt[t] = P_tt1[t] - gain[t].dot(H_view).dot(P_tt1[t])
    
    return beta_tt, P_tt, beta_tt1, P_tt1, y_tt1, eta_tt1, f_tt1, gain, ll