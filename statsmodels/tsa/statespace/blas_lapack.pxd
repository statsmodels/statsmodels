cimport numpy as np

ctypedef int ssymm_t(
    # Compute C := alpha*A*B + beta*C,
    char *side,  # {'L','R'}
    char *uplo,  # {'U','L'}
    int *m,        # Rows of C
    int *n,        # Columns C
    np.float32_t *alpha, # Scalar multiple
    np.float32_t *a,     # Matrix A: mxk
    int *lda,      # The size of the first dimension of A (in memory)
    np.float32_t *b,     # Matrix B: kxn
    int *ldb,      # The size of the first dimension of B (in memory)
    np.float32_t *beta,  # Scalar multiple
    np.float32_t *c,     # Matrix C: mxn
    int *ldc       # The size of the first dimension of C (in memory)
)

ctypedef int sgemm_t(
    # Compute C := alpha*A*B + beta*C
    char *transa,  # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    char *transb,  # {'T','C'}: o(B)=B'; {'N'}: o(B)=B
    int *m,        # Rows of o(A)    (and of C)
    int *n,        # Columns of o(B) (and of C)
    int *k,        # Columns of o(A) / Rows of o(B)
    np.float32_t *alpha, # Scalar multiple
    np.float32_t *a,     # Matrix A: mxk
    int *lda,      # The size of the first dimension of A (in memory)
    np.float32_t *b,     # Matrix B: kxn
    int *ldb,      # The size of the first dimension of B (in memory)
    np.float32_t *beta,  # Scalar multiple
    np.float32_t *c,     # Matrix C: mxn
    int *ldc       # The size of the first dimension of C (in memory)
)

ctypedef int sgemv_t(
    # Compute C := alpha*A*x + beta*y
    char *trans,   # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    int *m,        # Rows of o(A)
    int *n,        # Columns of o(A) / min(len(x))
    np.float32_t *alpha, # Scalar multiple
    np.float32_t *a,     # Matrix A: mxn
    int *lda,      # The size of the first dimension of A (in memory)
    np.float32_t *x,     # Vector x, min(len(x)) = n
    int *incx,     # The increment between elements of x (usually 1)
    np.float32_t *beta,  # Scalar multiple
    np.float32_t *y,     # Vector y, min(len(y)) = m
    int *incy      # The increment between elements of y (usually 1)
)

ctypedef int scopy_t(
    int *n,        # Number of vector elements to be copied.
    np.float32_t *x,     # Vector from which to copy.
    int *incx,     # Increment between elements of x.
    np.float32_t *y,     # array of dimension (n-1) * |incy| + 1, result vector.
    int *incy      # Increment between elements of y.
)

ctypedef int saxpy_t(
    # Compute y := alpha*x + y
    int *n,        # Columns of o(A) / min(len(x))
    np.float32_t *alpha, # Scalar multiple
    np.float32_t *x,     # Vector x, min(len(x)) = n
    int *incx,     # The increment between elements of x (usually 1)
    np.float32_t *y,     # Vector y, min(len(y)) = m
    int *incy      # The increment between elements of y (usually 1)
)

ctypedef np.float32_t sdot_t(
    # Compute DDOT := x.T * y
    int *n,        # Length of vectors
    np.float32_t *x,     # Vector x, min(len(x)) = n
    int *incx,     # The increment between elements of x (usually 1)
    np.float32_t *y,     # Vector y, min(len(y)) = m
    int *incy      # The increment between elements of y (usually 1)
)


ctypedef int sgetrf_t(
    int *m,        # Rows of A
    int *n,        # Columns of A
    np.float32_t *a,     # Matrix A: mxn
    int *lda,      # The size of the first dimension of A (in memory)
    int *ipiv,     # Matrix P: mxn (the pivot indices)
    int *info      # 0 if success, otherwise an error code (integer)
)

ctypedef int sgetri_t(
    int *n,        # Order of A
    np.float32_t *a,     # Matrix A: nxn (the LUP decomposed matrix from dgetrf)
    int *lda,      # The size of the first dimension of A (in memory)
    int *ipiv,     # Matrix P: nxn (the pivot indices from the LUP decomposition)
    np.float32_t *work,  # Matrix: nxn (a workspace for the inversion, optimal size=nxn)
    int *lwork,    # Number of elements in the workspace: optimal is n**2
    int *info      # 0 if success, otherwise an error code (integer)
)

ctypedef int spotrf_t(
    # Compute the Cholesky factorization of a
    # real  symmetric positive definite matrix A
    char *uplo,    # 'U':  A = U'U and U is stored, 'L': A = LL' and L is stored
    int *n,        #  The order of the matrix A.  n >= 0.
    np.float32_t *a,    # Matrix A: nxn
    int *lda,      # The size of the first dimension of A (in memory)
    int *info      # 0 if success, otherwise an error code (integer)
)


ctypedef int dsymm_t(
    # Compute C := alpha*A*B + beta*C,
    char *side,  # {'L','R'}
    char *uplo,  # {'U','L'}
    int *m,        # Rows of C
    int *n,        # Columns C
    double *alpha, # Scalar multiple
    double *a,     # Matrix A: mxk
    int *lda,      # The size of the first dimension of A (in memory)
    double *b,     # Matrix B: kxn
    int *ldb,      # The size of the first dimension of B (in memory)
    double *beta,  # Scalar multiple
    double *c,     # Matrix C: mxn
    int *ldc       # The size of the first dimension of C (in memory)
)

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
    # Compute y := alpha*A*x + beta*y
    char *trans,   # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    int *m,        # Rows of A (prior to transpose from *trans)
    int *n,        # Columns of A / min(len(x))
    double *alpha, # Scalar multiple
    double *a,     # Matrix A: mxn
    int *lda,      # The size of the first dimension of A (in memory)
    double *x,     # Vector x, min(len(x)) = n
    int *incx,     # The increment between elements of x (usually 1)
    double *beta,  # Scalar multiple
    double *y,     # Vector y, min(len(y)) = m
    int *incy      # The increment between elements of y (usually 1)
)

ctypedef int dcopy_t(
    int *n,        # Number of vector elements to be copied.
    double *x,     # Vector from which to copy.
    int *incx,     # Increment between elements of x.
    double *y,     # array of dimension (n-1) * |incy| + 1, result vector.
    int *incy      # Increment between elements of y.
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

ctypedef int dpotrf_t(
    # Compute the Cholesky factorization of a
    # real  symmetric positive definite matrix A
    char *uplo,    # 'U':  A = U'U and U is stored, 'L': A = LL' and L is stored
    int *n,        #  The order of the matrix A.  n >= 0.
    double *a,     # Matrix A: nxn
    int *lda,      # The size of the first dimension of A (in memory)
    int *info      # 0 if success, otherwise an error code (integer)
)

ctypedef int csymm_t(
    # Compute C := alpha*A*B + beta*C,
    char *side,  # {'L','R'}
    char *uplo,  # {'U','L'}
    int *m,        # Rows of C
    int *n,        # Columns C
    np.complex64_t *alpha, # Scalar multiple
    np.complex64_t *a,     # Matrix A: mxk
    int *lda,      # The size of the first dimension of A (in memory)
    np.complex64_t *b,     # Matrix B: kxn
    int *ldb,      # The size of the first dimension of B (in memory)
    np.complex64_t *beta,  # Scalar multiple
    np.complex64_t *c,     # Matrix C: mxn
    int *ldc       # The size of the first dimension of C (in memory)
)

ctypedef int cgemm_t(
    # Compute C := alpha*A*B + beta*C
    char *transa,   # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    char *transb,   # {'T','C'}: o(B)=B'; {'N'}: o(B)=B
    int *m,         # Rows of o(A)    (and of C)
    int *n,         # Columns of o(B) (and of C)
    int *k,         # Columns of o(A) / Rows of o(B)
    np.complex64_t *alpha, # Scalar multiple
    np.complex64_t *a,     # Matrix A: mxk
    int *lda,       # The size of the first dimension of A (in memory)
    np.complex64_t *b,     # Matrix B: kxn
    int *ldb,       # The size of the first dimension of B (in memory)
    np.complex64_t *beta,  # Scalar multiple
    np.complex64_t *c,     # Matrix C: mxn
    int *ldc        # The size of the first dimension of C (in memory)
)

ctypedef int cgemv_t(
    # Compute C := alpha*A*x + beta*y
    char *trans,    # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    int *m,         # Rows of o(A)
    int *n,         # Columns of o(A) / min(len(x))
    np.complex64_t *alpha, # Scalar multiple
    np.complex64_t *a,     # Matrix A: mxn
    int *lda,       # The size of the first dimension of A (in memory)
    np.complex64_t *x,     # Vector x, min(len(x)) = n
    int *incx,      # The increment between elements of x (usually 1)
    np.complex64_t *beta,  # Scalar multiple
    np.complex64_t *y,     # Vector y, min(len(y)) = m
    int *incy       # The increment between elements of y (usually 1)
)

ctypedef int ccopy_t(
    int *n,        # Number of vector elements to be copied.
    np.complex64_t *x,     # Vector from which to copy.
    int *incx,     # Increment between elements of x.
    np.complex64_t *y,     # array of dimension (n-1) * |incy| + 1, result vector.
    int *incy      # Increment between elements of y.
)

ctypedef int caxpy_t(
    # Compute y := alpha*x + y
    int *n,        # Columns of o(A) / min(len(x))
    np.complex64_t *alpha, # Scalar multiple
    np.complex64_t *x,     # Vector x, min(len(x)) = n
    int *incx,     # The increment between elements of x (usually 1)
    np.complex64_t *y,     # Vector y, min(len(y)) = m
    int *incy      # The increment between elements of y (usually 1)
)

ctypedef int cgetrf_t(
    int *m,        # Rows of A
    int *n,        # Columns of A
    np.complex64_t *a,    # Matrix A: mxn
    int *lda,      # The size of the first dimension of A (in memory)
    int *ipiv,     # Matrix P: mxn (the pivot indices)
    int *info      # 0 if success, otherwise an error code (integer)
)

ctypedef int cgetri_t(
    int *n,        # Order of A
    np.complex64_t *a,    # Matrix A: nxn (the LUP decomposed matrix from dgetrf)
    int *lda,      # The size of the first dimension of A (in memory)
    int *ipiv,     # Matrix P: nxn (the pivot indices from the LUP decomposition)
    np.complex64_t *work, # Matrix: nxn (a workspace for the inversion, optimal size=nxn)
    int *lwork,    # Number of elements in the workspace: optimal is n**2
    int *info      # 0 if success, otherwise an error code (integer)
)

ctypedef int cpotrf_t(
    # Compute the Cholesky factorization of a
    # real  symmetric positive definite matrix A
    char *uplo,    # 'U':  A = U'U and U is stored, 'L': A = LL' and L is stored
    int *n,        #  The order of the matrix A.  n >= 0.
    np.complex64_t *a,     # Matrix A: nxn
    int *lda,      # The size of the first dimension of A (in memory)
    int *info      # 0 if success, otherwise an error code (integer)
)

ctypedef int zsymm_t(
    # Compute C := alpha*A*B + beta*C,
    char *side,  # {'L','R'}
    char *uplo,  # {'U','L'}
    int *m,        # Rows of C
    int *n,        # Columns C
    complex *alpha, # Scalar multiple
    complex *a,     # Matrix A: mxk
    int *lda,      # The size of the first dimension of A (in memory)
    complex *b,     # Matrix B: kxn
    int *ldb,      # The size of the first dimension of B (in memory)
    complex *beta,  # Scalar multiple
    complex *c,     # Matrix C: mxn
    int *ldc       # The size of the first dimension of C (in memory)
)

ctypedef int zgemm_t(
    # Compute C := alpha*A*B + beta*C
    char *transa,   # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    char *transb,   # {'T','C'}: o(B)=B'; {'N'}: o(B)=B
    int *m,         # Rows of o(A)    (and of C)
    int *n,         # Columns of o(B) (and of C)
    int *k,         # Columns of o(A) / Rows of o(B)
    complex *alpha, # Scalar multiple
    complex *a,     # Matrix A: mxk
    int *lda,       # The size of the first dimension of A (in memory)
    complex *b,     # Matrix B: kxn
    int *ldb,       # The size of the first dimension of B (in memory)
    complex *beta,  # Scalar multiple
    complex *c,     # Matrix C: mxn
    int *ldc        # The size of the first dimension of C (in memory)
)

ctypedef int zgemv_t(
    # Compute C := alpha*A*x + beta*y
    char *trans,    # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    int *m,         # Rows of o(A)
    int *n,         # Columns of o(A) / min(len(x))
    complex *alpha, # Scalar multiple
    complex *a,     # Matrix A: mxn
    int *lda,       # The size of the first dimension of A (in memory)
    complex *x,     # Vector x, min(len(x)) = n
    int *incx,      # The increment between elements of x (usually 1)
    complex *beta,  # Scalar multiple
    complex *y,     # Vector y, min(len(y)) = m
    int *incy       # The increment between elements of y (usually 1)
)

ctypedef int zcopy_t(
    int *n,        # Number of vector elements to be copied.
    complex *x,     # Vector from which to copy.
    int *incx,     # Increment between elements of x.
    complex *y,     # array of dimension (n-1) * |incy| + 1, result vector.
    int *incy      # Increment between elements of y.
)

ctypedef int zaxpy_t(
    # Compute y := alpha*x + y
    int *n,        # Columns of o(A) / min(len(x))
    complex *alpha, # Scalar multiple
    complex *x,     # Vector x, min(len(x)) = n
    int *incx,     # The increment between elements of x (usually 1)
    complex *y,     # Vector y, min(len(y)) = m
    int *incy      # The increment between elements of y (usually 1)
)

ctypedef int zgetrf_t(
    int *m,        # Rows of A
    int *n,        # Columns of A
    complex *a,    # Matrix A: mxn
    int *lda,      # The size of the first dimension of A (in memory)
    int *ipiv,     # Matrix P: mxn (the pivot indices)
    int *info      # 0 if success, otherwise an error code (integer)
)

ctypedef int zgetri_t(
    int *n,        # Order of A
    complex *a,    # Matrix A: nxn (the LUP decomposed matrix from dgetrf)
    int *lda,      # The size of the first dimension of A (in memory)
    int *ipiv,     # Matrix P: nxn (the pivot indices from the LUP decomposition)
    complex *work, # Matrix: nxn (a workspace for the inversion, optimal size=nxn)
    int *lwork,    # Number of elements in the workspace: optimal is n**2
    int *info      # 0 if success, otherwise an error code (integer)
)

ctypedef int zpotrf_t(
    # Compute the Cholesky factorization of a
    # real  symmetric positive definite matrix A
    char *uplo,    # 'U':  A = U'U and U is stored, 'L': A = LL' and L is stored
    int *n,        #  The order of the matrix A.  n >= 0.
    complex *a,    # Matrix A: nxn
    int *lda,      # The size of the first dimension of A (in memory)
    int *info      # 0 if success, otherwise an error code (integer)
)
