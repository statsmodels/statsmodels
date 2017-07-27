cimport numpy as np

#
# BLAS
#

ctypedef int sgemm_t(
    # Compute C := alpha*A*B + beta*C
    char *transa,         # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    char *transb,         # {'T','C'}: o(B)=B'; {'N'}: o(B)=B
    int *m,               # Rows of o(A)    (and of C)
    int *n,               # Columns of o(B) (and of C)
    int *k,               # Columns of o(A) / Rows of o(B)
    np.float32_t *alpha,  # Scalar multiple
    np.float32_t *a,      # Matrix A: mxk
    int *lda,             # The size of the first dimension of A (in memory)
    np.float32_t *b,      # Matrix B: kxn
    int *ldb,             # The size of the first dimension of B (in memory)
    np.float32_t *beta,   # Scalar multiple
    np.float32_t *c,      # Matrix C: mxn
    int *ldc              # The size of the first dimension of C (in memory)
)

ctypedef int sgemv_t(
    # Compute C := alpha*A*x + beta*y
    char *trans,          # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    int *m,               # Rows of o(A)
    int *n,               # Columns of o(A) / min(len(x))
    np.float32_t *alpha,  # Scalar multiple
    np.float32_t *a,      # Matrix A: mxn
    int *lda,             # The size of the first dimension of A (in memory)
    np.float32_t *x,      # Vector x, min(len(x)) = n
    int *incx,            # The increment between elements of x (usually 1)
    np.float32_t *beta,   # Scalar multiple
    np.float32_t *y,      # Vector y, min(len(y)) = m
    int *incy             # The increment between elements of y (usually 1)
)

ctypedef int ssymm_t(
    # SSYMM - perform one of the matrix-matrix operations   C :=
    # alpha*A*B + beta*C,
    char *side,           # {'L', 'R'}: left, right
    char *uplo,           # {'U','L'}, upper, lower
    int *m,               # Rows of C
    int *n,               # Columns of C
    np.float32_t *alpha,  # Scalar multiple
    np.float32_t *a,      # Matrix A
    int *lda,             # The size of the first dimension of A (in memory)
    np.float32_t *b,      # Matrix B
    int *ldb,             # The size of the first dimension of B (in memory)
    np.float32_t *beta,   # Scalar multiple
    np.float32_t *c,      # Matrix C
    int *ldc,             # The size of the first dimension of C (in memory)
)

ctypedef int ssymv_t(
    # SSYMV - perform the matrix-vector operation   y := alpha*A*x
    # + beta*y,
    char *uplo,           # {'U','L'}, upper, lower
    int *n,               # Order of matrix A
    np.float32_t *alpha,  # Scalar multiple
    np.float32_t *a,      # Matrix A: mxn
    int *lda,             # The size of the first dimension of A (in memory)
    np.float32_t *x,      # Vector x, min(len(x)) = n
    int *incx,            # The increment between elements of x (usually 1)
    np.float32_t *beta,   # Scalar multiple
    np.float32_t *y,      # Vector y, min(len(y)) = n
    int *incy,            # The increment between elements of y (usually 1)
)

ctypedef int strmm_t(
    # STRMM - perform one of the matrix-matrix operations   B :=
    # alpha*op( A )*B, or B := alpha*B*op( A ),
    char *side,           # {'L', 'R'}: left, right
    char *uplo,           # {'U','L'}, upper, lower
    char *transa,         # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    char *diag,           # {'U','N'}: unit triangular or not
    int *m,               # Rows of B
    int *n,               # Columns of B
    np.float32_t *alpha,  # Scalar multiple
    np.float32_t *a,      # Matrix A
    int *lda,             # The size of the first dimension of A (in memory)
    np.float32_t *b,      # Matrix B
    int *ldb,             # The size of the first dimension of B (in memory)
)

ctypedef int strmv_t(
    # STRMV - perform one of the matrix-vector operations   x :=
    # A*x, or x := A'*x,
    char *uplo,           # {'U','L'}, upper, lower
    char *trans,          # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    char *diag,           # {'U','N'}: unit triangular or not
    int *n,               # Order of matrix A
    np.float32_t *a,      # Matrix A: mxn
    int *lda,             # The size of the first dimension of A (in memory)
    np.float32_t *x,      # Vector x, min(len(x)) = n
    int *incx,            # The increment between elements of x (usually 1)
)

ctypedef int scopy_t(
    int *n,           # Number of vector elements to be copied.
    np.float32_t *x,  # Vector from which to copy.
    int *incx,        # Increment between elements of x.
    np.float32_t *y,  # array of dimension (n-1) * |incy| + 1, result vector.
    int *incy         # Increment between elements of y.
)

ctypedef int sscal_t(
    # SSCAL - BLAS level one, scales a double precision vector
    int *n,               # Number of elements in the vector.
    np.float32_t *alpha,  # scalar alpha
    np.float32_t *x,      # Array of dimension (n-1) * |incx| + 1. Vector to be scaled.
    int *incx             # Increment between elements of x.
)

ctypedef int saxpy_t(
    # Compute y := alpha*x + y
    int *n,               # Columns of o(A) / min(len(x))
    np.float32_t *alpha,  # Scalar multiple
    np.float32_t *x,      # Vector x, min(len(x)) = n
    int *incx,            # The increment between elements of x (usually 1)
    np.float32_t *y,      # Vector y, min(len(y)) = m
    int *incy             # The increment between elements of y (usually 1)
)

ctypedef int sswap_t(
    # Swap y and x
    int *n,               # Columns of o(A) / min(len(x))
    np.float32_t *x,      # Vector x, min(len(x)) = n
    int *incx,            # The increment between elements of x (usually 1)
    np.float32_t *y,      # Vector y, min(len(y)) = m
    int *incy             # The increment between elements of y (usually 1)
)

ctypedef np.float64_t sdot_t(
    # Compute DDOT := x.T * y
    int *n,           # Length of vectors
    np.float32_t *x,  # Vector x, min(len(x)) = n
    int *incx,        # The increment between elements of x (usually 1)
    np.float32_t *y,  # Vector y, min(len(y)) = m
    int *incy         # The increment between elements of y (usually 1)
)

ctypedef int dgemm_t(
    # Compute C := alpha*A*B + beta*C
    char *transa,        # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    char *transb,        # {'T','C'}: o(B)=B'; {'N'}: o(B)=B
    int *m,              # Rows of o(A)    (and of C)
    int *n,              # Columns of o(B) (and of C)
    int *k,              # Columns of o(A) / Rows of o(B)
    np.float64_t *alpha, # Scalar multiple
    np.float64_t *a,     # Matrix A: mxk
    int *lda,            # The size of the first dimension of A (in memory)
    np.float64_t *b,     # Matrix B: kxn
    int *ldb,            # The size of the first dimension of B (in memory)
    np.float64_t *beta,  # Scalar multiple
    np.float64_t *c,     # Matrix C: mxn
    int *ldc             # The size of the first dimension of C (in memory)
)

ctypedef int dgemv_t(
    # Compute y := alpha*A*x + beta*y
    char *trans,         # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    int *m,              # Rows of A (prior to transpose from *trans)
    int *n,              # Columns of A / min(len(x))
    np.float64_t *alpha, # Scalar multiple
    np.float64_t *a,     # Matrix A: mxn
    int *lda,            # The size of the first dimension of A (in memory)
    np.float64_t *x,     # Vector x, min(len(x)) = n
    int *incx,           # The increment between elements of x (usually 1)
    np.float64_t *beta,  # Scalar multiple
    np.float64_t *y,     # Vector y, min(len(y)) = m
    int *incy            # The increment between elements of y (usually 1)
)

ctypedef int dsymm_t(
    # DSYMM - perform one of the matrix-matrix operations   C :=
    # alpha*A*B + beta*C,
    char *side,           # {'L', 'R'}: left, right
    char *uplo,           # {'U','L'}, upper, lower
    int *m,               # Rows of C
    int *n,               # Columns of C
    np.float64_t *alpha,  # Scalar multiple
    np.float64_t *a,      # Matrix A
    int *lda,             # The size of the first dimension of A (in memory)
    np.float64_t *b,      # Matrix B
    int *ldb,             # The size of the first dimension of B (in memory)
    np.float64_t *beta,   # Scalar multiple
    np.float64_t *c,      # Matrix C
    int *ldc,             # The size of the first dimension of C (in memory)
)

ctypedef int dsymv_t(
    # DSYMV - perform the matrix-vector operation   y := alpha*A*x
    # + beta*y,
    char *uplo,           # {'U','L'}, upper, lower
    int *n,               # Order of matrix A
    np.float64_t *alpha,  # Scalar multiple
    np.float64_t *a,      # Matrix A: mxn
    int *lda,             # The size of the first dimension of A (in memory)
    np.float64_t *x,      # Vector x, min(len(x)) = n
    int *incx,            # The increment between elements of x (usually 1)
    np.float64_t *beta,   # Scalar multiple
    np.float64_t *y,      # Vector y, min(len(y)) = n
    int *incy,            # The increment between elements of y (usually 1)
)

ctypedef int dtrmm_t(
    # DTRMM - perform one of the matrix-matrix operations   B :=
    # alpha*op( A )*B, or B := alpha*B*op( A ),
    char *side,           # {'L', 'R'}: left, right
    char *uplo,           # {'U','L'}, upper, lower
    char *transa,         # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    char *diag,           # {'U','N'}: unit triangular or not
    int *m,               # Rows of B
    int *n,               # Columns of B
    np.float64_t *alpha,  # Scalar multiple
    np.float64_t *a,      # Matrix A
    int *lda,             # The size of the first dimension of A (in memory)
    np.float64_t *b,      # Matrix B
    int *ldb,             # The size of the first dimension of B (in memory)
)

ctypedef int dtrmv_t(
    # DTRMV - perform one of the matrix-vector operations   x :=
    # A*x, or x := A'*x,
    char *uplo,           # {'U','L'}, upper, lower
    char *trans,          # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    char *diag,           # {'U','N'}: unit triangular or not
    int *n,               # Order of matrix A
    np.float64_t *a,      # Matrix A: mxn
    int *lda,             # The size of the first dimension of A (in memory)
    np.float64_t *x,      # Vector x, min(len(x)) = n
    int *incx,            # The increment between elements of x (usually 1)
)

ctypedef int dcopy_t(
    int *n,              # Number of vector elements to be copied.
    np.float64_t *x,     # Vector from which to copy.
    int *incx,           # Increment between elements of x.
    np.float64_t *y,     # array of dimension (n-1) * |incy| + 1, result vector.
    int *incy            # Increment between elements of y.
)

ctypedef int dscal_t(
    # DSCAL - BLAS level one, scales a double precision vector
    int *n,               # Number of elements in the vector.
    np.float64_t *alpha,  # scalar alpha
    np.float64_t *x,      # Array of dimension (n-1) * |incx| + 1. Vector to be scaled.
    int *incx             # Increment between elements of x.
)

ctypedef int daxpy_t(
    # Compute y := alpha*x + y
    int *n,              # Columns of o(A) / min(len(x))
    np.float64_t *alpha, # Scalar multiple
    np.float64_t *x,     # Vector x, min(len(x)) = n
    int *incx,           # The increment between elements of x (usually 1)
    np.float64_t *y,     # Vector y, min(len(y)) = m
    int *incy            # The increment between elements of y (usually 1)
)

ctypedef int dswap_t(
    # Swap y and x
    int *n,               # Columns of o(A) / min(len(x))
    np.float64_t *x,      # Vector x, min(len(x)) = n
    int *incx,            # The increment between elements of x (usually 1)
    np.float64_t *y,      # Vector y, min(len(y)) = m
    int *incy             # The increment between elements of y (usually 1)
)

ctypedef double ddot_t(
    # Compute DDOT := x.T * y
    int *n,              # Length of vectors
    np.float64_t *x,     # Vector x, min(len(x)) = n
    int *incx,           # The increment between elements of x (usually 1)
    np.float64_t *y,     # Vector y, min(len(y)) = m
    int *incy            # The increment between elements of y (usually 1)
)

ctypedef int cgemm_t(
    # Compute C := alpha*A*B + beta*C
    char *transa,           # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    char *transb,           # {'T','C'}: o(B)=B'; {'N'}: o(B)=B
    int *m,                 # Rows of o(A)    (and of C)
    int *n,                 # Columns of o(B) (and of C)
    int *k,                 # Columns of o(A) / Rows of o(B)
    np.complex64_t *alpha,  # Scalar multiple
    np.complex64_t *a,      # Matrix A: mxk
    int *lda,               # The size of the first dimension of A (in memory)
    np.complex64_t *b,      # Matrix B: kxn
    int *ldb,               # The size of the first dimension of B (in memory)
    np.complex64_t *beta,   # Scalar multiple
    np.complex64_t *c,      # Matrix C: mxn
    int *ldc                # The size of the first dimension of C (in memory)
)

ctypedef int cgemv_t(
    # Compute C := alpha*A*x + beta*y
    char *trans,            # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    int *m,                 # Rows of o(A)
    int *n,                 # Columns of o(A) / min(len(x))
    np.complex64_t *alpha,  # Scalar multiple
    np.complex64_t *a,      # Matrix A: mxn
    int *lda,               # The size of the first dimension of A (in memory)
    np.complex64_t *x,      # Vector x, min(len(x)) = n
    int *incx,              # The increment between elements of x (usually 1)
    np.complex64_t *beta,   # Scalar multiple
    np.complex64_t *y,      # Vector y, min(len(y)) = m
    int *incy               # The increment between elements of y (usually 1)
)

ctypedef int csymm_t(
    # CSYMM - perform one of the matrix-matrix operations   C :=
    # alpha*A*B + beta*C,
    char *side,             # {'L', 'R'}: left, right
    char *uplo,             # {'U','L'}, upper, lower
    int *m,                 # Rows of C
    int *n,                 # Columns of C
    np.complex64_t *alpha,  # Scalar multiple
    np.complex64_t *a,      # Matrix A
    int *lda,               # The size of the first dimension of A (in memory)
    np.complex64_t *b,      # Matrix B
    int *ldb,               # The size of the first dimension of B (in memory)
    np.complex64_t *beta,   # Scalar multiple
    np.complex64_t *c,      # Matrix C
    int *ldc,               # The size of the first dimension of C (in memory)
)

ctypedef int csymv_t(
    # CSYMV - perform the matrix-vector operation   y := alpha*A*x
    # + beta*y,
    char *uplo,             # {'U','L'}, upper, lower
    int *n,                 # Order of matrix A
    np.complex64_t *alpha,  # Scalar multiple
    np.complex64_t *a,      # Matrix A: mxn
    int *lda,               # The size of the first dimension of A (in memory)
    np.complex64_t *x,      # Vector x, min(len(x)) = n
    int *incx,              # The increment between elements of x (usually 1)
    np.complex64_t *beta,   # Scalar multiple
    np.complex64_t *y,      # Vector y, min(len(y)) = n
    int *incy,              # The increment between elements of y (usually 1)
)

ctypedef int ctrmm_t(
    # CTRMM - perform one of the matrix-matrix operations   B :=
    # alpha*op( A )*B, or B := alpha*B*op( A ),
    char *side,             # {'L', 'R'}: left, right
    char *uplo,             # {'U','L'}, upper, lower
    char *transa,           # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    char *diag,             # {'U','N'}: unit triangular or not
    int *m,                 # Rows of B
    int *n,                 # Columns of B
    np.complex64_t *alpha,  # Scalar multiple
    np.complex64_t *a,      # Matrix A
    int *lda,               # The size of the first dimension of A (in memory)
    np.complex64_t *b,      # Matrix B
    int *ldb,               # The size of the first dimension of B (in memory)
)

ctypedef int ctrmv_t(
    # CTRMV - perform one of the matrix-vector operations   x :=
    # A*x, or x := A'*x,
    char *uplo,           # {'U','L'}, upper, lower
    char *trans,          # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    char *diag,           # {'U','N'}: unit triangular or not
    int *n,               # Order of matrix A
    np.complex64_t *a,    # Matrix A: mxn
    int *lda,             # The size of the first dimension of A (in memory)
    np.complex64_t *x,    # Vector x, min(len(x)) = n
    int *incx,            # The increment between elements of x (usually 1)
)

ctypedef int ccopy_t(
    int *n,             # Number of vector elements to be copied.
    np.complex64_t *x,  # Vector from which to copy.
    int *incx,          # Increment between elements of x.
    np.complex64_t *y,  # array of dimension (n-1) * |incy| + 1, result vector.
    int *incy           # Increment between elements of y.
)

ctypedef int cscal_t(
    # CSCAL - BLAS level one, scales a double precision vector
    int *n,                 # Number of elements in the vector.
    np.complex64_t *alpha,  # scalar alpha
    np.complex64_t *x,      # Array of dimension (n-1) * |incx| + 1. Vector to be scaled.
    int *incx               # Increment between elements of x.
)

ctypedef int caxpy_t(
    # Compute y := alpha*x + y
    int *n,                 # Columns of o(A) / min(len(x))
    np.complex64_t *alpha,  # Scalar multiple
    np.complex64_t *x,      # Vector x, min(len(x)) = n
    int *incx,              # The increment between elements of x (usually 1)
    np.complex64_t *y,      # Vector y, min(len(y)) = m
    int *incy               # The increment between elements of y (usually 1)
)

ctypedef int cswap_t(
    # Swap y and x
    int *n,               # Columns of o(A) / min(len(x))
    np.complex64_t *x,      # Vector x, min(len(x)) = n
    int *incx,            # The increment between elements of x (usually 1)
    np.complex64_t *y,      # Vector y, min(len(y)) = m
    int *incy             # The increment between elements of y (usually 1)
)

ctypedef np.complex64_t cdotu_t(
    # Compute CDOTU := x.T * y
    int *n,             # Length of vectors
    np.complex64_t *x,  # Vector x, min(len(x)) = n
    int *incx,          # The increment between elements of x (usually 1)
    np.complex64_t *y,  # Vector y, min(len(y)) = m
    int *incy           # The increment between elements of y (usually 1)
)

ctypedef int zgemm_t(
    # Compute C := alpha*A*B + beta*C
    char *transa,           # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    char *transb,           # {'T','C'}: o(B)=B'; {'N'}: o(B)=B
    int *m,                 # Rows of o(A)    (and of C)
    int *n,                 # Columns of o(B) (and of C)
    int *k,                 # Columns of o(A) / Rows of o(B)
    np.complex128_t *alpha, # Scalar multiple
    np.complex128_t *a,     # Matrix A: mxk
    int *lda,               # The size of the first dimension of A (in memory)
    np.complex128_t *b,     # Matrix B: kxn
    int *ldb,               # The size of the first dimension of B (in memory)
    np.complex128_t *beta,  # Scalar multiple
    np.complex128_t *c,     # Matrix C: mxn
    int *ldc                # The size of the first dimension of C (in memory)
)

ctypedef int zgemv_t(
    # Compute C := alpha*A*x + beta*y
    char *trans,    # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    int *m,         # Rows of o(A)
    int *n,         # Columns of o(A) / min(len(x))
    np.complex128_t *alpha, # Scalar multiple
    np.complex128_t *a,     # Matrix A: mxn
    int *lda,       # The size of the first dimension of A (in memory)
    np.complex128_t *x,     # Vector x, min(len(x)) = n
    int *incx,      # The increment between elements of x (usually 1)
    np.complex128_t *beta,  # Scalar multiple
    np.complex128_t *y,     # Vector y, min(len(y)) = m
    int *incy       # The increment between elements of y (usually 1)
)

ctypedef int zsymm_t(
    # ZSYMM - perform one of the matrix-matrix operations   C :=
    # alpha*A*B + beta*C,
    char *side,             # {'L', 'R'}: left, right
    char *uplo,             # {'U','L'}, upper, lower
    int *m,                 # Rows of C
    int *n,                 # Columns of C
    np.complex128_t *alpha, # Scalar multiple
    np.complex128_t *a,     # Matrix A
    int *lda,               # The size of the first dimension of A (in memory)
    np.complex128_t *b,     # Matrix B
    int *ldb,               # The size of the first dimension of B (in memory)
    np.complex128_t *beta,  # Scalar multiple
    np.complex128_t *c,     # Matrix C
    int *ldc,               # The size of the first dimension of C (in memory)
)

ctypedef int zsymv_t(
    # ZSYMV - perform the matrix-vector operation   y := alpha*A*x
    # + beta*y,
    char *uplo,             # {'U','L'}, upper, lower
    int *n,                 # Order of matrix A
    np.complex128_t *alpha, # Scalar multiple
    np.complex128_t *a,     # Matrix A: mxn
    int *lda,               # The size of the first dimension of A (in memory)
    np.complex128_t *x,     # Vector x, min(len(x)) = n
    int *incx,              # The increment between elements of x (usually 1)
    np.complex128_t *beta,  # Scalar multiple
    np.complex128_t *y,     # Vector y, min(len(y)) = n
    int *incy,              # The increment between elements of y (usually 1)
)

ctypedef int ztrmm_t(
    # ZTRMM - perform one of the matrix-matrix operations   B :=
    # alpha*op( A )*B, or B := alpha*B*op( A ),
    char *side,             # {'L', 'R'}: left, right
    char *uplo,             # {'U','L'}, upper, lower
    char *transa,           # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    char *diag,             # {'U','N'}: unit triangular or not
    int *m,                 # Rows of B
    int *n,                 # Columns of B
    np.complex128_t *alpha, # Scalar multiple
    np.complex128_t *a,     # Matrix A
    int *lda,               # The size of the first dimension of A (in memory)
    np.complex128_t *b,     # Matrix B
    int *ldb,               # The size of the first dimension of B (in memory)
)

ctypedef int ztrmv_t(
    # ZTRMV - perform one of the matrix-vector operations   x :=
    # A*x, or x := A'*x,
    char *uplo,           # {'U','L'}, upper, lower
    char *trans,          # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    char *diag,           # {'U','N'}: unit triangular or not
    int *n,               # Order of matrix A
    np.complex128_t *a,   # Matrix A: mxn
    int *lda,             # The size of the first dimension of A (in memory)
    np.complex128_t *x,   # Vector x, min(len(x)) = n
    int *incx,            # The increment between elements of x (usually 1)
)

ctypedef int zcopy_t(
    int *n,         # Number of vector elements to be copied.
    np.complex128_t *x,     # Vector from which to copy.
    int *incx,      # Increment between elements of x.
    np.complex128_t *y,     # array of dimension (n-1) * |incy| + 1, result vector.
    int *incy       # Increment between elements of y.
)

ctypedef int zscal_t(
    # ZSCAL - BLAS level one, scales a double np.complex128_t precision vector
    int *n,          # Number of elements in the vector.
    np.complex128_t *alpha,  # scalar alpha
    np.complex128_t *x,      # Array of dimension (n-1) * |incx| + 1. Vector to be scaled.
    int *incx        # Increment between elements of x.
)

ctypedef int zaxpy_t(
    # Compute y := alpha*x + y
    int *n,         # Columns of o(A) / min(len(x))
    np.complex128_t *alpha, # Scalar multiple
    np.complex128_t *x,     # Vector x, min(len(x)) = n
    int *incx,      # The increment between elements of x (usually 1)
    np.complex128_t *y,     # Vector y, min(len(y)) = m
    int *incy       # The increment between elements of y (usually 1)
)

ctypedef int zswap_t(
    # Swap y and x
    int *n,               # Columns of o(A) / min(len(x))
    np.complex128_t *x,      # Vector x, min(len(x)) = n
    int *incx,            # The increment between elements of x (usually 1)
    np.complex128_t *y,      # Vector y, min(len(y)) = m
    int *incy             # The increment between elements of y (usually 1)
)

ctypedef np.complex128_t zdotu_t(
    # Compute ZDOTU := x.T * y
    int *n,      # Length of vectors
    np.complex128_t *x,  # Vector x, min(len(x)) = n
    int *incx,   # The increment between elements of x (usually 1)
    np.complex128_t *y,  # Vector y, min(len(y)) = m
    int *incy    # The increment between elements of y (usually 1)
)

#
# LAPACK
#

ctypedef int sgetrf_t(
    # SGETRF - compute an LU factorization of a general M-by-N
    # matrix A using partial pivoting with row interchanges
    int *m,          # Rows of A
    int *n,          # Columns of A
    np.float32_t *a, # Matrix A: mxn
    int *lda,        # The size of the first dimension of A (in memory)
    int *ipiv,       # Matrix P: mxn (the pivot indices)
    int *info        # 0 if success, otherwise an error code (integer)
)

ctypedef int sgetri_t(
    # SGETRI - compute the inverse of a matrix using the LU fac-
    # torization computed by SGETRF
    int *n,             # Order of A
    np.float32_t *a,    # Matrix A: nxn (the LUP decomposed matrix from dgetrf)
    int *lda,           # The size of the first dimension of A (in memory)
    int *ipiv,          # Matrix P: nxn (the pivot indices from the LUP decomposition)
    np.float32_t *work, # Matrix: nxn (a workspace for the inversion, optimal size=nxn)
    int *lwork,         # Number of elements in the workspace: optimal is n**2
    int *info           # 0 if success, otherwise an error code (integer)
)

ctypedef int sgetrs_t(
    # SGETRS - solve a system of linear equations  A * X = B or A'
    # * X = B with a general N-by-N matrix A using the LU factori-
    # zation computed by SGETRF
    char *trans,        # Specifies the form of the system of equations
    int *n,             # Order of A
    int *nrhs,          # The number of right hand sides
    np.float32_t *a,    # Matrix A: nxn (the LUP decomposed matrix from dgetrf)
    int *lda,           # The size of the first dimension of A (in memory)
    int *ipiv,          # Matrix P: nxn (the pivot indices from the LUP decomposition)
    np.float32_t *b,    # Matrix B: nxnrhs
    int *ldb,           # The size of the first dimension of B (in memory)
    int *info           # 0 if success, otherwise an error code (integer)
)

ctypedef int spotrf_t(
    # Compute the Cholesky factorization of a
    # real  symmetric positive definite matrix A
    char *uplo,       # 'U':  A = U'U and U is stored, 'L': A = LL' and L is stored
    int *n,           # The order of the matrix A.  n >= 0.
    np.float32_t *a,  # Matrix A: nxn
    int *lda,         # The size of the first dimension of A (in memory)
    int *info         # 0 if success, otherwise an error code (integer)
)

ctypedef int spotri_t(
    # SPOTRI - compute the inverse of a real symmetric positive
    # definite matrix A using the Cholesky factorization A =
    # U**T*U or A = L*L**T computed by SPOTRF
    char *uplo,       # 'U':  A = U'U and U is stored, 'L': A = LL' and L is stored
    int *n,           # The order of the matrix A.  n >= 0.
    np.float32_t *a,  # Matrix A: nxn
    int *lda,         # The size of the first dimension of A (in memory)
    int *info         # 0 if success, otherwise an error code (integer)
)

ctypedef int spotrs_t(
    # SPOTRS - solve a system of linear equations A*X = B with a
    # symmetric positive definite matrix A using the Cholesky fac-
    # torization A = U**T*U or A = L*L**T computed by SPOTRF
    char *uplo,       # 'U':  A = U'U and U is stored, 'L': A = LL' and L is stored
    int *n,           # The order of the matrix A.  n >= 0.
    int *nrhs,        # The number of right hand sides
    np.float32_t *a,  # Matrix A: nxn
    int *lda,         # The size of the first dimension of A (in memory)
    np.float32_t *b,  # Matrix B: nxnrhs
    int *ldb,         # The size of the first dimension of B (in memory)
    int *info         # 0 if success, otherwise an error code (integer)
)

ctypedef int strtrs_t(
    #  STRTRS solves a triangular system of the form
    #      A * X = B,  A**T * X = B,  or  A**H * X = B,
    # where A is a triangular matrix of order N, and B is an N-by-NRHS
    # matrix.  A check is made to verify that A is nonsingular.
    char *uplo,          # 'U':  A is upper triangular
    char *trans,         # N: A * X = B; T: A**T * X = B; C: A**H * X = B
    char *diag,          # {'U','N'}: unit triangular or not
    int *n,              # The order of the matrix A.  n >= 0.
    int *nrhs,           # The number of right hand sides
    np.float32_t *a,     # Matrix A: nxn
    int *lda,            # The size of the first dimension of A (in memory)
    np.float32_t *b,     # Matrix B: nxnrhs
    int *ldb,            # The size of the first dimension of B (in memory)
    int *info            # 0 if success, otherwise an error code (integer)
)

ctypedef int dgetrf_t(
    # DGETRF - compute an LU factorization of a general M-by-N
    # matrix A using partial pivoting with row interchanges
    int *m,          # Rows of A
    int *n,          # Columns of A
    np.float64_t *a, # Matrix A: mxn
    int *lda,        # The size of the first dimension of A (in memory)
    int *ipiv,       # Matrix P: mxn (the pivot indices)
    int *info        # 0 if success, otherwise an error code (integer)
)

ctypedef int dgetri_t(
    # DGETRI - compute the inverse of a matrix using the LU fac-
    # torization computed by DGETRF
    int *n,              # Order of A
    np.float64_t *a,     # Matrix A: nxn (the LUP decomposed matrix from dgetrf)
    int *lda,            # The size of the first dimension of A (in memory)
    int *ipiv,           # Matrix P: nxn (the pivot indices from the LUP decomposition)
    np.float64_t *work,  # Matrix: nxn (a workspace for the inversion, optimal size=nxn)
    int *lwork,          # Number of elements in the workspace: optimal is n**2
    int *info            # 0 if success, otherwise an error code (integer)
)

ctypedef int dgetrs_t(
    # DGETRS - solve a system of linear equations  A * X = B or A'
    # * X = B with a general N-by-N matrix A using the LU factori-
    # zation computed by DGETRF
    char *trans,        # Specifies the form of the system of equations
    int *n,             # Order of A
    int *nrhs,          # The number of right hand sides
    np.float64_t *a,    # Matrix A: nxn (the LUP decomposed matrix from dgetrf)
    int *lda,           # The size of the first dimension of A (in memory)
    int *ipiv,          # Matrix P: nxn (the pivot indices from the LUP decomposition)
    np.float64_t *b,    # Matrix B: nxnrhs
    int *ldb,           # The size of the first dimension of B (in memory)
    int *info           # 0 if success, otherwise an error code (integer)
)

ctypedef int dpotrf_t(
    # Compute the Cholesky factorization of a
    # real  symmetric positive definite matrix A
    char *uplo,      # 'U':  A = U'U and U is stored, 'L': A = LL' and L is stored
    int *n,          # The order of the matrix A.  n >= 0.
    np.float64_t *a, # Matrix A: nxn
    int *lda,        # The size of the first dimension of A (in memory)
    int *info        # 0 if success, otherwise an error code (integer)
)

ctypedef int dpotri_t(
    # DPOTRI - compute the inverse of a real symmetric positive
    # definite matrix A using the Cholesky factorization A =
    # U**T*U or A = L*L**T computed by DPOTRF
    char *uplo,      # 'U':  A = U'U and U is stored, 'L': A = LL' and L is stored
    int *n,          # The order of the matrix A.  n >= 0.
    np.float64_t *a, # Matrix A: nxn
    int *lda,        # The size of the first dimension of A (in memory)
    int *info        # 0 if success, otherwise an error code (integer)
)

ctypedef int dpotrs_t(
    # DPOTRS - solve a system of linear equations A*X = B with a
    # symmetric positive definite matrix A using the Cholesky fac-
    # torization A = U**T*U or A = L*L**T computed by DPOTRF
    char *uplo,       # 'U':  A = U'U and U is stored, 'L': A = LL' and L is stored
    int *n,           # The order of the matrix A.  n >= 0.
    int *nrhs,        # The number of right hand sides
    np.float64_t *a,  # Matrix A: nxn
    int *lda,         # The size of the first dimension of A (in memory)
    np.float64_t *b,  # Matrix B: nxnrhs
    int *ldb,         # The size of the first dimension of B (in memory)
    int *info         # 0 if success, otherwise an error code (integer)
)

ctypedef int dtrtrs_t(
    #  DTRTRS solves a triangular system of the form
    #      A * X = B,  A**T * X = B,  or  A**H * X = B,
    # where A is a triangular matrix of order N, and B is an N-by-NRHS
    # matrix.  A check is made to verify that A is nonsingular.
    char *uplo,          # 'U':  A is upper triangular
    char *trans,         # N: A * X = B; T: A**T * X = B; C: A**H * X = B
    char *diag,          # {'U','N'}: unit triangular or not
    int *n,              # The order of the matrix A.  n >= 0.
    int *nrhs,           # The number of right hand sides
    np.float64_t *a,     # Matrix A: nxn
    int *lda,            # The size of the first dimension of A (in memory)
    np.float64_t *b,     # Matrix B: nxnrhs
    int *ldb,            # The size of the first dimension of B (in memory)
    int *info            # 0 if success, otherwise an error code (integer)
)

ctypedef int cgetrf_t(
    # CGETRF - compute an LU factorization of a general M-by-N
    # matrix A using partial pivoting with row interchanges
    int *m,             # Rows of A
    int *n,             # Columns of A
    np.complex64_t *a,  # Matrix A: mxn
    int *lda,           # The size of the first dimension of A (in memory)
    int *ipiv,          # Matrix P: mxn (the pivot indices)
    int *info           # 0 if success, otherwise an error code (integer)
)

ctypedef int cgetri_t(
    # CGETRI - compute the inverse of a matrix using the LU fac-
    # torization computed by CGETRF
    int *n,               # Order of A
    np.complex64_t *a,    # Matrix A: nxn (the LUP decomposed matrix from dgetrf)
    int *lda,             # The size of the first dimension of A (in memory)
    int *ipiv,            # Matrix P: nxn (the pivot indices from the LUP decomposition)
    np.complex64_t *work, # Matrix: nxn (a workspace for the inversion, optimal size=nxn)
    int *lwork,           # Number of elements in the workspace: optimal is n**2
    int *info             # 0 if success, otherwise an error code (integer)
)

ctypedef int cgetrs_t(
    # CGETRS - solve a system of linear equations  A * X = B, A**T
    # * X = B, or A**H * X = B with a general N-by-N matrix A
    # using the LU factorization computed by CGETRF
    char *trans,          # Specifies the form of the system of equations
    int *n,               # Order of A
    int *nrhs,            # The number of right hand sides
    np.complex64_t *a,    # Matrix A: nxn (the LUP decomposed matrix from dgetrf)
    int *lda,             # The size of the first dimension of A (in memory)
    int *ipiv,            # Matrix P: nxn (the pivot indices from the LUP decomposition)
    np.complex64_t *b,    # Matrix B: nxnrhs
    int *ldb,             # The size of the first dimension of B (in memory)
    int *info             # 0 if success, otherwise an error code (integer)
)

ctypedef int cpotrf_t(
    # Compute the Cholesky factorization of a
    # np.complex128_t Hermitian positive definite matrix A
    char *uplo,         # 'U':  A = U'U and U is stored, 'L': A = LL' and L is stored
    int *n,             # The order of the matrix A.  n >= 0.
    np.complex64_t *a,  # Matrix A: nxn
    int *lda,           # The size of the first dimension of A (in memory)
    int *info           # 0 if success, otherwise an error code (integer)
)

ctypedef int cpotri_t(
    # CPOTRI - compute the inverse of a np.complex128_t Hermitian positive
    # definite matrix A using the Cholesky factorization A =
    # U**T*U or A = L*L**T computed by CPOTRF
    char *uplo,        # 'U':  A = U'U and U is stored, 'L': A = LL' and L is stored
    int *n,            # The order of the matrix A.  n >= 0.
    np.complex64_t *a, # Matrix A: nxn
    int *lda,          # The size of the first dimension of A (in memory)
    int *info          # 0 if success, otherwise an error code (integer)
)

ctypedef int cpotrs_t(
    # ZPOTRS - solve a system of linear equations A*X = B with a
    # Hermitian positive definite matrix A using the Cholesky fac-
    # torization A = U**H*U or A = L*L**H computed by ZPOTRF
    char *uplo,         # 'U':  A = U'U and U is stored, 'L': A = LL' and L is stored
    int *n,             # The order of the matrix A.  n >= 0.
    int *nrhs,          # The number of right hand sides
    np.complex64_t *a,  # Matrix A: nxn
    int *lda,           # The size of the first dimension of A (in memory)
    np.complex64_t *b,  # Matrix B: nxnrhs
    int *ldb,           # The size of the first dimension of B (in memory)
    int *info           # 0 if success, otherwise an error code (integer)
)

ctypedef int ctrtrs_t(
    #  CTRTRS solves a triangular system of the form
    #      A * X = B,  A**T * X = B,  or  A**H * X = B,
    # where A is a triangular matrix of order N, and B is an N-by-NRHS
    # matrix.  A check is made to verify that A is nonsingular.
    char *uplo,          # 'U':  A is upper triangular
    char *trans,          # N: A * X = B; T: A**T * X = B; C: A**H * X = B
    char *diag,           # {'U','N'}: unit triangular or not
    int *n,              # The order of the matrix A.  n >= 0.
    int *nrhs,           # The number of right hand sides
    np.complex64_t *a,   # Matrix A: nxn
    int *lda,            # The size of the first dimension of A (in memory)
    np.complex64_t *b,   # Matrix B: nxnrhs
    int *ldb,            # The size of the first dimension of B (in memory)
    int *info            # 0 if success, otherwise an error code (integer)
)

ctypedef int zgetrf_t(
    # ZGETRF - compute an LU factorization of a general M-by-N
    # matrix A using partial pivoting with row interchanges
    int *m,                # Rows of A
    int *n,                # Columns of A
    np.complex128_t *a,    # Matrix A: mxn
    int *lda,              # The size of the first dimension of A (in memory)
    int *ipiv,             # Matrix P: mxn (the pivot indices)
    int *info              # 0 if success, otherwise an error code (integer)
)

ctypedef int zgetri_t(
    # ZGETRI - compute the inverse of a matrix using the LU fac-
    # torization computed by ZGETRF
    int *n,                # Order of A
    np.complex128_t *a,    # Matrix A: nxn (the LUP decomposed matrix from dgetrf)
    int *lda,              # The size of the first dimension of A (in memory)
    int *ipiv,             # Matrix P: nxn (the pivot indices from the LUP decomposition)
    np.complex128_t *work, # Matrix: nxn (a workspace for the inversion, optimal size=nxn)
    int *lwork,            # Number of elements in the workspace: optimal is n**2
    int *info              # 0 if success, otherwise an error code (integer)
)

ctypedef int zgetrs_t(
    # ZGETRS - solve a system of linear equations  A * X = B, A**T
    # * X = B, or A**H * X = B with a general N-by-N matrix A
    # using the LU factorization computed by ZGETRF
    char *trans,           # Specifies the form of the system of equations
    int *n,                # Order of A
    int *nrhs,             # The number of right hand sides
    np.complex128_t *a,    # Matrix A: nxn (the LUP decomposed matrix from dgetrf)
    int *lda,              # The size of the first dimension of A (in memory)
    int *ipiv,             # Matrix P: nxn (the pivot indices from the LUP decomposition)
    np.complex128_t *b,    # Matrix B: nxnrhs
    int *ldb,              # The size of the first dimension of B (in memory)
    int *info              # 0 if success, otherwise an error code (integer)
)

ctypedef int zpotrf_t(
    # Compute the Cholesky factorization of a
    # np.complex128_t Hermitian positive definite matrix A
    char *uplo,         # 'U':  A = U'U and U is stored, 'L': A = LL' and L is stored
    int *n,             # The order of the matrix A.  n >= 0.
    np.complex128_t *a, # Matrix A: nxn
    int *lda,           # The size of the first dimension of A (in memory)
    int *info           # 0 if success, otherwise an error code (integer)
)

ctypedef int zpotri_t(
    # ZPOTRI - compute the inverse of a np.complex128_t Hermitian positive
    # definite matrix A using the Cholesky factorization A =
    # U**T*U or A = L*L**T computed by ZPOTRF
    char *uplo,          # 'U':  A = U'U and U is stored, 'L': A = LL' and L is stored
    int *n,              # The order of the matrix A.  n >= 0.
    np.complex128_t *a,  # Matrix A: nxn
    int *lda,            # The size of the first dimension of A (in memory)
    int *info            # 0 if success, otherwise an error code (integer)
)

ctypedef int zpotrs_t(
    # ZPOTRS - solve a system of linear equations A*X = B with a
    # Hermitian positive definite matrix A using the Cholesky fac-
    # torization A = U**H*U or A = L*L**H computed by ZPOTRF
    char *uplo,          # 'U':  A = U'U and U is stored, 'L': A = LL' and L is stored
    int *n,              # The order of the matrix A.  n >= 0.
    int *nrhs,           # The number of right hand sides
    np.complex128_t *a,  # Matrix A: nxn
    int *lda,            # The size of the first dimension of A (in memory)
    np.complex128_t *b,  # Matrix B: nxnrhs
    int *ldb,            # The size of the first dimension of B (in memory)
    int *info            # 0 if success, otherwise an error code (integer)
)

ctypedef int ztrtrs_t(
    #  ZTRTRS solves a triangular system of the form
    #      A * X = B,  A**T * X = B,  or  A**H * X = B,
    # where A is a triangular matrix of order N, and B is an N-by-NRHS
    # matrix.  A check is made to verify that A is nonsingular.
    char *uplo,          # 'U':  A is upper triangular
    char *trans,          # N: A * X = B; T: A**T * X = B; C: A**H * X = B
    char *diag,           # {'U','N'}: unit triangular or not
    int *n,              # The order of the matrix A.  n >= 0.
    int *nrhs,           # The number of right hand sides
    np.complex128_t *a,  # Matrix A: nxn
    int *lda,            # The size of the first dimension of A (in memory)
    np.complex128_t *b,  # Matrix B: nxnrhs
    int *ldb,            # The size of the first dimension of B (in memory)
    int *info            # 0 if success, otherwise an error code (integer)
)