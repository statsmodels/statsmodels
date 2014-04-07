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

ctypedef int zgemm_t(
    # Compute C := alpha*A*B + beta*C
    char *transa,   # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    char *transb,   # {'T','C'}: o(B)=B'; {'N'}: o(B)=B
    int *m,         # Rows of o(A)    (and of C)
    int *n,         # Columns of o(B) (and of C)
    int *k,         # Columns of o(A) / Rows of o(B)
    complex *alpha, # Scalar multiple
    void *a,     # Matrix A: mxk
    int *lda,       # The size of the first dimension of A (in memory)
    void *b,     # Matrix B: kxn
    int *ldb,       # The size of the first dimension of B (in memory)
    complex *beta,  # Scalar multiple
    void *c,     # Matrix C: mxn
    int *ldc        # The size of the first dimension of C (in memory)
)

ctypedef double ddot_t(
    # Compute DDOT := x.T * y
    int *n,        # Length of vectors
    double *x,     # Vector x, min(len(x)) = n
    int *incx,     # The increment between elements of x (usually 1)
    double *y,     # Vector y, min(len(y)) = m
    int *incy      # The increment between elements of y (usually 1)
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

ctypedef complex zdotu_t(
    # Compute ZDOT := x.T * y
    int *n,        # Length of vectors
    complex *x,     # Vector x, min(len(x)) = n
    int *incx,     # The increment between elements of x (usually 1)
    complex *y,     # Vector y, min(len(y)) = m
    int *incy      # The increment between elements of y (usually 1)
)

ctypedef double dger_t(
    # Compute A := alpha*x*y.T + A
    int *m,         # Length of x
    int *n,         # Length of y
    double *alpha,  # Scalar multiple
    double *x,      # Vector X
    int *incx,      # Increment between elements of x (usually 1)
    double *y,      # Vector y
    int *incy,      # Increment between elements of y (usually 1)
    double *A,      # Matrix A: m x n
    int *lda        # The size of the first dimension of A in memory
)
