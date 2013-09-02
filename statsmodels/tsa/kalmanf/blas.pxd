cdef extern from "cblas.h":
    void dgemm "cblas_dgemm"(char storage, char transa, char transb, int m,
                             int n, int k, double alpha, double *A, int lda,
                             double *B, int ldb, double beta, double *C,
                             int ldc)
    # m - rows of A
    # n - cols of B
    # k - cols of A
    # lda - leading dimension of A in sub-program (ie., obeys transpose arg)
    # ldb - leading dimension of B in sub-program "" ""
    # ldc - leading dimension of C
    # 101 = c-order, 111 = no transpose, 112 = transpose

    void dcopy "cblas_dcopy"(const int N, const double *X, const int incX,
                             double *Y, const int incY)

    void daxpy "cblas_daxpy"(const int N, const double alpha, const double *X,
                             const int incX, double *Y, const int incY)

    void dscal "cblas_dscal"(const int N, const double alpha, double *X,
                             const int incX)

    void dsyrk "cblas_dsyrk"(const int N, const int uplo,
                 const int, const int N, const int K,
                 const double alpha, const double *A, const int lda,
                 const double beta, double *C, const int ldc)

    void dsymm "cblas_dsymm"(char storage, const int Side,
                 const int Uplo, const int M, const int N,
                 const double alpha, const double *A, const int lda,
                 const double *B, const int ldb, const double beta,
                 double *C, const int ldc)

    # complex functions - note that Visual Studio doesn't support C99, so
    # doesn't have a complex type -- client code has to handle this.
    void zgemm "cblas_zgemm"(char storage, char transa, char transb, int m,
                             int n, int k, const double *alpha, void *A,
                             int lda, void *B, int ldb,
                             const double *beta, void *C, int ldc)

    void zaxpy "cblas_zaxpy"(const int N, const double *alpha, void *X,
                 const int incX, void *Y, const int incY)

    void zscal "cblas_zscal"(const int N, const double *alpha, void *X,
                             const int incX)
