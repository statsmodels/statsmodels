/*
 *             Automatically Tuned Linear Algebra Software v3.10.1
 *                    (C) Copyright 1997 R. Clint Whaley
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *   1. Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *   2. Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions, and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *   3. The name of the ATLAS group or the names of its contributers may
 *      not be used to endorse or promote products derived from this
 *      software without specific written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE ATLAS GROUP OR ITS CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */
/*
 * ===========================================================================
 * Prototypes for level 3 BLAS
 * ===========================================================================
 */
#ifndef ATLAS_LEVEL3_H
#define ATLAS_LEVEL3_H
#include "atlas_refalias3.h"


/*
 * Routines with standard 4 prefixes (S, D, C, Z)
 */
int ATL_sGetNB(void);
int ATL_sGetNCNB(void);
void ATL_sgemm(const enum ATLAS_TRANS TransA, const enum ATLAS_TRANS TransB,
               const int M, const int N, const int K, const float alpha,
               const float *A, const int lda, const float *B, const int ldb,
               const float beta, float *C, const int ldc);
void ATL_ssymm(const enum ATLAS_SIDE Side, const enum ATLAS_UPLO Uplo,
               const int M, const int N, const float alpha,
               const float *A, const int lda, const float *B, const int ldb,
               const float beta, float *C, const int ldc);
void ATL_ssyrk(const enum ATLAS_UPLO Uplo, const enum ATLAS_TRANS Trans,
               const int N, const int K, const float alpha,
               const float *A, const int lda, const float beta,
               float *C, const int ldc);
void ATL_ssyr2k(const enum ATLAS_UPLO Uplo, const enum ATLAS_TRANS Trans,
                const int N, const int K, const float alpha,
                const float *A, const int lda, const float *B, const int ldb,
                const float beta, float *C, const int ldc);
void ATL_strmm(const enum ATLAS_SIDE Side, const enum ATLAS_UPLO Uplo,
               const enum ATLAS_TRANS TransA, const enum ATLAS_DIAG Diag,
               const int M, const int N, const float alpha,
               const float *A, const int lda, float *B, const int ldb);
void ATL_strsm(const enum ATLAS_SIDE Side, const enum ATLAS_UPLO Uplo,
               const enum ATLAS_TRANS TransA, const enum ATLAS_DIAG Diag,
               const int M, const int N, const float alpha,
               const float *A, const int lda, float *B, const int ldb);

int ATL_dGetNB(void);
int ATL_dGetNCNB(void);
void ATL_dgemm(const enum ATLAS_TRANS TransA, const enum ATLAS_TRANS TransB,
               const int M, const int N, const int K, const double alpha,
               const double *A, const int lda, const double *B, const int ldb,
               const double beta, double *C, const int ldc);
void ATL_dsymm(const enum ATLAS_SIDE Side, const enum ATLAS_UPLO Uplo,
               const int M, const int N, const double alpha,
               const double *A, const int lda, const double *B, const int ldb,
               const double beta, double *C, const int ldc);
void ATL_dsyrk(const enum ATLAS_UPLO Uplo, const enum ATLAS_TRANS Trans,
               const int N, const int K, const double alpha,
               const double *A, const int lda, const double beta,
               double *C, const int ldc);
void ATL_dsyr2k(const enum ATLAS_UPLO Uplo, const enum ATLAS_TRANS Trans,
                const int N, const int K, const double alpha,
                const double *A, const int lda, const double *B, const int ldb,
                const double beta, double *C, const int ldc);
void ATL_dtrmm(const enum ATLAS_SIDE Side, const enum ATLAS_UPLO Uplo,
               const enum ATLAS_TRANS TransA, const enum ATLAS_DIAG Diag,
               const int M, const int N, const double alpha,
               const double *A, const int lda, double *B, const int ldb);
void ATL_dtrsm(const enum ATLAS_SIDE Side, const enum ATLAS_UPLO Uplo,
               const enum ATLAS_TRANS TransA, const enum ATLAS_DIAG Diag,
               const int M, const int N, const double alpha,
               const double *A, const int lda, double *B, const int ldb);

int ATL_cGetNB(void);
int ATL_cGetNCNB(void);
void ATL_cgemm(const enum ATLAS_TRANS TransA, const enum ATLAS_TRANS TransB,
               const int M, const int N, const int K, const float *alpha,
               const float *A, const int lda, const float *B, const int ldb,
               const float *beta, float *C, const int ldc);
void ATL_csymm(const enum ATLAS_SIDE Side, const enum ATLAS_UPLO Uplo,
               const int M, const int N, const float *alpha,
               const float *A, const int lda, const float *B, const int ldb,
               const float *beta, float *C, const int ldc);
void ATL_csyrk(const enum ATLAS_UPLO Uplo, const enum ATLAS_TRANS Trans,
               const int N, const int K, const float *alpha,
               const float *A, const int lda, const float *beta,
               float *C, const int ldc);
void ATL_csyr2k(const enum ATLAS_UPLO Uplo, const enum ATLAS_TRANS Trans,
                const int N, const int K, const float *alpha,
                const float *A, const int lda, const float *B, const int ldb,
                const float *beta, float *C, const int ldc);
void ATL_ctrmm(const enum ATLAS_SIDE Side, const enum ATLAS_UPLO Uplo,
               const enum ATLAS_TRANS TransA, const enum ATLAS_DIAG Diag,
               const int M, const int N, const float *alpha,
               const float *A, const int lda, float *B, const int ldb);
void ATL_ctrsm(const enum ATLAS_SIDE Side, const enum ATLAS_UPLO Uplo,
               const enum ATLAS_TRANS TransA, const enum ATLAS_DIAG Diag,
               const int M, const int N, const float *alpha,
               const float *A, const int lda, float *B, const int ldb);

int ATL_zGetNB(void);
int ATL_zGetNCNB(void);
void ATL_zgemm(const enum ATLAS_TRANS TransA, const enum ATLAS_TRANS TransB,
               const int M, const int N, const int K, const double *alpha,
               const double *A, const int lda, const double *B, const int ldb,
               const double *beta, double *C, const int ldc);
void ATL_zsymm(const enum ATLAS_SIDE Side, const enum ATLAS_UPLO Uplo,
               const int M, const int N, const double *alpha,
               const double *A, const int lda, const double *B, const int ldb,
               const double *beta, double *C, const int ldc);
void ATL_zsyrk(const enum ATLAS_UPLO Uplo, const enum ATLAS_TRANS Trans,
               const int N, const int K, const double *alpha,
               const double *A, const int lda, const double *beta,
               double *C, const int ldc);
void ATL_zsyr2k(const enum ATLAS_UPLO Uplo, const enum ATLAS_TRANS Trans,
                const int N, const int K, const double *alpha,
                const double *A, const int lda, const double *B, const int ldb,
                const double *beta, double *C, const int ldc);
void ATL_ztrmm(const enum ATLAS_SIDE Side, const enum ATLAS_UPLO Uplo,
               const enum ATLAS_TRANS TransA, const enum ATLAS_DIAG Diag,
               const int M, const int N, const double *alpha,
               const double *A, const int lda, double *B, const int ldb);
void ATL_ztrsm(const enum ATLAS_SIDE Side, const enum ATLAS_UPLO Uplo,
               const enum ATLAS_TRANS TransA, const enum ATLAS_DIAG Diag,
               const int M, const int N, const double *alpha,
               const double *A, const int lda, double *B, const int ldb);


/*
 * Routines with prefixes C and Z only
 */
void ATL_chemm(const enum ATLAS_SIDE Side, const enum ATLAS_UPLO Uplo,
               const int M, const int N, const float *alpha,
               const float *A, const int lda, const float *B, const int ldb,
               const float *beta, float *C, const int ldc);
void ATL_cherk(const enum ATLAS_UPLO Uplo, const enum ATLAS_TRANS Trans,
               const int N, const int K, const float alpha,
               const float *A, const int lda, const float beta,
               float *C, const int ldc);
void ATL_cher2k(const enum ATLAS_UPLO Uplo, const enum ATLAS_TRANS Trans,
                const int N, const int K, const float *alpha,
                const float *A, const int lda, const float *B, const int ldb,
                const float beta, float *C, const int ldc);

void ATL_zhemm(const enum ATLAS_SIDE Side, const enum ATLAS_UPLO Uplo,
               const int M, const int N, const double *alpha,
               const double *A, const int lda, const double *B, const int ldb,
               const double *beta, double *C, const int ldc);
void ATL_zherk(const enum ATLAS_UPLO Uplo, const enum ATLAS_TRANS Trans,
               const int N, const int K, const double alpha,
               const double *A, const int lda, const double beta,
               double *C, const int ldc);
void ATL_zher2k(const enum ATLAS_UPLO Uplo, const enum ATLAS_TRANS Trans,
                const int N, const int K, const double *alpha,
                const double *A, const int lda, const double *B, const int ldb,
                const double beta, double *C, const int ldc);


#endif
