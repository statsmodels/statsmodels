/*
 *             Automatically Tuned Linear Algebra Software v3.10.1
 *                    (C) Copyright 1999 R. Clint Whaley
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
#include "atlas_misc.h"
#ifdef ATL_USEPTHREADS
   #include "atlas_ptalias3.h"
#endif
#include "cblas.h"

#include "atlas_level1.h"
#include "atlas_level2.h"
#include "atlas_level3.h"

void cblas_dgemm(const enum CBLAS_ORDER Order,
                 const enum CBLAS_TRANSPOSE TA, const enum CBLAS_TRANSPOSE TB,
                 const int M, const int N, const int K,
                 const double  alpha, const double *A, const int lda,
                 const double *B, const int ldb, const double  beta,
                 double *C, const int ldc)
{
   int info=2000;

#ifndef NoCblasErrorChecks
   if (M < 0) info = cblas_errprn(4, info,
                     "M cannot be less than zero 0,; is set to %d.", M);
   if (N < 0) info = cblas_errprn(5, info,
                     "N cannot be less than zero 0,; is set to %d.", N);
   if (K < 0) info = cblas_errprn(6, info,
                     "K cannot be less than zero 0,; is set to %d.", K);

   if (Order == CblasRowMajor)
   {
      if (TA == CblasNoTrans)
      {
         if ( (lda < K) || (lda < 1) )
            info = cblas_errprn(9, info, "lda must be >= MAX(K,1): lda=%d K=%d",
                                lda, K);
      }
      else
      {
         if (TA != CblasConjTrans && TA != CblasTrans)
            info = cblas_errprn(2, info,
                                "TransA must be %d, %d or %d, but is set to %d",
                                CblasNoTrans, CblasTrans, CblasConjTrans, TA);
         if ( (lda < M) || (lda < 1) )
            info = cblas_errprn(9, info, "lda must be >= MAX(M,1): lda=%d M=%d",
                                lda, M);
      }
      if (TB == CblasNoTrans)
      {
         if ( (ldb < N) || (ldb < 1) )
            info = cblas_errprn(11, info,"ldb must be >= MAX(N,1): ldb=%d N=%d",
                                ldb, N);
      }
      else
      {
         if (TB != CblasConjTrans && TB != CblasTrans)
            info = cblas_errprn(3, info,
                                "TransB must be %d, %d or %d, but is set to %d",
                                CblasNoTrans, CblasTrans, CblasConjTrans, TB);
         if ( (ldb < K) || (ldb < 1) )
            info = cblas_errprn(11, info,"ldb must be >= MAX(K,1): ldb=%d K=%d",
                                ldb, K);
      }
      if ( (ldc < N) || (ldc < 1) )
         info = cblas_errprn(14, info,"ldc must be >= MAX(N,1): ldc=%d N=%d",
                             ldc, N);
   }
   else if (Order == CblasColMajor)
   {
      if (TA == CblasNoTrans)
      {
         if ( (lda < M) || (lda < 1) )
            info = cblas_errprn(9, info, "lda must be >= MAX(M,1): lda=%d M=%d",
                                lda, M);
      }
      else
      {
         if (TA != CblasConjTrans && TA != CblasTrans)
            info = cblas_errprn(2, info,
                                "TransA must be %d, %d or %d, but is set to %d",
                                CblasNoTrans, CblasTrans, CblasConjTrans, TA);
         if ( (lda < K) || (lda < 1) )
            info = cblas_errprn(9, info, "lda must be >= MAX(K,1): lda=%d K=%d",
                                lda, K);
      }
      if (TB == CblasNoTrans)
      {
         if ( (ldb < K) || (ldb < 1) )
            info = cblas_errprn(11,info, "ldb must be >= MAX(K,1): ldb=%d K=%d",
                                ldb, K);
      }
      else
      {
         if (TB != CblasConjTrans && TB != CblasTrans)
            info = cblas_errprn(3, info,
                                "TransB must be %d, %d or %d, but is set to %d",
                                CblasNoTrans, CblasTrans, CblasConjTrans, TB);
         if ( (ldb < N) || (ldb < 1) )
            info = cblas_errprn(11,info, "ldb must be >= MAX(N,1): ldb=%d N=%d",
                                ldb, N);
      }
      if ( (ldc < M) || (ldc < 1) )
         info = cblas_errprn(14, info,"ldc must be >= MAX(M,1): ldc=%d M=%d",
                             ldc, M);
   }
   else
      info = cblas_errprn(1, info, "Order must be %d or %d, but is set to %d",
                          CblasRowMajor, CblasColMajor, Order);
   if (info != 2000)
   {
      cblas_xerbla(info, "cblas_dgemm", "");
      return;
   }
#endif
/*
 * Call SYRK when that's what the user is actually asking for; just handle
 * beta=0, because beta=X requires we copy C and then subtract to preserve
 * asymmetry
 */
   if (A == B && M == N && TA != TB && lda == ldb && beta == 0.0)
   {
      ATL_dsyrk(CblasUpper, (Order == CblasColMajor) ? TA : TB, N, K,
                alpha, A, lda, beta, C, ldc);
      ATL_dsyreflect(CblasUpper, N, C, ldc);
      return;
   }
   if (Order == CblasColMajor)
      ATL_dgemm(TA, TB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
   else
      ATL_dgemm(TB, TA, N, M, K, alpha, B, ldb, A, lda, beta, C, ldc);
}
