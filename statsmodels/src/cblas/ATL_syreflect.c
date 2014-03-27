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

void Mjoin(PATL,syreflect)
   (const enum ATLAS_UPLO Uplo, const int N, TYPE *C, const int ldc)
/*
 * If Uplo == Lower, reflect lower triangle into upper,
 * If Uplo == Upper, reflect upper triangle into lower.
 */
{
   int j;
   #ifdef TCPLX
      const size_t ldc2 = ldc+ldc, incC = ldc2+2;
   #else
      const size_t incC = ldc+1, ldc2=ldc;
   #endif
   TYPE *pC;
   if (Uplo == AtlasLower)
   {
      for (j=0; j < N-1; j++, C += incC)
         Mjoin(PATL,copy)(N-j-1, C+(1 SHIFT), 1, C+ldc2, ldc);
   }
   else
   {
      pC = C + (((size_t)(N-1))SHIFT);
      C += ldc2*(N-1);
      for (j=0; j < N-1; j++, C -= ldc2, pC -= (1 SHIFT))
         Mjoin(PATL,copy)(N-j-1, C, 1, pC, ldc);
   }
}
#ifdef ldc2
   #undef ldc2
#endif
