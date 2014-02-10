#ifndef ATLAS_REFALIAS3_H
#define ATLAS_REFALIAS3_H
/*
 * Real BLAS
 */
   #define ATL_ssyr2k  ATL_s@(rep2c)syr2k
   #define ATL_ssyrk   ATL_s@(rep2c)syrk
   #define ATL_ssymm   ATL_s@(rep2c)symm
   #define ATL_strmm   ATL_s@(rep2c)trmm
   #define ATL_strsm   ATL_s@(rep2c)trsm
   #define ATL_sgemm   ATL_s@(rep2c)gemm

   #define ATL_dsyr2k  ATL_d@(rep2c)syr2k
   #define ATL_dsyrk   ATL_d@(rep2c)syrk
   #define ATL_dsymm   ATL_d@(rep2c)symm
   #define ATL_dtrmm   ATL_d@(rep2c)trmm
   #define ATL_dtrsm   ATL_d@(rep2c)trsm
   #define ATL_dgemm   ATL_d@(rep2c)gemm

/*
 * Complex BLAS
 */
   #define ATL_ctrmm     ATL_c@(rep2c)trmm
   #define ATL_cher2k    ATL_c@(rep2c)her2k
   #define ATL_csyr2k    ATL_c@(rep2c)syr2k
   #define ATL_cherk     ATL_c@(rep2c)herk
   #define ATL_csyrk     ATL_c@(rep2c)syrk
   #define ATL_chemm     ATL_c@(rep2c)hemm
   #define ATL_csymm     ATL_c@(rep2c)symm
   #define ATL_cgemm     ATL_c@(rep2c)gemm
   #define ATL_ctrsm     ATL_c@(rep2c)trsm

   #define ATL_ztrmm     ATL_z@(rep2c)trmm
   #define ATL_zher2k    ATL_z@(rep2c)her2k
   #define ATL_zsyr2k    ATL_z@(rep2c)syr2k
   #define ATL_zherk     ATL_z@(rep2c)herk
   #define ATL_zsyrk     ATL_z@(rep2c)syrk
   #define ATL_zhemm     ATL_z@(rep2c)hemm
   #define ATL_zsymm     ATL_z@(rep2c)symm
   #define ATL_zgemm     ATL_z@(rep2c)gemm
   #define ATL_ztrsm     ATL_z@(rep2c)trsm

#endif
