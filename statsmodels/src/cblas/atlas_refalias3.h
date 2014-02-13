#ifndef ATLAS_REFALIAS3_H
#define ATLAS_REFALIAS3_H
/*
 * Real BLAS
 */
   #define ATL_ssyr2k  ATL_srefsyr2k
   #define ATL_ssyrk   ATL_srefsyrk
   #define ATL_ssymm   ATL_srefsymm
   #define ATL_strmm   ATL_sreftrmm
   #define ATL_strsm   ATL_sreftrsm
   #define ATL_sgemm   ATL_srefgemm

   #define ATL_dsyr2k  ATL_drefsyr2k
   #define ATL_dsyrk   ATL_drefsyrk
   #define ATL_dsymm   ATL_drefsymm
   #define ATL_dtrmm   ATL_dreftrmm
   #define ATL_dtrsm   ATL_dreftrsm
   #define ATL_dgemm   ATL_drefgemm

/*
 * Complex BLAS
 */
   #define ATL_ctrmm     ATL_creftrmm
   #define ATL_cher2k    ATL_crefher2k
   #define ATL_csyr2k    ATL_crefsyr2k
   #define ATL_cherk     ATL_crefherk
   #define ATL_csyrk     ATL_crefsyrk
   #define ATL_chemm     ATL_crefhemm
   #define ATL_csymm     ATL_crefsymm
   #define ATL_cgemm     ATL_crefgemm
   #define ATL_ctrsm     ATL_creftrsm

   #define ATL_ztrmm     ATL_zreftrmm
   #define ATL_zher2k    ATL_zrefher2k
   #define ATL_zsyr2k    ATL_zrefsyr2k
   #define ATL_zherk     ATL_zrefherk
   #define ATL_zsyrk     ATL_zrefsyrk
   #define ATL_zhemm     ATL_zrefhemm
   #define ATL_zsymm     ATL_zrefsymm
   #define ATL_zgemm     ATL_zrefgemm
   #define ATL_ztrsm     ATL_zreftrsm

#endif
