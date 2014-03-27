#ifndef ATLAS_REFALIAS2_H
#define ATLAS_REFALIAS2_H
/*
 * Real BLAS
 */
   #define ATL_sger    ATL_srefger
   #define ATL_sgemv   ATL_srefgemv

   #define ATL_dger    ATL_drefger
   #define ATL_dgemv   ATL_drefgemv

/*
 * Complex BLAS
 */
   #define ATL_cgemv     ATL_crefgemv
   #define ATL_cgerc     ATL_crefgerc
   #define ATL_cgeru     ATL_crefgeru

   #define ATL_zgemv     ATL_zrefgemv
   #define ATL_zgerc     ATL_zrefgerc
   #define ATL_zgeru     ATL_zrefgeru

#endif
