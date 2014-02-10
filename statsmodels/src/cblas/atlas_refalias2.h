#ifndef ATLAS_REFALIAS2_H
#define ATLAS_REFALIAS2_H
/*
 * Real BLAS
 */
   #define ATL_sger    ATL_s@(rep2c)ger
   #define ATL_sgemv   ATL_s@(rep2c)gemv

   #define ATL_dger    ATL_d@(rep2c)ger
   #define ATL_dgemv   ATL_d@(rep2c)gemv

/*
 * Complex BLAS
 */
   #define ATL_cgemv     ATL_c@(rep2c)gemv
   #define ATL_cgerc     ATL_c@(rep2c)gerc
   #define ATL_cgeru     ATL_c@(rep2c)geru

   #define ATL_zgemv     ATL_z@(rep2c)gemv
   #define ATL_zgerc     ATL_z@(rep2c)gerc
   #define ATL_zgeru     ATL_z@(rep2c)geru

#endif
