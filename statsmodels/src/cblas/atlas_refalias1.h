#ifndef ATLAS_REFALIAS1_H
#define ATLAS_REFALIAS1_H
/*
 * Real BLAS
 */
   #define ATL_dsdot   ATL_ds@(rep2c)dot
   #define ATL_sdsdot  ATL_sds@(rep2c)dot
   #define ATL_sasum   ATL_s@(rep2c)asum
   #define ATL_snrm2   ATL_s@(rep2c)nrm2
   #define ATL_sdot    ATL_s@(rep2c)dot
   #define ATL_saxpy   ATL_s@(rep2c)axpy
   #define ATL_scopy   ATL_s@(rep2c)copy
   #define ATL_sscal   ATL_s@(rep2c)scal
   #define ATL_sswap   ATL_s@(rep2c)swap
   #define ATL_srotm   ATL_s@(rep2c)rotm
   #define ATL_srot    ATL_s@(rep2c)rot
   #define ATL_srotmg  ATL_s@(rep2c)rotmg
   #define ATL_srotg   ATL_s@(rep2c)rotg
   #define ATL_isamax  ATL_is@(rep2c)amax

   #define ATL_dasum   ATL_d@(rep2c)asum
   #define ATL_dnrm2   ATL_d@(rep2c)nrm2
   #define ATL_ddot    ATL_d@(rep2c)dot
   #define ATL_daxpy   ATL_d@(rep2c)axpy
   #define ATL_dcopy   ATL_d@(rep2c)copy
   #define ATL_dscal   ATL_d@(rep2c)scal
   #define ATL_dswap   ATL_d@(rep2c)swap
   #define ATL_drotm   ATL_d@(rep2c)rotm
   #define ATL_drot    ATL_d@(rep2c)rot
   #define ATL_drotmg  ATL_d@(rep2c)rotmg
   #define ATL_drotg   ATL_d@(rep2c)rotg
   #define ATL_idamax  ATL_id@(rep2c)amax

/*
 * Complex BLAS
 */
   #define ATL_cdotc_sub ATL_c@(rep2c)dotc_sub
   #define ATL_cdotu_sub ATL_c@(rep2c)dotu_sub
   #define ATL_caxpy     ATL_c@(rep2c)axpy
   #define ATL_ccopy     ATL_c@(rep2c)copy
   #define ATL_cscal     ATL_c@(rep2c)scal
   #define ATL_cswap     ATL_c@(rep2c)swap
   #define ATL_icamax    ATL_ic@(rep2c)amax
   #define ATL_csscal    ATL_cs@(rep2c)scal
   #define ATL_scnrm2    ATL_sc@(rep2c)nrm2
   #define ATL_scasum    ATL_sc@(rep2c)asum

   #define ATL_zdotc_sub ATL_z@(rep2c)dotc_sub
   #define ATL_zdotu_sub ATL_z@(rep2c)dotu_sub
   #define ATL_zaxpy     ATL_z@(rep2c)axpy
   #define ATL_zcopy     ATL_z@(rep2c)copy
   #define ATL_zscal     ATL_z@(rep2c)scal
   #define ATL_zswap     ATL_z@(rep2c)swap
   #define ATL_izamax    ATL_iz@(rep2c)amax
   #define ATL_zdscal    ATL_zd@(rep2c)scal
   #define ATL_dznrm2    ATL_dz@(rep2c)nrm2
   #define ATL_dzasum    ATL_dz@(rep2c)asum

#endif
