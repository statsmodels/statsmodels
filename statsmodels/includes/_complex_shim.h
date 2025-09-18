#include <stdio.h>
#include <complex.h>

#ifdef _MSC_VER // using Microsoft's implementation:

typedef _Dcomplex double_complex;

#else // using standard C:

typedef double complex double_complex;

#endif

inline double sm_cabs(double_complex z){
   return cabs(z);
}

inline double_complex sm_clog(double_complex z){
   return clog(z);
}

inline double_complex sm_cexp(double_complex z){
   return cexp(z);
}
