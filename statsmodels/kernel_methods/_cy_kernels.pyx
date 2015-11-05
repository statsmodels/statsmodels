#cython: boundscheck=False
#cython: wraparound=False
#cython: embedsignature=True

import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt, M_PI, pow, sin, cos, fabs
from numpy.math cimport isfinite, copysign

cdef extern from "erf.h" nogil:
    double _sm_erf(double)
    double _sm_erfc(double)

np.import_array()

def sm_erf(double a):
    return _sm_erf(a)

ctypedef np.npy_float64 float64_t
ctypedef np.npy_complex128 complex128_t
#ctypedef np.npy_float128 float128_t

cdef:
    float64_t SPI = sqrt(M_PI)
    float64_t S2PI = sqrt(2.0*M_PI)
    float64_t S2 = sqrt(2.0)
    float64_t PI2 = M_PI*M_PI

s2pi = S2PI
s2 = S2
pi2 = PI2

cdef void _vectorize(object z,
                     object out,
                     float64_t (*fct)(float64_t v)):
    cdef np.broadcast it = np.broadcast(z, out)
    while np.PyArray_MultiIter_NOTDONE(it):
        (<float64_t*> np.PyArray_MultiIter_DATA(it, 1))[0] = fct((<float64_t*> np.PyArray_MultiIter_DATA(it, 0))[0])
        np.PyArray_MultiIter_NEXT(it)

cdef object vectorize(object z,
                      object out,
                      float64_t (*fct)(float64_t v)):
    cdef np.ndarray zz = np.asfarray(z)
    if zz.ndim > 0:
        if out is None:
            out = np.PyArray_EMPTY(zz.ndim, zz.shape, np.NPY_FLOAT64, False)
        _vectorize(<object>zz, out, fct)
        return out
    else:
        return fct(<float64_t>zz)

cdef void _vectorize_cplx(object z,
                          object out,
                          complex128_t (*fct)(float64_t v)):
    cdef np.broadcast it = np.broadcast(z, out)
    while np.PyArray_MultiIter_NOTDONE(it):
        (<complex128_t*> np.PyArray_MultiIter_DATA(it, 1))[0] = fct((<float64_t*> np.PyArray_MultiIter_DATA(it, 0))[0])
        np.PyArray_MultiIter_NEXT(it)

cdef object vectorize_cplx(object z,
                           object out,
                           complex128_t (*fct)(float64_t v)):
    cdef np.ndarray zz = np.asfarray(z)
    if zz.ndim > 0:
        if out is None:
            out = np.PyArray_EMPTY(zz.ndim, zz.shape, np.NPY_COMPLEX128, False)
        _vectorize_cplx(<object>zz, out, fct)
        return out
    else:
        return fct(<float64_t>zz)

cdef float64_t _norm1d_pdf(float64_t z):
    return exp(-z*z/2)/S2PI

def norm1d_pdf(object z, object out = None):
    return vectorize(z, out, _norm1d_pdf)

cdef float64_t _norm1d_convolution(float64_t z):
    return exp(-z*z/4)/(2*SPI)

def norm1d_convolution(object z, object out = None):
    return vectorize(z, out, _norm1d_convolution)

cdef float64_t _norm1d_cdf(float64_t z):
    return _sm_erf(z/S2) / 2 + 0.5

def norm1d_cdf(object z, object out = None):
    return vectorize(z, out, _norm1d_cdf)

cdef float64_t _norm1d_pm1(float64_t z):
    return -exp(-z*z/2) / S2PI

def norm1d_pm1(object z, object out = None):
    return vectorize(z, out, _norm1d_pm1)

cdef float64_t _norm1d_pm2(float64_t z):
    if isfinite(z):
        return 0.5*_sm_erf(z/S2) + 0.5 - z/S2PI*exp(-z*z/2)
    return 0.5*_sm_erf(z/S2)+0.5

def norm1d_pm2(object z, object out = None):
    return vectorize(z, out, _norm1d_pm2)

cdef float64_t tricube_a = sqrt(35./243)
#cdef float128_t tricube_al = sqrtl(35./243)
tricube_width = tricube_a

cdef float64_t _tricube_pdf(float64_t z):
    z *= tricube_a
    if z < 0:
        z = -z
    if z > 1:
        return 0
    cdef float64_t z3_1 = 1-z*z*z
    return 70./81*(z3_1*z3_1*z3_1) * tricube_a

def tricube_pdf(object z, object out = None):
    return vectorize(z, out, _tricube_pdf)

cdef float64_t _tricube_cdf(float64_t z):
    z *= tricube_a
    if z < -1:
        return 0.
    if z > 1:
        return 1.
    if z > 0:
        return 1./162*(60*pow(z, 7.) - 7.*(2*pow(z, 10.) + 15.*pow(z, 4.)) + 140*z + 81)
    else:
        return 1./162*(60*pow(z, 7.) + 7.*(2*pow(z, 10.) + 15.*pow(z, 4.)) + 140*z + 81)

def tricube_cdf(object z, object out = None):
    return vectorize(z, out, _tricube_cdf)

cdef float64_t _tricube_pm1(float64_t zc):
    cdef float64_t z = zc
    z *= tricube_a
    if z < -1 or z > 1:
        return 0
    if z < 0:
        z = -z
    cdef float64_t z2 = z*z
    cdef float64_t z3 = z2*z
    cdef float64_t z5 = z3*z2
    cdef float64_t z8 = z5*z3
    cdef float64_t z11 = z8*z3
    return 7./(tricube_a*3564)*(165*z8 - 8.*(5*z11 + 33.*z5) + 220*z2 - 81)

def tricube_pm1(object z, object out = None):
    return vectorize(z, out, _tricube_pm1)

cdef float64_t _tricube_pm2(float64_t z):
    z *= tricube_a
    if z < -1:
        return 0
    if z > 1:
        return 1
    cdef float64_t z3 = z*z*z
    cdef float64_t z6 = z3*z3
    cdef float64_t z9 = z6*z3
    cdef float64_t z12 = z9*z3
    if z > 0:
        return 35./(tricube_a*tricube_a*486)*(4*z9 - (z12 + 6.*z6) + 4*z3 + 1)
    else:
        return 35./(tricube_a*tricube_a*486)*(4*z9 + (z12 + 6.*z6) + 4*z3 + 1)

def tricube_pm2(object z, object out = None):
    return vectorize(z, out, _tricube_pm2)

cdef float64_t _tricube_convolution_above_1(float64_t z):
    cdef:
        float64_t z2 = z*z
        float64_t z3 = z2*z
        float64_t z4 = z3*z
        float64_t z5 = z4*z
        float64_t z6 = z5*z
        float64_t z7 = z6*z
        float64_t z8 = z7*z
        float64_t z9 = z8*z
        float64_t z10 = z9*z
        float64_t z13 = z10*z3
        float64_t z16 = z13*z3
        float64_t z19 = z16*z3
    return (-z19/923780 + 3*z16/40040 - 57*z13/20020 + 31*z10/140 - 81*z9/70 +
            729*z8/220 - 969*z7/140 + 9963*z6/910 - 729*z5/55 + 66*z4/5 - 972*z3/91 +
            5832*z2/935 - 16*z/5 + 2592./1729.)

cdef float64_t _tricube_convolution_below_1(float64_t z):
    cdef:
        float64_t z2 = z*z
        float64_t z4 = z2*z2
        float64_t z6 = z4*z2
        float64_t z7 = z6*z
        float64_t z8 = z7*z
        float64_t z9 = z8*z
        float64_t z10 = z9*z
        float64_t z13 = z9*z4
        float64_t z16 = z9*z7
        float64_t z19 = z13*z6
    return (3*z19/923780 - 3*z16/40040 + 111*z13/20020 - 31*z10/140 + 81*z9/70 - 729*z8/220 +
            747*z7/140 - 729*z6/182 + 9*z4/5 - 19683 * z2 / 13090 + 6561. / 6916.)

cdef float64_t _tricube_convolution(float64_t z):
    z *= tricube_a
    if z < 0:
        z = -z
    if z > 2:
        return 0
    if z > 1:
        return _tricube_convolution_above_1(z) * tricube_a * 4900/6561
    return _tricube_convolution_below_1(z) * tricube_a * 4900/6561

def tricube_convolution(object z, object out = None):
    return vectorize(z, out, _tricube_convolution)

cdef float64_t epanechnikov_a = 1./sqrt(5.)
epanechnikov_width = epanechnikov_a

cdef float64_t _epanechnikov_pdf(float64_t z):
    z *= epanechnikov_a
    if z < -1:
        return 0
    if z > 1:
        return 0
    return .75*(1-z*z)*epanechnikov_a

def epanechnikov_pdf(object z, object out = None):
    return vectorize(z, out, _epanechnikov_pdf)

cdef float64_t _epanechnikov_convolution(float64_t z):
    if z < 0:
        z = -z
    z *= epanechnikov_a
    cdef float64_t z2 = 2 - z
    if z2 < 0:
        return 0
    return 3./160.*(z2*z2*z2)*(4 + 6*z + z*z)*epanechnikov_a

def epanechnikov_convolution(object z, object out = None):
    return vectorize(z, out, _epanechnikov_convolution)

cdef float64_t _epanechnikov_cdf(float64_t z):
    z *= epanechnikov_a
    if z < -1:
        return 0
    if z > 1:
        return 1
    return .25*(2+3*z-z*z*z)

def epanechnikov_cdf(object z, object out = None):
    return vectorize(z, out, _epanechnikov_cdf)

cdef float64_t _epanechnikov_pm1(float64_t z):
    z *= epanechnikov_a
    if z < -1:
        return 0
    if z > 1:
        return 0
    cdef float64_t z2 = z*z
    return -3./16.*(1-2*z2+z2*z2)/epanechnikov_a

def epanechnikov_pm1(object z, object out = None):
    return vectorize(z, out, _epanechnikov_pm1)

cdef float64_t _epanechnikov_pm2(float64_t z):
    z *= epanechnikov_a
    if z < -1:
        return 0
    if z > 1:
        return 1
    cdef float64_t z3 = z*z*z
    return 0.25*(2+5*z3-3*z3*z*z)

def epanechnikov_pm2(object z, object out = None):
    return vectorize(z, out, _epanechnikov_pm2)

cdef float64_t _epanechnikov_fft(float64_t z):
    if z == 0:
        return 1
    z *= 2*M_PI/epanechnikov_a
    cdef float64_t z2 = z*z
    cdef float64_t z3 = z2*z
    return 3/z3 * (sin(z) - z*cos(z))

def epanechnikov_fft(object z, object out = None):
    return vectorize(z, out, _epanechnikov_fft)

cdef complex128_t _epanechnikov_fft_xfx(float64_t z):
    if z == 0:
        return complex128_t(0, 0)
    z *= 2*M_PI/epanechnikov_a
    cdef:
        float64_t sin_z = sin(z)
        float64_t cos_z = cos(z)
        float64_t z2 = z*z
        float64_t z4 = z2*z2
    return complex128_t(0, 9/(epanechnikov_a*z4)*(z*cos_z - sin_z + z2*sin_z/3))

def epanechnikov_fft_xfx(object z, object out = None):
    return vectorize_cplx(z, out, _epanechnikov_fft_xfx)

cdef float64_t _epanechnikov_o4_pdf(float64_t z):
    if z < -1:
        return 0
    if z > 1:
        return 0
    return 0.125*(9-15*z*z)

def epanechnikov_o4_pdf(object z, object out = None):
    return vectorize(z, out, _epanechnikov_o4_pdf)

cdef float64_t _epanechnikov_o4_cdf(float64_t z):
    if z < -1:
        return 0
    if z > 1:
        return 1
    return .125*(4+9*z-5*z*z*z)

def epanechnikov_o4_cdf(object z, object out = None):
    return vectorize(z, out, _epanechnikov_o4_cdf)

cdef float64_t _epanechnikov_o4_pm1(float64_t z):
    if z < -1:
        return 0
    if z > 1:
        return 0
    cdef float64_t z2 = z*z
    return 1./32.*(18*z2-3-15*z2*z2)

def epanechnikov_o4_pm1(object z, object out = None):
    return vectorize(z, out, _epanechnikov_o4_pm1)

cdef float64_t _epanechnikov_o4_pm2(float64_t z):
    if z < -1:
        return 0
    if z > 1:
        return 0
    cdef float64_t z2 = z*z
    cdef float64_t z3 = z2*z
    return .375*(z3 - z2*z3)

def epanechnikov_o4_pm2(object z, object out = None):
    return vectorize(z, out, _epanechnikov_o4_pm2)

cdef float64_t _normal_o4_pdf(float64_t z):
    return (3-z*z)*_norm1d_pdf(z)/2

def normal_o4_pdf(object z, object out = None):
    return vectorize(z, out, _normal_o4_pdf)

cdef float64_t _normal_o4_cdf(float64_t z):
    if not isfinite(z):
        if z < 0:
            return 0
        return 1
    return _norm1d_cdf(z)+z*_norm1d_pdf(z)/2

def normal_o4_cdf(object z, object out = None):
    return vectorize(z, out, _normal_o4_cdf)

cdef float64_t _normal_o4_pm1(float64_t z):
    if not isfinite(z):
        return 0.0
    return _norm1d_pdf(z) - _normal_o4_pdf(z)

def normal_o4_pm1(object z, object out = None):
    return vectorize(z, out, _normal_o4_pm1)

cdef float64_t _normal_o4_pm2(float64_t z):
    if not isfinite(z):
        return 0.0
    return z*z*z/2*_norm1d_pdf(z)

def normal_o4_pm2(object z, object out = None):
    return vectorize(z, out, _normal_o4_pm2)

