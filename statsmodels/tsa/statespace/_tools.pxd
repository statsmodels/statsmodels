#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=False
"""
State Space Models - Cython Tools declarations

Author: Chad Fulton  
License: Simplified-BSD
"""

cimport numpy as np

cdef validate_matrix_shape(str name, Py_ssize_t *shape, int nrows, int ncols, object nobs=*)

cdef validate_vector_shape(str name, Py_ssize_t *shape, int nrows, object nobs=*)

cdef int _ssolve_discrete_lyapunov(np.float32_t * a, np.float32_t * q, int n, int complex_step=*) except *
cdef int _dsolve_discrete_lyapunov(np.float64_t * a, np.float64_t * q, int n, int complex_step=*) except *
cdef int _csolve_discrete_lyapunov(np.complex64_t * a, np.complex64_t * q, int n, int complex_step=*) except *
cdef int _zsolve_discrete_lyapunov(np.complex128_t * a, np.complex128_t * q, int n, int complex_step=*) except *

cdef int _sldl(np.float32_t * A, int n) except *
cdef int _dldl(np.float64_t * A, int n) except *
cdef int _cldl(np.complex64_t * A, int n) except *
cdef int _zldl(np.complex128_t * A, int n) except *

cpdef int sldl(np.float32_t [::1, :] A) except *
cpdef int dldl(np.float64_t [::1, :] A) except *
cpdef int cldl(np.complex64_t [::1, :] A) except *
cpdef int zldl(np.complex128_t [::1, :] A) except *