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

cpdef int sreorder_missing_matrix(np.float32_t [::1, :, :] A, int [::1, :] missing, int reorder_rows, int reorder_cols, int diagonal) except *
cpdef int dreorder_missing_matrix(np.float64_t [::1, :, :] A, int [::1, :] missing, int reorder_rows, int reorder_cols, int diagonal) except *
cpdef int creorder_missing_matrix(np.complex64_t [::1, :, :] A, int [::1, :] missing, int reorder_rows, int reorder_cols, int diagonal) except *
cpdef int zreorder_missing_matrix(np.complex128_t [::1, :, :] A, int [::1, :] missing, int reorder_rows, int reorder_cols, int diagonal) except *

cpdef int sreorder_missing_vector(np.float32_t [::1, :] A, int [::1, :] missing) except *
cpdef int dreorder_missing_vector(np.float64_t [::1, :] A, int [::1, :] missing) except *
cpdef int creorder_missing_vector(np.complex64_t [::1, :] A, int [::1, :] missing) except *
cpdef int zreorder_missing_vector(np.complex128_t [::1, :] A, int [::1, :] missing) except *

cpdef int scopy_missing_matrix(np.float32_t [::1, :, :] A, np.float32_t [::1, :, :] B, int [::1, :] missing, int copy_rows, int copy_cols, int diagonal) except *
cpdef int dcopy_missing_matrix(np.float64_t [::1, :, :] A, np.float64_t [::1, :, :] B, int [::1, :] missing, int copy_rows, int copy_cols, int diagonal) except *
cpdef int ccopy_missing_matrix(np.complex64_t [::1, :, :] A, np.complex64_t [::1, :, :] B, int [::1, :] missing, int copy_rows, int copy_cols, int diagonal) except *
cpdef int zcopy_missing_matrix(np.complex128_t [::1, :, :] A, np.complex128_t [::1, :, :] B, int [::1, :] missing, int copy_rows, int copy_cols, int diagonal) except *

cpdef int scopy_missing_vector(np.float32_t [::1, :] A, np.float32_t [::1, :] B, int [::1, :] missing) except *
cpdef int dcopy_missing_vector(np.float64_t [::1, :] A, np.float64_t [::1, :] B, int [::1, :] missing) except *
cpdef int ccopy_missing_vector(np.complex64_t [::1, :] A, np.complex64_t [::1, :] B, int [::1, :] missing) except *
cpdef int zcopy_missing_vector(np.complex128_t [::1, :] A, np.complex128_t [::1, :] B, int [::1, :] missing) except *

cpdef int scopy_index_matrix(np.float32_t [::1, :, :] A, np.float32_t [::1, :, :] B, int [::1, :] index, int copy_rows, int copy_cols, int diagonal) except *
cpdef int dcopy_index_matrix(np.float64_t [::1, :, :] A, np.float64_t [::1, :, :] B, int [::1, :] index, int copy_rows, int copy_cols, int diagonal) except *
cpdef int ccopy_index_matrix(np.complex64_t [::1, :, :] A, np.complex64_t [::1, :, :] B, int [::1, :] index, int copy_rows, int copy_cols, int diagonal) except *
cpdef int zcopy_index_matrix(np.complex128_t [::1, :, :] A, np.complex128_t [::1, :, :] B, int [::1, :] index, int copy_rows, int copy_cols, int diagonal) except *

cpdef int scopy_index_vector(np.float32_t [::1, :] A, np.float32_t [::1, :] B, int [::1, :] index) except *
cpdef int dcopy_index_vector(np.float64_t [::1, :] A, np.float64_t [::1, :] B, int [::1, :] index) except *
cpdef int ccopy_index_vector(np.complex64_t [::1, :] A, np.complex64_t [::1, :] B, int [::1, :] index) except *
cpdef int zcopy_index_vector(np.complex128_t [::1, :] A, np.complex128_t [::1, :] B, int [::1, :] index) except *

cdef int _sselect_cov(int k_states, int k_posdef, int k_states_total,
                           np.float32_t * tmp,
                           np.float32_t * selection,
                           np.float32_t * cov,
                           np.float32_t * selected_cov)

cdef int _dselect_cov(int k_states, int k_posdef, int k_states_total,
                           np.float64_t * tmp,
                           np.float64_t * selection,
                           np.float64_t * cov,
                           np.float64_t * selected_cov)

cdef int _cselect_cov(int k_states, int k_posdef, int k_states_total,
                           np.complex64_t * tmp,
                           np.complex64_t * selection,
                           np.complex64_t * cov,
                           np.complex64_t * selected_cov)

cdef int _zselect_cov(int k_states, int k_posdef, int k_states_total,
                           np.complex128_t * tmp,
                           np.complex128_t * selection,
                           np.complex128_t * cov,
                           np.complex128_t * selected_cov)

