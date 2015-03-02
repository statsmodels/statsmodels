cimport libc.stdint as stdint
ctypedef stdint.uintptr_t uintptr_t
ctypedef stdint.intptr_t intptr_t

cdef extern from "grid_interp.h":
    intptr_t binary_search(double, double*, intptr_t, intptr_t)

