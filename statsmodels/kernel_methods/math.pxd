cdef extern from "s_erf.h" nogil:
    double erf "sm_erf"(double)
    double erfc "sm_erfc"(double)
