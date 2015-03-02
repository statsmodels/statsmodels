cdef extern from "math.h" nogil:
    int isfinite(double v)
    long double powl(long double, long double)
    long double expl(long double)
    long double logl(long double)
    long double sqrtl(long double)
    long double lgammal(long double)
    double erf(double)
    double fabs(double)
