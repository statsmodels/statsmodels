cimport cython
cimport numpy as np
import numpy as np
from libc.math cimport floor, fmod, round, NAN
from grid_interp cimport binary_search
cimport grid_interp

ctypedef np.float64_t DOUBLE
ctypedef np.int_t INT
ctypedef grid_interp.uintptr_t uintptr_t
ctypedef grid_interp.intptr_t intptr_t

DEF MAX_DIM = 64

DEF BOUNDED = 0
DEF REFLECTED = 1
DEF CYCLIC = 2
DEF DISCRETE = 3

cdef object bin_type_map = dict(B=BOUNDED,
                                R=REFLECTED,
                                C=CYCLIC,
                                D=DISCRETE)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def interp1d(np.ndarray[DOUBLE] X not None,
             double lower, double upper,
             np.ndarray[DOUBLE] mesh not None,
             np.ndarray[DOUBLE] values not None,
             str s_bin_type,
             np.ndarray[DOUBLE] out not None
             ):
    cdef:
        Py_ssize_t i
        int nobs = X.shape[0]
        int M = mesh.shape[0]
        double span = upper - lower
        double rem, delta
        double val, cur_val
        np.npy_intp init_inf, init_sup
        np.npy_intp inf, sup, cur
        int bin_type
        double *data = <double*>np.PyArray_DATA(mesh)

    try:
        bin_type = bin_type_map[s_bin_type]
    except KeyError as err:
        raise ValueError('Error, invalid bin type: {0}'.format(err.args[0]))

    if bin_type == CYCLIC:
        if lower < mesh[0]: # Put the 'cyclic' part at the right end
            rem = mesh[0]-lower
            lower += rem
            upper += rem
        init_inf = 0
        init_sup = M
    elif bin_type == DISCRETE:
        init_inf = 0
        init_sup = M-1
    else:
        init_inf = -1
        init_sup = M

    for i in range(nobs):
        val = X[i]
        if bin_type == CYCLIC:
            if val < lower:
                rem = fmod(lower - val, span)
                val = upper - rem
            if val >= upper:
                rem = fmod(val - upper, span)
                val = lower + rem
        elif bin_type == REFLECTED:
            if val < lower:
                rem = fmod(lower - val, 2*span)
                if rem < span:
                    val = lower + rem
                else:
                    val = upper - rem + span
            elif val > upper:
                rem = fmod(val - upper, 2*span)
                if rem < span:
                    val = upper - rem
                else:
                    val = lower + rem - span
            if val <= mesh[0]:
                out[i] = values[0]
                continue
            elif val >= mesh[M-1]:
                out[i] = values[M-1];
                continue
        elif bin_type == BOUNDED:
            if val <= mesh[0]:
                out[i] = values[0]
                continue
            elif val >= mesh[M-1]:
                out[i] = values[M-1]
                continue
        else: # DISCRETE
            val = round(val)
            if val < lower or val > upper:
                out[i] = NAN
                continue # Nothing to do ... ignore the sample

        # Search for the sample in the mesh
        inf = binary_search(val, data, init_inf, init_sup)

        if bin_type == DISCRETE:
            if mesh[inf] == val:
                out[i] = values[inf]
            else:
                out[i] = values[inf+1]
        elif bin_type == CYCLIC:
            if inf >= M-1:
                sup = 0
                delta = upper - mesh[inf]
            else:
                sup = inf+1
                delta = mesh[sup] - mesh[inf]
            rem = (val - mesh[inf]) / delta
            out[i] = (1-rem)*values[inf] + rem*values[sup]
        else: # BOUNDED or REFLECTED
            if inf < 0:
                out[i] = values[0]
            elif inf >= M-1:
                out[i] = values[inf]
            else:
                rem = (val - mesh[inf]) / (mesh[inf+1] - mesh[inf])
                out[i] = (1-rem)*values[inf] + rem*values[inf+1]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def interpnd(np.ndarray[DOUBLE, ndim=2] X not None,
             np.ndarray[DOUBLE] _lower,
             np.ndarray[DOUBLE] _upper,
             list mesh not None,
             object values not None,
             str s_bin_type,
             np.ndarray[DOUBLE] out not None
             ):
    cdef:
        Py_ssize_t i
        int nobs = X.shape[0]
        uintptr_t D = X.shape[1]
        double* meshes[MAX_DIM]
        intptr_t M[MAX_DIM]
        intptr_t init_inf[MAX_DIM]
        intptr_t init_sup[MAX_DIM]
        intptr_t inf[MAX_DIM]
        intptr_t sup[MAX_DIM]
        double lower[MAX_DIM]
        double upper[MAX_DIM]
        double span[MAX_DIM]
        uintptr_t corner, d, pos
        uintptr_t nb_corners = 1 << D
        int bin_types[MAX_DIM]
        double rem[MAX_DIM]
        double val[MAX_DIM]
        int fail
        double tmp, wc, acc, delta
        void *data = np.PyArray_DATA(values)
        np.npy_intp *strides = np.PyArray_STRIDES(values)

    for d in range(D):
        try:
            bin_types[d] = bin_type_map[s_bin_type[d]]
        except KeyError as err:
            raise ValueError('Error, invalid bin type: {0}'.format(err.args[0]))

        meshes[d] = <double*>np.PyArray_DATA(mesh[d])
        M[d] = np.PyArray_DIMS(mesh[d])[0]

        lower[d] = _lower[d]
        upper[d] = _upper[d]
        span[d] = upper[d] - lower[d]
        if bin_types[d] == CYCLIC:
            if lower[d] < meshes[d][0]: # Put the 'cyclic' part at the right end
                tmp = meshes[d][0]-lower[d]
                lower[d] += tmp
                upper[d] += tmp
            init_inf[d] = 0
            init_sup[d] = M[d]
        elif bin_types[d] == DISCRETE:
            init_inf[d] = 0
            init_sup[d] = M[d]-1
        else:
            init_inf[d] = -1
            init_sup[d] = M[d]

    for i in range(nobs):
        fail = 0
        for d in range(D):
            val[d] = X[i,d]
            sup[d] = inf[d] = -1
            if bin_types[d] == CYCLIC:
                if val[d] < lower[d]:
                    tmp = fmod(lower[d] - val[d], span[d])
                    val[d] = upper[d] - tmp
                if val[d] >= upper[d]:
                    tmp = fmod(val[d] - upper[d], span[d])
                    val[d] = lower[d] + tmp
            elif bin_types[d] == REFLECTED:
                if val[d] < lower[d]:
                    tmp = fmod(lower[d] - val[d], 2*span[d])
                    if tmp < span[d]:
                        val[d] = lower[d] + tmp
                    else:
                        val[d] = upper[d] - tmp + span[d]
                elif val[d] > upper[d]:
                    tmp = fmod(val[d] - upper[d], 2*span[d])
                    if tmp < span[d]:
                        val[d] = upper[d] - tmp
                    else:
                        val[d] = lower[d] + tmp - span[d]
                if val[d] < meshes[d][0]:
                    sup[d] = inf[d] = 0
                    rem[d] = 0
                elif val[d] >= meshes[d][M[d]-1]:
                    sup[d] = inf[d] = M[d]-1
                    rem[d] = 0
            elif bin_types[d] == BOUNDED:
                if val[d] <= meshes[d][0]:
                    sup[d] = inf[d] = 0
                    rem[d] = 0
                elif val[d] >= meshes[d][M[d]-1]:
                    inf[d] = M[d]-1
                    sup[d] = 0
                    rem[d] = 0
            else: # DISCRETE
                val[d] = round(val[d])
                if val[d] < lower[d] or val[d] > upper[d]:
                    out[i] = NAN
                    fail = 1
                    break # Nothing to do ... ignore the sample
        if fail: continue

        for d in range(D):
            # Search for the sample in the mesh
            inf[d] = binary_search(val[d], meshes[d], init_inf[d], init_sup[d])

            if bin_types[d] == DISCRETE:
                sup[d] = inf[d]+1
                if meshes[d][inf[d]] == val[d]:
                    rem[d] = 0
                else:
                    rem[d] = 1
            elif bin_types[d] == CYCLIC:
                if inf[d] >= M[d]-1:
                    sup[d] = 0
                    delta = upper[d] - meshes[d][inf[d]]
                else:
                    sup[d] = inf[d]+1
                    delta = meshes[d][sup[d]] - meshes[d][inf[d]]
                rem[d] = (val[d] - meshes[d][inf[d]]) / delta
            else: # BOUNDED or REFLECTED
                if inf[d] < 0:
                    inf[d] = sup[d] = 0
                    rem[d] = 0
                elif inf[d] >= M[d]-1:
                    inf[d] = sup[d] = M[d]-1
                    rem[d] = 0
                else:
                    sup[d] = inf[d]+1
                    rem[d] = (val[d] - meshes[d][inf[d]]) / (meshes[d][inf[d]+1] - meshes[d][inf[d]])

        acc = 0
        # This uses the binary representation of the corner id (from 0 to 2**d-1) to identify where it is
        # for each bit: 0 means lower index, 1 means upper index
        # This means we are limited by the number of bits in Py_ssize_t. But also that we couldn't possibly allocate 
        # an array too big for this to work.
        for corner in range(nb_corners):
            pos = 0
            wc = 1
            for d in range(D):
                if corner & 1:
                    wc *= 1-rem[d]
                    pos += strides[d]*inf[d]
                else:
                    wc *= rem[d]
                    pos += strides[d]*sup[d]
                if wc == 0:
                    break
                corner >>= 1
            acc += wc*(<double*>(data+pos))[0]
        out[i] = acc