cimport cython
cimport numpy as np
import numpy as np
from libc.math cimport floor, fmod, ceil

np.import_array()

cdef inline double round(double a):
    if a < 0.0:
        return ceil(a - 0.5)
    return floor(a + 0.5)

ctypedef np.float64_t DOUBLE
ctypedef np.int_t INT
ctypedef np.uint8_t uint8_t

DEF BOUNDED = 0
DEF REFLECTED = 1
DEF CYCLIC = 2
DEF DISCRETE = 3

cdef object bin_type_map = dict(b=BOUNDED,
                                r=REFLECTED,
                                c=CYCLIC,
                                d=DISCRETE)

DEF bin_type_error=str("Error, letter '{0}' is invalid: must be one of 'b', 'c', 'r' or 'd'")

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.embedsignature(True)
def fast_linbin(np.ndarray[DOUBLE] X not None,
                double a, double b,
                np.ndarray[DOUBLE] grid not None,
                np.ndarray[DOUBLE] weights not None,
                str s_bin_type):
    cdef:
        Py_ssize_t i
        int nobs = X.shape[0]
        int M = grid.shape[0]
        np.ndarray[DOUBLE] mesh
        double delta
        double shift
        double rem
        double val, dval
        double lower
        double upper
        double w
        int base_idx
        int N
        int has_weight = len(weights) > 0
        int bin_type
        object bounds

    try:
        bin_type = bin_type_map[s_bin_type]
    except KeyError as err:
        raise ValueError(bin_type_error.format(s_bin_type))

    if bin_type == CYCLIC:
        lower = 0
        upper = M
        delta = (b - a) / M
        shift = -a-delta/2
    elif bin_type == DISCRETE:
        shift = -a
        lower = 0
        upper = M-1
        delta = (b - a) / (M-1)
    else: # REFLECTED of BOUNDED
        lower = -0.5
        upper = M-0.5
        delta = (b - a) / M
        shift = -a-delta/2

    for i in range(nobs):
        val = (X[i] + shift) / delta
        if bin_type == CYCLIC:
            if val < lower:
                rem = fmod(lower - val, M)
                val = upper - rem
            if val >= upper:
                rem = fmod(val - upper, M)
                val = lower + rem
        elif bin_type == REFLECTED:
            if val < lower:
                rem = fmod(lower - val, 2*M)
                if rem < M:
                    val = lower + rem
                else:
                    val = upper - rem + M
            elif val > upper:
                rem = fmod(val - upper, 2*M)
                if rem < M:
                    val = upper - rem
                else:
                    val = lower + rem - M
        elif bin_type == BOUNDED:
            if val < lower or val > upper:
                continue # Skip this sample
        else: # DISCRETE
            val = round(val)
            if val < lower or val > upper:
                continue

        base_idx = <int> floor(val);
        if has_weight:
            w = weights[i]
        else:
            w = 1.
        if bin_type == DISCRETE:
            grid[base_idx] += w
        else:
            rem = val - base_idx
            if bin_type == CYCLIC:
                grid[base_idx] += (1-rem)*w
                if base_idx == M-1:
                    grid[0] += rem*w
                else:
                    grid[base_idx+1] += rem*w
            else: # BOUNDED or REFLECTED
                if base_idx < 0:
                    grid[0] += w
                elif base_idx >= M-1:
                    grid[base_idx] += w
                else:
                    grid[base_idx] += (1-rem)*w
                    grid[base_idx+1] += rem*w

    if bin_type == DISCRETE:
        mesh = np.linspace(a, b, M)
        bounds = [a, b]
    else:
        mesh = np.linspace(a+delta/2, b-delta/2, M)
        bounds = [a, b]

    return mesh, bounds


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.embedsignature(True)
def fast_bin(np.ndarray[DOUBLE] X not None,
             double a, double b,
             np.ndarray[DOUBLE] grid not None,
             np.ndarray[DOUBLE] weights not None,
             str s_bin_type):
    r"""
    Linear Binning as described in Fan and Marron (1994)

    :param X ndarray: Input data
    :param a float: Lowest value to consider
    :param b float: Highest valus to consider
    :param M int: Number of bins
    :param weights ndarray: Array of same size as X with weights for each point, or None if all weights are 1
    :param cyclic bool: Consider the data cyclic or not

    :Returns: The weights in each bin

    For a point :math:`x` between bins :math:`b_i` and :math:`b_{i+1}` at positions :math:`p_i` and :math:`p_{i+1}`, the 
    bins will be updated as:

    .. math::

        b_i = b_i + \frac{b_{i+1} - x}{b_{i+1} - b_i}

        b_{i+1} = b_{i+1} + \frac{x - b_i}{b_{i+1} - b_i}

    By default the bins will be placed at :math:`\{a+\delta/2, \ldots, a+k \delta + \delta/1, \ldots b-\delta/2\}` with 
    :math:`delta = \frac{M-1}{b-a}`.

    If cyclic is true, then the bins are placed at :math:`\{a, \ldots, a+k \delta, \ldots, b-\delta\}` with 
    :math:`\delta = \frac{M}{b-a}` and there is a virtual bin in :math:`b` which is fused with :math:`a`.

    """
    cdef:
        Py_ssize_t i
        int nobs = X.shape[0]
        int M = grid.shape[0]
        double delta
        double shift
        double rem
        double val
        double lower
        double upper
        double w
        int base_idx
        int N
        int has_weight = len(weights) > 0
        int bin_type

    try:
        bin_type = bin_type_map[s_bin_type]
    except KeyError as err:
        raise ValueError(bin_type_error.format(s_bin_type))

    if bin_type == DISCRETE:
        delta = (b - a)/(M - 1)
        upper = M - 1
    else:
        delta = (b - a)/M
        upper = M
    shift = -a
    lower = 0

    for i in range(nobs):
        val = (X[i] + shift) / delta

        if bin_type == CYCLIC:
            if val < lower:
                rem = fmod(lower - val, M)
                val = upper - rem
            if val >= upper:
                rem = fmod(val - upper, M)
                val = lower + rem
        elif bin_type == REFLECTED:
            if val < lower:
                rem = fmod(lower - val, 2*M)
                if rem < M:
                    val = lower + rem
                else:
                    val = upper - rem + M
            elif val > upper:
                rem = fmod(val - upper, 2*M)
                if rem < M:
                    val = upper - rem
                else:
                    val = lower + rem - M
        elif bin_type == BOUNDED:
            if val < lower or val > upper:
                continue # Skip this sample
        else: # DISCRETE
            val = round(val)
            if val < lower or val > upper:
                continue # Skip this sample


        base_idx = <int> floor(val);
        if has_weight:
            w = weights[i]
        else:
            w = 1.
        if base_idx == M:
            base_idx -= 1
        grid[base_idx] += w

    if bin_type == DISCRETE:
        return np.linspace(a, b, M), [a, b]
    else:
        return np.linspace(a+delta/2, b-delta/2, M), [a, b]

# specialized version of fast_linbin_nd for 2 and 3d


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(True)
def fast_linbin_2d(np.ndarray[DOUBLE, ndim=2] X not None,
                    np.ndarray[DOUBLE] a not None,
                    np.ndarray[DOUBLE] b not None,
                    np.ndarray[DOUBLE, ndim=2] grid,
                    np.ndarray[DOUBLE] weights not None,
                    str s_bin_types):
    cdef:
        Py_ssize_t i, d, c, N
        int nobs = X.shape[0]
        object mesh
        object bounds
        double shift[2]
        double rem[2]
        double val[2]
        double lower[2]
        double upper[2]
        double delta[2]
        double w
        int base_idx[2]
        int next_idx[2]
        int idx[2]
        int is_out
        Py_ssize_t nb_corner = 4
        double wc
        Py_ssize_t pos
        #uint8_t *data = <uint8_t*>(np.PyArray_DATA(grid))
        #np.npy_intp *strides = np.PyArray_STRIDES(grid)
        np.npy_intp *M = np.PyArray_DIMS(grid)
        int bin_types[2]
        int has_weight = weights.shape[0] > 0

    for d in range(2):
        try:
            bin_types[d] = bin_type_map[s_bin_types[d]]
        except KeyError as err:
            raise ValueError(bin_type_error.format(s_bin_types[d]))

        if bin_types[d] == CYCLIC:
            delta[d] = (b[d] - a[d]) / M[d]
            shift[d] = -a[d]-delta[d]/2
            lower[d] = 0
            upper[d] = M[d]
        elif bin_types[d] == DISCRETE:
            delta[d] = (b[d] - a[d]) / (M[d] - 1)
            shift[d] = -a[d]
            lower[d] = 0
            upper[d] = M[d]-1
        else:
            delta[d] = (b[d] - a[d]) / M[d]
            shift[d] = -a[d]-delta[d]/2
            lower[d] = -0.5
            upper[d] = M[d]-0.5

    for i in range(nobs):
        is_out = 0
        for d in range(2):
            val[d] = (X[i,d] + shift[d]) / delta[d]

            if bin_types[d] == CYCLIC:
                if val[d] < lower[d]:
                    rem[d] = fmod(lower[d] - val[d], M[d])
                    val[d] = upper[d] - rem[d]
                if val[d] >= upper[d]:
                    rem[d] = fmod(val[d] - upper[d], M[d])
                    val[d] = lower[d] + rem[d]
            elif bin_types[d] == REFLECTED:
                if val[d] < lower[d]:
                    rem[d] = fmod(lower[d] - val[d], 2*M[d])
                    if rem[d] < M[d]:
                        val[d] = lower[d] + rem[d]
                    else:
                        val[d] = upper[d] - rem[d] + M[d]
                elif val[d] > upper[d]:
                    rem[d] = fmod(val[d] - upper[d], 2*M[d])
                    if rem[d] < M[d]:
                        val[d] = upper[d] - rem[d]
                    else:
                        val[d] = lower[d] + rem[d] - M[d]
            elif bin_types[d] == BOUNDED:
                if val[d] < lower[d] or val[d] > upper[d]:
                    is_out = 1
                    break
            else: # DISCRETE
                val[d] = round(val[d])
                if val[d] < lower[d] or val[d] > upper[d]:
                    is_out = 1
                    break

        if is_out: continue
        if has_weight:
            w = weights[i]
        else:
            w = 1.

        for d in range(2):
            base_idx[d] = <int> floor(val[d])
            if bin_types[d] == DISCRETE:
                rem[d] = 0
            else:
                rem[d] = val[d] - base_idx[d]
            if bin_types[d] == CYCLIC:
                if base_idx[d] == M[d]-1:
                    next_idx[d] = 0
                else:
                    next_idx[d] = base_idx[d]+1
            else:
                if base_idx[d] < 0:
                    base_idx[d] = 0
                    next_idx[d] = 1
                    rem[d] = 0
                elif base_idx[d] >= M[d]-1:
                    rem[d] = 0
                    next_idx[d] = 0
                else:
                    next_idx[d] = base_idx[d]+1

        # This uses the binary representation of the corner id (from 0 to 2**d-1) to identify where it is
        # for each bit: 0 means lower index, 1 means upper index
        # This means we are limited by the number of bits in Py_ssize_t. But also that we couldn't possibly allocate 
        # an array too big for this to work.
        for c in range(nb_corner):
            wc = w
            if c & 1:
                wc *= 1 - rem[0]
                idx[0] = base_idx[0]
            else:
                wc *= rem[0]
                idx[0] = next_idx[0]
            if c & 2:
                wc *= 1 - rem[1]
                idx[1] = base_idx[1]
            else:
                wc *= rem[1]
                idx[1] = next_idx[1]
            grid[idx[0],idx[1]] += wc

    mesh = [None]*2
    bounds = np.zeros((2,2), dtype=np.float)
    for d in range(2):
        if bin_types[d] == DISCRETE:
            mesh[d] = np.linspace(a[d], b[d], M[d])
            bounds[d,0] = a[d]
            bounds[d,1] = b[d]
        else: # BOUNDED or REFLECTED
            mesh[d] = np.linspace(a[d]+delta[d]/2, b[d]-delta[d]/2, M[d])
            bounds[d,0] = a[d]
            bounds[d,1] = b[d]

    return mesh, bounds



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(True)
def fast_linbin_3d(np.ndarray[DOUBLE, ndim=2] X not None,
                    np.ndarray[DOUBLE] a not None,
                    np.ndarray[DOUBLE] b not None,
                    np.ndarray[DOUBLE,ndim=3] grid not None,
                    np.ndarray[DOUBLE] weights not None,
                    str s_bin_types):
    cdef:
        Py_ssize_t i, d, c, N
        int nobs = X.shape[0]
        object mesh
        object bounds
        double shift[3]
        double rem[3]
        double val[3]
        double lower[3]
        double upper[3]
        double delta[3]
        double w
        int base_idx[3]
        int next_idx[3]
        int idx[3]
        int is_out
        Py_ssize_t nb_corner = 1 << 3
        double wc
        Py_ssize_t pos
        #uint8_t *data = <uint8_t*>(np.PyArray_DATA(grid))
        #np.npy_intp *strides = np.PyArray_STRIDES(grid)
        np.npy_intp *M = np.PyArray_DIMS(grid)
        int bin_types[3]
        int has_weight = weights.shape[0] > 0

    for d in range(3):
        try:
            bin_types[d] = bin_type_map[s_bin_types[d]]
        except KeyError as err:
            raise ValueError(bin_type_error.format(s_bin_types[d]))

        if bin_types[d] == CYCLIC:
            delta[d] = (b[d] - a[d]) / M[d]
            shift[d] = -a[d]-delta[d]/2
            lower[d] = 0
            upper[d] = M[d]
        elif bin_types[d] == DISCRETE:
            delta[d] = (b[d] - a[d]) / (M[d] - 1)
            shift[d] = -a[d]
            lower[d] = 0
            upper[d] = M[d]-1
        else:
            delta[d] = (b[d] - a[d]) / M[d]
            shift[d] = -a[d]-delta[d]/2
            lower[d] = -0.5
            upper[d] = M[d]-0.5

    for i in range(nobs):
        is_out = 0
        for d in range(3):
            val[d] = (X[i,d] + shift[d]) / delta[d]

            if bin_types[d] == CYCLIC:
                if val[d] < lower[d]:
                    rem[d] = fmod(lower[d] - val[d], M[d])
                    val[d] = upper[d] - rem[d]
                if val[d] >= upper[d]:
                    rem[d] = fmod(val[d] - upper[d], M[d])
                    val[d] = lower[d] + rem[d]
            elif bin_types[d] == REFLECTED:
                if val[d] < lower[d]:
                    rem[d] = fmod(lower[d] - val[d], 2*M[d])
                    if rem[d] < M[d]:
                        val[d] = lower[d] + rem[d]
                    else:
                        val[d] = upper[d] - rem[d] + M[d]
                elif val[d] > upper[d]:
                    rem[d] = fmod(val[d] - upper[d], 2*M[d])
                    if rem[d] < M[d]:
                        val[d] = upper[d] - rem[d]
                    else:
                        val[d] = lower[d] + rem[d] - M[d]
            elif bin_types[d] == BOUNDED:
                if val[d] < lower[d] or val[d] > upper[d]:
                    is_out = 1
                    break
            else: # DISCRETE
                val[d] = round(val[d])
                if val[d] < lower[d] or val[d] > upper[d]:
                    is_out = 1
                    break

        if is_out: continue
        if has_weight:
            w = weights[i]
        else:
            w = 1.

        for d in range(3):
            base_idx[d] = <int> floor(val[d])
            if bin_types[d] == DISCRETE:
                rem[d] = 0
            else:
                rem[d] = val[d] - base_idx[d]
            if bin_types[d] == CYCLIC:
                if base_idx[d] == M[d]-1:
                    next_idx[d] = 0
                else:
                    next_idx[d] = base_idx[d]+1
            else:
                if base_idx[d] < 0:
                    base_idx[d] = 0
                    next_idx[d] = 1
                    rem[d] = 0
                elif base_idx[d] >= M[d]-1:
                    rem[d] = 0
                    next_idx[d] = 0
                else:
                    next_idx[d] = base_idx[d]+1

        # This uses the binary representation of the corner id (from 0 to 2**d-1) to identify where it is
        # for each bit: 0 means lower index, 1 means upper index
        # This means we are limited by the number of bits in Py_ssize_t. But also that we couldn't possibly allocate 
        # an array too big for this to work.
        for c in range(nb_corner):
            wc = w
            if c & 1:
                wc *= 1 - rem[0]
                idx[0] = base_idx[0]
            else:
                wc *= rem[0]
                idx[0] = next_idx[0]
            if c & 2:
                wc *= 1 - rem[1]
                idx[1] = base_idx[1]
            else:
                wc *= rem[1]
                idx[1] = next_idx[1]
            if c & 4:
                wc *= 1 - rem[2]
                idx[2] = base_idx[2]
            else:
                wc *= rem[2]
                idx[2] = next_idx[2]
            grid[idx[0],idx[1],idx[2]] += wc

    mesh = [None]*3
    bounds = np.zeros((3,2), dtype=np.float)
    for d in range(3):
        if bin_types[d] == DISCRETE:
            mesh[d] = np.linspace(a[d], b[d], M[d])
            bounds[d,0] = a[d]
            bounds[d,1] = b[d]
        else: # BOUNDED or REFLECTED
            mesh[d] = np.linspace(a[d]+delta[d]/2, b[d]-delta[d]/2, M[d])
            bounds[d,0] = a[d]
            bounds[d,1] = b[d]

    return mesh, bounds


# Note: this define is NOT the limiting factor in the algorithm. See the code for details.
# Ideally, this constant should be the number of bits in Py_ssize_t
DEF MAX_DIM = 64

MAX_DIMENSION = min(MAX_DIM, 8*sizeof(Py_ssize_t))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(True)
def fast_linbin_nd(np.ndarray[DOUBLE, ndim=2] X not None,
                    np.ndarray[DOUBLE] a not None,
                    np.ndarray[DOUBLE] b not None,
                    object grid,
                    np.ndarray[DOUBLE] weights not None,
                    str s_bin_types):
    cdef:
        Py_ssize_t D = a.shape[0]
        np.npy_intp nD = D
        Py_ssize_t i, d, c, N
        int nobs = X.shape[0]
        object mesh
        object bounds
        double shift[MAX_DIM]
        double rem[MAX_DIM]
        double val[MAX_DIM]
        double lower[MAX_DIM]
        double upper[MAX_DIM]
        double delta[MAX_DIM]
        double w
        int base_idx[MAX_DIM]
        int next_idx[MAX_DIM]
        int is_out
        Py_ssize_t nb_corner = 1 << D
        double wc
        Py_ssize_t pos
        uint8_t *data = <uint8_t*>(np.PyArray_DATA(grid))
        np.npy_intp *strides = np.PyArray_STRIDES(grid)
        np.npy_intp *M = np.PyArray_DIMS(grid)
        int bin_types[MAX_DIM]
        int has_weight = weights.shape[0] > 0

    for d in range(D):
        try:
            bin_types[d] = bin_type_map[s_bin_types[d]]
        except KeyError as err:
            raise ValueError(bin_type_error.format(s_bin_types[d]))

        if bin_types[d] == CYCLIC:
            delta[d] = (b[d] - a[d]) / M[d]
            shift[d] = -a[d]-delta[d]/2
            lower[d] = 0
            upper[d] = M[d]
        elif bin_types[d] == DISCRETE:
            delta[d] = (b[d] - a[d]) / (M[d] - 1)
            shift[d] = -a[d]
            lower[d] = 0
            upper[d] = M[d]-1
        else:
            delta[d] = (b[d] - a[d]) / M[d]
            shift[d] = -a[d]-delta[d]/2
            lower[d] = -0.5
            upper[d] = M[d]-0.5

    for i in range(nobs):
        is_out = 0
        for d in range(D):
            val[d] = (X[i,d] + shift[d]) / delta[d]

            if bin_types[d] == CYCLIC:
                if val[d] < lower[d]:
                    rem[d] = fmod(lower[d] - val[d], M[d])
                    val[d] = upper[d] - rem[d]
                if val[d] >= upper[d]:
                    rem[d] = fmod(val[d] - upper[d], M[d])
                    val[d] = lower[d] + rem[d]
            elif bin_types[d] == REFLECTED:
                if val[d] < lower[d]:
                    rem[d] = fmod(lower[d] - val[d], 2*M[d])
                    if rem[d] < M[d]:
                        val[d] = lower[d] + rem[d]
                    else:
                        val[d] = upper[d] - rem[d] + M[d]
                elif val[d] > upper[d]:
                    rem[d] = fmod(val[d] - upper[d], 2*M[d])
                    if rem[d] < M[d]:
                        val[d] = upper[d] - rem[d]
                    else:
                        val[d] = lower[d] + rem[d] - M[d]
            elif bin_types[d] == BOUNDED:
                if val[d] < lower[d] or val[d] > upper[d]:
                    is_out = 1
                    break
            else: # DISCRETE
                val[d] = round(val[d])
                if val[d] < lower[d] or val[d] > upper[d]:
                    is_out = 1
                    break

        if is_out: continue
        if has_weight:
            w = weights[i]
        else:
            w = 1.

        for d in range(D):
            base_idx[d] = <int> floor(val[d])
            if bin_types[d] == DISCRETE:
                rem[d] = 0
            else:
                rem[d] = val[d] - base_idx[d]
            if bin_types[d] == CYCLIC:
                if base_idx[d] == M[d]-1:
                    next_idx[d] = 0
                else:
                    next_idx[d] = base_idx[d]+1
            else:
                if base_idx[d] < 0:
                    base_idx[d] = 0
                    next_idx[d] = 1
                    rem[d] = 0
                elif base_idx[d] >= M[d]-1:
                    rem[d] = 0
                    next_idx[d] = 0
                else:
                    next_idx[d] = base_idx[d]+1

        # This uses the binary representation of the corner id (from 0 to 2**d-1) to identify where it is
        # for each bit: 0 means lower index, 1 means upper index
        # This means we are limited by the number of bits in Py_ssize_t. But also that we couldn't possibly allocate 
        # an array too big for this to work.
        for c in range(nb_corner):
            wc = w
            pos = 0
            for d in range(D):
                if c & 1:
                    wc *= 1-rem[d]
                    pos += strides[d]*base_idx[d]
                else:
                    wc *= rem[d]
                    pos += strides[d]*next_idx[d]
                c >>= 1
            (<double*>(data+pos))[0] += wc

    mesh = [None]*D
    bounds = np.zeros((D,2), dtype=np.float)
    for d in range(D):
        if bin_types[d] == DISCRETE:
            mesh[d] = np.linspace(a[d], b[d], M[d])
            bounds[d,0] = a[d]
            bounds[d,1] = b[d]
        else: # BOUNDED or REFLECTED
            mesh[d] = np.linspace(a[d]+delta[d]/2, b[d]-delta[d]/2, M[d])
            bounds[d,0] = a[d]
            bounds[d,1] = b[d]

    return mesh, bounds


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.embedsignature(True)
def fast_bin_nd(np.ndarray[DOUBLE, ndim=2] X not None,
                np.ndarray[DOUBLE] a not None,
                np.ndarray[DOUBLE] b not None,
                object grid,
                np.ndarray[DOUBLE] weights not None,
                str s_bin_types):
    cdef:
        Py_ssize_t i, pos, d
        Py_ssize_t D = X.shape[1]
        int nobs = X.shape[0]
        object mesh
        object bounds
        double delta[MAX_DIM]
        double shift[MAX_DIM]
        double val[MAX_DIM]
        double lower[MAX_DIM]
        double upper[MAX_DIM]
        double w
        int base_idx[MAX_DIM]
        int N, is_out
        int bin_types[MAX_DIM]
        int has_weight = weights.shape[0] > 0
        uint8_t *data = <uint8_t*>(np.PyArray_DATA(grid))
        np.npy_intp *strides = np.PyArray_STRIDES(grid)
        np.npy_intp *M = np.PyArray_DIMS(grid)

    for d in range(D):
        try:
            bin_types[d] = bin_type_map[s_bin_types[d]]
        except KeyError as err:
            raise ValueError(bin_type_error.format(s_bin_types[d]))
        if bin_types[d] == DISCRETE:
            delta[d] = (b[d] - a[d])/(M[d] - 1)
            upper[d] = M[d] - 1
        else:
            delta[d] = (b[d] - a[d])/M[d]
            upper[d] = M[d]
        shift[d] = -a[d]
        lower[d] = 0

    for i in range(nobs):
        is_out = 0
        for d in range(D):
            val[d] = (X[i,d] + shift[d]) / delta[d]
            if bin_types[d] == CYCLIC:
                while val[d] < lower[d]:
                    val[d] += M[d]
                while val[d] > upper[d]:
                    val[d] -= M[d]
            elif bin_types[d] == REFLECTED:
                while val[d] < lower[d] or val[d] > upper[d]:
                    if val[d] < lower[d]:
                        val[d] =  2*lower[d] - val[d]
                    if val[d] > upper[d]:
                        val[d] =  2*upper[d] - val[d]
            elif bin_types[d] == BOUNDED:
                if val[d] < lower[d] or val[d] > upper[d]:
                    is_out = 1
                    break
            else: # DISCRETE
                val[d] = round(val[d])
                if val[d] < lower[d] or val[d] > upper[d]:
                    is_out = 1
                    break
        if is_out: continue
        if has_weight:
            w = weights[i]
        else:
            w = 1.
        pos = 0
        for d in range(D):
            base_idx[d] = <int> floor(val[d])
            if base_idx[d] == M[d]:
                base_idx[d] -= 1
            pos += strides[d]*base_idx[d]
        (<double*>(data+pos))[0] += w

    mesh = [None]*D
    bounds = np.zeros((D,2), dtype=np.float)
    for d in range(D):
        if bin_types[d] == DISCRETE:
            mesh[d] = np.linspace(a[d], b[d], M[d])
            bounds[d,0] = a[d]
            bounds[d,1] = b[d]
        else: # BOUNDED or REFLECTED
            mesh[d] = np.linspace(a[d]+delta[d]/2, b[d]-delta[d]/2, M[d])
            bounds[d,0] = a[d]
            bounds[d,1] = b[d]
    return mesh, bounds
