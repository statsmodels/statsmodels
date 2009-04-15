import numpy as np

# Taken from python doc site, exists in python2.6
def combinations(iterable, r):
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = range(r)
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)

def complex(maximal=[(0, 3, 2, 7),
                     (0, 6, 2, 7),
                     (0, 7, 5, 4),
                     (0, 7, 5, 1),
                     (0, 7, 4, 6),
                     (0, 3, 1, 7)],
            vertices=None):

    """
    Take a list of maximal simplices (by
    default a triangulation of a cube into 6 tetrahedra) and
    computes all faces, edges, vertices.

    If vertices is not None, then the
    vertices in 'maximal' are replaced with
    these vertices, by index.
    """

    faces = {}

    l = [len(list(x)) for x in maximal]
    for i in range(np.max(l)):
        faces[i+1] = set([])

    for simplex in maximal:
        simplex = list(simplex)
        simplex.sort()
        for k in range(1,len(simplex)+1):
            for v in combinations(simplex, k):
                if len(v) == 1:
                    v = v[0]
                faces[k].add(v)
    return faces

def cube_with_strides_center(center=[0,0,0],
                             strides=np.empty((2,2,2), np.bool).strides):
    """
    Cube in an array of voxels with a given center and strides.

    This triangulates a cube with vertices [center[i] + 1].

    The dimension of the cube is determined by len(center)
    which should agree with len(center).

    The allowable dimensions are [1,2,3].

    Inputs:
    =======

    center : [int]

    strides : [int]

    Outputs:
    ========

    complex : {}
         A dictionary with integer keys representing a simplicial
         complex. The vertices of the simplicial complex
         are the indices of the corners of the cube
         in a 'flattened' array with specified strides.

    """

    d = len(center)
    if len(strides) != d:
        raise ValueError, 'center and strides must have the same length'

    if d == 3:
        maximal = [(0, 3, 2, 7),
                   (0, 6, 2, 7),
                   (0, 7, 5, 4),
                   (0, 7, 5, 1),
                   (0, 7, 4, 6),
                   (0, 3, 1, 7)]

        vertices = []
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    vertices.append((center[0]+i)*strides[0] +
                                    (center[1]+j)*strides[1] +
                                    (center[2]+k)*strides[2])
    elif d == 2:
        maximal = [(0,1,3), (0,2,3)]
        vertices = []
        for j in range(2):
            for i in range(2):
                    vertices.append((center[0]+i)*strides[0] +
                                    (center[1]+j)*strides[1])
    elif d == 1:
        maximal = [(0,1)]
        vertices = [center[0],center[0]+strides[0]]

    mm = []
    for m in maximal:
        nm = [vertices[j] for j in m]
        mm.append(nm)
    maximal = [tuple([vertices[j] for j in m]) for m in maximal]
    return complex(maximal)

def join_complexes(*complexes):
    """
    Join a sequence of simplicial complexes.
    Returns the union of all the particular faces.
    """
    faces = {}

    nmax = np.array([len(c) for c in complexes]).max()
    for i in range(nmax):
        faces[i+1] = set([])
    for c in complexes:
        for i in range(nmax):
            if c.has_key(i+1):
                faces[i+1] = faces[i+1].union(c[i+1])
    return faces

def decompose3d(shape, dim=4):
    """
    Return all (dim-1)-dimensional simplices in a triangulation
    of a cube of a given shape. The vertices in the triangulation
    are indices in a 'flattened' array of the specified shape.
    """

    # First do the interior contributions.
    # We first figure out which vertices, edges, triangles, tetrahedra
    # are uniquely associated with an interior voxel

    unique = {}
    strides = np.empty(shape, np.bool).strides
    union = join_complexes(*[cube_with_strides_center((0,0,-1), strides),
                             cube_with_strides_center((0,-1,0), strides),
                             cube_with_strides_center((0,-1,-1), strides),
                             cube_with_strides_center((-1,0,0), strides),
                             cube_with_strides_center((-1,0,-1), strides),
                             cube_with_strides_center((-1,-1,0), strides),
                             cube_with_strides_center((-1,-1,-1), strides)])

    c = cube_with_strides_center((0,0,0), strides)
    for i in range(4):
        unique[i+1] = c[i+1].difference(union[i+1])

    if unique.has_key(dim) and dim > 1:
        d = unique[dim]

        for i in range(shape[0]-1):
            for j in range(shape[1]-1):
                for k in range(shape[2]-1):
                    index = i*strides[0]+j*strides[1]+k*strides[2]
                    for l in d:
                        yield [index+ii for ii in l]

    # There are now contributions from three two-dimensional faces

    for _strides, _shape in zip([(strides[0], strides[1]),
                                 (strides[0], strides[2]),
                                 (strides[1], strides[2])],
                                [(shape[0], shape[1]),
                                 (shape[0], shape[2]),
                                 (shape[1], shape[2])]):

        unique = {}
        union = join_complexes(*[cube_with_strides_center((0,-1), _strides),
                                 cube_with_strides_center((-1,0), _strides),
                                 cube_with_strides_center((-1,-1), _strides)])

        c = cube_with_strides_center((0,0), _strides)
        for i in range(3):
            unique[i+1] = c[i+1].difference(union[i+1])

        if unique.has_key(dim) and dim > 1:
            d = unique[dim]

            for i in range(_shape[0]-1):
                for j in range(_shape[1]-1):
                        index = i*_strides[0]+j*_strides[1]
                        for l in d:
                            yield [index+ii for ii in l]

    # Finally the one-dimensional faces

    for _stride, _shape in zip(strides, shape):

        unique = {}
        union = cube_with_strides_center((-1,), [_stride])
        c = cube_with_strides_center((0,), [_stride])
        for i in range(2):
            unique[i+1] = c[i+1].difference(union[i+1])

        if unique.has_key(dim) and dim > 1:
            d = unique[dim]

            for i in range(_shape-1):
                index = i*_stride
                for l in d:
                    yield [index+ii for ii in l]

    if dim == 1:
        for i in range(np.product(shape)):
            yield i


def decompose2d(shape, dim=3):
    """
    Return all (dim-1)-dimensional simplices in a triangulation
    of a square of a given shape. The vertices in the triangulation
    are indices in a 'flattened' array of the specified shape.
    """

    # First do the interior contributions.
    # We first figure out which vertices, edges, triangles
    # are uniquely associated with an interior pixel

    unique = {}
    strides = np.empty(shape, np.bool).strides
    union = join_complexes(*[cube_with_strides_center((0,-1), strides),
                             cube_with_strides_center((-1,0), strides),
                             cube_with_strides_center((-1,-1), strides)])
    c = cube_with_strides_center((0,0), strides)
    for i in range(3):
        unique[i+1] = c[i+1].difference(union[i+1])

    if unique.has_key(dim) and dim > 1:
        d = unique[dim]

        for i in range(shape[0]-1):
            for j in range(shape[1]-1):
                    index = i*strides[0]+j*strides[1]
                    for l in d:
                        yield [index+ii for ii in l]

    # Now, the one-dimensional faces

    for _stride, _shape in zip(strides, shape):

        unique = {}
        union = cube_with_strides_center((-1,), [_stride])
        c = cube_with_strides_center((0,), [_stride])
        for i in range(2):
            unique[i+1] = c[i+1].difference(union[i+1])

        if unique.has_key(dim) and dim > 1:
            d = unique[dim]

            for i in range(_shape-1):
                index = i*_stride
                for l in d:
                    yield [index+ii for ii in l]

    if dim == 1:
        for i in range(np.product(shape)):
            yield i

def test_EC3(shape):

    ts = 0
    fs = 0
    es = 0
    vs = 0
    ec = 0

    for t in decompose3d(shape, dim=4):
        ec -= 1; ts += 1
    for f in decompose3d(shape, dim=3):
        ec += 1; fs += 1
    for e in decompose3d(shape, dim=2):
        ec -= 1; es += 1
    for v in decompose3d(shape, dim=1):
        ec += 1; vs += 1
    return ts, fs, es, vs, ec

def test_EC2(shape):

    fs = 0
    es = 0
    vs = 0
    ec = 0

    for f in decompose2d(shape, dim=3):
        ec += 1; fs += 1
    for e in decompose2d(shape, dim=2):
        ec -= 1; es += 1
    for v in decompose2d(shape, dim=1):
        ec += 1; vs += 1
    return fs, es, vs, ec
