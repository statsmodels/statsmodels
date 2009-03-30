"""
The estimators for the intrinsic volumes appearing in this module
were partially supported by NSF grant DMS-0405970.

Taylor, J.E. & Worsley, K.J. (2007). "Detecting sparse signal in random fields,
   with an application to brain mapping."
   Journal of the American Statistical Association, 102(479):913-928.

"""

import numpy as np
cimport numpy as np

# local imports

from utils import cube_with_strides_center, join_complexes

DTYPE_float = np.float
ctypedef np.float_t DTYPE_float_t

DTYPE_int = np.int
ctypedef np.int_t DTYPE_int_t

def mu3_tet(np.ndarray[DTYPE_float_t, ndim=2] coords,
            long v0, long v1, long v2, long v3):

  """
  Compute the 3rd intrinsic volume (just volume in this case) of
  a tetrahedron with coordinates [coords[v0], coords[v1], coords[v2], coords[v3]].

  Inputs:
  -------

  coords : ndarray((*,2))
       An array of coordinates of vertices of tetrahedron.

  v0, v1, v2, v3 : int
       Indices for vertices of the tetrahedron.

  Outputs:
  --------

  mu3 : float

  """

  cdef double XTX00 = 0
  cdef double XTX01 = 0
  cdef double XTX02 = 0
  cdef double XTX11 = 0
  cdef double XTX12 = 0
  cdef double XTX22 = 0
  cdef int q, j

  q = coords.shape[0]

  for j in range(q):
      XTX00 += (coords[j,v0] - coords[j,v3]) * (coords[j,v0] - coords[j,v3])
      XTX01 += (coords[j,v0] - coords[j,v3]) * (coords[j,v1] - coords[j,v3])
      XTX02 += (coords[j,v0] - coords[j,v3]) * (coords[j,v2] - coords[j,v3])
      XTX11 += (coords[j,v1] - coords[j,v3]) * (coords[j,v1] - coords[j,v3])
      XTX12 += (coords[j,v1] - coords[j,v3]) * (coords[j,v2] - coords[j,v3])
      XTX22 += (coords[j,v2] - coords[j,v3]) * (coords[j,v2] - coords[j,v3])

  return np.sqrt((XTX00 * (XTX11 * XTX22 - XTX12 * XTX12) -
                XTX01 * (XTX01 * XTX22 - XTX02 * XTX12) +
                XTX02 * (XTX01 * XTX12 - XTX11 * XTX02))) / 6.;


def mu2_tet(np.ndarray[DTYPE_float_t, ndim=2] coords,
            long v0, long v1, long v2, long v3):

  """
  Compute the 2nd intrinsic volume (half the surface area) of
  a tetrahedron with coordinates [coords[v0], coords[v1], coords[v2], coords[v3]].

  Inputs:
  -------

  coords : ndarray((*,2))
       An array of coordinates of vertices of tetrahedron.

  v0, v1, v2, v3 : int
       Indices for vertices of the tetrahedron.

  Outputs:
  --------

  mu2 : float

  """

  cdef double mu = 0

  mu += mu2_tri(coords, v0, v1, v2)
  mu += mu2_tri(coords, v0, v1, v3)
  mu += mu2_tri(coords, v0, v2, v3)
  mu += mu2_tri(coords, v1, v2, v3)
  return mu * 0.5

def mu1_tet(np.ndarray[DTYPE_float_t, ndim=2] coords,
            long v0, long v1, long v2, long v3):

  """
  Compute the 3rd intrinsic volume (sum of external angles * edge lengths) of
  a tetrahedron with coordinates [coords[v0], coords[v1], coords[v2], coords[v3]].

  Inputs:
  -------

  coords : ndarray((*,2))
       An array of coordinates of vertices of tetrahedron.

  v0, v1, v2, v3 : int
       Indices for vertices of the tetrahedron.

  Outputs:
  --------

  mu1 : float

  """

  cdef double XTX00 = 0
  cdef double XTX01 = 0
  cdef double XTX02 = 0
  cdef double XTX03 = 0
  cdef double XTX11 = 0
  cdef double XTX12 = 0
  cdef double XTX13 = 0
  cdef double XTX22 = 0
  cdef double XTX23 = 0
  cdef double XTX33 = 0
  cdef double XTX10, XTX20, XTX30, XTX21, XTX31, XTX32
  cdef int q, j

  q = coords.shape[0]

  for j in range(q):
      XTX00 += coords[j,v0] * coords[j,v0]
      XTX01 += coords[j,v0] * coords[j,v1]
      XTX02 += coords[j,v0] * coords[j,v2]
      XTX03 += coords[j,v0] * coords[j,v3]
      XTX11 += coords[j,v1] * coords[j,v1]
      XTX12 += coords[j,v1] * coords[j,v2]
      XTX13 += coords[j,v1] * coords[j,v3]
      XTX22 += coords[j,v2] * coords[j,v2]
      XTX23 += coords[j,v2] * coords[j,v3]
      XTX33 += coords[j,v3] * coords[j,v3]

  XTX10 = XTX01
  XTX20 = XTX02
  XTX30 = XTX03
  XTX21 = XTX12
  XTX31 = XTX13
  XTX32 = XTX23

  mu = 0
  mu += _mu1_tetface(XTX00, XTX01, XTX11, XTX02, XTX03, XTX12, XTX13, XTX22, XTX23, XTX33)
  mu += _mu1_tetface(XTX00, XTX02, XTX22, XTX01, XTX03, XTX21, XTX23, XTX11, XTX13, XTX33)
  mu += _mu1_tetface(XTX00, XTX03, XTX33, XTX01, XTX02, XTX31, XTX32, XTX11, XTX12, XTX22)
  mu += _mu1_tetface(XTX11, XTX12, XTX22, XTX10, XTX13, XTX20, XTX23, XTX00, XTX03, XTX33)
  mu += _mu1_tetface(XTX11, XTX13, XTX33, XTX10, XTX12, XTX30, XTX32, XTX00, XTX02, XTX22)
  mu += _mu1_tetface(XTX22, XTX23, XTX33, XTX20, XTX21, XTX30, XTX31, XTX00, XTX01, XTX11)

  return mu

cdef double _mu1_tetface(double XTXs0s0,
                         double XTXs0s1,
                         double XTXs1s1,
                         double XTXs0t0,
                         double XTXs0t1,
                         double XTXs1t0,
                         double XTXs1t1,
                         double XTXt0t0,
                         double XTXt0t1,
                         double XTXt1t1):

    cdef double A00, A01, A02, A11, A12, A22
    cdef double length, norm_proj0, norm_proj1, inner_prod_proj

    A00 = XTXs1s1 - 2 * XTXs0s1 + XTXs0s0
    A11 = XTXt0t0 - 2 * XTXs0t0 + XTXs0s0
    A22 = XTXt1t1 - 2 * XTXs0t1 + XTXs0s0

    A01 = XTXs1t0 - XTXs0t0 - XTXs0s1 + XTXs0s0
    A02 = XTXs1t1 - XTXs0t1 - XTXs0s1 + XTXs0s0
    A12 = XTXt0t1 - XTXs0t0 - XTXs0t1 + XTXs0s0

    length = np.sqrt(A00)

    norm_proj0 = A11 - A01 * A01 / A00
    norm_proj1 = A22 - A02 * A02 / A00
    inner_prod_proj = A12 - A01 * A02 / A00

    return (np.pi - np.arccos(inner_prod_proj / np.sqrt(norm_proj0 * norm_proj1))) * length / (2 * np.pi)


def mu2_tri(np.ndarray[DTYPE_float_t, ndim=2] coords,
            long v0, long v1, long v2):

  """
  Compute the 2nd intrinsic volume (just area in this case) of
  a triangle with coordinates [coords[v0], coords[v1], coords[v2]].

  Inputs:
  -------

  coords : ndarray((*,2))
       An array of coordinates of vertices of tetrahedron.

  v0, v1, v2 : int
       Indices for vertices of the tetrahedron.

  Outputs:
  --------

  mu2 : float
  """


  cdef double XTX00 = 0
  cdef double XTX01 = 0
  cdef double XTX11 = 0
  cdef int q, j

  q = coords.shape[0]

  for j in range(q):
      XTX00 += (coords[j,v0] - coords[j,v2]) * (coords[j,v0] - coords[j,v2])
      XTX01 += (coords[j,v0] - coords[j,v2]) * (coords[j,v1] - coords[j,v2])
      XTX11 += (coords[j,v1] - coords[j,v2]) * (coords[j,v1] - coords[j,v2])

  return np.sqrt((XTX00 * XTX11 - XTX01 * XTX01)) * 0.5

def mu1_tri(np.ndarray[DTYPE_float_t, ndim=2] coords,
            long v0, long v1, long v2):
  """
  Compute the 1st intrinsic volume (1/2 the perimeter of
  a triangle with coordinates [coords[v0], coords[v1], coords[v2]].

  Inputs:
  -------

  coords : ndarray((*,2))
       An array of coordinates of vertices of tetrahedron.

  v0, v1, v2 : int
       Indices for vertices of the tetrahedron.

  Outputs:
  --------

  mu1 : float
  """

  cdef double mu = 0
  mu += mu1_edge(coords, v0, v1)
  mu += mu1_edge(coords, v1, v2)
  mu += mu1_edge(coords, v0, v2)
  return mu * 0.5

def mu1_edge(np.ndarray[DTYPE_float_t, ndim=2] coords,
             long v0, long v1):
  """
  Compute the 1st intrinsic volume (length)
  of a line segment with coordinates [coords[v0], coords[v1]]

  Inputs:
  -------

  coords : ndarray((*,2))
       An array of coordinates of vertices of tetrahedron.

  v0, v1 : int
       Indices for vertices of the tetrahedron.

  Outputs:
  --------

  mu0 : float
  """

  cdef int q

  q = coords.shape[0]

  mu = 0
  for j in range(q):
    mu += (coords[j,v0] - coords[j,v1])**2
  return np.sqrt(mu)

def Lips0_3d(np.ndarray[DTYPE_int_t, ndim=3] mask):
    """
    Given a mask and coordinates, estimate the 0th intrinsic volume
    (Euler characteristic)
    of the masked region. The region is broken up into tetrahedra /
    triangles / edges / vertices, which are included based on whether
    all voxels in the tetrahedron / triangle / edge / vertex are
    in the mask or not.

    Inputs:
    -------

    coords : ndarray((*,i,j,k))
         Coordinates for the voxels in the mask

    mask : ndarray((i,j,k), np.int)
         Binary mask determining whether or not
         a voxel is in the mask.

    Outputs:
    --------

    mu0 : int

    Notes:
    ------

    The array mask is assumed to be binary. At the time of
    writing, it is not clear how to get cython to use np.bool
    arrays.

    The 3d cubes are triangulated into 6 tetrahedra of equal volume,
    as described in the reference below.

    References:
    -----------

    Taylor, J.E. & Worsley, K.J. (2007). "Detecting sparse signal in random fields,
      with an application to brain mapping."
      Journal of the American Statistical Association, 102(479):913-928.


    """

    if not set(np.unique(mask)).issubset([0,1]):
      raise ValueError('mask should be filled with 0/1 values, but be of type np.int')

    # 'flattened' mask (1d array)

    cdef np.ndarray[DTYPE_int_t, ndim=1] fmask

    # d3 and d4 are lists of triangles and tetrahedra
    # associated to particular voxels in the cuve

    cdef np.ndarray[DTYPE_int_t, ndim=2] d2
    cdef np.ndarray[DTYPE_int_t, ndim=2] d3
    cdef np.ndarray[DTYPE_int_t, ndim=2] d4

    cdef long i, j, k, l, s0, s1, s2, ds2, ds3, ds4, index, m
    cdef long ss0, ss1, ss2 # strides
    cdef long v0, v1, v2, v3 # vertices
    cdef long l0 = 0

    s0, s1, s2 = (mask.shape[0], mask.shape[1], mask.shape[2])

    fmask = mask.reshape((s0*s1*s2))

    strides = np.empty((s0, s1, s2), np.bool).strides

    # First do the interior contributions.
    # We first figure out which vertices, edges, triangles, tetrahedra
    # are uniquely associated with an interior voxel

    union = join_complexes(*[cube_with_strides_center((0,0,-1), strides),
                             cube_with_strides_center((0,-1,0), strides),
                             cube_with_strides_center((0,-1,-1), strides),
                             cube_with_strides_center((-1,0,0), strides),
                             cube_with_strides_center((-1,0,-1), strides),
                             cube_with_strides_center((-1,-1,0), strides),
                             cube_with_strides_center((-1,-1,-1), strides)])
    c = cube_with_strides_center((0,0,0), strides)

    d4 = np.array(list(c[4].difference(union[4])))
    d3 = np.array(list(c[3].difference(union[3])))
    d2 = np.array(list(c[2].difference(union[2])))

    ds2 = d2.shape[0]
    ds3 = d3.shape[0]
    ds4 = d4.shape[0]

    ss0 = strides[0]
    ss1 = strides[1]
    ss2 = strides[2]

    for i in range(s0-1):
        for j in range(s1-1):
            for k in range(s2-1):
                index = i*ss0+j*ss1+k*ss2
                for l in range(ds4):
                    v0 = index + d4[l,0]
                    m = fmask[v0]
                    if m:
                        v1 = index + d4[l,1]
                        v2 = index + d4[l,2]
                        v3 = index + d4[l,3]
                        m = m * fmask[v1] * fmask[v2] * fmask[v3]
                        l0 = l0 - m

                for l in range(ds3):
                    v0 = index + d3[l,0]
                    m = fmask[v0]
                    if m:
                        v1 = index + d3[l,1]
                        v2 = index + d3[l,2]
                        m = m * fmask[v1] * fmask[v2]
                        l0 = l0 + m

                for l in range(ds2):
                    v0 = index + d2[l,0]
                    m = fmask[v0]
                    if m:
                        v1 = index + d2[l,1]
                        m = m * fmask[v1]
                        l0 = l0 - m

    # There are now contributions from three two-dimensional faces
    # on the boundary

    for _strides, _shape in zip([(strides[0], strides[1]),
                                 (strides[0], strides[2]),
                                 (strides[1], strides[2])],
                                [(mask.shape[0], mask.shape[1]),
                                 (mask.shape[0], mask.shape[2]),
                                 (mask.shape[1], mask.shape[2])]):

        unique = {}
        union = join_complexes(*[cube_with_strides_center((0,-1), _strides),
                                 cube_with_strides_center((-1,0), _strides),
                                 cube_with_strides_center((-1,-1), _strides)])

        c = cube_with_strides_center((0,0), _strides)
        d3 = np.array(list(c[3].difference(union[3])))
        d2 = np.array(list(c[2].difference(union[2])))

        s0, s1 = _shape
        ss0, ss1 = _strides
        ds3 = d3.shape[0]
        ds2 = d2.shape[0]

        for i in range(s0-1):
            for j in range(s1-1):
                index = i*ss0+j*ss1
                for l in range(ds3):
                    v0 = index + d3[l,0]
                    m = fmask[v0]
                    if m:
                        v1 = index + d3[l,1]
                        v2 = index + d3[l,2]
                        m = m * fmask[v1] * fmask[v2]
                        l0 = l0 + m

                for l in range(ds2):
                    v0 = index + d2[l,0]
                    m = fmask[v0]
                    if m:
                        v1 = index + d2[l,1]
                        m = m * fmask[v1]
                        l0 = l0 - m

    # Contribution of edges along the boundary

    s0, s1, s2 = (mask.shape[0], mask.shape[1], mask.shape[2])
    ss0 = strides[0]
    ss1 = strides[1]
    ss2 = strides[2]

    for _stride, _shape in zip((ss0, ss1, ss2), (s0, s1, s2)):

        unique = {}
        union = cube_with_strides_center((-1,), [_stride])
        c = cube_with_strides_center((0,), [_stride])
        d2 = np.array(list(c[2].difference(union[2])))

        s0 = _shape
        ss0 = _stride
        for i in range(s0-1):
                index = i*ss0
                v0 = index + d2[0,0]
                m = fmask[v0]
                if m:
                    v1 = index + d2[0,1]
                    m = m * fmask[v1]
                    l0 = l0 - m

    l0 += mask.sum()
    return l0

def Lips1_3d(np.ndarray[DTYPE_float_t, ndim=4] coords,
             np.ndarray[DTYPE_int_t, ndim=3] mask):
    """
    Given a mask and coordinates, estimate the 1st intrinsic volume
    of the masked region. The region is broken up into tetrahedra /
    triangles / edges / vertices, which are included based on whether
    all voxels in the tetrahedron / triangle / edge / vertex are
    in the mask or not.

    Inputs:
    -------

    coords : ndarray((*,i,j,k))
         Coordinates for the voxels in the mask

    mask : ndarray((i,j,k), np.int)
         Binary mask determining whether or not
         a voxel is in the mask.

    Outputs:
    --------

    mu1 : float

    Notes:
    ------

    The array mask is assumed to be binary. At the time of
    writing, it is not clear how to get cython to use np.bool
    arrays.

    The 3d cubes are triangulated into 6 tetrahedra of equal volume,
    as described in the reference below.

    References:
    -----------

    Taylor, J.E. & Worsley, K.J. (2007). "Detecting sparse signal in random fields,
      with an application to brain mapping."
      Journal of the American Statistical Association, 102(479):913-928.



    """

    if not set(np.unique(mask)).issubset([0,1]):
      raise ValueError('mask should be filled with 0/1 values, but be of type np.int')

    # 'flattened' coords (2d array)

    cdef np.ndarray[DTYPE_float_t, ndim=2] fcoords

    # 'flattened' mask (1d array)

    cdef np.ndarray[DTYPE_int_t, ndim=1] fmask

    # d3 and d4 are lists of triangles and tetrahedra
    # associated to particular voxels in the cuve

    cdef np.ndarray[DTYPE_int_t, ndim=2] d2
    cdef np.ndarray[DTYPE_int_t, ndim=2] d3
    cdef np.ndarray[DTYPE_int_t, ndim=2] d4

    cdef long i, j, k, l, s0, s1, s2, ds2, ds3, ds4, index, m
    cdef long ss0, ss1, ss2 # strides
    cdef long v0, v1, v2, v3 # vertices
    cdef double l1 = 0

    s0, s1, s2 = (coords.shape[1], coords.shape[2], coords.shape[3])

    if (mask.shape[0], mask.shape[1], mask.shape[2]) != (s0, s1, s2):
        raise ValueError('shape of mask does not match coordinates')

    fcoords = coords.reshape((coords.shape[0], (s0*s1*s2)))
    fmask = mask.reshape((s0*s1*s2))

    strides = np.empty((s0, s1, s2), np.bool).strides

    # First do the interior contributions.
    # We first figure out which vertices, edges, triangles, tetrahedra
    # are uniquely associated with an interior voxel

    union = join_complexes(*[cube_with_strides_center((0,0,-1), strides),
                             cube_with_strides_center((0,-1,0), strides),
                             cube_with_strides_center((0,-1,-1), strides),
                             cube_with_strides_center((-1,0,0), strides),
                             cube_with_strides_center((-1,0,-1), strides),
                             cube_with_strides_center((-1,-1,0), strides),
                             cube_with_strides_center((-1,-1,-1), strides)])
    c = cube_with_strides_center((0,0,0), strides)

    d4 = np.array(list(c[4].difference(union[4])))
    d3 = np.array(list(c[3].difference(union[3])))
    d2 = np.array(list(c[2].difference(union[2])))

    ds2 = d2.shape[0]
    ds3 = d3.shape[0]
    ds4 = d4.shape[0]

    ss0 = strides[0]
    ss1 = strides[1]
    ss2 = strides[2]

    for i in range(s0-1):
        for j in range(s1-1):
            for k in range(s2-1):
                index = i*ss0+j*ss1+k*ss2
                for l in range(ds4):
                    v0 = index + d4[l,0]
                    m = fmask[v0]
                    if m:
                        v1 = index + d4[l,1]
                        v2 = index + d4[l,2]
                        v3 = index + d4[l,3]
                        m = m * fmask[v1] * fmask[v2] * fmask[v3]
                        l1 = l1 + mu1_tet(fcoords, v0, v1, v2, v3) * m

                for l in range(ds3):
                    v0 = index + d3[l,0]
                    m = fmask[v0]
                    if m:
                        v1 = index + d3[l,1]
                        v2 = index + d3[l,2]
                        m = m * fmask[v1] * fmask[v2]
                        l1 = l1 - mu1_tri(fcoords, v0, v1, v2) * m

                for l in range(ds2):
                    v0 = index + d2[l,0]
                    m = fmask[v0]
                    if m:
                        v1 = index + d2[l,1]
                        m = m * fmask[v1]
                        l1 = l1 + mu1_edge(fcoords, v0, v1) * m

    # There are now contributions from three two-dimensional faces
    # on the boundary

    for _strides, _shape in zip([(strides[0], strides[1]),
                                 (strides[0], strides[2]),
                                 (strides[1], strides[2])],
                                [(coords.shape[1], coords.shape[2]),
                                 (coords.shape[1], coords.shape[3]),
                                 (coords.shape[2], coords.shape[3])]):

        unique = {}
        union = join_complexes(*[cube_with_strides_center((0,-1), _strides),
                                 cube_with_strides_center((-1,0), _strides),
                                 cube_with_strides_center((-1,-1), _strides)])

        c = cube_with_strides_center((0,0), _strides)
        d3 = np.array(list(c[3].difference(union[3])))
        d2 = np.array(list(c[2].difference(union[2])))

        s0, s1 = _shape
        ss0, ss1 = _strides
        ds3 = d3.shape[0]
        ds2 = d2.shape[0]

        for i in range(s0-1):
            for j in range(s1-1):
                index = i*ss0+j*ss1
                for l in range(ds3):
                    v0 = index + d3[l,0]
                    m = fmask[v0]
                    if m:
                        v1 = index + d3[l,1]
                        v2 = index + d3[l,2]
                        m = m * fmask[v1] * fmask[v2]
                        l1 = l1 - mu1_tri(fcoords, v0, v1, v2) * m

                for l in range(ds2):
                    v0 = index + d2[l,0]
                    m = fmask[v0]
                    if m:
                        v1 = index + d2[l,1]
                        m = m * fmask[v1]
                        l1 = l1 + mu1_edge(fcoords, v0, v1) * m

    # Contribution of edges along the boundary

    s0, s1, s2 = (coords.shape[1], coords.shape[2], coords.shape[3])
    ss0 = strides[0]
    ss1 = strides[1]
    ss2 = strides[2]

    for _stride, _shape in zip((ss0, ss1, ss2), (s0, s1, s2)):

        unique = {}
        union = cube_with_strides_center((-1,), [_stride])
        c = cube_with_strides_center((0,), [_stride])
        d2 = np.array(list(c[2].difference(union[2])))

        s0 = _shape
        ss0 = _stride
        for i in range(s0-1):
                index = i*ss0
                v0 = index + d2[0,0]
                m = fmask[v0]
                if m:
                    v1 = index + d2[0,1]
                    m = m * fmask[v1]
                    l1 = l1 + mu1_edge(fcoords, v0, v1) * m

    return l1

def Lips2_3d(np.ndarray[DTYPE_float_t, ndim=4] coords,
             np.ndarray[DTYPE_int_t, ndim=3] mask):
    """
    Given a mask and coordinates, estimate the 2nd intrinsic volume
    of the masked region. The region is broken up into tetrahedra /
    triangles / edges / vertices, which are included based on whether
    all voxels in the tetrahedron / triangle / edge / vertex are
    in the mask or not.

    Inputs:
    -------

    coords : ndarray((*,i,j,k))
         Coordinates for the voxels in the mask

    mask : ndarray((i,j,k), np.int)
         Binary mask determining whether or not
         a voxel is in the mask.

    Outputs:
    --------

    mu2 : float

    Notes:
    ------

    The array mask is assumed to be binary. At the time of
    writing, it is not clear how to get cython to use np.bool
    arrays.

    The 3d cubes are triangulated into 6 tetrahedra of equal volume,
    as described in the reference below.

    References:
    -----------

    Taylor, J.E. & Worsley, K.J. (2007). "Detecting sparse signal in random fields,
      with an application to brain mapping."
      Journal of the American Statistical Association, 102(479):913-928.

    """

    if not set(np.unique(mask)).issubset([0,1]):
      raise ValueError('mask should be filled with 0/1 values, but be of type np.int')

    # 'flattened' coords (2d array)

    cdef np.ndarray[DTYPE_float_t, ndim=2] fcoords

    # 'flattened' mask (1d array)

    cdef np.ndarray[DTYPE_int_t, ndim=1] fmask

    # d3 and d4 are lists of triangles and tetrahedra
    # associated to particular voxels in the cuve

    cdef np.ndarray[DTYPE_int_t, ndim=2] d3
    cdef np.ndarray[DTYPE_int_t, ndim=2] d4

    cdef long i, j, k, l, s0, s1, s2, ds3, ds4, index, m
    cdef long ss0, ss1, ss2 # strides
    cdef long v0, v1, v2, v3 # vertices
    cdef double l2 = 0

    s0, s1, s2 = (coords.shape[1], coords.shape[2], coords.shape[3])

    if (mask.shape[0], mask.shape[1], mask.shape[2]) != (s0, s1, s2):
        raise ValueError('shape of mask does not match coordinates')

    fcoords = coords.reshape((coords.shape[0], (s0*s1*s2)))
    fmask = mask.reshape((s0*s1*s2))

    strides = np.empty((s0, s1, s2), np.bool).strides

    # First do the interior contributions.
    # We first figure out which vertices, edges, triangles, tetrahedra
    # are uniquely associated with an interior voxel

    union = join_complexes(*[cube_with_strides_center((0,0,-1), strides),
                             cube_with_strides_center((0,-1,0), strides),
                             cube_with_strides_center((0,-1,-1), strides),
                             cube_with_strides_center((-1,0,0), strides),
                             cube_with_strides_center((-1,0,-1), strides),
                             cube_with_strides_center((-1,-1,0), strides),
                             cube_with_strides_center((-1,-1,-1), strides)])
    c = cube_with_strides_center((0,0,0), strides)

    d4 = np.array(list(c[4].difference(union[4])))
    d3 = np.array(list(c[3].difference(union[3])))

    ds3 = d3.shape[0]
    ds4 = d4.shape[0]

    ss0 = strides[0]
    ss1 = strides[1]
    ss2 = strides[2]

    for i in range(s0-1):
        for j in range(s1-1):
            for k in range(s2-1):
                index = i*ss0+j*ss1+k*ss2
                for l in range(ds4):
                    v0 = index + d4[l,0]
                    m = fmask[v0]
                    if m:
                        v1 = index + d4[l,1]
                        v2 = index + d4[l,2]
                        v3 = index + d4[l,3]
                        m = m * fmask[v1] * fmask[v2] * fmask[v3]
                        l2 = l2 - mu2_tet(fcoords, v0, v1, v2, v3) * m

                for l in range(ds3):
                    v0 = index + d3[l,0]
                    m = fmask[v0]
                    if m:
                        v1 = index + d3[l,1]
                        v2 = index + d3[l,2]
                        m = m * fmask[v1] * fmask[v2]
                        l2 = l2 + mu2_tri(fcoords, v0, v1, v2) * m

    # There are now contributions from three two-dimensional faces

    for _strides, _shape in zip([(strides[0], strides[1]),
                                 (strides[0], strides[2]),
                                 (strides[1], strides[2])],
                                [(coords.shape[1], coords.shape[2]),
                                 (coords.shape[1], coords.shape[3]),
                                 (coords.shape[2], coords.shape[3])]):

        unique = {}
        union = join_complexes(*[cube_with_strides_center((0,-1), _strides),
                                 cube_with_strides_center((-1,0), _strides),
                                 cube_with_strides_center((-1,-1), _strides)])

        c = cube_with_strides_center((0,0), _strides)
        d3 = np.array(list(c[3].difference(union[3])))

        s0, s1 = _shape
        ss0, ss1 = _strides
        ds3 = d3.shape[0]
        for i in range(s0-1):
            for j in range(s1-1):
                index = i*ss0+j*ss1
                for l in range(ds3):
                    v0 = index + d3[l,0]
                    m = fmask[v0]
                    if m:
                        v1 = index + d3[l,1]
                        v2 = index + d3[l,2]
                        m = m * fmask[v1] * fmask[v2]
                        l2 = l2 + mu2_tri(fcoords, v0, v1, v2) * m
    return l2

def Lips3_3d(np.ndarray[DTYPE_float_t, ndim=4] coords,
             np.ndarray[DTYPE_int_t, ndim=3] mask):
    """
    Given a mask and coordinates, estimate the 3rd intrinsic volume
    of the masked region. The region is broken up into tetrahedra /
    triangles / edges / vertices, which are included based on whether
    all voxels in the tetrahedron / triangle / edge / vertex are
    in the mask or not.

    Inputs:
    -------

    coords : ndarray((*,i,j,k))
         Coordinates for the voxels in the mask

    mask : ndarray((i,j,k), np.int)
         Binary mask determining whether or not
         a voxel is in the mask.

    Outputs:
    --------

    mu3 : float

    Notes:
    ------

    The array mask is assumed to be binary. At the time of
    writing, it is not clear how to get cython to use np.bool
    arrays.

    The 3d cubes are triangulated into 6 tetrahedra of equal volume,
    as described in the reference below.

    References:
    -----------

    Taylor, J.E. & Worsley, K.J. (2007). "Detecting sparse signal in random fields,
      with an application to brain mapping."
      Journal of the American Statistical Association, 102(479):913-928.

    """

    if not set(np.unique(mask)).issubset([0,1]):
      raise ValueError('mask should be filled with 0/1 values, but be of type np.int')

    # 'flattened' coords (2d array)

    cdef np.ndarray[DTYPE_float_t, ndim=2] fcoords

    # 'flattened' mask (1d array)

    cdef np.ndarray[DTYPE_int_t, ndim=1] fmask

    # d3 and d4 are lists of triangles and tetrahedra
    # associated to particular voxels in the cuve

    cdef np.ndarray[DTYPE_int_t, ndim=2] d4

    cdef long i, j, k, l, s0, s1, s2, ds4, index, m
    cdef long ss0, ss1, ss2 # strides
    cdef long v0, v1, v2, v3 # vertices
    cdef double l3 = 0

    s0, s1, s2 = (coords.shape[1], coords.shape[2], coords.shape[3])

    if (mask.shape[0], mask.shape[1], mask.shape[2]) != (s0, s1, s2):
        raise ValueError('shape of mask does not match coordinates')

    fcoords = coords.reshape((coords.shape[0], (s0*s1*s2)))
    fmask = mask.reshape((s0*s1*s2))

    strides = np.empty((s0, s1, s2), np.bool).strides

    # First do the interior contributions.
    # We first figure out which vertices, edges, triangles, tetrahedra
    # are uniquely associated with an interior voxel

    union = join_complexes(*[cube_with_strides_center((0,0,-1), strides),
                             cube_with_strides_center((0,-1,0), strides),
                             cube_with_strides_center((0,-1,-1), strides),
                             cube_with_strides_center((-1,0,0), strides),
                             cube_with_strides_center((-1,0,-1), strides),
                             cube_with_strides_center((-1,-1,0), strides),
                             cube_with_strides_center((-1,-1,-1), strides)])
    c = cube_with_strides_center((0,0,0), strides)

    d4 = np.array(list(c[4].difference(union[4])))
    ds4 = d4.shape[0]

    ss0 = strides[0]
    ss1 = strides[1]
    ss2 = strides[2]

    for i in range(s0-1):
        for j in range(s1-1):
            for k in range(s2-1):
                index = i*ss0+j*ss1+k*ss2
                for l in range(ds4):
                    v0 = index + d4[l,0]
                    m = fmask[v0]
                    if m:
                        v1 = index + d4[l,1]
                        v2 = index + d4[l,2]
                        v3 = index + d4[l,3]
                        m = m * fmask[v1] * fmask[v2] * fmask[v3]
                        l3 = l3 + mu3_tet(fcoords, v0, v1, v2, v3) * m

    return l3



def Lips2_2d(np.ndarray[DTYPE_float_t, ndim=3] coords,
             np.ndarray[DTYPE_int_t, ndim=2] mask):
    """
    Given a mask and coordinates, estimate the 2nd intrinsic volume
    of the masked region. The region is broken up into
    triangles / edges / vertices, which are included based on whether
    all voxels in the triangle / edge / vertex are
    in the mask or not.

    Inputs:
    -------

    coords : ndarray((*,i,j))
         Coordinates for the voxels in the mask

    mask : ndarray((i,j), np.int)
         Binary mask determining whether or not
         a voxel is in the mask.

    Outputs:
    --------

    mu2 : float

    Notes:
    ------

    The array mask is assumed to be binary. At the time of
    writing, it is not clear how to get cython to use np.bool
    arrays.

    References:
    -----------

    Taylor, J.E. & Worsley, K.J. (2007). "Detecting sparse signal in random fields,
      with an application to brain mapping."
      Journal of the American Statistical Association, 102(479):913-928.


    """

    if not set(np.unique(mask)).issubset([0,1]):
      raise ValueError('mask should be filled with 0/1 values, but be of type np.int')

    # 'flattened' coords (2d array)

    cdef np.ndarray[DTYPE_float_t, ndim=2] fcoords

    # 'flattened' mask (1d array)

    cdef np.ndarray[DTYPE_int_t, ndim=1] fmask

    # d3 and d4 are lists of triangles
    # associated to particular voxels in the square

    cdef np.ndarray[DTYPE_int_t, ndim=2] d3

    cdef long i, j, k, l, s0, s1, ds3, index, m
    cdef long ss0, ss1 # strides
    cdef long v0, v1, v2 # vertices
    cdef double l2 = 0

    s0, s1 = (coords.shape[1], coords.shape[2])

    if (mask.shape[0], mask.shape[1]) != (s0, s1):
        raise ValueError('shape of mask does not match coordinates')

    fcoords = coords.reshape((coords.shape[0], (s0*s1)))
    fmask = mask.reshape((s0*s1))

    strides = np.empty((s0, s1), np.bool).strides

    # First do the interior contributions.
    # We first figure out which vertices, edges, triangles, tetrahedra
    # are uniquely associated with an interior voxel

    union = join_complexes(*[cube_with_strides_center((0,-1), strides),
                             cube_with_strides_center((-1,0), strides),
                             cube_with_strides_center((-1,-1), strides)])

    c = cube_with_strides_center((0,0), strides)
    d3 = np.array(list(c[3].difference(union[3])))
    ds3 = d3.shape[0]

    ss0 = strides[0]
    ss1 = strides[1]

    for i in range(s0-1):
        for j in range(s1-1):
          index = i*ss0+j*ss1
          for l in range(ds3):
            v0 = index + d3[l,0]
            m = fmask[v0]
            if m:
              v1 = index + d3[l,1]
              v2 = index + d3[l,2]
              m = m * fmask[v1] * fmask[v2]
              l2 = l2 + mu2_tri(fcoords, v0, v1, v2) * m

    return l2

def Lips1_2d(np.ndarray[DTYPE_float_t, ndim=3] coords,
             np.ndarray[DTYPE_int_t, ndim=2] mask):
    """
    Given a mask and coordinates, estimate the 1st intrinsic volume
    of the masked region. The region is broken up into
    triangles / edges / vertices, which are included based on whether
    all voxels in the triangle / edge / vertex are
    in the mask or not.

    Inputs:
    -------

    coords : ndarray((*,i,j))
         Coordinates for the voxels in the mask

    mask : ndarray((i,j), np.int)
         Binary mask determining whether or not
         a voxel is in the mask.

    Outputs:
    --------

    mu1 : float

    Notes:
    ------

    The array mask is assumed to be binary. At the time of
    writing, it is not clear how to get cython to use np.bool
    arrays.

    References:
    -----------

    Taylor, J.E. & Worsley, K.J. (2007). "Detecting sparse signal in random fields,
      with an application to brain mapping."
      Journal of the American Statistical Association, 102(479):913-928.

    """

    if not set(np.unique(mask)).issubset([0,1]):
      raise ValueError('mask should be filled with 0/1 values, but be of type np.int')

    # 'flattened' coords (2d array)

    cdef np.ndarray[DTYPE_float_t, ndim=2] fcoords

    # 'flattened' mask (1d array)

    cdef np.ndarray[DTYPE_int_t, ndim=1] fmask

    # d3 and d4 are lists of triangles
    # associated to particular voxels in the square

    cdef np.ndarray[DTYPE_int_t, ndim=2] d3
    cdef np.ndarray[DTYPE_int_t, ndim=2] d2

    cdef long i, j, k, l, s0, s1, ds3, ds2, index, m
    cdef long ss0, ss1 # strides
    cdef long v0, v1, v2 # vertices
    cdef double l1 = 0

    s0, s1 = (coords.shape[1], coords.shape[2])

    if (mask.shape[0], mask.shape[1]) != (s0, s1):
        raise ValueError('shape of mask does not match coordinates')

    fcoords = coords.reshape((coords.shape[0], (s0*s1)))
    fmask = mask.reshape((s0*s1))

    strides = np.empty((s0, s1), np.bool).strides

    union = join_complexes(*[cube_with_strides_center((0,-1), strides),
                             cube_with_strides_center((-1,0), strides),
                             cube_with_strides_center((-1,-1), strides)])

    c = cube_with_strides_center((0,0), strides)
    d3 = np.array(list(c[3].difference(union[3])))
    d2 = np.array(list(c[2].difference(union[2])))

    ds3 = d3.shape[0]
    ds2 = d2.shape[0]

    ss0 = strides[0]
    ss1 = strides[1]

    # First do the interior contributions.
    # We first figure out which vertices, edges, triangles, tetrahedra
    # are uniquely associated with an interior voxel

    for i in range(s0-1):
        for j in range(s1-1):
          index = i*ss0+j*ss1

          for l in range(ds3):
            v0 = index + d3[l,0]
            m = fmask[v0]
            if m:
              v1 = index + d3[l,1]
              v2 = index + d3[l,2]
              m = m * fmask[v1] * fmask[v2]
              l1 = l1 - mu1_tri(fcoords, v0, v1, v2) * m

          for l in range(ds2):
            v0 = index + d2[l,0]
            m = fmask[v0]
            if m:
              v1 = index + d2[l,1]
              m = m * fmask[v1]
              l1 = l1 + mu1_edge(fcoords, v0, v1) * m

    s0, s1 = (coords.shape[1], coords.shape[2])
    ss0 = strides[0]
    ss1 = strides[1]

    for _stride, _shape in zip((ss0, ss1), (s0, s1)):

        unique = {}
        union = cube_with_strides_center((-1,), [_stride])
        c = cube_with_strides_center((0,), [_stride])
        d2 = np.array(list(c[2].difference(union[2])))

        s0 = _shape
        ss0 = _stride
        for i in range(s0-1):
                index = i*ss0
                v0 = index + d2[0,0]
                m = fmask[v0]
                if m:
                    v1 = index + d2[0,1]
                    m = m * fmask[v1]
                    l1 = l1 + mu1_edge(fcoords, v0, v1) * m

    return l1

def Lips0_2d(np.ndarray[DTYPE_int_t, ndim=2] mask):

    """
    Given a mask and coordinates, estimate the 0th intrinsic volume
    (Euler characteristic) of the masked region. The region is broken
    up into triangles, edges, vertices, which are included based on whether
    all voxels in the triangle / edge / vertex are
    in the mask or not.

    Inputs:
    -------

    mask : ndarray((i,j), np.int)
         Binary mask determining whether or not
         a voxel is in the mask.

    Outputs:
    --------

    mu0 : int

    Notes:
    ------

    The array mask is assumed to be binary. At the time of
    writing, it is not clear how to get cython to use np.bool
    arrays.

    References:
    -----------

    Taylor, J.E. & Worsley, K.J. (2007). "Detecting sparse signal in random fields,
      with an application to brain mapping."
      Journal of the American Statistical Association, 102(479):913-928.

    """

    if not set(np.unique(mask)).issubset([0,1]):
      raise ValueError('mask should be filled with 0/1 values, but be of type np.int')

    # 'flattened' mask (1d array)

    cdef np.ndarray[DTYPE_int_t, ndim=1] fmask

    # d3 and d4 are lists of triangles
    # associated to particular voxels in the square

    cdef np.ndarray[DTYPE_int_t, ndim=2] d3
    cdef np.ndarray[DTYPE_int_t, ndim=2] d2

    cdef long i, j, k, l, s0, s1, ds3, ds2, index, m
    cdef long ss0, ss1 # strides
    cdef long v0, v1, v2 # vertices
    cdef long l0 = 0

    s0, s1 = (mask.shape[0], mask.shape[1])

    fmask = mask.reshape((s0*s1))

    strides = np.empty((s0, s1), np.bool).strides

    union = join_complexes(*[cube_with_strides_center((0,-1), strides),
                             cube_with_strides_center((-1,0), strides),
                             cube_with_strides_center((-1,-1), strides)])

    c = cube_with_strides_center((0,0), strides)
    d3 = np.array(list(c[3].difference(union[3])))
    d2 = np.array(list(c[2].difference(union[2])))

    ds3 = d3.shape[0]
    ds2 = d2.shape[0]

    ss0 = strides[0]
    ss1 = strides[1]

    # First do the interior contributions.
    # We first figure out which edges, triangles
    # are uniquely associated with an interior voxel

    for i in range(s0-1):
        for j in range(s1-1):
          index = i*ss0+j*ss1

          for l in range(ds3):
            v0 = index + d3[l,0]
            m = fmask[v0]
            if m:
              v1 = index + d3[l,1]
              v2 = index + d3[l,2]
              m = m * fmask[v1] * fmask[v2]
              l0 = l0 + m

          for l in range(ds2):
            v0 = index + d2[l,0]
            m = fmask[v0]
            if m:
              v1 = index + d2[l,1]
              m = m * fmask[v1]
              l0 = l0 - m

    s0, s1 = (mask.shape[0], mask.shape[1])
    ss0 = strides[0]
    ss1 = strides[1]

    for _stride, _shape in zip((ss0, ss1), (s0, s1)):

        unique = {}
        union = cube_with_strides_center((-1,), [_stride])
        c = cube_with_strides_center((0,), [_stride])
        d2 = np.array(list(c[2].difference(union[2])))

        s0 = _shape
        ss0 = _stride
        for i in range(s0-1):
                index = i*ss0
                v0 = index + d2[0,0]
                m = fmask[v0]
                if m:
                    v1 = index + d2[0,1]
                    m = m * fmask[v1]
                    l0 = l0 - m

    l0 += mask.sum()
    return l0


def Lips1_1d(np.ndarray[DTYPE_float_t, ndim=2] coords,
             np.ndarray[DTYPE_int_t, ndim=1] mask):

    """
    Given a mask and coordinates, estimate the 1st intrinsic volume
    (Euler characteristic) of the masked region. The region is broken
    up into edges / vertices, which are included based on whether
    all voxels in the edge / vertex are
    in the mask or not.

    Inputs:
    -------

    coords : ndarray((*,i))
         Coordinates for the voxels in the mask


    mask : ndarray((i,), np.int)
         Binary mask determining whether or not
         a voxel is in the mask.

    Outputs:
    --------

    mu1 : float

    Notes:
    ------

    The array mask is assumed to be binary. At the time of
    writing, it is not clear how to get cython to use np.bool
    arrays.

    References:
    -----------

    Taylor, J.E. & Worsley, K.J. (2007). "Detecting sparse signal in random fields,
      with an application to brain mapping."
      Journal of the American Statistical Association, 102(479):913-928.

    """

    if not set(np.unique(mask)).issubset([0,1]):
      raise ValueError('mask should be filled with 0/1 values, but be of type np.int')

    # 'flattened' coords (2d array)

    cdef np.ndarray[DTYPE_float_t, ndim=2] fcoords

    # 'flattened' mask (1d array)

    cdef np.ndarray[DTYPE_int_t, ndim=1] fmask

    cdef long i, s0, m
    cdef double l1 = 0

    s0 = coords.shape[1]

    if mask.shape[0] != s0:
        raise ValueError('shape of mask does not match coordinates')

    fcoords = coords.reshape((coords.shape[0], s0))
    fmask = mask.reshape((s0,))

    # First do the interior contributions.
    # We first figure out which vertices, edges, triangles, tetrahedra
    # are uniquely associated with an interior voxel

    for i in range(s0-1):
      m = fmask[i] * fmask[i+1]
      l1 = l1 + mu1_edge(coords, i, i+1) * m
    return l1

def Lips0_1d(np.ndarray[DTYPE_int_t, ndim=1] mask):
    """
    Given a mask and coordinates, estimate the 0th intrinsic volume (Euler characteristic)
    (Euler characteristic) of the masked region. The region is broken
    up into edges / vertices, which are included based on whether
    all voxels in the edge / vertex are
    in the mask or not.

    Inputs:
    -------

    coords : ndarray((*,i))
         Coordinates for the voxels in the mask


    mask : ndarray((i,), np.int)
         Binary mask determining whether or not
         a voxel is in the mask.

    Outputs:
    --------

    mu0 : int

    Notes:
    ------

    The array mask is assumed to be binary. At the time of
    writing, it is not clear how to get cython to use np.bool
    arrays.

    References:
    -----------

    Taylor, J.E. & Worsley, K.J. (2007). "Detecting sparse signal in random fields,
      with an application to brain mapping."
      Journal of the American Statistical Association, 102(479):913-928.

    """

    if not set(np.unique(mask)).issubset([0,1]):
      raise ValueError('mask should be filled with 0/1 values, but be of type np.int')

    # 'flattened' mask (1d array)

    cdef np.ndarray[DTYPE_int_t, ndim=1] fmask

    cdef long i, s0, m
    cdef long l0 = 0

    s0 = mask.shape[0]
    fmask = mask.reshape((s0,))

    # First do the interior contributions.
    # We first figure out which vertices, edges, triangles, tetrahedra
    # are uniquely associated with an interior voxel

    for i in range(s0-1):
      m = fmask[i] * fmask[i+1]
      l0 = l0 - m
    return l0 + mask.sum()
