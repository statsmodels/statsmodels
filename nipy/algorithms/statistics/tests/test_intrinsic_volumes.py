import numpy as np
import numpy.linalg as L
import numpy.random as R

from nipy.testing import *

from nipy.algorithms.statistics import intvol, utils

def symnormal(p=10):
    M = R.standard_normal((p,p))
    return (M + M.T) / np.sqrt(2)

def randorth(p=10):
    """
    A random orthogonal matrix.
    """
    A = symnormal(p)
    return L.eig(A)[1]

def box(shape, edges):
    data = np.zeros(shape)
    sl = []
    for i in range(len(shape)):
        sl.append(slice(edges[i][0], edges[i][1],1))
    data[sl] = 1
    return data.astype(np.int)

def randombox(shape):
    """
    Generate a random box, returning the box and the edge lengths
    """
    edges = [R.random_integers(0, shape[j], size=(2,))
             for j in range(len(shape))]

    for j in range(len(shape)):
        edges[j].sort()
        if edges[j][0] == edges[j][1]:
            edges[j][0] = 0; edges[j][1] = shape[j]/2+1
    return edges, box(shape, edges)

def elsym(edgelen, order=1):
    """
    Elementary symmetric polynoimal of a given order
    """

    l = len(edgelen)
    r = 0
    for v in utils.combinations(range(l), order):
        r += np.product([edgelen[vv] for vv in v])
    return r

def nonintersecting_boxes(shape):
    """
    The Lips's are supposed to be additive, so disjoint things
    should be additive. But, if they ALMOST intersect, different
    things get added to the triangulation.

    >>> b1 = np.zeros(40, np.int)
    >>> b1[:11] = 1
    >>> b2 = np.zeros(40, np.int)
    >>> b2[11:] = 1
    >>> (b1*b2).sum()
    0
    >>> c = np.indices((40,)).astype(np.float)
    >>> intvol.Lips1_1d(c, b1)
    10.0
    >>> intvol.Lips1_1d(c, b2)
    28.0
    >>> intvol.Lips1_1d(c, b1+b2)
    39.0

    The function creates two boxes such that
    the 'dilated' box1 does not intersect with box2.
    Additivity works in this case.
    """

    while True:
        edge1, box1 = randombox(shape)
        edge2, box2 = randombox(shape)

        diledge1 = [[max(ed[0]-1, 0), min(ed[1]+1, sh)]
                    for ed, sh in zip(edge1, box1.shape)]

        dilbox1 = box(box1.shape, diledge1)

        if set(np.unique(dilbox1 + box2)).issubset([0,1]):
            break
    return box1, box2, edge1, edge2

def test_ec():
    for i in range(1, 4):
        _, box1 = randombox((40,)*i)
        f = {3:intvol.Lips0_3d,
             2:intvol.Lips0_2d,
             1:intvol.Lips0_1d}[i]
        yield assert_almost_equal, f(box1), 1

def test_ec_disjoint():
    for i in range(1, 4):
        e = {3:intvol.Lips0_3d,
             2:intvol.Lips0_2d,
             1:intvol.Lips0_1d}[i]
        box1, box2, _, _ = nonintersecting_boxes((40,)*i)
        yield assert_almost_equal, e(box1 + box2), e(box1) + e(box2)


def test_lips1_disjoint():
    for i in range(1, 4):
        phi = {3:intvol.Lips1_3d,
             2:intvol.Lips1_2d,
             1:intvol.Lips1_1d}[i]

        box1, box2, edge1, edge2 = nonintersecting_boxes((30,)*i)
        c = np.indices((30,)*i).astype(np.float)
        d = np.random.standard_normal((10,)+(30,)*i)

        U = randorth(p=6)[0:i]
        e = np.dot(U.T, c.reshape((c.shape[0], np.product(c.shape[1:]))))
        e.shape = (e.shape[0],) +  c.shape[1:]

        yield assert_almost_equal, phi(c, box1 + box2), \
            phi(c, box1) + phi(c, box2)
        yield assert_almost_equal, phi(d, box1 + box2), \
            phi(d, box1) + phi(d, box2)
        yield assert_almost_equal, phi(e, box1 + box2), \
            phi(e, box1) + phi(e, box2)
        yield assert_almost_equal, phi(e, box1 + box2), phi(c, box1 + box2)
        yield assert_almost_equal, phi(e, box1 + box2), \
            elsym([e[1]-e[0]-1 for e in edge1], 1) + \
            elsym([e[1]-e[0]-1 for e in edge2], 1)


def test_lips2_disjoint():
    for i in range(2, 4):
        phi = {3:intvol.Lips2_3d,
               2:intvol.Lips2_2d}[i]

        box1, box2, edge1, edge2 = nonintersecting_boxes((40,)*i)
        c = np.indices((40,)*i).astype(np.float)
        d = np.random.standard_normal((40,)+(40,)*i)

        U = randorth(p=6)[0:i]
        e = np.dot(U.T, c.reshape((c.shape[0], np.product(c.shape[1:]))))
        e.shape = (e.shape[0],) +  c.shape[1:]

        yield assert_almost_equal, phi(c, box1 + box2), phi(c, box1) + \
            phi(c, box2)
        yield assert_almost_equal, phi(d, box1 + box2), phi(d, box1) + \
            phi(d, box2)
        yield assert_almost_equal, phi(e, box1 + box2), phi(e, box1) + \
            phi(e, box2)
        yield assert_almost_equal, phi(e, box1 + box2), phi(c, box1 + box2)
        yield assert_almost_equal, phi(e, box1 + box2), \
            elsym([e[1]-e[0]-1 for e in edge1], 2) + \
            elsym([e[1]-e[0]-1 for e in edge2], 2)


def test_lips3_disjoint():
    phi = intvol.Lips3_3d
    box1, box2, edge1, edge2 = nonintersecting_boxes((40,)*3)
    c = np.indices((40,)*3).astype(np.float)
    d = np.random.standard_normal((40,40,40,40))

    U = randorth(p=6)[0:3]
    e = np.dot(U.T, c.reshape((c.shape[0], np.product(c.shape[1:]))))
    e.shape = (e.shape[0],) +  c.shape[1:]

    yield assert_almost_equal, phi(c, box1 + box2), phi(c, box1) + phi(c, box2)
    yield assert_almost_equal, phi(d, box1 + box2), phi(d, box1) + phi(d, box2)
    yield assert_almost_equal, phi(e, box1 + box2), phi(e, box1) + phi(e, box2)
    yield assert_almost_equal, phi(e, box1 + box2), phi(c, box1 + box2)
    yield assert_almost_equal, phi(e, box1 + box2), \
        elsym([e[1]-e[0]-1 for e in edge1], 3) + \
        elsym([e[1]-e[0]-1 for e in edge2], 3)


def test_slices():
    # Slices have EC 1...

    e = intvol.Lips0_3d

    m = np.zeros((40,)*3, np.int)
    m[10,10,10] = 1
    yield assert_almost_equal, e(m), 1

    m = np.zeros((40,)*3, np.int)
    m[10,10:12,10] = 1
    yield assert_almost_equal, e(m), 1

    m = np.zeros((40,)*3, np.int)
    m[10,10:12,10:12] = 1
    yield assert_almost_equal, e(m), 1




