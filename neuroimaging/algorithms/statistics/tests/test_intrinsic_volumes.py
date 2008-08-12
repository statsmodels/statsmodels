import numpy as N
import numpy.linalg as L
import numpy.random as R

from neuroimaging.testing import *

from neuroimaging.algorithms.statistics import intrinsic_volumes

def symnormal(p=10):
    M = R.standard_normal((p,p))
    return (M + M.T) / N.sqrt(2)

def randorth(p=10):
    A = symnormal(p)
    return L.eig(A)[1]

def box(shape, edges):
    data = N.zeros(shape)
    sl = []
    for i in range(len(shape)):
        sl.append(slice(edges[i][0], edges[i][1],1))
    data[sl] = 1
    return data

def randombox(shape):
    edges = [R.random_integers(0, shape[j], size=(2,)) for j in range(len(shape))]
    for j in range(len(shape)):
        edges[j].sort()
        if edges[j][0] == edges[j][1]:
            edges[j][0] = 0; edges[j][1] = shape[j]/2+1
    return edges, box(shape, edges)


class test_iv(TestCase):

    def test1(self):
        for i in range(1, 4):
            _, box1 = randombox((30,)*i)
            assert_almost_equal(intrinsic_volumes.EC(box1), 1)

    # FIXME: AssertionError:
    #    Items are not equal:
    #    ACTUAL: 1.0
    #    DESIRED: 2.0
    @dec.skipknownfailure
    def test2(self):
        e = intrinsic_volumes.EC
        for i in range(1, 4):
            _, box1 = randombox((30,)*i)
            _, box2 = randombox((30,)*i)
            assert_almost_equal(e(box1 + box2),
                                          e(box1) + e(box2) - e(box1*box2))

    # FIXME: AssertionError:
    #    Items are not equal:
    #    ACTUAL: 1.0
    #    DESIRED: 2.0
    @dec.skipknownfailure
    def test3(self):
        e = intrinsic_volumes.EC
        for i in range(1, 4):
            e1, box1 = randombox((30,)*i)
            e2, box2 = randombox((30,)*i)
            e3, box3 = randombox((30,)*i)
            a = e(box1 + box2 + box3)
            b = (e(box1) + e(box2) + e(box3) -
                 e(box1*box2) - e(box2*box3) - e(box1*box3) +
                 e(box1*box2*box3))
            if a != b:
                print a, b
                print e1, e2, e3
                print e(box1), e(box2), e(box3)
                print e(box1*box2), e(box1*box3), e(box2*box3)
                print e(box1*box2*box3)
            assert_almost_equal(e(box1 + box2 + box3),
                                          (e(box1) + e(box2) + e(box3) -
                                           e(box1*box2) - e(box2*box3) - e(box1*box3) +
                                           e(box1*box2*box3)))
            print """
            This test is failing.
            This is probably due to a bad test case. This needs to be
            addressed by someone that understand this (Jonathan ?).
            """


    def test4(self):
        e = intrinsic_volumes.EC

        m = N.zeros((40,)*3)
        m[10,10,10] = 1
        assert_almost_equal(e(m), 1)

        m = N.zeros((40,)*3)
        m[10,10:12,10] = 1
        assert_almost_equal(e(m), 1)

        m[10,10:12,10:12] = 1
        m = N.zeros((40,)*3)
        m[10,10:12,10:12] = 1
        assert_almost_equal(e(m), 1)


##     for i in range(1, 6):
##         X = N.zeros((10,)*i)
##         if i in range(1,4):

##             for l in range(i+1):

##                 if i == 3:
##                     X[0:2,0:4,0:6] = 1
##                     answer = {0:1, 1:9, 2:23, 3:15}[l]
##                 elif i == 2:
##                     X[1:6,1:4] = 1
##                     answer = {0:1, 1:6, 2:8}[l]
##                 elif i == 1:
##                     X[4:6] = 1
##                     answer = {0:1, 1:1}[l]

##                 Y = N.indices(X.shape).astype(N.float)
##                 Y.shape = (i, 10**i)
##                 U = randorth()[0:i]
##                 Y = N.dot(U.T, Y)
##                 Y.shape = (p,) + (10,)*i
##                 print i, l, LK(X, 0.5, coords=Y, lk=l), answer
## ##                 if i == 3:
## ##                     file("LK%ddim%d.c" % (l, i), 'w').write(code(i, lk=l, explorer=True))







