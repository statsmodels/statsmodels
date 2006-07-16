import unittest
import numpy as N
import numpy.random as R
from neuroimaging.statistics import rft
from scipy.special import gammaln, hermitenorm
import scipy.stats

def polyF(dim, dfd=N.inf, dfn=1):
    """
    Return the polynomial part of the EC density when
    evaluating the polynomial on the sqrt(F) scale (or sqrt(chi^2)=chi scale).

    The polynomial is such that, if dfd=inf, the F EC density in is just

    polyF(dim,dfn=dfn)(sqrt(dfn*x)) * exp(-dfn*x/2) * (2\pi)^{-(dim+1)/2}

    """

    n = float(dfn)
    m = float(dfd)
    D = float(dim)

    p = rft.K(dim=D, dfd=m, dfn=n)
    c = p.c

    # Take care of the powers of n (i.e. we want polynomial K evaluated
    # at */n).

    for i in range(p.order+1):
        c[i] /= N.power(n, p.order-i)

    # Now, turn it into a polynomial of x when evaluated at x**2

    C = N.zeros((2*c.shape[0]-1,), N.float64)

    for i in range(c.shape[0]):
        C[2*i] = c[i]

    # Multiply by the factor x^(dfn-dim) in front (see Theorem 4.6 of
    # Worsley (1994), cited above.

    if dim > dfn: # divide by x^(dim-dfn)
        C = C[0:(C.shape[0] - (dim-dfn))]
    else: # multiply by x^(dim-dfn)
        C = N.hstack([C, N.zeros((dfn-dim,))])

    # Fix up constant in front

    if N.isfinite(m):
        C *= N.exp(gammaln((m+n-D)/2.) - gammaln(m/2.)) * N.power(m, -(n-D)/2.)
    else:
        C *= N.power(2, -(n-D)/2.)

    C /= N.power(2, (dim-2)/2.) * N.exp(gammaln(n/2.))
    C *= N.sqrt(2*N.pi)
    return N.poly1d(C)

def F_alternative(x, dim, dfd=N.inf, dfn=1):
    """
    Another way to compute F EC density as a product of a
    polynomial and a power of (1+x^2/m).
    """

    n = float(dfn)
    m = float(dfd)
    D = float(dim)

    x = N.asarray(x, N.float64)
    p = polyF(dim=dim, dfd=dfd, dfn=dfn)
    v = p(N.sqrt(n*x))

    if N.isfinite(m):
        v *= N.power(1 + n*x/m, -(m+n-2.) / 2.)
    else:
        v *= N.exp(-n*x/2)
    v *= N.power(2*N.pi, -(dim+1)/2.)
    return v


class RFTTest(unittest.TestCase):


    def test_polynomial1(self):
        """
        Polynomial part of Gaussian densities are Hermite polynomials.
        """
        for dim in range(1,10):
            q = rft.Gaussian().quasi(dim)
            h = hermitenorm(dim-1)
            N.testing.assert_almost_equal(q.c, h.c)

    def test_polynomial2(self):
        """
        EC density of chi^2(1) is 2 * EC density of Gaussian so
        polynomial part is a factor of 2 as well.
        """
        for dim in range(1,10):
            q = rft.ChiSquared(dfn=1).quasi(dim)
            h = hermitenorm(dim-1)
            N.testing.assert_almost_equal(q.c, 2*h.c)



    def test_polynomial3(self):
        """
        EC density of F with infinite dfd is the same as chi^2 --
        polynomials should be the same.
        """
        for dim in range(1,10):
            for dfn in range(5,10):
                q1 = rft.FStat(dfn=dfn, dfd=N.inf).quasi(dim)
                q2 = rft.ChiSquared(dfn=dfn).quasi(dim)
                N.testing.assert_almost_equal(q1.c, q2.c)


    def test_chi1(self):
        """
        EC density of F with infinite dfd is the same as chi^2 --
        EC should be the same.
        """

        x = N.linspace(0.1,10,100)
        for dim in range(1,10):
            for dfn in range(5,10):
                c = rft.ChiSquared(dfn=dfn)
                f = rft.FStat(dfn=dfn, dfd=N.inf)
                chi1 = c.density(dfn*x, dim)
                chi2 = f.density(x, dim)
                N.testing.assert_almost_equal(chi1, chi2)

    def test_chi2(self):
        """
        Quasi-polynomial part of the chi^2 EC density should
        be the limiting polyF.
        """
        x = N.linspace(0.1,10,100)
        for dim in range(1,10):
            for dfn in range(5,10):
                c = rft.ChiSquared(dfn=dfn)
                p1 = c.quasi(dim=dim)
                p2 = polyF(dim=dim, dfn=dfn)
                N.testing.assert_almost_equal(p1.c, p2.c)

    def test_chi3(self):
        """
        EC density of chi^2(1) is 2 * EC density of Gaussian squared so
        EC densities factor of 2 as well.

        """

        x = N.linspace(0.1,10,100)
        for dim in range(1,10):
            g = rft.Gaussian()
            c = rft.ChiSquared(dfn=1)
            ec1 = g.density(N.sqrt(x), dim)
            ec2 = c.density(x, dim)
            N.testing.assert_almost_equal(2*ec1, ec2)



    def test_F1(self):
        x = N.linspace(0.1,10,100)
        for dim in range(1,10):
            for dfn in range(5,10):
                for dfd in [40,50,N.inf]:
                    f1 = rft.F(x, dim, dfn=dfn, dfd=dfd)
                    f2 = F_alternative(x, dim, dfn=dfn, dfd=dfd)
                    N.testing.assert_almost_equal(f1, f2)

    def test_F2(self):
        x = N.linspace(0.1,10,100)
        for dim in range(3,7):
            for dfn in range(5,10):
                for dfd in [40,50,N.inf]:
                    f1 = rft.FStat(dfn=dfn, dfd=dfd).density(x, dim)
                    f2 = F_alternative(x, dim, dfn=dfn, dfd=dfd)
                    N.testing.assert_almost_equal(f1, f2)

    def test_F3(self):
        x = N.linspace(0.1,10,100)
        for dim in range(3,7):
            for dfn in range(5,10):
                for dfd in [40,50,N.inf]:
                    f1 = rft.FStat(dfn=dfn, dfd=dfd).density(x, dim)
                    f2 = rft.F(x, dim, dfn=dfn, dfd=dfd)
                    N.testing.assert_almost_equal(f1, f2)

    def test_T1(self):
        """
        O-dim EC density should be tail probality.
        """

        x = N.linspace(0.1,10,100)

        for dfd in [40,50]:
            t = rft.TStat(dfd=dfd)
            N.testing.assert_almost_equal(t(x), scipy.stats.t.sf(x, dfd))

        t = rft.TStat(dfd=N.inf)
        N.testing.assert_almost_equal(t(x), scipy.stats.norm.sf(x))

    def test_T2(self):
        """
        T is an F with dfn=1
        """

        x = N.linspace(0.1,10,100)

        for dfd in [40,50,N.inf]:
            t = rft.TStat(dfd=dfd)
            f = rft.FStat(dfd=dfd, dfn=1)
            for dim in range(7):
                N.testing.assert_almost_equal(t.density(x, dim), f.density(x**2, dim))

    def test_hotelling1(self):
        """
        Asymptotically, Hotelling is the same as F which is the same
        as chi^2.
        """
        x = N.linspace(0.1,10,100)
        for dim in range(1,7):
            for dfn in range(5,10):
                h = rft.Hotelling(k=dfn).density(x, dim)
                f = rft.FStat(dfn=dfn).density(x, dim)
                N.testing.assert_almost_equal(h, f)

    def test_hotelling2(self):
        """
        Marginally, Hotelling is an F field.
        """

        x = N.linspace(0.1,10,100)
        for dfn in range(5, 10):
            for dfd in [40,50]:
                h = rft.Hotelling(dfd=dfd,k=dfn)(x*dfn)
                f = scipy.stats.f.sf(x, dfn, dfd)
                N.testing.assert_almost_equal(h, f)

            h = rft.Hotelling(k=dfn)(x)
            chi2 = scipy.stats.chi2.sf(x, dfn, dfd)
            N.testing.assert_almost_equal(h, chi2)



    def test_search3(self):
        """
        In the Gaussian case, test that search and product give same results.

        """
        search = rft.IntrinsicVolumes([3,4,5,7])
        g1 = rft.Gaussian(search=search)
        g2 = rft.Gaussian(product=search)
        x = N.linspace(0.1,10,100)
        y1 = g1(x)
        y2 = g2(x)
        N.testing.assert_almost_equal(y1, y2)



    def test_search(self):
        """
        Test that the search region works.
        """

        search = rft.IntrinsicVolumes([3,4,5])
        x = N.linspace(0.1,10,100)

        stat = rft.Gaussian(search=search)

        v1 = stat(x)
        v2 = ((5*x + 4*N.sqrt(2*N.pi)) *
              N.exp(-x**2/2.) / N.power(2*N.pi, 1.5) +
              3 * scipy.stats.norm.sf(x))
        N.testing.assert_almost_equal(v1, v2)

    def test_search2(self):
        """
        Test that the search region works.
        """

        search = rft.IntrinsicVolumes([3,4,5])
        x = N.linspace(0.1,10,100)

        stats = [rft.Gaussian(search=search)]
        ostats = [rft.Gaussian()]

        for dfn in range(5,10):
            for dfd in [40,50,N.inf]:
                stats.append(rft.FStat(dfn=dfn, dfd=dfd, search=search))
                ostats.append(rft.FStat(dfn=dfn, dfd=dfd))
                stats.append(rft.TStat(dfd=dfd, search=search))
                ostats.append(rft.TStat(dfd=dfd))
            stats.append(rft.ChiSquared(dfn=dfn, search=search))
            ostats.append(rft.ChiSquared(dfn=dfn))

        for i in range(len(stats)):
            stat = stats[i]
            ostat = ostats[i]
            v1 = stat(x)
            v2 = 0

            for j in range(search.mu.shape[0]):
                v2 += ostat.density(x, j) * search.mu[j]
            N.testing.assert_almost_equal(v1, v2)

    def test_T2(self):
        """
        T is an F with dfn=1
        """

        x = N.linspace(0,5,101)

        for dfd in [40,50,N.inf]:
            t = rft.TStat(dfd=dfd)
            f = rft.FStat(dfd=dfd, dfn=1)
            for dim in range(7):
                y = 2*t.density(x, dim)
                z = f.density(x**2, dim)
                N.testing.assert_almost_equal(y, z)



    def test_search1(self):
        """
        Test that the search region works.
        """

        search = rft.IntrinsicVolumes([3,4,5])
        x = N.linspace(0.1,10,100)

        stats = [rft.Gaussian()]

        for dfn in range(5,10):
            for dfd in [40,50,N.inf]:
                stats.append(rft.FStat(dfn=dfn, dfd=dfd))
                stats.append(rft.TStat(dfd=dfd))
            stats.append(rft.ChiSquared(dfn=dfn))

        for dim in range(7):
            for stat in stats:
                v1 = stat(x, search=search)
                v2 = 0
                for i in range(search.mu.shape[0]):
                    v2 += stat.density(x, i) * search.mu[i]
                import pylab


    def test_search4(self):
        """
        Test that the search/product work well together
        """

        search = rft.IntrinsicVolumes([3,4,5])
        product = rft.IntrinsicVolumes([1,2])
        x = N.linspace(0.1,10,100)

        g1 = rft.Gaussian()
        g2 = rft.Gaussian(product=product)

        y = g2(x, search=search)
        z = g1(x, search=search*product)
        N.testing.assert_almost_equal(y, z)


    def test_search5(self):
        """
        Test that the search/product work well together
        """

        search = rft.IntrinsicVolumes([3,4,5])
        product = rft.IntrinsicVolumes([1,2])
        prodsearch = product * search
        x = N.linspace(0,5,101)

        g1 = rft.Gaussian()
        g2 = rft.Gaussian(product=product)

        z = 0

        for i in range(prodsearch.mu.shape[0]):
            z += g1.density(x, i) * prodsearch.mu[i]
        y = g2(x, search=search)
        N.testing.assert_almost_equal(y, z)



    def test_hotelling4(self):
        """
        Hotelling T^2 should just be like taking product with sphere.

        """

        x = N.linspace(0.1,10,100)
        for dim in range(1,7):
            for k in range(5, 10):
                p = rft.spherical_search(k)
                for dfd in [N.inf,40,50]:
                    t = rft.FStat(dfd=dfd, dfn=1)(x, search=p)
                    h = rft.Hotelling(k=k, dfd=dfd).density(x, dim)
                    N.testing.assert_almost_equal(h, t)
