import unittest
import numpy as N
import numpy.random as R
from neuroimaging.statistics import rft
from scipy.special import gammaln

class keithT:

    def __init__(self, m):
        self.m = m

    def __call__(self, x, j=0):
        x = N.asarray(x, N.float64)
        _j = j - 1
        if _j >= 0:
            c = pow(1. + x**2/m, -0.5*(m-1.))
            c *= pow(2*N.pi, -0.5*(j+1))

            q = 0
            for i in range(int(N.floor((_j/2.))+1)):

                tmp = pow(x, _j-2*i) / (factorial(_j-2*i) * factorial(i) * pow(-2.0, i))
                tmp *= N.exp(gammaln((m+1.)/2) - gammaln((m+1.-_j+2*i)/2.)) * pow(2./m, (_j-2*i)/2.)
                tmp *= factorial(_j)
                q += tmp

            return q * c
        else:
            return scipy.stats.t.sf(x, self.m)

class keithF:

    def __init__(self, m, n):
        self.m = m
        self.n = n

    def __call__(self, x, j=0):
        x = N.asarray(x, N.float64)
        m = self.m
        n = self.n
        if j >= 1:
            c = pow(n*x/m, 0.5*(n-j))
            c *= pow(2*N.pi, -0.5*j)
            c *= pow(2, -0.5*(j-2))
            c *= N.exp(gammaln((m+n-j)/2.) - gammaln(m/2.) - gammaln(n/2.))
            c *= pow(1. + n*x/m, -0.5*(m+n-2.))
            c *= pow(-1., j-1) * N.exp(gammaln(j))

            q = 0

            for i in range(int(N.floor(((j-1)/2.))+1)):
                cc = N.exp(gammaln((m+n-j)/2. + i) - gammaln((m+n-j)/2.) - gammaln(i+1))
                for k in range(j - 2*i):
                    tmp = rft.binomial(m-1,k) * rft.binomial(n-1, j-1-2*i-k)
                    tmp *= pow(-1., i+k) * pow(n*x/m, i+k)
                q += tmp * cc

            return q * c
        else:
            return scipy.stats.f.sf(x,self.n, self.m)

class keithChi:

    def __init__(self, n):
        self.n = n

    def __call__(self, x, j=0):
        x = N.array(x, N.float64)
        n = self.n
        if j >= 1:
            c = N.power(x, 0.5*(n-j))
            c *= N.power(2*N.pi, -0.5*j)
            c *= N.power(2, -0.5*(n-2))
            c *= N.exp(-gammaln(n/2.))
            c *= N.exp(-x/2.)

            q = 0

            for i in range(int(N.floor(((j-1)/2.))+1)):
                for k in range(j - 2*i):
                    tmp = rft.binomial(n-1, j-1-2*i-k)
                    tmp *= N.power(-1., j-1+i+k) * N.power(x, i+k)
                    tmp *= N.exp(gammaln(j) - gammaln(i+1) - gammaln(k+1))
                    tmp *= N.power(2., -i)
                    q += tmp

            return q * c
        else:
            return scipy.stats.chisqprob(x, self.n)

class FDensity:
    """
    F EC density from Worsley(1994).
    """

    def __init__(self, dim, df_denom, df_num):
        self.dim = dim
        self.df_denom = df_denom
        self.df_num = df_num

        if dim > 1:
            self.coef = K(D=dim, df_denom=df_denom, df_num=df_num)[dim-1]

        self.multiplier = N.exp(gammaln((df_denom+df_num-dim)/2.) -
                                gammaln(df_num/2.) -
                                gammaln(df_denom/2.) -
                                N.log(2) * (dim - 2) / 2. -
                                N.log(2*N.pi) * dim / 2.)

    def __call__(self, x):

        if self.dim > 0:
            x *= self.df_num * 1. / self.df_denom
            f = N.array(x, copy=True)

            p = self.coef[0]
            for i in range(1, self.coef.shape[0]):
                p += self.coef[i] * x
                x *= x

            p *= (N.power(f, (self.df_num - self.dim) / 2.) *
                  N.power(1 + f, -(self.df_denom + self.df_num - 2.) / 2.) *
                  self.multiplier)
            return p
        else:
            return scipy.stats.f.sf(x, self.df_num, self.df_denom)

class KQTest(unittest.TestCase):

    """
    Verify Q and K agree
    """

    def test_QK(self):
        from neuroimaging.statistics.rft import Q, K
        df_denom=30
        dim = 3
        x = N.fabs(R.standard_normal((10,)))
        q = Q(dim, df_denom=df_denom)(x)
        k = K(dim=dim, df_denom=df_denom, df_num=1)(x**2/df_denom)
        N.testing.assert_almost_equal(q, k)

class FDensityTest(unittest.TestCase):

    def setUp(self):
        df_denom = range(10,60,10)
        df_num = range(4,15)
        dim = range(4)
        self.kF = {}
        self.F = {}

        for m in df_denom:
            self.kF[m] = {}
            self.F[m] = {}
            for n in df_num:
                self.kF[m][n] = {}
                self.F[m][n] = rft.FStat(n=n, m=m)
                for k in dim:
                    self.kF[m][n][k] = FDensity(k, m, n)


    def test_F(self):
        df_denom, df_num, dim = (N.inf, 5, 3)
        x = N.fabs(R.standard_normal((10,)))
        N.testing.assert_almost_equal(self.F[df_denom][df_num].density(x, dim),
                                      self.kF[df_denom][df_num][dim](x))

class FHermiteDensityTest(unittest.TestCase):

    def setUp(self):
        df_denom = N.inf
        df_num = 1
        dim = range(5)
        self.kF = {}
        self.F = rft.FStat(m=N.inf,n=1)

        for k in dim:
            self.kF[k] = FDensity(k, df_denom, df_num)

    def test_ratio(self):
        dim = 4

        x = N.fabs(R.standard_normal((10,))) * 3
        a = self.kF[dim](x**2)
        b = (x**3 - 3 * x) * N.exp(-x**2/2) / N.power(N.pi, (dim+1)/2.)
        N.testing.assert_almost_equal(a, b)


class TDensityTest(unittest.TestCase):

    def setUp(self):
        df_denom = range(10,60,10)
        dim = range(4)
        self.kT = {}
        self.T = {}

        for m in df_denom:
            self.kT[m] = {}
            self.T[m] = rft.TStat(m=m)
            for k in dim:
                self.kT[m][k] = FDensity(k, m, 1)


    def test_T(self):
        df_denom, df_num, dim = (10, 5, 3)
        x = N.fabs(R.standard_normal((10,)))
        a = self.T[df_denom].density(x, dim),
        b = 0.5 * self.kT[df_denom][dim](N.sqrt(x))
        print a/b
        N.testing.assert_almost_equal(a, b)

if __name__ == '__main__':
    unittest.main()


## p = K()
## print p[1](4.5)
## c = ChiBarSquared(4)
## c = ChiBarSquared(5)
## f = FStat(n=7, m=20)
## g = Gaussian()
## import pylab
## x = N.linspace(0,10,100)
## print f.polynomial(x, 3)

## pylab.plot(x, f.polynomial(x, 3))

## from scipy.sandbox.models.regression import OLSModel
## from scipy.sandbox.models.formula import Formula, Quantitative, I

## namespace = {'x':x}
## X = Quantitative('x')
## formula = I + X
## order = 8
## for i in range(2, order+1):
##     formula += X**i
## design = formula.design(namespace=namespace)
## model = OLSModel(design)
## results = model.fit(f.polynomial(x, 3))
## print results.beta, formula.names()


## def _f(x):
##     t = results.beta[0]
##     names = formula.names()
##     for term in formula.terms:
##         if hasattr(term, 'power'):
##             i = names.index(term.termname)
##             t += results.beta[i] * x**i
##         elif term.termname == 'x':
##             i = names.index(term.termname)
##             t += results.beta[i] * x

##     return t
## print dir(results)
## pylab.plot(x, _f(x))

## ## pylab.plot(x, (x**2 - 1) / N.power(2*N.pi, 2))
## ## pylab.figure()
## ## a = g.polynomial(x, 3) / (rho(x, 3) * N.exp(x**2/2) * N.power(2*N.pi, -2))
## ## print N.log(a.mean()) / N.log(N.sqrt(2*N.pi))
## ## pylab.plot(x, a)
## pylab.show()

## ## m=1000
## ## f = Fstat(3,m, search=[3,4])
## ## x = ChiSquared(3, search=[3,4])
## ## r = Roy(3,m,1, search=[3,4])

## ## print f(2, j=2), x(2*3, j=2), r(2, j=2), r.LK, f.LK
