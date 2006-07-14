import unittest
import numpy as N
import numpy.random as R
from neuroimaging.statistics import rft
from scipy.special import gammaln

def K(D=4, df_num=7, df_denom=20):
    """
    Determine coefficients of polynomial in Worsley (1994).
    """
    rhoF=N.zeros((D,D), N.float64)
    rhoCT=N.zeros((D,D), N.float64)
    a = N.arange(D)

    for dim in range(1, D+1):
        s1 = 0

        for j in range(int(N.floor((dim-1)/2.)+1)):
            t = (gammaln(dim) +
                 gammaln((df_denom+df_num-dim)/2.+j) -
                 gammaln(j+1) -
                 gammaln((df_denom+df_num-dim)/2.) +
                 gammaln(df_denom) -
                 gammaln(a-j+1) -
                 gammaln(df_denom-a+j) +
                 gammaln(df_num) -
                 gammaln(dim-a-j) -
                 gammaln(df_num-dim+j+a+1))
            s1 += N.exp(t) * N.power(-1., a+dim-1.)
        rhoF[dim-1] = s1

        s2=0
        for i in range(int(N.floor((df_num-1)/2.)+1)):
            k = df_num-1-2.*i
            mu = (N.log(2) * (k+1.) + N.log(N.pi) * k/2. +
                   gammaln((df_num+1)/2.) - gammaln(k+1.) - gammaln(i+1.))
            for j in range(int(N.floor((dim+k-1)/2.)+1)):

                t = (gammaln(dim+k) -
                     gammaln(j+1) -
                     gammaln(dim+k-2.*j) +
                     N.log(2) * ((dim+k-1)/2.-2*j) +
                     gammaln((df_denom+1)/2.) -
                     gammaln((df_denom+2-dim-k)/2.+j) -
                     N.log(2*N.pi)*(dim+k+1)/2. +
                     gammaln(i+1) -
                     gammaln(a-dim+i+j+2) -
                     gammaln(dim-a-j))

                s2 += N.exp(mu+t)* N.power(-1.,j)

        rhoCT[dim-1] = s2 / N.exp(gammaln((df_denom+df_num-dim)/2.) -
                                  gammaln(df_denom/2.) -
                                  gammaln(df_num/2.) - N.log(2)*(dim-1) -
                                  N.log(N.pi)*dim/2.)

    return rhoF

##     if N.allclose(rhoCT, rhoF):
##         return rhoF
##     else:
##         raise ValueError, 'cone T and F result don\'t agree!'

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
            for i in range(1, self.coef.ndim):
                p += self.coef[i] * x
                x *= x

            p *= (N.power(f, (self.df_num - self.dim) / 2.) *
                  N.power(1 + f, -(self.df_denom + self.df_num - 2.) / 2.) *
                  self.multiplier)
            return p
        else:
            return scipy.stats.f.sf(x, self.df_num, self.df_denom)

class ECDensityTest(unittest.TestCase):

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
        df_denom, df_num, dim = (10, 5, 3)
        x = N.fabs(R.standard_normal((10,)))
        N.testing.assert_almost_equal(self.F[df_denom][df_num].density(x, dim),
                                      self.kF[df_denom][df_num][dim](x))

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
