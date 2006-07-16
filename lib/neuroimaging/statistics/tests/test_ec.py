## import unittest
import numpy as N
from neuroimaging.statistics import rft
from scipy.special import gammaln

## class ECDensityTest(unittest.TestCase):
##     pass

def K(D=4, d=7, m=20):

    rhoF=N.zeros((D,D), N.float64)
    rhoCT=N.zeros((D,D), N.float64)
    a = N.arange(D)

    for n in range(1, D+1):
        s1 = 0

        for j in range(int(N.floor((n-1)/2.)+1)):
            t = (gammaln(n) +
                 gammaln((m+d-n)/2.+j) - gammaln(j+1) - gammaln((m+d-n)/2.) +
                 gammaln(m) - gammaln(a-j+1) - gammaln(m-a+j) +
                 gammaln(d) - gammaln(n-a-j) - gammaln(d-n+j+a+1))
            s1 += N.exp(t) * N.power(-1., a+n-1.)
        rhoF[n-1] = s1

        s2=0
        for i in range(int(N.floor((d-1)/2.)+1)):
            k = d-1-2.*i
            muk = (N.log(2) * (k+1.) + N.log(N.pi) * k/2. +
                   gammaln((d+1)/2.) - gammaln(k+1.) - gammaln(i+1.))
            for j in range(int(N.floor((n+k-1)/2.)+1)):
                t = (gammaln(n+k) - gammaln(j+1) - gammaln(n+k-2.*j))
                t +=  N.log(2) * ((n+k-1)/2.-2*j) + gammaln((m+1)/2.)
                t = (t - gammaln((m+2-n-k)/2.+j) -
                     N.log(2*N.pi)*(n+k+1)/2. +
                     gammaln(i+1) - gammaln(a-n+i+j+2) - gammaln(n-a-j))
                s2 += N.exp(muk+t)* N.power(-1.,j)

        rhoCT[n-1] = s2 / N.exp(gammaln((m+d-n)/2.) - gammaln(m/2.) -
                                gammaln(d/2.) - N.log(2)*(n-1) -
                                N.log(N.pi)*n/2.)
    return rhoCT

## print K()
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
