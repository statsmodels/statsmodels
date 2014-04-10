'''Cox proportional hazards regression model.


some dimension problems
fixed import errors
currently produces parameter estimate but then raises exception for other results


finally, after running the script several times, I get a OSError with too many
open file handles

updates and changes :

as of 2010-05-15
AttributeError: 'CoxPH' object has no attribute 'cachedir'
Traceback (most recent call last):
  File "C:\...\scikits\statsmodels\sandbox\cox.py", line 244, in <module>
    res = c.newton([0.4])
AttributeError: 'CoxPH' object has no attribute 'newton'

replaced newton by call to new fit method for mle with bfgs

feels very slow
need testcase before trying to fix

'''
from __future__ import print_function
from statsmodels.compat.python import iterkeys, range
import shutil
import tempfile

import numpy as np


from statsmodels.base import model
from . import survival

class Discrete(object):

    """
    A simple little class for working with discrete random vectors.

    Note: assumes x is 2-d and observations are in 0 axis, variables in 1 axis
    """

    def __init__(self, x, w=None):
        self.x = np.squeeze(x)
        if self.x.shape == ():
            self.x = np.array([self.x])
##        #JP added and removed again b/c still broadcast error
##        if self.x.ndim == 1:
##            self.x = self.x[:,None]
        self.n = self.x.shape[0]
        if w is None:
            w = np.ones(self.n, np.float64)
        else:
            if w.shape[0] != self.n:
                raise ValueError('incompatible shape for weights w')
            if np.any(np.less(w, 0)):
                raise ValueError('weights should be non-negative')
        self.w = w*1.0 / w.sum()

    def mean(self, f=None):   #JP: this is expectation, "expect" in mine
        if f is None:
            fx = self.x
        else:
            fx = f(self.x)
        return (fx * self.w).sum()

    def cov(self):
        mu = self.mean()  #JP: call to method (confusing name)
        dx = self.x - mu#np.multiply.outer(mu, self.x.shape[1])
        return np.dot(dx, np.transpose(dx))
##        if dx.ndim == 1:
##            dx = dx[:,None]
##        return np.dot(dx.T, dx)

class Observation(survival.RightCensored):

    def __getitem__(self, item):
        if self.namespace is not None:
            return self.namespace[item]
        else:
            return getattr(self, item)

    def __init__(self, time, delta, namespace=None):
        self.namespace = namespace
        survival.RightCensored.__init__(self, time, delta)

    def __call__(self, formula, time=None, **extra):
        return formula(namespace=self, time=time, **extra)

class CoxPH(model.LikelihoodModel):
    """Cox proportional hazards regression model."""

    def __init__(self, subjects, formula, time_dependent=False):
        self.subjects, self.formula = subjects, formula
        self.time_dependent = time_dependent
        self.initialize(self.subjects)

    def initialize(self, subjects):
        print('called initialize')
        self.failures = {}
        for i in range(len(subjects)):
            s = subjects[i]
            if s.delta:
                if s.time not in self.failures:
                    self.failures[s.time] = [i]
                else:
                    self.failures[s.time].append(i)

        self.failure_times = list(iterkeys(self.failures))
        self.failure_times.sort()

    def cache(self):
        if self.time_dependent:
            self.cachedir = tempfile.mkdtemp()

        self.design = {}
        self.risk = {}
        first = True

        for t in iterkeys(self.failures):
            if self.time_dependent:
                d = np.array([s(self.formula, time=t)
                             for s in self.subjects]).astype(float)[:,None]
                dshape = d.shape
                dfile = file(tempfile.mkstemp(dir=self.cachedir)[1], 'w')
                d.tofile(dfile)
                dfile.close()
                del(d)
                self.design[t] = np.memmap(dfile.name,
                                          dtype=np.dtype(float),
                                          shape=dshape)
            elif first:
                d = np.array([s(self.formula, time=t)
                             for s in self.subjects]).astype(np.float64)
                self.design[t] = d
            else:
                self.design[t] = d
            self.risk[t] = np.compress([s.atrisk(t) for s in self.subjects],
                                      np.arange(self.design[t].shape[0]),axis=-1)
# this raised exception on exit,
    def __del__(self):
        try:
            shutil.rmtree(self.cachedir, ignore_errors=True)
        except AttributeError:
            print("AttributeError: 'CoxPH' object has no attribute 'cachedir'")
            pass

    def loglike(self, b, ties='breslow'):

        logL = 0
        for t in iterkeys(self.failures):
            fail = self.failures[t]
            d = len(fail)
            risk = self.risk[t]
            Zb = np.dot(self.design[t], b)

            logL += Zb[fail].sum()

            if ties == 'breslow':
                s = np.exp(Zb[risk]).sum()
                logL -= np.log(np.exp(Zb[risk]).sum()) * d
            elif ties == 'efron':
                s = np.exp(Zb[risk]).sum()
                r = np.exp(Zb[fail]).sum()
                for j in range(d):
                    logL -= np.log(s - j * r / d)
            elif ties == 'cox':
                raise NotImplementedError('Cox tie breaking method not \
implemented')
            else:
                raise NotImplementedError('tie breaking method not recognized')
        return logL

    def score(self, b, ties='breslow'):

        score = 0
        for t in iterkeys(self.failures):
            fail = self.failures[t]
            d = len(fail)
            risk = self.risk[t]
            Z = self.design[t]

            score += Z[fail].sum()

            if ties == 'breslow':
                w = np.exp(np.dot(Z, b))
                rv = Discrete(Z[risk], w=w[risk])
                score -= rv.mean() * d
            elif ties == 'efron':
                w = np.exp(np.dot(Z, b))
                score += Z[fail].sum()
                for j in range(d):
                    efron_w = w
                    efron_w[fail] -= i * w[fail] / float(d)
                    rv = Discrete(Z[risk], w=efron_w[risk])
                    score -= rv.mean()
            elif ties == 'cox':
                raise NotImplementedError('Cox tie breaking method not \
implemented')
            else:
                raise NotImplementedError('tie breaking method not recognized')
        return np.array([score])

    def information(self, b, ties='breslow'):

        info = 0 #np.zeros((len(b),len(b))) #0
        score = 0
        for t in iterkeys(self.failures):
            fail = self.failures[t]
            d = len(fail)
            risk = self.risk[t]
            Z = self.design[t]

            if ties == 'breslow':
                w = np.exp(np.dot(Z, b))
                rv = Discrete(Z[risk], w=w[risk])
                info += rv.cov()
            elif ties == 'efron':
                w = np.exp(np.dot(Z, b))
                score += Z[fail].sum()
                for j in range(d):
                    efron_w = w
                    efron_w[fail] -= i * w[fail] / d
                    rv = Discrete(Z[risk], w=efron_w[risk])
                    info += rv.cov()
            elif ties == 'cox':
                raise NotImplementedError('Cox tie breaking method not \
implemented')
            else:
                raise NotImplementedError('tie breaking method not recognized')
        return score

if __name__ == '__main__':
    import numpy.random as R
    n = 100
    X = np.array([0]*n + [1]*n)
    b = 0.4
    lin = 1 + b*X
    Y = R.standard_exponential((2*n,)) / lin
    delta = R.binomial(1, 0.9, size=(2*n,))

    subjects = [Observation(Y[i], delta[i]) for i in range(2*n)]
    for i in range(2*n):
        subjects[i].X = X[i]

    import statsmodels.sandbox.formula as F
    x = F.Quantitative('X')
    f = F.Formula(x)

    c = CoxPH(subjects, f)

#    c.cache()
    # temp file cleanup doesn't work on windows
    c = CoxPH(subjects, f, time_dependent=True)
    c.cache() #this creates  tempfile cache,
    # no tempfile cache is created in normal use of CoxPH


    #res = c.newton([0.4])  #doesn't work anymore
    res=c.fit([0.4],method="bfgs")
    print(res.params)
    print(dir(c))
    #print c.fit(Y)
    #c.information(res.params)  #raises exception

    '''
    Note: Replacement for c.newton

    >>> c.fit()
    Traceback (most recent call last):
      File "<pyshell#1>", line 1, in <module>
        c.fit()
      File "C:\Josef\eclipsegworkspace\statsmodels-josef-experimental\scikits\statsmodels\model.py", line 132, in fit
        start_params = [0]*self.exog.shape[1] # will fail for shape (K,)
    AttributeError: 'CoxPH' object has no attribute 'exog'
    >>> c.fit([0.4])
    Traceback (most recent call last):
      File "<pyshell#2>", line 1, in <module>
        c.fit([0.4])
      File "C:\Josef\eclipsegworkspace\statsmodels-josef-experimental\scikits\statsmodels\model.py", line 148, in fit
        H = self.hessian(history[-1])
      File "C:\Josef\eclipsegworkspace\statsmodels-josef-experimental\scikits\statsmodels\model.py", line 115, in hessian
        raise NotImplementedError
    NotImplementedError
    >>> c.fit([0.4],method="bfgs")
    Optimization terminated successfully.
             Current function value: 802.354181
             Iterations: 3
             Function evaluations: 5
             Gradient evaluations: 5
    <statsmodels.model.LikelihoodModelResults object at 0x01D48B70>
    >>> res=c.fit([0.4],method="bfgs")
    Optimization terminated successfully.
             Current function value: 802.354181
             Iterations: 3
             Function evaluations: 5
             Gradient evaluations: 5
    >>> res.params
    array([ 0.34924421])
'''
