# -*- coding: utf-8 -*-
"""

Created on Fri Mar 09 16:00:27 2012

Author: Josef Perktold
"""
from __future__ import print_function
from statsmodels.compat.python import iterkeys, cPickle, BytesIO
import numpy as np
import statsmodels.api as sm

from numpy.testing import assert_

from nose import SkipTest
import platform


iswin = platform.system() == 'Windows'
npversionless15 = np.__version__ < '1.5'
winoldnp = iswin & npversionless15


def check_pickle(obj):
    fh = BytesIO()
    cPickle.dump(obj, fh, protocol=cPickle.HIGHEST_PROTOCOL)
    plen = fh.tell()
    fh.seek(0, 0)
    res = cPickle.load(fh)
    fh.close()
    return res, plen


class RemoveDataPickle(object):

    def __init__(self):
        self.predict_kwds = {}

    @classmethod
    def setup_class(self):

        nobs = 10000
        np.random.seed(987689)
        x = np.random.randn(nobs, 3)
        x = sm.add_constant(x)
        self.exog = x
        self.xf = 0.25 * np.ones((2, 4))

    def test_remove_data_pickle(self):
        if winoldnp:
            raise SkipTest
        results = self.results
        xf = self.xf

        pred_kwds = self.predict_kwds
        pred1 = results.predict(xf, **pred_kwds)
        #create some cached attributes
        results.summary()
        res = results.summary2()  # SMOKE test also summary2

        # uncomment the following to check whether tests run (7 failures now)
        #np.testing.assert_equal(res, 1)

        #check pickle unpickle works on full results
        #TODO: drop of load save is tested
        res, l = check_pickle(results._results)

        #remove data arrays, check predict still works
        results.remove_data()
        pred2 = results.predict(xf, **pred_kwds)
        np.testing.assert_equal(pred2, pred1)

        #pickle, unpickle reduced array
        res, l = check_pickle(results._results)

        #for testing attach res
        self.res = res

        #Note: l_max is just a guess for the limit on the length of the pickle
        l_max = 20000
        assert_(l < l_max, msg='pickle length not %d < %d' % (l, l_max))

        pred3 = results.predict(xf, **pred_kwds)
        np.testing.assert_equal(pred3, pred1)

    def test_remove_data_docstring(self):
        assert_(self.results.remove_data.__doc__ is not None)

    def test_pickle_wrapper(self):

        fh = BytesIO()  # use cPickle with binary content

        # test unwrapped results load save pickle
        self.results._results.save(fh)
        fh.seek(0, 0)
        res_unpickled = self.results._results.__class__.load(fh)
        assert_(type(res_unpickled) is type(self.results._results))

        # test wrapped results load save
        fh.seek(0, 0)
        self.results.save(fh)
        fh.seek(0, 0)
        res_unpickled = self.results.__class__.load(fh)
        fh.close()
        # print type(res_unpickled)
        assert_(type(res_unpickled) is type(self.results))

        before = sorted(iterkeys(self.results.__dict__))
        after = sorted(iterkeys(res_unpickled.__dict__))
        assert_(before == after, msg='not equal %r and %r' % (before, after))

        before = sorted(iterkeys(self.results._results.__dict__))
        after = sorted(iterkeys(res_unpickled._results.__dict__))
        assert_(before == after, msg='not equal %r and %r' % (before, after))

        before = sorted(iterkeys(self.results.model.__dict__))
        after = sorted(iterkeys(res_unpickled.model.__dict__))
        assert_(before == after, msg='not equal %r and %r' % (before, after))

        before = sorted(iterkeys(self.results._cache))
        after = sorted(iterkeys(res_unpickled._cache))
        assert_(before == after, msg='not equal %r and %r' % (before, after))


class TestRemoveDataPickleOLS(RemoveDataPickle):

    def setup(self):
        #fit for each test, because results will be changed by test
        x = self.exog
        np.random.seed(987689)
        y = x.sum(1) + np.random.randn(x.shape[0])
        self.results = sm.OLS(y, self.exog).fit()


class TestRemoveDataPickleWLS(RemoveDataPickle):

    def setup(self):
        #fit for each test, because results will be changed by test
        x = self.exog
        np.random.seed(987689)
        y = x.sum(1) + np.random.randn(x.shape[0])
        self.results = sm.WLS(y, self.exog, weights=np.ones(len(y))).fit()


class TestRemoveDataPicklePoisson(RemoveDataPickle):

    def setup(self):
        #fit for each test, because results will be changed by test
        x = self.exog
        np.random.seed(987689)
        y_count = np.random.poisson(np.exp(x.sum(1) - x.mean()))
        model = sm.Poisson(y_count, x)  #, exposure=np.ones(nobs), offset=np.zeros(nobs)) #bug with default
        # use start_params to converge faster
        start_params = np.array([0.75334818, 0.99425553, 1.00494724, 1.00247112])
        self.results = model.fit(start_params=start_params, method='bfgs',
                                 disp=0)

        #TODO: temporary, fixed in master
        self.predict_kwds = dict(exposure=1, offset=0)

class TestRemoveDataPickleNegativeBinomial(RemoveDataPickle):

    def setup(self):
        #fit for each test, because results will be changed by test
        np.random.seed(987689)
        data = sm.datasets.randhie.load()
        exog = sm.add_constant(data.exog, prepend=False)
        mod = sm.NegativeBinomial(data.endog, data.exog)
        self.results = mod.fit(disp=0)

class TestRemoveDataPickleLogit(RemoveDataPickle):

    def setup(self):
        #fit for each test, because results will be changed by test
        x = self.exog
        nobs = x.shape[0]
        np.random.seed(987689)
        y_bin = (np.random.rand(nobs) < 1.0 / (1 + np.exp(x.sum(1) - x.mean()))).astype(int)
        model = sm.Logit(y_bin, x)  #, exposure=np.ones(nobs), offset=np.zeros(nobs)) #bug with default
        # use start_params to converge faster
        start_params = np.array([-0.73403806, -1.00901514, -0.97754543, -0.95648212])
        self.results = model.fit(start_params=start_params, method='bfgs', disp=0)


class TestRemoveDataPickleRLM(RemoveDataPickle):

    def setup(self):
        #fit for each test, because results will be changed by test
        x = self.exog
        np.random.seed(987689)
        y = x.sum(1) + np.random.randn(x.shape[0])
        self.results = sm.RLM(y, self.exog).fit()


class TestRemoveDataPickleGLM(RemoveDataPickle):

    def setup(self):
        #fit for each test, because results will be changed by test
        x = self.exog
        np.random.seed(987689)
        y = x.sum(1) + np.random.randn(x.shape[0])
        self.results = sm.GLM(y, self.exog).fit()

if __name__ == '__main__':
    for cls in [TestRemoveDataPickleOLS, TestRemoveDataPickleWLS,
                TestRemoveDataPicklePoisson,
                TestRemoveDataPickleNegativeBinomial,
                TestRemoveDataPickleLogit, TestRemoveDataPickleRLM,
                TestRemoveDataPickleGLM]:
        print(cls)
        cls.setup_class()
        tt = cls()
        tt.setup()
        tt.test_remove_data_pickle()
        tt.test_remove_data_docstring()
        tt.test_pickle_wrapper()
