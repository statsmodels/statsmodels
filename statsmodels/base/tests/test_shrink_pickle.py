# -*- coding: utf-8 -*-
"""

Created on Fri Mar 09 16:00:27 2012

Author: Josef Perktold
"""

import pickle, StringIO
import numpy as np
import statsmodels.api as sm

from numpy.testing import assert_, assert_almost_equal, assert_equal


def check_pickle(obj):
    from statsmodels.compatnp.py3k import BytesIO
    fh = BytesIO()
    pickle.dump(obj, fh)
    plen = fh.tell()
    fh.seek(0,0)
    res = pickle.load(fh)
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
        x = sm.add_constant(x, prepend=True)
        self.exog = x
        self.xf = 0.25 * np.ones((2,4))


    def test_remove_data_pickle(self):
        results = self.results
        xf = self.xf

        pred_kwds = self.predict_kwds
        pred1 = results.predict(xf, **pred_kwds)
        #create some cached attributes
        results.summary()

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

        #Note: 10000 is just a guess for the limit on the length of the pickle
        assert_(l < 10000, msg='pickle length not %d < %d' % (l, 10000))

        pred3 = results.predict(xf, **pred_kwds)
        np.testing.assert_equal(pred3, pred1)

    def test_remove_data_docstring(self):
        assert_(self.results.remove_data.__doc__ is not None)

    def test_pickle_wrapper(self):

        from statsmodels.iolib.smpickle import save_pickle, load_pickle

        from statsmodels.compatnp.py3k import BytesIO

        fh = BytesIO()  #use cPickle with binary content

        #test unwrapped results load save pickle
        self.results._results.save(fh)
        fh.seek(0,0)
        res_unpickled = self.results._results.__class__.load(fh)
        assert_(type(res_unpickled) is type(self.results._results))

        #test wrapped results load save
        fh.seek(0,0)
        #save_pickle(self.results, fh)
        self.results.save(fh)
        fh.seek(0,0)
        #res_unpickled = load_pickle(fh)
        res_unpickled = self.results.__class__.load(fh)
        fh.close()
        #print type(res_unpickled)
        assert_(type(res_unpickled) is type(self.results))

        before = sorted(self.results.__dict__.keys())
        after = sorted(res_unpickled.__dict__.keys())
        assert_(before == after, msg='not equal %r and %r' % (before, after))

        before = sorted(self.results._results.__dict__.keys())
        after = sorted(res_unpickled._results.__dict__.keys())
        assert_(before == after, msg='not equal %r and %r' % (before, after))

        before = sorted(self.results.model.__dict__.keys())
        after = sorted(res_unpickled.model.__dict__.keys())
        assert_(before == after, msg='not equal %r and %r' % (before, after))

        before = sorted(self.results._cache.keys())
        after = sorted(res_unpickled._cache.keys())
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
        y_count = np.random.poisson(np.exp(x.sum(1)-x.mean()))
        model = sm.Poisson(y_count, x)#, exposure=np.ones(nobs), offset=np.zeros(nobs)) #bug with default
        #use start_params to converge faster
        start_params = np.array([ 0.75334818,  0.99425553,  1.00494724,  1.00247112])
        self.results = model.fit(start_params=start_params, method='bfgs',
                                 disp=0)

        #TODO: temporary, fixed in master
        self.predict_kwds = dict(exposure=1, offset=0)

class TestRemoveDataPickleLogit(RemoveDataPickle):

    def setup(self):
        #fit for each test, because results will be changed by test
        x = self.exog
        nobs = x.shape[0]
        np.random.seed(987689)
        y_bin = (np.random.rand(nobs) < 1./(1+np.exp(x.sum(1)-x.mean()))).astype(int)
        model = sm.Logit(y_bin, x)#, exposure=np.ones(nobs), offset=np.zeros(nobs)) #bug with default
        #use start_params to converge faster
        start_params = np.array([-0.73403806, -1.00901514, -0.97754543, -0.95648212])
        self.results = model.fit(start_params=start_params, method='bfgs',
                disp=0)

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
                TestRemoveDataPickleLogit, TestRemoveDataPickleRLM,
                TestRemoveDataPickleGLM]:
        print cls
        cls.setup_class()
        tt = cls()
        tt.setup()
        tt.test_remove_data_pickle()
        tt.test_remove_data_docstring()
        tt.test_pickle_wrapper()
