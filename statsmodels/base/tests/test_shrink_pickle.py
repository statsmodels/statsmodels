# -*- coding: utf-8 -*-
"""

Created on Fri Mar 09 16:00:27 2012

Author: Josef Perktold
"""

import pickle
import numpy as np
import statsmodels.api as sm

from numpy.testing import assert_, assert_almost_equal, assert_equal


def check_pickle(obj):
    import StringIO
    fh = StringIO.StringIO()
    pickle.dump(obj, fh)
    plen = fh.pos
    fh.seek(0,0)
    res = pickle.load(fh)
    fh.close()
    return res, plen


class RemoveDataPickle(object):

    def __init__(self):
        self.predict_kwds = {}

    @classmethod
    def setupclass(self):

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



class TestRemoveDataPickleOLS(RemoveDataPickle):

    def __init__(self):
        super(self.__class__, self).__init__()
        #fit for each test, because results will be changed by test
        x = self.exog
        np.random.seed(987689)
        y = x.sum(1) + np.random.randn(x.shape[0])
        self.results = sm.OLS(y, self.exog).fit()

class TestRemoveDataPicklePoisson(RemoveDataPickle):

    def __init__(self):
        super(self.__class__, self).__init__()
        #fit for each test, because results will be changed by test
        x = self.exog
        np.random.seed(987689)
        y_count = np.random.poisson(np.exp(x.sum(1)-x.mean()))
        model = sm.Poisson(y_count, x)#, exposure=np.ones(nobs), offset=np.zeros(nobs)) #bug with default
        #use start_params to converge faster
        start_params = np.array([ 0.75334818,  0.99425553,  1.00494724,  1.00247112])
        self.results = model.fit(start_params=start_params, method='bfgs')

        #TODO: temporary, fixed in master
        self.predict_kwds = dict(exposure=1, offset=0)
        #TODO: needs to go into pickle save
        self.results.mle_settings['callback'] = None

class TestRemoveDataPickleLogit(RemoveDataPickle):

    def __init__(self):
        super(self.__class__, self).__init__()
        #fit for each test, because results will be changed by test
        x = self.exog
        nobs = x.shape[0]
        np.random.seed(987689)
        y_bin = (np.random.rand(nobs) < 1./(1+np.exp(x.sum(1)-x.mean()))).astype(int)
        model = sm.Logit(y_bin, x)#, exposure=np.ones(nobs), offset=np.zeros(nobs)) #bug with default
        #use start_params to converge faster
        start_params = np.array([-0.73403806, -1.00901514, -0.97754543, -0.95648212])
        self.results = model.fit(start_params=start_params, method='bfgs')

        #TODO: needs to go into pickle save
        self.results.mle_settings['callback'] = None



if __name__ == '__main__':
    for cls in [TestRemoveDataPickleOLS, TestRemoveDataPicklePoisson,
                TestRemoveDataPickleLogit]:
        cls.setupclass()
        tt = cls()

        tt.test_remove_data_pickle()

    raise


    #print results.predict(xf)
    print results.model.predict(results.params, xf)
    results.summary()

    shrinkit = 1
    if shrinkit:
        results.remove_data()

    import pickle
    fname = 'try_shrink%d_ols.pickle' % shrinkit
    fh = open(fname, 'w')
    pickle.dump(results._results, fh)  #pickling wrapper doesn't work
    fh.close()
    fh = open(fname, 'r')
    results2 = pickle.load(fh)
    fh.close()
    print results2.predict(xf)
    print results2.model.predict(results.params, xf)


    y_count = np.random.poisson(np.exp(x.sum(1)-x.mean()))
    model = sm.Poisson(y_count, x)#, exposure=np.ones(nobs), offset=np.zeros(nobs)) #bug with default
    results = model.fit(method='bfgs')

    results.summary()

    print results.model.predict(results.params, xf, exposure=1, offset=0)

    if shrinkit:
        results.remove_data()
    else:
        #work around pickling bug
        results.mle_settings['callback'] = None

    import pickle
    fname = 'try_shrink%d_poisson.pickle' % shrinkit
    fh = open(fname, 'w')
    pickle.dump(results._results, fh)  #pickling wrapper doesn't work
    fh.close()
    fh = open(fname, 'r')
    results3 = pickle.load(fh)
    fh.close()
    print results3.predict(xf, exposure=1, offset=0)
    print results3.model.predict(results.params, xf, exposure=1, offset=0)

    def check_pickle(obj):
        import StringIO
        fh = StringIO.StringIO()
        pickle.dump(obj, fh)
        plen = fh.pos
        fh.seek(0,0)
        res = pickle.load(fh)
        fh.close()
        return res, plen

    def test_remove_data_pickle(results, xf):
        res, l = check_pickle(results)
        #Note: 10000 is just a guess for the limit on the length of the pickle
        np.testing.assert_(l < 10000, msg='pickle length not %d < %d' % (l, 10000))
        pred1 = results.predict(xf, exposure=1, offset=0)
        pred2 = res.predict(xf, exposure=1, offset=0)
        np.testing.assert_equal(pred2, pred1)

    test_remove_data_pickle(results._results, xf)
