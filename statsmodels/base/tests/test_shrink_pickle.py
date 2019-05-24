# -*- coding: utf-8 -*-
"""

Created on Fri Mar 09 16:00:27 2012

Author: Josef Perktold
"""
from __future__ import print_function
from statsmodels.compat.python import iterkeys, cPickle, BytesIO

import warnings

import numpy as np
from numpy.testing import assert_
import pandas as pd

import statsmodels.api as sm


def check_pickle(obj):
    fh = BytesIO()
    cPickle.dump(obj, fh, protocol=cPickle.HIGHEST_PROTOCOL)
    plen = fh.tell()
    fh.seek(0, 0)
    res = cPickle.load(fh)
    fh.close()
    return res, plen


class RemoveDataPickle(object):

    @classmethod
    def setup_class(cls):
        nobs = 10000
        np.random.seed(987689)
        x = np.random.randn(nobs, 3)
        x = sm.add_constant(x)
        cls.exog = x
        cls.xf = 0.25 * np.ones((2, 4))
        cls.nbytes_max = 20000
        cls.predict_kwds = {}

    def test_remove_data_pickle(self):
        import pandas as pd
        from pandas.util.testing import assert_series_equal

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
        res, _ = check_pickle(results._results)

        #remove data arrays, check predict still works
        with warnings.catch_warnings(record=True) as w:
            results.remove_data()

        pred2 = results.predict(xf, **pred_kwds)

        if isinstance(pred1, pd.Series) and isinstance(pred2, pd.Series):
            assert_series_equal(pred1, pred2)
        elif isinstance(pred1, pd.DataFrame) and isinstance(pred2, pd.DataFrame):
            assert_(pred1.equals(pred2))
        else:
            np.testing.assert_equal(pred2, pred1)

        # pickle and unpickle reduced array
        res, nbytes = check_pickle(results._results)

        #for testing attach res
        self.res = res

        # Note: nbytes_max is just a guess for the limit on the length
        #  of the pickle
        nbytes_max = self.nbytes_max
        assert_(nbytes < nbytes_max,
                msg='pickle length not %d < %d' % (nbytes, nbytes_max))

        pred3 = results.predict(xf, **pred_kwds)

        if isinstance(pred1, pd.Series) and isinstance(pred3, pd.Series):
            assert_series_equal(pred1, pred3)
        elif isinstance(pred1, pd.DataFrame) and isinstance(pred3, pd.DataFrame):
            assert_(pred1.equals(pred3))
        else:
            np.testing.assert_equal(pred3, pred1)

    def test_remove_data_docstring(self):
        assert_(self.results.remove_data.__doc__ is not None)

    def test_pickle_wrapper(self):

        fh = BytesIO()  # use cPickle with binary content

        # test unwrapped results load save pickle
        self.results._results.save(fh)
        fh.seek(0, 0)
        res_unpickled = self.results._results.__class__.load(fh)
        assert type(res_unpickled) is type(self.results._results)  # noqa: E721

        # test wrapped results load save
        fh.seek(0, 0)
        self.results.save(fh)
        fh.seek(0, 0)
        res_unpickled = self.results.__class__.load(fh)
        fh.close()
        assert type(res_unpickled) is type(self.results)  # noqa: E721

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
        data = sm.datasets.randhie.load(as_pandas=False)
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


class TestPickleFormula(RemoveDataPickle):
    @classmethod
    def setup_class(cls):
        super(TestPickleFormula, cls).setup_class()
        nobs = 10000
        np.random.seed(987689)
        x = np.random.randn(nobs, 3)
        cls.exog = pd.DataFrame(x, columns=["A", "B", "C"])
        cls.xf = pd.DataFrame(0.25 * np.ones((2, 3)),
                              columns=cls.exog.columns)
        cls.nbytes_max = 900000  # have to pickle endo/exog to unpickle form.

    def setup(self):
        x = self.exog
        np.random.seed(123)
        y = x.sum(1) + np.random.randn(x.shape[0])
        y = pd.Series(y, name="Y")
        X = self.exog.copy()
        X["Y"] = y
        self.results = sm.OLS.from_formula("Y ~ A + B + C", data=X).fit()


class TestPickleFormula2(RemoveDataPickle):
    @classmethod
    def setup_class(cls):
        super(TestPickleFormula2, cls).setup_class()
        nobs = 500
        np.random.seed(987689)
        data = np.random.randn(nobs, 4)
        data[:,0] = data[:, 1:].sum(1)
        cls.data = pd.DataFrame(data, columns=["Y", "A", "B", "C"])
        cls.xf = pd.DataFrame(0.25 * np.ones((2, 3)),
                              columns=cls.data.columns[1:])
        cls.nbytes_max = 900000  # have to pickle endo/exog to unpickle form.

    def setup(self):
        self.results = sm.OLS.from_formula("Y ~ A + B + C", data=self.data).fit()


class TestPickleFormula3(TestPickleFormula2):

    def setup(self):
        self.results = sm.OLS.from_formula("Y ~ A + B * C", data=self.data).fit()


class TestPickleFormula4(TestPickleFormula2):

    def setup(self):
        self.results = sm.OLS.from_formula("Y ~ np.log(abs(A) + 1) + B * C", data=self.data).fit()


# we need log in module namespace for TestPickleFormula5
from numpy import log  # noqa:F401


class TestPickleFormula5(TestPickleFormula2):

    def setup(self):
        # if we import here, then unpickling fails -> exception in test
        #from numpy import log
        self.results = sm.OLS.from_formula("Y ~ log(abs(A) + 1) + B * C", data=self.data).fit()


class TestRemoveDataPicklePoissonRegularized(RemoveDataPickle):

    def setup(self):
        #fit for each test, because results will be changed by test
        x = self.exog
        np.random.seed(987689)
        y_count = np.random.poisson(np.exp(x.sum(1) - x.mean()))
        model = sm.Poisson(y_count, x)
        self.results = model.fit_regularized(method='l1', disp=0, alpha=10)
