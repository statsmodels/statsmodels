"""

Created on Fri Mar 09 16:00:27 2012

Author: Josef Perktold
"""

from statsmodels.compat.pandas import assert_series_equal

from io import BytesIO
import pickle

import numpy as np
import pandas as pd
import pytest

import statsmodels.api as sm

# log used in TestPickleFormula5
log = np.log


def check_pickle(obj):
    fh = BytesIO()
    pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    plen = fh.tell()
    fh.seek(0, 0)
    res = pickle.load(fh)
    fh.close()
    return res, plen


class RemoveDataPickle:

    @classmethod
    def setup_class(cls):
        nobs = 1000
        rs = np.random.RandomState(987689)
        x = rs.randn(nobs, 3)
        x = sm.add_constant(x)
        cls.exog = x
        cls.xf = 0.25 * np.ones((2, 4))
        cls.predict_kwds = {}
        cls.reduction_factor = 0.1

    def setup_method(self):
        self.model_class = sm.OLS
        self.formula = None
        self.model_args = ()
        self.model_kwargs = {}
        self.fit_kwargs = {}
        self.constrained = False
        self.regularized = False

    def estimate_model(self):
        if self.formula:
            mod = self.model_class.from_formula(self.formula, data=self.data)
        else:
            mod = self.model_class(*self.model_args, **self.model_kwargs)
        if self.constrained:
            results = mod.fit_constrained(*self.constrained_args)
        elif self.regularized:
            results = mod.fit_regularized(method="l1", disp=0, alpha=10)
        else:
            results = mod.fit(**self.fit_kwargs)
        return results

    @pytest.mark.thread_unsafe
    def test_remove_data_pickle(self):
        results = self.estimate_model()
        xf = self.xf
        pred_kwds = self.predict_kwds
        pred1 = results.predict(xf, **pred_kwds)
        # create some cached attributes
        results.summary()
        results.summary2()  # SMOKE test also summary2

        # uncomment the following to check whether tests run (7 failures now)
        # np.testing.assert_equal(res, 1)

        # check pickle unpickle works on full results
        # TODO: drop of load save is tested
        res, orig_nbytes = check_pickle(results._results)

        # remove data arrays, check predict still works
        results.remove_data()

        pred2 = results.predict(xf, **pred_kwds)

        if isinstance(pred1, pd.Series) and isinstance(pred2, pd.Series):
            assert_series_equal(pred1, pred2)
        elif isinstance(pred1, pd.DataFrame) and isinstance(pred2, pd.DataFrame):
            assert pred1.equals(pred2)
        else:
            np.testing.assert_equal(pred2, pred1)

        # pickle and unpickle reduced array
        res, nbytes = check_pickle(results._results)

        # for testing attach res
        msg = "pickle length not %d < %d" % (nbytes, orig_nbytes)
        assert nbytes < orig_nbytes * self.reduction_factor, msg
        pred3 = results.predict(xf, **pred_kwds)

        if isinstance(pred1, pd.Series) and isinstance(pred3, pd.Series):
            assert_series_equal(pred1, pred3)
        elif isinstance(pred1, pd.DataFrame) and isinstance(pred3, pd.DataFrame):
            assert pred1.equals(pred3)
        else:
            np.testing.assert_equal(pred3, pred1)

    @pytest.mark.thread_unsafe
    def test_remove_data_docstring(self):
        assert self.results.remove_data.__doc__ is not None

    @pytest.mark.thread_unsafe
    def test_pickle_wrapper(self):

        fh = BytesIO()  # use pickle with binary content

        # test unwrapped results load save pickle
        self.results._results.save(fh)
        fh.seek(0, 0)
        res_unpickled = self.results._results.__class__.load(fh)
        assert type(res_unpickled) is type(self.results._results)

        # test wrapped results load save
        fh.seek(0, 0)
        self.results.save(fh)
        fh.seek(0, 0)
        res_unpickled = self.results.__class__.load(fh)
        fh.close()
        assert type(res_unpickled) is type(self.results)

        before = sorted(self.results.__dict__.keys())
        after = sorted(res_unpickled.__dict__.keys())
        assert before == after, f"not equal {before!r} and {after!r}"

        before = sorted(self.results._results.__dict__.keys())
        after = sorted(res_unpickled._results.__dict__.keys())
        assert before == after, f"not equal {before!r} and {after!r}"

        before = sorted(self.results.model.__dict__.keys())
        after = sorted(res_unpickled.model.__dict__.keys())
        assert before == after, f"not equal {before!r} and {after!r}"

        before = sorted(self.results._cache.keys())
        after = sorted(res_unpickled._cache.keys())
        assert before == after, f"not equal {before!r} and {after!r}"


class TestRemoveDataPickleOLS(RemoveDataPickle):

    def setup_method(self):
        super().setup_method()  # fit for each test, because results will be changed by test
        x = self.exog
        rs = np.random.RandomState(987689)
        y = x.sum(axis=1) + rs.randn(x.shape[0])
        self.results = sm.OLS(y, self.exog).fit()

        self.model_class = sm.OLS
        self.model_args = (y, self.exog)


class TestRemoveDataPickleWLS(RemoveDataPickle):

    def setup_method(self):
        super().setup_method()
        # fit for each test, because results will be changed by test
        x = self.exog
        rs = np.random.RandomState(987689)
        y = x.sum(axis=1) + rs.randn(x.shape[0])
        self.results = sm.WLS(y, self.exog, weights=np.ones(len(y))).fit()

        self.model_class = sm.WLS
        self.model_args = (y, self.exog)
        self.model_kwargs = dict(weights=np.ones(len(y)))


class TestRemoveDataPicklePoisson(RemoveDataPickle):

    def setup_method(self):
        super().setup_method()
        # fit for each test, because results will be changed by test
        x = self.exog
        rs = np.random.RandomState(987689)
        y_count = rs.poisson(np.exp(x.sum(1) - x.mean()))

        # bug with default
        model = sm.Poisson(y_count, x)

        # use start_params to converge faster
        start_params = np.array([0.75334818, 0.99425553, 1.00494724, 1.00247112])
        self.results = model.fit(start_params=start_params, method="bfgs", disp=0)

        # TODO: temporary, fixed in main
        self.predict_kwds = dict(exposure=1, offset=0)

        self.model_class = sm.Poisson
        self.model_args = (y_count, x)
        self.fit_kwargs = dict(start_params=start_params, method="bfgs", disp=0)


class TestRemoveDataPickleNegativeBinomial(RemoveDataPickle):

    def setup_method(self):
        super().setup_method()
        # fit for each test, because results will be changed by test
        data = sm.datasets.randhie.load()
        mod = sm.NegativeBinomial(data.endog, data.exog)
        self.results = mod.fit(disp=0)
        self.model_class = sm.NegativeBinomial
        self.model_args = (data.endog, data.exog)
        self.fit_kwargs = dict(disp=0)


class TestRemoveDataPickleLogit(RemoveDataPickle):

    def setup_method(self):
        super().setup_method()
        # fit for each test, because results will be changed by test
        x = self.exog
        nobs = x.shape[0]
        rs = np.random.RandomState(987689)
        y_bin = (rs.rand(nobs) < 1.0 / (1 + np.exp(x.sum(1) - x.mean()))).astype(int)

        # bug with default
        model = sm.Logit(y_bin, x)

        # use start_params to converge faster
        start_params = np.array([-0.73403806, -1.00901514, -0.97754543, -0.95648212])
        self.results = model.fit(start_params=start_params, method="bfgs", disp=0)
        self.model_class = sm.Logit
        self.model_args = (y_bin, x)
        self.fit_kwargs = dict(start_params=start_params, method="bfgs", disp=0)


class TestRemoveDataPickleRLM(RemoveDataPickle):

    def setup_method(self):
        super().setup_method()
        # fit for each test, because results will be changed by test
        x = self.exog
        rs = np.random.RandomState(987689)
        y = x.sum(axis=1) + rs.randn(x.shape[0])
        self.results = sm.RLM(y, self.exog).fit()
        self.model_class = sm.RLM
        self.model_args = (y, self.exog)


class TestRemoveDataPickleGLM(RemoveDataPickle):

    def setup_method(self):
        super().setup_method()
        # fit for each test, because results will be changed by test
        x = self.exog
        rs = np.random.RandomState(987689)
        y = x.sum(axis=1) + rs.randn(x.shape[0])
        self.results = sm.GLM(y, self.exog).fit()
        self.model_class = sm.GLM
        self.model_args = (y, self.exog)
        self.constrained = False

    def test_cached_data_removed(self):
        res = self.estimate_model()
        # fill data-like members of the cache
        names = ["resid_response", "resid_deviance", "resid_pearson", "resid_anscombe"]
        for name in names:
            getattr(res, name)
        # check that the attributes are present before calling remove_data
        for name in names:
            assert name in res._cache
            assert res._cache[name] is not None
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            # FutureWarning for BIC changes
            res.remove_data()

        for name in names:
            assert res._cache[name] is None

    def test_cached_values_evaluated(self):
        # check that value-like attributes are evaluated before data
        # is removed
        res = self.estimate_model()
        assert res._cache == {}
        res.remove_data()
        assert "aic" in res._cache


class TestRemoveDataPickleGLMConstrained(RemoveDataPickle):

    def setup_method(self):
        # fit for each test, because results will be changed by test
        super().setup_method()
        x = self.exog
        rs = np.random.RandomState(987689)
        y = x.sum(axis=1) + rs.randn(x.shape[0])
        self.results = sm.GLM(y, self.exog).fit_constrained("x1=x2")
        self.model_class = sm.GLM
        self.model_args = (y, self.exog)
        self.constrained = True
        self.constrained_args = ("x1=x2",)


class TestPickleFormula(RemoveDataPickle):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        nobs = 10000
        rs = np.random.RandomState(987689)
        x = rs.randn(nobs, 3)
        cls.exog = pd.DataFrame(x, columns=["A", "B", "C"])
        cls.xf = pd.DataFrame(0.25 * np.ones((2, 3)), columns=cls.exog.columns)
        cls.reduction_factor = 0.5

    def setup_method(self):
        super().setup_method()
        x = self.exog
        rs = np.random.RandomState(123)
        y = x.sum(axis=1) + rs.randn(x.shape[0])
        y = pd.Series(y, name="Y")
        X = self.exog.copy()
        X["Y"] = y
        self.results = sm.OLS.from_formula("Y ~ A + B + C", data=X).fit()
        self.model_class = sm.OLS
        self.formula = "Y ~ A + B + C"
        self.data = X


class TestPickleFormula2(RemoveDataPickle):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        nobs = 500
        rs = np.random.RandomState(987689)
        data = rs.randn(nobs, 4)
        data[:, 0] = data[:, 1:].sum(1)
        cls.data = pd.DataFrame(data, columns=["Y", "A", "B", "C"])
        cls.xf = pd.DataFrame(0.25 * np.ones((2, 3)), columns=cls.data.columns[1:])
        cls.reduction_factor = 0.666

    def setup_method(self):
        super().setup_method()
        self.results = sm.OLS.from_formula("Y ~ A + B + C", data=self.data).fit()
        self.model_class = sm.OLS
        self.formula = "Y ~ A + B + C"


class TestPickleFormula3(TestPickleFormula2):

    def setup_method(self):
        super().setup_method()
        self.results = sm.OLS.from_formula("Y ~ A + B * C", data=self.data).fit()
        self.model_class = sm.OLS
        self.formula = "Y ~ A + B * C"


class TestPickleFormula4(TestPickleFormula2):

    def setup_method(self):
        super().setup_method()
        self.results = sm.OLS.from_formula(
            "Y ~ np.log(abs(A) + 1) + B * C", data=self.data
        ).fit()
        self.model_class = sm.OLS
        self.formula = "Y ~ np.log(abs(A) + 1) + B * C"


# we need log in module namespace for TestPickleFormula5


class TestPickleFormula5(TestPickleFormula2):

    def setup_method(self):
        super().setup_method()
        self.results = sm.OLS.from_formula(
            "Y ~ log(abs(A) + 1) + B * C", data=self.data
        ).fit()
        self.model_class = sm.OLS
        self.formula = "Y ~ log(abs(A) + 1) + B * C"
        self.model_args = ()
        self.model_kwargs = {}
        self.fit_kwargs = {}
        self.constrained = False


class TestRemoveDataPicklePoissonRegularized(RemoveDataPickle):

    def setup_method(self):
        # fit for each test, because results will be changed by test
        super().setup_method()
        x = self.exog
        rs = np.random.RandomState(987689)
        y_count = rs.poisson(np.exp(x.sum(1) - x.mean()))
        model = sm.Poisson(y_count, x)
        self.results = model.fit_regularized(method="l1", disp=0, alpha=10)
        self.model_class = sm.Poisson
        self.model_args = (y_count, x)
        self.fit_kwargs = dict(method="l1", disp=0, alpha=10)
        self.regularized = True
