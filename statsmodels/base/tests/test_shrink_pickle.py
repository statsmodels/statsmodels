"""

Created on Fri Mar 09 16:00:27 2012

Author: Josef Perktold
"""

from statsmodels.compat.pandas import assert_series_equal

import copy
from io import BytesIO
import pickle

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLMResults
from statsmodels.tools._decorators import cache_readonly

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


def remove_data_none(results):
    # remove_data as in statsmodels 0.14 and 0.15.0, which stored None in the
    # cache for every cached attribute of the class and for _data_in_cache
    results.remove_data()
    for name, attr in vars(type(results)).items():
        if isinstance(attr, cache_readonly):
            results._cache[name] = None
    for key in results._data_in_cache:
        results._cache[key] = None


def n_none(cache):
    return sum(val is None for val in cache.values())


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
        msg = f"pickle length not {nbytes:d} < {orig_nbytes:d}"
        assert nbytes < orig_nbytes * self.reduction_factor, msg
        pred3 = results.predict(xf, **pred_kwds)

        if isinstance(pred1, pd.Series) and isinstance(pred3, pd.Series):
            assert_series_equal(pred1, pred3)
        elif isinstance(pred1, pd.DataFrame) and isinstance(pred3, pd.DataFrame):
            assert pred1.equals(pred3)
        else:
            np.testing.assert_equal(pred3, pred1)

    @pytest.mark.thread_unsafe
    def test_pickle_none_in_cache(self):
        # Results pickled after remove_data by statsmodels 0.14 or 0.15.0 hold
        # None in the cache for the removed values. Loading drops them, so an
        # old pickle has the same cache and summary as a new one.
        expected = self.estimate_model()._results
        expected.summary()
        expected.remove_data()
        expected, _ = check_pickle(expected)

        results = self.estimate_model()._results
        results.summary()
        remove_data_none(results)
        assert n_none(results._cache) > 0
        res, _ = check_pickle(results)
        assert n_none(res._cache) == 0
        assert sorted(res._cache) == sorted(expected._cache)
        assert str(res.summary().tables[1]) == str(expected.summary().tables[1])

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
        # default maxiter=35 does not converge
        self.results = mod.fit(maxiter=100, disp=0)
        self.model_class = sm.NegativeBinomial
        self.model_args = (data.endog, data.exog)
        self.fit_kwargs = dict(maxiter=100, disp=0)


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
        self.reduction_factor = 0.2


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
            # The cached value is deleted, not stored as None, so the next
            # access computes it again, which fails without the data
            assert name not in res._cache
            with pytest.raises(TypeError):
                getattr(res, name)

    def test_cached_values_evaluated(self):
        # remove_data neither evaluates the value-like attributes nor stores
        # None for them
        res = self.estimate_model()
        assert res._cache == {}
        res.remove_data()
        assert res._cache == {}

        # A value computed before remove_data is deleted from the cache and
        # computed again from what is left (llf keeps its own copy)
        res = self.estimate_model()
        aic = res.aic
        assert res._cache["aic"] is aic
        res.remove_data()
        assert "aic" not in res._cache
        assert_allclose(res.aic, aic, rtol=1e-13)

    def test_old_pickle_statistics(self):
        # An old pickle read llf and aic as None, and summary raised TypeError
        full = self.estimate_model()
        results = self.estimate_model()._results
        results.summary()
        remove_data_none(results)
        res, _ = check_pickle(results)
        for name in ["llf", "aic", "bic", "bse", "pvalues"]:
            assert_allclose(getattr(res, name), getattr(full, name), rtol=1e-13)
        res.summary()

    def test_copy_keeps_original_cache(self):
        # copy.copy passes the original __dict__ to __setstate__, so the
        # original keeps its cache
        results = self.estimate_model()._results
        remove_data_none(results)
        cache = dict(results._cache)
        res = copy.copy(results)
        assert results._cache == cache
        assert n_none(res._cache) == 0


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


def test_pickle_fit_constrained_formula():
    # GH#9018
    rs = np.random.RandomState(987689)
    df = pd.DataFrame(rs.randn(50, 2), columns=["x1", "x2"])
    df["g"] = np.repeat(["a", "b"], 25)
    df["y"] = df["x1"] + df["x2"] + rs.randn(50)
    model = sm.GLM.from_formula("y ~ x1 * x2 + C(g)", data=df)
    results = model.fit_constrained("x1 + x2 = 1")
    res, _ = check_pickle(results)
    assert_series_equal(res.params, results.params)

    # predict still goes through the formula of the original model
    exog = model.exog[:5]
    expected = exog @ results.params.to_numpy()
    assert_allclose(results.predict(df.iloc[:5]).to_numpy(), expected)
    assert_allclose(res.predict(df.iloc[:5]).to_numpy(), expected)


# The attributes that GLMResults marks as data with cached_data
GLM_DATA = ["resid_response", "resid_pearson", "resid_deviance", "mu"]


def glm_data():
    rs = np.random.RandomState(34234)
    exog = sm.add_constant(rs.standard_normal((200, 2)))
    endog = exog.sum(1) + rs.standard_normal(200)
    return endog, exog


def test_remove_data_base_class_data():
    # GEEResults inherits the data attributes of GLMResults. remove_data
    # deletes them and keeps the statistics that the base classes cache
    endog, exog = glm_data()
    groups = np.repeat(np.arange(40), 5)
    res = sm.GEE(endog, exog, groups).fit()._results
    for name in GLM_DATA:
        getattr(res, name)
    tvalues = res.tvalues
    pvalues = res.pvalues
    res.remove_data()
    assert not set(GLM_DATA) & set(res._cache)
    assert res._cache["tvalues"] is tvalues
    assert res._cache["pvalues"] is pvalues
    res2, _ = check_pickle(res)
    assert not set(GLM_DATA) & set(res2._cache)
    assert_allclose(res2.pvalues, pvalues, rtol=1e-13)


class _OverrideResults(GLMResults):
    # resid_pearson is no longer marked as data
    @cache_readonly
    def resid_pearson(self):
        return self._endog - self.mu


class _InheritResults(_OverrideResults):
    pass


def test_remove_data_overridden_base_class_data():
    # The class nearest in the method resolution order decides whether an
    # attribute is data: resid_pearson of _OverrideResults is kept, and
    # resid_response, which it inherits from GLMResults, is deleted
    endog, exog = glm_data()
    res = sm.GLM(endog, exog).fit()._results
    res.__class__ = _InheritResults
    resid_pearson = res.resid_pearson
    assert res.resid_response.shape == endog.shape
    assert "resid_response" in res._cache
    res.remove_data()
    assert res._cache["resid_pearson"] is resid_pearson
    assert "resid_response" not in res._cache
