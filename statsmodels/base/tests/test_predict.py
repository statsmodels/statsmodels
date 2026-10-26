"""
Tests for Results.predict
"""

from statsmodels.compat.pandas import testing as pdt

from io import BytesIO

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest
from scipy.special import expit

from statsmodels.discrete.discrete_model import NegativeBinomial
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.regression.linear_model import OLS


@pytest.mark.parametrize("model_class", [NegativeBinomial, OrderedModel])
@pytest.mark.parametrize("nobs", [1, 3])
def test_predict_series_extra_params(model_class, nobs):
    rng = np.random.default_rng(76452)
    x = pd.Series(rng.normal(size=200), name="x")
    if model_class is NegativeBinomial:
        y = rng.negative_binomial(2, 2 / (2 + np.exp(0.4 * x)))
        model = model_class(y, x)
    else:
        y = np.digitize(0.4 * x + rng.logistic(size=200), [-0.5, 0.8])
        model = model_class(y, x, distr="logit")
    res = model.fit(method="bfgs", disp=False)
    assert res.mle_retvals["converged"]

    exog = pd.Series(
        [-0.4, 0.1, 0.7], index=pd.Index([13, 8, 2], name="case"), name="x"
    ).iloc[:nobs]
    params = np.asarray(res.params)
    xb = exog.to_numpy() * params[0]
    if model_class is NegativeBinomial:
        # The NB2 conditional mean is exp(x beta), independent of dispersion.
        expected = np.exp(xb)
    else:
        # Ordered logit probabilities are differences of logistic CDFs at
        # the cutpoints minus x beta; the second cutpoint is an increment.
        cutpoints = [-np.inf, params[1], params[1] + np.exp(params[2]), np.inf]
        expected = np.diff(expit(np.asarray(cutpoints) - xb[:, None]), axis=1)

    pred_frame = res.predict(exog.to_frame())
    pdt.assert_index_equal(pred_frame.index, exog.index)
    assert_allclose(pred_frame, expected, rtol=1e-13)
    pred = res.predict(exog)
    pdt.assert_index_equal(pred.index, exog.index)
    assert_allclose(pred, expected, rtol=1e-13)


@pytest.mark.parametrize("formula", ["y ~ x", "y ~ x - 1"])
@pytest.mark.parametrize("nobs", [1, 3])
@pytest.mark.parametrize("persist", [False, True])
def test_formula_series_after_remove_data(formula, nobs, persist):
    data = pd.DataFrame(
        {"x": [-2., -1., 0., 1., 2., 3.], "y": [-3., -0.8, 1.1, 2.7, 5.2, 6.8]}
    )
    result = OLS.from_formula(formula, data).fit()
    values = pd.Series(
        [-0.5, 0.2, 1.3], index=pd.Index([9, 4, 2], name="case"), name="x"
    ).iloc[:nobs]
    expected = np.asarray(
        result.params["x"] * values + result.params.get("Intercept", 0)
    )
    if persist:
        stream = BytesIO()
        result.save(stream, remove_data=True)
        stream.seek(0)
        result = type(result).load(stream)
    else:
        result.remove_data()
    assert result.model.exog is None
    prediction = result.predict(values)
    pdt.assert_index_equal(prediction.index, values.index)
    assert_allclose(prediction, expected, rtol=1e-13)


class CheckPredictReturns:

    def test_2d(self):
        res = self.res
        data = self.data

        fitted = res.fittedvalues.iloc[1:10:2]

        pred = res.predict(data.iloc[1:10:2])
        pdt.assert_index_equal(pred.index, fitted.index)
        assert_allclose(pred.values, fitted.values, rtol=1e-13)

        # plain dict
        xd = dict(zip(data.columns, data.iloc[1:10:2].values.T, strict=True))
        with pytest.warns(DeprecationWarning, match="Using"):
            pred = res.predict(xd)
        assert_equal(pred.index, np.arange(len(pred)))
        assert_allclose(pred.values, fitted.values, rtol=1e-13)

    def test_1d(self):
        # one observation
        res = self.res
        data = self.data

        pred = res.predict(data.iloc[:1])
        pdt.assert_index_equal(pred.index, data.iloc[:1].index)
        fv = np.asarray(res.fittedvalues)
        assert_allclose(pred.values, fv[0], rtol=1e-13)

        fittedm = res.fittedvalues.mean()
        xmean = data.mean()
        pred = res.predict(xmean.to_frame().T)
        assert_equal(pred.index, np.arange(1))
        assert_allclose(pred, fittedm, rtol=1e-13)

        # Series
        pred = res.predict(data.mean())
        assert_equal(pred.index, np.arange(1))
        assert_allclose(pred.values, fittedm, rtol=1e-13)

        # dict with scalar value (is plain dict)
        # Note: this warns about dropped nan, even though there are None -FIXED
        with pytest.warns(DeprecationWarning, match="Using"):
            pred = res.predict(data.mean().to_dict())
        assert_equal(pred.index, np.arange(1))
        assert_allclose(pred.values, fittedm, rtol=1e-13)

    def test_without_formula(self):
        res = self.res
        data = self.data
        fitted = res.fittedvalues.iloc[1:10:2]

        # plain numpy array
        pred = res.predict(res.model.exog[1:10:2], transform=False)
        assert_allclose(pred, fitted.values, rtol=1e-13)

        # pandas DataFrame
        x = pd.DataFrame(
            res.model.exog[1:10:2],
            index=data.index[1:10:2],
            columns=res.model.exog_names,
        )
        pred = res.predict(x)
        pdt.assert_index_equal(pred.index, fitted.index)
        assert_allclose(pred.values, fitted.values, rtol=1e-13)

        # one observation - 1-D
        pred = res.predict(res.model.exog[1], transform=False)
        assert_allclose(pred, fitted.values[0], rtol=1e-13)

        # one observation - pd.Series
        pred = res.predict(x.iloc[0])
        pdt.assert_index_equal(pred.index, fitted.index[:1])
        assert_allclose(pred.values[0], fitted.values[0], rtol=1e-13)


class TestPredictOLS(CheckPredictReturns):

    @classmethod
    def setup_class(cls):
        nobs = 30
        rs = np.random.RandomState(987128)
        x = rs.randn(nobs, 3)
        y = x.sum(1) + rs.randn(nobs)
        index = [f"obs{i:02d}" for i in range(nobs)]
        # add one extra column to check that it does not matter
        cls.data = pd.DataFrame(
            np.round(np.column_stack((y, x)), 4),
            columns="y var1 var2 var3".split(),
            index=index,
        )

        cls.res = OLS.from_formula("y ~ var1 + var2", data=cls.data).fit()


class TestPredictGLM(CheckPredictReturns):

    @classmethod
    def setup_class(cls):
        nobs = 30
        rs = np.random.RandomState(987128)
        x = rs.randn(nobs, 3)
        y = x.sum(1) + rs.randn(nobs)
        index = [f"obs{i:02d}" for i in range(nobs)]
        # add one extra column to check that it does not matter
        cls.data = pd.DataFrame(
            np.round(np.column_stack((y, x)), 4),
            columns="y var1 var2 var3".split(),
            index=index,
        )

        cls.res = GLM.from_formula("y ~ var1 + var2", data=cls.data).fit()

    def test_predict_offset(self):
        res = self.res
        data = self.data

        fitted = res.fittedvalues.iloc[1:10:2]
        offset = np.arange(len(fitted))
        fitted = fitted + offset

        pred = res.predict(data.iloc[1:10:2], offset=offset)
        pdt.assert_index_equal(pred.index, fitted.index)
        assert_allclose(pred.values, fitted.values, rtol=1e-13)

        # plain dict
        xd = dict(zip(data.columns, data.iloc[1:10:2].values.T, strict=True))
        with pytest.warns(DeprecationWarning, match="Using"):
            pred = res.predict(xd, offset=offset)
        assert_equal(pred.index, np.arange(len(pred)))
        assert_allclose(pred.values, fitted.values, rtol=1e-13)

        # offset as pandas.Series
        data2 = data.iloc[1:10:2].copy()
        data2["offset"] = offset
        pred = res.predict(data2, offset=data2["offset"])
        pdt.assert_index_equal(pred.index, fitted.index)
        assert_allclose(pred.values, fitted.values, rtol=1e-13)

        # check nan in exog is ok, preserves index matching offset length
        data2 = data.iloc[1:10:2].copy()
        data2["offset"] = offset
        data2.iloc[0, 1] = np.nan
        pred = res.predict(data2, offset=data2["offset"])
        pdt.assert_index_equal(pred.index, fitted.index)
        fitted_nan = fitted.copy()
        fitted_nan.iloc[0] = np.nan
        assert_allclose(pred.values, fitted_nan.values, rtol=1e-13)
