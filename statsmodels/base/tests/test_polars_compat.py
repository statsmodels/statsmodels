"""
Tests for Polars DataFrame compatibility with statsmodels.

Tests conversion of Polars DataFrames/Series to pandas for use with
statsmodels models and the formula interface.
"""

import numpy as np
import pandas as pd
import pytest

from statsmodels.discrete.discrete_model import Logit
import statsmodels.formula.api as smf
from statsmodels.genmod.api import GLM
from statsmodels.regression.linear_model import OLS

# Skip entire module if polars is not installed
pl = pytest.importorskip("polars")


class TestPolarsDirectConstructor:
    """Test passing Polars DataFrame/Series directly to model constructors."""

    @classmethod
    def setup_class(cls):
        """Create test data in both pandas and Polars formats."""
        cls.n = 50
        np.random.seed(42)

        # Create pandas versions
        cls.pandas_y = np.random.randn(cls.n)
        cls.pandas_X = np.column_stack(
            [np.ones(cls.n), np.random.randn(cls.n), np.random.randn(cls.n)]
        )

        # Create Polars versions
        cls.polars_y_series = pl.Series(name="y", values=cls.pandas_y)
        cls.polars_df = pl.DataFrame(
            {
                "y": cls.pandas_y,
                "x1": cls.pandas_X[:, 1],
                "x2": cls.pandas_X[:, 2],
            }
        )

    def test_ols_polars_series_and_polars_df(self):
        """Test OLS with Polars Series endog and Polars DataFrame exog."""
        # Extract exog from Polars DataFrame
        polars_exog = self.polars_df[["x1", "x2"]].with_columns(
            pl.lit(1).alias("const")
        ).select(["const", "x1", "x2"])

        # Fit models
        result_polars = OLS(self.polars_y_series, polars_exog).fit()

        # Compare with pandas reference
        pandas_y = pd.Series(self.pandas_y, name="y")
        pandas_X = self.pandas_X
        result_pandas = OLS(pandas_y, pandas_X).fit()

        # Assert parameters are close
        np.testing.assert_allclose(
            result_polars.params.values, result_pandas.params.values, rtol=1e-10
        )

    def test_ols_polars_df_select_columns(self):
        """Test OLS by selecting columns from Polars DataFrame."""
        # Add a constant column for regression
        polars_data = self.polars_df.with_columns(
            pl.lit(1).alias("const")
        ).select(["const", "y", "x1", "x2"])

        result_polars = OLS(
            polars_data["y"], polars_data[["const", "x1", "x2"]]
        ).fit()

        pandas_y = pd.Series(self.pandas_y)
        result_pandas = OLS(pandas_y, self.pandas_X).fit()

        np.testing.assert_allclose(
            result_polars.params.values, result_pandas.params.values, rtol=1e-10
        )

    def test_glm_polars(self):
        """Test GLM with Polars data."""
        # Create binary response for GLM
        y_binary = (self.pandas_y > 0).astype(int)
        polars_y_binary = pl.Series("y_binary", values=y_binary)
        polars_exog = self.polars_df[["x1", "x2"]]

        result_polars = GLM(polars_y_binary, polars_exog).fit(disp=False)

        pandas_y_binary = pd.Series(y_binary)
        result_pandas = GLM(pandas_y_binary, self.pandas_X[:, 1:]).fit(disp=False)

        np.testing.assert_allclose(
            result_polars.params.values, result_pandas.params.values, rtol=1e-5
        )

    def test_logit_polars(self):
        """Test Logit model with Polars data."""
        y_binary = (self.pandas_y > 0).astype(int)
        polars_y_binary = pl.Series(values=y_binary)
        polars_exog = self.polars_df[["x1", "x2"]]

        result_polars = Logit(polars_y_binary, polars_exog).fit(disp=False)

        pandas_y_binary = pd.Series(y_binary)
        result_pandas = Logit(pandas_y_binary, self.pandas_X[:, 1:]).fit(disp=False)

        np.testing.assert_allclose(
            result_polars.params.values, result_pandas.params.values, rtol=1e-5
        )

    def test_mixed_polars_pandas(self):
        """Test mixing Polars and pandas inputs."""
        polars_y = self.polars_y_series
        pandas_X = pd.DataFrame(self.pandas_X[:, 1:], columns=["x1", "x2"])
        pandas_X.insert(0, "const", 1)

        result_mixed = OLS(polars_y, pandas_X).fit()

        pandas_y = pd.Series(self.pandas_y)
        result_pandas = OLS(pandas_y, pandas_X).fit()

        np.testing.assert_allclose(
            result_mixed.params.values, result_pandas.params.values, rtol=1e-10
        )


class TestPolarsFormulaAPI:
    """Test using Polars DataFrames with the formula interface."""

    @classmethod
    def setup_class(cls):
        """Create test data."""
        cls.n = 50
        np.random.seed(123)
        cls.y = np.random.randn(cls.n)
        cls.x1 = np.random.randn(cls.n)
        cls.x2 = np.random.randn(cls.n)

        cls.pandas_df = pd.DataFrame({"y": cls.y, "x1": cls.x1, "x2": cls.x2})
        cls.polars_df = pl.DataFrame(
            {"y": cls.y, "x1": cls.x1, "x2": cls.x2}
        )

    def test_ols_formula_polars(self):
        """Test OLS.from_formula with Polars DataFrame."""
        result_polars = smf.ols("y ~ x1 + x2", data=self.polars_df).fit()
        result_pandas = smf.ols("y ~ x1 + x2", data=self.pandas_df).fit()

        np.testing.assert_allclose(
            result_polars.params.values, result_pandas.params.values, rtol=1e-10
        )

    def test_glm_formula_polars(self):
        """Test GLM.from_formula with Polars DataFrame."""
        result_polars = smf.glm(
            "y ~ x1 + x2", data=self.polars_df
        ).fit(disp=False)
        result_pandas = smf.glm(
            "y ~ x1 + x2", data=self.pandas_df
        ).fit(disp=False)

        np.testing.assert_allclose(
            result_polars.params.values, result_pandas.params.values, rtol=1e-5
        )

    def test_formula_with_multiple_terms(self):
        """Test formula with interaction and polynomial terms."""
        result_polars = smf.ols("y ~ x1 + x2 + x1:x2", data=self.polars_df).fit()
        result_pandas = smf.ols("y ~ x1 + x2 + x1:x2", data=self.pandas_df).fit()

        np.testing.assert_allclose(
            result_polars.params.values, result_pandas.params.values, rtol=1e-10
        )


class TestPolarsOutputMetadata:
    """Test that output metadata (column names, index) is preserved correctly."""

    @classmethod
    def setup_class(cls):
        """Create test data with named columns."""
        cls.n = 30
        np.random.seed(456)
        cls.y = np.random.randn(cls.n)
        cls.x1 = np.random.randn(cls.n)
        cls.x2 = np.random.randn(cls.n)

        cls.pandas_df = pd.DataFrame({"y": cls.y, "x1": cls.x1, "x2": cls.x2})
        cls.polars_df = pl.DataFrame(
            {"y": cls.y, "x1": cls.x1, "x2": cls.x2}
        )

    def test_formula_result_exog_names(self):
        """Test that exog column names are preserved in formula results."""
        result_polars = smf.ols("y ~ x1 + x2", data=self.polars_df).fit()
        result_pandas = smf.ols("y ~ x1 + x2", data=self.pandas_df).fit()

        assert list(result_polars.model.exog_names) == list(
            result_pandas.model.exog_names
        )

    def test_formula_result_endog_name(self):
        """Test that endog name is preserved in formula results."""
        result_polars = smf.ols("y ~ x1 + x2", data=self.polars_df).fit()
        result_pandas = smf.ols("y ~ x1 + x2", data=self.pandas_df).fit()

        assert result_polars.model.endog_names == result_pandas.model.endog_names


class TestPolarsPredict:
    """Test predictions with Polars DataFrames."""

    @classmethod
    def setup_class(cls):
        """Create train and test data."""
        cls.n_train = 40
        cls.n_test = 10
        np.random.seed(789)

        cls.y_train = np.random.randn(cls.n_train)
        cls.x1_train = np.random.randn(cls.n_train)
        cls.x2_train = np.random.randn(cls.n_train)

        cls.x1_test = np.random.randn(cls.n_test)
        cls.x2_test = np.random.randn(cls.n_test)

        # Pandas versions
        cls.pandas_df_train = pd.DataFrame(
            {"y": cls.y_train, "x1": cls.x1_train, "x2": cls.x2_train}
        )
        cls.pandas_df_test = pd.DataFrame(
            {"x1": cls.x1_test, "x2": cls.x2_test}
        )

        # Polars versions
        cls.polars_df_train = pl.DataFrame(
            {"y": cls.y_train, "x1": cls.x1_train, "x2": cls.x2_train}
        )
        cls.polars_df_test = pl.DataFrame(
            {"x1": cls.x1_test, "x2": cls.x2_test}
        )

    def test_predict_with_polars_exog(self):
        """Test predict method with Polars DataFrame exog."""
        result_polars = smf.ols("y ~ x1 + x2", data=self.polars_df_train).fit()
        pred_polars = result_polars.predict(self.polars_df_test)

        result_pandas = smf.ols("y ~ x1 + x2", data=self.pandas_df_train).fit()
        pred_pandas = result_pandas.predict(self.pandas_df_test)

        np.testing.assert_allclose(pred_polars.values, pred_pandas.values, rtol=1e-10)

    def test_predict_returns_pandas(self):
        """Test that predict returns pandas Series regardless of input type."""
        result_polars = smf.ols("y ~ x1 + x2", data=self.polars_df_train).fit()
        pred = result_polars.predict(self.polars_df_test)

        assert isinstance(pred, pd.Series)


class TestPolarsEdgeCases:
    """Test edge cases and special scenarios."""

    def test_polars_with_null_values(self):
        """Test Polars DataFrame with null values and missing='drop'."""
        np.random.seed(321)
        n = 30
        y = np.random.randn(n)
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)

        # Create Polars DataFrame with some nulls
        df_polars = pl.DataFrame({"y": y, "x1": x1, "x2": x2})
        # Insert some null values
        df_polars = df_polars.with_columns([
            pl.when(pl.int_range(0, pl.len()) % 5 == 0).then(None).otherwise(pl.col("y")).alias("y_with_nulls")
        ])
        df_polars = df_polars.drop("y").rename({"y_with_nulls": "y"})

        # This should work with missing='drop'
        # Note: formula API defaults to missing='drop' so this should work without explicit parameter
        try:
            result = smf.ols("y ~ x1 + x2", data=df_polars).fit()
            # Verify we have fewer observations due to dropped NaNs
            assert result.nobs < n
        except Exception as e:
            pytest.fail(f"Failed to handle null values: {e}")

    def test_polars_single_variable(self):
        """Test with Polars Series as both endog and simple exog."""
        np.random.seed(654)
        n = 25
        y = np.random.randn(n)
        x = np.random.randn(n)

        poly_y = pl.Series("y", values=y)
        poly_X = pl.DataFrame({"x": x, "const": [1.0] * n})[["const", "x"]]

        result_polars = OLS(poly_y, poly_X).fit()

        pandas_y = pd.Series(y)
        pandas_X = pd.DataFrame({"const": [1.0] * n, "x": x})
        result_pandas = OLS(pandas_y, pandas_X).fit()

        np.testing.assert_allclose(
            result_polars.params.values, result_pandas.params.values, rtol=1e-10
        )
