"""Python front end for the causal compiler prototype."""

from collections.abc import Sequence
from dataclasses import dataclass
import json
from numbers import Real
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd

from statsmodels.regression.linear_model import OLS


@dataclass
class CounterfactualResults:
    """Results from evaluating a treatment contrast on a dataset.

    Attributes
    ----------
    row_effects : pandas.Series
        Predicted treated-minus-control contrast for each input row.
    ate : float
        Mean of `row_effects` over the input dataset.

    Notes
    -----
    This is an average of model-predicted contrasts. Its interpretation as a
    causal average treatment effect depends on assumptions beyond this
    calculation.
    """

    row_effects: pd.Series
    ate: float


class Workflow:
    """A counterfactual prediction workflow.

    ``run`` uses Statsmodels prediction as a correctness reference. The MLIR
    methods emit and lower the supported interaction contrast but do not yet
    execute the lowered module.
    """

    def __init__(
        self,
        results,
        treatment_var: str,
        treatment_values: tuple[Real, Real] | list[Real],
        data: pd.DataFrame,
        interaction_data: np.ndarray,
        coefficients: np.ndarray,
        main_effect_index: int,
        interaction_columns: tuple[str, ...],
        interaction_coefficients: tuple[int, ...],
    ):
        self._results = results
        self._treatment_var = treatment_var
        self._treatment_values = treatment_values
        self._data = data.copy()
        self._interaction_data = interaction_data
        self._coefficients = coefficients
        self._main_effect_index = main_effect_index
        self._interaction_columns = interaction_columns
        self._interaction_coefficients = interaction_coefficients

    def run(self) -> CounterfactualResults:
        """Evaluate row-level contrasts and their mean.

        Returns
        -------
        CounterfactualResults
            Predicted treated-minus-control contrast for each row and its
            arithmetic mean.
        """
        control_value, treated_value = self._treatment_values
        control_data = self._data.assign(**{self._treatment_var: control_value})
        treated_data = self._data.assign(**{self._treatment_var: treated_value})

        control_prediction = np.asarray(self._results.predict(control_data))
        treated_prediction = np.asarray(self._results.predict(treated_data))
        row_effects = pd.Series(
            treated_prediction - control_prediction,
            index=self._data.index,
            name="treatment_effect",
        )
        return CounterfactualResults(
            row_effects=row_effects, ate=float(row_effects.mean())
        )

    def run_direct(self) -> CounterfactualResults:
        """Evaluate the contrast directly from fitted interaction terms.

        This avoids constructing two counterfactual formula design matrices.
        The emitted MLIR represents the same calculation, but this method
        currently executes it with NumPy rather than compiled code.

        Returns
        -------
        CounterfactualResults
            Predicted treated-minus-control contrast for each input row and
            its arithmetic mean.
        """
        control_value, treated_value = self._treatment_values
        interaction_coefficients = self._coefficients[
            list(self._interaction_coefficients)
        ]
        treatment_slopes = (
            self._coefficients[self._main_effect_index]
            + self._interaction_data @ interaction_coefficients
        )
        row_effects = pd.Series(
            (treated_value - control_value) * treatment_slopes,
            index=self._data.index,
            name="treatment_effect",
        )
        return CounterfactualResults(
            row_effects=row_effects, ate=float(row_effects.mean())
        )

    def emit_mlir(self) -> str:
        """Emit a module invoking the causal treatment-effect operation.

        Returns
        -------
        str
            Generic MLIR containing ``ci.treatment_effect``. The matrix and
            coefficient vector are function arguments so the module does not
            embed potentially large dataset values as IR constants.
        """
        row_count = self._interaction_data.shape[0]
        interaction_count = self._interaction_data.shape[1]
        coefficient_count = self._coefficients.size
        control_value, treated_value = self._treatment_values
        interaction_columns = ", ".join(
            json.dumps(column) for column in self._interaction_columns
        )
        coefficient_indices = ", ".join(
            str(index) for index in self._interaction_coefficients
        )
        return f"""module {{
  func.func public @counterfactual(
      %data: tensor<{row_count}x{interaction_count}xf64>,
      %coefficients: tensor<{coefficient_count}xf64>
  ) -> tensor<{row_count}xf64> {{
    %effects = "ci.treatment_effect"(%data, %coefficients) {{
      control_value = {control_value:.17e} : f64,
      interaction_columns = [{interaction_columns}],
      interaction_coefficients = array<i64: {coefficient_indices}>,
      main_effect_index = {self._main_effect_index} : i64,
      treated_value = {treated_value:.17e} : f64,
      treatment_name = {json.dumps(self._treatment_var)}
    }} : (
      tensor<{row_count}x{interaction_count}xf64>,
      tensor<{coefficient_count}xf64>
    ) -> tensor<{row_count}xf64>
    return %effects : tensor<{row_count}xf64>
  }}
}}"""

    def mlir_inputs(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the feature matrix and coefficients expected by emitted IR.

        Returns
        -------
        tuple of ndarray
            Interaction feature values with shape ``(nobs, n_interactions)``
            and the fitted OLS coefficients in design-column order.
        """
        return self._interaction_data.copy(), self._coefficients.copy()

    def compile(self, ci_opt_path: str | Path | Sequence[str]) -> str:
        """Lower the emitted module with the project ``ci-opt`` executable.

        Parameters
        ----------
        ci_opt_path : str, pathlib.Path, or sequence of str
            Path to the built ``ci-opt`` executable, or a command sequence
            such as ``["wsl", "-d", "Ubuntu", "--", "/path/to/ci-opt"]``.

        Returns
        -------
        str
            MLIR after lowering ``ci.treatment_effect``.

        Raises
        ------
        ValueError
            If ``ci_opt_path`` is an empty command.
        subprocess.CalledProcessError
            If ``ci-opt`` rejects or fails to lower the emitted module.
        """
        command = (
            [str(ci_opt_path)]
            if isinstance(ci_opt_path, (str, Path))
            else list(ci_opt_path)
        )
        if not command or not command[0]:
            raise ValueError("ci_opt_path must specify a command")
        completed = subprocess.run(
            [*command, "--ci-lower-treatment-effect"],
            input=self.emit_mlir(),
            capture_output=True,
            check=True,
            text=True,
        )
        return completed.stdout


class CompilerApi:
    """Prototype front end for creating counterfactual workflows.

    The current implementation supports fitted formula-based OLS results and
    in-memory pandas DataFrames. The MLIR path is a prototype and does not
    execute the lowered module.
    """

    def compile_workflow(
        self,
        results,
        treatment_var: str,
        treatment_values: tuple[Real, Real] | list[Real],
        data: pd.DataFrame,
    ) -> Workflow:
        """Create a workflow for a fitted formula-based OLS result.

        Parameters
        ----------
        results : RegressionResultsWrapper or OLSResults
            Fitted OLS results returned by ``fit()``.
        treatment_var : str
            Name of the treatment column in ``data`` and the fitted formula.
        treatment_values : tuple of float or list of float
            Control and treated values, in that order.
        data : pandas.DataFrame
            Rows on which to evaluate the counterfactual predictions.

        Returns
        -------
        Workflow
            Reference workflow that produces row-level contrasts and their
            mean when run.

        Notes
        -----
        The mean returned by this workflow is an average of model-predicted
        contrasts. Its interpretation as a causal average treatment effect
        depends on assumptions beyond this calculation.

        The compiler prototype supports raw numeric treatment columns and
        direct two-factor interactions with raw numeric columns. Categorical
        terms and transformed treatment or interaction factors are not
        supported.

        Raises
        ------
        TypeError
            If the fitted results or scoring data have an unsupported type.
        ValueError
            If the results are not fitted formula-based OLS, the treatment
            column is missing, or invalid treatment values are provided.
        """
        model = getattr(results, "model", None)
        if not isinstance(model, OLS):
            raise TypeError("results must be fitted OLS results")
        if not hasattr(results, "params"):
            raise ValueError("results must contain fitted parameters")
        if not hasattr(model, "formula"):
            raise ValueError("results must come from a formula-based OLS model")
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not isinstance(treatment_var, str) or not treatment_var:
            raise TypeError("treatment_var must be a non-empty column name")
        if treatment_var not in data.columns:
            raise ValueError(f"treatment column {treatment_var!r} is missing from data")
        if not pd.api.types.is_numeric_dtype(data[treatment_var]):
            raise TypeError("treatment column must contain numeric values")
        if (
            not isinstance(treatment_values, (tuple, list))
            or len(treatment_values) != 2
            or not all(
                isinstance(value, Real) and np.isfinite(value)
                for value in treatment_values
            )
            or treatment_values[0] == treatment_values[1]
        ):
            raise ValueError(
                "treatment_values must be a pair of distinct finite numeric values"
            )
        if data.empty:
            raise ValueError("data must contain at least one row")

        names = list(model.exog_names)
        if treatment_var not in names:
            raise ValueError(
                f"model must include a main effect for {treatment_var!r}"
            )
        interaction_columns = []
        interaction_coefficients = []
        for index, name in enumerate(names):
            factors = name.split(":")
            if name == treatment_var:
                continue
            treatment_factors = [
                factor
                for factor in factors
                if re.search(rf"\b{re.escape(treatment_var)}\b", factor)
            ]
            if not treatment_factors:
                continue
            if treatment_var not in factors:
                raise ValueError(
                    "transformed treatment terms are not supported"
                )
            if len(factors) != 2 or factors.count(treatment_var) != 1:
                raise ValueError(
                    "only two-factor interactions with one treatment factor "
                    "are supported"
                )
            interaction_column = (
                factors[1] if factors[0] == treatment_var else factors[0]
            )
            if interaction_column not in data.columns:
                raise ValueError(
                    "interaction factors must be raw numeric data columns"
                )
            if not pd.api.types.is_numeric_dtype(data[interaction_column]):
                raise TypeError("interaction columns must contain numeric values")
            interaction_columns.append(interaction_column)
            interaction_coefficients.append(index)
        if not interaction_columns:
            raise ValueError(
                "the model must include at least one treatment interaction"
            )

        coefficients = np.asarray(results.params, dtype=float)
        if coefficients.ndim != 1 or coefficients.size != len(names):
            raise ValueError("fitted parameters do not match model design columns")
        if not np.isfinite(coefficients).all():
            raise ValueError("fitted coefficients must all be finite")
        try:
            interaction_data = data[interaction_columns].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError("interaction columns must contain numeric values") from exc
        if not np.isfinite(interaction_data).all():
            raise ValueError("interaction data must contain only finite values")

        return Workflow(
            results,
            treatment_var,
            tuple(treatment_values),
            data,
            interaction_data,
            coefficients,
            names.index(treatment_var),
            tuple(interaction_columns),
            tuple(interaction_coefficients),
        )
