import numpy as np
import pandas as pd
import pytest

from statsmodels.formula.api import ols
from causal_compiler import CompilerApi


def test_formula_ols_counterfactual_interaction():
    training = pd.DataFrame(
        {
            "A": [0, 1, 0, 1, 0, 1],
            "x": [0, 0, 1, 1, 2, 2],
        }
    )
    training["y"] = (
        2 + 3 * training["A"] + 4 * training["x"] + 5 * training["A"] * training["x"]
    )
    results = ols("y ~ A * x", data=training).fit()
    scoring_data = pd.DataFrame(
        {"A": [0, 1, 0], "x": [-1, 2, 4]}, index=["a", "b", "c"]
    )

    workflow = CompilerApi().compile_workflow(results, "A", [0, 1], scoring_data)
    output = workflow.run()

    expected_effects = pd.Series(
        3 + 5 * scoring_data["x"],
        index=scoring_data.index,
        name="treatment_effect",
    )
    np.testing.assert_allclose(output.row_effects, expected_effects)
    assert output.ate == pytest.approx(expected_effects.mean())

    control_data = scoring_data.assign(A=0)
    treated_data = scoring_data.assign(A=1)
    reference = results.predict(treated_data) - results.predict(control_data)
    np.testing.assert_allclose(output.row_effects, reference)

    interaction_data, coefficients = workflow.mlir_inputs()
    np.testing.assert_allclose(interaction_data, scoring_data[["x"]])
    np.testing.assert_allclose(coefficients, results.params)

    mlir = workflow.emit_mlir()
    assert '"ci.treatment_effect"' in mlir
    assert "tensor<3x1xf64>" in mlir
    assert 'interaction_columns = ["x"]' in mlir
    assert 'interaction_coefficients = array<i64: 3>' in mlir


def test_compile_workflow_requires_formula_ols_results():
    data = pd.DataFrame({"y": [1.0, 2.0], "A": [0, 1]})
    unfitted_model = ols("y ~ A", data=data)

    with pytest.raises(TypeError, match="fitted OLS results"):
        CompilerApi().compile_workflow(unfitted_model, "A", (0, 1), data)


def test_compile_workflow_requires_treatment_in_formula():
    data = pd.DataFrame({"y": [1.0, 2.0, 4.0], "x": [0.0, 1.0, 2.0]})
    results = ols("y ~ x", data=data).fit()
    scoring_data = data.assign(A=[0, 1, 0])

    with pytest.raises(ValueError, match="main effect"):
        CompilerApi().compile_workflow(results, "A", (0, 1), scoring_data)


def test_compile_workflow_requires_distinct_numeric_treatment_values():
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 4.0, 3.0],
            "A": [0, 1, 0, 1],
            "x": [0.0, 0.0, 1.0, 1.0],
        }
    )
    results = ols("y ~ A * x", data=data).fit()

    with pytest.raises(ValueError, match="distinct finite numeric"):
        CompilerApi().compile_workflow(results, "A", (1, 1), data)
