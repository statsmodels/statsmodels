import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from causal_compiler import emit_treatment_effect_mlir, extract_ols_contrast


def test_emit_treatment_effect_mlir_structure():
    rng = np.random.default_rng(7)
    n = 12
    df = pd.DataFrame(
        {
            "T": rng.uniform(0.0, 1.0, size=n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "x3": rng.normal(size=n),
        }
    )
    df["y"] = 1.0 + 0.5 * df["T"] + 0.2 * df["T"] * df["x1"] - 0.3 * df["T"] * df["x2"]
    results = smf.ols("y ~ T * (x1 + x2) + x3", data=df).fit()

    inputs = extract_ols_contrast(results, "T", (0.0, 1.5), df.iloc[:5])
    mlir = emit_treatment_effect_mlir(inputs)

    assert "func.func public @counterfactual(" in mlir
    assert "%data: tensor<5x2xf64>" in mlir
    assert f"%coefficients: tensor<{len(results.params)}xf64>" in mlir
    assert "-> tensor<5xf64>" in mlir
    assert '"ci.treatment_effect"(%data, %coefficients)' in mlir
    assert 'interaction_columns = ["x1", "x2"]' in mlir
    assert f"main_effect_index = {inputs.main_effect_index} : i64" in mlir
    assert (
        f"interaction_coefficients = array<i64: {inputs.interaction_indices[0]}, {inputs.interaction_indices[1]}>"
        in mlir
    )
    assert 'treatment_name = "T"' in mlir
    assert "arith.constant dense" not in mlir
