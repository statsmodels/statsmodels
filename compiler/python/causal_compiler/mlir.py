"""MLIR IR emitter for causal inference operations."""

import json

from .adapter import ContrastInputs, InterventionInputs


def emit_treatment_effect_mlir(inputs: ContrastInputs) -> str:
    """Emit a generic MLIR module containing ``ci.treatment_effect``.

    The data matrix and coefficient vector are passed as function arguments so
    large datasets are never embedded as IR constants.

    Parameters
    ----------
    inputs : ContrastInputs
        Extracted model parameters, indices, and data shapes from the
        ``statsmodels`` adapter.

    Returns
    -------
    str
        MLIR module text defining ``@counterfactual``.
    """
    row_count, interaction_count = inputs.interaction_data.shape
    coefficient_count = inputs.coefficients.size
    interaction_columns = ", ".join(
        json.dumps(col) for col in inputs.interaction_columns
    )
    coefficient_indices = ", ".join(str(idx) for idx in inputs.interaction_indices)

    return f"""module {{
  func.func public @counterfactual(
      %data: tensor<{row_count}x{interaction_count}xf64>,
      %coefficients: tensor<{coefficient_count}xf64>
  ) -> tensor<{row_count}xf64> {{
    %effects = "ci.treatment_effect"(%data, %coefficients) {{
      control_value = {inputs.control_value:.17e} : f64,
      interaction_columns = [{interaction_columns}],
      interaction_coefficients = array<i64: {coefficient_indices}>,
      main_effect_index = {inputs.main_effect_index} : i64,
      treated_value = {inputs.treated_value:.17e} : f64,
      treatment_name = {json.dumps(inputs.treatment_var)}
    }} : (
      tensor<{row_count}x{interaction_count}xf64>,
      tensor<{coefficient_count}xf64>
    ) -> tensor<{row_count}xf64>
    return %effects : tensor<{row_count}xf64>
  }}
}}"""


def emit_predict_sequence_mlir(inputs: InterventionInputs) -> str:
    """Emit K separate ``ci.predict`` ops (pre-fusion IR)."""
    n_rows, n_covs = inputs.covariate_data.shape
    n_coeff = inputs.coefficients.size
    n_doses = len(inputs.doses)
    base_idx = ", ".join(str(i) for i in inputs.base_indices)
    int_idx = ", ".join(str(i) for i in inputs.interaction_indices)
    t_name = json.dumps(inputs.treatment_var)
    link_str = json.dumps(inputs.link)

    ops = []
    for k, dose in enumerate(inputs.doses):
        ops.append(
            f'    %p{k} = "ci.predict"(%data, %coefficients) {{\n'
            f"      treatment_name = {t_name},\n"
            f"      link = {link_str},\n"
            f"      dose = {dose:.17e} : f64,\n"
            f"      base_coefficients = array<i64: {base_idx}>,\n"
            f"      interaction_coefficients = array<i64: {int_idx}>,\n"
            f"      intercept_index = {inputs.intercept_index} : i64,\n"
            f"      treatment_index = {inputs.treatment_index} : i64\n"
            f"    }} : (tensor<{n_rows}x{n_covs}xf64>, tensor<{n_coeff}xf64>) -> tensor<{n_rows}xf64>"
        )
    ret_types = ", ".join(f"tensor<{n_rows}xf64>" for _ in range(n_doses))
    ret_vals = ", ".join(f"%p{k}" for k in range(n_doses))
    body = "\n".join(ops)
    return f"""module {{
  func.func public @counterfactual_unfused(
      %data: tensor<{n_rows}x{n_covs}xf64>,
      %coefficients: tensor<{n_coeff}xf64>
  ) -> ({ret_types}) {{
{body}
    return {ret_vals} : {ret_types}
  }}
}}"""


def emit_intervention_predict_mlir(inputs: InterventionInputs) -> str:
    """Emit a fused ``ci.intervention_predict`` op returning ``tensor<NxKxf64>``."""
    n_rows, n_covs = inputs.covariate_data.shape
    n_coeff = inputs.coefficients.size
    n_doses = len(inputs.doses)
    base_idx = ", ".join(str(i) for i in inputs.base_indices)
    int_idx = ", ".join(str(i) for i in inputs.interaction_indices)
    doses_str = ", ".join(f"{d:.17e}" for d in inputs.doses)

    return f"""module {{
  func.func public @counterfactual_k(
      %data: tensor<{n_rows}x{n_covs}xf64>,
      %coefficients: tensor<{n_coeff}xf64>
  ) -> tensor<{n_rows}x{n_doses}xf64> {{
    %preds = "ci.intervention_predict"(%data, %coefficients) {{
      treatment_name = {json.dumps(inputs.treatment_var)},
      link = {json.dumps(inputs.link)},
      doses = array<f64: {doses_str}>,
      base_coefficients = array<i64: {base_idx}>,
      interaction_coefficients = array<i64: {int_idx}>,
      intercept_index = {inputs.intercept_index} : i64,
      treatment_index = {inputs.treatment_index} : i64
    }} : (
      tensor<{n_rows}x{n_covs}xf64>,
      tensor<{n_coeff}xf64>
    ) -> tensor<{n_rows}x{n_doses}xf64>
    return %preds : tensor<{n_rows}x{n_doses}xf64>
  }}
}}"""
