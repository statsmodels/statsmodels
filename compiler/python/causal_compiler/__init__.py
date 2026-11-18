"""Python interface for the causal compiler prototype."""

from .adapter import (
    ContrastInputs,
    InterventionInputs,
    evaluate_intervention_predict,
    evaluate_ols_contrast,
    extract_intervention_inputs,
    extract_ols_contrast,
)
from .api import CompilerApi, CounterfactualResults, Workflow
from .compiler import compile_iree, lower_mlir, run_iree
from .contrast import intervention_predict, treatment_effect
from .mlir import (
    emit_intervention_predict_mlir,
    emit_predict_sequence_mlir,
    emit_treatment_effect_mlir,
)

__all__ = [
    "CompilerApi",
    "ContrastInputs",
    "CounterfactualResults",
    "InterventionInputs",
    "Workflow",
    "compile_iree",
    "emit_intervention_predict_mlir",
    "emit_predict_sequence_mlir",
    "emit_treatment_effect_mlir",
    "evaluate_intervention_predict",
    "evaluate_ols_contrast",
    "extract_intervention_inputs",
    "extract_ols_contrast",
    "intervention_predict",
    "lower_mlir",
    "run_iree",
    "treatment_effect",
]
