import os
import shlex

import numpy as np
import pytest

from causal_compiler.synthetic import (
    _benchmark_iree_cpu_module,
    _compile_iree_cpu_module,
    _compile_iree_empty_output_module,
    _load_iree_cpu_kernel,
    make_synthetic_workflow,
    run_benchmark,
)


def test_synthetic_direct_contrast_matches_formula_predictions():
    workflow = make_synthetic_workflow(n_rows=128, n_covariates=3, seed=10)

    reference = workflow.run()
    direct = workflow.run_direct()

    np.testing.assert_allclose(
        direct.row_effects.to_numpy(),
        reference.row_effects.to_numpy(),
        rtol=1e-10,
        atol=1e-10,
    )
    assert direct.ate == pytest.approx(reference.ate)


def test_synthetic_benchmark_reports_workload_and_timing():
    workflow = make_synthetic_workflow(n_rows=64, n_covariates=2, seed=11)

    measurements = run_benchmark(workflow, warmup=0, repeats=1)

    assert measurements["rows"] == 64
    assert measurements["interactions"] == 2
    assert measurements["full_design_numpy_gemv_count"] == 2
    assert measurements["direct_numpy_gemv_count"] == 1
    assert (
        measurements["direct_gemv_estimated_flops"]
        < measurements["full_design_gemv_estimated_flops"]
    )
    assert measurements["full_design_numpy_gemv_median_seconds"] >= 0
    assert measurements["direct_numpy_gemv_median_seconds"] >= 0
    assert (
        measurements["counterfactual_design_matrices_total_materialized_mib"]
        > 0
    )


def test_synthetic_benchmark_checks_optional_compiled_output():
    workflow = make_synthetic_workflow(n_rows=32, n_covariates=2, seed=12)
    compiled = workflow.run_direct().row_effects.to_numpy

    measurements = run_benchmark(
        workflow, warmup=0, repeats=1, compiled_operation=compiled
    )

    assert (
        measurements["iree_python_call_median_seconds_including_runtime"] >= 0
    )


@pytest.mark.skipif(
    not os.environ.get("CI_OPT_COMMAND"),
    reason="set CI_OPT_COMMAND to run the optional WSL/IREE integration test",
)
def test_synthetic_compiled_mlir_matches_statsmodels():
    workflow = make_synthetic_workflow(n_rows=128, n_covariates=3, seed=13)
    command = shlex.split(os.environ["CI_OPT_COMMAND"])
    module = _compile_iree_cpu_module(workflow, command, "host")
    compiled = _load_iree_cpu_kernel(workflow, module)

    np.testing.assert_allclose(
        compiled(),
        workflow.run().row_effects.to_numpy(),
        rtol=1e-10,
        atol=1e-10,
    )
    assert _benchmark_iree_cpu_module(
        workflow, module, repeats=5, device="local-sync"
    ) > 0
    assert _benchmark_iree_cpu_module(
        workflow, module, repeats=5, device="local-task"
    ) > 0
    interaction_data, coefficients = workflow.mlir_inputs()
    empty_module = _compile_iree_empty_output_module(
        interaction_data.shape[0],
        interaction_data.shape[1],
        coefficients.size,
        "host",
    )
    assert (
        _benchmark_iree_cpu_module(workflow, empty_module, repeats=5) > 0
    )


@pytest.mark.parametrize(
    "n_rows,n_covariates",
    [(1, 1), (10, 0)],
)
def test_synthetic_workload_requires_positive_dimensions(n_rows, n_covariates):
    with pytest.raises(ValueError):
        make_synthetic_workflow(n_rows=n_rows, n_covariates=n_covariates)
