import os
import shlex

import numpy as np
import pytest

from causal_compiler.synthetic import (
    _compile_iree_cpu_kernel,
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
    assert measurements["reference_full_design_matvecs"] == 2
    assert measurements["numpy_direct_interaction_matvecs"] == 1
    assert (
        measurements["direct_estimated_flops"]
        < measurements["reference_estimated_flops"]
    )
    assert measurements["reference_median_seconds"] >= 0
    assert measurements["direct_median_seconds"] >= 0
    assert measurements["reference_design_matrix_materialization_mib"] > 0


def test_synthetic_benchmark_checks_optional_compiled_output():
    workflow = make_synthetic_workflow(n_rows=32, n_covariates=2, seed=12)
    compiled = lambda: workflow.run_direct().row_effects.to_numpy()

    measurements = run_benchmark(
        workflow, warmup=0, repeats=1, compiled_operation=compiled
    )

    assert measurements["compiled_median_seconds"] >= 0
    assert measurements["compiled_speedup_vs_reference"] >= 0


@pytest.mark.skipif(
    not os.environ.get("CI_OPT_COMMAND"),
    reason="set CI_OPT_COMMAND to run the optional WSL/IREE integration test",
)
def test_synthetic_compiled_mlir_matches_statsmodels():
    workflow = make_synthetic_workflow(n_rows=128, n_covariates=3, seed=13)
    command = shlex.split(os.environ["CI_OPT_COMMAND"])
    compiled = _compile_iree_cpu_kernel(workflow, command, "host")

    measurements = run_benchmark(
        workflow, warmup=0, repeats=1, compiled_operation=compiled
    )

    assert measurements["compiled_linalg_generic_ops"] == 1


@pytest.mark.parametrize(
    "n_rows,n_covariates",
    [(1, 1), (10, 0)],
)
def test_synthetic_workload_requires_positive_dimensions(n_rows, n_covariates):
    with pytest.raises(ValueError):
        make_synthetic_workflow(n_rows=n_rows, n_covariates=n_covariates)
