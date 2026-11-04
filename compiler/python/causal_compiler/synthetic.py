"""Synthetic-data benchmark for post-fit OLS treatment contrasts."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import shlex
import time

import numpy as np
import pandas as pd

from statsmodels.formula.api import ols

from .api import CompilerApi, Workflow


def make_synthetic_workflow(
    n_rows: int = 100_000,
    n_covariates: int = 8,
    seed: int = 12345,
) -> Workflow:
    """Create a reproducible OLS workflow with treatment interactions.

    Parameters
    ----------
    n_rows : int, optional
        Number of observations to generate.
    n_covariates : int, optional
        Number of continuous covariates, each interacted with treatment.
    seed : int, optional
        Seed for the local NumPy random number generator.

    Returns
    -------
    Workflow
        A fitted formula-OLS workflow over the generated observations.
    """
    if n_rows < 2:
        raise ValueError("n_rows must be at least 2")
    if n_covariates < 1:
        raise ValueError("n_covariates must be at least 1")

    rng = np.random.default_rng(seed)
    treatment = rng.integers(0, 2, size=n_rows)
    covariate_names = [f"x{index}" for index in range(n_covariates)]
    covariates = rng.normal(size=(n_rows, n_covariates))
    main_effect = 0.75
    baseline_effects = np.linspace(0.2, 1.0, n_covariates)
    interaction_effects = np.linspace(0.1, 0.6, n_covariates)
    outcome = (
        1.5
        + main_effect * treatment
        + covariates @ baseline_effects
        + treatment * (covariates @ interaction_effects)
        + rng.normal(scale=0.1, size=n_rows)
    )

    data = pd.DataFrame(covariates, columns=covariate_names)
    data["A"] = treatment
    data["y"] = outcome
    formula = f"y ~ A * ({' + '.join(covariate_names)})"
    results = ols(formula, data=data).fit()
    return CompilerApi().compile_workflow(results, "A", (0, 1), data)


def _median_latency(
    operation: Callable[[], object], warmup: int, repeats: int
) -> float:
    for _ in range(warmup):
        operation()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples))


def _compile_iree_cpu_kernel(
    workflow: Workflow,
    ci_opt_command: list[str],
    target_cpu: str,
) -> Callable[[], np.ndarray]:
    try:
        from iree.compiler import compile_str
        import iree.runtime as ireert
    except ImportError as exc:
        raise ImportError(
            "compiled CPU benchmarking requires the optional "
            "'iree-compiler' and 'iree-runtime' packages"
        ) from exc

    lowered_mlir = workflow.compile(ci_opt_command)
    compiled_module = compile_str(
        lowered_mlir,
        target_backends=["llvm-cpu"],
        extra_args=[
            "--iree-input-demote-f64-to-f32=false",
            "--iree-vm-target-extension-f64",
            "--iree-vm-target-truncate-unsupported-floats=false",
            f"--iree-llvmcpu-target-cpu={target_cpu}",
        ],
    )
    context = ireert.SystemContext(config=ireert.Config("local-task"))
    vm_module = ireert.VmModule.from_flatbuffer(
        context.instance, compiled_module, warn_if_copy=False
    )
    context.add_vm_module(vm_module)
    entrypoint = context.modules.module.counterfactual
    interaction_data, coefficients = workflow.mlir_inputs()

    def run() -> np.ndarray:
        return np.asarray(entrypoint(interaction_data, coefficients))

    return run


def run_benchmark(
    workflow: Workflow,
    warmup: int = 2,
    repeats: int = 7,
    compiled_operation: Callable[[], np.ndarray] | None = None,
) -> dict[str, float | int]:
    """Check equivalence and benchmark reference and direct NumPy paths.

    If ``compiled_operation`` is provided, also validate and time the lowered
    MLIR through IREE's CPU runtime.
    """
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    reference = workflow.run()
    direct = workflow.run_direct()
    np.testing.assert_allclose(
        direct.row_effects.to_numpy(),
        reference.row_effects.to_numpy(),
        rtol=1e-10,
        atol=1e-10,
    )
    if compiled_operation is not None:
        np.testing.assert_allclose(
            compiled_operation(),
            reference.row_effects.to_numpy(),
            rtol=1e-10,
            atol=1e-10,
        )

    interaction_data, coefficients = workflow.mlir_inputs()
    n_rows, n_interactions = interaction_data.shape
    n_parameters = coefficients.size
    reference_flops = 2 * n_rows * (2 * n_parameters - 1) + n_rows
    direct_flops = n_rows * (2 * n_interactions + 1)
    reference_seconds = _median_latency(workflow.run, warmup, repeats)
    direct_seconds = _median_latency(workflow.run_direct, warmup, repeats)
    measurements = {
        "rows": n_rows,
        "design_columns": n_parameters,
        "interactions": n_interactions,
        "reference_full_design_matvecs": 2,
        "numpy_direct_interaction_matvecs": 1,
        "reference_estimated_flops": reference_flops,
        "direct_estimated_flops": direct_flops,
        "reference_median_seconds": reference_seconds,
        "direct_median_seconds": direct_seconds,
        "direct_speedup_vs_reference": reference_seconds / direct_seconds,
        "reference_design_matrix_materialization_mib": (
            2 * n_rows * n_parameters * np.dtype(np.float64).itemsize / 2**20
        ),
        "direct_interaction_matrix_mib": (
            n_rows * n_interactions * np.dtype(np.float64).itemsize / 2**20
        ),
    }
    if compiled_operation is not None:
        compiled_seconds = _median_latency(
            compiled_operation, warmup, repeats
        )
        measurements["compiled_linalg_generic_ops"] = 1
        measurements["compiled_median_seconds"] = compiled_seconds
        measurements["compiled_speedup_vs_reference"] = (
            reference_seconds / compiled_seconds
        )
        measurements["compiled_speedup_vs_direct_numpy"] = (
            direct_seconds / compiled_seconds
        )
    return measurements


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare formula-based counterfactual prediction with a direct "
            "NumPy treatment contrast and optional IREE CPU execution on "
            "synthetic OLS data."
        )
    )
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--covariates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--ci-opt-command",
        help=(
            "optional ci-opt command, e.g. "
            "'wsl -d Ubuntu -- /path/to/ci-opt'"
        ),
    )
    parser.add_argument("--target-cpu", default="host")
    args = parser.parse_args()

    workflow = make_synthetic_workflow(
        n_rows=args.rows,
        n_covariates=args.covariates,
        seed=args.seed,
    )
    compiled_operation = None
    if args.ci_opt_command:
        ci_opt_command = shlex.split(args.ci_opt_command)
        if not ci_opt_command:
            parser.error("--ci-opt-command must not be empty")
        compiled_operation = _compile_iree_cpu_kernel(
            workflow, ci_opt_command, args.target_cpu
        )
    measurements = run_benchmark(
        workflow,
        warmup=args.warmup,
        repeats=args.repeats,
        compiled_operation=compiled_operation,
    )
    for name, value in measurements.items():
        if isinstance(value, float):
            print(f"{name}: {value:.6f}")
        else:
            print(f"{name}: {value}")
    print(
        "Formula and direct timings are Statsmodels/NumPy CPU measurements. "
        "An IREE-compiled CPU timing is included only when ci-opt is provided."
    )


if __name__ == "__main__":
    main()
