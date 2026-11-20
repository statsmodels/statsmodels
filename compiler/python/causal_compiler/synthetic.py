"""Synthetic-data benchmark for post-fit OLS treatment contrasts."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd
import patsy

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


def _iree_cpu_codegen_args(
    target_cpu: str,
    loop_vectorization: bool = True,
    slp_vectorization: bool = True,
) -> list[str]:
    extra_args = [
        "--iree-input-demote-f64-to-f32=false",
        "--iree-vm-target-extension-f64",
        "--iree-vm-target-truncate-unsupported-floats=false",
        f"--iree-llvmcpu-target-cpu={target_cpu}",
    ]
    if loop_vectorization:
        extra_args.append("--iree-llvmcpu-loop-vectorization")
    if slp_vectorization:
        extra_args.append("--iree-llvmcpu-slp-vectorization")
    return extra_args


def _compile_iree_cpu_module(
    workflow: Workflow,
    ci_opt_command: list[str],
    target_cpu: str,
    loop_vectorization: bool = True,
    slp_vectorization: bool = True,
) -> bytes:
    try:
        from iree.compiler import compile_str
    except ImportError as exc:
        raise ImportError(
            "compiled CPU benchmarking requires the optional "
            "'iree-compiler' package"
        ) from exc

    lowered_mlir = workflow.compile(ci_opt_command)
    return compile_str(
        lowered_mlir,
        target_backends=["llvm-cpu"],
        extra_args=_iree_cpu_codegen_args(
            target_cpu, loop_vectorization, slp_vectorization
        ),
    )


def _load_iree_cpu_kernel(
    workflow: Workflow, compiled_module: bytes
) -> Callable[[], np.ndarray]:
    import iree.runtime as ireert

    context = ireert.SystemContext(config=ireert.Config("local-task"))
    vm_module = ireert.VmModule.from_flatbuffer(
        context.instance, compiled_module, warn_if_copy=False
    )
    context.add_vm_module(vm_module)
    entrypoint = context.modules.module.counterfactual
    interaction_data, coefficients = workflow.mlir_inputs()

    def run() -> np.ndarray:
        # Copy while the IREE result is alive; np.asarray alone returns a view
        # that does not keep the device buffer alive.
        result = entrypoint(interaction_data, coefficients)
        return np.array(np.asarray(result), copy=True)

    return run


def _benchmark_iree_module(
    compiled_module: bytes,
    interaction_data: np.ndarray,
    coefficients: np.ndarray,
    repeats: int,
    device: str,
) -> float:
    executable = shutil.which("iree-benchmark-module")
    if executable is None:
        raise RuntimeError(
            "iree-benchmark-module is required for compiled compute timings; "
            "install iree-compiler and iree-runtime"
        )

    with tempfile.TemporaryDirectory(prefix="causal-compiler-benchmark-") as directory:
        directory_path = Path(directory)
        module_path = directory_path / "counterfactual.vmfb"
        data_path = directory_path / "interaction_data.npy"
        coefficients_path = directory_path / "coefficients.npy"
        module_path.write_bytes(compiled_module)
        np.save(data_path, interaction_data)
        np.save(coefficients_path, coefficients)

        command = [
            executable,
            f"--module={module_path}",
            "--function=counterfactual",
            f"--input=@{data_path}",
            f"--input=@{coefficients_path}",
            f"--device={device}://",
            "--benchmark_min_time=0.2s",
            "--benchmark_min_warmup_time=0.1",
            f"--benchmark_repetitions={repeats}",
            "--benchmark_report_aggregates_only=true",
            "--benchmark_color=false",
            "--benchmark_time_unit=us",
        ]
        completed = subprocess.run(
            command, capture_output=True, check=False, text=True
        )
        if completed.returncode:
            raise RuntimeError(
                f"iree-benchmark-module failed ({completed.returncode}):\n"
                f"{completed.stderr}"
            )

    match = re.search(
        r"BM_counterfactual/process_time/real_time_median\s+"
        r"([0-9.]+)\s+(ns|us|ms|s)\b",
        completed.stdout,
    )
    if match is None:
        raise RuntimeError(
            "could not parse iree-benchmark-module median from output:\n"
            f"{completed.stdout}"
        )
    value = float(match.group(1))
    scale = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}[match.group(2)]
    return value * scale


def _benchmark_iree_cpu_module(
    workflow: Workflow,
    compiled_module: bytes,
    repeats: int,
    device: str = "local-sync",
) -> float:
    interaction_data, coefficients = workflow.mlir_inputs()
    return _benchmark_iree_module(
        compiled_module, interaction_data, coefficients, repeats, device
    )


def _compile_iree_empty_output_module(
    n_rows: int,
    n_interactions: int,
    n_coefficients: int,
    target_cpu: str,
    loop_vectorization: bool = True,
    slp_vectorization: bool = True,
) -> bytes:
    from iree.compiler import compile_str

    module = f"""module {{
  func.func public @counterfactual(
      %data: tensor<{n_rows}x{n_interactions}xf64>,
      %coefficients: tensor<{n_coefficients}xf64>
  ) -> tensor<{n_rows}xf64> {{
    %output = tensor.empty() : tensor<{n_rows}xf64>
    return %output : tensor<{n_rows}xf64>
  }}
}}"""
    return compile_str(
        module,
        target_backends=["llvm-cpu"],
        extra_args=_iree_cpu_codegen_args(
            target_cpu, loop_vectorization, slp_vectorization
        ),
    )


def _full_design_matrices(workflow: Workflow) -> tuple[np.ndarray, np.ndarray]:
    model_data = workflow._results.model.data
    model_spec = getattr(model_data, "model_spec", None)
    if model_spec is None:
        model_spec = getattr(model_data, "design_info", None)
    if model_spec is None:
        raise ValueError("fitted formula model does not expose design metadata")
    design_matrices = []
    for treatment_value in workflow._treatment_values:
        scoring_data = workflow._data.assign(
            **{workflow._treatment_var: treatment_value}
        )
        design_matrix = patsy.build_design_matrices(
            [model_spec], scoring_data
        )[0]
        design_matrices.append(np.asarray(design_matrix, dtype=np.float64))
    return design_matrices[0], design_matrices[1]


def run_benchmark(
    workflow: Workflow,
    warmup: int = 2,
    repeats: int = 7,
    compiled_operation: Callable[[], np.ndarray] | None = None,
) -> dict[str, float | int]:
    """Validate outputs and benchmark prediction and NumPy kernel paths.

    If ``compiled_operation`` is provided, validate its results and report its
    Python-call latency separately from the native IREE benchmark.
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
    interaction_coefficients = coefficients[
        list(workflow._interaction_coefficients)
    ]
    main_effect = coefficients[workflow._main_effect_index]
    control_value, treated_value = workflow._treatment_values
    treatment_delta = treated_value - control_value

    control_design, treated_design = _full_design_matrices(workflow)

    def full_design_kernel() -> np.ndarray:
        return treated_design @ coefficients - control_design @ coefficients

    def direct_numpy_kernel() -> np.ndarray:
        effects = interaction_data @ interaction_coefficients
        effects += main_effect
        effects *= treatment_delta
        return effects

    preallocated_effects = np.empty(n_rows, dtype=np.float64)

    def direct_numpy_preallocated_kernel() -> np.ndarray:
        np.matmul(
            interaction_data, interaction_coefficients, out=preallocated_effects
        )
        np.add(preallocated_effects, main_effect, out=preallocated_effects)
        np.multiply(
            preallocated_effects, treatment_delta, out=preallocated_effects
        )
        return preallocated_effects

    np.testing.assert_allclose(
        full_design_kernel(),
        reference.row_effects.to_numpy(),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        direct_numpy_kernel(),
        reference.row_effects.to_numpy(),
        rtol=1e-10,
        atol=1e-10,
    )

    reference_flops = 2 * n_rows * (2 * n_parameters - 1) + n_rows
    direct_flops = n_rows * (2 * n_interactions + 1)
    formula_path_seconds = _median_latency(workflow.run, warmup, repeats)
    full_design_kernel_seconds = _median_latency(
        full_design_kernel, warmup, repeats
    )
    direct_kernel_seconds = _median_latency(
        direct_numpy_kernel, warmup, repeats
    )
    direct_preallocated_seconds = _median_latency(
        direct_numpy_preallocated_kernel, warmup, repeats
    )
    measurements = {
        "rows": n_rows,
        "design_columns": n_parameters,
        "interactions": n_interactions,
        "formula_path_median_seconds_including_pandas_and_design_build": (
            formula_path_seconds
        ),
        "full_design_numpy_gemv_count": 2,
        "full_design_numpy_gemv_median_seconds": full_design_kernel_seconds,
        "direct_numpy_gemv_count": 1,
        "direct_numpy_gemv_median_seconds": direct_kernel_seconds,
        "direct_numpy_preallocated_median_seconds": direct_preallocated_seconds,
        "direct_numpy_speedup_vs_formula_path": (
            formula_path_seconds / direct_kernel_seconds
        ),
        "full_design_gemv_estimated_flops": reference_flops,
        "direct_gemv_estimated_flops": direct_flops,
        "counterfactual_design_matrices_total_materialized_mib": (
            2 * n_rows * n_parameters * np.dtype(np.float64).itemsize / 2**20
        ),
        "direct_interaction_input_matrix_mib": (
            n_rows * n_interactions * np.dtype(np.float64).itemsize / 2**20
        ),
    }
    if compiled_operation is not None:
        compiled_seconds = _median_latency(
            compiled_operation, warmup, repeats
        )
        measurements[
            "iree_python_call_median_seconds_including_runtime"
        ] = compiled_seconds
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
    parser.add_argument(
        "--target-cpu",
        default="host",
        help="LLVM CPU target, such as 'host' or AMD Zen 4's 'znver4'",
    )
    parser.add_argument(
        "--loop-vectorization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable LLVM loop vectorization for IREE CPU codegen",
    )
    parser.add_argument(
        "--slp-vectorization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable LLVM SLP vectorization for IREE CPU codegen",
    )
    args = parser.parse_args()

    workflow = make_synthetic_workflow(
        n_rows=args.rows,
        n_covariates=args.covariates,
        seed=args.seed,
    )
    compiled_operation = None
    compiled_module = None
    if args.ci_opt_command:
        ci_opt_command = shlex.split(args.ci_opt_command)
        if not ci_opt_command:
            parser.error("--ci-opt-command must not be empty")
        compiled_module = _compile_iree_cpu_module(
            workflow,
            ci_opt_command,
            args.target_cpu,
            args.loop_vectorization,
            args.slp_vectorization,
        )
        compiled_operation = _load_iree_cpu_kernel(workflow, compiled_module)
    measurements = run_benchmark(
        workflow,
        warmup=args.warmup,
        repeats=args.repeats,
        compiled_operation=compiled_operation,
    )
    if compiled_module is not None:
        print(
            "IREE CPU codegen: "
            f"target={args.target_cpu}, "
            f"loop_vectorization={args.loop_vectorization}, "
            f"slp_vectorization={args.slp_vectorization}"
        )
        interaction_data, coefficients = workflow.mlir_inputs()
        empty_module = _compile_iree_empty_output_module(
            interaction_data.shape[0],
            interaction_data.shape[1],
            coefficients.size,
            args.target_cpu,
            args.loop_vectorization,
            args.slp_vectorization,
        )
        for device in ("local-sync", "local-task"):
            compiled_seconds = _benchmark_iree_cpu_module(
                workflow,
                compiled_module,
                repeats=max(args.repeats, 5),
                device=device,
            )
            device_prefix = f"iree_{device}"
            measurements[f"{device_prefix}_median_seconds"] = compiled_seconds
            measurements[f"{device_prefix}_speedup_vs_full_design_gemv"] = (
                measurements["full_design_numpy_gemv_median_seconds"]
                / compiled_seconds
            )
            measurements[f"{device_prefix}_speedup_vs_direct_numpy"] = (
                measurements["direct_numpy_gemv_median_seconds"]
                / compiled_seconds
            )
            measurements[f"{device_prefix}_empty_output_floor_seconds"] = (
                _benchmark_iree_module(
                    empty_module,
                    interaction_data,
                    coefficients,
                    repeats=max(args.repeats, 5),
                    device=device,
                )
            )
    for name, value in measurements.items():
        if isinstance(value, float):
            print(f"{name}: {value:.6f}")
        else:
            print(f"{name}: {value}")
    print(
        "Formula path is end-to-end; NumPy timings are kernel-only with "
        "prebuilt inputs. IREE native timings exclude compilation and are "
        "included only when --ci-opt-command is provided."
    )


if __name__ == "__main__":
    main()
