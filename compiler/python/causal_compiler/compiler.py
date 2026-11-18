"""Subprocess and IREE wrappers for lowering and running causal MLIR."""

from pathlib import Path
import subprocess

import numpy as np

from .adapter import ContrastInputs


def lower_mlir(mlir_text: str, ci_opt_path: str | Path = "ci-opt") -> str:
    """Lower ``ci.treatment_effect`` MLIR to Linalg using ``ci-opt``.

    Parameters
    ----------
    mlir_text : str
        MLIR module text containing ``ci.treatment_effect``.
    ci_opt_path : str or pathlib.Path, optional
        Path or executable name for the ``ci-opt`` binary.

    Returns
    -------
    str
        Lowered MLIR module text.

    Raises
    ------
    FileNotFoundError
        If ``ci_opt_path`` cannot be found.
    RuntimeError
        If ``ci-opt`` exits with a non-zero status.
    """
    completed = subprocess.run(
        [str(ci_opt_path), "--ci-lower-treatment-effect"],
        input=mlir_text,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"ci-opt failed with exit code {completed.returncode}:\n"
            f"{completed.stderr}"
        )
    return completed.stdout


def compile_iree(lowered_mlir: str, target_cpu: str = "host") -> bytes:
    """Compile lowered Linalg MLIR to an IREE CPU flatbuffer.

    Parameters
    ----------
    lowered_mlir : str
        Lowered MLIR module text (e.g. from :func:`lower_mlir`).
    target_cpu : str, optional
        Target CPU passed to ``--iree-llvmcpu-target-cpu``.

    Returns
    -------
    bytes
        Compiled IREE VMFB flatbuffer bytes.
    """
    try:
        from iree.compiler import compile_str
    except ImportError as exc:
        raise ImportError(
            "compiling MLIR requires the optional 'iree-compiler' package"
        ) from exc

    extra_args = [
        "--iree-input-demote-f64-to-f32=false",
        "--iree-vm-target-extension-f64",
        "--iree-vm-target-truncate-unsupported-floats=false",
        f"--iree-llvmcpu-target-cpu={target_cpu}",
        "--iree-llvmcpu-loop-vectorization",
        "--iree-llvmcpu-slp-vectorization",
    ]
    return compile_str(
        lowered_mlir,
        target_backends=["llvm-cpu"],
        extra_args=extra_args,
    )


def run_iree(compiled_module: bytes, inputs: ContrastInputs) -> np.ndarray:
    """Execute a compiled IREE ``@counterfactual`` module on ``inputs``.

    Parameters
    ----------
    compiled_module : bytes
        Compiled IREE VMFB flatbuffer bytes from :func:`compile_iree`.
    inputs : ContrastInputs
        Extracted model arrays to pass to ``@counterfactual``.

    Returns
    -------
    ndarray
        One-dimensional ``float64`` array of row-level treatment contrasts.
    """
    try:
        import iree.runtime as ireert
    except ImportError as exc:
        raise ImportError(
            "running compiled modules requires the optional 'iree-runtime' package"
        ) from exc

    context = ireert.SystemContext(config=ireert.Config("local-sync"))
    vm_module = ireert.VmModule.from_flatbuffer(
        context.instance, compiled_module, warn_if_copy=False
    )
    context.add_vm_module(vm_module)
    entrypoint = context.modules.module.counterfactual

    # Copy while `result` is still alive so the IREE buffer cannot be freed
    # before NumPy reads it.
    result = entrypoint(inputs.interaction_data, inputs.coefficients)
    return np.array(np.asarray(result), copy=True)
