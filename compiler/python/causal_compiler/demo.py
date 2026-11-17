"""End-to-end demo: IHDP -> fitted OLS -> ci dialect -> IREE -> verified results.

Run from ``compiler/python``::

    python -m causal_compiler.demo --ci-opt build/tools/ci-opt/ci-opt
"""

from __future__ import annotations

import argparse
import csv
import shlex
import time

import numpy as np

from statsmodels.formula.api import ols

from .api import CompilerApi
from .datasets import load_ihdp, resample_rows
from .synthetic import (
    _compile_iree_cpu_module,
    _full_design_matrices,
    _median_latency,
)



def _owned(device_array):
    """Copy an IREE result while it is still alive (np.asarray alone is a
    view that does not keep the device buffer alive)."""
    return np.array(np.asarray(device_array), copy=True)

def load_resident_kernel(workflow, module: bytes, threads: int):
    """Load the module and copy inputs to the IREE device once.

    ``synthetic._load_iree_cpu_kernel`` passes NumPy arrays on every call,
    so each call re-copies the whole input matrix; at 1M x 25 f64 that
    copy (~200 MB) dominates the measured time. Real workloads keep data
    resident, so the demo times the kernel with device-resident inputs.
    """
    import iree.runtime as ireert

    config = ireert.Config(
        "local-task" if threads > 1 else "local-sync")
    context = ireert.SystemContext(config=config)
    context.add_vm_module(ireert.VmModule.from_flatbuffer(
        context.instance, module, warn_if_copy=False))
    entry = context.modules.module.counterfactual
    x, beta = workflow.mlir_inputs()
    x_dev = ireert.asdevicearray(config.device, x)
    b_dev = ireert.asdevicearray(config.device, beta)
    return (lambda: entry(x_dev, b_dev)), (lambda: _owned(entry(x_dev, b_dev)))


def banner(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")


def head(text: str, lines: int) -> str:
    rows = text.strip().splitlines()
    extra = len(rows) - lines
    return "\n".join(rows[:lines] + ([f"... ({extra} more lines)"] if extra > 0 else []))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ci-opt", required=True, help="ci-opt path or command")
    parser.add_argument("--replication", type=int, default=1)
    parser.add_argument(
        "--sizes", default="747,10000,100000,1000000",
        help="comma-separated row counts (747 = original IHDP)",
    )
    parser.add_argument("--target-cpu", default="host")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--csv", default="demo_results.csv")
    parser.add_argument(
        "--threads", type=int, default=1,
        help="1 = single-threaded IREE (local-sync), fair vs OPENBLAS_NUM_THREADS=1")
    args = parser.parse_args()
    ci_opt = shlex.split(args.ci_opt)

    # 1. Data + fitted model -------------------------------------------------
    banner("1. Load IHDP (Yao et al. Table 2) and fit OLS with treatment interactions")
    data = load_ihdp(args.replication)
    covariates = [f"x{i}" for i in range(1, 26)]
    formula = f"y ~ A * ({' + '.join(covariates)})"
    results = ols(formula, data=data).fit()
    print(f"rows={len(data)}  covariates={len(covariates)}  "
          f"treated={int(data.A.sum())}  params={results.params.size}")
    print(f"formula: y ~ A * (x1 + ... + x25)")

    api = CompilerApi()
    workflow = api.compile_workflow(results, "A", (0.0, 1.0), data)

    # 2. IR at each stage ---------------------------------------------------
    banner("2. Front end emits the high-level `ci` dialect")
    print(head(workflow.emit_mlir(), 14))
    banner("3. ci-opt --ci-lower-treatment-effect lowers it to linalg/arith")
    print(head(workflow.compile(ci_opt), 25))

    # 3. Correctness --------------------------------------------------------
    banner("4. Correctness: compiled IREE output vs statsmodels reference")
    module = _compile_iree_cpu_module(workflow, ci_opt, args.target_cpu)
    _, compiled = load_resident_kernel(workflow, module, args.threads)
    reference = workflow.run().row_effects.to_numpy()
    out = compiled()
    abs_err = np.max(np.abs(out - reference))
    rel_err = abs_err / np.max(np.abs(reference))
    print(f"max |compiled - reference| = {abs_err:.3e}  (relative {rel_err:.3e})")
    print(f"bitwise identical rows: {np.mean(out == reference):.1%}")
    print(f"ATE (model-predicted, compiled) = {out.mean():.4f}")
    print(f"ATE (statsmodels reference)     = {reference.mean():.4f}")
    print(f"ATE (IHDP simulated truth, mean(mu1-mu0)) = "
          f"{(data.mu1 - data.mu0).mean():.4f}   <- sanity check of the model, "
          "not of the compiler")
    np.testing.assert_allclose(out, reference, rtol=1e-10, atol=1e-10)
    print("PASS: agreement within 1e-10")

    # 4. Scaling ------------------------------------------------------------
    banner("5. Scaling: bootstrap-resampled IHDP rows, same fitted model")
    header = ["rows", "statsmodels_ms", "two_gemv_numpy_ms",
              "hand_fused_numpy_ms", "compiled_iree_ms",
              "speedup_vs_statsmodels", "speedup_vs_two_gemv",
              "speedup_vs_hand_fused", "counterfactual_mib", "compiled_input_mib"]
    print("  ".join(f"{h:>14}" for h in header[:5]) + "  vs_sm  vs_2gemv vs_fused")
    rows_out = []
    for n in [int(s) for s in args.sizes.split(",")]:
        frame = data if n == len(data) else resample_rows(data, n, seed=n)
        wf = api.compile_workflow(results, "A", (0.0, 1.0), frame)
        t0 = time.perf_counter()
        kernel, kernel_np = load_resident_kernel(
            wf, _compile_iree_cpu_module(wf, ci_opt, args.target_cpu),
            args.threads)
        compile_s = time.perf_counter() - t0
        ref = wf.run().row_effects.to_numpy()
        np.testing.assert_allclose(kernel_np(), ref, rtol=1e-10, atol=1e-10)

        x, beta = wf.mlir_inputs()
        b_int = beta[list(wf._interaction_coefficients)]
        b_main = beta[wf._main_effect_index]
        d0, d1 = _full_design_matrices(wf)

        t_sm = _median_latency(wf.run, 1, args.repeats)
        t_2g = _median_latency(lambda: d1 @ beta - d0 @ beta, 2, args.repeats)
        t_hf = _median_latency(lambda: (x @ b_int + b_main) * 1.0, 2, args.repeats)
        t_cc = _median_latency(kernel, 2, args.repeats)
        row = [n, t_sm * 1e3, t_2g * 1e3, t_hf * 1e3, t_cc * 1e3,
               t_sm / t_cc, t_2g / t_cc, t_hf / t_cc,
               (d0.nbytes + d1.nbytes) / 2**20, x.nbytes / 2**20]
        rows_out.append(row)
        print("  ".join(f"{v:>14.3f}" if isinstance(v, float) else f"{v:>14}"
                        for v in row[:5])
              + f"  {row[5]:5.1f}x  {row[6]:5.2f}x  {row[7]:5.2f}x"
              + f"   (compile {compile_s:.1f}s)")

    with open(args.csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows_out)
    print(f"\nwrote {args.csv}")
    print("Timings: statsmodels is end-to-end (pandas + patsy + predict); "
          "NumPy/IREE are kernel-only with prebuilt inputs; IREE includes "
          "the Python->runtime call with device-resident inputs.")


if __name__ == "__main__":
    main()
