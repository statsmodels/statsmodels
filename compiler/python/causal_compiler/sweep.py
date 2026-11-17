"""Covariate sweep: current ci lowering vs a matvec formulation vs NumPy.

Tests whether the compiled kernel falls behind NumPy as the number of
interaction covariates d grows, and whether lowering to a single
``linalg.matvec`` fixes it. All variants are single-threaded with inputs
already resident on the IREE device. Run from ``compiler/python``::

    OPENBLAS_NUM_THREADS=1 python -m causal_compiler.sweep --ci-opt PATH
"""

from __future__ import annotations

import argparse
import csv
import shlex

import numpy as np

from .synthetic import (
    _compile_iree_cpu_module,
    _iree_cpu_codegen_args,
    _median_latency,
    make_synthetic_workflow,
)

MATVEC_MLIR = """
#id = affine_map<(i) -> (i)>
#scalar = affine_map<(i) -> ()>
func.func public @effect(%x: tensor<{n}x{d}xf64>, %b: tensor<{d}xf64>,
                         %m: tensor<f64>) -> tensor<{n}xf64> {{
  %zero = arith.constant 0.0 : f64
  %empty = tensor.empty() : tensor<{n}xf64>
  %acc = linalg.fill ins(%zero : f64) outs(%empty : tensor<{n}xf64>) -> tensor<{n}xf64>
  %mv = linalg.matvec ins(%x, %b : tensor<{n}x{d}xf64>, tensor<{d}xf64>)
                      outs(%acc : tensor<{n}xf64>) -> tensor<{n}xf64>
  %r = linalg.generic {{indexing_maps = [#id, #scalar, #id],
                        iterator_types = ["parallel"]}}
       ins(%mv, %m : tensor<{n}xf64>, tensor<f64>) outs(%empty : tensor<{n}xf64>) {{
    ^bb0(%a: f64, %c: f64, %o: f64):
      %s = arith.addf %a, %c : f64
      linalg.yield %s : f64
  }} -> tensor<{n}xf64>
  return %r : tensor<{n}xf64>
}}
"""



def _owned(device_array):
    """Copy an IREE result while it is still alive (np.asarray alone is a
    view that does not keep the device buffer alive)."""
    return np.array(np.asarray(device_array), copy=True)

def _resident(module: bytes, name: str, *arrays):
    import iree.runtime as rt

    config = rt.Config("local-sync")
    ctx = rt.SystemContext(config=config)
    ctx.add_vm_module(rt.VmModule.from_flatbuffer(
        ctx.instance, module, warn_if_copy=False))
    fn = getattr(ctx.modules.module, name)
    dev = [rt.asdevicearray(config.device, a) for a in arrays]
    return (lambda: fn(*dev)), (lambda: _owned(fn(*dev)))


def main() -> None:
    from iree.compiler import compile_str

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ci-opt", required=True)
    p.add_argument("--rows", type=int, default=1_000_000)
    p.add_argument("--covariates", default="4,8,16,25,32,64")
    p.add_argument("--repeats", type=int, default=11)
    p.add_argument("--target-cpu", default="host")
    p.add_argument("--csv", default="sweep_results.csv")
    a = p.parse_args()
    ci_opt = shlex.split(a.ci_opt)

    header = ["rows", "covariates", "numpy_ms", "current_lowering_ms",
              "matvec_lowering_ms", "current_vs_numpy", "matvec_vs_numpy",
              "numpy_GBps", "current_GBps", "matvec_GBps"]
    print("  ".join(f"{h:>18}" for h in header))
    out = []
    for d in [int(s) for s in a.covariates.split(",")]:
        wf = make_synthetic_workflow(n_rows=a.rows, n_covariates=d)
        x, beta = wf.mlir_inputs()
        b = np.ascontiguousarray(beta[list(wf._interaction_coefficients)])
        m = np.array(beta[wf._main_effect_index])
        ref = wf.run_direct().row_effects.to_numpy()

        cur, cur_np = _resident(
            _compile_iree_cpu_module(wf, ci_opt, a.target_cpu),
            "counterfactual", x, beta)
        mv_module = compile_str(
            MATVEC_MLIR.format(n=a.rows, d=d), target_backends=["llvm-cpu"],
            extra_args=_iree_cpu_codegen_args(a.target_cpu))
        mv, mv_np = _resident(mv_module, "effect", x, b, m)
        for got in (cur_np(), mv_np()):
            np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-10)

        t_np = _median_latency(lambda: x @ b + m, 2, a.repeats)
        t_cur = _median_latency(cur, 2, a.repeats)
        t_mv = _median_latency(mv, 2, a.repeats)
        gb = x.nbytes / 1e9
        row = [a.rows, d, t_np * 1e3, t_cur * 1e3, t_mv * 1e3,
               t_np / t_cur, t_np / t_mv, gb / t_np, gb / t_cur, gb / t_mv]
        out.append(row)
        print("  ".join(f"{v:>18.3f}" if isinstance(v, float) else f"{v:>18}"
                        for v in row), flush=True)

    with open(a.csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(out)
    print(f"wrote {a.csv}  (vs_numpy > 1 means compiled is faster; "
          "GBps = input bytes read / time)")


if __name__ == "__main__":
    main()
