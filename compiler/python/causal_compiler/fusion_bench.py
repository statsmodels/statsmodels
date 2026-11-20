"""K interventions through the *compiler*: fusion pass vs direct op vs hand-written.

Same workload as ``kinterv.py`` (IHDP covariates, simulated dose, statsmodels
Logit, ``y ~ T * (x1 + ... + x25)``), but the fused kernel now comes from
``ci-opt`` instead of hand-written MLIR:

- per_dose:     the fused kernel compiled for K = 1 and called K times
                (re-reads X K times). Baseline for "no sharing".
- handwritten:  ``kinterv.FUSED_MLIR`` (one matmul + one N x K sigmoid).
- direct_op:    ``ci.intervention_predict`` -> ``--ci-lower-intervention-predict``.
- fusion_pass:  K x ``ci.predict`` -> ``--ci-fuse-predictions`` ->
                ``--ci-lower-intervention-predict``. This is the path where
                the compiler itself finds the shared work.
- numpy_hoisted: expert NumPy (X @ W once, N x K epilogue).

Every variant is checked against statsmodels (rows <= --statsmodels-max-rows)
or naive NumPy, and the fusion pass is checked to have merged all K predicts.

Run from ``compiler/python``::

    OPENBLAS_NUM_THREADS=1 python -m causal_compiler.fusion_bench --ci-opt PATH
"""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess

import numpy as np

from . import kinterv
from .adapter import extract_intervention_inputs
from .mlir import emit_intervention_predict_mlir, emit_predict_sequence_mlir
from .synthetic import _iree_cpu_codegen_args


def ci_opt(command: list[str], text: str, *passes: str) -> str:
    done = subprocess.run([*command, *passes], input=text, text=True,
                          capture_output=True)
    if done.returncode:
        raise RuntimeError(f"ci-opt {' '.join(passes)} failed:\n{done.stderr}")
    return done.stdout


def load_named(source: str, name: str, target_cpu: str):
    """Compile MLIR with IREE and load ``name`` on kinterv's shared device."""
    import iree.runtime as rt
    from iree.compiler import compile_str

    if kinterv._CONFIG is None:
        kinterv._CONFIG = rt.Config("local-sync")
    vmfb = compile_str(source, target_backends=["llvm-cpu"],
                       extra_args=_iree_cpu_codegen_args(target_cpu))
    ctx = rt.SystemContext(config=kinterv._CONFIG)
    ctx.add_vm_module(rt.VmModule.from_flatbuffer(ctx.instance, vmfb,
                                                  warn_if_copy=False))
    return getattr(ctx.modules.module, name)


def as_columns(result) -> np.ndarray:
    """N x K array from either one N x K result or K separate N results."""
    if isinstance(result, (list, tuple)):
        return np.column_stack([kinterv.to_numpy(r).reshape(-1) for r in result])
    return kinterv.to_numpy(result)


def main() -> None:
    import iree.runtime as rt

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ci-opt", required=True)
    ap.add_argument("--rows", default="100000,1000000")
    ap.add_argument("--k", default="1,2,4,8,16,32,64")
    ap.add_argument("--fit-rows", type=int, default=100_000)
    ap.add_argument("--statsmodels-max-rows", type=int, default=100_000)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--target-cpu", default="host")
    ap.add_argument("--csv", default="fusion_results.csv")
    a = ap.parse_args()
    command = shlex.split(a.ci_opt)

    results = kinterv.fit_logit(a.fit_rows)
    w, c = kinterv.unpack(results)
    d = w.shape[0]
    print(f"Logit fitted on {a.fit_rows} rows, {results.params.size} params")

    names = ["numpy_hoisted", "per_dose", "handwritten", "direct_op", "fusion_pass"]
    header = (["rows", "K"] + [f"{n}_ms" for n in names]
              + ["fusion_pass_vs_per_dose", "direct_op_vs_per_dose",
                 "fusion_pass_vs_handwritten", "direct_op_vs_numpy_hoisted",
                 "max_abs_err"])
    print("  ".join(f"{h:>13}" for h in ["rows", "K"] + names)
          + "   pass/perK  direct/perK  pass/hand  direct/numpy   max_err")
    rows_out = []
    for n in [int(s) for s in a.rows.split(",")]:
        frame = kinterv.make_data(n, seed=n)
        x = np.ascontiguousarray(frame[kinterv.COVARIATES].to_numpy())

        single = load_named(kinterv.FUSED_MLIR.format(n=n, d=d, k=1),
                            "predict", a.target_cpu)
        device = kinterv._CONFIG.device
        x_dev, w_dev, c_dev = (rt.asdevicearray(device, v) for v in (x, w, c))

        for k in [int(s) for s in a.k.split(",")]:
            doses = np.linspace(0.0, 4.0, k) if k > 1 else np.array([2.0])
            inputs = extract_intervention_inputs(results, "T", list(doses), frame)
            ix_dev = rt.asdevicearray(device, np.ascontiguousarray(inputs.covariate_data))
            ib_dev = rt.asdevicearray(device, inputs.coefficients)

            # K = 1: nothing to fuse, and a lone ci.predict has no lowering
            # (it stays in the IR), so the fusion-pass variant is n/a.
            fusion = None
            if k > 1:
                fused_ir = ci_opt(command, emit_predict_sequence_mlir(inputs),
                                  "--ci-fuse-predictions")
                merged = fused_ir.count('"ci.intervention_predict"')
                leftover = fused_ir.count('"ci.predict"')
                if merged != 1 or leftover:
                    raise AssertionError(
                        f"fusion pass did not merge all predicts at K={k}: "
                        f"{merged} intervention_predict, {leftover} predict left")
                fusion = load_named(
                    ci_opt(command, fused_ir, "--ci-lower-intervention-predict"),
                    "counterfactual_unfused", a.target_cpu)
            direct = load_named(
                ci_opt(command, emit_intervention_predict_mlir(inputs),
                       "--ci-lower-intervention-predict"),
                "counterfactual_k", a.target_cpu)
            hand = load_named(kinterv.FUSED_MLIR.format(n=n, d=d, k=k),
                              "predict", a.target_cpu)
            t_dev = rt.asdevicearray(device, doses)
            singles = [rt.asdevicearray(device, np.array([t])) for t in doses]

            def numpy_hoisted():
                m = x @ w
                eta = np.multiply.outer(m[:, 1] + c[1], doses)
                eta += (m[:, 0] + c[0])[:, None]
                np.negative(eta, out=eta)
                np.exp(eta, out=eta)
                eta += 1.0
                return np.reciprocal(eta, out=eta)

            run = {
                "numpy_hoisted": numpy_hoisted,
                "per_dose": lambda: [single(x_dev, w_dev, c_dev, s) for s in singles],
                "handwritten": lambda: hand(x_dev, w_dev, c_dev, t_dev),
                "direct_op": lambda: direct(ix_dev, ib_dev),
                "fusion_pass": lambda: fusion(ix_dev, ib_dev),
            }
            active = [name for name in names if name != "fusion_pass" or fusion]

            if n <= a.statsmodels_max_rows:
                ref = np.column_stack([results.predict(frame.assign(T=t)).to_numpy()
                                       for t in doses])
            else:
                ref = numpy_hoisted().copy()
            errors = {"numpy_hoisted": float(np.abs(numpy_hoisted() - ref).max())}
            for name in [v for v in ("handwritten", "direct_op", "fusion_pass")
                         if v in active]:
                errors[name] = float(np.abs(as_columns(run[name]()) - ref).max())
            per_dose = np.column_stack([
                kinterv.to_numpy(single(x_dev, w_dev, c_dev, s))[:, 0] for s in singles])
            errors["per_dose"] = float(np.abs(per_dose - ref).max())
            err = max(errors.values())
            if err > 1e-10:
                raise AssertionError(f"mismatch at rows={n} K={k}: {errors}")

            ms = {name: (kinterv.timed(run[name], a.repeats) * 1e3
                         if name in active else float("nan")) for name in names}
            row = ([n, k] + [ms[name] for name in names]
                   + [ms["per_dose"] / ms["fusion_pass"],
                      ms["per_dose"] / ms["direct_op"],
                      ms["handwritten"] / ms["fusion_pass"],
                      ms["numpy_hoisted"] / ms["direct_op"], err])
            rows_out.append(row)
            print(f"{n:>13}  {k:>13}  "
                  + "  ".join(f"{ms[name]:>13.3f}" for name in names)
                  + f"   {row[7]:8.2f}x  {row[8]:10.2f}x  {row[9]:8.2f}x"
                  + f"  {row[10]:11.2f}x   {err:.1e}", flush=True)

    with open(a.csv, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
        wr.writerows(rows_out)
    print(f"wrote {a.csv}. Ratios > 1 mean the second-named variant is slower.")


if __name__ == "__main__":
    main()
