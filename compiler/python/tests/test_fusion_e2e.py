"""End-to-end: K x ci.predict -> --ci-fuse-predictions -> lowering -> IREE.

Covers the C++ passes, which the emitter-only tests in test_kinterv.py do not
run. Requires CI_OPT_COMMAND (path to ci-opt) and the IREE Python packages.
"""

import os
import shlex
import subprocess

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from causal_compiler import (
    emit_intervention_predict_mlir,
    emit_predict_sequence_mlir,
    extract_intervention_inputs,
)

pytestmark = pytest.mark.skipif(
    not os.environ.get("CI_OPT_COMMAND"),
    reason="set CI_OPT_COMMAND to run the ci-opt/IREE integration test",
)

IREE_ARGS = [
    "--iree-input-demote-f64-to-f32=false",
    "--iree-vm-target-extension-f64",
    "--iree-vm-target-truncate-unsupported-floats=false",
    "--iree-llvmcpu-target-cpu=generic",
]


def _ci_opt(text, *passes):
    command = shlex.split(os.environ["CI_OPT_COMMAND"])
    done = subprocess.run(
        [*command, *passes], input=text, text=True, capture_output=True
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def _run(lowered, name, *arrays):
    rt = pytest.importorskip("iree.runtime")
    compiler = pytest.importorskip("iree.compiler")
    vmfb = compiler.compile_str(
        lowered, target_backends=["llvm-cpu"], extra_args=IREE_ARGS
    )
    config = rt.Config("local-sync")
    ctx = rt.SystemContext(config=config)
    ctx.add_vm_module(
        rt.VmModule.from_flatbuffer(ctx.instance, vmfb, warn_if_copy=False)
    )
    result = getattr(ctx.modules.module, name)(
        *[rt.asdevicearray(config.device, a) for a in arrays]
    )
    results = result if isinstance(result, (list, tuple)) else [result]
    # Copy while the device arrays are alive: np.asarray is only a view.
    return [np.array(np.asarray(r), copy=True) for r in results]


@pytest.fixture(scope="module")
def logit_case():
    # 25 covariates: realistic width. tensor.from_elements of 2*D scalars
    # is miscompiled by IREE at this size, which this test guards against.
    rng = np.random.default_rng(0)
    n, d = 400, 25
    cols = [f"x{i}" for i in range(1, d + 1)]
    frame = pd.DataFrame(rng.normal(size=(n, d)), columns=cols)
    frame["T"] = rng.uniform(0.0, 4.0, n)
    eta = -0.5 + 0.3 * frame["T"] + frame[cols].to_numpy() @ rng.normal(0, 0.2, d)
    frame["y"] = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    results = smf.logit(f"y ~ T * ({' + '.join(cols)})", data=frame).fit(disp=0)
    doses = [0.0, 1.0, 2.5, 4.0]
    inputs = extract_intervention_inputs(results, "T", doses, frame.iloc[:64])
    expected = np.column_stack(
        [results.predict(frame.iloc[:64].assign(T=t)).to_numpy() for t in doses]
    )
    return inputs, expected


def test_fuse_pass_merges_predicts(logit_case):
    inputs, _ = logit_case
    fused = _ci_opt(emit_predict_sequence_mlir(inputs), "--ci-fuse-predictions")
    assert fused.count('"ci.intervention_predict"') == 1
    assert '"ci.predict"' not in fused


def test_fused_pipeline_matches_statsmodels(logit_case):
    inputs, expected = logit_case
    lowered = _ci_opt(
        emit_predict_sequence_mlir(inputs),
        "--ci-fuse-predictions",
        "--ci-lower-intervention-predict",
    )
    outs = _run(lowered, "counterfactual_unfused",
                inputs.covariate_data, inputs.coefficients)
    got = np.column_stack([o.reshape(-1) for o in outs])
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


def test_direct_intervention_predict_matches_statsmodels(logit_case):
    inputs, expected = logit_case
    lowered = _ci_opt(
        emit_intervention_predict_mlir(inputs), "--ci-lower-intervention-predict"
    )
    (got,) = _run(lowered, "counterfactual_k",
                  inputs.covariate_data, inputs.coefficients)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
