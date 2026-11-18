import subprocess

import pytest

from causal_compiler import lower_mlir


def test_lower_mlir_invokes_ci_opt_and_returns_stdout(monkeypatch):
    seen = {}

    def fake_run(cmd, *, input, text, capture_output, check):
        seen["cmd"] = cmd
        seen["input"] = input
        return subprocess.CompletedProcess(
            cmd, returncode=0, stdout="module { // lowered }", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = lower_mlir("module {}", ci_opt_path="/tmp/ci-opt")

    assert seen["cmd"] == ["/tmp/ci-opt", "--ci-lower-treatment-effect"]
    assert seen["input"] == "module {}"
    assert out == "module { // lowered }"


def test_lower_mlir_raises_on_nonzero_exit(monkeypatch):
    def fake_run(cmd, *, input, text, capture_output, check):
        return subprocess.CompletedProcess(
            cmd, returncode=1, stdout="", stderr="parse error"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="parse error"):
        lower_mlir("bad mlir")
