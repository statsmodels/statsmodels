from statsmodels.compat.python import PYTHON_IMPL_WASM

from pathlib import Path
import subprocess
import sys

import pytest

import statsmodels


@pytest.mark.skipif(
    PYTHON_IMPL_WASM,
    reason="Can't start subprocess in WASM/Pyodide"
)
def test_lazy_imports():
    # Check that when statsmodels.api is imported, matplotlib is _not_ imported
    cmd = ("import statsmodels.api as sm; "
           "import sys; "
           "mods = [x for x in sys.modules if 'matplotlib.pyplot' in x]; "
           "assert not mods, mods")
    cmd = sys.executable + ' -c "' + cmd + '"'
    p = subprocess.Popen(cmd, shell=True, close_fds=True)
    p.wait()
    rc = p.returncode
    assert rc == 0


@pytest.mark.skipif(
    PYTHON_IMPL_WASM,
    reason="Can't start subprocess in WASM/Pyodide"
)
def test_docstring_optimization_compat():
    # GH#5235 check that importing with stripped docstrings does not raise
    cmd = sys.executable + ' -OO -c "import statsmodels.api as sm"'
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    out = p.communicate()
    rc = p.returncode
    assert rc == 0, out


def test_test_builds_pytest_command_and_reports_success(monkeypatch):
    # statsmodels.test() is a thin shim around PytestTester; mock
    # pytest.main so this does not actually recurse into the full suite
    calls = {}

    def fake_main(cmd):
        calls["cmd"] = cmd
        return 0

    monkeypatch.setattr(pytest, "main", fake_main)

    result = statsmodels.test(extra_args=["-k", "doesnotexist"], exit=False)

    assert result is True
    assert calls["cmd"][0] == str(Path(statsmodels.__file__).parent)
    assert calls["cmd"][1:] == ["-k", "doesnotexist"]


def test_test_default_args_and_failure_status(monkeypatch):
    def fake_main(cmd):
        assert cmd[1:] == ["--tb=short", "--disable-pytest-warnings"]
        return 1

    monkeypatch.setattr(pytest, "main", fake_main)

    assert statsmodels.test() is False


def test_test_exit_true_calls_sys_exit(monkeypatch):
    monkeypatch.setattr(pytest, "main", lambda cmd: 3)

    with pytest.raises(SystemExit) as excinfo:
        statsmodels.test(exit=True)
    assert excinfo.value.code == 3
