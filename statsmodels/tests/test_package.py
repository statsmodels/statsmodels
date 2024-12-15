import subprocess
import sys

from statsmodels.compat.python import PYTHON_IMPL_WASM

import pytest


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
