import glob
from pathlib import Path
import sys

import pytest

try:
    import jupyter_client
    from nbconvert.preprocessors import ExecutePreprocessor
    import nbformat

    plat_win = sys.platform.startswith("win")
    if plat_win and (3, 8) <= sys.version_info < (3, 14):  # pragma: no cover
        import asyncio

        try:
            from asyncio import WindowsSelectorEventLoopPolicy
        except ImportError:
            pass  # Can't assign a policy which doesn't exist.
        else:
            pol = asyncio.get_event_loop_policy()
            if not isinstance(pol, WindowsSelectorEventLoopPolicy):
                asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())
except ImportError:
    pytestmark = pytest.mark.skip(reason="Required packages not available")

try:
    import rpy2

    HAS_RPY2 = True
except ImportError:
    HAS_RPY2 = False

try:
    import joblib

    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    import pymc

    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False

try:
    import arviz

    HAS_ARVIZ = True
except ImportError:
    HAS_ARVIZ = False


KNOWN_FAILURES = []
JOBLIB_NOTEBOOKS = ["distributed_estimation"]
RPY2_NOTEBOOKS = ["mixed_lm_example", "robust_models_1"]
PYMC_NOTEBOOKS = ["statespace_custom_models", "statespace_sarimax_pymc"]
ARVIZ_NOTEBOOKS = ["statespace_tvpvar_mcmc_cfa"]

kernel_name = f"python{sys.version_info.major}"

head = Path(__file__).resolve().parent
NOTEBOOK_DIR = head / ".." / ".." / ".." / "examples" / "notebooks"

nbs = sorted(NOTEBOOK_DIR.glob("*.ipynb"))

if nbs:
    ids = [p.name for p in nbs]

    @pytest.fixture(params=nbs, ids=ids)
    def notebook(request):
        return request.param

    @pytest.mark.slow
    @pytest.mark.example
    @pytest.mark.thread_unsafe(reason="notebooks use matplotlib")
    def test_notebook(notebook):
        fullfile = notebook.resolve()
        filename = notebook.stem

        if filename in KNOWN_FAILURES:
            pytest.skip(f"{filename} is known to fail")
        if filename in RPY2_NOTEBOOKS and not HAS_RPY2:
            pytest.skip(f"{filename} since rpy2 is not installed")
        if filename in JOBLIB_NOTEBOOKS and not HAS_JOBLIB:
            pytest.skip(f"{filename} since joblib is not installed")
        if filename in PYMC_NOTEBOOKS and not HAS_PYMC:
            pytest.skip(f"{filename} since pymc is not installed")
        if filename in ARVIZ_NOTEBOOKS and not HAS_ARVIZ:
            pytest.skip(f"{filename} since arviz is not installed")

        with open(fullfile, encoding="utf-8") as fp:
            nb = nbformat.read(fp, as_version=4)

        # The slowest cells in the simulation/MCMC notebooks take ~35s on a
        # fast desktop, so a 20s budget failed them spuriously on any loaded
        # or slower machine.
        ep = ExecutePreprocessor(
            allow_errors=False, timeout=120, kernel_name=kernel_name
        )
        ep.preprocess(nb, {"metadata": {"path": NOTEBOOK_DIR}})

else:
    pytestmark = pytest.mark.skip(reason="No notebooks found so no tests run")
