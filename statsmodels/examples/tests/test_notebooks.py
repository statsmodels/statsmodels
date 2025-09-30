import glob
import os
import sys

import pytest

try:
    import jupyter_client
    from nbconvert.preprocessors import ExecutePreprocessor
    import nbformat

    plat_win = sys.platform.startswith("win")
    if plat_win and sys.version_info >= (3, 8):  # pragma: no cover
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


KNOWN_FAILURES = []
JOBLIB_NOTEBOOKS = ["distributed_estimation"]
RPY2_NOTEBOOKS = ["mixed_lm_example", "robust_models_1"]

kernel_name = "python%s" % sys.version_info.major

head, _ = os.path.split(__file__)
NOTEBOOK_DIR = os.path.join(head, "..", "..", "..", "examples", "notebooks")
NOTEBOOK_DIR = os.path.abspath(NOTEBOOK_DIR)

nbs = sorted(glob.glob(os.path.join(NOTEBOOK_DIR, "*.ipynb")))

if nbs:
    ids = [os.path.split(p)[-1] for p in nbs]

    @pytest.fixture(params=nbs, ids=ids)
    def notebook(request):
        return request.param

    @pytest.mark.slow
    @pytest.mark.example
    def test_notebook(notebook):
        fullfile = os.path.abspath(notebook)
        _, filename = os.path.split(fullfile)
        filename, _ = os.path.splitext(filename)

        if filename in KNOWN_FAILURES:
            pytest.skip(f"{filename} is known to fail")
        if filename in RPY2_NOTEBOOKS and not HAS_RPY2:
            pytest.skip(f"{filename} since rpy2 is not installed")
        if filename in JOBLIB_NOTEBOOKS and not JOBLIB_NOTEBOOKS:
            pytest.skip(f"{filename} since joblib is not installed")

        with open(fullfile, encoding="utf-8") as fp:
            nb = nbformat.read(fp, as_version=4)

        ep = ExecutePreprocessor(
            allow_errors=False, timeout=20, kernel_name=kernel_name
        )
        ep.preprocess(nb, {"metadata": {"path": NOTEBOOK_DIR}})

else:
    pytestmark = pytest.mark.skip(reason="No notebooks found so no tests run")
