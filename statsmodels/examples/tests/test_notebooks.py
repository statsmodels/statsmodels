import glob
import io
import os
import sys

import pytest

try:
    import jupyter_client
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
except ImportError:
    pytestmark = pytest.mark.skip(reason='Required packages not available')

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
JOBLIB_NOTEBOOKS = ['distributed_estimation']
RPY2_NOTEBOOKS = ['mixed_lm_example', 'robust_models_1']

kernel_name = 'python%s' % sys.version_info.major

head, _ = os.path.split(__file__)
NOTEBOOK_DIR = os.path.abspath(os.path.join(head, '..', '..', '..', 'examples', 'notebooks'))

nbs = sorted(glob.glob(os.path.join(NOTEBOOK_DIR, '*.ipynb')))
ids = list(map(lambda p: os.path.split(p)[-1], nbs))


@pytest.fixture(params=nbs, ids=ids)
def notebook(request):
    return request.param


if not nbs:
    pytestmark = pytest.mark.skip(reason='No notebooks found so not tests run')


@pytest.mark.example
def test_notebook(notebook):
    fullfile = os.path.abspath(notebook)
    filename, _ = os.path.splitext(notebook)

    for known_fail in KNOWN_FAILURES:
        if filename == known_fail:
            pytest.skip('{0} is known to fail'.format(filename))

    with io.open(fullfile, encoding='utf-8') as f:
        nb = nbformat.read(fullfile, as_version=4)

    ep = ExecutePreprocessor(allow_errors=False,
                             timeout=20,
                             kernel_name=kernel_name)
    ep.preprocess(nb, {'metadata': {'path': NOTEBOOK_DIR}})
