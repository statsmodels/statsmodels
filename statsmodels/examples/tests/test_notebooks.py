import glob
import io
import os
import sys

import pytest

try:
    import jupyter_client  # noqa: F401
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
except ImportError:
    pytestmark = pytest.mark.skip(reason='Required packages not available')

try:
    import rpy2  # noqa: F401
    HAS_RPY2 = True
except ImportError:
    HAS_RPY2 = False

try:
    import joblib  # noqa: F401
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


KNOWN_FAILURES = []
JOBLIB_NOTEBOOKS = ['distributed_estimation']
RPY2_NOTEBOOKS = ['mixed_lm_example', 'robust_models_1']

kernel_name = 'python%s' % sys.version_info.major

head, _ = os.path.split(__file__)
NOTEBOOK_DIR = os.path.join(head, '..', '..', '..', 'examples', 'notebooks')
NOTEBOOK_DIR = os.path.abspath(NOTEBOOK_DIR)

nbs = sorted(glob.glob(os.path.join(NOTEBOOK_DIR, '*.ipynb')))
ids = list(map(lambda p: os.path.split(p)[-1], nbs))


@pytest.fixture(params=nbs, ids=ids)
def notebook(request):
    return request.param


if not nbs:
    pytestmark = pytest.mark.skip(reason='No notebooks found so not tests run')


@pytest.mark.slow
@pytest.mark.example
def test_notebook(notebook):
    fullfile = os.path.abspath(notebook)
    _, filename = os.path.split(fullfile)
    filename, _ = os.path.splitext(filename)

    if filename in KNOWN_FAILURES:
        pytest.skip('{0} is known to fail'.format(filename))
    if filename in RPY2_NOTEBOOKS and not HAS_RPY2:
        pytest.skip('{0} since rpy2 is not installed'.format(filename))
    if filename in JOBLIB_NOTEBOOKS and not JOBLIB_NOTEBOOKS:
        pytest.skip('{0} since joblib is not installed'.format(filename))

    with io.open(fullfile, encoding='utf-8') as fp:
        nb = nbformat.read(fp, as_version=4)

    ep = ExecutePreprocessor(allow_errors=False,
                             timeout=20,
                             kernel_name=kernel_name)
    ep.preprocess(nb, {'metadata': {'path': NOTEBOOK_DIR}})
