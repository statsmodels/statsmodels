import glob
import io
import os
import sys

import pytest

from statsmodels.compat.testing import example

try:
    print('In try')
    import pytest
    import jupyter_client
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
except ImportError:
    print('In Except')
    pytestmark = pytest.mark.skip(reason='Required packages not available')

KNOWN_FAILURES = ['distributed_estimation']
if os.name == 'nt':
    KNOWN_FAILURES += ['mixed_lm_example']

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


@example
def test_notebook(notebook):
    fullfile = os.path.abspath(notebook)
    _, filename = os.path.split(fullfile)
    filename, _ = os.path.splitext(notebook)
    
    for known_fail in KNOWN_FAILURES:
        if filename == known_fail:
            pytest.skip(msg='{0} is known to fail'.format(filename))
    
    with io.open(fullfile, encoding='utf-8') as f:
        nb = nbformat.read(fullfile, as_version=4)
    
    ep = ExecutePreprocessor(allow_errors=False,
                             timeout=20,
                             kernel_name=kernel_name)
    ep.preprocess(nb, {'metadata': {'path': NOTEBOOK_DIR}})
