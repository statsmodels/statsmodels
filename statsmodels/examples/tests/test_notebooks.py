import glob
import io
import os
import sys

from nose import SkipTest

try:
    import jupyter_client
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
except ImportError:
    raise SkipTest('Required packages not available')

KNOWN_FAILURES = ['distributed_estimation']
if os.name == 'nt':
    KNOWN_FAILURES += ['mixed_lm_example']

kernels = jupyter_client.kernelspec.find_kernel_specs()
kernel_name = 'python%s' % sys.version_info.major

head, _ = os.path.split(__file__)
NOTEBOOK_DIR = os.path.abspath(os.path.join(head, '..', '..', '..', 'examples', 'notebooks'))

nbs = sorted(glob.glob(os.path.join(NOTEBOOK_DIR, '*.ipynb')))

if not nbs:
    raise SkipTest('No notebooks found so not tests run')


def execute_notebook(src):
    filename, _ = os.path.splitext(src)
    for known_fail in KNOWN_FAILURES:
        if filename == known_fail:
            raise SkipTest('{0} is known to fail'.format(filename))

    fullfile = os.path.join(NOTEBOOK_DIR, src)
    with io.open(fullfile, encoding='utf-8') as f:
        nb = nbformat.read(fullfile, as_version=4)

    ep = ExecutePreprocessor(allow_errors=False,
                             timeout=20,
                             kernel_name=kernel_name)
    ep.preprocess(nb, {'metadata': {'path': NOTEBOOK_DIR}})


def test_notebook():
    for nb in nbs:
        fullfile = os.path.abspath(nb)
        _, filename = os.path.split(fullfile)
        yield (execute_notebook, filename)
