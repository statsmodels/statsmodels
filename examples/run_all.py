"""
Run all python examples to make sure they do not raise
"""
from __future__ import print_function

import tempfile

SHOW_PLOT = False
BAD_FILES = ['robust_models_1']


def no_show(*args):
    pass


if __name__ == '__main__':
    import glob
    import sys
    import matplotlib.pyplot as plt

    if not SHOW_PLOT:
        PLT_SHOW = plt.show
        plt.show = no_show

    SAVE_STDOUT = sys.stdout
    SAVE_STDERR = sys.stderr
    REDIRECT_STDOUT = tempfile.TemporaryFile('w')
    REDIRECT_STDERR = tempfile.TemporaryFile('w')

    EXAMPLE_FILES = glob.glob('python/*.py')
    for example in EXAMPLE_FILES:
        KNOWN_BAD_FILE = any([bf in example for bf in BAD_FILES])
        with open(example, 'r') as pyfile:
            code = pyfile.read()
            try:
                sys.stdout = REDIRECT_STDOUT
                sys.stderr = REDIRECT_STDERR
                exec(code)
            except Exception as e:
                sys.stderr = SAVE_STDERR
                print('FAIL: {0}'.format(example), file=sys.stderr)
                if KNOWN_BAD_FILE:
                    print('This FAIL is expected', file=sys.stderr)
                else:
                    print('The last error was: ', file=sys.stderr)
                    print(e.__class__.__name__, file=sys.stderr)
                    print(e, file=sys.stderr)
            else:
                sys.stdout = SAVE_STDOUT
                print('SUCCESS: {0}'.format(example))
            finally:
                plt.close('all')

    REDIRECT_STDOUT.close()
    REDIRECT_STDERR.close()
    sys.stdout = SAVE_STDOUT
    sys.stderr = SAVE_STDERR
    if not SHOW_PLOT:
        plt.show = PLT_SHOW
