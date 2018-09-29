"""
Run all python examples to make sure they do not raise
"""
from __future__ import print_function

BAD_FILES = ['robust_models_1']

if __name__ == '__main__':
    import glob
    import sys
    import matplotlib.pyplot as plt

    SAVE_STDOUT = sys.stdout
    SAVE_STDERR = sys.stderr
    REDIRECT_STDOUT = open('trash', 'w')
    REDIRECT_STDERR = open('trash-err', 'w')

    EXAMPLE_FILES = glob.glob('python/*.py')
    for example in EXAMPLE_FILES:
        KNOWN_BAD_FILE = any([bf in example for bf in BAD_FILES])
        with open(example, 'r') as pyfile:
            code = pyfile.read()
            try:
                sys.stdout = REDIRECT_STDOUT
                sys.stderr = REDIRECT_STDERR
                exec(code)
                plt.close('all')
                sys.stdout = SAVE_STDOUT
                sys.stderr = SAVE_STDERR
                print('SUCCESS: {0}'.format(example))
            except Exception as e:
                plt.close('all')
                sys.stdout = SAVE_STDOUT
                sys.stderr = SAVE_STDERR
                print('FAIL: {0}'.format(example), file=sys.stderr)
                if KNOWN_BAD_FILE:
                    print('This FAIL is expected', file=sys.stderr)
                else:
                    print('The last error was: ', file=sys.stderr)
                    print(e.__class__.__name__, file=sys.stderr)
                    print(e, file=sys.stderr)
