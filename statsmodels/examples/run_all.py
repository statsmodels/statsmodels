"""
Run all examples to make sure we don't get an exception.

Note:
If an example contaings plt.show(), then all plot windows have to be closed
manually, at least in my [josef's] setup.

Usage Notes
-----------
- Un-comment plt.show() to show all plot windows.
"""
from __future__ import print_function
import os

from statsmodels.compat.python import lzip
import matplotlib.pyplot as plt  # matplotlib is required for many examples

stop_on_error = False

here = os.path.dirname(__file__)

filelist = ['example_glsar.py', 'example_wls.py', 'example_gls.py',
            'example_glm.py', 'example_ols_tftest.py',
            'example_ols.py', 'example_ols_minimal.py', 'example_rlm.py',
            'example_discrete.py', 'example_predict.py',
            'example_ols_table.py',
            'tut_ols.py', 'tut_ols_rlm.py', 'tut_ols_wls.py']
# Note: we have intentionally excluded example_rpy.py

use_glob = True
if use_glob:
    import glob
    filelist = glob.glob(os.path.join(here, '*.py'))
    # TODO: get examples/tsa/ in there too

print(lzip(range(len(filelist)), filelist))

filelist.sort()
filelist = [x for x in filelist
            if os.path.split(x)[-1] not in ['run_all.py', 'example_rpy.py']]


def run_example(path):
    # FIXME: don't use `exec`
    with open(path, "rb") as fd:
        content = fd.read()
    exec(content.replace('__name__ == "__main__"', "True"))


def run_all():
    # temporarily disable show
    plt_show = plt.show

    def noop(*args):
        pass

    plt.show = noop

    has_errors = []
    for run_all_f in filelist:
        print("\n\nExecuting example file", run_all_f)
        print("-----------------------" + "-"*len(run_all_f))
        rc = run_example(run_all_f)
        if rc != 0:
            print("**********************" + "*"*len(run_all_f))
            print("ERROR in example file", run_all_f)
            print("**********************" + "*"*len(run_all_f))
            has_errors.append(run_all_f)
            if stop_on_error:
                raise

    print('\nModules that raised exception:')
    print(has_errors)

    # reenable show after closing windows
    plt.close('all')
    plt.show = plt_show
    plt.show()


if __name__ == "__main__":
    run_all()
