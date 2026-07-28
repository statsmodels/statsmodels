"""run all examples to make sure we do not get an exception

Note:
If an example contaings plt.show(), then all plot windows have to be closed
manually, at least in my setup.

uncomment plt.show() to show all plot windows

"""

from statsmodels.compat.python import input, lzip

from pathlib import Path

import matplotlib.pyplot as plt

stop_on_error = True
filelist = [
    "example_glsar.py",
    "example_wls.py",
    "example_gls.py",
    "example_glm.py",
    "example_ols_tftest.py",
    "example_ols.py",
    "example_ols_minimal.py",
    "example_rlm.py",
    "example_discrete.py",
    "example_predict.py",
    "example_ols_table.py",
    "tut_ols.py",
    "tut_ols_rlm.py",
    "tut_ols_wls.py",
]
use_glob = True
if use_glob:
    filelist = [p.name for p in Path().glob("*.py")]
print(lzip(range(len(filelist)), filelist))
for fname in ["run_all.py", "example_rpy.py"]:
    filelist.remove(fname)
plt_show = plt.show


def noop(*args):
    pass


plt.show = noop
cont = input(
    "Are you sure you want to run all of the examples?\nThis is done mainly to check that they are up to date.\n(y/n) >>> "
)
has_errors = []
if "y" in cont.lower():
    for run_all_f in filelist:
        try:
            print("\n\nExecuting example file", run_all_f)
            print("-----------------------" + "-" * len(run_all_f))
            with Path(run_all_f).open(encoding="utf-8") as f:
                exec(f.read())  # noqa: S102
        except Exception:
            print("**********************" + "*" * len(run_all_f))
            print("ERROR in example file", run_all_f)
            print("**********************" + "*" * len(run_all_f))
            has_errors.append(run_all_f)
            if stop_on_error:
                raise
print("\nModules that raised exception:")
print(has_errors)
plt.close("all")
plt.show = plt_show
plt.show()
