"""run all examples to make sure we do not get an exception

Note:
If an example contaings plt.show(), then all plot windows have to be closed
manually, at least in my setup.

uncomment plt.show() to show all plot windows

"""

from statsmodels.compat.python import input

from pathlib import Path

stop_on_error = True
filelist = [
    "example_pca.py",
    "example_sysreg.py",
    "example_mle.py",
    "example_pca_regression.py",
]
cont = input(
    "Are you sure you want to run all of the examples?\nThis is done mainly to check that they are up to date.\n(y/n) >>> "
)
if "y" in cont.lower():
    for run_all_f in filelist:
        try:
            print("Executing example file", run_all_f)
            print("-----------------------" + "-" * len(run_all_f))
            with Path(run_all_f).open(encoding="utf-8") as f:
                exec(f.read())
        except Exception as exc:
            print("*********************")
            print("ERROR in example file", run_all_f)
            print("**********************" + "*" * len(run_all_f))
            if stop_on_error:
                raise exc
