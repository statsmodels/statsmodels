incdir_numpy = run_command(
    py,
    [
        '-c',
        '''
import os
import numpy as np
try:
    # Check if include directory is inside the dir
    # e.g. a venv created inside the dir
    # If so, convert it to a relative path
    incdir = os.path.relpath(np.get_include())
except Exception:
    incdir = np.get_include()
print(incdir)
     ''',
    ],
    check: true,
).stdout().strip()

inc_np = include_directories(incdir_numpy)
math_pxd = files('include/math.pxd')
inc_sm = include_directories('include')

subdir('robust')
subdir('nonparametric')
subdir('tsa')

subdirs_list = [
    'base',
    'compat',
    'datasets',
    'discrete',
    'distributions',
    'duration',
    'emplike',
    'examples',
    'formula',
    'gam',
    'genmod',
    'graphics',
    'imputation',
    'interface',
    'iolib',
    'miscmodels',
    'multivariate',
    'nonparametric',
    'othermod',
    'regression',
    'robust',
    'sandbox',
    'include',
    'stats',
    'tests',
    'tools',
    'treatment',
    'tsa',
]

foreach subdir : subdirs_list
    install_subdir(subdir, install_dir: py.get_install_dir() / 'statsmodels')
endforeach

top_level_py_list = [
    '__init__.py',
    'api.py',
#    '_version.py',  # TODO: generate _version.py
    'conftest.py',
]
py.install_sources(top_level_py_list, subdir: 'statsmodels')