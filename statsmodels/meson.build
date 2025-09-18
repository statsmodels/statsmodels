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
inc_sm = include_directories('includes')
_math_pxd = [
]

# Copy the main __init__.py to the build dir.
# Some submodules (linalg, special, optimize) add pxd files to this.
# Needed to trick Cython, it won't do a relative import outside a package
#_cython_tree = declare_dependency(sources: [
_cython_tree = [fs.copyfile('__init__.py')]
cython_args = []
cython_c_args =  ['-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION']
if get_option('cython-coverage')
  cython_args += ['--directive', 'linetrace=true']
  cython_c_args += ['-DCYTHON_TRACE_NOGIL=1']
endif
cython_args += ['-3', '--output-file', '@OUTPUT@', '--include-dir', '@BUILD_ROOT@', '@INPUT@']
# -X linetrace=true
#

version_link_args = []

sm_dir = py.get_install_dir() / 'statsmodels'
# Generate version.py for sdist
meson.add_dist_script(
   ['_build/git_version.py', '--meson-dist', '--write',
     'statsmodels/_version.py']
)
if not fs.exists('_version.py')
  generate_version = custom_target(
    'generate-version',
    install: true,
    build_always_stale: true,
    build_by_default: true,
    output: '_version.py',
    input: '_build/git_version.py',
    command: [py, '@INPUT@', '--write', '@OUTPUT@'],
    install_dir: sm_dir,
    install_tag: 'python-runtime'
  )
else
  # When building from sdist, version.py exists and should be included
  py.install_sources(
    ['_version.py'],
    subdir : 'statsmodels'
  )
endif

subdir('includes')
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
