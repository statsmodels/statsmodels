project(
  'statsmodels',
  'c', 'cython',
  license: 'BSD-3',
  meson_version: '>= 1.9.0',
  default_options: [],
  version:  run_command(['statsmodels/_build/git_version.py'], check: true).stdout().strip(),
)

cc = meson.get_compiler('c')
cy = meson.get_compiler('cython')
cython = find_program(cy.cmd_array()[0])
tempita = find_program('statsmodels/_build/tempita.py')

# Check if scipy's cython_blas exports blas_int (ILP64 support).
scipy_has_blas_int = cy.compiles(
  'from scipy.linalg.cython_blas cimport blas_int',
  name: 'scipy cython_blas blas_int'
)

_blas_int_conf = configuration_data()
if scipy_has_blas_int
  _blas_int_conf.set('BLAS_INT_DEF', 'from scipy.linalg.cython_blas cimport blas_int')
else
  _blas_int_conf.set('BLAS_INT_DEF', 'ctypedef int blas_int')
endif

if not cy.version().version_compare('>=3.0.10')
  error('statsmodels requires Cython >= 3.0.10')
endif

py = import('python').find_installation(pure: false)
fs = import('fs')

py.install_sources('pyproject.toml', subdir: 'statsmodels')
subdir('statsmodels')
