project(
  'statsmodels',
  'c', 'cython',
  license: 'BSD-3',
  meson_version: '>= 1.9.0',
  default_options: [],
  version: '0.15.0',
)

cc = meson.get_compiler('c')
cy = meson.get_compiler('cython')
tempita = find_program('statsmodels/_build/tempita.py')

if not cy.version().version_compare('>=3.0.10')
  error('statsmodels requires Cython >= 3.0.10')
endif

py = import('python').find_installation(pure: false)
py_dep = py.dependency()
fs = import('fs')

py.install_sources('pyproject.toml', subdir: 'statsmodels')
subdir('statsmodels')
