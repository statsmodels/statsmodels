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

if not cy.version().version_compare('>=3.0.10')
  error('statsmodels requires Cython >= 3.0.10')
endif

# We need -lm for C code that uses math functions, so add it for all targets.
m_dep = cc.find_library('m', required : false)
if m_dep.found()
  add_project_link_arguments('-lm', language : 'c')
endif

py = import('python').find_installation(pure: false)
fs = import('fs')

py.install_sources('pyproject.toml', subdir: 'statsmodels')
subdir('statsmodels')
