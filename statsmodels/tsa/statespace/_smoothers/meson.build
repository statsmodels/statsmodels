_statspece_smoothers_tree = _tsa_tree + [
    fs.copyfile('__init__.py'),
    fs.copyfile('_alternative.pxd'),
    fs.copyfile('_classical.pxd'),
    fs.copyfile('_conventional.pxd'),
    fs.copyfile('_univariate_diffuse.pxd'),
    fs.copyfile('_univariate.pxd'),
]

statspece_smoothers_cython_gen = generator(cython,
  arguments : cython_args,
  output : '@BASENAME@.c',
  depends : [_math_pxd, _cython_tree, _special_tree, _statspece_smoothers_tree]
)

_alternative_pyx = custom_target(
  '_alternative_pyx',
  input: '_alternative.pyx.in',
  output: '_alternative.pyx',
  command: [tempita, '@INPUT@', '--outfile', '@OUTPUT@'],
)

py.extension_module('_alternative',
  statspece_smoothers_cython_gen.process(_alternative_pyx),
  c_args: cython_c_args,
  include_directories: [inc_np, inc_sm],
  install: true,
  subdir: 'statsmodels/tsa/statespace/_smoothers',
)

_classical_pyx = custom_target(
  '_classical_pyx',
  input: '_classical.pyx.in',
  output: '_classical.pyx',
  command: [tempita, '@INPUT@', '--outfile', '@OUTPUT@'],
)

py.extension_module('_classical',
  statspece_smoothers_cython_gen.process(_classical_pyx),
  c_args: cython_c_args,
  include_directories: [inc_np, inc_sm],
  install: true,
  subdir: 'statsmodels/tsa/statespace/_smoothers',
)

_conventional_pyx = custom_target(
  '_conventional_pyx',
  input: '_conventional.pyx.in',
  output: '_conventional.pyx',
  command: [tempita, '@INPUT@', '--outfile', '@OUTPUT@'],
)

py.extension_module('_conventional',
  statspece_smoothers_cython_gen.process(_conventional_pyx),
  c_args: cython_c_args,
  include_directories: [inc_np, inc_sm],
  install: true,
  subdir: 'statsmodels/tsa/statespace/_smoothers',
)

_univariate_diffuse_pyx = custom_target(
  '_univariate_diffuse_pyx',
  input: '_univariate_diffuse.pyx.in',
  output: '_univariate_diffuse.pyx',
  command: [tempita, '@INPUT@', '--outfile', '@OUTPUT@'],
)

py.extension_module('_univariate_diffuse',
  statspece_smoothers_cython_gen.process(_univariate_diffuse_pyx),
  c_args: cython_c_args,
  include_directories: [inc_np, inc_sm],
  install: true,
  subdir: 'statsmodels/tsa/statespace/_smoothers',
)

_univariate_pyx = custom_target(
  '_univariate_pyx',
  input: '_univariate.pyx.in',
  output: '_univariate.pyx',
  command: [tempita, '@INPUT@', '--outfile', '@OUTPUT@'],
)

py.extension_module('_univariate',
  statspece_smoothers_cython_gen.process(_univariate_pyx),
  c_args: cython_c_args,
  include_directories: [inc_np, inc_sm],
  install: true,
  subdir: 'statsmodels/tsa/statespace/_smoothers',
)