_statspece_filters_tree = _tsa_tree + [
    fs.copyfile('__init__.py'),
    fs.copyfile('_conventional.pxd'),
    fs.copyfile('_inversions.pxd'),
    fs.copyfile('_univariate_diffuse.pxd'),
    fs.copyfile('_univariate.pxd'),
]

statspece_filters_cython_gen = generator(cython,
  arguments : cython_args,
  output : '@BASENAME@.c',
  depends : [_math_pxd, _cython_tree, _special_tree, _statspece_filters_tree]
)

_conventional_pyx = custom_target(
  '_conventional_pyx',
  input: '_conventional.pyx.in',
  output: '_conventional.pyx',
  command: [tempita, '@INPUT@', '--outfile', '@OUTPUT@'],
)

py.extension_module('_conventional',
  statspece_filters_cython_gen.process(_conventional_pyx),
  c_args: cython_c_args,
  include_directories: [inc_np, inc_sm],
  install: true,
  subdir: 'statsmodels/tsa/statespace/_filters',
)

_inversions_pyx = custom_target(
  '_inversions_pyx',
  input: '_inversions.pyx.in',
  output: '_inversions.pyx',
  command: [tempita, '@INPUT@', '--outfile', '@OUTPUT@'],
)

py.extension_module('_inversions',
  statspece_filters_cython_gen.process(_inversions_pyx),
  c_args: cython_c_args,
  include_directories: [inc_np, inc_sm],
  install: true,
  subdir: 'statsmodels/tsa/statespace/_filters',
)

_univariate_diffuse_pyx = custom_target(
  '_univariate_diffuse_pyx',
  input: '_univariate_diffuse.pyx.in',
  output: '_univariate_diffuse.pyx',
  command: [tempita, '@INPUT@', '--outfile', '@OUTPUT@'],
)

py.extension_module('_univariate_diffuse',
  statspece_filters_cython_gen.process(_univariate_diffuse_pyx),
  c_args: cython_c_args,
  include_directories: [inc_np, inc_sm],
  install: true,
  subdir: 'statsmodels/tsa/statespace/_filters',
)

_univariate_pyx = custom_target(
  '_univariate_pyx',
  input: '_univariate.pyx.in',
  output: '_univariate.pyx',
  command: [tempita, '@INPUT@', '--outfile', '@OUTPUT@'],
)

py.extension_module('_univariate',
  statspece_filters_cython_gen.process(_univariate_pyx),
  c_args: cython_c_args,
  include_directories: [inc_np, inc_sm],
  install: true,
  subdir: 'statsmodels/tsa/statespace/_filters',
)